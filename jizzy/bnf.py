from __future__ import annotations

import ast
import regex
import functools
import numpy as np

from jizzy.tokenizer import tokenize
from collections import defaultdict
from dataclasses import dataclass, field
from numpy.typing import NDArray
from enum import IntEnum, Enum
from typing import Type, TypeVar, Callable, cast, Generic


T = TypeVar("T")


def cache(x: T) -> T:
    return cast(T, functools.cache(cast(Callable, x)))


class TokenType(Enum):
    COMMENT = "#.*+"
    T = "\w+"
    NT = "<\w+>"
    EQ = "::="
    STRING = r"\"([^\\\"]|\\.)*\""


class ActionType(IntEnum):
    SHIFT = 0
    REDUCE = 1


@cache
@dataclass(frozen=True, order=True, slots=True)
class TestTuple(Generic[T]):
    values: tuple[T, ...]

    def __hash__(self) -> int:
        return id(self)


@cache
@dataclass(frozen=True, order=True, slots=True)
class Action:
    type: ActionType
    value: int

    def __hash__(self) -> int:
        return id(self)


@cache
@dataclass(frozen=True, order=True, slots=True)
class ClosureItem:
    lhs: int
    rhs: tuple[int, ...]
    dot: int
    lookahead: frozenset[int]

    def at_end(self) -> bool:
        try:
            self.rhs[self.dot]
            return False
        except IndexError:
            return True
        # return self.dot == len(self.rhs)

    def almost_at_end(self) -> bool:
        return self.dot == len(self.rhs) - 1

    def current_symbol(self) -> int:
        return self.rhs[self.dot]

    def next_symbol(self) -> int:
        return self.rhs[self.dot + 1]

    def next_item(self) -> ClosureItem:
        return ClosureItem(lhs=self.lhs, rhs=self.rhs, dot=self.dot + 1, lookahead=self.lookahead)

    def __hash__(self) -> int:
        return id(self)


@dataclass
class ShiftReduceParser:
    start_idx: int
    eof_idx: int
    name_to_idx: dict[str, int]
    idx_to_name: dict[int, str]
    terminals: dict[int, str]
    nonterminals: tuple[tuple[int, tuple[int, ...]], ...]
    rule_sets: dict[int, tuple[tuple[int, ...], ...]] = field(init=False)
    firsts: dict[int, frozenset[int]] = field(init=False)
    final_automaton: NDArray[np.int16] = field(init=False)

    @cache
    def default_closure(self, lhs: int, lookahead: frozenset[int]) -> TestTuple[ClosureItem, ...]:
        return tuple(
            ClosureItem(lhs=lhs, rhs=rhs, dot=0, lookahead=lookahead)
            for rhs in self.rule_sets[lhs]
        )

    @cache
    def expand_closure(self, closure: tuple[ClosureItem, ...]) -> tuple[ClosureItem, ...]:
        new_closure: dict[int, ClosureItem] = {}

        idx = 0
        stack: list[ClosureItem] = list(closure)
        while True:
            try:
                item = stack[idx]
            except IndexError:
                break

            idx += 1

            item_id = id(item)
            if item_id in new_closure:
                continue

            new_closure.setdefault(item_id, item)

            if item.at_end():
                continue

            lhs = item.current_symbol()
            if lhs in self.terminals:
                continue

            if item.almost_at_end():
                lookahead = item.lookahead
            else:
                lookahead = self.firsts[item.next_symbol()]

            stack.extend(self.default_closure(lhs, lookahead))

        lookaheads: dict[tuple[int, tuple[int, ...], int], set[int]] = defaultdict(set)
        for _, item in new_closure.items():
            lookaheads[(item.lhs, item.rhs, item.dot)].update(item.lookahead)

        return tuple(
            dict.fromkeys(
                ClosureItem(lhs=lhs, rhs=rhs, dot=dot, lookahead=frozenset(lookahead))
                for (lhs, rhs, dot), lookahead in lookaheads.items()
            )
        )

    @cache
    def make_token_type(self) -> Type[Enum]:
        return Enum("enum", {(f"group_{key}", value) for key, value in self.terminals.items()})  # type: ignore

    def parse(self, text: str):
        token_type = self.make_token_type()
        tokens = tokenize(token_type, text)
        terminal_types = map(lambda x: x.type.name, tokens)
        terminal_indices = map(lambda x: int(x.split("_")[1]), terminal_types)
        terminal_inputs = list(terminal_indices) + [self.eof_idx]

        stack = [(1, 0)]
        while terminal_inputs:
            token_idx = terminal_inputs.pop(0)

            action = self.final_automaton[stack[-1][0] - 1, token_idx]
            assert action != 0

            if action > 0:
                stack.append((action, token_idx))

            if action == -1:
                assert token_idx == self.eof_idx
                return ":)"

            token_idx = terminal_inputs[0]
            while (action := self.final_automaton[stack[-1][0] - 1, token_idx]) < -1:
                symbol, rule = self.nonterminals[-action - 1]
                del stack[-len(rule):]

                action = self.final_automaton[stack[-1][0] - 1, symbol]
                assert action > 0

                stack.append((action, symbol))

    def __post_init__(self):
        rule_sets: dict[int, tuple[tuple[int, ...], ...]] = {}
        for symbol in list(dict.fromkeys(key for key, _ in self.nonterminals)):
            rule_sets[symbol] = tuple(rhs for lhs, rhs in self.nonterminals if symbol == lhs)

        firsts: dict[int, frozenset[int]] = {}
        for symbol in self.idx_to_name:
            if symbol in self.terminals:
                firsts[symbol] = frozenset({symbol})
            else:
                first_set = set()

                first_passed: set[int] = set()
                first_stack: list[int] = [symbol]
                while first_stack:
                    current = first_stack.pop(0)
                    if current in first_passed:
                        continue

                    first_passed.add(current)

                    for r in rule_sets[current]:
                        first = r[0]
                        if first in first_stack:
                            continue

                        if first in self.terminals:
                            first_set.add(first)
                        else:
                            first_stack.append(first)

                firsts[symbol] = frozenset(first_set)

        self.rule_sets = rule_sets
        self.firsts = firsts

        initial_closure = self.default_closure(self.start_idx, frozenset({self.eof_idx}))
        initial_closure = self.expand_closure(initial_closure)

        closures: dict[tuple[ClosureItem, ...], int] = {}
        automaton: dict[int, dict[int, Action]] = defaultdict(dict)

        def closure_id(closure: tuple[ClosureItem, ...]) -> int:
            return closures.setdefault(closure, len(closures))

        closure_passed: set[int] = set()
        closure_stack: list[tuple[ClosureItem, ...]] = [initial_closure]
        while closure_stack:
            current_closure = closure_stack.pop(0)
            current_id = closure_id(current_closure)
            if current_id in closure_passed:
                continue

            closure_passed.add(current_id)

            symbol_to_shift: defaultdict[int, list[ClosureItem]] = defaultdict(list)
            symbol_to_reduce: defaultdict[int, list[ClosureItem]] = defaultdict(list)
            for item in current_closure:
                try:
                    symbol = item.current_symbol()
                    symbol_to_shift[symbol].append(item)
                except IndexError:
                    for symbol in item.lookahead:
                        symbol_to_reduce[symbol].append(item)

            for symbol, subclosure in symbol_to_shift.items():
                shifts: dict[ClosureItem, None] = dict.fromkeys(item.next_item() for item in subclosure)

                new_closure = self.expand_closure(tuple(shifts))

                automaton[current_id][symbol] = Action(
                    type=ActionType.SHIFT,
                    value=closure_id(new_closure) + 1
                )

                closure_stack.append(new_closure)

            for symbol, subclosure in symbol_to_reduce.items():
                assert len(subclosure) == 1

                item, = subclosure
                reduction = (item.lhs, item.rhs)

                if symbol in automaton[current_id]:
                    assert automaton[current_id][symbol].type is ActionType.SHIFT, f"Conflict {symbol} vs {automaton[current_id][symbol]}"

                automaton[current_id][symbol] = Action(
                    type=ActionType.REDUCE,
                    value=self.nonterminals.index(reduction) + 1
                )

            assert current_id in automaton

        remap_indices: dict[int, int] = {}
        remap_closures: dict[tuple[tuple[int, tuple[int, ...], int], ...], tuple[ClosureItem, ...]] = {}
        for closure, idx in closures.items():
            key = tuple((item.lhs, item.rhs, item.dot) for item in closure)
            value = remap_closures.setdefault(key, closure)
            if value is not closure:
                assert all(
                    (a.lhs, a.rhs, a.dot) == (b.lhs, b.rhs, b.dot)
                    for a, b in zip(closure, value)
                )

                remap_closures[key] = tuple(
                    ClosureItem(
                        lhs=a.lhs,
                        rhs=a.rhs,
                        dot=a.dot,
                        lookahead=a.lookahead | b.lookahead
                    ) for a, b in zip(closure, value)
                )

            remap_indices.setdefault(idx, list(remap_closures).index(key))

        new_automaton: dict[int, dict[int, Action]] = defaultdict(dict)
        for old, new in remap_indices.items():
            old_state = automaton[old]
            new_state = new_automaton[new]
            for symbol in self.idx_to_name:
                if symbol not in old_state:
                    continue

                old_action = old_state[symbol]
                if old_action.type is ActionType.SHIFT:
                    old_action = Action(
                        type=ActionType.SHIFT,
                        value=remap_indices[old_action.value - 1] + 1
                    )

                if symbol not in new_state:
                    new_state[symbol] = old_action

                new_action = new_state.setdefault(symbol, old_action)
                if old_action is new_action:
                    continue

                if old_action.type is ActionType.SHIFT:
                    if new_action.type is ActionType.SHIFT:
                        assert old_action.value == new_action.value
                    if new_action.type is ActionType.REDUCE:
                        new_state[symbol] = old_action
                if old_action.type is ActionType.REDUCE:
                    if new_action.type is ActionType.REDUCE:
                        assert old_action.value == new_action.value, "Reduce-reduce conflict"

        automaton = new_automaton

        final_automaton = np.zeros(shape=(len(automaton), len(self.idx_to_name)), dtype=np.int16)

        for state, actions in automaton.items():
            for symbol in self.idx_to_name:
                if symbol not in actions:
                    continue

                action = actions[symbol]
                final_automaton[state][symbol] = action.value if action.type is ActionType.SHIFT else -action.value

        self.final_automaton = final_automaton

    def __hash__(self) -> int:
        return id(self)


def parse(text: str) -> ShiftReduceParser:
    tokens = tokenize(TokenType, text)
    tokens = list(filter(lambda x: x.type is not TokenType.COMMENT, tokens))

    name_to_idx: dict[str, int] = {}
    idx_to_name: dict[int, str] = {}
    terminals: dict[int, str] = {}
    nonterminals: list[tuple[int, tuple[int, ...]]] = []

    terminal_count = len(set(map(lambda x: x.text, filter(lambda x: x.type is TokenType.T, tokens)))) + 1
    nonterminal_count = len(set(map(lambda x: x.text, filter(lambda x: x.type is TokenType.NT, tokens)))) + 1

    eof_idx = 0
    start_idx = terminal_count

    symbol_tokens = filter(lambda x: x.type in (TokenType.T, TokenType.NT), tokens)
    names = dict.fromkeys(map(lambda x: x.text, symbol_tokens))

    for name in names:
        if len(name_to_idx) == eof_idx:
            name_to_idx.setdefault("$", eof_idx)
        elif len(name_to_idx) == start_idx:
            name_to_idx.setdefault("<>", start_idx)

        name_to_idx.setdefault(name, len(name_to_idx))

    terminals[eof_idx] = "a^"

    for name, idx in name_to_idx.items():
        idx_to_name[idx] = name

    while tokens:
        lhs = tokens.pop(0)
        assert lhs.type in (TokenType.T, TokenType.NT)

        eq = tokens.pop(0)
        assert eq.type is TokenType.EQ

        lhs_idx = name_to_idx.setdefault(lhs.text, len(name_to_idx))

        if lhs.type is TokenType.T:
            pattern = tokens.pop(0)
            assert pattern.type is TokenType.STRING

            terminals[lhs_idx] = ast.literal_eval(pattern.text)

            regex.compile(terminals[lhs_idx])

        if lhs.type is TokenType.NT:
            rhs = tokens[:]
            for idx, token in enumerate(tokens):
                if token.type is not TokenType.EQ:
                    continue

                assert idx != 0

                rhs = tokens[:idx - 1]

                break

            del tokens[:len(rhs)]

            assert all(token.type in (TokenType.T, TokenType.NT) for token in rhs)

            nonterminals.append((lhs_idx, tuple(map(name_to_idx.get, map(lambda x: x.text, rhs)))))

    nonterminals.insert(0, (start_idx, (nonterminals[0][0],)))

    assert len(name_to_idx) == terminal_count + nonterminal_count

    return ShiftReduceParser(
        start_idx=start_idx,
        eof_idx=eof_idx,
        name_to_idx=name_to_idx,
        idx_to_name=idx_to_name,
        terminals=terminals,
        nonterminals=tuple(nonterminals)
    )
