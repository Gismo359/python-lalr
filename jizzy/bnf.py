from __future__ import annotations

import ast
import regex
import functools
import numpy as np

from jizzy.tokenizer import tokenize
from collections import defaultdict
from dataclasses import dataclass, field
from numpy.typing import NDArray
from enum import Enum
from typing import Type, TypeVar, Callable, cast


T = TypeVar("T")


def cache(x: T) -> T:
    return cast(T, functools.cache(cast(Callable, x)))


class TokenType(Enum):
    COMMENT = "#.*+"
    T = "\w+"
    NT = "<\w+>"
    EQ = "::="
    STRING = r"\"([^\\\"]|\\.)*\""


@cache
@dataclass(frozen=True, slots=True)
class ClosureItem:
    idx: int
    lhs: int
    rhs: tuple[int, ...]
    dot: int
    lookahead: frozenset[int]

    @cache
    def next_item(self) -> ClosureItem:
        return ClosureItem(idx=self.idx, lhs=self.lhs, rhs=self.rhs, dot=self.dot + 1, lookahead=self.lookahead)

    def __hash__(self) -> int:
        return id(self)


@dataclass(slots=True)
class ShiftReduceParser:
    start_idx: int
    eof_idx: int
    name_to_idx: dict[str, int]
    idx_to_name: dict[int, str]
    terminals: tuple[str, ...]
    nonterminals: tuple[tuple[int, tuple[int, ...]], ...]
    terminal_cutoff: int = field(init=False)
    rule_sets: tuple[tuple[tuple[int, tuple[int, ...]], ...], ...] = field(init=False)
    firsts: list[frozenset[int]] = field(init=False)
    final_automaton: NDArray[np.int16] = field(init=False)

    @cache
    def default_closure(self, lhs: int, lookahead: frozenset[int]) -> tuple[ClosureItem, ...]:
        return tuple(
            ClosureItem(idx=idx, lhs=lhs, rhs=rhs, dot=0, lookahead=lookahead)
            for idx, rhs in self.rule_sets[lhs] if rhs is not None
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

            try:
                lhs = item.rhs[item.dot]
            except IndexError:
                continue

            if lhs < self.terminal_cutoff:
                continue

            try:
                next_symbol = item.rhs[item.dot + 1]
                lookahead = self.firsts[next_symbol]
            except IndexError:
                lookahead = item.lookahead

            default_closure = self.default_closure(lhs, lookahead)
            stack.extend(default_closure)

        lookaheads: dict[tuple[int, int], list[frozenset[int]]] = defaultdict(list)
        for item in new_closure.values():
            lookaheads[(item.idx, item.dot)].append(item.lookahead)

        return tuple(
            dict.fromkeys(
                ClosureItem(
                    idx,
                    *self.nonterminals[idx],
                    dot,
                    frozenset.union(*lookahead)
                )
                for (idx, dot), lookahead in lookaheads.items()
            )
        )

    @cache
    def make_token_type(self) -> Type[Enum]:
        return Enum("enum", {(f"group_{key}", value) for key, value in enumerate(self.terminals)})  # type: ignore

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
        rule_sets: list[tuple[tuple[int, tuple[int, ...]], ...]] = [None] * len(self.idx_to_name)
        for symbol in list(dict.fromkeys(key for key, _ in self.nonterminals)):
            rule_sets[symbol] = tuple((idx, rhs) for idx, (lhs, rhs) in enumerate(self.nonterminals) if symbol == lhs)

        terminal_cutoff = len(self.terminals)

        firsts: list[frozenset[int]] = []
        for symbol in self.idx_to_name:
            if symbol < terminal_cutoff:
                firsts.append(frozenset({symbol}))
            else:
                first_set = set()

                first_passed: set[int] = set()
                first_stack: list[int] = [symbol]
                while first_stack:
                    current = first_stack.pop(0)
                    if current in first_passed:
                        continue

                    first_passed.add(current)

                    for _, r in rule_sets[current]:
                        first = r[0]
                        if first in first_stack:
                            continue

                        if first < terminal_cutoff:
                            first_set.add(first)
                        else:
                            first_stack.append(first)

                firsts.append(frozenset(first_set))

        self.terminal_cutoff = terminal_cutoff
        self.rule_sets = tuple(rule_sets)
        self.firsts = firsts

        initial_closure = self.default_closure(self.start_idx, frozenset({self.eof_idx}))
        initial_closure = self.expand_closure(initial_closure)

        default_state = np.zeros(len(self.idx_to_name), dtype=np.int16)

        closures: dict[tuple[ClosureItem, ...], int] = {}
        automaton: dict[int, NDArray[np.int16]] = defaultdict(default_state.copy)

        closure_passed: set[int] = set()
        closure_stack: list[tuple[ClosureItem, ...]] = [initial_closure]
        while closure_stack:
            fuck_austin = closure_stack.pop(0)

            current_id = closures.setdefault(fuck_austin, len(closures))
            if current_id in closure_passed:
                continue

            closure_passed.add(current_id)

            symbol_to_shift: defaultdict[int, list[ClosureItem]] = defaultdict(list)
            symbol_to_reduce: defaultdict[int, list[ClosureItem]] = defaultdict(list)
            for item in fuck_austin:
                try:
                    symbol_to_shift[item.rhs[item.dot]].append(item)
                except IndexError:
                    for symbol in item.lookahead:
                        symbol_to_reduce[symbol].append(item)

            current_state = automaton[current_id]
            for symbol, subclosure in symbol_to_reduce.items():
                assert len(subclosure) == 1

                item, = subclosure
                current_state[symbol] = -item.idx - 1

            for symbol, subclosure in symbol_to_shift.items():
                shifts: dict[ClosureItem, None] = dict.fromkeys(item.next_item() for item in subclosure)

                new_closure = self.expand_closure(tuple(shifts))

                current_state[symbol] = closures.setdefault(new_closure, len(closures)) + 1

                closure_stack.append(new_closure)

            assert current_id in automaton

        remap_indices: NDArray[np.int16] = np.zeros(len(closures) + 1, dtype=np.int16)
        remap_lookaheads: dict[tuple[tuple[int, int], ...], int] = {}
        for closure, idx in closures.items():
            remap_indices[idx + 1] = remap_lookaheads.setdefault(
                tuple((item.idx, item.dot) for item in closure),
                len(remap_lookaheads) + 1
            )

        old_numpy_automaton = np.vstack(tuple(automaton.values()))

        remap_indices_sort_idx = np.argsort(remap_indices[1:])
        remap_indices_sorted = remap_indices[1:][remap_indices_sort_idx]
        _, remap_indices_counts = np.unique(remap_indices_sorted, return_counts=True)

        remap_indices_reverse = np.split(remap_indices_sort_idx, np.cumsum(remap_indices_counts))

        new_final_states_please_work = []
        for old_indices in remap_indices_reverse[:-1]:
            old_states = old_numpy_automaton[old_indices]

            shift_mask = (old_states >= 0)
            shift_states = remap_indices[old_states * shift_mask]
            reduce_states = old_states * np.logical_not(shift_mask)

            shift_state = np.max(shift_states, axis=0)
            reduce_state = np.min(reduce_states, axis=0)

            new_final_states_please_work.append(np.where(shift_state > 0, shift_state, reduce_state))

        self.final_automaton = np.vstack(new_final_states_please_work)

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
        terminals=tuple(terminals.values()),
        nonterminals=tuple(nonterminals)
    )
