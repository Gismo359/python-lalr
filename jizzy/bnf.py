from __future__ import annotations
from jizzy.tokenizer import tokenize
from numpy.typing import NDArray
import regex
import numpy as np

import ast
import functools
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pyroaring import BitMap, FrozenBitMap
from typing import Callable, Generic, Type, TypeVar, TypeAlias, cast


T = TypeVar("T")


class _bitset(Generic[T], BitMap):
    pass


class _frozenbitset(Generic[T], FrozenBitMap):
    pass


bitset: TypeAlias = cast(Type[set], set)
frozenbitset: TypeAlias = cast(Type[frozenset], frozenset)


def cache(x: T) -> T:
    return cast(T, functools.cache(cast(Callable, x)))


class TokenType(Enum):
    T = "\w+"
    NT = "<\w+>"
    EQ = "::="
    CALLS = "=>"
    STRING = r"\"([^\\\"]|\\.)*\""
    COMMENT = "#.*+"


ClosureItemLR0 = tuple[int, int]
ClosureItemCLR = tuple[int, int, frozenbitset[int]]

ClosureLR0 = tuple[ClosureItemLR0, ...]
ClosureCLR = tuple[ClosureItemCLR, ...]


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
    firsts: list[frozenbitset[int]] = field(init=False)
    final_automaton: NDArray[np.int16] = field(init=False)

    @cache
    def lr0_make_closure(self, lhs: int) -> ClosureLR0:
        return tuple(
            (idx, 0) for idx, rhs in self.rule_sets[lhs] if rhs is not None
        )

    @cache
    def lr0_expand_closure(self, closure: ClosureLR0) -> ClosureLR0:
        new_closures: dict[ClosureItemLR0, None] = {}

        ddddd = 0
        stack: list[ClosureItemLR0] = list(closure)
        while True:
            try:
                item = stack[ddddd]
            except IndexError:
                break

            ddddd += 1

            if item in new_closures:
                continue

            new_closures.setdefault(item)

            idx, dot = item
            rule = self.nonterminals[idx][1]

            try:
                lhs = rule[dot]
            except IndexError:
                continue

            if lhs < self.terminal_cutoff:
                continue

            default_closure = self.lr0_make_closure(lhs)
            stack.extend(default_closure)

        return tuple(new_closures)

    @cache
    def lalr_expand_closure(self, lr0_closure: ClosureLR0, lalr_lookaheads: tuple[frozenbitset[int], ...]) -> tuple[ClosureLR0, list[bitset[int]]]:
        lr0_closure = self.lr0_expand_closure(lr0_closure)
        lr_closure_to_idx = {closure: idx for idx, closure in enumerate(lr0_closure)}

        updated: set[int] = set()
        lookaheads: list[bitset[int]] = [bitset() for _ in range(len(lr0_closure))]
        # for lr0_item_idx, (lr0_item, lalr_input_lookahead) in enumerate(zip(lr0_closure, lalr_lookaheads)):
        for lr0_item_idx in range(len(lalr_lookaheads)):
            lr0_item = lr0_closure[lr0_item_idx]
            lalr_input_lookahead = lalr_lookaheads[lr0_item_idx]
            lookaheads[lr0_item_idx].update(lalr_input_lookahead)

            try:
                rule_idx, rule_dot = lr0_item
                self.nonterminals[rule_idx][1][rule_dot]
                updated.add(lr0_item_idx)
            except IndexError:
                pass

        while updated:
            lr0_item_idx = updated.pop()
            lr0_item = lr0_closure[lr0_item_idx]

            rule_idx, rule_dot = lr0_item
            rule = self.nonterminals[rule_idx][1]

            next_idx = rule[rule_dot]

            if next_idx < self.terminal_cutoff:
                continue

            lookahead: bitset[int] | frozenbitset[int]
            try:
                lookahead = self.firsts[rule[rule_dot + 1]]
            except IndexError:
                lookahead = lookaheads[lr0_item_idx]

            for a in self.lr0_make_closure(next_idx):
                updated_idx = lr_closure_to_idx[a]
                updated_lookahead = lookaheads[updated_idx]

                before_size = len(updated_lookahead)
                updated_lookahead |= lookahead
                after_size = len(updated_lookahead)

                if before_size != after_size:
                    updated.add(updated_idx)

        return lr0_closure, lookaheads

    def clr_make_automaton(self) -> tuple[dict[ClosureCLR, int], NDArray[np.int16]]:
        start_lr0_item = ((0, 0),)
        start_clr_lookahead = (frozenbitset({self.eof_idx}),)
        initial_lr0_closure, initial_clr_lookaheads = self.lalr_expand_closure(start_lr0_item, start_clr_lookahead)

        default_state = [0] * len(self.idx_to_name)

        initial_clr_closure = tuple(
            (idx, dot, frozenbitset(clr_lookahead))
            for (idx, dot), clr_lookahead in zip(initial_lr0_closure, initial_clr_lookaheads)
        )

        closures: dict[ClosureCLR, int] = {initial_clr_closure: 0}
        automaton: dict[int, list[int]] = defaultdict(default_state.copy)

        stack: list[tuple[ClosureCLR, int]] = [(initial_clr_closure, 0)]
        while stack:
            current, current_idx = stack.pop(0)

            symbol_to_shift: defaultdict[int, list[ClosureItemCLR]] = defaultdict(list)
            symbol_to_reduce: defaultdict[int, list[ClosureItemCLR]] = defaultdict(list)
            for item in current:
                idx, dot, lookahead = item
                try:
                    symbol_to_shift[self.nonterminals[idx][1][dot]].append(item)
                except IndexError:
                    for symbol in lookahead:
                        symbol_to_reduce[symbol].append(item)

            state = automaton[current_idx]
            for symbol, subclosure in symbol_to_reduce.items():
                item, = subclosure
                state[symbol] = -item[0] - 1

            for symbol, subclosure in symbol_to_shift.items():
                next_lr0_closure, next_clr_lookaheads = self.lalr_expand_closure(
                    tuple((a, b + 1) for a, b, _ in subclosure),
                    tuple(frozenbitset(c) for _, _, c in subclosure)
                )

                new_closure = tuple(
                    (idx, dot, frozenbitset(clr_lookahead))
                    for (idx, dot), clr_lookahead in zip(next_lr0_closure, next_clr_lookaheads)
                )

                closure_count = len(closures)
                next_closure_idx = closures.setdefault(new_closure, closure_count)
                state[symbol] = next_closure_idx + 1

                if closure_count == next_closure_idx:
                    stack.append((new_closure, next_closure_idx))

        return closures, np.vstack(tuple(automaton.values()))

    def lalr_make_automaton(self) -> NDArray[np.int16]:
        start_lr0_item = ((0, 0),)
        start_lalr_lookahead = (frozenbitset({self.eof_idx}),)
        initial_lr0_closure, initial_lalr_lookaheads = self.lalr_expand_closure(start_lr0_item, start_lalr_lookahead)

        default_state = [0] * len(self.idx_to_name)

        lr0_closures: list[ClosureLR0] = [initial_lr0_closure]
        lr0_closure_to_idx: dict[ClosureLR0, int] = {initial_lr0_closure: 0}
        lalr_lookaheads: list[list[bitset[int]]] = [initial_lalr_lookaheads]
        lalr_automaton: dict[int, list[int]] = defaultdict(default_state.copy)

        updated: bitset[int] = {0}
        while updated:
            current_closure_idx = updated.pop()
            current_closure = lr0_closures[current_closure_idx]
            current_lookaheads = lalr_lookaheads[current_closure_idx]

            symbol_to_shift: defaultdict[int, tuple[list[tuple[int, int]],
                                                    list[bitset[int]]]] = defaultdict(lambda: ([], []))
            symbol_to_reduce: defaultdict[int, list[tuple[int, int]]] = defaultdict(list)
            for item in zip(current_closure, current_lookaheads):
                (idx, dot), lookahead = item
                try:
                    shift_closure, shift_lookaheads = symbol_to_shift[self.nonterminals[idx][1][dot]]
                    shift_closure.append((idx, dot + 1))
                    shift_lookaheads.append(lookahead)
                except IndexError:
                    for symbol in lookahead:
                        symbol_to_reduce[symbol].append((idx, dot))

            state = lalr_automaton[current_closure_idx]
            for symbol, subclosure in symbol_to_reduce.items():
                (idx, _), = subclosure
                next_state = state[symbol]
                if next_state < 0:
                    assert next_state == -idx - 1
                elif next_state == 0:
                    state[symbol] = -idx - 1

            for symbol, (shift_closure, shift_lookaheads) in symbol_to_shift.items():
                next_lr0_closure, next_updated_lookaheads = self.lalr_expand_closure(
                    tuple(shift_closure),
                    tuple(frozenbitset(sa) for sa in shift_lookaheads)
                )
                next_closure_idx = lr0_closure_to_idx.setdefault(next_lr0_closure, len(lr0_closure_to_idx))
                if next_closure_idx == len(lr0_closures):
                    lr0_closures.append(next_lr0_closure)
                    lalr_lookaheads.append(next_updated_lookaheads)
                    updated.add(next_closure_idx)
                else:
                    needs_update = False
                    next_lookaheads = lalr_lookaheads[next_closure_idx]
                    for lookahead, updated_lookahead in zip(next_lookaheads, next_updated_lookaheads):
                        old_count = len(lookahead)
                        lookahead |= updated_lookahead
                        new_count = len(lookahead)

                        needs_update |= old_count != new_count

                    if needs_update:
                        updated.add(next_closure_idx)

                state[symbol] = next_closure_idx + 1

        return np.vstack(tuple(lalr_automaton.values()))

    def clr_to_lalr_automaton(
        self,
        clr_closures: dict[ClosureCLR, int],
        clr_automaton: NDArray[np.int16]
    ) -> NDArray[np.int16]:
        lalr_old_to_new_idx: NDArray[np.int16] = np.zeros(len(clr_closures) + 1, dtype=np.int16)
        lalr_closure_to_idx: dict[tuple[tuple[int, int], ...], int] = {}
        for closure, closure_idx in clr_closures.items():
            lalr_old_to_new_idx[closure_idx + 1] = lalr_closure_to_idx.setdefault(
                tuple((idx, dot) for idx, dot, _ in closure),
                len(lalr_closure_to_idx) + 1
            )

        lalr_old_to_new_sort_idx = np.argsort(lalr_old_to_new_idx[1:])
        lalr_old_to_new_sorted = lalr_old_to_new_idx[1:][lalr_old_to_new_sort_idx]
        _, lalr_new_to_old_counts = np.unique(lalr_old_to_new_sorted, return_counts=True)

        lalr_cumulative_sum = np.cumsum(lalr_new_to_old_counts)
        lalr_new_to_old_idx = np.split(lalr_old_to_new_sort_idx, lalr_cumulative_sum)

        lalr_merged_states = []
        for old_indices in lalr_new_to_old_idx[:-1]:
            old_states = clr_automaton[old_indices]

            shift_mask = (old_states >= 0)
            shift_states = lalr_old_to_new_idx[old_states * shift_mask]
            reduce_states = old_states * np.logical_not(shift_mask)

            shift_state = np.max(shift_states, axis=0)
            reduce_state = np.min(reduce_states, axis=0)

            lalr_merged_states.append(
                np.where(
                    shift_state > 0,
                    shift_state,
                    reduce_state
                )
            )

        return np.vstack(lalr_merged_states)

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

        firsts: list[frozenbitset[int]] = []
        for symbol in self.idx_to_name:
            if symbol < terminal_cutoff:
                firsts.append(frozenbitset({symbol}))
            else:
                first_set = bitset()

                first_passed: bitset[int] = bitset()
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

                firsts.append(frozenbitset(first_set))

        self.terminal_cutoff = terminal_cutoff
        self.rule_sets = tuple(rule_sets)
        self.firsts = firsts

        # a, b = self.clr_make_automaton()
        # self.final_automaton = self.clr_to_lalr_automaton(a, b)
        self.final_automaton = self.lalr_make_automaton()

        # assert np.all(self.clr_to_lalr_automaton(a, b) == self.lalr_make_automaton())

    def __hash__(self) -> int:
        return id(self)


def parse(text: str) -> ShiftReduceParser:
    tokens = tokenize(TokenType, text)
    tokens = list(filter(lambda x: x.type is not TokenType.COMMENT, tokens))

    name_to_idx: dict[str, int] = {}
    idx_to_name: dict[int, str] = {}
    terminals: dict[int, str] = {}
    nonterminals: list[tuple[int, tuple[int, ...]]] = []
    callbacks: list[str] = []

    after_calls = False
    terminal_names = []
    nonterminal_names = []
    for token in tokens:
        if token.type is TokenType.T and not after_calls:
            terminal_names.append(token.text)
        if token.type is TokenType.NT:
            nonterminal_names.append(token.text)
        after_calls = token.type is TokenType.CALLS

    terminal_count = len(set(terminal_names)) + 1
    nonterminal_count = len(set(nonterminal_names)) + 1

    eof_idx = 0
    start_idx = terminal_count

    names = dict.fromkeys(terminal_names + nonterminal_names)
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

        lhs_idx = name_to_idx.setdefault(lhs.text, len(name_to_idx))

        if lhs.type is TokenType.T:
            eq = tokens.pop(0)
            assert eq.type is TokenType.EQ

            pattern = tokens.pop(0)
            assert pattern.type is TokenType.STRING

            terminals[lhs_idx] = ast.literal_eval(pattern.text)

            regex.compile(terminals[lhs_idx])

        if lhs.type is TokenType.NT:
            eq_or_calls = tokens.pop(0)
            assert eq_or_calls.type in (TokenType.EQ, TokenType.CALLS)

            callback: str = None
            if eq_or_calls.type is TokenType.CALLS:
                callback_token = tokens.pop(0)
                assert callback_token.type is TokenType.T

                eq = tokens.pop(0)
                assert eq.type is TokenType.EQ

                callback = callback_token.text

            rhs = tokens[:]
            for idx, token in enumerate(tokens):
                if token.type not in (TokenType.EQ, TokenType.CALLS):
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
