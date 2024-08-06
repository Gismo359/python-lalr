from __future__ import annotations

import regex
import numpy as np

from regex import VERSION1
from functools import cache
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Match, Iterator, Callable, Generic, TypeVar, TypeVarTuple, cast
from numpy.typing import NDArray

T = TypeVar("T")
Ts = TypeVarTuple("Ts")


class frozenlist(list[T]):
    def __hash__(self) -> int:  # type: ignore
        return hash(tuple(self))


class SymbolType(Enum):
    TERMINAL = auto()
    NONTERMINAL = auto()


@dataclass(kw_only=True)
class Node:
    start: int
    stop: int


@dataclass(kw_only=True)
class Token(Node):
    text: str
    type: Symbol

    def __str__(self) -> str:
        return self.text


@dataclass(kw_only=True)
class Rule:
    idx: int = None
    callback: Callable[[Any, int, int, *Ts], Node]
    lhs: NonTerminal
    rhs: list[Symbol]
    parameter_indices: list[int] = field(default_factory=list)

    def __post_init__(self):
        for idx, symbol in enumerate(self.rhs):
            if not isinstance(symbol, DummySymbol):
                continue

            self.rhs[idx] = symbol.symbol
            self.parameter_indices.append(idx)

    def __hash__(self) -> int:
        return self.idx


@dataclass(kw_only=True)
class Symbol:
    idx: int = None
    name: str = None
    symbol_type: SymbolType

    def __hash__(self) -> int:
        return self.idx

    def __iter__(self) -> Iterator[DummySymbol]:
        yield DummySymbol(symbol=self)


@dataclass(kw_only=True)
class Terminal(Symbol):
    pattern: str
    symbol_type: SymbolType = SymbolType.TERMINAL

    def __hash__(self) -> int:
        return self.idx

    def __repr__(self) -> str:
        return f"<terminal {self.name}: {self.idx}>"


@dataclass(kw_only=True)
class NonTerminal(Symbol):
    rules: list[Rule] = field(default_factory=list)
    symbol_type: SymbolType = SymbolType.NONTERMINAL

    def __hash__(self) -> int:
        return self.idx

    def __repr__(self) -> str:
        return f"<nonterminal {self.name}: {self.idx}>"


@dataclass(kw_only=True)
class DummySymbol:
    symbol: Symbol


@dataclass(kw_only=True, frozen=True)
class ClosureItemLR0:
    rule: Rule
    position: int

    def current(self):
        return self.rule.rhs[self.position]


@dataclass(kw_only=True, frozen=True)
class ClosureItemCLR(ClosureItemLR0):
    lookahead: frozenset[Terminal]

    def __hash__(self) -> int:
        return super().__hash__()


@dataclass(kw_only=True)
class ParseState:
    state: int
    value: Node


class GrammarMeta(type, Generic[T]):
    cutoff: int
    symbols: list[Symbol]

    def __init__(self, name, bases, attrs):
        super().__init__(name, bases, attrs)

        _EOF = Terminal(pattern=None)
        _START = NonTerminal()
        terminal_names: dict[str, Terminal] = dict(_EOF=_EOF)
        nonterminal_names: dict[str, Symbol] = dict(_START=_START)
        for name, value in self.__dict__.items():
            if not isinstance(value, Symbol):
                continue

            match value:
                case Terminal():
                    terminal_names[name] = value
                case NonTerminal():
                    nonterminal_names[name] = value

        all_names = terminal_names | nonterminal_names
        for idx, (name, symbol) in enumerate(all_names.items()):
            symbol.idx = idx
            symbol.name = symbol.name or name

        self.cutoff = len(terminal_names)
        self.symbols = [*terminal_names.values(), *nonterminal_names.values()]

        rules = self.rules()
        rules.insert(0, Rule(callback=None, lhs=_START, rhs=[self.start()]))
        for idx, rule in enumerate(rules):
            rule.idx = idx
            rule.lhs.rules.append(rule)

    @cache
    def lr0_make_closure(
        self,
        symbol: NonTerminal
    ) -> frozenlist[ClosureItemLR0]:
        return frozenlist([
            ClosureItemLR0(rule=rule, position=0)
            for rule in symbol.rules
        ])

    @cache
    def lr0_expand_closure(
        self,
        closure: frozenlist[ClosureItemLR0]
    ) -> frozenlist[ClosureItemLR0]:
        new_closures: dict[ClosureItemLR0, None] = {}

        idx = 0
        stack: list[ClosureItemLR0] = list(closure)
        while True:
            try:
                item = stack[idx]
            except IndexError:
                break

            idx += 1

            if item in new_closures:
                continue

            new_closures.setdefault(item)

            try:
                lhs = item.current()
            except IndexError:
                continue

            if lhs.symbol_type is SymbolType.TERMINAL:
                continue

            default_closure = self.lr0_make_closure(lhs)
            stack.extend(default_closure)

        return frozenlist(new_closures)

    @cache
    def lalr_expand_closure(
        self,
        closure: frozenlist[ClosureItemCLR],
    ) -> frozenlist[ClosureItemCLR]:
        lr0_closure = self.lr0_expand_closure(closure)
        lr_closure_to_idx = {
            closure: idx
            for idx, closure
            in enumerate(lr0_closure)
        }

        updated: set[int] = set()
        lalr_closure: list[ClosureItemCLR] = []
        for idx, item in enumerate(lr0_closure):
            try:
                lookahead = closure[idx].lookahead
            except IndexError:
                lookahead = frozenset()

            lalr_closure.append(
                ClosureItemCLR(
                    rule=item.rule,
                    position=item.position,
                    lookahead=lookahead
                )
            )

            try:
                item.current()
                updated.add(idx)
            except IndexError:
                pass

        while updated:
            item = lalr_closure[updated.pop()]

            current_symbol = item.current()
            if current_symbol.symbol_type is SymbolType.TERMINAL:
                continue

            follow = self.follow(item)
            for lr0_item in self.lr0_make_closure(current_symbol):
                item_idx = lr_closure_to_idx[lr0_item]
                lalr_item = lalr_closure[item_idx]

                old_lookahead = lalr_item.lookahead
                new_lookahead = lalr_item.lookahead | follow

                if len(old_lookahead) != len(new_lookahead):
                    lalr_closure[item_idx] = ClosureItemCLR(
                        rule=lalr_item.rule,
                        position=lalr_item.position,
                        lookahead=new_lookahead
                    )
                    updated.add(item_idx)

        return frozenlist(lalr_closure)

    def lalr_make_automaton(self) -> NDArray[np.int16]:
        rules = self.rules()

        initial_lalr_closure = self.lalr_expand_closure(
            frozenlist([
                ClosureItemCLR(
                    rule=rules[0],
                    position=0,
                    lookahead=frozenset({
                        cast(Terminal, self.symbols[0])
                    })
                )
            ])
        )

        def slice_closure(
            closure: frozenlist[ClosureItemCLR]
        ) -> frozenlist[ClosureItemLR0]:
            return frozenlist(
                ClosureItemLR0(
                    rule=item.rule,
                    position=item.position
                )
                for item in closure
            )

        default_state = [0] * len(self.symbols)

        closures = [initial_lalr_closure]
        closure_to_idx = {slice_closure(initial_lalr_closure): 0}

        automaton: dict[int, list[int]] = defaultdict(default_state.copy)

        updated: set[int] = {0}
        while updated:
            current_closure_idx = updated.pop()
            current_closure = closures[current_closure_idx]

            symbol_to_shift: defaultdict[Symbol, list[ClosureItemCLR]] = defaultdict(list)
            symbol_to_reduce: defaultdict[Terminal, list[Rule]] = defaultdict(list)
            for item in current_closure:
                try:
                    symbol = item.current()
                    shift_closure = symbol_to_shift[symbol]
                    symbol_to_shift[symbol]
                    shift_closure.append(
                        ClosureItemCLR(
                            rule=item.rule,
                            position=item.position + 1,
                            lookahead=item.lookahead
                        )
                    )
                except IndexError:
                    for symbol in item.lookahead:
                        symbol_to_reduce[symbol].append(item.rule)

            state = automaton[current_closure_idx]
            for symbol, rules in symbol_to_reduce.items():
                assert len(rules) == 1
                rule, = rules

                next_state = state[symbol.idx]
                if next_state < 0:
                    assert next_state == -rule.idx - 1
                elif next_state == 0:
                    state[symbol.idx] = -rule.idx - 1

            for symbol, closure in symbol_to_shift.items():
                closure = self.lalr_expand_closure(frozenlist(closure))
                closure_idx = closure_to_idx.setdefault(
                    slice_closure(closure),
                    len(closure_to_idx)
                )
                if closure_idx == len(closures):
                    closures.append(closure)
                    updated.add(closure_idx)
                else:
                    needs_update = False
                    for idx, (old_item, new_item) in enumerate(
                        zip(
                            closures[closure_idx],
                            closure
                        )
                    ):
                        new_lookahead = old_item.lookahead | new_item.lookahead

                        closures[closure_idx][idx] = ClosureItemCLR(
                            rule=old_item.rule,
                            position=old_item.position,
                            lookahead=new_lookahead
                        )

                        needs_update |= len(new_lookahead) != len(old_item.lookahead)

                    if needs_update:
                        updated.add(closure_idx)

                state[symbol.idx] = closure_idx + 1

        return np.vstack(tuple(automaton.values()))

    def parse(self, text: str, engine: T):
        tokens = self.tokenize(text)
        tokens.append(
            Token(
                start=len(text),
                stop=len(text),
                text="$",
                type=cast(Terminal, self.symbols[0])
            )
        )

        automaton = self.lalr_make_automaton()
        rules = self.rules()

        stack: list[ParseState] = [
            ParseState(
                state=1,
                value=None
            )
        ]
        while tokens:
            token = tokens.pop(0)

            action: int = automaton[stack[-1].state - 1, token.type.idx]
            assert action != 0

            if action > 0:
                stack.append(
                    ParseState(
                        state=action,
                        value=token
                    )
                )

            if action == -1:
                assert token.type.idx == 0
                return stack[-1].value

            token = tokens[0]
            while (action := automaton[stack[-1].state - 1, token.type.idx]) < -1:
                rule = rules[-action - 1]
                arg_stack = stack[-len(rule.rhs):]
                del stack[-len(rule.rhs):]

                arguments = [
                    arg_stack[idx].value
                    for idx in rule.parameter_indices
                ]

                result = rule.callback(
                    engine,
                    arg_stack[0].value.start,
                    arg_stack[-1].value.stop,
                    *arguments
                )
                action = automaton[stack[-1].state - 1, rule.lhs.idx]
                assert action > 0

                stack.append(
                    ParseState(
                        state=action,
                        value=result
                    )
                )

    @cache
    def follow(
        self,
        item: ClosureItemCLR
    ) -> frozenset[Terminal]:
        try:
            return self.first(item.rule.rhs[item.position + 1])
        except:
            return item.lookahead

    @cache
    def first(
        self,
        symbol: Symbol
    ) -> frozenset[Terminal]:
        match symbol:
            case Terminal():
                return frozenset({symbol})
            case NonTerminal():
                terminals: set[Terminal] = set()
                nonterminals: set[NonTerminal] = set()
                stack: set[NonTerminal] = {symbol}
                while stack:
                    current = stack.pop()
                    if current in nonterminals:
                        continue

                    nonterminals.add(current)

                    for rule in current.rules:
                        first = rule.rhs[0]
                        match first:
                            case Terminal():
                                terminals.add(first)
                            case NonTerminal():
                                stack.add(first)

                return frozenset(terminals)
            case _:
                return None

    def terminals(self) -> list[Terminal]:
        return cast(list[Terminal], self.symbols[:self.cutoff])

    def nonterminals(self) -> list[NonTerminal]:
        return cast(list[NonTerminal], self.symbols[self.cutoff:])

    def tokenize(self, text: str) -> list[Token]:
        regexes = [
            f"(?P<_{terminal.idx}>{terminal.pattern})"
            for terminal in self.terminals()
            if terminal.pattern is not None
        ]
        gigaregex = "|".join(regexes)

        def make_token(match: Match) -> Token:
            for key, value in match.groupdict().items():
                if value is None:
                    continue

                return Token(
                    start=match.start(),
                    stop=match.end(),
                    text=value,
                    type=self.symbols[int(key[1:])]
                )
            return None

        return list(map(make_token, regex.finditer(gigaregex, text, flags=VERSION1)))

    @cache
    def rules(self) -> list[Rule]:
        pass

    @cache
    def start(self) -> NonTerminal:
        pass

    def __iter__(self) -> Iterator[Symbol]:
        yield from self.symbols
