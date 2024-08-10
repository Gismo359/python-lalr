from __future__ import annotations
from abc import abstractmethod

import regex
import numpy as np

from regex import VERSION1
from functools import cache
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Match, Iterator, TypeVar, cast
from numpy.typing import NDArray

from jizzy.common import Parameter, LexicalElement, NonTerminal, ParseError, Rule, Terminal, Symbol, Token, Node
from jizzy.builder import Builder
from jizzy.helpers import frozenlist
from jizzy.operators import Repeat

T = TypeVar("T", bound=Builder)
U = TypeVar("U", bound=Node)


@dataclass(kw_only=True, frozen=True)
class ClosureItemLR0:
    rule: Rule
    position: int

    def current(self) -> NonTerminal | Terminal:
        return cast(
            NonTerminal | Terminal,
            self.rule.rhs[self.position]
        )


@dataclass(kw_only=True, frozen=True)
class ClosureItemCLR(ClosureItemLR0):
    lookahead: frozenset[Terminal]

    def __hash__(self) -> int:
        return super().__hash__()


@dataclass(kw_only=True)
class ParseState:
    action: int
    value: Node


class GrammarMeta[T: Builder, U: Node](type):
    _terminals: list[Terminal]
    _nonterminals: list[NonTerminal]

    def emplace_terminal(
        self,
        name: str,
        pattern: str | None
    ):
        for terminal in self._terminals:
            if terminal.name == name:
                terminal.pattern = pattern
                return terminal, False

        terminal = Terminal(
            name=name,
            pattern=pattern
        )
        self._terminals.append(terminal)

        setattr(self, name, terminal)

        return terminal, True

    def emplace_nonterminal(
        self,
        name: str,
        generated: bool = False
    ):
        for nonterminal in self._nonterminals:
            if nonterminal.name == name:
                assert nonterminal.generated == generated
                return nonterminal, False

        nonterminal = NonTerminal(
            name=name,
            generated=generated
        )
        self._nonterminals.append(nonterminal)

        setattr(self, name, nonterminal)

        return nonterminal, True

    def generate_nonterminal(
        self,
        name: str,
        idx: int
    ) -> tuple[NonTerminal, int]:

        nonterminal, new = self.emplace_nonterminal(
            name=f"[{idx}] {name}",
            generated=True
        )

        return nonterminal, new

    def unpack_parameter(
        self,
        rule: Rule,
        idx: int,
        element: Parameter
    ) -> LexicalElement:
        rule.rhs[idx] = element.symbol
        rule.parameter_indices.append(idx)
        return element.symbol

    def unpack_repeat(
        self,
        rule: Rule,
        idx: int,
        element: Repeat
    ):
        name = (
            f"{element.name or 'List'} "
            f"({element.element.name}, "
            f"{element.separator.name if element.separator else None}, "
            f"{element.allow_empty})"
        )

        list_builder = element.list_builder

        non_empty_list_symbol, new = self.generate_nonterminal(
            name=name,
            idx=0
        )

        recursive_rhs: list[LexicalElement] = [
            *non_empty_list_symbol,
            *element.element
        ]

        if element.separator is not None:
            recursive_rhs.insert(1, element.separator)

        rules = [
            Rule(
                callback=list_builder.make_list,
                lhs=non_empty_list_symbol,
                rhs=[*element.element]
            ),
            Rule(
                callback=list_builder.expand_list,
                lhs=non_empty_list_symbol,
                rhs=recursive_rhs
            )
        ]

        final_list_symbol = non_empty_list_symbol
        if element.allow_empty:
            list_symbol, new = self.generate_nonterminal(
                name=name,
                idx=1
            )

            rules.extend([
                Rule(
                    callback=self.builder().identity,
                    lhs=list_symbol,
                    rhs=[*non_empty_list_symbol]
                ),
                Rule(
                    callback=list_builder.make_list,
                    lhs=list_symbol,
                    rhs=[]
                )
            ])

            final_list_symbol = list_symbol

        if new:
            self.rules().extend(rules)

        return final_list_symbol

    def unpack_lexical_element(
        self,
        rule: Rule,
        idx: int,
        element: LexicalElement
    ):
        while not isinstance(element, NonTerminal | Terminal):
            if isinstance(element, Parameter):
                element = self.unpack_parameter(rule, idx, element)
            if isinstance(element, Repeat):
                element = self.unpack_repeat(rule, idx, element)
        return element

    def __init__(
        self,
        name: str,
        bases: tuple[type, ...],
        attrs: dict[str, Any]
    ):
        super().__init__(name, bases, attrs)

        self._terminals = []
        self._nonterminals = []

        if not any(isinstance(base, GrammarMeta) for base in bases):
            return

        self.rules = classmethod(cache(self.rules.__func__))  # type: ignore

        _EOF, _ = self.emplace_terminal(name="_EOF", pattern=None)
        _START, _ = self.emplace_nonterminal(name="_START",)
        for base in bases:
            if not isinstance(base, GrammarMeta):
                continue

            if base is Grammar:
                continue

            for terminal in base.terminals():
                self.emplace_terminal(
                    name=terminal.name,
                    pattern=terminal.pattern
                )

            for nonterminal in base.nonterminals():
                if nonterminal.generated:
                    continue

                self.emplace_nonterminal(
                    name=nonterminal.name
                )

        for name, value in self.__dict__.items():
            if isinstance(value, Terminal):
                self.emplace_terminal(
                    name=value.name or name,
                    pattern=value.pattern
                )
            elif isinstance(value, NonTerminal):
                self.emplace_nonterminal(
                    name=value.name or name
                )

        initial_rule = Rule(
            callback=self.builder().noop,
            lhs=_START,
            rhs=[self.start()]
        )
        rules = self.rules()
        rules.insert(0, initial_rule)

        rule_idx = 0
        while rule_idx != len(rules):
            rule = rules[rule_idx]
            rule.idx = rule_idx
            rule.lhs.rules.append(rule)

            rule_idx += 1

            for symbol_idx, symbol in enumerate(rule.rhs):
                rule.rhs[symbol_idx] = self.unpack_lexical_element(
                    rule,
                    symbol_idx,
                    symbol
                )

        for idx, symbol in enumerate(self.symbols()):
            symbol.idx = idx

        updated: set[NonTerminal] = set()
        for nonterminal in self.nonterminals():
            nonterminal.nullable = not all(nonterminal.rules)
            if nonterminal.nullable:
                updated.add(nonterminal)

        while updated:
            current = updated.pop()
            for rule in self.rules():
                if rule.lhs.nullable:
                    continue

                if rule.rhs != [current]:
                    continue

                rule.lhs.nullable = True
                updated.add(rule.lhs)

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

            if isinstance(lhs, NonTerminal):
                default_closure = self.lr0_make_closure(lhs)
                stack.extend(default_closure)

        return frozenlist(new_closures)

    @cache
    def lalr_expand_closure(
        self,
        closure: frozenlist[ClosureItemCLR],
    ) -> list[ClosureItemCLR]:
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
                lookahead = frozenset[Terminal]()

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

            try:
                current_symbol: NonTerminal | Terminal = item.current()
            except IndexError:
                continue

            if isinstance(current_symbol, Terminal):
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

        return lalr_closure

    def lalr_make_automaton(self) -> NDArray[np.int16]:
        rules = self.rules()

        initial_lalr_closure = self.lalr_expand_closure(
            frozenlist([
                ClosureItemCLR(
                    rule=rules[0],
                    position=0,
                    lookahead=frozenset({
                        self._terminals[0]
                    })
                )
            ])
        )

        def slice_closure(
            closure: list[ClosureItemCLR]
        ) -> frozenlist[ClosureItemLR0]:
            return frozenlist(
                ClosureItemLR0(
                    rule=item.rule,
                    position=item.position
                )
                for item in closure
            )

        default_state = [0] * len(self.symbols())

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

    def parse(self, text: str, builder: T | None = None) -> U:
        if builder is None:
            builder = self.builder()()

        eof_token = Token(
            start=len(text),
            stop=len(text),
            text="$",
            type=self._terminals[0]
        )
        tokens = self.tokenize(text)

        automaton = self.lalr_make_automaton()
        rules = self.rules()

        stack: list[ParseState] = [
            ParseState(
                action=1,
                value=Node(start=0, stop=0)
            )
        ]
        while True:
            try:
                token = tokens.pop(0)
            except IndexError:
                token = eof_token

            action: int = automaton[stack[-1].action - 1, token.type.idx]
            if action == 0:
                state = automaton[stack[-1].action - 1]
                state = np.reshape(state, [-1])
                state = cast(list[int], np.flatnonzero(state))
                symbols = self.symbols()
                terminals: list[str] = [
                    symbols[idx].name
                    for idx in state
                    if isinstance(symbols[idx], Terminal)
                ]
                expectation = ", ".join(terminals)
                raise ParseError(
                    f"Unexpected token: {token.text!r} ({token.type.name}), "
                    f"expected one of: {expectation}"
                )

            if action > 0:
                stack.append(
                    ParseState(
                        action=action,
                        value=token
                    )
                )

            try:
                token = tokens[0]
            except IndexError:
                token = eof_token

            while (action := automaton[stack[-1].action - 1, token.type.idx]) < 0:
                if action == -1:
                    # TODO@Daniel:
                    #   Allow for partial parsing
                    assert token.type.idx == 0
                    return cast(U, stack[-1].value)

                rule = rules[-action - 1]

                start_idx = len(stack) - len(rule.rhs)
                stop_idx = len(stack)
                arg_stack = stack[start_idx:stop_idx]
                del stack[start_idx:stop_idx]

                arguments = [
                    arg_stack[idx].value
                    for idx in rule.parameter_indices
                ]

                if arg_stack:
                    start = arg_stack[0].value.start
                    stop = arg_stack[-1].value.stop
                else:
                    start = stack[-1].value.stop
                    stop = token.start

                result = rule.callback(
                    builder,
                    start,
                    stop,
                    *arguments
                )

                action = automaton[stack[-1].action - 1, rule.lhs.idx]
                assert action > 0

                stack.append(
                    ParseState(
                        action=action,
                        value=result
                    )
                )

    @cache
    def follow(
        self,
        item: ClosureItemCLR
    ) -> frozenset[Terminal]:
        right = cast(
            list[Terminal | NonTerminal],
            item.rule.rhs[item.position + 1:]
        )
        if not right:
            return item.lookahead

        follow: set[Terminal] = set()
        while right:
            current = right[0]
            if not isinstance(current, NonTerminal):
                break

            if not current.nullable:
                break

            follow.update(self.first(right.pop(0)))

        if right:
            follow.update(self.first(right.pop(0)))
        else:
            follow.update(item.lookahead)

        return frozenset(follow)

    @cache
    def first(
        self,
        symbol: Terminal | NonTerminal
    ) -> frozenset[Terminal]:
        if isinstance(symbol, Terminal):
            return frozenset({symbol})

        closure = self.lr0_make_closure(symbol)
        closure = self.lr0_expand_closure(closure)

        terminals: set[Terminal] = set()
        for item in closure:
            if not item.rule.rhs:
                continue

            first = item.rule.rhs[0]
            if not isinstance(first, Terminal):
                continue

            terminals.add(first)

        return frozenset(terminals)

    def terminals(self) -> list[Terminal]:
        return self._terminals

    def nonterminals(self) -> list[NonTerminal]:
        return self._nonterminals

    def symbols(self) -> list[Terminal | NonTerminal]:
        return self._terminals + self._nonterminals

    def tokenize(self, text: str) -> list[Token]:
        regexes = [
            f"(?P<_{terminal.idx}>{terminal.pattern})"
            for terminal in self.terminals()
            if terminal.pattern is not None
        ]
        gigaregex = "|".join(regexes)

        symbols = self.symbols()

        def make_token(match: Match[str]) -> Token:
            for key, value in match.groupdict().items():
                if value is None:
                    continue

                return Token(
                    start=match.start(),
                    stop=match.end(),
                    text=value,
                    type=symbols[int(key[1:])]
                )
            assert False

        iterable = cast(
            Iterable[Match[str]],
            regex.finditer(
                gigaregex,
                text,
                flags=VERSION1
            )
        )
        return list(map(make_token, iterable))

    @abstractmethod
    def rules(self) -> list[Rule]:
        pass

    @abstractmethod
    def start(self) -> NonTerminal:
        pass

    @abstractmethod
    def builder(self) -> type[T]:
        pass

    def __iter__(self) -> Iterator[Symbol]:
        yield from self.symbols()


class Grammar[T: Builder, U: Node](metaclass=GrammarMeta):
    @classmethod
    def builder(cls) -> type[T]:
        return GrammarMeta.builder(cls)

    @classmethod
    def parse(cls, text: str, builder: T | None = None) -> U:
        return GrammarMeta.parse(cls, text, builder=builder)
