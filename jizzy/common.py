from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator, TYPE_CHECKING

if TYPE_CHECKING:
    from jizzy.builder import Builder

class ParseError(Exception):
    pass


class ReduceReduceConflict(Exception):
    pass


class ShiftReduceConflict(Exception):
    pass


@dataclass(kw_only=True)
class Node:
    start: int
    stop: int



@dataclass(kw_only=True)
class List[T: Node](Node):
    items: list[T]

    def append(self, item: T):
        self.items.append(item)

    def __getitem__(self, key: int) -> T:
        return self.items[key]

    def __iter__(self) -> Iterator[T]:
        yield from self.items


@dataclass(kw_only=True)
class Token(Node):
    text: str
    type: Symbol

    def __str__(self) -> str:
        return self.text


@dataclass(kw_only=True)
class LexicalElement:
    name: str = ""

    def __iter__(self) -> Iterator[Parameter]:
        yield Parameter(symbol=self)



@dataclass(kw_only=True)
class Symbol(LexicalElement):
    idx: int = 0


@dataclass(kw_only=True)
class Terminal(Symbol):
    pattern: str | None

    def __hash__(self) -> int:
        return self.idx

    def __repr__(self) -> str:
        return f"<terminal {self.name!r}: {self.idx}>"


@dataclass(kw_only=True)
class NonTerminal(Symbol):
    rules: list[Rule] = field(default_factory=list)
    nullable: bool = False
    generated: bool = False

    def __hash__(self) -> int:
        return self.idx

    def __repr__(self) -> str:
        return f"<nonterminal {self.name!r}: {self.idx}>"


@dataclass(kw_only=True)
class Repeated(LexicalElement):
    symbol: Symbol


@dataclass(kw_only=True)
class Parameter(LexicalElement):
    symbol: LexicalElement


@dataclass(kw_only=True)
class Rule:
    idx: int = 0
    callback: Callable[..., Node]
    lhs: NonTerminal
    rhs: list[LexicalElement]
    parameter_indices: list[int] = field(default_factory=list)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Rule):
            return False

        return (
            self.idx,
            self.callback,
            self.lhs.name,
            tuple(rhs.name for rhs in self.rhs),
            self.parameter_indices
        ) == (
            other.idx,
            self.callback,
            other.lhs.name,
            tuple(rhs.name for rhs in other.rhs),
            other.parameter_indices
        )

    def __hash__(self) -> int:
        return self.idx
