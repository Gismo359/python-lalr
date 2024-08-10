from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar
from enum import IntEnum, auto

from jizzy.builder import Builder
from jizzy.common import List, Node, Token


T = TypeVar("T", bound=Node)


class ListBreakType(IntEnum):
    Always = auto()
    Never = auto()
    Wrap = auto()


class BraceType(IntEnum):
    Round = auto()
    Square = auto()
    Curly = auto()


def indent(any: T) -> str:
    return "\n".join("    " + line for line in str(any).splitlines())


@dataclass(kw_only=True)
class Expression(Node):
    pass


@dataclass(kw_only=True)
class UnaryPrefix(Expression):
    op: Token
    expression: Expression

    def __str__(self) -> str:
        return f"{self.op}{self.expression}"


@dataclass(kw_only=True)
class UnaryPostfix(Expression):
    op: Token
    expression: Expression

    def __str__(self) -> str:
        return f"{self.expression}{self.op}"


@dataclass(kw_only=True)
class BinaryExpression(Expression):
    lhs: Expression
    op: Token
    rhs: Expression

    def __str__(self) -> str:
        return f"{self.lhs} {self.op} {self.rhs}"


@dataclass(kw_only=True)
class Statement(Node):
    expression: Expression
    terminator: Token | None = None

    def __str__(self) -> str:
        if self.terminator is not None:
            return f"{self.expression}{self.terminator}"
        else:
            return f"{self.expression}"


@dataclass(kw_only=True)
class ExpressionList(List[Expression]):
    def __str__(self) -> str:
        return "\n".join(map(str, self))


@dataclass(kw_only=True)
class Block(Node):
    brace_type: BraceType
    body: ExpressionList

    def __str__(self) -> str:
        match self.brace_type:
            case BraceType.Round:
                start = "("
                stop = ")"
            case BraceType.Square:
                start = "["
                stop = "]"
            case BraceType.Curly:
                start = "{"
                stop = "}"

        return f"{start}\n{indent(self.body)}\n{stop}"


@dataclass(kw_only=True)
class BlockExpression(Expression):
    expression: Expression
    block: Block

    def __str__(self) -> str:
        return f"{self.expression}{self.block}"


class JizzBuilder(Builder):
    def make_paren_block(
        self,
        start: int,
        stop: int,
        body: ExpressionList
    ) -> Block:
        return Block(
            start=start,
            stop=stop,
            brace_type=BraceType.Round,
            body=body
        )

    def make_brace_block(
        self,
        start: int,
        stop: int,
        body: ExpressionList
    ) -> Block:
        return Block(
            start=start,
            stop=stop,
            brace_type=BraceType.Square,
            body=body
        )

    def make_curly_block(
        self,
        start: int,
        stop: int,
        body: ExpressionList
    ) -> Block:
        return Block(
            start=start,
            stop=stop,
            brace_type=BraceType.Curly,
            body=body
        )

    def make_block_expression(
        self,
        start: int,
        stop: int,
        expression: Expression,
        block: Block
    ):
        return BlockExpression(
            start=start,
            stop=stop,
            expression=expression,
            block=block
        )

    def make_prefix_expression(
        self,
        start: int,
        stop: int,
        op: Token,
        expression: Expression
    ) -> UnaryPrefix:
        return UnaryPrefix(
            start=start,
            stop=stop,
            op=op,
            expression=expression
        )

    def make_postfix_expression(
        self,
        start: int,
        stop: int,
        expression: Expression,
        op: Token
    ) -> UnaryPostfix:
        return UnaryPostfix(
            start=start,
            stop=stop,
            op=op,
            expression=expression
        )

    def make_binary_expression(
        self,
        start: int,
        stop: int,
        lhs: Expression,
        op: Token,
        rhs: Expression
    ) -> BinaryExpression:
        return BinaryExpression(
            start=start,
            stop=stop,
            lhs=lhs,
            op=op,
            rhs=rhs
        )

    def make_statement(
        self,
        start: int,
        stop: int,
        expression: Expression,
        terminator: Token | None = None
    ) -> Statement:
        return Statement(
            start=start,
            stop=stop,
            expression=expression,
            terminator=terminator
        )

    def make_expression_list(
        self,
        start: int,
        stop: int,
        expression: Expression | None = None
    ) -> ExpressionList:
        return ExpressionList(
            start=start,
            stop=stop,
            items=[expression] if expression is not None else []
        )

    def expand_expression_list(
        self,
        start: int,
        stop: int,
        list: ExpressionList,
        expression: Expression
    ) -> ExpressionList:
        return ExpressionList(
            start=start,
            stop=stop,
            items=list.items + [expression]
        )
