from dataclasses import dataclass
from typing import TypeVar, Generic, Iterator
from enum import IntEnum, auto

from jizzy.grammar import Token, Node

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
    if any is None:
        return ""
    return "\n".join("    " + line for line in str(any).splitlines())


@dataclass(kw_only=True)
class List(Node, Generic[T]):
    items: list[T]

    def append(self, item: T):
        self.items.append(item)

    def __iter__(self) -> Iterator[T]:
        return iter(self.items)

    def __str__(self) -> str:
        pass


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
    terminator: Token = None

    def __str__(self) -> str:
        if self.terminator is not None:
            return f"{self.expression}{self.terminator}"
        else:
            return f"{self.expression}"


@dataclass(kw_only=True)
class Block(Node):
    brace_type: BraceType
    break_type: ListBreakType
    items: List[Expression]

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

        if self.items is None:
            return f"{start}{stop}"

        match self.break_type:
            case ListBreakType.Wrap:
                # TODO@Daniel: Implement this some day
                return ""
            case ListBreakType.Always:
                body = "\n".join(map(str, self.items))
                return f"{start}\n{indent(body)}\n{stop}"
            case ListBreakType.Never:
                body = " ".join(map(str, self.items))
                return f"{start}{body}{stop}"


@dataclass(kw_only=True)
class BlockExpression(Expression):
    expression: Expression
    block: Block

    def __str__(self) -> str:
        return f"{self.expression}{self.block}"


@dataclass(kw_only=True)
class Program(Node):
    items: List[Expression]

    def __str__(self) -> str:
        return "\n".join(map(str, self.items))


@dataclass
class Engine:
    indentation: int = 0

    def indent(
        self,
        start: int,
        stop: int,
        token: Token
    ) -> Token:
        self.indentation += 1
        return token

    def unindent(
        self,
        start: int,
        stop: int,
        token: Token
    ) -> Token:
        self.indentation -= 1
        return token

    def identity(
        self,
        start: int,
        stop: int,
        any: T
    ) -> T:
        return any

    def make_paren_block(
        self,
        start: int,
        stop: int,
        items: List[Expression] = None
    ) -> Block:
        return Block(
            start=start,
            stop=stop,
            brace_type=BraceType.Round,
            break_type=ListBreakType.Always,
            items=items
        )

    def make_brace_block(
        self,
        start: int,
        stop: int,
        items: List[Expression] = None
    ) -> Block:
        return Block(
            start=start,
            stop=stop,
            brace_type=BraceType.Square,
            break_type=ListBreakType.Always,
            items=items
        )

    def make_curly_block(
        self,
        start: int,
        stop: int,
        items: List[Expression] = None
    ) -> Block:
        return Block(
            start=start,
            stop=stop,
            brace_type=BraceType.Curly,
            break_type=ListBreakType.Always,
            items=items
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
        terminator: Token = None
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
        expression: Expression
    ) -> List[Expression]:
        return List[Expression](
            start=start,
            stop=stop,
            items=[expression]
        )

    def expand_expression_list(
        self,
        start: int,
        stop: int,
        list: List[Expression],
        expression: Expression
    ) -> List[Expression]:
        return List[Expression](
            start=start,
            stop=stop,
            items=list.items + [expression]
        )

    def finalize_program(
        self,
        start: int,
        stop: int,
        items: List[Expression]
    ) -> Program:
        return Program(
            start=start,
            stop=stop,
            items=items
        )
