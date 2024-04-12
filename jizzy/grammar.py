from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Iterator, Protocol


class SymbolType(Enum):
    TERMINAL = auto()
    NONTERMINAL = auto()


@dataclass
class T:
    pattern: str = None
    name: str = None


@dataclass
class NT:
    name: str = None



class Rule(ABC):
    def name(self):
        return self.__class__.name
    @staticmethod
    @abstractmethod
    def rules():
        pass
    


@dataclass
class Symbol:
    idx: int
    symbol_type: SymbolType
    name: str = None
    pattern: str = None

    def __repr__(self) -> str:
        if self.symbol_type is SymbolType.TERMINAL:
            type = "terminal"
        else:
            type = "nonterminal"
        return f"<{type} {self.name}: {self.idx}>"


def t(name: str = None, pattern: str = None) -> Symbol:
    return Symbol(
        idx=0,
        symbol_type=SymbolType.TERMINAL,
        name=name,
        pattern=pattern
    )


def nt(name: str = None) -> Symbol:
    return Symbol(
        idx=0,
        symbol_type=SymbolType.NONTERMINAL,
        name=name
    )


class TestEnum(type):
    class_dict: str
    symbols: list[Symbol]

    def __init__(self, member_name, bases, clsdict: dict[str, Any]):
        self.class_name = clsdict["__qualname__"]
        self.symbols = []

        idx: int = 0
        for member_name, value in clsdict.items():
            if callable(value):
                print(member_name, value)

            if not isinstance(value, Symbol):
                continue

            value.name = value.name or member_name
            value.name = f"{self.class_name}.{value.name}"
            value.idx = idx

            idx += 1

            self.symbols.append(value)

    def terminals(self) -> list[Symbol]:
        return [x for x in self if x.symbol_type is SymbolType.TERMINAL]

    def nonterminals(self) -> list[Symbol]:
        return [x for x in self if x.symbol_type is SymbolType.NONTERMINAL]

    def __iter__(self) -> Iterator[Symbol]:
        yield from self.symbols


class Symbols(metaclass=TestEnum):
    IF = t(pattern=r"if")
    ELSE = t(pattern=r"else")
    FOR = t(pattern=r"for")
    IN = t(pattern=r"in")
    WHILE = t(pattern=r"while")
    STRUCT = t(pattern=r"struct")
    RETURN = t(pattern=r"return")

    DOUBLECOLON = t(pattern=r"::")
    SEMICOLON = t(pattern=r";")
    COLON = t(pattern=r":")
    DOT = t(pattern=r"\\.")
    COMMA = t(pattern=r",")

    OP = t(pattern=r"\\(")
    CP = t(pattern=r"\\)")
    OB = t(pattern=r"\\[")
    CB = t(pattern=r"\\]")
    OC = t(pattern=r"\\{")
    CC = t(pattern=r"\\}")

    ASS_IGN = t(pattern=r"=")
    ASS_ADD = t(pattern=r"\\+=")
    ASS_SUB = t(pattern=r"-=")
    ASS_MUL = t(pattern=r"\\*=")
    ASS_DIV = t(pattern=r"/=")
    ASS_SHL = t(pattern=r"<<=")
    ASS_SHR = t(pattern=r">>=")

    BIT_LEFT = t(pattern=r"<<")
    BIT_RIGHT = t(pattern=r">>")

    CMP_IE = t(pattern=r"<=>")

    CMP_LT = t(pattern=r"<")
    CMP_GT = t(pattern=r">")
    CMP_LE = t(pattern=r"<=")
    CMP_GE = t(pattern=r">=")

    CMP_EQ = t(pattern=r"==")
    CMP_NE = t(pattern=r"!=")

    BIT_AND = t(pattern=r"&")
    BIT_XOR = t(pattern=r"\\^")
    BIT_OR = t(pattern=r"\\|")

    LOG_AND = t(pattern=r"&&")
    LOG_OR = t(pattern=r"\\|\\|")

    MATH_ADD = t(pattern=r"\\+")
    MATH_SUB = t(pattern=r"-")

    MATH_MUL = t(pattern=r"\\*")
    MATH_DIV = t(pattern=r"/")
    MATH_MOD = t(pattern=r"%")

    DEC = t(pattern=r"--")
    INC = t(pattern=r"\\+\\+")

    IDENTIFIER = t(pattern=r"\w+")

    LIT_S8 = t(pattern=r"u8\"([^\\\"]|\\.)*\"")
    LIT_S16 = t(pattern=r"u16\"([^\\\"]|\\.)*\"")
    LIT_S32 = t(pattern=r"u32\"([^\\\"]|\\.)*\"")

    LIT_C8 = t(pattern=r"u8\'([^\\\']|\\.)*\'")
    LIT_C16 = t(pattern=r"u16\'([^\\\']|\\.)*\'")
    LIT_C32 = t(pattern=r"u32\'([^\\\']|\\.)*\'")

    LIT_I8 = t(pattern=r"\d+i8")
    LIT_I16 = t(pattern=r"\d+i16")
    LIT_I32 = t(pattern=r"\d+i32")
    LIT_I64 = t(pattern=r"\d+i64")

    LIT_U8 = t(pattern=r"\d+u8")
    LIT_U16 = t(pattern=r"\d+u16")
    LIT_U32 = t(pattern=r"\d+u32")
    LIT_U64 = t(pattern=r"\d+u64")

    LIT_F32 = t(pattern=r"(\d+.\d+|\d+.|.\d+|\d+)f32")
    LIT_F64 = t(pattern=r"(\d+.\d+|\d+.|.\d+|\d+)f64")

    def base_expression(self): ...
    def base_expression(self): ...


class Test2(metaclass=TestEnum):
    A = t("asd")
    B = t("asdasdasd")
    C = A


for test in Symbols:
    print(test)


class TestHint(Protocol):
    def __call__(self, *args: int | float) -> Any: ...

def test(list: list[int | str | float], callback: TestHint):
    b = []
    for a in list:
        if not isinstance(a, str):
            b.append(a)

    b