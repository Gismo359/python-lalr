from __future__ import annotations

import ast

from typing import TypeVar
from dataclasses import dataclass

from jizzy.builder import Builder
from jizzy.common import List, Node, Token


T = TypeVar("T", bound=Node)


def indent(any: T) -> str:
    return "\n".join("    " + line for line in str(any).splitlines())


@dataclass(kw_only=True)
class Value(Node):
    def to_python(self) -> object:
        pass


@dataclass(kw_only=True)
class Number(Value):
    value: Token

    def to_python(self) -> float:
        return ast.literal_eval(self.value.text)

    def __str__(self) -> str:
        return str(self.value)


@dataclass(kw_only=True)
class String(Value):
    value: Token

    def to_python(self) -> str:
        return ast.literal_eval(self.value.text)

    def __str__(self) -> str:
        return str(self.value)


@dataclass(kw_only=True)
class Boolean(Value):
    value: Token

    def to_python(self) -> bool:
        return self.value.text == "true"

    def __str__(self) -> str:
        return str(self.value)


@dataclass(kw_only=True)
class Null(Value):
    def to_python(self) -> None:
        return None

    def __str__(self) -> str:
        return "null"


@dataclass(kw_only=True)
class Pair(Node):
    key: Value
    value: Value

    def __str__(self) -> str:
        return f"{self.key}: {self.value}"


@dataclass(kw_only=True)
class ListBody(List[Value]):
    def __str__(self) -> str:
        return ",\n".join(map(str, self))


@dataclass(kw_only=True)
class DictBody(List[Pair]):
    def __str__(self) -> str:
        return ",\n".join(map(str, self))


@dataclass(kw_only=True)
class Array(Node):
    body: ListBody

    def to_python(self) -> list:
        return [value.to_python() for value in self.body]

    def __str__(self) -> str:
        return f"[\n{indent(self.body)}\n]"


@dataclass(kw_only=True)
class Object(Value):
    body: DictBody

    def to_python(self) -> dict:
        return {
            pair.key.to_python(): pair.value.to_python()
            for pair in self.body
        }

    def __str__(self) -> str:
        return f"{{\n{indent(self.body)}\n}}"


class JsonBuilder(Builder):
    class ListBodyBuilder(Builder.ListBuilder[Value, ListBody]):
        @classmethod
        def list_type(cls):
            return ListBody

    class DictBodyBuilder(Builder.ListBuilder[Pair, DictBody]):
        @classmethod
        def list_type(cls):
            return DictBody

    def make_number(
        self,
        start: int,
        stop: int,
        value: Token
    ) -> Number:
        return Number(
            start=start,
            stop=stop,
            value=value
        )

    def make_string(
        self,
        start: int,
        stop: int,
        value: Token
    ) -> String:
        return String(
            start=start,
            stop=stop,
            value=value
        )

    def make_boolean(
        self,
        start: int,
        stop: int,
        value: Token
    ) -> Boolean:
        return Boolean(
            start=start,
            stop=stop,
            value=value
        )

    def make_null(
        self,
        start: int,
        stop: int
    ) -> Null:
        return Null(
            start=start,
            stop=stop
        )

    def make_pair(
        self,
        start: int,
        stop: int,
        key: Value,
        value: Value
    ) -> Pair:
        return Pair(
            start=start,
            stop=stop,
            key=key,
            value=value
        )

    def make_array(
        self,
        start: int,
        stop: int,
        body: ListBody
    ) -> Array:
        return Array(
            start=start,
            stop=stop,
            body=body
        )

    def make_object(
        self,
        start: int,
        stop: int,
        body: DictBody
    ) -> Object:
        return Object(
            start=start,
            stop=stop,
            body=body
        )
