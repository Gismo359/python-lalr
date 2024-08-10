from __future__ import annotations
from typing import cast

from jizzy.grammar import Terminal, NonTerminal, Rule, Grammar
from jizzy.json.builder import JsonBuilder, Value, Object
from jizzy.operators import Repeat


class StrictJson(Grammar[JsonBuilder, Object]):
    NUMBER = Terminal(pattern=r"\d+(\.\d+)?([eE][\-\+]\d+)?")
    STRING = Terminal(pattern=r"\"([^\\\"]|\\.)*\"")
    BOOLEAN = Terminal(pattern=r"(true|false)")
    NULL = Terminal(pattern=r"null")

    COLON = Terminal(pattern=r":")
    COMMA = Terminal(pattern=r",")

    OC = Terminal(pattern=r"\{")
    CC = Terminal(pattern=r"\}")
    OB = Terminal(pattern=r"\[")
    CB = Terminal(pattern=r"\]")

    VALUE = NonTerminal()
    PAIR = NonTerminal()
    KEY = NonTerminal()

    OBJECT = NonTerminal()
    ARRAY = NonTerminal()

    LIST_BODY = NonTerminal()
    DICT_BODY = NonTerminal()

    NON_EMPTY_LIST_BODY = NonTerminal()
    NON_EMPTY_DICT_BODY = NonTerminal()

    @classmethod
    def builder(cls):
        return JsonBuilder

    @classmethod
    def start(cls):
        return cls.OBJECT

    @classmethod
    def rules(cls):
        builder = cls.builder()
        return [
            Rule(
                callback=builder.make_object,
                lhs=cls.OBJECT,
                rhs=[cls.OC, *cls.DICT_BODY, cls.CC]
            ),
            Rule(
                callback=builder.make_array,
                lhs=cls.ARRAY,
                rhs=[cls.OB, *cls.LIST_BODY, cls.CB]
            ),

            Rule(
                callback=builder.identity,
                lhs=cls.DICT_BODY,
                rhs=[
                    *Repeat(
                        list_builder=builder.DictBodyBuilder,
                        element=cls.PAIR,
                        separator=cls.COMMA
                    )
                ]
            ),

            Rule(
                callback=builder.make_pair,
                lhs=cls.PAIR,
                rhs=[*cls.KEY, cls.COLON, *cls.VALUE]
            ),
            Rule(
                callback=builder.make_string,
                lhs=cls.KEY,
                rhs=[*cls.STRING]
            ),

            Rule(
                callback=builder.identity,
                lhs=cls.LIST_BODY,
                rhs=[
                    *Repeat(
                        list_builder=builder.ListBodyBuilder,
                        element=cls.VALUE,
                        separator=cls.COMMA
                    )
                ]
            ),

            Rule(
                callback=builder.identity,
                lhs=cls.VALUE,
                rhs=[*cls.OBJECT]
            ),
            Rule(
                callback=builder.identity,
                lhs=cls.VALUE,
                rhs=[*cls.ARRAY]
            ),

            Rule(
                callback=builder.make_number,
                lhs=cls.VALUE,
                rhs=[*cls.NUMBER]
            ),
            Rule(
                callback=builder.make_string,
                lhs=cls.VALUE,
                rhs=[*cls.STRING]
            ),
            Rule(
                callback=builder.make_boolean,
                lhs=cls.VALUE,
                rhs=[*cls.BOOLEAN]
            ),
            Rule(
                callback=builder.make_null,
                lhs=cls.VALUE,
                rhs=[cls.NULL]
            )
        ]


class LenientJson(StrictJson):
    @classmethod
    def parse(  # type: ignore
        cls,
        text: str,
        builder: JsonBuilder | None = None
    ) -> Value:
        return super().parse(text, builder=builder)

    @classmethod
    def start(cls) -> NonTerminal:
        return cls.VALUE

    @classmethod
    def rules(cls) -> list[Rule]:
        rules = super().rules()
        for rule in rules:
            if rule.lhs is cls.PAIR:
                rule.rhs[0:1] = [*cls.VALUE]
        return rules
