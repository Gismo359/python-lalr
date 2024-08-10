from __future__ import annotations

import numpy as np

from unittest.mock import patch

from jizzy.builder import Builder
from jizzy.grammar import Grammar, Repeat, Rule, Terminal, NonTerminal, Token, Node


class TestBuilder(Builder):
    def first_rule(self, start: int, stop: int):
        return Node(start=start, stop=stop)

    def second_rule(self, start: int, stop: int):
        return Node(start=start, stop=stop)


class TestLanguage(Grammar[TestBuilder, Node]):
    A = Terminal(pattern=r"\(")
    B = Terminal(pattern=r"\)")
    C = Terminal(pattern=r"\w+")

    D = NonTerminal()

    @classmethod
    def builder(cls) -> type[TestBuilder]:
        return TestBuilder

    @classmethod
    def start(cls) -> NonTerminal:
        return cls.D

    @classmethod
    def rules(cls) -> list[Rule]:
        return [
            Rule(
                callback=cls.builder().first_rule,
                lhs=cls.D,
                rhs=[cls.C]
            ),
            Rule(
                callback=cls.builder().second_rule,
                lhs=cls.D,
                rhs=[cls.A, cls.D, cls.B]
            )
        ]


def test_metaclass():
    assert TestLanguage.A.idx == 1
    assert TestLanguage.B.idx == 2
    assert TestLanguage.C.idx == 3
    assert TestLanguage.D.idx == 5

    assert TestLanguage.A.name == "A"
    assert TestLanguage.B.name == "B"
    assert TestLanguage.C.name == "C"
    assert TestLanguage.D.name == "D"

    dummy_rule, first_rule, second_rule = TestLanguage.rules()

    assert dummy_rule.idx == 0
    assert first_rule.idx == 1
    assert second_rule.idx == 2


def test_tokenizer():
    assert TestLanguage.tokenize("(a)") == [
        Token(start=0, stop=1, text="(", type=TestLanguage.A),
        Token(start=1, stop=2, text="a", type=TestLanguage.C),
        Token(start=2, stop=3, text=")", type=TestLanguage.B)
    ]


def test_parse_result():
    result = TestLanguage.parse("(a)")

    assert result.start == 0
    assert result.stop == 3


def test_inheritance():
    class InheritedLanguage(TestLanguage):
        @classmethod
        def rules(cls) -> list[Rule]:
            return super().rules()

    assert InheritedLanguage.A == TestLanguage.A
    assert InheritedLanguage.B == TestLanguage.B
    assert InheritedLanguage.C == TestLanguage.C
    assert InheritedLanguage.D == TestLanguage.D

    assert InheritedLanguage.A is not TestLanguage.A
    assert InheritedLanguage.B is not TestLanguage.B
    assert InheritedLanguage.C is not TestLanguage.C
    assert InheritedLanguage.D is not TestLanguage.D

    assert InheritedLanguage.rules() == TestLanguage.rules()
    assert InheritedLanguage.rules() is not TestLanguage.rules()

    for lhs, rhs in zip(
        InheritedLanguage.rules(),
        TestLanguage.rules()
    ):
        assert lhs == rhs
        assert lhs is not rhs

    assert np.all(
        InheritedLanguage.lalr_make_automaton() ==
        TestLanguage.lalr_make_automaton()
    )

    text = "(a)"
    assert InheritedLanguage.parse(text) == TestLanguage.parse(text)


def test_repeat():
    class RepeatLanguage(Grammar[Builder, Node]):
        T = Terminal(pattern=r"a")
        U = Terminal(pattern=r"b")

        NT = NonTerminal()

        @classmethod
        def builder(cls) -> type[Builder]:
            return Builder

        @classmethod
        def start(cls) -> NonTerminal:
            return cls.NT

        @classmethod
        def rules(cls) -> list[Rule]:
            return [
                Rule(
                    callback=cls.builder().noop,
                    lhs=cls.NT,
                    rhs=[Repeat(element=cls.T)]
                )
            ]

    RepeatLanguage.parse("")
    RepeatLanguage.parse("a")
    RepeatLanguage.parse("aa")
    RepeatLanguage.parse("aaa")
    RepeatLanguage.parse("aaaa")
    RepeatLanguage.parse("aaaaa")
    RepeatLanguage.parse("aaaaaa")


def test_parse_calls():
    with (
        patch.object(TestBuilder, TestBuilder.first_rule.__name__) as first_rule,
        patch.object(TestBuilder, TestBuilder.second_rule.__name__) as second_rule
    ):
        class DummyLanguage(TestLanguage):
            pass

        DummyLanguage.parse("(a)")
        DummyLanguage.rules()

    first_rule.assert_called_once()
    second_rule.assert_called_once()
