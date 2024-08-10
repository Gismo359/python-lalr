from __future__ import annotations

import pytest

from jizzy.grammar import ParseError
from jizzy.json.parser import StrictJson, LenientJson
from jizzy.json.builder import Object, Array, DictBody, ListBody


def test_lenient_json():
    assert LenientJson.parse("{}") == Object(
        start=0,
        stop=2,
        body=DictBody(
            start=1,
            stop=1,
            items=[]
        )
    )
    assert LenientJson.parse("[]") == Array(
        start=0,
        stop=2,
        body=ListBody(
            start=1,
            stop=1,
            items=[]
        )
    )

    assert LenientJson.parse("{0: 0}").to_python() == {0: 0}
    assert LenientJson.parse("{null: null}").to_python() == {None: None}
    assert LenientJson.parse("{true: true}").to_python() == {True: True}

    assert LenientJson.parse("[]").to_python() == []
    assert LenientJson.parse("0e+1").to_python() == 0
    assert LenientJson.parse("1e+1").to_python() == 10

    assert LenientJson.parse("null").to_python() == None
    assert LenientJson.parse("[1, 2, 3, 4, 5]").to_python() == [1, 2, 3, 4, 5]


def test_strict_json():
    assert StrictJson.parse("{}").to_python() == {}

    assert StrictJson.parse("{\"key\": 0}").to_python() == dict(key=0)

    with pytest.raises(ParseError, match=r"Unexpected token: '\[' \(OB\)"):
        StrictJson.parse("[]")

    with pytest.raises(ParseError, match=r"Unexpected token: 'true' \(BOOLEAN\)"):
        StrictJson.parse("true")
