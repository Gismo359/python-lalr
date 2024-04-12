from __future__ import annotations

import regex
from regex import VERSION1
from dataclasses import dataclass
from enum import Enum
from typing import Match, Type, TypeVar, Generic

T = TypeVar("T", bound=Enum)


@dataclass(frozen=True)
class Token(Generic[T]):
    start: int
    stop: int
    text: str
    type: T

    def __str__(self) -> str:
        return self.text


def make_token(token_type: Type[T], match: Match) -> Token[T]:
    start = match.start()
    stop = match.end()
    for key, value in match.groupdict().items():
        if value is not None:
            return Token[T](
                start=start,
                stop=stop,
                text=value,
                type=token_type[key]
            )
    return None


def tokenize(token_type: Type[T], text: str) -> list[Token[T]]:
    regexes = [f"(?P<{type.name}>{type.value})" for type in token_type]
    gigaregex = "|".join(regexes)

    return [
        make_token(token_type, match)
        for match in regex.finditer(gigaregex, text, flags=VERSION1)
    ]
