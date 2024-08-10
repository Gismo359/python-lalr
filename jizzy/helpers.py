from __future__ import annotations

from typing import TypeVar

T = TypeVar("T")


class frozenlist(list[T]):
    def __hash__(self) -> int:  # type: ignore
        return hash(tuple(self))
