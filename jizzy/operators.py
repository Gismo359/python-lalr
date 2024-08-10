from __future__ import annotations

from dataclasses import dataclass

from jizzy.builder import Builder
from jizzy.common import LexicalElement, Symbol


@dataclass(kw_only=True)
class Repeat(LexicalElement):
    list_builder: type[Builder.ListBuilder] = Builder.ListBuilder
    element: Symbol
    separator: Symbol | None = None
    allow_empty: bool = True
