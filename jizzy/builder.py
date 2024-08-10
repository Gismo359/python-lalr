from __future__ import annotations

from typing import Any, TypeVar, cast

from jizzy.common import Node, List


T = TypeVar("T", bound=Node)
U = TypeVar("U", bound=List[Node])


class Builder:
    def noop(
        self,
        start: int,
        stop: int,
        *_: Node
    ) -> Node:
        return Node(
            start=start,
            stop=stop
        )

    def identity[T: Node](
        self,
        start: int,
        stop: int,
        value: T
    ) -> T:
        assert value.start == start
        assert value.stop == stop

        return value

    class ListBuilder[T: Node, U: List]:
        @classmethod
        def list_type(cls):
            return cast(Any, List[T])

        @classmethod
        def make_list(
            cls,
            builder: Builder,
            start: int,
            stop: int,
            value: T | None = None
        ) -> U:
            return cls.list_type()(
                start=start,
                stop=stop,
                items=[value] if value is not None else []
            )

        @classmethod
        def expand_list(
            cls,
            builder: Builder,
            start: int,
            stop: int,
            list: List[T],
            value: T
        ) -> U:
            return cls.list_type()(
                start=start,
                stop=stop,
                items=list.items + [value]
            )
