from __future__ import annotations

from jizzy.engine import Engine
from jizzy.jizzy_grammar import Symbols

if __name__ == "__main__":
    result = Symbols.parse(
        """
            a + c
            b + a * d - f
            if {f(a - v, asdasd)}

            A: B() {}
        """,
        Engine()
    )

    print(result)
