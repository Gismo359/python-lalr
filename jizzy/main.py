from __future__ import annotations

# from tabulate import tabulate
# from pandas import DataFrame
# from pathlib import Path

# from jizzy.bnf import parse
# from jizzy.engine import Engine
from jizzy.grammar import T, NT


if __name__ == "__main__":
    input_path = Path("sample.bnf")
    input = input_path.read_text(encoding='utf8')
    for _ in range(1):
        parser = parse(input)

    df = DataFrame(data=parser.final_automaton)
    df.index += 1
    df.columns = list(parser.name_to_idx)
    df.index.names = ["State"]
    df.columns.names = ["Symbol"]
    h = [df.index.names[0] + '/' + df.columns.names[0]] + list(df.columns)
    print(tabulate(df, headers=h, tablefmt='pipe'), file=Path("test.txt").open("w"))

    lines = []
    for lhs, rhs in parser.nonterminals:
        parts = [
            parser.idx_to_name[lhs],
            "->",
            *map(parser.idx_to_name.get, rhs)
        ]
        lines.append(" ".join(parts))
    print("\n".join(lines).replace("<", "").replace(">", "").replace("-", "->"), file=Path("test.bnf").open("w"))

    # Path("test.cpp").write_text(parser.generate_cpp())

    result = parser.parse(
        """
            a + c
            b + a * d - f
            if {f(a - v, asdasd)}

            A: B() {}
        """,
        Engine()
    )

    # print(result)
