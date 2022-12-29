from __future__ import annotations

from tabulate import tabulate
from pandas import DataFrame
from pathlib import Path

from jizzy.bnf import parse

input_path = Path("sample.bnf")
parser = parse(input_path.read_text(encoding='utf8'))

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


parser.parse(
    """
    main: void (){
        a + b.c;
    }   
    """
)
