"""Cut one species out of upstream's raw NASA-9 table and write it in the per-species layout the readers take.

Usage: python make_nasa9_file.py <new_nasa9.TXT> <header substring> <out file>
The per-species file holds the 200-1000 K set, then the 1000-6000 K set, two lines each, then the header as a comment.
The raw table packs five 16-character fields per line; the per-species files keep the first field as it is and put one
space before each later one.
"""

import sys
from pathlib import Path

FIELD = 16
RANGES_K = ((200.0, 1000.0), (1000.0, 6000.0))  # the two sets the readers use, in file order

raw, key, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
lines = raw.read_text().splitlines()
starts = [i for i, line in enumerate(lines) if key in line and "HF298" in line]
if len(starts) != 1:
    raise SystemExit(f"{key!r} matches {len(starts)} header lines in {raw}; give a substring that matches one")
start = starts[0]
n_ranges = int(lines[start + 1].split()[0])
blocks = {}
for j in range(n_ranges):
    head, c1, c2 = lines[start + 2 + 3 * j : start + 5 + 3 * j]
    blocks[(float(head.split()[0]), float(head.split()[1]))] = (c1, c2)
missing = [r for r in RANGES_K if r not in blocks]
if missing:
    raise SystemExit(f"{key!r} has ranges {sorted(blocks)} K; missing {missing}")


def spaced(line: str) -> str:
    """Return a raw-table line in the per-species layout: the first field as is, one space before each later one."""
    fields = [line[i : i + FIELD] for i in range(0, 5 * FIELD, FIELD)]
    return fields[0] + "".join(" " + f.strip() for f in fields[1:])


rows = [spaced(c) for r in RANGES_K for c in blocks[r]]
out.write_text("\n".join(rows) + "\n# " + lines[start])
print(f"{out.name}: ranges in the table {sorted(blocks)} K; wrote {RANGES_K}")
