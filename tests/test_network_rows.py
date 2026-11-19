"""Network rows VULCAN 2 cannot read the way the port would: refused at parse time."""

from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    ("rows", "match"),
    [
        ("# photo\n3  [ H2O -> H + OH ]", r"net\.txt:4: .*H2O -> H \+ OH.*species and branch"),
        ("# photo\n3  [ H2O -> H + OH ]  H2O", r"net\.txt:4: .*H2O -> H \+ OH.*species and branch"),
        ("3  [ H + H + M + M -> H2 + M ]  1.0E-30  0.0  0.0", r"net\.txt:3: .*more than one M"),
    ],
    ids=["photo row, no columns", "photo row, no branch", "M + M"],
)
def test_a_photo_row_without_its_branch_and_a_doubled_M_are_refused(tmp_path, rows, match):
    """VULCAN 2 raises at `int(columns[1])` (`op.py:245`) and builds M*M from `M + M`; the port read neither."""
    from vulcan3 import network as netmod

    f = tmp_path / "net.txt"
    f.write_text(f"1  [ H + OH -> H2O ]  1.0E-10  0.0  0.0\n# reverse stops\n{rows}\n")
    with pytest.raises(ValueError, match=match):
        netmod.parse_network(f)
