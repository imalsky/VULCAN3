"""Run the two gradient examples in `examples/` to completion.

They build `AtmStatic` by hand, so a new field breaks them; no other test or workflow runs them.
"""

from __future__ import annotations

import math
import re
import subprocess
import sys
from pathlib import Path

import pytest
from _helpers import SELF_RUN_TIMEOUT_S

ROOT = Path(__file__).resolve().parent.parent
HD189_FIXTURE = ROOT / "tests" / "data" / "adj_state_hd189.npz"

# Each script and the printed line carrying one derivative it computed.
EXAMPLES = [
    pytest.param("grad_jvp_example.py", r"tangent dy_out\[0, 0\]\s+=\s+(\S+)", id="jvp"),
    pytest.param(
        "grad_reverse_example.py",
        r"r\s+13\s+FD=\S+\s+adjoint=(\S+)",
        id="reverse",
        marks=pytest.mark.skipif(not HD189_FIXTURE.exists(), reason="HD189 adjoint fixture missing"),
    ),
]


@pytest.mark.slow
@pytest.mark.parametrize(("script", "derivative_line"), EXAMPLES)
def test_gradient_example_runs(script: str, derivative_line: str) -> None:
    """The script exits 0 and prints a finite derivative."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "examples" / script)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=SELF_RUN_TIMEOUT_S,
        check=False,
    )
    output = f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    assert result.returncode == 0, f"{script} exited {result.returncode}\n{output}"
    match = re.search(derivative_line, result.stdout)
    assert match is not None, f"{script} printed no line matching {derivative_line!r}\n{output}"
    assert math.isfinite(float(match.group(1))), f"{script} derivative is {match.group(1)}\n{output}"
