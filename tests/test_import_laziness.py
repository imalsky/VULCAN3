"""`import vulcan3` leaves the solver choice open and freezes only the network.

`jax_step` and `solver_fast` read `$VULCAN3_SOLVER` and the reservoir tables when first imported, so the package import
must not reach them: a caller may set the variable or `cfg.atom_list` between `import vulcan3` and the first setup.
"""

from __future__ import annotations

import os
import subprocess
import sys

_CHILD = r"""
import os
import sys

import vulcan3

for name in ("jax_step", "solver_fast", "outer_loop", "op_jax", "atm_setup", "legacy_io", "steady_state_grad"):
    assert f"vulcan3.{name}" not in sys.modules, f"import vulcan3 loaded vulcan3.{name}"

os.environ["VULCAN3_SOLVER"] = "ffi"
from vulcan3 import solver_fast

assert solver_fast.BACKEND == "ffi", solver_fast.BACKEND
"""


def test_import_leaves_solver_and_step_modules_unloaded():
    env = {k: v for k, v in os.environ.items() if k != "VULCAN3_SOLVER"}
    res = subprocess.run(
        [sys.executable, "-c", _CHILD], capture_output=True, text=True, env=env, timeout=300, check=False
    )
    assert res.returncode == 0, res.stderr[-2000:]
