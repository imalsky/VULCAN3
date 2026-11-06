"""Smoke-check the shipped example configs (`vulcan3/configs/*.yaml`).

Each subprocess loads a named config with `load_config`, validates it with
`runtime_validation`, and runs the full pre-loop setup through photo / ion-rate
initialization. This catches missing vendored assets and stale config paths
without mutating the process default config. A non-default network/atom set
(e.g. W39b's SNCHO) is import-frozen, so the child sets `$VULCAN3_*` from the
config's own frozen keys BEFORE importing vulcan3.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


_SETUP_SCRIPT = r"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import yaml

warnings.filterwarnings("ignore")

pkg_root = Path(sys.argv[1])   # .../src/vulcan3
cfg_name = sys.argv[2]
os.chdir(pkg_root)

# Read the import-frozen knobs from the config YAML BEFORE importing vulcan3
# so a non-default network/atom_list/com_file freezes correctly.
raw = yaml.safe_load((pkg_root / "configs" / (cfg_name + ".yaml")).read_text())
if raw.get("network"):
    os.environ["VULCAN3_NETWORK"] = raw["network"]
if raw.get("atom_list"):
    os.environ["VULCAN3_ATOM_LIST"] = ",".join(raw["atom_list"])
if raw.get("com_file"):
    os.environ["VULCAN3_COM_FILE"] = raw["com_file"]

import vulcan3  # noqa: F401  (freezes network/atom_list/com_file here)
from vulcan3.config import load_config

cfg = load_config(cfg_name)
cfg.use_print_prog = False
cfg.save_evolution = False
cfg.count_max = 1
cfg.count_min = 1
cfg.runtime = 1.0e22

from vulcan3.runtime_validation import validate_runtime_config

validate_runtime_config(cfg)

# Full pre-loop pipeline (atm + rates + initial abundance + photo
# cross-sections + sflux + remove pass) in one call.
from vulcan3.state import RunState

rs = RunState.with_pre_loop_setup(cfg)
assert rs.metadata is not None
assert rs.atm.Tco.shape[0] > 0

print("PASS")
"""


@pytest.mark.parametrize("cfg_name", ["HD209", "W39b"])
def test_cfg_example_setup(cfg_name: str) -> None:
    from vulcan3._paths import PACKAGE_ROOT

    result = subprocess.run(
        [sys.executable, "-c", _SETUP_SCRIPT, str(PACKAGE_ROOT), cfg_name],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    assert result.returncode == 0, (
        f"{cfg_name} setup failed with exit code {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    assert result.stdout.strip().endswith("PASS")
