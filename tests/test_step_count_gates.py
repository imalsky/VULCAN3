"""Free-convergence step counts: the CLI gates and the 10 % rule against VULCAN 2.

The gates pin vulcan3's own counts, HD189 1495, HD209 1206, W39b 1084: a
physics-neutral change keeps them. HD189_vulcan3 is left out because its count
is platform-dependent. The oracle test runs the pinned VULCAN 2 on the same
three cases and holds vulcan3's count within 10 % of it (oracle.md "Step
counts"); it also needs `$VULCAN_MASTER_DIR`. Each run is a full integration,
so the file is slow-gated; oracle.yml's full tier runs both tests in the release gate.
"""

from __future__ import annotations

import os
import pickle
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from test_default_master_parity import MASTER_ONLY

from tools.audit_master_parity import INTENTIONAL_JAX_DELTAS
from vulcan3 import config
from vulcan3._paths import PACKAGE_ROOT
from vulcan3.atm_setup import surface_gravity
from vulcan3.config import load_config

_GATES = {"HD189": 1495, "HD209": 1206, "W39b": 1084}
# The CLI selects each config's import-frozen network by re-executing itself;
# an inherited selection would run the config on the wrong network.
_FROZEN_ENV = (*config._FROZEN_ENV.values(), "VULCAN3_CLI_RELAUNCHED")
# oracle.md "Step counts": vulcan3's converged count within 10 % of VULCAN 2's.
STEP_RTOL = 0.10
# Inputs copied from the package into the oracle copy, as `oracle.apply_data_deltas` copies the corrected networks:
# the clone's 2025 sulfur network predates the vendored one and its declared row fixes, and lacks W39b's inputs.
_STAGED_KEYS = ("network", "atm_file", "sflux_file")
_MASTER_DONE = re.compile(r"has successfully run to steady-state with (\d+) steps")


def _cli_count(name: str, cwd: Path) -> int:
    """Run the vulcan3 CLI on config `name` to convergence and return its step count."""
    env = {k: v for k, v in os.environ.items() if k not in _FROZEN_ENV}
    res = subprocess.run(
        [sys.executable, "-m", "vulcan3.cli", "--config", name],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,
        check=False,
    )
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    cfg = load_config(name)
    with open(cwd / cfg.output_dir / cfg.out_name, "rb") as fh:
        par = pickle.load(fh)["parameter"]
    assert int(par["end_case"]) == 1, (name, int(par["count"]), par["end_case"])
    return int(par["count"])


def _master_count(name: str) -> int:
    """Run the pinned VULCAN 2 on config `name`'s case to convergence and return its step count.

    The cfg is upstream's HD189 example overlaid with every vulcan3 key except
    the declared departures (`INTENTIONAL_JAX_DELTAS`: VULCAN 2 keeps its own
    `dt_max`, `dt_min` and `nega_cut`), plus `gs` from `Mp`/`Rp`. Both seeds
    read the config's abundance file.
    """
    from oracle import ORACLE_DATA_DELTAS, ORACLE_NETWORK_TRANSPLANTS, oracle_worktree, write_master_cfg

    inherited = sorted(k for k in _FROZEN_ENV if k in os.environ)
    assert not inherited, f"{inherited} are set: the case must come from its config, not from the environment"
    cfg = load_config(name)
    abundance = Path(cfg.fastchem_solar_abundance_file)
    assert abundance.parent == Path("thermo"), abundance
    overrides = {k: v for k, v in vars(cfg).items() if not k.startswith("_") and k not in INTENTIONAL_JAX_DELTAS}
    overrides["gs"] = surface_gravity(cfg)
    overrides.update(MASTER_ONLY)
    with oracle_worktree("vulcan2_ncho", fastchem_abundance=abundance.name) as root:
        for key in _STAGED_KEYS:
            rel = getattr(cfg, key)
            target, source = root / rel, PACKAGE_ROOT / rel
            # Replacing a file the clone ships is declared in tests/oracle.py, never done on the side.
            declared = rel in ORACLE_DATA_DELTAS + ORACLE_NETWORK_TRANSPLANTS
            changes_clone = target.exists() and target.read_bytes() != source.read_bytes()
            assert declared or not changes_clone, f"{rel} changes a clone file and is not declared"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        write_master_cfg(
            root,
            (root / "cfg_examples" / "vulcan_cfg_HD189.py").read_text()
            + f"\n# === {name} overrides ===\n"
            + "\n".join(f"{k} = {v!r}" for k, v in sorted(overrides.items()))
            + "\n",
        )
        res = subprocess.run(
            [sys.executable, "vulcan.py", "-n"],
            cwd=root,
            env={**os.environ, "PYTHONHASHSEED": "0"},
            capture_output=True,
            text=True,
            timeout=1800,
            check=False,
        )
    log = f"{res.stdout[-3000:]}\n{res.stderr[-3000:]}"
    assert res.returncode == 0, f"VULCAN 2 {name} exited {res.returncode}\n{log}"
    done = _MASTER_DONE.search(res.stdout)
    assert done, f"VULCAN 2 {name} did not converge\n{log}"
    return int(done.group(1))


@pytest.mark.slow
@pytest.mark.parametrize(("name", "count"), list(_GATES.items()))
def test_cli_step_count_gate(name, count, tmp_path):
    assert _cli_count(name, tmp_path) == count, name


@pytest.mark.slow
@pytest.mark.parametrize("name", list(_GATES))
def test_step_count_within_10_percent_of_vulcan2(name, tmp_path):
    master = _master_count(name)
    print(f"{name}: VULCAN 2 {master} steps", flush=True)
    port = _cli_count(name, tmp_path)
    print(f"{name}: vulcan3 {port} steps, ratio {port / master:.4f}")
    assert abs(port - master) <= STEP_RTOL * master, (name, port, master)
