"""Obtain every numerical-regression fixture the test suite requires.

    python tests/gen_fixtures.py --all

That one command, run in a clean clone with no sibling repositories, produces
all five fixtures and writes this machine's manifest (`MANIFEST`) recording, per file, its
SHA-256 and the commit, config and dependency versions that made it.

The fixtures are required (conftest fails collection when one is missing)
but gitignored (~36 MB), so a clean clone must regenerate them. The network
is frozen at the first `import vulcan3` and the two adjoint states need
different networks, so each generator runs in its own subprocess. The
adjoint states and the HD209 state each converge a real run (minutes); the
photo fixtures are fast.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = ROOT / "tests" / "data"
# Converged states are machine-dependent, so each machine has its own manifest. The tracked FIXTURES.json is the
# arm64 laptop's; any other machine's is local and gitignored.
_MACHINE = platform.machine()
MANIFEST = FIXTURE_DIR / ("FIXTURES.json" if _MACHINE == "arm64" else f"FIXTURES.{_MACHINE}.json")

# fixture file -> (generator argv relative to ROOT, which fixtures it writes).
# Grouped by generator so one subprocess can emit several files.
GENERATORS = {
    "photo": {
        "argv": [sys.executable, "tests/_gen_photo_baseline.py"],
        "writes": ["photo_setup_hd189_baseline.npz", "photo_setup_hd189_T_dep.npz"],
        "note": "HD189 photo cross-section setup (default and T-dependent)",
    },
    "adj_hd189": {
        "argv": [sys.executable, "tests/_gen_adj_state.py", "hd189"],
        "writes": ["adj_state_hd189.npz"],
        "note": "HD189 converged state, photo OFF, C-H-N-O network",
    },
    "adj_w39b": {
        "argv": [sys.executable, "tests/_gen_adj_state.py", "w39b"],
        "writes": ["adj_state_w39b.npz"],
        "note": "WASP-39 b converged state, photo ON (frozen), SNCHO network",
    },
    "hd209": {
        "argv": [sys.executable, "tests/_gen_adj_state.py", "hd209"],
        "writes": ["state_hd209.npz"],
        "note": "HD209 converged state (shipped config, photo ON), C-H-N-O network",
    },
}

ALL_FIXTURES = [f for g in GENERATORS.values() for f in g["writes"]]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head() -> str:
    head = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    ).stdout.strip()
    return head + ("-dirty" if dirty else "")


def _versions() -> dict:
    out = {"python": sys.version.split()[0]}
    for mod in ("jax", "jaxlib", "numpy", "scipy"):
        out[mod] = importlib.import_module(mod).__version__
    # The source of truth: under an editable install importlib.metadata reports the version at `pip install -e`.
    out["vulcan3"] = re.search(
        r"__version__\s*=\s*[\"']([^\"']+)", (ROOT / "src/vulcan3/_version.py").read_text()
    ).group(1)
    return out


def _config_identity() -> dict:
    """Hash the shipped configs the generators read.

    A fixture built from a different config is a different oracle, and the
    failure mode is a confusing numerical mismatch rather than a clear error.
    """
    cfg_dir = ROOT / "src" / "vulcan3" / "configs"
    out = {}
    for name in ("default.yaml", "W39b.yaml", "HD189.yaml", "HD209.yaml"):
        p = cfg_dir / name
        if p.is_file():
            out[name] = sha256(p)[:16]
    return out


def run_generator(key: str) -> None:
    spec = GENERATORS[key]
    print(f"\n=== {key}: {spec['note']} ===", flush=True)
    env = dict(os.environ)
    # PYTHONSAFEPATH strips the cwd from sys.path in some sandboxes, which makes
    # the generators fail for reasons unrelated to the repo.
    env.pop("PYTHONSAFEPATH", None)
    t0 = time.time()
    r = subprocess.run(spec["argv"], cwd=ROOT, env=env, check=False)
    if r.returncode != 0:
        raise SystemExit(
            f"gen_fixtures: generator {key} failed (exit {r.returncode}). "
            f"Command: {' '.join(spec['argv'])} (cwd={ROOT})"
        )
    print(f"=== {key}: done in {time.time() - t0:.1f}s ===", flush=True)


def record_generated(keys: list[str]) -> dict:
    """Record this checkout's provenance for the fixtures of generators `keys`; every other record stays as it was."""
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.is_file() else {"fixtures": {}}
    provenance = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "vulcan3_commit": _git_head(),
        "platform_machine": platform.machine(),
        "versions": _versions(),
        "config_sha256_16": _config_identity(),
    }
    for key in keys:
        spec = GENERATORS[key]
        for name in spec["writes"]:
            path = FIXTURE_DIR / name
            manifest["fixtures"][name] = {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "generator": " ".join(spec["argv"][1:]),
                "note": spec["note"],
                **provenance,
            }
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {MANIFEST}")
    return manifest


def built_on(name: str) -> str | None:
    """Return the machine that built fixture `name`, or None when this machine's manifest does not say."""
    if not MANIFEST.is_file():
        return None
    return json.loads(MANIFEST.read_text())["fixtures"].get(name, {}).get("platform_machine")


def verify() -> int:
    """Check every fixture against its record in the manifest. Returns an exit code."""
    if not MANIFEST.is_file():
        print(f"gen_fixtures: no manifest at {MANIFEST}; run --all --force first", file=sys.stderr)
        return 1
    recorded = json.loads(MANIFEST.read_text())["fixtures"]
    bad = []
    for name in ALL_FIXTURES:
        p = FIXTURE_DIR / name
        if not p.is_file():
            bad.append(f"{name}: MISSING")
            continue
        if name not in recorded:
            bad.append(f"{name}: present but not in the manifest; rebuild it with --force")
            continue
        if "vulcan3_commit" not in recorded[name]:
            bad.append(f"{name}: an old-format record with no provenance; rebuild it with --force")
            continue
        got = sha256(p)
        if got != recorded[name]["sha256"]:
            bad.append(
                f"{name}: sha256 {got[:16]}... != recorded {recorded[name]['sha256'][:16]}...; rebuild it with --force"
            )
    if bad:
        print("gen_fixtures: verification FAILED", file=sys.stderr)
        for b in bad:
            print(f"  - {b}", file=sys.stderr)
        return 1
    for name in ALL_FIXTURES:
        print(f"{name}: matches the record, built from {recorded[name]['vulcan3_commit']}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--all", action="store_true", help="build every missing fixture and record it, then verify all of them"
    )
    g.add_argument("--only", choices=sorted(GENERATORS), help="run one generator")
    g.add_argument("--verify", action="store_true", help="check the existing fixtures against the manifest")
    ap.add_argument("--force", action="store_true", help="rebuild even when the fixture is already present")
    args = ap.parse_args(argv)

    if args.verify:
        return verify()

    keys = sorted(GENERATORS) if args.all else [args.only]
    generated = []
    for key in keys:
        have = all((FIXTURE_DIR / n).is_file() for n in GENERATORS[key]["writes"])
        if have and not args.force:
            print(f"=== {key}: already present, skipping (--force to rebuild) ===", flush=True)
            continue
        run_generator(key)
        generated.append(key)

    if generated:
        record_generated(generated)
    # A skipped fixture keeps its record, and must still match it.
    return verify() if args.all else 0


if __name__ == "__main__":
    sys.exit(main())
