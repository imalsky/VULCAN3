"""Pinned, isolated access to an upstream VULCAN checkout for oracle tests."""

# Four rules:
#
# 1. `VULCAN_MASTER_DIR` names the checkout; there is no default path.
# 2. `require_oracle()` checks HEAD against tests/science_sources.yaml and a
#    clean tree before any calculation (reaction indices are positional).
# 3. `oracle_worktree()` hands out a temporary copy, since upstream setup
#    rewrites files in place, and proves the original unchanged afterwards.
# 4. An upstream defect corrected on both sides is listed once in
#    `ORACLE_CODE_DELTAS` and applied to the copy only; each delta must match
#    the pinned text once.
#
# Local runs skip without an oracle; release CI sets
# `VULCAN3_REQUIRE_ORACLE=1` so a missing oracle fails.

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = Path(__file__).resolve().parent / "science_sources.yaml"

ENV_DIR = "VULCAN_MASTER_DIR"
ENV_REQUIRE = "VULCAN3_REQUIRE_ORACLE"

_MANIFEST_CACHE: dict | None = None


def manifest() -> dict:
    global _MANIFEST_CACHE
    if _MANIFEST_CACHE is None:
        _MANIFEST_CACHE = yaml.safe_load(MANIFEST_PATH.read_text())
    return _MANIFEST_CACHE


def oracle_spec(family: str) -> dict:
    """Return the manifest entry for one oracle-test family."""
    oracles = manifest()["oracles"]
    if family not in oracles:
        raise KeyError(f"unknown oracle family {family!r}; science_sources.yaml defines {sorted(oracles)}")
    return oracles[family]


def _git(repo: Path, *args: str) -> tuple[int, str]:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=30, check=False)
    return r.returncode, r.stdout.strip()


def oracle_dir() -> Path | None:
    """Return the configured oracle checkout, or None when unset."""
    raw = os.environ.get(ENV_DIR)
    return Path(raw).expanduser().resolve() if raw else None


def _fail_or_skip(msg: str):
    """Fail when an oracle is REQUIRED, else skip with the same message."""
    if os.environ.get(ENV_REQUIRE) == "1":
        pytest.fail(msg)
    pytest.skip(msg)


def run_oracle_subprocess(test_file, family: str, config_rel: str | None = None) -> None:
    """Run `test_file`'s `main()` in a fresh process against an oracle COPY.

    Upstream's modules are top-level names bound to one copy's cfg, so every
    upstream-comparison test re-execs itself from a cold Python start.
    """
    with oracle_worktree(family, config_rel) as master:
        regenerate_chem_funs(master)
        env = os.environ.copy()
        env["VULCAN_MASTER_DIR"] = str(master)
        result = subprocess.run(
            [sys.executable, str(Path(test_file).resolve())],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
    assert result.returncode == 0, (
        f"subprocess exited {result.returncode}\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )


# --- inside the re-exec'd child ------------------------------------------------
# The parent verified the pin and set VULCAN_MASTER_DIR to a copy with no .git.
# Upstream imports these modules by top-level name; none of them exists in
# tests/ or among the port's imports (src/vulcan3 is a package).
UPSTREAM_MODULES = ("vulcan_cfg", "store", "build_atm", "op", "chem_funs")


def enter_master() -> Path:
    """Chdir into the oracle copy (upstream reads its inputs by relative path) and put it first on sys.path."""
    master = oracle_dir()
    if master is None or not master.is_dir():
        raise SystemExit(f"${ENV_DIR} must name the oracle copy; run this file through pytest, which passes one")
    os.chdir(master)
    sys.path.insert(0, str(master))
    return master


def leave_master() -> None:
    """Drop upstream's modules and path entry and chdir back to ROOT, so no later import resolves to upstream.

    References already held stay valid.
    """
    master = str(oracle_dir())
    for name in UPSTREAM_MODULES:
        sys.modules.pop(name, None)
    while master in sys.path:
        sys.path.remove(master)
    os.chdir(ROOT)


def upstream_preloop(cfg, var, atm, make_atm, rate, ini=None, output=None):
    """Upstream vulcan.py's pre-loop, in its order, on the caller's objects.

    The caller picks each object's side (upstream or port); only the call
    sequence is shared. Stops after read_rate without `ini`, after ini_y
    without `output`, else runs through BC_flux. Returns `(var, atm)`.
    """
    atm = make_atm.f_pico(atm)
    atm = make_atm.load_TPK(atm)
    if cfg.use_condense:
        make_atm.sp_sat(atm)
    var = rate.read_rate(var, atm)
    if ini is None:
        return var, atm
    var = rate.rev_rate(var, atm)
    var = ini.ini_y(var, atm)
    if output is None:
        return var, atm
    var = ini.ele_sum(var)
    atm = make_atm.f_mu_dz(var, atm, output)
    make_atm.mol_diff(atm)
    make_atm.BC_flux(atm)
    return var, atm


def assert_below(scores: dict[str, tuple[float, float]]) -> None:
    """Print every `name: (error, bound)`, then fail on each error not below its bound (NaN included)."""
    for name, (err, bound) in scores.items():
        print(f"{name}: {err:.3e} (bound {bound:.0e})")
    over = [name for name, (err, bound) in scores.items() if not err < bound]
    assert not over, f"over bound: {over}"


def write_master_cfg(root: Path, cfg_text: str) -> None:
    """Write `vulcan_cfg.py` into the oracle COPY at `root`, and regenerate its chem_funs.py with upstream's generator.

    The old module is deleted first, so a failed generator cannot pass by
    leaving the shipped one importable. make_chem_funs.py writes chem_funs.py
    before its check_conserv(), which raises under numpy>=1.24; master's
    vulcan.py ignores that exit code, so the module's importability decides.
    PYTHONHASHSEED=0 as for every master run (its trajectory depends on set
    order).
    """
    (root / "vulcan_cfg.py").write_text(cfg_text)
    regenerate_chem_funs(root)


def regenerate_chem_funs(root: Path) -> None:
    """Rebuild the oracle COPY's chem_funs.py from its current cfg and network.

    The clone's tracked module numbers the pinned network's rows, so it is stale once
    `ORACLE_DATA_DELTAS` has replaced the network: rows past a removed one shift by two.
    """
    (root / "chem_funs.py").unlink(missing_ok=True)
    env = {**os.environ, "PYTHONHASHSEED": "0"}
    gen = subprocess.run(
        [sys.executable, "make_chem_funs.py"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    probe = subprocess.run(
        [sys.executable, "-c", "import chem_funs as c; assert c.ni > 0 and c.nr > 0"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert probe.returncode == 0, (
        f"make_chem_funs.py exited {gen.returncode} and left no importable "
        f"chem_funs:\n{gen.stdout[-2000:]}\n{gen.stderr[-2000:]}\n"
        f"{probe.stderr[-2000:]}"
    )


def require_oracle(family: str) -> Path:
    """Return the oracle path for `family`, or skip/fail with one clear reason.

    Verifies the checkout exists, is a git repository, sits at the manifest's
    exact commit, and is clean. Never returns a path that fails any of those.
    """
    spec = oracle_spec(family)
    want = spec["commit"]
    path = oracle_dir()

    if path is None:
        _fail_or_skip(
            f"oracle family {family!r} needs an upstream checkout. Set "
            f"{ENV_DIR} to a clean clone of {spec['repo']} at commit "
            f"{want[:12]}:\n"
            f"    git clone {spec['repo']} /tmp/vulcan-oracle\n"
            f"    git -C /tmp/vulcan-oracle checkout {want}\n"
            f"    export {ENV_DIR}=/tmp/vulcan-oracle\n"
            "There is no default path."
        )

    if not path.is_dir():
        _fail_or_skip(f"{ENV_DIR}={path} does not exist")

    rc, head = _git(path, "rev-parse", "HEAD")
    if rc != 0 or not head:
        _fail_or_skip(
            f"{ENV_DIR}={path} is not a git checkout, so its revision cannot "
            "be verified; an unversioned copy cannot serve as an oracle. Clone "
            f"{spec['repo']} and check out {want[:12]}."
        )

    if head != want:
        _fail_or_skip(
            f"UNSUPPORTED ORACLE REVISION for family {family!r}.\n"
            f"  expected: {want}\n"
            f"  actual:   {head}\n"
            f"  repo:     {spec['repo']}\n"
            f"  why this revision: {spec.get('why', '').strip()}\n"
            "Reaction indices are POSITIONAL, so a different revision shifts "
            "every rate/Gibbs comparison and produces failures unrelated to "
            "the ported kernels. Check out the pinned commit, or update "
            "tests/science_sources.yaml and re-run the oracle tests."
        )

    rc, dirty = _git(path, "status", "--porcelain")
    if rc != 0:
        _fail_or_skip(f"cannot read git status of {path}")
    if dirty:
        _fail_or_skip(
            f"oracle checkout {path} is DIRTY:\n"
            + "\n".join(f"    {ln}" for ln in dirty.splitlines()[:20])
            + "\nAn oracle must be pristine: upstream setup rewrites files in "
            "place (make_chem_funs renumbers a network file; FastChem writes "
            "into fastchem_vulcan/), so a dirty tree is usually residue of a "
            "run outside oracle_worktree(). Reset it: git -C "
            f"{path} checkout . && git -C {path} clean -fd"
        )

    return path


def tree_fingerprint(path: Path) -> str:
    """SHA-256 over the checkout's tracked-file state (git-based, cheap)."""
    rc, head = _git(path, "rev-parse", "HEAD")
    rc2, dirty = _git(path, "status", "--porcelain")
    if rc != 0 or rc2 != 0:
        raise RuntimeError(f"cannot fingerprint {path}: not a git checkout")
    return hashlib.sha256(f"{head}\n{dirty}".encode()).hexdigest()


# Rule 4. {relative file: ((OLD, NEW, tag), ...)}; OLD/NEW are exact upstream
# text (24-space indent, LF).
ORACLE_CODE_DELTAS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "op.py": (
        (
            (
                "                        nn = 0.75 - 1.27*np.log(Fc)\n"
                "                        ff = np.exp( np.log(Fc)/(1.+ (np.log(k[i]*M/k_inf)/nn)**2 ) )\n"
            ),
            (
                "                        nn = 0.75 - 1.27*np.log10(Fc)\n"
                "                        ff = Fc**( 1./(1.+ (np.log10(k[i]*M/k_inf)/nn)**2 ) )\n"
            ),
            "OH+CH3+M Troe width in log10 (Visscher & Moses 2011 eq 14)",
        ),
        (
            "        return var\n        \n    \n    def remove_rate(self, var):",
            (
                "        for i in rev_list:\n"
                "            r = [np.zeros(nz)]\n"
                "            for j, side in zip((i-1, i), var.Rf[i-1].split('->')):\n"
                "                names = [q.strip() for q in side.split('+')]\n"
                "                m = sum(q != 'M' for q in names)\n"
                "                if m + ('M' in names) >= 2 and j not in vulcan_cfg.remove_list:\n"
                "                    r.append(var.k[j] * atm.M**(('M' in names) + m - 2.) / 1.e-9)\n"
                "            c = 1./np.maximum(np.max(r, axis=0), 1.)\n"
                "            var.k[i-1], var.k[i] = var.k[i-1]*c, var.k[i]*c\n"
                "        return var\n        \n    \n    def remove_rate(self, var):"
            ),
            "collision limit on thermal pairs, as rates_jax.apply_collision_limit (notes Parity ledger)",
        ),
    ),
    # FastChem's table holds the same N(2D) fit as thermo/NASA9/N_2D.txt (`n2d_thermo`), so the seed needs it too.
    "fastchem_vulcan/input/nasa9_logK_SNCHOPTi.dat": (
        (
            (
                "-9.03708588e+03  2.48128703e+00 4.48951881e+00 -2.46852066e-03 5.07361986e-06 -4.80560489e-09  "
                "1.73150465e-12 0.             8.32331232e+04  1.76541859e+00 -2.29928631e+01 -4.63139448e-01 "
                "4.00395298e+00 -1.48851690e-05 2.76628541e-08 -2.41078078e-11  7.93668376e-15 0.             "
                "8.33519446e+04  4.02697144e+00"
            ),
            (
                "0. 0. 2.500000000E+00 0. 0. 0. 0. 0. 8.376936653E+04 5.110195768E+00 "
                "0. 0. 2.500000000E+00 0. 0. 0. 0. 0. 8.376936653E+04 5.110195768E+00"
            ),
            "N(2D) thermo in FastChem's table, as thermo/NASA9/N_2D.txt (n2d_thermo)",
        ),
    ),
}


# Rule 4b. Vendored data files VULCAN-JAX corrects (science_sources.yaml
# `intentional_divergences` low_t_reverse_rates, n2d_thermo, hno3_thermo, sulfur_compose_masses):
# copied from this tree into the COPY so index-by-index rate and step comparisons
# run the same network on both sides. tools/audit_master_parity.py checks that
# these files differ from upstream only in the declared rows.
ORACLE_DATA_DELTAS: tuple[str, ...] = (
    "thermo/NCHO_photo_network.txt",
    "thermo/SNCHO_photo_network.txt",
    "thermo/all_compose.txt",
    "thermo/NASA9/N_2D.txt",
    "thermo/NASA9/HNO3.txt",
)


def apply_data_deltas(root: Path) -> list[str]:
    """Copy `ORACLE_DATA_DELTAS` from this tree into the COPY at `root`."""
    pkg = Path(__file__).resolve().parent.parent / "src" / "vulcan3"
    for rel in ORACLE_DATA_DELTAS:
        if not (root / rel).is_file():
            raise RuntimeError(f"{root / rel}: the oracle copy has no such file to replace")
        shutil.copy2(pkg / rel, root / rel)
    return list(ORACLE_DATA_DELTAS)


def apply_code_deltas(root: Path) -> list[str]:
    """Apply `ORACLE_CODE_DELTAS` to the COPY at `root`; return the applied tags."""
    applied: list[str] = []
    for rel, deltas in ORACLE_CODE_DELTAS.items():
        path = root / rel
        text = path.read_bytes().decode("utf-8")
        for old, new, tag in deltas:
            n = text.count(old)
            if n != 1:
                raise RuntimeError(
                    f"{path}: delta {tag!r} expects its OLD block exactly once, "
                    f"found {n}; the pinned upstream no longer matches "
                    "ORACLE_CODE_DELTAS -- re-derive or delete the entry"
                )
            text = text.replace(old, new)
            applied.append(tag)
        path.write_bytes(text.encode("utf-8"))
    return applied


# One FastChem build per oracle family per session: `make` takes ~1 min and
# every EQ comparison needs the same binary. The build happens in a throwaway
# copy (make writes objects in place), never in the pinned checkout; the
# TemporaryDirectory handles stay alive so the binaries outlive each worktree.
_FASTCHEM_BUILDS: dict[str, Path] = {}
_FASTCHEM_BUILD_DIRS: list[tempfile.TemporaryDirectory] = []


def _oracle_fastchem_binary(family: str, src: Path) -> Path:
    """Build the oracle's own FastChem and return the binary."""
    cached = _FASTCHEM_BUILDS.get(family)
    if cached is not None and cached.is_file():
        return cached
    holder = tempfile.TemporaryDirectory(prefix="vulcan-oracle-fastchem-")
    _FASTCHEM_BUILD_DIRS.append(holder)
    tree = Path(holder.name) / "fastchem_vulcan"
    shutil.copytree(src / "fastchem_vulcan", tree, symlinks=True, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    (tree / "obj").mkdir(exist_ok=True)
    build = subprocess.run(["make"], cwd=str(tree), capture_output=True, text=True, timeout=1800, check=False)
    binary = tree / "fastchem"
    if build.returncode != 0 or not binary.is_file():
        raise RuntimeError(
            f"building the oracle's FastChem in {tree} failed "
            f"({build.returncode}):\n{build.stdout[-2000:]}\n{build.stderr[-2000:]}"
        )
    _FASTCHEM_BUILDS[family] = binary
    return binary


@contextmanager
def oracle_worktree(
    family: str,
    config_rel: str | None = None,
    fastchem_abundance: str | None = None,
):
    """Yield a temporary COPY of the oracle; the original must not be touched.

    Upstream data-generation code mutates its own checkout, so every test that
    runs it must work on a copy. The copy is deleted on exit and the original is
    proven unchanged. The copy carries `ORACLE_CODE_DELTAS` (rule 4) and
    `ORACLE_DATA_DELTAS` (rule 4b).
    """
    src = require_oracle(family)
    before = tree_fingerprint(src)
    with tempfile.TemporaryDirectory(prefix="vulcan-oracle-") as tmp:
        dst = Path(tmp) / src.name
        # copy2 preserves mtimes; skip .git (large, and the copy is disposable)
        shutil.copytree(src, dst, symlinks=True, ignore=shutil.ignore_patterns(".git", "__pycache__"))
        apply_code_deltas(dst)
        apply_data_deltas(dst)
        if config_rel is not None:
            config_source = dst / config_rel
            if not config_source.is_file():
                raise FileNotFoundError(f"oracle config does not exist: {config_source}")
            shutil.copy2(config_source, dst / "vulcan_cfg.py")
        oracle_fastchem = dst / "fastchem_vulcan"
        # Upstream ships FastChem source only, so an ini_mix='EQ' oracle run needs a
        # build of upstream's own source (the oracle stays unmodified).
        if oracle_fastchem.is_dir() and not (oracle_fastchem / "fastchem").exists():
            shutil.copy2(_oracle_fastchem_binary(family, src), oracle_fastchem / "fastchem")
        if fastchem_abundance is not None:
            abundance = ROOT / "src" / "vulcan3" / "thermo" / fastchem_abundance
            if not abundance.is_file():
                raise FileNotFoundError(f"no such abundance preset: {abundance}")
            shutil.copy2(abundance, oracle_fastchem / "input" / "solar_element_abundances.dat")
        try:
            yield dst
        finally:
            after = tree_fingerprint(src)
            if after != before:
                _rc, dirty = _git(src, "status", "--porcelain")
                raise AssertionError(
                    f"the oracle checkout at {src} CHANGED during the test. "
                    "Upstream code must only ever run against the temporary "
                    "copy.\nDirty files now:\n" + "\n".join(f"    {ln}" for ln in dirty.splitlines()[:20])
                )
