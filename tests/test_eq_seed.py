"""The `ini_mix: EQ` Gibbs seed: agreement with upstream, and its own invariants."""

# The seed is a JAX Gibbs minimizer over the loaded network's gas species, on
# the NASA-9 polynomials of the reverse rates. Two things need proving:
#
# 1. It is the same equilibrium. `test_seed_matches_upstream_fastchem` runs the
#    pinned upstream VULCAN's own initializer on the same column with the same
#    elemental abundances and compares dex by dex. Upstream's FastChem reads the
#    same NASA-9 polynomials (its HO2 row is older) but shares no algorithm and
#    no species list, so the bar is the measured agreement (see MAX_DEX below),
#    not machine precision.
#
# 2. It is exact where it must be: element ratios, normalisation, excluded
#    species, and the jit / vmap / jvp boundary. Those need no oracle.
#
# Both sides run in subprocesses: the network is import-frozen, so the W39b
# (SNCHO) case cannot share a process with the NCHO default, and master's
# modules must never be imported into a pytest worker.

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml

from vulcan3._paths import PACKAGE_ROOT

# 3x the worst max |log10(y_jax/y_master)| over cells with master ymix >
# YMIX_FLOOR (HD189 7.86e-3, W39b 7.72e-3, both HO2). The full-solar Lodders
# 2009 preset is not a case: upstream's FastChem locks O into rock species
# an NCHO network lacks.
MAX_DEX = 2.4e-2
YMIX_FLOOR = 1.0e-15
ATM_RTOL = 1e-14  # both codes build Tco and M from the same inputs
ELEMENT_RTOL = 3.2e-8  # 3x worst recovery (1.04e-8, isothermal 300 K) at fastchem_newton_tol 1e-12

# (case id, VULCAN-JAX config, abundance preset)
CASES = [
    ("hd189", "default", "solar_element_abundances.dat"),
    ("w39b", "W39b", "solar_element_abundances.dat"),
]


_JAX_PROBE = r"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import yaml

root = Path(sys.argv[1])       # PACKAGE_ROOT (.../src/vulcan3)
out_npz = Path(sys.argv[2])
config_name = sys.argv[3]
abundance = sys.argv[4]

os.chdir(root)

# The network is import-frozen: set $VULCAN3_* from the config's own keys
# (over default.yaml, as load_config reads it) BEFORE importing vulcan3 so a
# non-default network is parsed.
raw = {**yaml.safe_load((root / "configs" / "default.yaml").read_text()),
       **yaml.safe_load((root / "configs" / f"{config_name}.yaml").read_text())}
os.environ["VULCAN3_NETWORK"] = raw["network"]
os.environ["VULCAN3_ATOM_LIST"] = ",".join(raw["atom_list"])
os.environ["VULCAN3_COM_FILE"] = raw["com_file"]

from vulcan3.config import default_config, load_config

cfg = default_config()
for key, value in vars(load_config(config_name)).items():
    setattr(cfg, key, value)
cfg.fastchem_solar_abundance_file = f"thermo/{abundance}"

from vulcan3.runtime_validation import validate_runtime_config

validate_runtime_config(cfg)

import vulcan3.chem_funs as chem_funs
from vulcan3.atm_setup import Atm
from vulcan3.ini_abun import InitialAbun
from vulcan3.state import _AtmData, _Variables

data_var, data_atm = _Variables(), _AtmData()
make_atm = Atm()
data_atm = make_atm.f_pico(data_atm)
data_atm = make_atm.load_TPK(data_atm)
data_var = InitialAbun().ini_y(data_var, data_atm)

np.savez_compressed(
    out_npz,
    species=np.array(list(chem_funs.spec_list), dtype=object),
    pco=np.asarray(data_atm.pco, dtype=np.float64),
    Tco=np.asarray(data_atm.Tco, dtype=np.float64),
    M=np.asarray(data_atm.M, dtype=np.float64),
    y_ini=np.asarray(data_var.y_ini, dtype=np.float64),
)
"""


_MASTER_PROBE = r"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])       # the DISPOSABLE oracle copy
out_npz = Path(sys.argv[2])
cfg_src = Path(sys.argv[3])

(root / "vulcan_cfg.py").write_text(cfg_src.read_text())
# A failed generator must not be mistaken for success by importing the
# checkout's stale, pre-generated module.
(root / "chem_funs.py").unlink(missing_ok=True)
res = subprocess.run(
    [sys.executable, "make_chem_funs.py"],
    cwd=str(root), capture_output=True, text=True, timeout=900,
)
# make_chem_funs.py writes chem_funs.py before its check_conserv(), which
# raises under numpy>=1.24; master's vulcan.py ignores that exit code, so
# only the module's importability matters.
if res.returncode != 0:
    probe = subprocess.run(
        [sys.executable, "-c",
         "import chem_funs as c; assert c.ni > 0 and c.nr > 0"],
        cwd=str(root), capture_output=True, text=True, timeout=300,
    )
    if probe.returncode != 0:
        print(res.stdout[-2000:])
        print(res.stderr[-2000:])
        sys.exit(res.returncode)

os.chdir(root)
sys.path.insert(0, str(root))

import build_atm
import chem_funs
import store

data_var, data_atm = store.Variables(), store.AtmData()
make_atm = build_atm.Atm()
data_atm = make_atm.f_pico(data_atm)
data_atm = make_atm.load_TPK(data_atm)
data_var = build_atm.InitialAbun().ini_y(data_var, data_atm)

np.savez_compressed(
    out_npz,
    species=np.array(list(chem_funs.spec_list), dtype=object),
    pco=np.asarray(data_atm.pco, dtype=np.float64),
    Tco=np.asarray(data_atm.Tco, dtype=np.float64),
    M=np.asarray(data_atm.M, dtype=np.float64),
    y_ini=np.asarray(data_var.y_ini, dtype=np.float64),
)
print("MASTER_OK")
"""


_TRACER_PROBE = r"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

root = Path(sys.argv[1])       # PACKAGE_ROOT (.../src/vulcan3)
os.chdir(root)

from vulcan3 import ini_abun
from vulcan3.atm_setup import Atm
from vulcan3.config import default_config
from vulcan3.state import _AtmData

cfg = default_config()
make_atm = Atm()
data_atm = make_atm.load_TPK(make_atm.f_pico(_AtmData()))
Tco = jnp.asarray(np.asarray(data_atm.Tco, dtype=np.float64))
p_bar = jnp.asarray(np.asarray(data_atm.pco, dtype=np.float64) / 1.0e6)
b = jnp.asarray(ini_abun._element_vector(cfg))
seed = jax.jit(lambda T, p, b: ini_abun.eq_seed(T, p, b, cfg))

# `_element_vector` needs the element order, so drop the setup again: the
# first seed CALL has to be the traced one, so the lazy setup is built inside
# that trace.
ini_abun._SEED.clear()
y = np.asarray(seed(Tco, p_bar, b))
assert np.isfinite(y).all()
fm = ini_abun._seed(bool(cfg.use_photo))[0].formula_matrix
assert not isinstance(fm, jax.core.Tracer), f"formula_matrix is a {type(fm)}"

# Anything later must be able to reuse the cached setup: another trace at
# another nz, and an untraced call. A tracer anywhere in the cached setup
# (the formula matrix, or the NASA-9 coefficients the hvector closure holds)
# raises UnexpectedTracerError here.
k = 40
np.asarray(seed(Tco[:k], p_bar[:k], b))
np.asarray(ini_abun.eq_seed(Tco, p_bar, b, cfg))
print("TRACER_PROBE_OK")
"""


def _run(script: str, *args) -> None:
    """Run a probe subprocess and surface both streams on failure."""
    result = subprocess.run(
        [sys.executable, "-c", script, *map(str, args)],
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    assert result.returncode == 0, (
        f"probe exited {result.returncode}\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )


def _master_cfg(master_root: Path, config_name: str, scratch: Path) -> Path:
    """Write the master-side `vulcan_cfg.py` for one case.

    Upstream has no W39b config, so that one is resolved from the shipped
    YAML on top of the oracle's own legacy surface (its modules import `gs`,
    which VULCAN-JAX derives from Mp/Rp), and the
    checksummed W39b inputs are staged into the disposable copy.
    """
    from vulcan3.config import load_config
    from vulcan3.phy_const import G_grav

    if config_name != "W39b":
        return master_root / "cfg_examples" / "vulcan_cfg_HD189.py"

    cfg = load_config("W39b")
    legacy = (master_root / "vulcan_cfg.py").read_text()
    lines = [f"{key} = {getattr(cfg, key)!r}" for key in sorted(vars(cfg)) if not key.startswith("_")]
    lines.append(f"gs = {G_grav * float(cfg.Mp) / float(cfg.Rp) ** 2!r}")
    cfg_path = scratch / "vulcan_cfg_W39b_resolved.py"
    cfg_path.write_text(legacy + "\n# resolved W39b overrides\n" + "\n".join(lines) + "\n")
    for rel in (
        "thermo/SNCHO_photo_network_2025.txt",
        "atm/atm_W39b_evening_TP_Kzz.txt",
        "atm/stellar_flux/sflux-W39b_Tsai2023.txt",
    ):
        target = master_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PACKAGE_ROOT / rel, target)
    return cfg_path


@pytest.mark.parametrize(("config_name", "abundance"), [c[1:] for c in CASES], ids=[c[0] for c in CASES])
def test_seed_matches_upstream_fastchem(config_name: str, abundance: str) -> None:
    """The Gibbs seed reproduces upstream's FastChem seed to MAX_DEX."""
    from oracle import oracle_worktree

    with (
        tempfile.TemporaryDirectory(prefix="eq_seed_") as tmp,
        oracle_worktree(
            "vulcan2_ncho",
            # Both sides must start from the SAME composition: an
            # unmatched preset compares two atmospheres, not two codes.
            fastchem_abundance=abundance,
        ) as master_root,
    ):
        scratch = Path(tmp)
        jax_npz, master_npz = scratch / "jax.npz", scratch / "master.npz"
        cfg_path = _master_cfg(master_root, config_name, scratch)

        _run(_JAX_PROBE, PACKAGE_ROOT, jax_npz, config_name, abundance)
        _run(_MASTER_PROBE, master_root, master_npz, cfg_path)

        jax = np.load(jax_npz, allow_pickle=True)
        master = np.load(master_npz, allow_pickle=True)

    assert list(jax["species"]) == list(master["species"])
    np.testing.assert_array_equal(jax["pco"], master["pco"])
    np.testing.assert_allclose(jax["Tco"], master["Tco"], rtol=ATM_RTOL, atol=0.0)
    np.testing.assert_allclose(jax["M"], master["M"], rtol=ATM_RTOL, atol=0.0)

    m = np.asarray(master["y_ini"]) / np.asarray(master["M"])[:, None]
    j = np.asarray(jax["y_ini"]) / np.asarray(jax["M"])[:, None]
    sig = m > YMIX_FLOOR
    dex = np.abs(np.log10(np.where(sig, j, 1.0) / np.where(sig, m, 1.0)))
    worst = int(np.argmax(np.where(sig, dex, -1.0)))
    z, i = np.unravel_index(worst, dex.shape)
    assert dex[sig].max() <= MAX_DEX, (
        f"seed vs upstream FastChem: max {dex[sig].max():.3e} dex > "
        f"{MAX_DEX:.3e} on {list(jax['species'])[i]} at layer {z} "
        f"(T={float(jax['Tco'][z]):.1f} K, p={float(jax['pco'][z]) / 1e6:.3e} bar, "
        f"jax={j[z, i]:.6e}, master={m[z, i]:.6e}); median "
        f"{np.median(dex[sig]):.3e} over {int(sig.sum())} cells"
    )


@pytest.mark.parametrize("Tiso", [None, 300.0, 3000.0])
def test_seed_reproduces_its_element_vector_and_normalisation(Tiso) -> None:
    """The seed's own exact invariants, on the nominal column and two corners.

    Element ratios recovered from the output must be the requested `b`, the
    mixing ratios must sum to one, and species the seed excludes must be
    exactly zero. Nothing here depends on an oracle.
    """
    import jax.numpy as jnp

    from vulcan3 import composition, ini_abun
    from vulcan3.config import default_config

    cfg = default_config()
    Tco, p_bar = _hd189_column()
    if Tiso is not None:
        Tco = np.full_like(Tco, Tiso)
    b = ini_abun._element_vector(cfg)
    ymix = np.asarray(ini_abun.eq_seed(jnp.asarray(Tco), jnp.asarray(p_bar), jnp.asarray(b), cfg))

    assert np.isfinite(ymix).all()
    seed_idx = set(ini_abun._seed(bool(cfg.use_photo))[1].tolist())
    excluded = [i for i in range(len(composition.species)) if i not in seed_idx]
    assert np.all(ymix[:, excluded] == 0.0)
    np.testing.assert_allclose(ymix.sum(axis=1), 1.0, rtol=1e-12, atol=0.0)

    counts = np.asarray(composition.compo_array)
    elements = ini_abun.seed_elements(cfg)
    cols = [composition.atom_list.index(e) for e in elements]
    got = ymix @ counts[:, cols]  # (nz, E)
    got = got / got[:, [elements.index("H")]]
    np.testing.assert_allclose(got, np.broadcast_to(b / b[0], got.shape), rtol=ELEMENT_RTOL, atol=0.0)


def test_ratios_path_matches_the_config_path_and_moves_carbon() -> None:
    """The per-lane `ratios` seed equals the config path, bit for bit.

    Then doubling C/H must move the carbon carriers and leave helium alone:
    the retrieval's cold start drives this path.
    """
    import jax.numpy as jnp

    from vulcan3 import composition, ini_abun
    from vulcan3.config import default_config

    cfg = default_config()
    elements = [a for a in cfg.atom_list if a != "H"]
    ratios = {a: float(getattr(cfg, a + "_H")) for a in elements}
    # use_solar=False makes the config path read the same <X>_H the ratios
    # dict carries; with use_solar=True it would read the preset file instead
    # and the two would agree only to the preset's four-decimal dex.
    cfg.use_solar = False
    try:
        b_cfg = ini_abun._element_vector(cfg)
        np.testing.assert_array_equal(ini_abun._element_vector(cfg, ratios), b_cfg)
        idx = ini_abun.ratio_indices(elements, cfg)
        np.testing.assert_array_equal(
            np.asarray(ini_abun.element_vector(jnp.asarray([ratios[a] for a in elements]), idx, cfg)),
            b_cfg,
        )
        # No-override case, both branches of `use_solar`: the traced path
        # must read the same vector the host path reads, preset or <X>_H.
        empty = ini_abun.ratio_indices([], cfg)
        np.testing.assert_array_equal(np.asarray(ini_abun.element_vector(jnp.zeros(0), empty, cfg)), b_cfg)
        b_c2 = ini_abun._element_vector(cfg, {**ratios, "C": 2.0 * ratios["C"]})
    finally:
        cfg.use_solar = True

    np.testing.assert_array_equal(
        np.asarray(ini_abun.element_vector(jnp.zeros(0), empty, cfg)),
        ini_abun._element_vector(cfg),
    )
    # A partial override under use_solar=True (the retrieval's case): elements
    # it does not name keep the preset on both paths.
    c_only = {"C": 2.0 * ratios["C"]}
    np.testing.assert_array_equal(
        np.asarray(
            ini_abun.element_vector(jnp.asarray(list(c_only.values())), ini_abun.ratio_indices(list(c_only), cfg), cfg)
        ),
        ini_abun._element_vector(cfg, c_only),
    )

    Tco, p_bar = _hd189_column()
    y_cfg = np.asarray(ini_abun.eq_seed(jnp.asarray(Tco), jnp.asarray(p_bar), jnp.asarray(b_cfg), cfg))
    y_c2 = np.asarray(ini_abun.eq_seed(jnp.asarray(Tco), jnp.asarray(p_bar), jnp.asarray(b_c2), cfg))
    sp = composition.species
    assert np.all(y_c2[:, sp.index("CH4")] > y_cfg[:, sp.index("CH4")])
    np.testing.assert_allclose(y_c2[:, sp.index("He")], y_cfg[:, sp.index("He")], rtol=2e-2)


def test_solver_controls_force_a_retrace() -> None:
    """Changing the seed's solver controls retraces.

    `eq_seed` reads tol and max_iter at trace time, so after a converged run `max_iter = 1` must return the all-NaN
    non-convergence signal.
    """
    import jax.numpy as jnp

    from vulcan3 import ini_abun
    from vulcan3.config import default_config

    cfg = default_config()
    Tco, p_bar = _hd189_column()
    b = ini_abun._element_vector(cfg)
    args = (jnp.asarray(Tco), jnp.asarray(p_bar), jnp.asarray(b))

    def seed_jit():
        return ini_abun._seed_fns(*ini_abun._seed_key(cfg))[2]

    warm = np.asarray(seed_jit()(*args))
    assert np.isfinite(warm).all(), "the shipped controls must converge"

    keep = cfg.fastchem_newton_max_iter
    cfg.fastchem_newton_max_iter = 1
    try:
        one = np.asarray(seed_jit()(*args))
    finally:
        cfg.fastchem_newton_max_iter = keep
        ini_abun._seed_fns.cache_clear()

    seeded = ini_abun._seed(bool(cfg.use_photo))[1]
    assert np.isnan(one[:, seeded]).all(), (
        "max_iter=1 reused the converged trace: the config's solver controls "
        "are baked in at trace time and a new key must retrace"
    )


def test_seed_setup_survives_being_built_inside_a_trace() -> None:
    """The lazy setup is cached, so it must hold no tracers.

    vulcan-forward calls `eq_seed` from a jitted forward model, so the first
    call of a process can be a traced one; a JAX array built there escapes
    into every later trace. Runs in a subprocess because the check needs a
    process whose first seed call is the traced one.
    """
    _run(_TRACER_PROBE, PACKAGE_ROOT)


def test_seed_lane_is_bitwise_the_solo_seed_at_any_batch_width() -> None:
    """A lane's seed is the solo seed bit for bit at any `vmap` width, nested or under `jvp` (tangent exactly zero).

    Callers vmap the seed over lanes (vulcan-forward's batched prep,
    `run_queue`'s refill) and the solver amplifies a 1-ulp seed difference to
    the convergence scale, so a lane must not depend on how many lanes are
    seeded with it. A `jvp` must never reach ExoGibbs's kernel (a custom_vjp
    with no forward rule), so the seed declares a zero tangent. Every side is
    jitted: the eager and the jitted solo seed differ in the last bits.
    """
    import jax
    import jax.numpy as jnp

    from vulcan3 import ini_abun
    from vulcan3.config import default_config

    cfg = default_config()
    Tco, p_bar = _hd189_column()
    p = jnp.asarray(p_bar)
    T = jnp.asarray(Tco) * jnp.asarray([1.0, 1.05, 0.95, 1.10, 0.90])[:, None]
    B = jnp.tile(jnp.asarray(ini_abun._element_vector(cfg)), (5, 1))

    def seed(T, p, b):
        return ini_abun.eq_seed(T, p, b, cfg)

    lanes = jax.vmap(seed, in_axes=(0, None, 0))  # vulcan-forward's pattern

    def lane_1(tree):
        return jax.tree_util.tree_map(lambda x: x[1], tree)

    widths = {
        "width 1": lambda: lanes(T[1:2], p, B[:1])[0],
        "width 2": lambda: lanes(T[:2], p, B[:2])[1],
        "width 5": lambda: lanes(T, p, B)[1],
        "nested 2x2": lambda: jax.vmap(lanes, in_axes=(0, None, 0))(
            T[:4].reshape(2, 2, -1), p, B[:4].reshape(2, 2, -1)
        )[0, 1],
    }
    jvps = {  # (primal, tangent) along d/d(T scale) at scale 1
        "jvp": lambda: jax.jvp(lambda s: seed(T[1] * s, p, B[1]), (1.0,), (1.0,)),
        "jvp of vmap": lambda: lane_1(jax.jvp(lambda s: lanes(T * s, p, B), (1.0,), (1.0,))),
        "vmap of jvp": lambda: lane_1(
            jax.vmap(lambda t, b: jax.jvp(lambda s: seed(t * s, p, b), (1.0,), (1.0,)))(T, B)
        ),
    }
    solo = np.asarray(jax.jit(seed)(T[1], p, B[1]))
    assert np.isfinite(solo).all()
    for name, run in {**widths, **jvps}.items():
        out = jax.jit(run)()
        y, dy = (out, None) if name in widths else out
        y = np.asarray(y)
        assert np.array_equal(y, solo), (
            f"{name}: {int(np.sum(y != solo))} entries differ from the solo seed, max {np.abs(y - solo).max():.3e}"
        )
        assert dy is None or not np.asarray(dy).any(), f"{name}: nonzero tangent"


def _hd189_column() -> tuple[np.ndarray, np.ndarray]:
    """Return the shipped HD189 T-P column as (Tco [K], p [bar])."""
    from vulcan3.atm_setup import Atm
    from vulcan3.state import _AtmData

    make_atm = Atm()
    data_atm = make_atm.load_TPK(make_atm.f_pico(_AtmData()))
    return (
        np.asarray(data_atm.Tco, dtype=np.float64),
        np.asarray(data_atm.pco, dtype=np.float64) / 1.0e6,
    )


_PHOTO_OFF_PROBE = r"""
import os
import sys
from pathlib import Path

import numpy as np

root, out_npz, network, use_photo = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4] == "1"
os.chdir(root)
os.environ["VULCAN3_NETWORK"] = network

import jax.numpy as jnp

from vulcan3 import chem_funs, ini_abun, make_config
from vulcan3.atm_setup import Atm
from vulcan3.config import default_config
from vulcan3.state import _AtmData

# The process default keeps the shipped use_photo, as after a consumer's setup: the seed follows the run's config.
assert default_config().use_photo is True
cfg = make_config(use_photo=use_photo)
data_atm = Atm().load_TPK(Atm().f_pico(_AtmData()))
Tco = jnp.asarray(np.asarray(data_atm.Tco, dtype=np.float64))
p_bar = jnp.asarray(np.asarray(data_atm.pco, dtype=np.float64) / 1.0e6)
b = ini_abun.element_vector(jnp.zeros(0), ini_abun.ratio_indices([], cfg), cfg)
y = ini_abun.eq_seed(Tco, p_bar, b, cfg)
np.savez(out_npz, y=np.asarray(y), species=np.array(chem_funs.spec_list))
"""


def test_photo_off_seed_leaves_out_the_photolysis_only_species(tmp_path) -> None:
    """With use_photo off, VULCAN 2 drops the photo rows and their species (`make_chem_funs.py:74`).

    On CHO_photo_network_lowT that is O_1 and CH2_1 (38 species against 36). The seed must give them exactly 0 and
    give the other 36 what the network without its photo rows gives.
    """
    full = PACKAGE_ROOT / "thermo" / "CHO_photo_network_lowT.txt"
    lines = full.read_text().splitlines(keepends=True)
    cut = next(i for i, line in enumerate(lines) if line.startswith("# photo"))
    stripped = tmp_path / "CHO_photo_network_lowT_no_photo.txt"
    stripped.write_text("".join(lines[:cut]))

    _run(_PHOTO_OFF_PROBE, PACKAGE_ROOT, tmp_path / "full.npz", full, "0")
    _run(_PHOTO_OFF_PROBE, PACKAGE_ROOT, tmp_path / "stripped.npz", stripped, "1")
    a, b = np.load(tmp_path / "full.npz"), np.load(tmp_path / "stripped.npz")
    sp_full, sp_kept = list(a["species"]), list(b["species"])
    assert sp_full[: len(sp_kept)] == sp_kept
    assert sp_full[len(sp_kept) :] == ["O_1", "CH2_1"]
    assert np.isfinite(b["y"]).all()
    assert np.all(a["y"][:, len(sp_kept) :] == 0.0), "a photolysis-only species entered the photo-off seed"
    np.testing.assert_array_equal(a["y"][:, : len(sp_kept)], b["y"])


_SEQUESTER_PROBE = r"""
import os
import sys
from pathlib import Path

import numpy as np
import yaml

root, config_name, element_rtol, scale = Path(sys.argv[1]), sys.argv[2], float(sys.argv[3]), sys.argv[4] == "1"
os.chdir(root)
raw = {**yaml.safe_load((root / "configs" / "default.yaml").read_text()),
       **yaml.safe_load((root / "configs" / f"{config_name}.yaml").read_text())}
os.environ["VULCAN3_NETWORK"] = raw["network"]
os.environ["VULCAN3_ATOM_LIST"] = ",".join(raw["atom_list"])
os.environ["VULCAN3_COM_FILE"] = raw["com_file"]

import jax.numpy as jnp

from vulcan3 import composition, ini_abun
from vulcan3.atm_setup import Atm
from vulcan3.config import default_config, load_config
from vulcan3.state import _AtmData

cfg = load_config(config_name)
for key, value in vars(cfg).items():  # the process default is the run's config, as setup's overlay makes it
    setattr(default_config(), key, value)
assert cfg.sequestered_frac == {}


def same_bits(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


# {}: the vector the seed solves on is the unscaled one, bit for bit, on both paths.
unscaled = ini_abun._preset_vector(cfg) if cfg.use_solar is True else ini_abun._base_vector(cfg)
assert same_bits(ini_abun._element_vector(cfg), unscaled)
assert same_bits(ini_abun.element_vector(jnp.zeros(0), ini_abun.ratio_indices([], cfg), cfg), unscaled)

if scale:
    fracs = {"O": 0.2, "S": 0.1}
    seq = load_config(config_name, sequestered_frac=fracs)
    elements = ini_abun.seed_elements(cfg)
    factor = np.ones(len(elements))
    factor[elements.index("O")], factor[elements.index("S")] = 0.8, 0.9

    # Per-run path: the run's config is the argument, the process default keeps {}. The ratios replace O and S, and
    # the fraction applies after them.
    names = ["O", "C", "S"]
    ratios = jnp.asarray([5.0e-4, 3.0e-4, 1.0e-5])
    pairs = [
        (ini_abun._element_vector(cfg), ini_abun._element_vector(seq)),
        *(
            tuple(np.asarray(ini_abun.element_vector(r, ini_abun.ratio_indices(n, c), c)) for c in (cfg, seq))
            for r, n in ((ratios, names), (jnp.zeros(0), []))
        ),
    ]
    assert default_config().sequestered_frac == {}
    for before, after in pairs:
        for i, f in enumerate(factor):
            assert same_bits(after[i], before[i]) if f == 1.0 else after[i] == f * before[i], (elements[i], f)

    # Setup path: the host seed on the overlaid process config holds the scaled column.
    data_atm = Atm().load_TPK(Atm().f_pico(_AtmData()))
    column = (data_atm.pco, data_atm.Tco, data_atm.M)
    counts = np.asarray(composition.compo_array)[:, [composition.atom_list.index(e) for e in elements]]
    h = elements.index("H")
    per_h = {}
    for name, value in (("none", {}), ("seq", fracs)):
        default_config().sequestered_frac = value
        atoms = ini_abun.eq_column(*column) @ counts
        per_h[name] = atoms / atoms[:, [h]]
    b0 = ini_abun._element_vector(cfg)
    for name, f in (("none", 1.0), ("seq", factor)):
        np.testing.assert_allclose(per_h[name], np.broadcast_to(b0 * f / b0[h], per_h[name].shape), rtol=element_rtol)
print("SEQUESTER_PROBE_OK")
"""

# Every shipped config the EQ seed starts; Earth's const_mix start never reads the elemental vector.
_EQ_CONFIGS = sorted(
    p.stem
    for p in (PACKAGE_ROOT / "configs").glob("*.yaml")
    if yaml.safe_load(p.read_text()).get("ini_mix", "EQ") == "EQ"
)


@pytest.mark.parametrize("config_name", _EQ_CONFIGS)
def test_sequestration_scales_only_the_named_elements(config_name: str) -> None:
    """`sequestered_frac` {} seeds bit for bit as without it; {O: 0.2, S: 0.1} leaves exactly 0.8 of O and 0.9 of S.

    Every other element is unchanged to the bit, on the host path and the per-run (config-argument) path, after the
    `<X>_H` overrides and any per-run ratios, and the setup-path seed holds the scaled column. Each config runs in a
    subprocess: the network is import-frozen, and the scaling needs W39b's sulfur.
    """
    _run(_SEQUESTER_PROBE, PACKAGE_ROOT, config_name, ELEMENT_RTOL, "1" if config_name == "W39b" else "0")


@pytest.mark.parametrize(
    ("fracs", "ini_mix", "match"),
    [
        ({"H": 0.1}, "EQ", r"sequestered_frac\['H'\]=0\.1: H is the reference element.*Remove the key"),
        ({"S": 0.1}, "EQ", r"sequestered_frac\['S'\]=0\.1: 'S' is not in atom_list.*Remove the key"),
        ({"O": 1.0}, "EQ", r"sequestered_frac\['O'\]=1\.0 is not a number in \[0, 1\).*Set a value in \[0, 1\)"),
        ({"O": -0.1}, "EQ", r"sequestered_frac\['O'\]=-0\.1 is not a number in \[0, 1\).*Set a value in \[0, 1\)"),
        (0.2, "EQ", r"sequestered_frac=0\.2 is not a mapping.*\{O: 0\.2\}"),
        ({"O": 0.2}, "const_mix", r"sequestered_frac=\{'O': 0\.2\} acts only on .*ini_mix='EQ'.*Set it to \{\}"),
    ],
)
def test_sequestration_refuses_a_bad_or_inert_fraction(monkeypatch, fracs, ini_mix, match) -> None:
    """Refused at setup and, for an EQ run, on the per-run path too."""
    import jax.numpy as jnp

    from vulcan3 import ini_abun, make_config

    monkeypatch.setattr(ini_abun._CFG, "sequestered_frac", fracs)
    monkeypatch.setattr(ini_abun._CFG, "ini_mix", ini_mix)
    with pytest.raises(ValueError, match=match):
        ini_abun.compute_initial_abundance(None)
    if ini_mix == "EQ":
        cfg = make_config(sequestered_frac=fracs)
        with pytest.raises(ValueError, match=match):
            ini_abun.element_vector(jnp.zeros(0), ini_abun.ratio_indices([], cfg), cfg)


def test_the_old_oxygen_key_is_refused_with_its_remedy() -> None:
    from vulcan3 import make_config, validate_overrides

    old = r"O_sequestered_frac` -- use sequestered_frac: \{O: f\}"
    with pytest.raises(ValueError, match=old):
        make_config(O_sequestered_frac=0.2)
    with pytest.raises(ValueError, match=old):
        validate_overrides({"O_sequestered_frac": 0.2})
