"""End-to-end condensation runtime coverage, in a subprocess.

The default NCHO network has no condensate species, so the curated suite never
exercises the assembled condensation runtime path. The network is
import-locked, so the child selects `SNCHO_photo_network_2025.txt` (the one
vendored network with both `H2O_l_s` and `S8_l_s`) via `$VULCAN3_NETWORK`.
Self-contained: isothermal TP + const_mix init, no photo, no EQ seed.

Checks: (1) settling velocities are downward for each condensate column and
exactly zero for gas species; (2) a short H2O-relaxation run stays finite and
actually moves mass into `H2O_l_s` (full runner conden path); (3)
`use_sat_surfaceH2O` reaches the runner's bottom pin on both config paths.
"""

from __future__ import annotations

from _helpers import run_child

CONDEN_NETWORK = "thermo/SNCHO_photo_network_2025.txt"


_CHILD = r"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")

from pathlib import Path
repo = Path(sys.argv[1])
sys.path.insert(0, str(repo / "src"))        # resolve `import vulcan3` to src/
os.chdir(repo / "src" / "vulcan3")         # data paths are package-relative

import numpy as np
import vulcan3  # first import -> parses the condensate network
import vulcan3.composition as composition
import vulcan3.chem_funs as chem_funs

sl = list(composition.species)
for need in ("H2O", "S8", "H2O_l_s", "S8_l_s"):
    assert need in sl, f"condensate network missing required species {need!r}"

from vulcan3.config import default_config
cfg = default_config()
# Synthetic isothermal atmosphere, const_mix init, no photo -> no data files.
cfg.atm_type = "isothermal"; cfg.Tiso = 250.0           # cold enough for H2O to supersaturate
cfg.atm_base = "N2"; cfg.use_moldiff = True             # settling requires use_moldiff
cfg.use_Kzz = True; cfg.Kzz_prof = "const"; cfg.const_Kzz = 1e5; cfg.use_vz = False
cfg.use_photo = False; cfg.use_ion = False
cfg.ini_mix = "const_mix"
cfg.const_mix = {"N2": 0.79, "O2": 0.20, "H2O": 1e-2, "CO2": 4e-4}
cfg.nz = 50; cfg.P_b = 1e6; cfg.P_t = 1e-1
cfg.use_topflux = False; cfg.use_botflux = False
cfg.use_fix_sp_bot = {}; cfg.use_ini_cold_trap = False; cfg.use_print_prog = False
cfg.use_condense = True

# ---- Check 1: settling velocities for both condensates ----
cfg.use_settling = True
cfg.condense_sp = ["H2O", "S8"]; cfg.non_gas_sp = ["H2O_l_s", "S8_l_s"]
cfg.r_p = {"H2O_l_s": 5e-3, "S8_l_s": 1e-4}
cfg.rho_p = {"H2O_l_s": 0.9, "S8_l_s": 2.07}
cfg.use_relax = []

from vulcan3.atm_setup import Atm
from vulcan3.state import _Variables, _AtmData
from vulcan3.ini_abun import InitialAbun
import vulcan3.legacy_io as op

dv = _Variables(); da = _AtmData(); make_atm = Atm()
da = make_atm.f_pico(da); da = make_atm.load_TPK(da); make_atm.sp_sat(da)
dv = InitialAbun().ini_y(dv, da)
da = make_atm.f_mu_dz(dv, da, op.Output(cfg=cfg))
vs = np.asarray(da.vs); nz = da.Tco.shape[0]
assert vs.shape == (nz - 1, len(sl)), f"vs shape {vs.shape}"
for cond in ("H2O_l_s", "S8_l_s"):
    j = sl.index(cond)
    assert np.all(vs[:, j] < 0.0), f"{cond} settling velocity not downward"
gas = [i for i in range(len(sl)) if sl[i] not in ("H2O_l_s", "S8_l_s")]
assert np.all(vs[:, gas] == 0.0), "settling velocity nonzero on a gas species"
print("SETTLING_OK")

# ---- Check 2: short H2O-relax run forms condensate ----
cfg.condense_sp = ["H2O"]; cfg.non_gas_sp = ["H2O_l_s"]
cfg.r_p = {"H2O_l_s": 0.01}; cfg.rho_p = {"H2O_l_s": 0.9}
cfg.use_relax = ["H2O"]; cfg.humidity = 1.0
cfg.start_conden_time = 0.0; cfg.stop_conden_time = 1e22
cfg.count_max = 50; cfg.count_min = 1

from vulcan3.state import RunState
import vulcan3.op_jax as op_jax
import vulcan3.outer_loop as outer_loop

t0 = time.time()
rs = RunState.with_pre_loop_setup(cfg)
integ = outer_loop.OuterLoop(op_jax.Ros2JAX(), op.Output(cfg=cfg), cfg=cfg)
rs_out = integ(rs)
y = np.asarray(rs_out.step.y)
assert np.all(np.isfinite(y)), "non-finite densities after conden run"
cond_total = float(y[:, sl.index("H2O_l_s")].sum())
assert cond_total > 0.0, f"no H2O condensate formed (total={cond_total:.3e})"
print(f"CONDEN_OK t={float(rs_out.step.t):.2e} count={int(rs_out.params.count)} "
      f"H2O_l_s_total={cond_total:.3e} ({time.time()-t0:.1f}s)")

# ---- Check 3: use_sat_surfaceH2O pins bottom H2O at its surface saturation ----
# On default_config() and on a separate Config (the load_config path, where
# setup runs on a copy of the config); neither config is mutated.
import copy
from vulcan3._build_statics import build_statics
from vulcan3.state import legacy_view
cfg.use_sat_surfaceH2O = True
for c in (cfg, copy.deepcopy(cfg)):
    rs = RunState.with_pre_loop_setup(c)
    var, atm, _ = legacy_view(rs)
    st = build_statics(outer_loop.OuterLoop(op_jax.Ros2JAX(), op.Output(cfg=c), cfg=c), var, atm)
    pins = {sl[int(i)]: float(m) for i, m in
            zip(np.asarray(st.fix_sp_bot_idx), np.asarray(st.fix_sp_bot_mix))}
    want = float(np.asarray(rs.metadata.sat_mix["H2O"])[0])
    assert st.use_fix_sp_bot and pins == {"H2O": want}, (pins, want)
    assert c.use_fix_sp_bot == {} and default_config().use_fix_sp_bot == {}
print("SAT_SURFACE_OK")
print("PASS")
"""


def test_condensation_runtime_subprocess():
    res = run_child(_CHILD, network=CONDEN_NETWORK, label="condensation")
    assert "SETTLING_OK" in res.stdout, res.stdout
    assert "CONDEN_OK" in res.stdout, res.stdout
    assert "SAT_SURFACE_OK" in res.stdout, res.stdout
    assert res.stdout.strip().endswith("PASS")
