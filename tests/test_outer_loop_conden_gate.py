"""The condensation and fix-species gates read entry-time `s.t`, not `t_next`.

Master gates condensation on var.t before save_step advances it (op.py:856; save_step at op.py:918), so the step that
crosses start_conden_time does not condense. One trial step of a condensing column, across each boundary.
"""

from __future__ import annotations

from _helpers import run_child

NETWORK = "thermo/NCHO_photo_network_lowT_Jupiter.txt"

_CHILD = r"""
import jax, jax.numpy as jnp, numpy as np
import vulcan3
from vulcan3 import legacy_io, op_jax, outer_loop
from vulcan3._loop_body import body_fn
from vulcan3.phy_const import G_grav

Rp = 7.1492e9
cfg = vulcan3.load_config(
    "default", atm_type="analytical", atm_base="H2", use_vm_mol=False, use_hybrid_vm_mol=False, nz=60, P_b=5e9,
    P_t=1e-2, Rp=Rp, Mp=2479.0 * Rp**2 / G_grav, use_Kzz=True, Kzz_prof="const", const_Kzz=1e8, use_vz=False,
    use_moldiff=True, use_photo=False, use_ion=False, ini_mix="const_mix",
    const_mix={"H2": 0.864, "He": 0.134, "CH4": 1.8e-3, "NH3": 7e-4, "H2O": 1.0e-3},
    use_topflux=False, use_botflux=False, use_fix_sp_bot={}, use_ini_cold_trap=False, use_print_prog=False,
    use_condense=True, use_settling=True, condense_sp=["H2O", "NH3"], non_gas_sp=["H2O_l_s", "NH3_l_s"],
    r_p={"H2O_l_s": 5e-5, "NH3_l_s": 5e-5}, rho_p={"H2O_l_s": 0.9, "NH3_l_s": 0.7}, use_relax=[],
    humidity=1.0, start_conden_time=0.0, stop_conden_time=1e22, count_max=30, count_min=1, fix_species=["H2O"],
    network="thermo/NCHO_photo_network_lowT_Jupiter.txt",
)
rs = vulcan3.RunState.with_pre_loop_setup(cfg)
integ = outer_loop.OuterLoop(op_jax.Ros2JAX(), legacy_io.Output(cfg=cfg), cfg=cfg)
state, atm = integ.prepare_runstate(rs)
DT = 1e-16  # s: short enough that this far-from-equilibrium start accepts the trial
REJECTED_DT = 1e-12  # s: this start rejects it
env = integ._make_runner_jvp.args[0]  # the runner's own closure inputs


@jax.jit
def step(t, start, stop, dt):
    return body_fn(env._replace(start_conden_time=start, stop_conden_time=stop), state._replace(t=t, dt=dt), atm)[0]


NEVER = 1e30


def trial(t, start, stop=NEVER):
    out = step(jnp.float64(t), jnp.float64(start), jnp.float64(stop), jnp.float64(DT))
    assert int(out.accept_count) == 1, "the trial step was rejected, so it says nothing about the gates"
    return np.asarray(out.k_arr), bool(out.fix_species_started)


off, _ = trial(0.0, NEVER)
same = lambda k: np.array_equal(k, off)  # the branch writes the condensation rows of k_arr

# Condensation: not before the time, not on the step that crosses it, from the step that starts on it onward.
assert same(trial(0.0, 2.0 * DT)[0]), "condensed before start_conden_time"
assert same(trial(0.0, 0.5 * DT)[0]), "the step crossing start_conden_time condensed: the gate read t_next"
assert not same(trial(0.0, 0.0)[0]), "no condensation on the step entering at start_conden_time"
assert not same(trial(DT, 0.5 * DT)[0]), "no condensation after start_conden_time"

# A rejected trial does not condense, whatever the time.
rejected = step(jnp.float64(0.0), jnp.float64(0.0), jnp.float64(NEVER), jnp.float64(REJECTED_DT))
assert int(rejected.accept_count) == 0, "the long trial was accepted, so it says nothing about the accept gate"
assert same(np.asarray(rejected.k_arr)), "a rejected trial condensed"

# The fix_species freeze: strictly after stop_conden_time, at entry time.
assert not trial(0.0, 0.0, 0.5 * DT)[1], "the step crossing stop_conden_time froze: the gate read t_next"
assert not trial(0.0, 0.0, 0.0)[1], "froze on the step entering at stop_conden_time"
assert trial(DT, 0.0, 0.5 * DT)[1], "no freeze after stop_conden_time"
print("PASS")
"""


def test_conden_gates_use_entry_time_t() -> None:
    res = run_child(_CHILD, network=NETWORK, label="condensation gates")
    assert res.stdout.strip().endswith("PASS"), res.stdout
