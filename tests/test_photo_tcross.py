"""Photochemistry with T-dependent cross sections, where they ride `ProfileVars` (`p_absp_T_cross`, `p_cross_J_T`).

The only test with `T_cross_sp` non-empty: the loop's photolysis branch must give, from one state, what the setup path
gives from its own T-dependent tables, and an integration must run through it.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
from _helpers import relerr

from vulcan3.phy_const import UNDERFLOW_DENOM

PHOTO_RTOL = 1e-13  # the same kernels on the same state, by two routes


@pytest.mark.strict_isolation
def test_photo_tcross_rates():
    import vulcan3.legacy_io as op
    from vulcan3 import chem_funs, op_jax, outer_loop, rates_jax
    from vulcan3._loop_branches import _make_photo_branch
    from vulcan3.state import RunState, legacy_view, runstate_from_store

    vulcan_cfg = op.default_config()

    vulcan_cfg.count_max = 12
    vulcan_cfg.count_min = 1
    vulcan_cfg.use_print_prog = False
    vulcan_cfg.use_photo = True
    vulcan_cfg.use_ion = False
    vulcan_cfg.ini_mix = "const_mix"
    vulcan_cfg.atm_type = "isothermal"
    vulcan_cfg.Kzz_prof = "Pfunc"
    vulcan_cfg.nz = 30
    vulcan_cfg.T_cross_sp = ["H2O"]  # the trigger: non-empty T-dep cross sections
    # H2O must be present, or its T-dependent absorption adds nothing to tau.
    vulcan_cfg.const_mix = {**vulcan_cfg.const_mix, "H2O": 1e-3}
    vulcan_cfg.ini_update_photo_frq = 4

    rs = RunState.with_pre_loop_setup(vulcan_cfg)
    assert rs.photo_static.absp_T_cross.shape[0] > 0

    # The setup route on one state.
    var, atm, para = legacy_view(rs)
    var.k_arr = np.array(var.k_arr)  # compute_J writes it in place
    var_entry = copy.deepcopy(var)
    solver = op_jax.Ros2JAX(rs.photo_static)
    solver.compute_tau(var, atm)
    solver.compute_flux(var, atm)
    solver.compute_J(var, atm)
    rates_jax.apply_photo_remove(vulcan_cfg, var, chem_funs.NETWORK)

    # The loop's branch on the same state, reading its T-dependent arrays off the carry.
    integ = outer_loop.OuterLoop(solver, op.Output())
    integ.ensure_runner(var_entry, atm)
    rs_entry = runstate_from_store(var_entry, atm, para)._replace(photo_static=rs.photo_static)
    final = _make_photo_branch(integ.photo_static)(integ.pack_state(rs_entry))

    assert np.any(np.asarray(integ.photo_static.branch_T_active))
    for name in ("tau", "aflux", "sflux"):
        err = relerr(getattr(final, name), getattr(var, name), floor=UNDERFLOW_DENOM)
        assert err <= PHOTO_RTOL, f"{name} relerr {err:.3e}"
    err = relerr(final.k_arr[1:], var.k_arr[1:], floor=1e-300)
    assert err <= PHOTO_RTOL, f"k_arr relerr {err:.3e}"

    rs_out = outer_loop.OuterLoop(op_jax.Ros2JAX(), op.Output())(rs)
    assert np.all(np.isfinite(np.asarray(rs_out.step.y))), "non-finite y after T_cross_sp run"
    assert int(rs_out.params.count) >= 12, f"run did not progress (count={int(rs_out.params.count)})"
