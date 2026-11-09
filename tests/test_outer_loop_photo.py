"""The in-runner photo branch must match the Python-side compute_tau / compute_flux / compute_J round-trip on HD189.

Both paths share the same JAX kernels, so tau / fluxes / aflux_change /
photo k_arr rows and the .vul writer's J_sp must agree to <= 1e-13; anything
beyond that is a wiring bug in the carry plumbing.
"""

from __future__ import annotations

import copy

import jax.numpy as jnp
from _helpers import relerr

from vulcan3.phy_const import UNDERFLOW_DENOM

PHOTO_RTOL = 1e-13


def test_photo_branch_matches_python_photo_path():
    from vulcan3.config import default_config

    vulcan_cfg = default_config()

    import vulcan3.legacy_io as op
    from vulcan3 import op_jax, outer_loop, photo as _photo_mod
    from vulcan3._loop_branches import _make_photo_branch
    from vulcan3._pack_state import unpack_state
    from vulcan3.state import RunState, legacy_view, runstate_from_store

    # The HD189 reference state with the photo pre-loop, as the CLI builds it before the integration loop.
    rs = RunState.with_pre_loop_setup(vulcan_cfg)
    data_var, data_atm, data_para = legacy_view(rs)
    output = op.Output()

    # with_pre_loop_setup already ran the photo pipeline once. Reset var.k_arr
    # to the post-rate-build, pre-photo values via setup_var_k, then re-run
    # Path A explicitly so there is a clean pre/post split.
    import vulcan3.rates_jax as _rates_mod

    _network = _rates_mod.setup_var_k(vulcan_cfg, data_var, data_atm)

    # Path B starts from a copy of the pre-photo state; otherwise it computes
    # the 2nd photo update, not the 1st.
    var_B = copy.deepcopy(data_var)

    # Path A: Python-side compute_tau / compute_flux / compute_J
    solver = op_jax.Ros2JAX()
    if rs.photo_static is not None:
        solver._photo_static = rs.photo_static
    solver.compute_tau(data_var, data_atm)
    solver.compute_flux(data_var, data_atm)
    solver.compute_J(data_var, data_atm)
    _rates_mod.apply_photo_remove(vulcan_cfg, data_var, _network)

    # Path B: photo branch inside the JAX runner
    integ = outer_loop.OuterLoop(solver, output)
    integ.ensure_runner(var_B, data_atm)
    rs_entry = runstate_from_store(var_B, data_atm, data_para)._replace(photo_static=rs.photo_static)
    init_state = integ.pack_state(rs_entry)
    photo_branch = _make_photo_branch(integ._photo_static)
    final_state = photo_branch(init_state)

    # Regression: the in-runner photo branch must use the dynamic dz carried
    # by JaxIntegState, not the initial photo_static.dz closed over at trace
    # time. update_mu_dz changes dz during long runs, and op.compute_tau reads
    # the current atm.dz on every photo update.
    dz_scale = jnp.linspace(0.95, 1.05, init_state.dz.shape[0])
    dynamic_state = init_state._replace(dz=init_state.dz * dz_scale)
    dynamic_final = photo_branch(dynamic_state)
    tau_dynamic_ref = _photo_mod.compute_tau_jax(
        dynamic_state.y,
        dynamic_state.dz,
        integ._photo_static.photo_data,
    )
    dyn_dz_err = relerr(dynamic_final.tau, tau_dynamic_ref, floor=UNDERFLOW_DENOM)
    assert dyn_dz_err <= PHOTO_RTOL, "photo branch did not use dynamic state.dz"

    for name in ("tau", "aflux", "sflux", "dflux_u"):
        err = relerr(getattr(final_state, name), getattr(data_var, name), floor=UNDERFLOW_DENOM)
        assert err <= PHOTO_RTOL, f"{name} relerr {err:.3e}"

    err_change = relerr(float(final_state.aflux_change), data_var.aflux_change, floor=1e-300)
    assert err_change <= PHOTO_RTOL, f"aflux_change relerr {err_change:.3e}"

    # Every k_arr row, photo-driven or not: both paths write the same J*.
    err_k = relerr(final_state.k_arr[1:], data_var.k_arr[1:], floor=1e-300)
    assert err_k <= PHOTO_RTOL, f"k_arr relerr {err_k:.3e}"

    # J_sp through the production .vul writer (every branch plus the (sp, 0)
    # totals, op.py:2764, 2783) against compute_J's.
    rs_B = unpack_state(integ, final_state, rs_entry)
    J_sp_B = op._synthesize_save_dicts(rs_B, vulcan_cfg)[0]["J_sp"]
    for key, ref in data_var.J_sp.items():
        assert key in J_sp_B, f"J_sp[{key}] missing from the writer"
        err = relerr(J_sp_B[key], ref, floor=1e-300)
        assert err <= PHOTO_RTOL, f"J_sp[{key}] relerr {err:.3e}"
