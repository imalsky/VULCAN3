"""In-runner atm refresh vs master's update_mu_dz / update_phi_esc on HD189.

Same kernel on both sides: master's sweeps write `zco[i+1]` before reading
it for `g[i+1]`, so its g(z) is self-consistent exactly as
`update_mu_dz_jax` is. Every
refreshed field, including the diffusion-limited escape flux (`diff_esc`
forced to ['H'] on both sides), is held to REFRESH_RTOL.
"""

from __future__ import annotations

import copy

import numpy as np
from _helpers import relerr
from oracle import assert_below, enter_master, leave_master, run_oracle_subprocess

REFRESH_RTOL = 1e-12
EXACT_RTOL = 1e-13  # the same formula on both sides, roundoff only


def main() -> None:
    enter_master()
    from vulcan3.config import default_config

    vulcan_cfg = default_config()
    import op
    import vulcan_cfg as master_cfg

    # Exercise update_phi_esc on both sides (HD189 ships diff_esc = []).
    vulcan_cfg.diff_esc = ["H"]
    master_cfg.diff_esc = ["H"]

    leave_master()
    from vulcan3 import atm_refresh, op_jax, outer_loop
    from vulcan3.atm_setup import Atm
    from vulcan3.state import RunState, legacy_view, runstate_from_store

    # --- Build HD189 reference state ---
    rs = RunState.with_pre_loop_setup(vulcan_cfg)
    data_var, data_atm, data_para = legacy_view(rs)
    make_atm = Atm()
    output = op.Output()

    # Perturb ymix slightly so the refresh produces non-trivial deltas
    # vs the initial state (otherwise mu_post == mu_pre and it's not a
    # real test of the loop).
    rng = np.random.default_rng(0)
    pert = 1.0 + 1e-3 * rng.standard_normal(data_var.ymix.shape)
    data_var.ymix = data_var.ymix * pert
    data_var.ymix = data_var.ymix / np.sum(data_var.ymix, axis=1, keepdims=True)
    data_var.y = data_atm.n_0[:, None] * data_var.ymix

    # --- Path A: Python-side update_mu_dz / update_phi_esc ---
    atm_A = copy.deepcopy(data_atm)
    var_A = copy.deepcopy(data_var)
    integ_ref = op.Integration(op.Ros2(), output)
    atm_A = integ_ref.update_mu_dz(var_A, atm_A, make_atm)
    atm_A = integ_ref.update_phi_esc(var_A, atm_A)

    # --- Path B: atm-refresh branch via JAX runner ---
    solver_B = op_jax.Ros2JAX()
    if vulcan_cfg.use_photo and rs.photo_static is not None:
        solver_B._photo_static = rs.photo_static
    integ = outer_loop.OuterLoop(solver_B, output)
    integ._ensure_runner(data_var, data_atm)

    # Drive the refresh kernels directly on a packed initial state: this
    # exercises update_mu_dz_jax + update_phi_esc_jax wiring without
    # depending on the photo branch / chem step.
    init_state = integ._pack_state_from_runstate(
        runstate_from_store(data_var, data_atm, data_para)._replace(photo_static=rs.photo_static)
    )
    st = integ._refresh_static
    mu_B, g_B, Hp_B, dz_B, zco_B, dzi_B, Hpi_B = atm_refresh.update_mu_dz_jax(init_state.ymix, st)
    top_flux_B = atm_refresh.update_phi_esc_jax(init_state.y, g_B, Hp_B, init_state.top_flux, st)

    from vulcan3.phy_const import UNDERFLOW_DENOM

    assert_below(
        {
            label: (relerr(B, A, floor=UNDERFLOW_DENOM), rtol)
            for label, A, B, rtol in (
                ("mu", atm_A.mu, mu_B, EXACT_RTOL),
                ("g", atm_A.g, g_B, REFRESH_RTOL),
                ("Hp", atm_A.Hp, Hp_B, REFRESH_RTOL),
                ("dz", atm_A.dz, dz_B, REFRESH_RTOL),
                ("dzi", atm_A.dzi, dzi_B, REFRESH_RTOL),
                ("Hpi", atm_A.Hpi, Hpi_B, REFRESH_RTOL),
                ("zco", atm_A.zco, zco_B, REFRESH_RTOL),
                ("top_flux", atm_A.top_flux, top_flux_B, REFRESH_RTOL),
            )
        }
    )


def test_main():
    run_oracle_subprocess(__file__, "vulcan2_ncho", "cfg_examples/vulcan_cfg_HD189.py")


if __name__ == "__main__":
    main()
