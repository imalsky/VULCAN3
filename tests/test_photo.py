"""Validate JAX photochemistry kernels against VULCAN-master.

Runs VULCAN's compute_tau on the canonical HD189 state, then runs the JAX
version on the same state and compares element-wise.
"""

from __future__ import annotations

import numpy as np
from _helpers import relerr
from oracle import assert_below, enter_master, leave_master, run_oracle_subprocess, upstream_preloop


# size-limit-ok: one oracle script, run in a subprocess on a copy of master
def main() -> None:
    enter_master()
    import jax.numpy as jnp

    from vulcan3.config import default_config

    vulcan_cfg = default_config()
    import op

    import vulcan3.photo as photo_mod
    from vulcan3.atm_setup import Atm
    from vulcan3.ini_abun import InitialAbun

    # Private containers on purpose: master's pipeline must mutate a shared
    # (var, atm) that the JAX side then reads (var.cross[sp] etc.).
    from vulcan3.state import _AtmData, _Variables

    # === Set up state with photochemistry ===
    make_atm = Atm()
    rate = op.ReadRate()
    data_var, data_atm = upstream_preloop(
        vulcan_cfg, _Variables(), _AtmData(), make_atm, rate, InitialAbun(), op.Output()
    )

    # Photochem setup
    if vulcan_cfg.use_photo:
        rate.make_bins_read_cross(data_var, data_atm)
        make_atm.read_sflux(data_var, data_atm)

    # Run VULCAN's compute_tau
    solver_v = op.Ros2()
    solver_v.compute_tau(data_var, data_atm)
    tau_ref = data_var.tau.copy()

    # === Pack photo data and run JAX version ===
    leave_master()
    from vulcan3 import chem_funs

    species_list = chem_funs.spec_list

    # Build the dense `PhotoStaticInputs` and derive the runtime
    # `PhotoData` from it; master's `make_bins_read_cross` above already
    # populated the dicts master's compute_tau needs.
    from vulcan3 import photo_setup

    static = photo_setup._build_photo_static_dense(data_var, data_atm)
    static = static.with_din12_indx(int(data_var.sflux_din12_indx))
    photo_data = photo_mod.photo_data_from_static(static, species_list)

    tau_jax = np.asarray(photo_mod.compute_tau_jax(jnp.asarray(data_var.y), jnp.asarray(data_atm.dz), photo_data))
    assert tau_jax.shape == tau_ref.shape, (tau_jax.shape, tau_ref.shape)

    # === compute_flux validation ===
    # Reset VULCAN's dflux_u/d to zero so JAX's zero-init matches first-call behavior
    data_var.dflux_u = np.zeros_like(data_var.dflux_u)
    data_var.dflux_d = np.zeros_like(data_var.dflux_d)
    data_var.tau = tau_ref  # restore tau (may have been modified)
    data_var.ymix = data_var.y / np.vstack(np.sum(data_var.y, axis=1))
    solver_v.compute_flux(data_var, data_atm)
    aflux_ref = data_var.aflux.copy()
    from vulcan3.phy_const import hc

    # Master's op.compute_flux reads sl_angle/edd from its OWN vulcan_cfg
    # (both ship 48 deg for HD189; 83 deg is W39b's value); feed the JAX kernel
    # the angle master actually used or the comparison is invalid.
    aflux_jax, _, _, _ = photo_mod.compute_flux_jax(
        jnp.asarray(tau_ref),
        jnp.asarray(data_var.sflux_top),
        jnp.asarray(data_var.ymix),
        photo_data,
        jnp.asarray(data_var.bins),
        float(np.cos(op.vulcan_cfg.sl_angle)),
        float(op.vulcan_cfg.edd),
        0.0,  # ag0=0 in phy_const
        float(hc),
        jnp.zeros_like(jnp.asarray(data_var.dflux_u)),  # dflux_u_prev = zeros (first-call match)
        ag0_is_zero=True,
    )
    aflux_jax = np.asarray(aflux_jax)

    # Filter to non-trivial flux values to avoid noise comparisons
    nonzero_mask = np.abs(aflux_ref) > 1e-30
    flux_err = relerr(aflux_jax, aflux_ref, mask=nonzero_mask) if nonzero_mask.any() else 0.0

    # === compute_J validation ===
    solver_v.compute_J(data_var, data_atm)
    J_ref = dict(data_var.J_sp)

    photo_J = photo_mod.photo_J_data_from_static(static)
    J_jax = photo_mod.compute_J_jax(jnp.asarray(aflux_jax), photo_J)

    branches = [k for k in photo_J.branch_keys + photo_J.branch_T_keys if k in J_ref]
    print(f"compute_J compared over {len(branches)} branches")
    J_err = float(np.max([relerr(J_jax[k], J_ref[k]) for k in branches], initial=0.0))

    assert_below(
        {
            "compute_tau": (relerr(tau_jax, tau_ref), 1e-10),
            "compute_flux (nontrivial entries)": (flux_err, 1e-6),
            "compute_J": (J_err, 1e-6),
        }
    )


def test_main():
    run_oracle_subprocess(__file__, "vulcan2_ncho", "cfg_examples/vulcan_cfg_HD189.py")


if __name__ == "__main__":
    main()
