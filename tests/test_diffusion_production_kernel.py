"""Validate the PRODUCTION JAX diffusion kernel against the NumPy reference (`diffusion_numpy_ref`).

The reference is master-validated elsewhere, so this needs no VULCAN-master oracle.

Pins, for the default 'gravity' mode, the upwind 'vm' mode and vertical advection ('vz'):
  1. coefficient arrays (A/B/C eddy + mol) at ~machine precision,
  2. the diffusion operator output on significant cells.
"""

from __future__ import annotations

import numpy as np
import pytest
from _helpers import relerr

COEF_RTOL = 1e-11  # C_mol FP-ordering noise reaches ~2e-12; eddy terms ~5e-16
OP_RTOL = 1e-4  # the operator is a small residue of large cancellations (floor ~1e-5)
OP_FLOOR = 1e-12  # absolute and peak-relative floor for significant operator cells


@pytest.mark.parametrize("mode", ["gravity", "vm", "vz", "-vz"])
def test_production_kernel_matches_reference(mode):
    """The 'vm' case pins the hot-path upwind (`use_vm_mol`) with a nonzero mixed-sign interface drift, (nz-1, ni).

    The 'vz' cases pin vertical advection (`use_vz`) with a mixed-sign wind, (nz-1,), on the central scheme, in both
    signs: each boundary row has one term per wind direction.
    """
    import diffusion_numpy_ref as diff_ref
    import jax.numpy as jnp

    from vulcan3 import jax_step
    from vulcan3.config import default_config
    from vulcan3.state import RunState, legacy_view

    vulcan_cfg = default_config()
    # The NumPy reference's gravity mode is the CENTRAL scheme, and the vm case
    # needs the all-zero-vm state that use_vm_mol=False builds.
    vulcan_cfg.use_vm_mol = False
    vulcan_cfg.use_hybrid_vm_mol = False
    rs = RunState.with_pre_loop_setup(vulcan_cfg)
    data_var, data_atm, _ = legacy_view(rs)
    y = np.asarray(data_var.y, dtype=np.float64)
    nz, ni = y.shape

    if mode == "vm":
        # Inject a nonzero, mixed-sign interface drift so both upwind branches
        # ((vm>0)/(vm<0)) fire in both the production kernel and the reference.
        rng = np.random.default_rng(1)
        data_atm.vm = (rng.standard_normal((nz - 1, ni)) * 50.0).astype(np.float64)
        vulcan_cfg.use_vm_mol = True
    if mode in ("vz", "-vz"):
        # A mixed-sign wind (cm/s) so both upwind branches of every interior row fire.
        data_atm.vz = np.random.default_rng(2).standard_normal(nz - 1) * (100.0 if mode == "vz" else -100.0)
        mode = "gravity"

    # Production kernel
    atm_static = jax_step.make_atm_static(data_atm, ni, nz, cfg=vulcan_cfg)
    grav = jax_step.compute_diff_grav(atm_static)
    A_eddy, B_eddy, C_eddy, A_mol, B_mol, C_mol, _ = jax_step._build_diff_coeffs_jax(jnp.asarray(y), atm_static, grav)
    diff_prod = np.asarray(
        jax_step._apply_diffusion_jax(jnp.asarray(y), A_eddy, B_eddy, C_eddy, A_mol, B_mol, C_mol, atm_static)
    )

    # NumPy reference
    coeffs = diff_ref.build_diffusion_coeffs(y, data_atm, vulcan_cfg, mode=mode)
    diff_numpy = diff_ref.apply_diffusion(y, coeffs)

    # 1. Coefficients are identical formulas -> machine precision (FP-ordering only).
    for label, p, r in (
        ("A_eddy", A_eddy, coeffs.A_eddy),
        ("B_eddy", B_eddy, coeffs.B_eddy),
        ("C_eddy", C_eddy, coeffs.C_eddy),
        ("A_mol", A_mol, coeffs.A_mol),
        ("B_mol", B_mol, coeffs.B_mol),
        ("C_mol", C_mol, coeffs.C_mol),
    ):
        r = np.asarray(r)
        err = relerr(p, r, floor=1e-30 * max(np.abs(r).max(), 1e-300))
        assert err < COEF_RTOL, f"{mode}-mode {label} relerr {err:.3e}"

    # 2. Operator output on significant cells.
    abs_tol = max(OP_FLOOR, OP_FLOOR * np.abs(diff_numpy).max())
    err = relerr(diff_prod, diff_numpy, floor=abs_tol)
    assert err < OP_RTOL, f"{mode}-mode diffusion operator relerr {err:.3e}"


CLOSED_COLUMN_TOL = 1e-10  # measured 3e-12 or less; upstream's boundary rows give 1.5e-4 in gravity mode (C35)


@pytest.mark.parametrize("mode", ["gravity", "vm", "vz"])
def test_closed_transport_conserves_the_weighted_column(mode):
    """With closed boundaries the transport operator moves each species and creates none.

    The interface fluxes telescope under `operator_column_weights`, so the weighted column tendency of every species
    is round-off against the weighted sum of its cell tendencies. HD189's gravity falls with height, which is what
    upstream's boundary rows get wrong: they weigh the neighbour cell with the boundary cell's gravity.
    """
    import jax.numpy as jnp

    from vulcan3 import jax_step
    from vulcan3.config import default_config
    from vulcan3.ini_abun import operator_column_weights
    from vulcan3.state import RunState, legacy_view

    vulcan_cfg = default_config()
    vulcan_cfg.use_vm_mol = False
    vulcan_cfg.use_hybrid_vm_mol = False
    rs = RunState.with_pre_loop_setup(vulcan_cfg)
    data_var, data_atm, _ = legacy_view(rs)
    y = np.asarray(data_var.y, dtype=np.float64)
    nz, ni = y.shape
    assert np.ptp(np.asarray(data_atm.g)) > 0.0, "a constant gravity cannot see the boundary rows"
    if mode == "vm":
        data_atm.vm = np.random.default_rng(1).standard_normal((nz - 1, ni)) * 50.0
        vulcan_cfg.use_vm_mol = True
    if mode == "vz":
        data_atm.vz = np.random.default_rng(2).standard_normal(nz - 1) * 100.0

    atm_static = jax_step.make_atm_static(data_atm, ni, nz, cfg=vulcan_cfg)
    coeffs = jax_step._build_diff_coeffs_jax(jnp.asarray(y), atm_static, jax_step.compute_diff_grav(atm_static))[:6]
    tendency = np.asarray(jax_step._apply_diffusion_jax(jnp.asarray(y), *coeffs, atm_static))
    weights = np.asarray(operator_column_weights(np.asarray(data_atm.dz)))[:, None]
    moved = (weights * np.abs(tendency)).sum(axis=0)
    created = np.abs((weights * tendency).sum(axis=0))
    assert moved.max() > 0.0
    worst = float(np.max(created[moved > 0.0] / moved[moved > 0.0]))
    assert worst < CLOSED_COLUMN_TOL, f"{mode}: weighted column tendency {worst:.2e} of the cell tendencies"
