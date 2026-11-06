"""Validate the NumPy diffusion reference against op.diffdf.

Op.lhs_jac_tot and the vm / settling / settling_vm variants.

The central operator and its Jacobian assembly are compared on master's own
pre-loop state. The variants run on the port's pre-loop state with synthetic
vm/vs populated.
"""

from __future__ import annotations

import numpy as np
from _helpers import relerr
from oracle import assert_below, enter_master, leave_master, run_oracle_subprocess, upstream_preloop

# Bars: operator 1e-3 (He's net flux is a ~12-digit cancellation; worst
# cell ~3e-4 at the roundoff floor). diag 2.0 is a sign check only (a flipped
# sign gives exactly 2): the ~1e-4 diffusion residue is extracted from ~1e10
# LHS terms, and lhs_jac_tot disagrees with op.diffdf at heavy condensables.
# sup/sub 1e-10 (no cancellation).
OP_RTOL = 1e-3
DIAG_RTOL = 2.0
OFFDIAG_RTOL = 1e-10
FLOOR_FRAC = 1e-12  # variants: cancellation-residue floor, a fraction of the reference peak


# size-limit-ok: one oracle script, run in a subprocess on a copy of master
def main() -> None:
    # === Set up VULCAN-master state for reference ===
    enter_master()
    import build_atm as ba_v
    import chem_funs as cf_v
    import op as op_v
    import store as st_v
    import vulcan_cfg as cfg_v

    data_var, data_atm = upstream_preloop(
        cfg_v, st_v.Variables(), st_v.AtmData(), ba_v.Atm(), op_v.ReadRate(), ba_v.InitialAbun(), op_v.Output()
    )
    data_var.dt = 1e-10  # arbitrary; affects c0 in lhs_jac_tot

    y = np.asarray(data_var.y, dtype=np.float64).copy()
    nz, ni = y.shape

    # Reference diffusion contribution and full LHS Jacobian
    odes = op_v.ODESolver()
    diff_ref = np.asarray(odes.diffdf(y, data_atm), dtype=np.float64)
    lhs_ref = np.asarray(odes.lhs_jac_tot(data_var, data_atm), dtype=np.float64)

    # Reference chemistry Jacobian (no diffusion). VULCAN's convention:
    # lhs = c0*I - chem_J - diff_J, so this is already the negated contribution.
    chem_jac_ref = -np.asarray(cf_v.symjac(y, data_atm.M, data_var.k), dtype=np.float64)

    # === Switch to VULCAN-JAX modules ===
    leave_master()

    import diffusion_numpy_ref as diff_mod

    from vulcan3.config import default_config
    from vulcan3.state import RunState, legacy_view

    cfg_jax = default_config()
    # The variants' state: the port's typed pre-loop on the unmodified config,
    # as a legacy `(var, atm)` shim for master's op.ODESolver (it reads only
    # var.y and atm.*).
    var_p, atm_p, _ = legacy_view(RunState.with_pre_loop_setup(cfg_jax))
    # op.diffdf is the central scheme, so keep the upwind vm term off even if
    # the default changes; the variants below test the upwind kernel.
    cfg_jax.use_vm_mol = False
    cfg_jax.use_hybrid_vm_mol = False

    coeffs = diff_mod.build_diffusion_coeffs(y, data_atm, cfg_jax)
    diff_jax = diff_mod.apply_diffusion(y, coeffs)

    # === Compare RHS (operator) ===
    # Absolute floor so cancellation residues near zero (physically zero
    # diffusion at near-zero abundance) don't blow up the relative error.
    abs_tol = max(1e-12, 1e-12 * np.abs(diff_ref).max())
    scores = {"diffdf": (relerr(diff_jax, diff_ref, floor=abs_tol), OP_RTOL)}

    # === Compare Jacobian blocks ===
    diag_d, sup_d, sub_d = diff_mod.diffusion_block_diags(coeffs)

    # Isolate the diffusion blocks as (lhs_ref - c0*I - chem_jac_ref).
    r = 1.0 + 1.0 / np.sqrt(2.0)
    c0 = 1.0 / (r * data_var.dt)
    diff_jac_only = lhs_ref - chem_jac_ref
    np.fill_diagonal(diff_jac_only, np.diag(diff_jac_only) - c0)

    # diff_jac_only mapping:
    #   diag:  [j*ni+i, j*ni+i]     = -(A_eddy[j] + Ai[j, i])
    #   super: [j*ni+i, (j+1)*ni+i] = -(B_eddy[j] + Bi[j, i])
    #   sub:   [j*ni+i, (j-1)*ni+i] = -(C_eddy[j] + Ci[j, i])

    # Floor for jac comparisons: typical entries are ~1e-4; below that the
    # extraction from the c0 + chem cancellation is FP noise, not physics.
    # An entry passes if its abs diff is below the floor OR its rel diff is small.
    jac_abs_tol = 1e-4
    idx = np.arange(nz * ni).reshape(nz, ni)

    def _err(ref, val):
        d = np.abs(ref - val)
        return float(np.max(np.where(d < jac_abs_tol, 0.0, d / np.maximum(np.abs(ref), jac_abs_tol))))

    scores["jac diag block"] = (_err(diff_jac_only[idx, idx], -diag_d), DIAG_RTOL)
    scores["jac super block"] = (_err(diff_jac_only[idx[:-1], idx[1:]], -sup_d), OFFDIAG_RTOL)
    scores["jac sub block"] = (_err(diff_jac_only[idx[1:], idx[:-1]], -sub_d), OFFDIAG_RTOL)

    # === Variants (= op.diffdf_vm / diffdf_settling / diffdf_settling_vm) ===
    # Synthetic mixed-sign interface velocities (shape (nz-1, ni)) so the
    # sign-dependent upwind branches are actually exercised; setup leaves
    # atm.vm all-zero with use_vm_mol=False.
    yp = var_p.y
    nzp, nip = yp.shape
    rng = np.random.default_rng(0)
    atm_p.vs = (rng.standard_normal((nzp - 1, nip)) * 0.01).astype(np.float64)
    atm_p.vm = (rng.standard_normal((nzp - 1, nip)) * 50.0).astype(np.float64)

    # Upstream inconsistency: master's op.diffdf_settling_vm omits the vm
    # advective term at j=0 while op.diffdf_vm keeps it (vm_branch carries the
    # same quirk). VULCAN-JAX stays self-consistent at j=0 across all modes,
    # so settling_vm is compared on rows 1.. and the j=0 gap is pinned to
    # that exact term below.
    odes_p = op_v.ODESolver()
    for mode, upstream, label, rows, bound in (
        ("vm", odes_p.diffdf_vm, "diffdf_vm", slice(None), 1e-3),
        ("settling", odes_p.diffdf_settling, "diffdf_settling", slice(None), 1e-5),
        ("settling_vm", odes_p.diffdf_settling_vm, "diffdf_settling_vm (rows 1..)", slice(1, None), 1e-5),
    ):
        ref = np.asarray(upstream(yp, atm_p), dtype=np.float64)
        got = diff_mod.apply_diffusion(yp, diff_mod.build_diffusion_coeffs(yp, atm_p, cfg_jax, mode=mode))
        scores[label] = (relerr(got[rows], ref[rows], floor=FLOOR_FRAC * np.abs(ref).max()), bound)

    # The loop ended on settling_vm. Operator gap at j=0 = (vm upwind that JAX
    # keeps but master drops) applied to (y0, y1):
    # -[ (vm0>0)*vm0*y0 + (vm0<0)*vm0*y1 ] / dzi0.
    vm0 = np.asarray(atm_p.vm)[0]
    dzi0 = float(np.asarray(atm_p.dzi)[0])
    y0 = np.asarray(yp)[0]
    y1 = np.asarray(yp)[1]
    expected_j0_gap = -((vm0 > 0) * vm0 * y0 + (vm0 < 0) * vm0 * y1) / dzi0
    j0_gap = got[0] - ref[0]
    scores["diffdf_settling_vm j=0 gap == omitted vm term"] = (
        relerr(j0_gap, expected_j0_gap, floor=1e-10 * np.abs(expected_j0_gap).max()),
        1e-6,
    )

    assert_below(scores)


def test_main():
    run_oracle_subprocess(__file__, "vulcan2_ncho", "cfg_examples/vulcan_cfg_HD189.py")


if __name__ == "__main__":
    main()
