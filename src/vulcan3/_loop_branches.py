"""The step's photolysis and condensation branches, and the atmosphere splices they read."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from . import atm_refresh as _atm_refresh_mod, conden as _conden_mod, photo as _photo_mod
from .phy_const import UNDERFLOW_DENOM

if TYPE_CHECKING:
    from collections.abc import Callable

    from ._carry import JaxIntegState, ProfileVars, _PhotoStatic
    from .jax_step import AtmStatic


def _make_photo_branch(photo_static: _PhotoStatic) -> Callable[[JaxIntegState], JaxIntegState]:
    """Build the in-loop photo branch closure.

    Computes tau / aflux / J inside the JIT'd body and writes the
    photolysis rows of `k_arr` directly via `.at[].set()`. Photo state
    stays on device between calls.
    """
    photo_data = photo_static.photo_data
    cross_J = photo_static.cross_J
    branch_re_idx = photo_static.branch_re_idx
    branch_active = photo_static.branch_active
    branch_T_re_idx = photo_static.branch_T_re_idx
    branch_T_active = photo_static.branch_T_active
    cross_Jion = photo_static.cross_Jion
    ion_branch_re_idx = photo_static.ion_branch_re_idx
    ion_branch_active = photo_static.ion_branch_active
    bins = photo_static.bins
    sflux_top = photo_static.sflux_top
    din12_indx = photo_static.din12_indx
    dbin1 = photo_static.dbin1
    dbin2 = photo_static.dbin2
    mu_zenith = photo_static.mu_zenith
    edd = photo_static.edd
    ag0 = photo_static.ag0
    hc = photo_static.hc
    f_diurnal = photo_static.f_diurnal
    flux_atol = photo_static.flux_atol
    ag0_is_zero = photo_static.ag0_is_zero

    def photo_branch(s: JaxIntegState) -> JaxIntegState:
        # This lane's T-dependent cross sections from the carry (see ProfileVars).
        pd = photo_data._replace(absp_T_cross=s.pv.p_absp_T_cross)

        # Optical depth (mirrors op.compute_tau via op_jax.Ros2JAX.compute_tau).
        tau_new = _photo_mod.compute_tau_jax(s.y, s.dz, pd)

        # Two-stream RT. `s.dflux_u` is the prior call's value (matches
        # op.py:2692, which reads dflux_u before the up-sweep at op.py:2693-2694).
        aflux_new, sflux_new, _, dflux_u_new = _photo_mod.compute_flux_jax(
            tau_new,
            sflux_top,
            s.ymix,
            pd,
            bins,
            mu_zenith,
            edd,
            ag0,
            hc,
            s.dflux_u,
            ag0_is_zero=ag0_is_zero,
        )

        # Per-branch J-rates (flat output; no Python dict).
        J_br_new, J_br_T_new = _photo_mod.compute_J_jax_flat(
            aflux_new, cross_J, s.pv.p_cross_J_T, din12_indx, dbin1, dbin2
        )

        # Write into k_arr: k_arr[re_idx] = J * f_diurnal for each active branch.
        k_arr_new = _photo_mod.update_k_with_J(
            s.k_arr,
            J_br_new,
            J_br_T_new,
            branch_re_idx,
            branch_active,
            branch_T_re_idx,
            branch_T_active,
            f_diurnal,
        )

        if cross_Jion.shape[0] > 0:
            Jion_br_new = _photo_mod.compute_J_jax_flat(
                aflux_new,
                cross_Jion,
                jnp.zeros((0, aflux_new.shape[0], cross_Jion.shape[1]), dtype=aflux_new.dtype),
                din12_indx,
                dbin1,
                dbin2,
            )[0]
            k_arr_new = _photo_mod.update_k_with_J(
                k_arr_new,
                Jion_br_new,
                jnp.zeros((0, aflux_new.shape[0]), dtype=aflux_new.dtype),
                ion_branch_re_idx,
                ion_branch_active,
                jnp.zeros((0,), dtype=jnp.int64),
                jnp.zeros((0,), dtype=jnp.bool_),
                f_diurnal,
            )

        # aflux_change: mirrors op.py:2737 / op_jax.py:79-88. `s.aflux` here
        # is the old aflux, master's `prev_aflux` in the ratio.
        mask = aflux_new > flux_atol
        diff = jnp.abs(aflux_new - s.aflux)
        ratio = jnp.where(mask, diff / jnp.maximum(jnp.abs(aflux_new), UNDERFLOW_DENOM), 0.0)
        aflux_change_new = jnp.where(jnp.any(mask), jnp.max(ratio), jnp.float64(0.0))

        return s._replace(
            k_arr=k_arr_new,
            tau=tau_new,
            aflux=aflux_new,
            sflux=sflux_new,
            dflux_u=dflux_u_new,
            aflux_change=aflux_change_new,
        )

    return photo_branch


def _splice_conden(conden_static: _conden_mod.CondenStatic, pv: ProfileVars) -> _conden_mod.CondenStatic:
    """Return the baked `CondenStatic` with this lane's per-profile arrays (see ProfileVars).

    Its other fields stay baked and must be batch-constant. nh3_conden_top is a 0-d int32 only compared against
    jnp.arange.
    """
    return conden_static._replace(
        Dg_per_re=pv.c_Dg_per_re,
        sat_n_per_re=pv.c_sat_n_per_re,
        h2o_Dg=pv.c_h2o_Dg,
        h2o_sat=pv.c_h2o_sat,
        nh3_Dg=pv.c_nh3_Dg,
        nh3_sat=pv.c_nh3_sat,
        nh3_conden_top=pv.c_nh3_conden_top,
        n_0=pv.n_0,
    )


def _splice_step_atm(atm_static: AtmStatic, s: JaxIntegState, use_vm_mol: bool, kb: float, Navo: float) -> AtmStatic:
    """Return the step's `AtmStatic`: the carry's refreshed geometry (op.py:905-907) and hybrid vm_mol phase.

    Under `use_vm_mol`, also the vm recomputed from them.

    `jax_ros2_step` reads use_vm_mol as a float multiplier; non-hybrid runs
    never flip it, so it equals `atm_static.use_vm_mol` bit-for-bit. vm depends
    on mu (via Hpi) and g, so it is refreshed with the geometry
    (vm_branch@84d010d op.py:992, update_mu_dz); its inputs change only at
    refresh cadence, so a per-step recompute keeps upstream's cadence.
    """
    atm = atm_static._replace(g=s.g, dzi=s.dzi, Hpi=s.Hpi, top_flux=s.top_flux, vs=s.vs, use_vm_mol=s.hybrid_use_vm)
    if use_vm_mol:
        atm = atm._replace(
            vm=_atm_refresh_mod.recompute_vm_jax(
                s.g, s.Hpi, s.dzi, atm_static.Dzz, atm_static.ms, atm_static.alpha, atm_static.Tco, kb, Navo
            )
        )
    return atm


def _make_conden_branch(conden_static: _conden_mod.CondenStatic) -> Callable[[JaxIntegState], JaxIntegState]:
    """Build the in-loop conden closure.

    Operates on the post-Ros2/post-clip s.y/s.ymix. Updates the conden
    rows of `s.k_arr`, then redistributes mass via the H2O/NH3 relax
    kernels (each gated by a Python bool on conden_static).
    """

    def conden_branch(s: JaxIntegState) -> JaxIntegState:
        st = _splice_conden(conden_static, s.pv)
        k_arr_new = _conden_mod.update_conden_rates(s.k_arr, s.y, st)
        y_new, ymix_new = _conden_mod.apply_h2o_relax_jax(s.y, s.ymix, s.dt, st)
        y_new, ymix_new = _conden_mod.apply_nh3_relax_jax(y_new, ymix_new, s.dt, st)
        return s._replace(y=y_new, ymix=ymix_new, k_arr=k_arr_new)

    return conden_branch
