"""Convert between a `RunState` and the integrator's carry."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from . import atm_refresh as _atm_refresh_mod, phy_const as _phy_const, state as _state_mod
from ._carry import JaxIntegState, ProfileVars
from .chem_funs import _NETWORK
from .ini_abun import column_atoms
from .state import TERM_CONVERGED, TERM_NONFINITE, TERM_RUNNING, TERM_RUNTIME, TERM_STEP_COUNT


def initial_photo_carry(integ: Any, rs: Any) -> dict:
    """Build the initial photo carry from `rs.rate.k` and `rs.photo_runtime` (placeholders when photo is off)."""
    nz = int(rs.atm.Tco.shape[0])
    nr = _NETWORK.nr
    k_arr = jnp.asarray(rs.rate.k, dtype=jnp.float64)
    if k_arr.shape != (nr + 1, nz):
        raise ValueError(f"rs.rate.k shape {k_arr.shape} != expected ({nr + 1}, {nz})")
    if integ._photo_static is None:
        tiny = jnp.zeros((1, 1), dtype=jnp.float64)
        return {
            "k_arr": k_arr,
            "tau": tiny,
            "aflux": tiny,
            "sflux": tiny,
            "dflux_u": tiny,
            "aflux_change": jnp.float64(0.0),
        }
    pr = rs.photo_runtime
    return {
        "k_arr": k_arr,
        "tau": jnp.asarray(pr.tau, dtype=jnp.float64),
        "aflux": jnp.asarray(pr.aflux, dtype=jnp.float64),
        "sflux": jnp.asarray(pr.sflux, dtype=jnp.float64),
        "dflux_u": jnp.asarray(pr.dflux_u, dtype=jnp.float64),
        "aflux_change": jnp.float64(float(pr.aflux_change)),
    }


def initial_atm_carry(rs: Any) -> dict:
    """Build the initial atm-refresh carry from a RunState slice."""
    return {
        "g": jnp.asarray(rs.atm.g, dtype=jnp.float64),
        "mu": jnp.asarray(rs.atm.mu, dtype=jnp.float64),
        "Hp": jnp.asarray(rs.atm.Hp, dtype=jnp.float64),
        "dz": jnp.asarray(rs.atm.dz, dtype=jnp.float64),
        "zco": jnp.asarray(rs.atm.zco, dtype=jnp.float64),
        "dzi": jnp.asarray(rs.atm.dzi, dtype=jnp.float64),
        "Hpi": jnp.asarray(rs.atm.Hpi, dtype=jnp.float64),
        "top_flux": jnp.asarray(rs.atm.top_flux, dtype=jnp.float64),
        "vs": jnp.asarray(rs.atm.vs, dtype=jnp.float64),
    }


def initial_conv_carry(integ: Any, rs: Any) -> dict:
    """Build the initial conv-history carry from a RunState slice."""
    nz = int(rs.atm.Tco.shape[0])
    ni = _NETWORK.ni
    conv_step = int(integ._cfg.conv_step)
    ini_frq = int(integ._cfg.ini_update_photo_frq)
    return {
        "y_time_ring": jnp.zeros((conv_step, nz, ni), dtype=jnp.float64),
        "t_time_ring": jnp.zeros((conv_step,), dtype=jnp.float64),
        "longdy": jnp.float64(float(rs.step.longdy)),
        "longdydt": jnp.float64(float(rs.step.longdydt)),
        "where_varies_most": jnp.asarray(
            rs.params.where_varies_most,
            dtype=jnp.float64,
        ),
        "rtol": jnp.float64(float(integ._cfg.rtol)),
        "loss_criteria": jnp.float64(float(integ._cfg.loss_criteria)),
        "update_photo_frq": jnp.int32(ini_frq),
        "is_final_photo_frq": jnp.bool_(False),
        "geom_ok": jnp.bool_(False),
        "budget_ok": jnp.bool_(False),
        "budget_ref": column_atoms(
            jnp.asarray(rs.metadata.y_ini, dtype=jnp.float64),
            jnp.asarray(rs.atm.dz, dtype=jnp.float64),
            jnp.asarray(integ._compo_arr),
        ),
        "budget_err": jnp.zeros((integ._compo_arr.shape[1],), dtype=jnp.float64),
        "budget_drift": jnp.zeros((integ._compo_arr.shape[1],), dtype=jnp.float64),
    }


def profile_vars(integ: Any, rs: Any) -> ProfileVars:
    """Snapshot this profile's per-profile constants into a `ProfileVars`.

    Rebuilds the per-run static bundles for this runstate (so the values
    match what the single-profile closure would bake) and copies out the
    per-profile arrays/scalars that the batched runner reads from the carry
    instead of the closure. Host-side NumPy; cheap. Conden arrays fall back
    to consistently-shaped placeholders when condensation is off (the body
    never reads them in that case).
    """
    var, atm, _ = _state_mod.legacy_view(rs)
    statics = integ._build_statics(var, atm)
    refresh = integ._build_refresh_static(atm)
    nz = int(atm.Tco.shape[0])
    conden = integ._build_conden_static(var, atm, jnp.asarray(integ._gas_mask(atm)))
    if conden is not None:
        c_Dg = jnp.asarray(conden.Dg_per_re, dtype=jnp.float64)
        c_sat_n = jnp.asarray(conden.sat_n_per_re, dtype=jnp.float64)
        c_h2o_Dg = jnp.asarray(conden.h2o_Dg, dtype=jnp.float64)
        c_h2o_sat = jnp.asarray(conden.h2o_sat, dtype=jnp.float64)
        c_nh3_Dg = jnp.asarray(conden.nh3_Dg, dtype=jnp.float64)
        c_nh3_sat = jnp.asarray(conden.nh3_sat, dtype=jnp.float64)
        # Pin int32 explicitly: under x64, asarray(int) would default to
        # int64 and lanes packed at different times could mismatch dtypes.
        c_nh3_top = jnp.asarray(int(conden.nh3_conden_top), dtype=jnp.int32)
    else:
        zz = jnp.zeros((nz,), dtype=jnp.float64)
        c_Dg = jnp.zeros((1, nz), dtype=jnp.float64)
        c_sat_n = jnp.zeros((1, nz), dtype=jnp.float64)
        c_h2o_Dg = zz
        c_h2o_sat = zz
        c_nh3_Dg = zz
        c_nh3_sat = zz
        c_nh3_top = jnp.int32(0)
    if integ._cfg.use_photo:
        if rs.photo_static is None:
            raise ValueError(
                "use_photo is on but this RunState carries no photo_static "
                "(the T-P-dependent cross sections each lane needs). Build "
                "it with RunState.with_pre_loop_setup(cfg), or attach one "
                "with rs._replace(photo_static=...)."
            )
        # The two T-P-dependent photo statics, as _build_photo_static
        # would bake them for this profile.
        p_absp_T_cross = jnp.asarray(rs.photo_static.absp_T_cross, dtype=jnp.float64)
        p_cross_J_T = jnp.asarray(rs.photo_static.cross_J_T, dtype=jnp.float64)
    else:
        p_absp_T_cross = jnp.zeros((0, 1, 1), dtype=jnp.float64)
        p_cross_J_T = jnp.zeros((0, 1, 1), dtype=jnp.float64)
    return ProfileVars(
        n_0=jnp.asarray(statics.n_0, dtype=jnp.float64),
        Kzz=jnp.asarray(statics.Kzz, dtype=jnp.float64),
        atom_ini=jnp.asarray(statics.atom_ini_arr, dtype=jnp.float64),
        y_ini=jnp.asarray(rs.metadata.y_ini, dtype=jnp.float64),
        fix_species_sat_mix=jnp.asarray(statics.fix_species_sat_mix, dtype=jnp.float64),
        r_Tco=jnp.asarray(refresh.Tco, dtype=jnp.float64),
        r_pico=jnp.asarray(refresh.pico, dtype=jnp.float64),
        r_Dzz_top=jnp.asarray(refresh.Dzz_top, dtype=jnp.float64),
        r_gs=jnp.float64(refresh.gs),
        r_zco_pref=jnp.float64(refresh.zco_pref),
        r_Rp=jnp.float64(refresh.Rp),
        c_Dg_per_re=c_Dg,
        c_sat_n_per_re=c_sat_n,
        c_h2o_Dg=c_h2o_Dg,
        c_h2o_sat=c_h2o_sat,
        c_nh3_Dg=c_nh3_Dg,
        c_nh3_sat=c_nh3_sat,
        c_nh3_conden_top=c_nh3_top,
        p_absp_T_cross=p_absp_T_cross,
        p_cross_J_T=p_cross_J_T,
    )


# size-limit-ok: one carry field per line
def pack_state(integ: Any, rs: Any) -> JaxIntegState:
    """Build the initial JaxIntegState from a fully-populated RunState.

    Static metadata (atom ordering, fix-species mapping) lives on the
    `OuterLoop` instance.
    """
    photo_fields = integ._initial_photo_carry_from_runstate(rs)
    atm_fields = integ._initial_atm_carry_from_runstate(rs)
    conv_fields = integ._initial_conv_carry_from_runstate(rs)
    nz = int(rs.atm.Tco.shape[0])
    ni = _NETWORK.ni
    n_fix = int(integ._statics.fix_species_idx.shape[0])
    fix_started = bool(rs.params.fix_species_start)
    fix_y_init = np.zeros((nz, ni), dtype=np.float64)
    fix_mask_init = np.zeros((nz, ni), dtype=bool)
    fix_pfix_idx_init = np.zeros((n_fix,), dtype=np.int32)
    if integ._statics.use_fix_species and rs.fix_species is not None and len(rs.fix_species.fix_species) > 0:
        fix_y_arr = np.asarray(rs.fix_species.fix_y, dtype=np.float64)
        coldtrap = np.asarray(rs.fix_species.conden_min_lev, dtype=np.int32)
        for i, sp in enumerate(rs.fix_species.fix_species):
            sp_idx = _NETWORK.species_idx[sp]
            fix_y_init[:, sp_idx] = fix_y_arr[i]
            if fix_started:
                if integ._statics.fix_species_from_coldtrap_lev:
                    pfix = int(coldtrap[i])
                    fix_mask_init[:pfix, sp_idx] = True
                    fix_pfix_idx_init[i] = pfix
                else:
                    fix_mask_init[:, sp_idx] = True
                    fix_pfix_idx_init[i] = nz
    return JaxIntegState(
        y=jnp.asarray(rs.step.y, dtype=jnp.float64),
        y_prev=jnp.asarray(rs.step.y_prev, dtype=jnp.float64),
        ymix=jnp.asarray(rs.step.ymix, dtype=jnp.float64),
        dt=jnp.asarray(float(rs.step.dt), dtype=jnp.float64),
        t=jnp.asarray(float(rs.step.t), dtype=jnp.float64),
        delta=jnp.asarray(float(rs.params.delta), dtype=jnp.float64),
        accept_count=jnp.int32(int(rs.params.count)),
        retry_count=jnp.int32(0),
        atom_loss=jnp.asarray(rs.atoms.atom_loss, dtype=jnp.float64),
        atom_loss_prev=jnp.asarray(rs.atoms.atom_loss_prev, dtype=jnp.float64),
        nega_count=jnp.int32(int(rs.params.nega_count)),
        loss_count=jnp.int32(int(rs.params.loss_count)),
        delta_count=jnp.int32(int(rs.params.delta_count)),
        small_y=jnp.float64(float(rs.params.small_y)),
        nega_y=jnp.float64(float(rs.params.nega_y)),
        **photo_fields,
        **atm_fields,
        **conv_fields,
        fix_species_started=jnp.bool_(fix_started),
        fix_y=jnp.asarray(fix_y_init, dtype=jnp.float64),
        fix_mask=jnp.asarray(fix_mask_init, dtype=jnp.bool_),
        fix_pfix_idx=jnp.asarray(fix_pfix_idx_init, dtype=jnp.int32),
        # Hycean: seed pinned=False, mix=[0, 0]; the body snapshots the
        # live ymix when (use_fix_H2He=True) & (~pinned) & (t > hycean_pin_time).
        h2he_pinned=jnp.bool_(False),
        h2he_mix=jnp.zeros((2,), dtype=jnp.float64),
        # save_evolution buffers. Allocated to `save_evo_n_max`
        # (ceil(count_max / save_evo_frq) + 1) when on; length-1
        # placeholder when off.
        y_evo=jnp.zeros(
            (int(integ._statics.save_evo_n_max), nz, ni),
            dtype=jnp.float64,
        ),
        t_evo=jnp.zeros(
            (int(integ._statics.save_evo_n_max),),
            dtype=jnp.float64,
        ),
        evo_idx=jnp.int32(0),
        # Batched-runner flags; the single-profile loop never reads them (its
        # runner sets `termination_reason` once at exit).
        is_done=jnp.bool_(False),
        termination_reason=jnp.int32(TERM_RUNNING),
        # Phase seed: upwind (1.0) when use_vm_mol, else central (0.0);
        # only hybrid runs ever flip it.
        hybrid_use_vm=jnp.float64(1.0 if bool(integ._statics.use_vm_mol) else 0.0),
        # Live termination budget, seeded to the static caps. Only the
        # hybrid phase flip mutates these (extends phase 1's allowance).
        count_min_dyn=jnp.int32(int(integ._statics.count_min)),
        count_max_dyn=jnp.int32(int(integ._statics.count_max)),
        runtime_dyn=jnp.float64(float(integ._statics.runtime)),
        # Per-profile constants must ride the carry (vmap does not batch
        # closures); value-identical to the closures single-profile.
        pv=integ._profile_vars_from_runstate(rs),
    )


# size-limit-ok: one RunState field per line
def unpack_state(integ: Any, state: JaxIntegState, rs_entry: Any) -> Any:
    """Build a fresh `RunState` from the runner's final JaxIntegState.

    The static atm fields (pco, Tco, Kzz, n_0, ms, alpha, ...) are
    preserved verbatim from `rs_entry.atm`; only the dynamic refresh
    slots (g, mu, Hp, dz, dzi, zco, zmco, Hpi, top_flux, vs, and vm
    under use_vm_mol) come from the carry. Step / params / atoms /
    photo_runtime / fix_species are rebuilt from the carry, in the
    OuterLoop's atom order and cfg `fix_species` order.
    """
    g = jnp.asarray(state.g, dtype=jnp.float64)
    zco = jnp.asarray(state.zco, dtype=jnp.float64)
    dzi = jnp.asarray(state.dzi, dtype=jnp.float64)
    Hpi = jnp.asarray(state.Hpi, dtype=jnp.float64)
    atm_out = rs_entry.atm._replace(
        g=g,
        mu=jnp.asarray(state.mu, dtype=jnp.float64),
        Hp=jnp.asarray(state.Hp, dtype=jnp.float64),
        dz=jnp.asarray(state.dz, dtype=jnp.float64),
        zco=zco,
        # op.py:972-973: cell-centre heights follow the refreshed zco.
        zmco=0.5 * (zco[:-1] + zco[1:]),
        dzi=dzi,
        Hpi=Hpi,
        top_flux=jnp.asarray(state.top_flux, dtype=jnp.float64),
        vs=jnp.asarray(state.vs, dtype=jnp.float64),
    )
    if integ._statics.use_vm_mol:
        # vm_branch op.py:945-992 refreshes vm inside update_mu_dz. The
        # runner recomputes it per step from the carry and never stores
        # it, so rebuild the terminal value here (same inputs as in-loop).
        atm_out = atm_out._replace(
            vm=_atm_refresh_mod.recompute_vm_jax(
                g,
                Hpi,
                dzi,
                jnp.asarray(rs_entry.atm.Dzz, dtype=jnp.float64),
                jnp.asarray(rs_entry.atm.ms, dtype=jnp.float64),
                jnp.asarray(rs_entry.atm.alpha, dtype=jnp.float64),
                jnp.asarray(rs_entry.atm.Tco, dtype=jnp.float64),
                float(_phy_const.kb),
                float(_phy_const.Navo),
            )
        )

    rate_out = rs_entry.rate._replace(
        k=jnp.asarray(state.k_arr, dtype=jnp.float64),
    )

    photo_out = rs_entry.photo  # not mutated by the runner

    # Slice the populated prefix of the save_evolution ring; when
    # save_evolution=False, evo_idx stays 0 and the slice is empty.
    evo_n = int(state.evo_idx)
    y_evo_out = jnp.asarray(state.y_evo[:evo_n], dtype=jnp.float64)
    t_evo_out = jnp.asarray(state.t_evo[:evo_n], dtype=jnp.float64)

    step_out = _state_mod.StepInputs(
        y=jnp.asarray(state.y, dtype=jnp.float64),
        y_prev=jnp.asarray(state.y_prev, dtype=jnp.float64),
        ymix=jnp.asarray(state.ymix, dtype=jnp.float64),
        t=float(state.t),
        dt=float(state.dt),
        longdy=float(state.longdy),
        longdydt=float(state.longdydt),
        y_evo=y_evo_out,
        t_evo=t_evo_out,
    )

    params_out = _state_mod.ParamInputs(
        count=int(state.accept_count),
        nega_count=int(state.nega_count),
        loss_count=int(state.loss_count),
        delta_count=int(state.delta_count),
        delta=float(state.delta),
        small_y=float(state.small_y),
        nega_y=float(state.nega_y),
        end_case=integ._classify_end_case(state),
        switch_final_photo_frq=bool(state.is_final_photo_frq),
        pic_count=int(rs_entry.params.pic_count),
        where_varies_most=jnp.asarray(
            state.where_varies_most,
            dtype=jnp.float64,
        ),
        fix_species_start=bool(state.fix_species_started),
        termination_reason=int(state.termination_reason),
    )

    # This run's atom_ini rides the carry; `integ._statics` holds the
    # profile the runner was first built for.
    atom_ini_arr = np.asarray(state.pv.atom_ini, dtype=np.float64)
    atom_loss_arr = np.asarray(state.atom_loss, dtype=np.float64)
    atom_sum_arr = (atom_loss_arr + 1.0) * atom_ini_arr
    atoms_out = _state_mod.AtomInputs(
        atom_order=tuple(integ._atom_order),
        atom_ini=jnp.asarray(atom_ini_arr),
        atom_loss=jnp.asarray(atom_loss_arr),
        atom_loss_prev=jnp.asarray(state.atom_loss_prev, dtype=jnp.float64),
        atom_sum=jnp.asarray(atom_sum_arr),
    )

    if integ._photo_static is not None:
        photo_runtime_out = _state_mod.PhotoRuntimeInputs(
            tau=jnp.asarray(state.tau, dtype=jnp.float64),
            aflux=jnp.asarray(state.aflux, dtype=jnp.float64),
            sflux=jnp.asarray(state.sflux, dtype=jnp.float64),
            dflux_u=jnp.asarray(state.dflux_u, dtype=jnp.float64),
            aflux_change=float(state.aflux_change),
        )
    else:
        photo_runtime_out = None

    nz = int(rs_entry.atm.Tco.shape[0])
    fix_species_cfg = list(integ._cfg.fix_species or [])
    if fix_species_cfg:
        fix_y_full = np.asarray(state.fix_y, dtype=np.float64)
        fix_y_per_sp = np.zeros((len(fix_species_cfg), nz), dtype=np.float64)
        for i, sp in enumerate(fix_species_cfg):
            fix_y_per_sp[i] = fix_y_full[:, _NETWORK.species_idx[sp]]
        if integ._statics.fix_species_from_coldtrap_lev:
            pfix_np = np.asarray(state.fix_pfix_idx, dtype=np.int32)
        else:
            pfix_np = np.zeros((len(fix_species_cfg),), dtype=np.int32)
        fix_species_out = _state_mod.FixSpeciesInputs(
            fix_species=tuple(fix_species_cfg),
            fix_y=jnp.asarray(fix_y_per_sp),
            conden_min_lev=jnp.asarray(pfix_np),
        )
    else:
        fix_species_out = _state_mod.FixSpeciesInputs(
            fix_species=(),
            fix_y=jnp.zeros((0, nz), dtype=jnp.float64),
            conden_min_lev=jnp.zeros((0,), dtype=jnp.int32),
        )

    return _state_mod.RunState(
        atm=atm_out,
        rate=rate_out,
        photo=photo_out,
        step=step_out,
        params=params_out,
        atoms=atoms_out,
        photo_runtime=photo_runtime_out,
        fix_species=fix_species_out,
        metadata=rs_entry.metadata,
        photo_static=rs_entry.photo_static,
    )


def classify_end_case(state: JaxIntegState) -> Any:
    """Classify end-of-run (op.py:1069-1085) from the in-loop reason.

    `_real_terminate` applies master's priority (converged over runtime
    over step-count) against the live budget, so a step that converges
    while hitting a cap reads end_case=1, and so does a hybrid phase-1
    convergence past the static count_max.

    end_case=5 is a VULCAN3 addition with no upstream counterpart: the
    run stopped without meeting the convergence criterion and without
    hitting either cap -- a run stopped on a non-finite state
    (termination_reason 5) or a "converged" state that is not finite.
    """
    reason = int(state.termination_reason)
    if reason in (TERM_RUNTIME, TERM_STEP_COUNT):
        return reason
    if reason == TERM_CONVERGED and bool(jnp.all(jnp.isfinite(state.y))):
        return TERM_CONVERGED
    return TERM_NONFINITE
