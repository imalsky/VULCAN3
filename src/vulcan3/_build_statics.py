"""Pack a run's setup into the runner's closure inputs: statics, photolysis, refresh and condensation tables."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from . import atm_refresh as _atm_refresh_mod, conden as _conden_mod, photo as _photo_mod, phy_const as _phy_const
from ._carry import _PhotoStatic, _Statics
from .chem_funs import NETWORK

if TYPE_CHECKING:
    import types

    from .outer_loop import OuterLoop

# Condensates pinned over the whole column once fix_species fires: master's list (exoclime@80f75b9 op.py:878).
_WHOLE_COLUMN_CONDENSATES = {"H2O_l_s", "H2SO4_l", "NH3_l_s", "S8_l_s"}


# size-limit-ok: packs one long NamedTuple, one field per line
def build_statics(integ: OuterLoop, var: types.SimpleNamespace, atm: types.SimpleNamespace) -> _Statics:
    """Pack scalar config + per-run arrays into the closure inputs."""
    ni = NETWORK.ni
    atom_ini_arr = np.asarray(
        [float(var.atom_ini[a]) for a in integ._atom_order],
        dtype=np.float64,
    )

    # plot_evo reads the same snapshots; only save_evolution writes them
    # to the .vul.
    capture_evo = bool(integ._cfg.save_evolution or integ._cfg.use_plot_evo)

    # conver_ignore species are zeroed in the longdy reduction (op.py:1045-1046).
    conver_ignore_np = np.zeros(ni, dtype=bool)
    for sp in integ._cfg.conver_ignore:
        if sp in NETWORK.species_idx:
            conver_ignore_np[NETWORK.species_idx[sp]] = True

    # condense_zero_conv (op.py:1048-1049): when use_condense, the
    # non_gas_sp columns are zeroed in longdy.
    nz = atm.Tco.shape[0]
    cond_zero_conv_np = np.zeros((nz, ni), dtype=bool)
    if integ._cfg.use_condense:
        for sp in integ._cfg.non_gas_sp:
            cond_zero_conv_np[:, NETWORK.species_idx[sp]] = True

    # The ion masks are zeros when use_ion is off; the body's Python
    # `if statics.use_ion:` skips them at trace time.
    use_ion = bool(integ._cfg.use_ion)
    if use_ion:
        charge_list = list(var.charge_list)
        charge_np = np.zeros(ni, dtype=np.float64)
        for sp in charge_list:
            charge_np[NETWORK.species_idx[sp]] = integ._compo_charge[NETWORK.species_idx[sp]]
        e_idx = NETWORK.species_idx["e"]
        # Exclude 'e' from the charge column so `e[:] = -dot(y, charge_arr)`
        # is consistent (op.py:3001 zeros e first).
        charge_np[e_idx] = 0.0
    else:
        charge_np = np.zeros(ni, dtype=np.float64)
        e_idx = 0

    fix_sp_bot_cfg = dict(integ._cfg.use_fix_sp_bot)
    # use_sat_surfaceH2O pins bottom H2O at the surface saturation that
    # setup computed (build_atm.py:227-228; applied every step, op.py:2945).
    if integ._cfg.use_condense and integ._cfg.use_sat_surfaceH2O and "H2O" in integ._cfg.condense_sp:
        fix_sp_bot_cfg["H2O"] = float(np.asarray(atm.sat_mix["H2O"])[0])
    use_fix_sp_bot = bool(fix_sp_bot_cfg)
    if use_fix_sp_bot:
        fix_sp_bot_idx = np.asarray(
            [NETWORK.species_idx[sp] for sp in fix_sp_bot_cfg],
            dtype=np.int32,
        )
        fix_sp_bot_mix = np.asarray(
            [float(fix_sp_bot_cfg[sp]) for sp in fix_sp_bot_cfg],
            dtype=np.float64,
        )
    else:
        fix_sp_bot_idx = np.zeros((0,), dtype=np.int32)
        fix_sp_bot_mix = np.zeros((0,), dtype=np.float64)

    use_fix_H2He = bool(integ._cfg.use_fix_H2He)
    if use_fix_H2He:
        h2_idx = int(NETWORK.species_idx["H2"])
        he_idx = int(NETWORK.species_idx["He"])
    else:
        h2_idx = -1
        he_idx = -1

    fix_species_cfg = list(integ._cfg.fix_species)
    use_fix_species = bool(integ._cfg.use_condense and fix_species_cfg)
    if use_fix_species:
        fix_species_idx = np.asarray(
            [NETWORK.species_idx[sp] for sp in fix_species_cfg],
            dtype=np.int32,
        )
        fix_species_wholecol = np.asarray(
            [sp in _WHOLE_COLUMN_CONDENSATES for sp in fix_species_cfg],
            dtype=bool,
        )
        fix_species_sat_mix = np.zeros((len(fix_species_cfg), nz), dtype=np.float64)
        for i, sp in enumerate(fix_species_cfg):
            if sp in atm.sat_mix:
                fix_species_sat_mix[i] = np.asarray(atm.sat_mix[sp], dtype=np.float64)
    else:
        fix_species_idx = np.zeros((0,), dtype=np.int32)
        fix_species_sat_mix = np.zeros((0, nz), dtype=np.float64)
        fix_species_wholecol = np.zeros((0,), dtype=bool)

    return _Statics(
        compo_arr=jnp.asarray(integ._compo_arr),
        atom_ini_arr=jnp.asarray(atom_ini_arr),
        loss_eps=float(integ._cfg.loss_eps),
        pos_cut=float(integ._cfg.pos_cut),
        nega_cut=float(integ._cfg.nega_cut),
        mtol=float(integ._cfg.mtol),
        atol=float(integ._cfg.atol),
        dt_var_min=float(integ._cfg.dt_var_min),
        dt_var_max=float(integ._cfg.dt_var_max),
        dt_min=float(integ._cfg.dt_min),
        dt_max=float(integ._cfg.dt_max),
        batch_max_retries=int(integ._cfg.batch_max_retries),
        conv_step=int(integ._cfg.conv_step),
        count_min=int(integ._cfg.count_min),
        count_max=int(integ._cfg.count_max),
        runtime=float(integ._cfg.runtime),
        trun_min=float(integ._cfg.trun_min),
        st_factor=float(integ._cfg.st_factor),
        yconv_cri=float(integ._cfg.yconv_cri),
        yconv_min=float(integ._cfg.yconv_min),
        slope_cri=float(integ._cfg.slope_cri),
        slope_min_hp_frac=float(integ._cfg.slope_min_hp_frac),
        slope_min_cap=float(integ._cfg.slope_min_cap),
        slope_min_floor=float(integ._cfg.slope_min_floor),
        flux_cri=float(integ._cfg.flux_cri),
        geom_conv_tol=float(integ._cfg.geom_conv_tol),
        element_budget_tol=float(integ._cfg.element_budget_tol),
        budget_ref_atom=integ._atom_order.index("H"),
        mtol_conv=float(integ._cfg.mtol_conv),
        conver_ignore_mask=jnp.asarray(conver_ignore_np),
        condense_zero_conv_mask=jnp.asarray(cond_zero_conv_np),
        n_0=jnp.asarray(atm.n_0, dtype=jnp.float64),
        Kzz=jnp.asarray(atm.Kzz, dtype=jnp.float64),
        use_vm_mol=bool(integ._cfg.use_vm_mol and integ._cfg.use_moldiff),
        hybrid_vm_mol=bool(integ._cfg.use_vm_mol and integ._cfg.use_hybrid_vm_mol and integ._cfg.use_moldiff),
        final_update_photo_frq=int(integ._cfg.final_update_photo_frq),
        update_frq=int(integ._cfg.update_frq),
        use_adapt_rtol=bool(integ._cfg.use_adapt_rtol),
        rtol_accept=float(integ._cfg.rtol),
        rtol_min=float(integ._cfg.rtol_min),
        rtol_max=float(integ._cfg.rtol_max),
        adapt_rtol_dec_period=int(integ._cfg.adapt_rtol_dec_period),
        adapt_rtol_inc_period=int(integ._cfg.adapt_rtol_inc_period),
        adapt_rtol_dec=float(integ._cfg.adapt_rtol_dec),
        adapt_rtol_inc=float(integ._cfg.adapt_rtol_inc),
        adapt_rtol_loss_mul=float(integ._cfg.adapt_rtol_loss_mul),
        adapt_rtol_inc_loss_thresh=float(integ._cfg.adapt_rtol_inc_loss_thresh),
        photo_switch_longdy_thresh=float(integ._cfg.photo_switch_longdy_thresh),
        photo_switch_longdydt_thresh=float(integ._cfg.photo_switch_longdydt_thresh),
        hycean_pin_time=float(integ._cfg.hycean_pin_time),
        step_size_safety=float(integ._cfg.step_size_safety),
        step_size_zero_delta_frac=float(integ._cfg.step_size_zero_delta_frac),
        use_ion=use_ion,
        e_idx=int(e_idx),
        charge_arr=jnp.asarray(charge_np),
        use_fix_sp_bot=use_fix_sp_bot,
        fix_sp_bot_idx=jnp.asarray(fix_sp_bot_idx),
        fix_sp_bot_mix=jnp.asarray(fix_sp_bot_mix),
        use_fix_H2He=use_fix_H2He,
        h2_idx=h2_idx,
        he_idx=he_idx,
        use_fix_species=use_fix_species,
        post_conden_rtol=float(integ._cfg.post_conden_rtol),
        fix_species_from_coldtrap_lev=bool(integ._cfg.fix_species_from_coldtrap_lev),
        fix_species_idx=jnp.asarray(fix_species_idx),
        fix_species_sat_mix=jnp.asarray(fix_species_sat_mix),
        fix_species_wholecol=jnp.asarray(fix_species_wholecol),
        save_evolution=capture_evo,
        save_evo_frq=int(integ._cfg.save_evo_frq),
        save_evo_n_max=(
            int(np.ceil(int(integ._cfg.count_max) / max(int(integ._cfg.save_evo_frq), 1))) + 1 if capture_evo else 1
        ),
    )


def gas_mask(integ: OuterLoop, atm: types.SimpleNamespace) -> np.ndarray:
    """(ni,) gas-species mask.

    `atm.gas_indx` when `non_gas_sp` is set, else all True.
    """
    if not integ._cfg.non_gas_sp:
        return np.ones(NETWORK.ni, dtype=bool)
    mask = np.zeros(NETWORK.ni, dtype=bool)
    mask[np.asarray(atm.gas_indx, dtype=int)] = True
    return mask


def build_photo_static(integ: OuterLoop, var: types.SimpleNamespace, atm: types.SimpleNamespace) -> _PhotoStatic | None:
    """Pack photo cross sections + scalar configs into a `_PhotoStatic`.

    Returns None if `use_photo=False` (the runner skips the photo branch
    entirely in that case). Reuses the photo data caches on the
    odesolver when set; otherwise builds them from its `_photo_static`
    and caches them there.
    """
    if not integ._cfg.use_photo:
        return None

    # Derive everything from the PhotoStaticInputs pytree; lazily build it
    # via Ros2JAX's own builder for unwired test sites.
    odesolver = integ.odesolver
    photo_static = odesolver._photo_static
    if photo_static is None:
        photo_static = odesolver._ensure_photo_static(var, atm)

    photo_data = odesolver._photo_data
    if photo_data is None:
        photo_data = _photo_mod.photo_data_from_static(photo_static, list(NETWORK.species))
        odesolver._photo_data = photo_data
    photo_J_data = odesolver._photo_J_data
    if photo_J_data is None:
        photo_J_data = _photo_mod.photo_J_data_from_static(photo_static)
        odesolver._photo_J_data = photo_J_data
    photo_ion_data = odesolver._photo_ion_data
    if integ._cfg.use_ion:
        if photo_ion_data is None:
            photo_ion_data = _photo_mod.photo_ion_data_from_static(photo_static)
            odesolver._photo_ion_data = photo_ion_data
    else:
        photo_ion_data = None

    remove_list = integ._cfg.remove_list
    branch_re_idx, branch_active = _photo_mod.pack_branch_to_k_index_map(
        photo_J_data.branch_keys, var.pho_rate_index, remove_list
    )
    branch_T_re_idx, branch_T_active = _photo_mod.pack_branch_to_k_index_map(
        photo_J_data.branch_T_keys, var.pho_rate_index, remove_list
    )
    if photo_ion_data is not None:
        ion_branch_re_idx, ion_branch_active = _photo_mod.pack_branch_to_k_index_map(
            photo_ion_data.branch_keys, var.ion_rate_index, remove_list
        )
        cross_Jion = photo_ion_data.cross_J
    else:
        ion_branch_re_idx = jnp.zeros((0,), dtype=jnp.int64)
        ion_branch_active = jnp.zeros((0,), dtype=jnp.bool_)
        cross_Jion = jnp.zeros((0, int(photo_static.nbin)), dtype=jnp.float64)

    bins_arr = jnp.asarray(photo_static.bins, dtype=jnp.float64)
    din12_indx = int(photo_static.din12_indx)
    dbin1 = float(photo_static.dbin1)
    dbin2 = float(photo_static.dbin2)

    ag0 = float(_phy_const.ag0)
    # Record the baked star's TOA flux here, so `prepare_runstate` also
    # rejects a later different-star batch on a pre-built runner.
    if integ._sflux_top_ref is None:
        integ._sflux_top_ref = np.asarray(var.sflux_top, dtype=np.float64)
    return _PhotoStatic(
        photo_data=photo_data,
        cross_J=photo_J_data.cross_J,
        branch_re_idx=branch_re_idx,
        branch_active=branch_active,
        branch_T_re_idx=branch_T_re_idx,
        branch_T_active=branch_T_active,
        cross_Jion=cross_Jion,
        ion_branch_re_idx=ion_branch_re_idx,
        ion_branch_active=ion_branch_active,
        bins=bins_arr,
        sflux_top=jnp.asarray(var.sflux_top, dtype=jnp.float64),
        din12_indx=din12_indx,
        dbin1=dbin1,
        dbin2=dbin2,
        mu_zenith=float(np.cos(integ._cfg.sl_angle)),
        edd=float(integ._cfg.edd),
        ag0=ag0,
        hc=float(_phy_const.hc),
        f_diurnal=float(integ._cfg.f_diurnal),
        flux_atol=float(integ._cfg.flux_atol),
        ag0_is_zero=(ag0 == 0.0),
    )


def build_refresh_static(integ: OuterLoop, atm: types.SimpleNamespace) -> _atm_refresh_mod.AtmRefreshStatic:
    """Pack the static inputs to `atm_refresh.update_mu_dz_jax`.

    Captures the T-P profile, planetary constants, species masses, and
    the reference layer / boundary z-value from `atm`, cfg and the
    composition table (no `var`). Built with the runner and again per
    profile by `_profile_vars_from_runstate`; the refresh statics are
    fixed for the run.
    """
    from . import composition

    species = composition.species
    nz = atm.Tco.shape[0]
    diff_esc_idx = np.array(
        [species.index(sp) for sp in integ._cfg.diff_esc],
        dtype=np.int32,
    )
    return _atm_refresh_mod.AtmRefreshStatic(
        Tco=jnp.asarray(atm.Tco, dtype=jnp.float64),
        pico=jnp.asarray(atm.pico, dtype=jnp.float64),
        ms=jnp.asarray(atm.ms, dtype=jnp.float64),
        Dzz_top=jnp.asarray(atm.Dzz[-1], dtype=jnp.float64),
        diff_esc_idx=jnp.asarray(diff_esc_idx),
        pref_indx=int(atm.pref_indx),
        zco_pref=float(atm.zco[atm.pref_indx]),
        gs=float(atm.gs),
        Rp=float(integ._cfg.Rp),
        kb=float(_phy_const.kb),
        Navo=float(_phy_const.Navo),
        max_flux=float(integ._cfg.max_flux),
        nz=int(nz),
    )


def build_conden_static(
    integ: OuterLoop, var: types.SimpleNamespace, atm: types.SimpleNamespace, gas_mask_jnp: jnp.ndarray
) -> _conden_mod.CondenStatic | None:
    """Pack the conden tables into a `CondenStatic`.

    None when `use_condense=False`; the runner then omits the conden branch.

    Reactions whose species is in `use_relax` get `coeff_per_re = 0`
    (matches the `var.k[re] = 0` short-circuits at op.py:1121-1123,
    1153-1155); the H2O/NH3 relax blocks are degenerate unless the
    species is in `use_relax` (the `*_active` bools short-circuit at
    trace time).
    """
    if not integ._cfg.use_condense:
        return None

    # Single-source: static metadata from make_conden_spec; every
    # T/structure-dependent array from the same build_conden_profile the
    # on-graph rebuild uses.
    spec = _conden_mod.make_conden_spec(integ._cfg, var, atm, NETWORK.species_idx)
    prof = _conden_mod.build_conden_profile(
        spec,
        jnp.asarray(atm.Tco, dtype=jnp.float64),
        jnp.asarray(atm.pco, dtype=jnp.float64),
        jnp.asarray(atm.n_0, dtype=jnp.float64),
        jnp.asarray(atm.Dzz, dtype=jnp.float64),
    )
    return _conden_mod.CondenStatic(
        conden_re_idx=jnp.asarray(np.asarray(spec.conden_re_idx, dtype=np.int32)),
        conden_sp_idx=jnp.asarray(np.asarray(spec.conden_sp_idx, dtype=np.int32)),
        Dg_per_re=prof.Dg_per_re,
        sat_n_per_re=prof.sat_n_per_re,
        coeff_per_re=jnp.asarray(np.asarray(spec.coeff_per_re, dtype=np.float64)),
        h2o_active=spec.h2o_active,
        h2o_idx=spec.h2o_idx,
        h2o_l_s_idx=spec.h2o_l_s_idx,
        h2o_Dg=prof.h2o_Dg,
        h2o_sat=prof.h2o_sat,
        h2o_m_over_rho_r2=spec.h2o_m_over_rho_r2,
        nh3_active=spec.nh3_active,
        nh3_idx=spec.nh3_idx,
        nh3_l_s_idx=spec.nh3_l_s_idx,
        nh3_Dg=prof.nh3_Dg,
        nh3_sat=prof.nh3_sat,
        nh3_m_over_rho_r2=spec.nh3_m_over_rho_r2,
        nh3_conden_top=int(prof.nh3_conden_top),
        n_0=jnp.asarray(atm.n_0, dtype=jnp.float64),
        gas_indx_mask=gas_mask_jnp,
    )
