"""The integrator's carry and closure inputs: pytrees, field lists and batch axes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

from .jax_step import AtmStatic

if TYPE_CHECKING:
    from collections.abc import Callable

    import jax.numpy as jnp

    from . import atm_refresh as _atm_refresh_mod, photo as _photo_mod
    from .chem import NetworkArrays

__all__ = ["ATM_STATIC_BATCH_AXES", "JaxIntegState", "ProfileVars"]

# JAX publishes no pytree type. A caller's jobs, summaries and tangent trees can be any tree of arrays.
PyTree = Any


class ProfileVars(NamedTuple):
    """Per-profile constants threaded through the carry.

    T-P, abundances, Kzz, gravity, radius, saturation and molecular diffusion.

    `jax.vmap` does not batch closure constants, so per-profile arrays must
    ride this carry, never a runner closure (a closure would share lane-0's
    atmosphere across the batch). Constant during a run; the single-profile
    path seeds the same values the closures bake. `pref_indx` stays a closure
    constant (it sizes a `jnp.arange`) and must be batch-constant
    (prepare_runstate rejects a mismatch).
    """

    # from _Statics
    n_0: jnp.ndarray  # (nz,)              total number density
    Kzz: jnp.ndarray  # (nz-1,)            eddy diffusion (cond_fn slope_min)
    atom_ini: jnp.ndarray  # (n_atoms,)    initial atom abundances (atom_loss)
    y_ini: jnp.ndarray  # (nz, ni)         initial column (element-budget term)
    fix_species_sat_mix: jnp.ndarray  # (n_fix_species, nz)
    # from AtmRefreshStatic
    r_Tco: jnp.ndarray  # (nz,)            atm-refresh temperature
    r_pico: jnp.ndarray  # (nz+1,)         atm-refresh interface pressure
    r_Dzz_top: jnp.ndarray  # (ni,)        top-interface molecular diffusion
    r_gs: jnp.ndarray  # ()               surface gravity
    r_zco_pref: jnp.ndarray  # ()          reference-layer height
    r_Rp: jnp.ndarray  # ()               planet radius
    # from CondenStatic
    c_Dg_per_re: jnp.ndarray  # (n_conden_re, nz)
    c_sat_n_per_re: jnp.ndarray  # (n_conden_re, nz)
    c_h2o_Dg: jnp.ndarray  # (nz,)
    c_h2o_sat: jnp.ndarray  # (nz,)
    c_nh3_Dg: jnp.ndarray  # (nz,)
    c_nh3_sat: jnp.ndarray  # (nz,)
    c_nh3_conden_top: jnp.ndarray  # () int32, argmin(sat_mix['NH3'])
    # from _PhotoStatic: the only T-P-dependent photo statics; the rest is
    # star/network/grid-fixed and must be batch-constant (prepare_runstate
    # guards this). Placeholder shape (0, 1, 1) when use_photo=False.
    p_absp_T_cross: jnp.ndarray  # (n_absp_T, nz, nbin)
    p_cross_J_T: jnp.ndarray  # (n_br_T, nz, nbin)


class JaxIntegState(NamedTuple):
    """Carry state for the JIT'd accept/reject loop.

    Shapes for the HD189 reference config: nz=150, ni=69, n_atoms=4,
    nbin=2588, n_br=48. Scalars are float64 unless noted; counts are
    int32. When use_photo=False, tau/aflux/sflux/dflux_u use placeholder shape (1, 1).
    """

    y: jnp.ndarray  # (nz, ni), cm^-3 current proposed state
    y_prev: jnp.ndarray  # (nz, ni)        last accepted state (revert target on reject)
    ymix: jnp.ndarray  # (nz, ni)        mixing ratios
    dt: jnp.ndarray  # (), s           step size to use for the next attempt
    t: jnp.ndarray  # (), s           elapsed integration time
    delta: jnp.ndarray  # ()              truncation-error proxy of last attempt
    accept_count: jnp.ndarray  # ()  int32       accepted steps
    retry_count: jnp.ndarray  # ()  int32       retries on the in-flight step
    keep_prev_run: jnp.ndarray  # ()  int32     force-accepts at dt_min that kept the state, since the last real accept
    atom_loss: jnp.ndarray  # (n_atoms,)
    atom_loss_prev: jnp.ndarray  # (n_atoms,)
    nega_count: jnp.ndarray  # ()  int32       cumulative over the run
    loss_count: jnp.ndarray  # ()  int32
    delta_count: jnp.ndarray  # ()  int32
    small_y: jnp.ndarray  # ()              cumulative |y| of clipped-small cells
    nega_y: jnp.ndarray  # ()              cumulative |y| of clipped-negative cells

    # Photo state. All zeros / unused when use_photo=False, except `k_arr`
    # (the full rate table, read every step).
    k_arr: jnp.ndarray  # (nr+1, nz)     reaction-rate table
    tau: jnp.ndarray  # (nz+1, nbin)   optical depth
    aflux: jnp.ndarray  # (nz, nbin)     actinic flux
    sflux: jnp.ndarray  # (nz+1, nbin)   direct beam
    dflux_u: jnp.ndarray  # (nz+1, nbin)   diffuse upward (carries between calls)
    aflux_change: jnp.ndarray  # ()             max relative aflux change

    # Atmosphere geometry, refreshed at the end of an accepted iteration every
    # `update_frq` steps (op.py:905-907); the next body_fn splices g/dzi/Hpi/
    # top_flux/vs into AtmStatic so jax_ros2_step sees refreshed diffusion.
    g: jnp.ndarray  # (nz,)          gravity
    mu: jnp.ndarray  # (nz,)          mean molar mass (g/mol)
    Hp: jnp.ndarray  # (nz,)          pressure scale height
    dz: jnp.ndarray  # (nz,)          layer thickness
    zco: jnp.ndarray  # (nz+1,)        interface heights
    dzi: jnp.ndarray  # (nz-1,)        interface dz
    Hpi: jnp.ndarray  # (nz-1,)        interface scale height
    top_flux: jnp.ndarray  # (ni,)          diffusion-limited escape flux at TOA
    vs: jnp.ndarray  # (nz-1, ni)     settling velocity (zeroed post-condense fix)

    # Convergence + step-history. The body writes (y, t) to the ring at
    # index `accept_count % conv_step` and cond_fn reads longdy/longdydt
    # to terminate. `rtol`/`loss_criteria` ride in the carry so the
    # adaptive-rtol updates fire inside the body without retracing.
    # `update_photo_frq`/`is_final_photo_frq` drive the ini->final switch.
    y_time_ring: jnp.ndarray  # (conv_step, nz, ni) float64
    t_time_ring: jnp.ndarray  # (conv_step,)        float64
    longdy: jnp.ndarray  # ()                  float64
    longdydt: jnp.ndarray  # ()                  float64
    where_varies_most: jnp.ndarray  # (nz, ni)            float64
    rtol: jnp.ndarray  # ()                  float64
    loss_criteria: jnp.ndarray  # ()                  float64
    update_photo_frq: jnp.ndarray  # ()                  int32
    is_final_photo_frq: jnp.ndarray  # ()                  bool
    geom_ok: jnp.ndarray  # ()  bool, True only on a certificate-candidate step whose refreshed geometry agreed
    budget_ok: jnp.ndarray  # () bool  True only on a candidate step whose column element budget held
    budget_ref: (
        jnp.ndarray
    )  # (n_atoms,) the t=0 operator-weighted atom column on the t=0 grid (fixed; the budget denominator)
    budget_err: jnp.ndarray  # (n_atoms,) accumulated per-step column change / budget_ref, each step on its own grid
    budget_drift: jnp.ndarray  # (n_atoms,) budget_err relative to H: the certificate's budget operand

    # Post-condensation fixed-species state.
    fix_species_started: jnp.ndarray  # ()                  bool
    fix_y: jnp.ndarray  # (nz, ni)           stored fixed values
    fix_mask: jnp.ndarray  # (nz, ni)           bool mask of fixed cells
    fix_pfix_idx: jnp.ndarray  # (n_fix_sp,)        int32 cold-trap indices

    # Hycean H2/He pin: one-shot snapshot at t>1e6 stored in h2he_mix
    # and applied to the bottom layer thereafter. When use_fix_H2He=False
    # the body branch is a static no-op.
    h2he_pinned: jnp.ndarray  # ()  bool
    h2he_mix: jnp.ndarray  # (2,) float64  [H2_mix, He_mix]

    # save_evolution capture buffer. When save_evolution=False the buffers
    # are length-1 placeholders that the body never writes.
    y_evo: jnp.ndarray  # (save_evo_n_max, nz, ni) float64
    t_evo: jnp.ndarray  # (save_evo_n_max,)        float64
    evo_idx: jnp.ndarray  # ()  int32  next slot to fill

    # Termination state. `is_done` is batched-runner only; every runner sets
    # `termination_reason` on its exit state.
    # `is_done` freezes a finished lane while stragglers finish.
    # `termination_reason`: one of the TERM_* codes.
    is_done: jnp.ndarray  # ()  bool
    termination_reason: jnp.ndarray  # ()  int32

    # Hybrid vm_mol phase blend for jax_ros2_step: 1.0 = upwind (phase 0),
    # 0.0 = central difference (phase 1). Hybrid runs flip 1.0 -> 0.0 the
    # first time phase 0 ends (convergence, runtime, OR step-count), so a run
    # stopping on convergence or a cap is in phase 1 -- a fixed point only if
    # phase 1 also converged. Non-hybrid runs never flip (bit-identical trace).
    hybrid_use_vm: jnp.ndarray  # ()  float64

    # Live termination budget, seeded from the static caps; the termination
    # test reads these, not the closure constants. Only the hybrid phase flip
    # extends them (see _HYBRID_FLIP_*); non-hybrid runs never touch them.
    count_min_dyn: jnp.ndarray  # ()  int32
    count_max_dyn: jnp.ndarray  # ()  int32
    runtime_dyn: jnp.ndarray  # ()  float64

    # Per-profile constants (see ProfileVars).
    pv: ProfileVars


class _PhotoStatic(NamedTuple):
    """Per-run static inputs closed over by the photo-branch closure."""

    photo_data: _photo_mod.PhotoData  # absp / scat cross sections
    cross_J: jnp.ndarray  # (n_br, nbin)
    branch_re_idx: jnp.ndarray  # (n_br,)   int64, k_arr row to write
    branch_active: jnp.ndarray  # (n_br,)   bool
    branch_T_re_idx: jnp.ndarray  # (n_br_T,) int64
    branch_T_active: jnp.ndarray  # (n_br_T,) bool
    cross_Jion: jnp.ndarray  # (n_ion_br, nbin)
    ion_branch_re_idx: jnp.ndarray  # (n_ion_br,) int64, k_arr row to write
    ion_branch_active: jnp.ndarray  # (n_ion_br,) bool
    bins: jnp.ndarray  # (nbin,)   wavelength grid (nm)
    sflux_top: jnp.ndarray  # (nbin,)   TOA stellar flux
    din12_indx: int  # static: wavelength split index for J integration
    dbin1: float
    dbin2: float
    mu_zenith: float  # cos(sl_angle)
    edd: float  # Eddington coefficient
    ag0: float  # asymmetry factor (0 for HD189)
    hc: float  # planck * c (erg nm)
    f_diurnal: float  # diurnal flux average (1.0 tidally locked)
    flux_atol: float  # aflux_change masking floor
    ag0_is_zero: bool  # static: selects compute_flux branch


class _Statics(NamedTuple):
    """Per-run static inputs to the JAX runner.

    Closed-over by the runner; never appears in the carry. The
    convergence/termination caps and the adaptive-rtol /
    photo-frequency-switch knobs all live here.
    """

    compo_arr: jnp.ndarray  # (ni, n_atoms)
    atom_ini_arr: jnp.ndarray  # (n_atoms,)
    loss_eps: float
    pos_cut: float
    nega_cut: float
    mtol: float
    atol: float
    dt_var_min: float
    dt_var_max: float
    dt_min: float
    dt_max: float
    batch_max_retries: int  # safety cap on inner retries per accepted step

    # Convergence + termination
    conv_step: int  # ring buffer length (cfg.conv_step)
    count_min: int
    count_max: int
    runtime: float
    trun_min: float
    st_factor: float
    yconv_cri: float
    yconv_min: float
    slope_cri: float
    flux_cri: float
    geom_conv_tol: float  # certificate: max relative move of the refreshed geometry
    element_budget_tol: float  # certificate: max cumulative column element drift
    budget_ref_atom: int  # column of compo_arr the budget is measured against (H)
    mtol_conv: float
    conver_ignore_mask: jnp.ndarray  # (ni,) bool, species to drop from longdy
    condense_zero_conv_mask: jnp.ndarray  # (nz, ni) bool, non_gas_sp columns
    n_0: jnp.ndarray  # (nz,) atm.n_0; seeds pv.n_0 (the (y-y_old)/n_0 ratio)
    Kzz: jnp.ndarray  # (nz-1,) atm.Kzz; seeds pv.Kzz (slope_min recompute)

    # Photo / adaptive-rtol cadence statics. The dynamic counterparts ride
    # in the carry (s.update_photo_frq / s.rtol / s.loss_criteria).
    use_vm_mol: bool  # upwind molecular diffusion -> refresh vm in-loop with mu
    hybrid_vm_mol: bool  # two-stage: converge upwind, then finish central-diff
    final_update_photo_frq: int
    update_frq: int
    use_adapt_rtol: bool
    rtol_accept: float  # cfg.rtol at build; acceptance never sees adapted rtol
    rtol_min: float
    rtol_max: float
    adapt_rtol_dec_period: int
    adapt_rtol_inc_period: int
    adapt_rtol_dec: float
    adapt_rtol_inc: float
    adapt_rtol_loss_mul: float
    adapt_rtol_inc_loss_thresh: float
    photo_switch_longdy_thresh: float
    photo_switch_longdydt_thresh: float
    hycean_pin_time: float

    # _step_size knobs (from cfg).
    step_size_safety: float
    step_size_zero_delta_frac: float

    # Ion. The bool branches at trace time; when off, the arrays are zero
    # placeholders the body never reads.
    use_ion: bool
    e_idx: int  # species index of 'e' (0 if use_ion=False)
    charge_arr: jnp.ndarray  # (ni,) compo[i]['e'] over charge_list, 0 elsewhere
    use_fix_sp_bot: bool
    fix_sp_bot_idx: jnp.ndarray  # (n_fix_sp_bot,) int32
    fix_sp_bot_mix: jnp.ndarray  # (n_fix_sp_bot,)
    # Hycean H2/He bottom-pin: snapshots ymix[0,H2]/ymix[0,He] once
    # t > hycean_pin_time and pins them in the bottom layer, with fix_sp_bot's
    # `delta[0] = 0`. Indices are -1 sentinels when use_fix_H2He is False.
    use_fix_H2He: bool
    h2_idx: int
    he_idx: int
    use_fix_species: bool
    post_conden_rtol: float
    fix_species_from_coldtrap_lev: bool
    fix_species_idx: jnp.ndarray  # (n_fix_species,) int32
    fix_species_sat_mix: jnp.ndarray  # (n_fix_species, nz)
    fix_species_wholecol: jnp.ndarray  # (n_fix_species,) bool

    # save_evolution capture. When on, the body writes (y, t) to the
    # ring every `save_evo_frq` accepted steps up to `save_evo_n_max`;
    # the populated prefix is returned as rs.step.y_evo / t_evo. When off,
    # the buffers are length-1 placeholders.
    save_evolution: bool
    save_evo_frq: int
    save_evo_n_max: int


class RunnerEnv(NamedTuple):
    """What every part of the step closes over: built once per runner, read at trace time.

    A branch is None when its process is off.
    """

    net: NetworkArrays
    statics: _Statics
    non_gas_present: bool
    gas_indx_mask: jnp.ndarray  # (ni,) bool
    start_conden_time: float  # s
    stop_conden_time: float  # s
    refresh_static: _atm_refresh_mod.AtmRefreshStatic
    clip_fn: Callable[[jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]
    agg_delta_fn: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray | None, jnp.ndarray], jnp.ndarray]
    delta_ratio_fn: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray | None, jnp.ndarray], jnp.ndarray]
    photo_branch: Callable[[JaxIntegState], JaxIntegState] | None
    conden_branch: Callable[[JaxIntegState], JaxIntegState] | None


# Carry leaves the body appends to one slot at a time. The convergence ring is
# written at `(accept_count-1) % conv_step`, the save_evolution buffer at the
# pre-step `evo_idx`.
_CONV_RING_FIELDS = ("y_time_ring", "t_time_ring")

_EVO_RING_FIELDS = ("y_evo", "t_evo")

# Certificate-only lane fields: the convergence ring, its two statistics and
# the cell it points at. They steer predicates, never the arithmetic, so their
# tangents are dead. The lockstep batch drops
# them on its own; the queue's refill `cond` rewrites lane state and keeps them
# (D tangent copies of the conv_step-slot ring per lane), so `runner_queue`
# zeroes them at its lane step. Every primal and every live tangent stays
# bitwise. `_make_jvp_step` (run_jvp) is untouched: its certificate reads
# the ring tangent.
_QUEUE_STOP_FIELDS = (*_CONV_RING_FIELDS, "longdy", "longdydt", "where_varies_most")

# vmap axis name of the batched runner. `body_fn_batch` / `body_fn` psum over
# it to build the lane-uniform predicates their photo / refresh conds need.
_LANE_AXIS = "lane"

# The JaxIntegState fields `photo_branch` writes. Under vmap the batched photo
# cond re-applies each lane's own gate to exactly these.
_PHOTO_FIELDS = (
    "k_arr",
    "tau",
    "aflux",
    "sflux",
    "dflux_u",
    "aflux_change",
)

# in_axes for vmapping the batched runner over AtmStatic: array leaves batch on axis 0; the toggle flags broadcast
# (None), identical within a (nz, toggle-combo) batch and consumed only as traced values, never in a Python `if`.
_ATM_TOGGLES = ("use_vm_mol", "use_settling", "use_topflux", "use_botflux")
ATM_STATIC_BATCH_AXES = AtmStatic._make(None if f in _ATM_TOGGLES else 0 for f in AtmStatic._fields)

# The AtmStatic fields that carry the batch axis, in AtmStatic order.
_ATM_ARRAY_FIELDS = tuple(f for f in AtmStatic._fields if getattr(ATM_STATIC_BATCH_AXES, f) == 0)
