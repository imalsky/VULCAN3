"""The accepted step's after-work: geometry refresh, element budget, hybrid flip, rtol, dt and the photo switch."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from . import atm_refresh as _atm_refresh_mod
from ._convergence import _convergence_ok
from ._step_control import _step_size
from .ini_abun import column_atoms
from .phy_const import UNDERFLOW_DENOM

if TYPE_CHECKING:
    from ._carry import JaxIntegState, _Statics
    from ._loop_env import RunnerEnv

# Hybrid vm_mol phase-flip budget, vm_branch@84d010d stop(): count_min =
# count + 100 on every flip (op.py:1107, 1128, 1152); count_max = count + 2000
# after convergence (op.py:1108), count + 1000 after the runtime or
# step-count limit (op.py:1129, 1153); runtime grows 10% on a runtime flip
# (op.py:1130).
_HYBRID_FLIP_COUNT_MIN_EXTRA = 100
_HYBRID_FLIP_COUNT_MAX_EXTRA_CONV = 2000
_HYBRID_FLIP_COUNT_MAX_EXTRA_BUDGET = 1000
_HYBRID_FLIP_RUNTIME_FACTOR = 1.1


# size-limit-ok: the cond and its two sides read one set of locals
def _refresh_geometry(env: RunnerEnv, s: JaxIntegState, ymix_new, sol_clip, do_accept, candidate, it, lane_axis):
    """Refresh the hydrostatic column and the escape flux on their cadence; return the carry updates."""
    statics, refresh_static = env.statics, env.refresh_static
    # Atm refresh (op.py:905-907): after conden, before hydrostatic
    # balance, on accepted steps only. `s.accept_count` is pre-increment,
    # matching master's `count % update_frq == 0` cadence; batched runs
    # key the cadence to the iteration tick instead so every lane
    # refreshes on the same iteration. The candidate-forced refresh
    # is unchanged either way.
    cadence_count = s.accept_count if lane_axis is None else it
    refresh_due = do_accept & ((jnp.mod(cadence_count, jnp.int32(statics.update_frq)) == jnp.int32(0)) | candidate)
    if lane_axis is not None:
        # Exclude finished lanes: a done lane always reads as a
        # candidate and would pull the batch into the refresh.
        refresh_due = refresh_due & jnp.logical_not(s.is_done)
    # Splice this lane's per-profile atmosphere into the refresh
    # static (see ProfileVars).
    refresh_lane = refresh_static._replace(
        Tco=s.pv.r_Tco,
        pico=s.pv.r_pico,
        Dzz_top=s.pv.r_Dzz_top,
        gs=s.pv.r_gs,
        zco_pref=s.pv.r_zco_pref,
        Rp=s.pv.r_Rp,
    )

    def _do_refresh(_):
        mu_n, g_n, Hp_n, dz_n, zco_n, dzi_n, Hpi_n = _atm_refresh_mod.update_mu_dz_jax(ymix_new, refresh_lane)
        top_flux_n = _atm_refresh_mod.update_phi_esc_jax(
            sol_clip,
            g_n,
            Hp_n,
            s.top_flux,
            refresh_lane,
        )
        return mu_n, g_n, Hp_n, dz_n, zco_n, dzi_n, Hpi_n, top_flux_n

    def _no_refresh(_):
        return (s.mu, s.g, s.Hp, s.dz, s.zco, s.dzi, s.Hpi, s.top_flux)

    if lane_axis is None:
        out = jax.lax.cond(
            refresh_due,
            _do_refresh,
            _no_refresh,
            operand=None,
        )
    else:

        def _do_refresh_gated(_):
            new, old = _do_refresh(None), _no_refresh(None)
            return tuple(jnp.where(refresh_due, n, o) for n, o in zip(new, old, strict=False))

        any_refresh = jax.lax.psum(refresh_due.astype(jnp.int32), lane_axis) > 0
        out = jax.lax.cond(
            any_refresh,
            _do_refresh_gated,
            _no_refresh,
            operand=None,
        )
    (
        mu_next,
        g_next,
        Hp_next,
        dz_next,
        zco_next,
        dzi_next,
        Hpi_next,
        top_flux_next,
    ) = out
    geom_rel = jnp.max(
        jnp.stack(
            [
                jnp.max(jnp.abs(new - old) / jnp.maximum(jnp.abs(new), UNDERFLOW_DENOM))
                for new, old in (
                    (mu_next, s.mu),
                    (g_next, s.g),
                    (Hp_next, s.Hp),
                    (dzi_next, s.dzi),
                    (Hpi_next, s.Hpi),
                )
            ]
        )
    )
    geom_ok_next = candidate & (geom_rel < jnp.float64(statics.geom_conv_tol))
    return {
        "mu": mu_next,
        "g": g_next,
        "Hp": Hp_next,
        "dz": dz_next,
        "zco": zco_next,
        "dzi": dzi_next,
        "Hpi": Hpi_next,
        "top_flux": top_flux_next,
        "geom_ok": geom_ok_next,
    }


def _element_budget(statics: _Statics, s: JaxIntegState, y_next, candidate):
    """Accumulate the cumulative element budget; return the carry updates."""
    # Cumulative element budget: loss_eps misses a slow drain.
    # Accumulate each step's change in the operator-weighted atom column
    # of the returned state (after renormalisation and bottom pins) on
    # that step's grid (s.dz), over the fixed t=0 column; a rejected step
    # adds zero. The gate reads it relative to H: the renormalisation
    # scales every atom column alike. Tolerance: element_budget_tol.
    _safe = jnp.where(s.budget_ref == 0.0, 1.0, s.budget_ref)
    _dcol = column_atoms(y_next, s.dz, statics.compo_arr) - column_atoms(s.y_prev, s.dz, statics.compo_arr)
    budget_err_next = s.budget_err + jnp.where(s.budget_ref == 0.0, 0.0, _dcol / _safe)
    budget_drift_next = (1.0 + budget_err_next) / (1.0 + budget_err_next[statics.budget_ref_atom]) - 1.0
    budget_ok_next = candidate & (jnp.max(jnp.abs(budget_drift_next)) < jnp.float64(statics.element_budget_tol))
    return {"budget_err": budget_err_next, "budget_drift": budget_drift_next, "budget_ok": budget_ok_next}


def _hybrid_flip(
    statics: _Statics,
    s: JaxIntegState,
    longdy_next,
    longdydt_next,
    t_next,
    accept_count_next,
    Hp_next,
    do_accept,
    ready_after,
):
    """Switch the hybrid scheme to central difference when its upwind phase ends.

    Returns `(longdy_next, longdydt_next, carry updates)`; a non-hybrid run passes everything through.
    """
    # Hybrid vm_mol phase flip: when phase 0 (upwind) ends by convergence,
    # runtime or step count, switch to central difference, reset the
    # convergence trackers and extend the budget (_HYBRID_FLIP_*).
    # Dropped at trace time for non-hybrid runs.
    hybrid_use_vm_next = s.hybrid_use_vm
    count_min_dyn_next = s.count_min_dyn
    count_max_dyn_next = s.count_max_dyn
    runtime_dyn_next = s.runtime_dyn
    if statics.hybrid_vm_mol:
        s_after = s._replace(
            longdy=longdy_next,
            longdydt=longdydt_next,
            t=t_next,
            accept_count=accept_count_next,
            Hp=Hp_next,
        )
        is_conv_after = _convergence_ok(s_after, statics)
        in_phase0 = s.hybrid_use_vm > jnp.float64(0.5)
        # Priority matches vm_branch stop(): convergence > runtime > count.
        conv_flip = in_phase0 & do_accept & ready_after & is_conv_after
        runtime_flip = in_phase0 & (t_next > s.runtime_dyn) & ~conv_flip
        count_flip = in_phase0 & (accept_count_next > s.count_max_dyn) & ~conv_flip & ~runtime_flip
        do_flip = conv_flip | runtime_flip | count_flip

        new_count_min = accept_count_next + jnp.int32(_HYBRID_FLIP_COUNT_MIN_EXTRA)
        count_max_conv = accept_count_next + jnp.int32(_HYBRID_FLIP_COUNT_MAX_EXTRA_CONV)
        count_max_budget = accept_count_next + jnp.int32(_HYBRID_FLIP_COUNT_MAX_EXTRA_BUDGET)

        hybrid_use_vm_next = jnp.where(do_flip, jnp.float64(0.0), s.hybrid_use_vm)
        count_min_dyn_next = jnp.where(do_flip, new_count_min, s.count_min_dyn)
        count_max_dyn_next = jnp.where(
            conv_flip,
            count_max_conv,
            jnp.where(runtime_flip | count_flip, count_max_budget, s.count_max_dyn),
        )
        runtime_dyn_next = jnp.where(
            runtime_flip,
            s.runtime_dyn * jnp.float64(_HYBRID_FLIP_RUNTIME_FACTOR),
            s.runtime_dyn,
        )
        longdy_next = jnp.where(do_flip, jnp.float64(jnp.inf), longdy_next)
        longdydt_next = jnp.where(do_flip, jnp.float64(jnp.inf), longdydt_next)
    updates = {
        "hybrid_use_vm": hybrid_use_vm_next,
        "count_min_dyn": count_min_dyn_next,
        "count_max_dyn": count_max_dyn_next,
        "runtime_dyn": runtime_dyn_next,
    }
    return longdy_next, longdydt_next, updates


def _adapt_rtol(statics: _Statics, s: JaxIntegState, atom_loss_new, do_accept, trigger_fix):
    """Return `(rtol_next, loss_criteria_after_dec)`, from the atom loss on accepted steps."""
    # Adaptive rtol (accepted steps only): periodic decrease while
    # |atom_loss| >= loss_criteria; periodic increase below a lower
    # atom-loss threshold.
    max_atom_loss = jnp.max(jnp.abs(atom_loss_new))
    do_dec = (
        jnp.bool_(statics.use_adapt_rtol)
        & do_accept
        & (jnp.mod(s.accept_count, jnp.int32(statics.adapt_rtol_dec_period)) == jnp.int32(0))
        & (max_atom_loss >= s.loss_criteria)
    )
    rtol_dec = jnp.maximum(s.rtol * jnp.float64(statics.adapt_rtol_dec), jnp.float64(statics.rtol_min))
    loss_crit_dec = s.loss_criteria * jnp.float64(statics.adapt_rtol_loss_mul)
    rtol_after_dec = jnp.where(do_dec, rtol_dec, s.rtol)
    loss_criteria_after_dec = jnp.where(do_dec, loss_crit_dec, s.loss_criteria)

    do_inc = (
        jnp.bool_(statics.use_adapt_rtol)
        & do_accept
        & (jnp.mod(s.accept_count, jnp.int32(statics.adapt_rtol_inc_period)) == jnp.int32(0))
        & (s.accept_count > jnp.int32(0))
        & (max_atom_loss < jnp.float64(statics.adapt_rtol_inc_loss_thresh))
    )
    rtol_inc = jnp.minimum(rtol_after_dec * jnp.float64(statics.adapt_rtol_inc), jnp.float64(statics.rtol_max))
    rtol_adapt = jnp.where(do_inc, rtol_inc, rtol_after_dec)
    rtol_next = jnp.where(trigger_fix, jnp.float64(statics.post_conden_rtol), rtol_adapt)
    return rtol_next, loss_criteria_after_dec


def _next_dt(statics: _Statics, s: JaxIntegState, delta, rtol_next, accept, force_accept, next_dt_if_reject):
    """Return the next step size; it runs after the adaptive rtol, so the new tolerance applies."""
    # Step-size control runs *after* adaptive rtol so the post-update
    # tolerance applies to the next step.
    dt_after_normal = _step_size(
        s.dt,
        delta,
        rtol_next,
        statics.dt_var_min,
        statics.dt_var_max,
        statics.dt_min,
        statics.dt_max,
        statics.step_size_safety,
        statics.step_size_zero_delta_frac,
    )
    dt_after_force = _step_size(
        jnp.float64(statics.dt_min),
        delta,
        rtol_next,
        statics.dt_var_min,
        statics.dt_var_max,
        statics.dt_min,
        statics.dt_max,
        statics.step_size_safety,
        statics.step_size_zero_delta_frac,
    )
    return jnp.where(
        force_accept,
        dt_after_force,
        jnp.where(accept, dt_after_normal, next_dt_if_reject),
    )


def _photo_switch(
    env: RunnerEnv, s: JaxIntegState, t_next, accept_count_next, count_min_dyn_next, longdy_next, longdydt_next
):
    """Return the carry updates of the photolysis ini->final frequency switch."""
    statics, photo_branch = env.statics, env.photo_branch
    # Photo-frequency ini->final switch when longdy / longdydt drop
    # below their respective cfg.photo_switch_* thresholds.
    # Upstream evaluates `conv()` (which writes longdy/longdydt) only once
    # `t > trun_min and count > count_min` (op.py:1069); before that both
    # sit at their initial 1.0 and the switch cannot fire. Mirror that
    # readiness gate instead of the always-updated ring-buffer longdy.
    ready_next = (t_next > jnp.float64(statics.trun_min)) & (accept_count_next > count_min_dyn_next)
    switch_to_final = (
        jnp.bool_(photo_branch is not None)
        & ~s.is_final_photo_frq
        & ready_next
        & (longdy_next < jnp.float64(statics.photo_switch_longdy_thresh))
        & (longdydt_next < jnp.float64(statics.photo_switch_longdydt_thresh))
    )
    update_photo_frq_next = jnp.where(switch_to_final, jnp.int32(statics.final_update_photo_frq), s.update_photo_frq)
    is_final_next = s.is_final_photo_frq | switch_to_final
    return {"update_photo_frq": update_photo_frq_next, "is_final_photo_frq": is_final_next}
