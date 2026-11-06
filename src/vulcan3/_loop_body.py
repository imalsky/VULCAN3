"""One trial step of the outer loop (`body_fn`), in program order.

The order per iteration; master's `op.Integration.__call__` is the same but for the refresh (see `_refresh_geometry`):
  1. photo (when accept_count % update_photo_frq == 0)
  2. Ros2 step, clip, atom_loss, delta, accept or reject
  3. conden (on accept, once t >= start_conden_time, until the fix_species freeze fires)
  4. hydrostatic balance (reads post-conden ymix)
  5. ring and save_evolution append (on accept), then (longdy, longdydt) against the ring
  6. atm_refresh (on accept, when accept_count % update_frq == 0 or on a certificate candidate; batched runs key both
     cadences to the loop's iteration tick)
  7. element budget, hybrid flip, adaptive rtol, next dt, photo-frequency ini->final switch
Moving an array operation moves the program.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp

from . import atm_refresh as _atm_refresh_mod
from ._carry import _PHOTO_FIELDS, JaxIntegState, _Statics
from ._convergence import _convergence_ok, conv_jax, ring_slot
from ._loop_branches import _splice_step_atm
from ._step_control import _adapt_rtol, _next_dt
from .ini_abun import column_atoms
from .jax_step import jax_ros2_step
from .phy_const import UNDERFLOW_DENOM

if TYPE_CHECKING:
    from ._carry import RunnerEnv
    from .jax_step import AtmStatic

# Hybrid vm_mol phase-flip budget, vm_branch@84d010d stop(): count_min =
# count + 100 on every flip (op.py:1107, 1128, 1152); count_max = count + 2000
# after convergence (op.py:1108), count + 1000 after the runtime or
# step-count limit (op.py:1129, 1153); runtime grows 10% on a runtime flip
# (op.py:1130).
_HYBRID_FLIP_COUNT_MIN_EXTRA = 100
_HYBRID_FLIP_COUNT_MAX_EXTRA_CONV = 2000
_HYBRID_FLIP_COUNT_MAX_EXTRA_BUDGET = 1000
_HYBRID_FLIP_RUNTIME_FACTOR = 1.1


def activate_fix_species(statics: _Statics, s_in: JaxIntegState) -> JaxIntegState:
    """Freeze the fix_species columns at their current density, up to the cold trap when asked."""
    nz_fix = s_in.pv.n_0.shape[0]
    n_fix = statics.fix_species_idx.shape[0]
    fix_y_new = s_in.fix_y.at[:, statics.fix_species_idx].set(s_in.y[:, statics.fix_species_idx])

    if statics.fix_species_from_coldtrap_lev:
        sat_rho = s_in.pv.n_0[None, :] * s_in.pv.fix_species_sat_mix
        cond_status = s_in.y[:, statics.fix_species_idx].T >= sat_rho
        masked_sat = jnp.where(cond_status, s_in.pv.fix_species_sat_mix, jnp.inf)
        coldtrap_idx = jnp.argmin(masked_sat, axis=1).astype(jnp.int32)
        has_cond = jnp.any(cond_status, axis=1)
        coldtrap_idx = jnp.where(
            statics.fix_species_wholecol,
            jnp.int32(nz_fix - 1),
            jnp.where(has_cond, coldtrap_idx, jnp.int32(0)),
        )
        layer_idx = jnp.arange(nz_fix, dtype=jnp.int32)[:, None]
        fix_cols_mask = layer_idx < coldtrap_idx[None, :]
        fix_pfix_idx = coldtrap_idx
    else:
        fix_cols_mask = jnp.ones((nz_fix, n_fix), dtype=jnp.bool_)
        fix_pfix_idx = jnp.full((n_fix,), jnp.int32(nz_fix))

    fix_mask_new = jnp.zeros_like(s_in.fix_mask)
    fix_mask_new = fix_mask_new.at[:, statics.fix_species_idx].set(fix_cols_mask)
    return s_in._replace(
        fix_species_started=jnp.bool_(True),
        fix_y=fix_y_new,
        fix_mask=fix_mask_new,
        fix_pfix_idx=fix_pfix_idx,
        rtol=jnp.float64(statics.post_conden_rtol),
        vs=jnp.zeros_like(s_in.vs),
    )


def _photo_update(
    env: RunnerEnv, s: JaxIntegState, lane_axis: str | None, photo_take: jnp.ndarray | None
) -> JaxIntegState:
    """Fire the photolysis branch on its cadence: `retry_count == 0` so reject loops do not re-fire it."""
    photo_branch = env.photo_branch
    if photo_branch is not None:
        if lane_axis is None:
            photo_due = (s.retry_count == jnp.int32(0)) & (jnp.mod(s.accept_count, s.update_photo_frq) == jnp.int32(0))
            s = jax.lax.cond(photo_due, photo_branch, lambda ss: ss, s)
        elif photo_take is not None:
            p = photo_branch(s)
            s = s._replace(**{f: jnp.where(photo_take, getattr(p, f), getattr(s, f)) for f in _PHOTO_FIELDS})
    return s


def _ros2_solve(
    env: RunnerEnv, s: JaxIntegState, atm_step: AtmStatic, matrix_free: bool
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run the Ros2 step with the row pins the config asks for; return `(sol, delta_arr)`."""
    statics, net = env.statics, env.net
    # use_ion: master pins the electron rows inside both Ros2 stages
    # (op.py:2908-2911, 2925-2926) so sol[e]=y[e], delta[e]=0; 'e' is then
    # recomputed by the post-step charge balance below. The fix_mask
    # row-pin implements that; unlike fix_species, the e column
    # must not be overwritten with fix_y (the pinned step already returns
    # sol[e] == y[e]).
    step_mask = None
    if statics.use_ion:
        e_mask = jnp.zeros_like(s.fix_mask).at[:, statics.e_idx].set(True)
        step_mask = (s.fix_mask | e_mask) if statics.use_fix_species else e_mask
    elif statics.use_fix_species:
        step_mask = s.fix_mask
    sol, delta_arr = jax_ros2_step(s.y, s.k_arr, s.dt, atm_step, net, fix_mask=step_mask, matrix_free=matrix_free)
    if statics.use_ion:
        delta_arr = jnp.where(step_mask, 0.0, delta_arr)
        if statics.use_fix_species:
            sol = jnp.where(s.fix_mask, s.fix_y, sol)
    elif statics.use_fix_species:
        sol = jnp.where(s.fix_mask, s.fix_y, sol)
        delta_arr = jnp.where(s.fix_mask, 0.0, delta_arr)
    return sol, delta_arr


def _bottom_pins(
    statics: _Statics, s: JaxIntegState, sol: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray | None]:
    """Apply the bottom-layer pins; return `(sol, h2he_pinned_next, h2he_mix_next, zero_bot_dyn)`."""
    # Bottom-layer pins are applied to `sol` before the truncation error
    # and the clip (exoclime@80f75b9 op.py:2935-2946): the carried ymix
    # and the hydrostatic rebalance then see the pinned value normalized
    # with its layer, as upstream. use_fix_H2He (op.py:2935-2941)
    # snapshots the pre-step ymix[0] once `t > hycean_pin_time` and turns
    # the fix_sp_bot pin (and its `delta[0] = 0`) on for the rest of the
    # run; no accept gate, since upstream snapshots before accept/reject.
    if statics.use_fix_sp_bot:
        sol = sol.at[0, statics.fix_sp_bot_idx].set(statics.fix_sp_bot_mix * s.pv.n_0[0])
    if statics.use_fix_H2He:
        trip = (~s.h2he_pinned) & (s.t > jnp.float64(statics.hycean_pin_time))
        h2_mix_snap = jnp.where(trip, s.ymix[0, statics.h2_idx], s.h2he_mix[0])
        he_mix_snap = jnp.where(trip, s.ymix[0, statics.he_idx], s.h2he_mix[1])
        h2he_mix_next = jnp.stack([h2_mix_snap, he_mix_snap])
        h2he_pinned_next = s.h2he_pinned | trip
        sol = sol.at[0, statics.h2_idx].set(
            jnp.where(
                h2he_pinned_next,
                h2_mix_snap * s.pv.n_0[0],
                sol[0, statics.h2_idx],
            )
        )
        sol = sol.at[0, statics.he_idx].set(
            jnp.where(
                h2he_pinned_next,
                he_mix_snap * s.pv.n_0[0],
                sol[0, statics.he_idx],
            )
        )
        zero_bot_dyn = h2he_pinned_next
    else:
        h2he_pinned_next = s.h2he_pinned
        h2he_mix_next = s.h2he_mix
        zero_bot_dyn = None
    return sol, h2he_pinned_next, h2he_mix_next, zero_bot_dyn


class _StepDecision(NamedTuple):
    """The accept or reject outcome of one trial."""

    accept: jnp.ndarray
    force_accept: jnp.ndarray
    do_accept: jnp.ndarray
    next_dt_if_reject: jnp.ndarray
    delta_count_inc: jnp.ndarray
    nega_count_inc: jnp.ndarray
    loss_count_inc: jnp.ndarray
    t_next: jnp.ndarray
    y_post_clip: jnp.ndarray
    ymix_new: jnp.ndarray


def _decide_step(
    statics: _Statics,
    s: JaxIntegState,
    sol: jnp.ndarray,
    sol_clip: jnp.ndarray,
    ymix_new: jnp.ndarray,
    atom_loss_new: jnp.ndarray,
    delta: jnp.ndarray,
) -> _StepDecision:
    """Accept, reject or force-accept the trial, and count why it failed."""
    # Acceptance uses the build-time rtol, never the carry's adapted one:
    # upstream binds `rtol = vulcan_cfg.rtol` as a default argument of
    # `step_ok`/`step_reject` at import (exoclime@80f75b9 op.py:2489,2495;
    # vm_branch@84d010d op.py:2568,2574), so the adaptive-rtol and
    # `post_conden_rtol` writes reach only `step_size`, which reads the
    # live value. `s.rtol` therefore feeds `_step_size` alone.
    all_nonneg = jnp.all(sol_clip >= 0)
    loss_diff = jnp.max(jnp.abs(atom_loss_new - s.atom_loss_prev))
    accept = all_nonneg & (loss_diff < statics.loss_eps) & (delta <= jnp.float64(statics.rtol_accept))

    # Force-accept when shrinking dt would underflow or the retry budget
    # is spent, so the runner cannot get stuck.
    next_dt_if_reject = s.dt * statics.dt_var_min
    dt_underflow = next_dt_if_reject < statics.dt_min
    retry_exhausted = s.retry_count >= jnp.int32(statics.batch_max_retries)
    force_accept = (dt_underflow | retry_exhausted) & ~accept
    do_accept = accept | force_accept

    # Exactly one counter increments per failed attempt, classified on
    # `~accept` (not `~do_accept`) so a force-accepted failure still
    # counts -- master bumps its reject counter before the dt<dt_min
    # force-accept clamp (op.py:2495-2518).
    delta_too_big = delta > jnp.float64(statics.rtol_accept)
    any_neg = jnp.any(sol_clip < 0)
    attempt_failed = ~accept
    delta_count_inc = (attempt_failed & delta_too_big).astype(jnp.int32)
    nega_count_inc = (attempt_failed & ~delta_too_big & any_neg).astype(jnp.int32)
    loss_count_inc = (attempt_failed & ~delta_too_big & ~any_neg).astype(jnp.int32)

    # Time advance: dt for accept, dt_min for force_accept, 0 for reject.
    dt_used_for_t = jnp.where(force_accept, jnp.float64(statics.dt_min), s.dt)
    t_next = jnp.where(do_accept, s.t + dt_used_for_t, s.t)

    # A force-accepted finite step keeps the previous state whole: master's
    # step_reject (op.py:2515-2520) resets only var.y at dt_min and keeps
    # the rejected solve's ymix, which the rebalance below turns back into
    # y, so it accepts the composition it rejected (notes "Parity ledger"). A
    # non-finite solve still goes through, so the run ends as non-finite
    # instead of idling to the step cap.
    y_prev_clipped = jnp.where(s.y_prev < 0, 0.0, s.y_prev)
    y_post_clip = jnp.where(force_accept, y_prev_clipped, sol_clip)
    keep_prev = force_accept & jnp.all(jnp.isfinite(sol))
    ymix_new = jnp.where(keep_prev, s.ymix, ymix_new)
    return _StepDecision(
        accept,
        force_accept,
        do_accept,
        next_dt_if_reject,
        delta_count_inc,
        nega_count_inc,
        loss_count_inc,
        t_next,
        y_post_clip,
        ymix_new,
    )


# size-limit-ok: the cond and its two sides read one set of locals
def _refresh_geometry(
    env: RunnerEnv,
    s: JaxIntegState,
    ymix_new: jnp.ndarray,
    sol_clip: jnp.ndarray,
    do_accept: jnp.ndarray,
    candidate: jnp.ndarray,
    it: jnp.ndarray | None,
    lane_axis: str | None,
) -> JaxIntegState:
    """Refresh the hydrostatic column and the escape flux on their cadence, and judge the geometry on a candidate.

    Master refreshes before the rebalance (op.py:905-907); this reads the post-conden `ymix_new` and `sol_clip`, which
    the rebalance leaves alone, so it can follow the certificate candidate that forces it. `s.accept_count` is
    pre-increment, as master's `count % update_frq == 0`; batched runs key the cadence to the iteration tick so every
    lane refreshes on the same iteration.
    """
    statics, refresh_static = env.statics, env.refresh_static
    cadence_count = s.accept_count if lane_axis is None else it
    refresh_due = do_accept & ((jnp.mod(cadence_count, jnp.int32(statics.update_frq)) == jnp.int32(0)) | candidate)
    if lane_axis is not None:
        # A done lane always reads as a candidate and would pull the batch into the refresh.
        refresh_due = refresh_due & jnp.logical_not(s.is_done)
    # This lane's atmosphere rides the carry (see ProfileVars).
    refresh_lane = refresh_static._replace(
        Tco=s.pv.r_Tco,
        pico=s.pv.r_pico,
        Dzz_top=s.pv.r_Dzz_top,
        gs=s.pv.r_gs,
        zco_pref=s.pv.r_zco_pref,
        Rp=s.pv.r_Rp,
    )

    def _do_refresh(_: None) -> tuple[jnp.ndarray, ...]:
        mu_n, g_n, Hp_n, dz_n, zco_n, dzi_n, Hpi_n = _atm_refresh_mod.update_mu_dz_jax(ymix_new, refresh_lane)
        top_flux_n = _atm_refresh_mod.update_phi_esc_jax(
            sol_clip,
            g_n,
            Hp_n,
            s.top_flux,
            refresh_lane,
        )
        return mu_n, g_n, Hp_n, dz_n, zco_n, dzi_n, Hpi_n, top_flux_n

    def _no_refresh(_: None) -> tuple[jnp.ndarray, ...]:
        return (s.mu, s.g, s.Hp, s.dz, s.zco, s.dzi, s.Hpi, s.top_flux)

    if lane_axis is None:
        out = jax.lax.cond(
            refresh_due,
            _do_refresh,
            _no_refresh,
            operand=None,
        )
    else:

        def _do_refresh_gated(_: None) -> tuple[jnp.ndarray, ...]:
            new, old = _do_refresh(None), _no_refresh(None)
            return tuple(jnp.where(refresh_due, n, o) for n, o in zip(new, old, strict=False))

        any_refresh = jax.lax.psum(refresh_due.astype(jnp.int32), lane_axis) > 0
        out = jax.lax.cond(
            any_refresh,
            _do_refresh_gated,
            _no_refresh,
            operand=None,
        )
    mu_next, g_next, Hp_next, dz_next, zco_next, dzi_next, Hpi_next, top_flux_next = out
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
    return s._replace(
        mu=mu_next,
        g=g_next,
        Hp=Hp_next,
        dz=dz_next,
        zco=zco_next,
        dzi=dzi_next,
        Hpi=Hpi_next,
        top_flux=top_flux_next,
        geom_ok=candidate & (geom_rel < jnp.float64(statics.geom_conv_tol)),
    )


# size-limit-ok: one trial step in program order, and each part reads the locals of the ones before it
def body_fn(
    env: RunnerEnv,
    s: JaxIntegState,
    atm_static_: AtmStatic,
    lane_axis: str | None = None,
    it: jnp.ndarray | None = None,
    hold: jnp.ndarray | None = None,
    photo_take: jnp.ndarray | None = None,
    defer_ring: bool = False,
    matrix_free: bool = True,
) -> tuple[JaxIntegState, jnp.ndarray, jnp.ndarray]:
    """Advance the carry by one trial step; return `(carry, ring slot, y for that slot)`.

    `lane_axis=None` is the single-profile trace. A string means the body runs under
    `jax.vmap(..., axis_name=lane_axis)` from `body_fn_batch`: the refresh cond then takes a lane-uniform predicate (a
    psum over the lanes, which vmap returns unmapped) so it lowers to a real `case` instead of both sides running on
    every lane, and each lane keeps its own gate as a `where` inside the taken branch. `it` is the batched loop's
    iteration tick (unmapped), which keys that cadence. Batched only: `hold` is the lanes the tick keeps whatever this
    step does (the ring writes fold it in), and `photo_take` the lanes that take photolysis now, None on a tick without
    any (`body_fn_batch` owns that cond).
    """
    statics, gas_indx_mask = env.statics, env.gas_indx_mask
    s = _photo_update(env, s, lane_axis, photo_take)
    atm_step = _splice_step_atm(atm_static_, s, statics.use_vm_mol, env.refresh_static.kb, env.refresh_static.Navo)
    sol, delta_arr = _ros2_solve(env, s, atm_step, matrix_free)
    sol, h2he_pinned_next, h2he_mix_next, zero_bot_dyn = _bottom_pins(statics, s, sol)
    sol_clip, ymix_new, small_y_inc, nega_y_inc = env.clip_fn(sol)

    # Per-atom relative conservation residual, (n_atoms,).
    atom_sum = jnp.einsum("zi,ia->a", sol_clip, statics.compo_arr)
    atom_loss_new = (atom_sum - s.pv.atom_ini) / s.pv.atom_ini
    # delta uses the pre-clip sol (master computes it before clip): sol_clip would erase the truncation error of cells
    # about to clip to zero and let overly aggressive steps through (HD209 exercises this). The trial mixing ratio of
    # the mask is pre-clip too: the clip zeroes a species the step creates below pos_cut (notes "Parity ledger"). Its
    # denominator is the clip's: the gas species only when condensates exist.
    gas_sum = jnp.sum(
        jnp.where(gas_indx_mask[None, :], sol, 0.0) if env.non_gas_present else sol, axis=1, keepdims=True
    )
    ymix_trial = sol / jnp.maximum(gas_sum, UNDERFLOW_DENOM)
    delta = env.agg_delta_fn(sol, delta_arr, s.ymix, zero_bot_dyn, ymix_trial)

    d = _decide_step(statics, s, sol, sol_clip, ymix_new, atom_loss_new, delta)
    do_accept, t_next = d.do_accept, d.t_next

    # Condensation gates on the pre-step `s.t`, not `t_next`: master's save_step advances var.t after conden runs.
    s_post = s._replace(y=d.y_post_clip, ymix=d.ymix_new)
    trigger_fix = jnp.bool_(False)
    if env.conden_branch is not None:
        in_conden_window = s.t >= jnp.float64(env.start_conden_time)
        fire_conden = do_accept & in_conden_window & ~s.fix_species_started
        # The pin activates before the relax, so `fix_y` freezes the post-solve reservoir master freezes: op.py:871-873
        # snapshots `fix_y` (and :882 reads the cold-trap level) from the same y, then op.py:898-902 relaxes inside the
        # same block. Order is free on every other step: `trigger_fix` implies `fire_conden`, and `conden_branch` reads
        # none of the fields the activation sets (it touches k_arr/y/ymix from y, ymix, dt).
        if statics.use_fix_species:
            trigger_fix = (
                do_accept & ~s.fix_species_started & in_conden_window & (s.t > jnp.float64(env.stop_conden_time))
            )
            s_post = jax.lax.cond(
                trigger_fix,
                functools.partial(activate_fix_species, statics),
                lambda ss: ss,
                s_post,
            )
        s_post = jax.lax.cond(
            fire_conden,
            env.conden_branch,
            lambda ss: ss,
            s_post,
        )
    y_after_conden, ymix_after_conden = s_post.y, s_post.ymix

    # Hydrostatic rebalance. With condensation on, only gas columns are rebalanced after Ros2.
    n_0 = atm_step.M[:, None]
    sol_balanced_full = n_0 * ymix_after_conden
    if env.conden_branch is not None:
        sol_balanced = jnp.where(gas_indx_mask[None, :], sol_balanced_full, y_after_conden)
    else:
        sol_balanced = sol_balanced_full
    # Charge balance: with `charge_arr[i] = compo[i]['e']` for charged species (and 0 for 'e' itself),
    # `e[:] = -dot(y, charge_arr)` enforces zero net charge per layer.
    if statics.use_ion:
        e_density = -jnp.einsum("zi,i->z", sol_balanced, statics.charge_arr)
        sol_balanced = sol_balanced.at[:, statics.e_idx].set(e_density)

    y_next = jnp.where(do_accept, sol_balanced, s.y_prev)
    # A retry starts from the previous state: master restores var.y only (op.py:2524-2531), so its retry masks with the
    # rejected ymix (notes "Parity ledger").
    ymix_next = jnp.where(do_accept, ymix_after_conden, s.ymix)
    atom_loss_next = jnp.where(d.accept, atom_loss_new, s.atom_loss)
    # y_prev is the revert target: on accept, the fresh state; on reject, the carry's (still retrying this step).
    y_prev_next = jnp.where(do_accept, y_next, s.y_prev)
    atom_loss_prev_next = jnp.where(do_accept, atom_loss_next, s.atom_loss_prev)
    accept_count_next = s.accept_count + jnp.where(do_accept, jnp.int32(1), jnp.int32(0))
    retry_count_next = jnp.where(do_accept, jnp.int32(0), s.retry_count + jnp.int32(1))

    # Ring append (op.save_step). On reject this rewrites the same slot with unchanged values. A batched lane the tick
    # keeps (`hold`, or this step went non-finite: `body_fn_batch`'s freeze rule) rewrites its pre-step slot instead,
    # and the freeze passes the ring through. Batched (`defer_ring`), the y slot is returned, not written:
    # `body_fn_batch` writes it after its per-tick photolysis cond, since a cond returning the written
    # (conv_step, nz, ni) ring made XLA:GPU rebuild and copy the whole ring every tick (notes "Efficiency program").
    keep = None
    ring_idx = ring_slot(statics, accept_count_next)
    y_slot, t_slot = y_next, t_next
    if hold is not None:
        keep = hold | jnp.logical_not(jnp.all(jnp.isfinite(y_next)))
        y_slot = jnp.where(keep, s.y_time_ring[ring_idx], y_next)
        t_slot = jnp.where(keep, s.t_time_ring[ring_idx], t_next)
    y_time_ring_new = s.y_time_ring if defer_ring else s.y_time_ring.at[ring_idx].set(y_slot)
    t_time_ring_new = s.t_time_ring.at[ring_idx].set(t_slot)

    # save_evolution. Master appends every accepted step and post-slices [::save_evo_frq], keeping accepted steps 1,
    # K+1, 2K+1...; gating on `(accept_count_next - 1) % K == 0` reproduces that.
    y_evo_new, t_evo_new, evo_idx_new = s.y_evo, s.t_evo, s.evo_idx
    if statics.save_evolution:
        do_evo_append = (
            do_accept
            & (jnp.mod(accept_count_next - jnp.int32(1), jnp.int32(statics.save_evo_frq)) == jnp.int32(0))
            & (s.evo_idx < jnp.int32(statics.save_evo_n_max))
        )
        if hold is not None:
            # The batched freeze passes the buffer through (see the ring).
            do_evo_append = do_evo_append & jnp.logical_not(keep)
        evo_slot = s.evo_idx
        zero_i32 = jnp.int32(0)
        y_evo_new = jax.lax.cond(
            do_evo_append,
            lambda yev: jax.lax.dynamic_update_slice(yev, y_next[None, :, :], (evo_slot, zero_i32, zero_i32)),
            lambda yev: yev,
            s.y_evo,
        )
        t_evo_new = jax.lax.cond(
            do_evo_append,
            lambda tev: tev.at[evo_slot].set(t_next),
            lambda tev: tev,
            s.t_evo,
        )
        evo_idx_new = s.evo_idx + jnp.where(do_evo_append, jnp.int32(1), jnp.int32(0))

    # longdy against the ring, with the y ring and the slot just written. Solo, that is the written ring: a read of the
    # pre-write ring after the in-place write made XLA:CPU copy the whole (conv_step, nz, ni) ring twice an iteration
    # (notes "Efficiency program"). Batched (`defer_ring`) it is the pre-write ring; the overlay gives the same values.
    s_for_conv = s._replace(y=y_next, ymix=ymix_next, t=t_next, t_time_ring=t_time_ring_new)
    longdy_new_val, longdydt_new_val, where_varies_most_new = conv_jax(
        statics,
        s_for_conv,
        accept_count_next,
        s.y_time_ring if defer_ring else y_time_ring_new,
        ring_idx,
        y_slot,
    )
    longdy_next = jnp.where(do_accept, longdy_new_val, s.longdy)
    longdydt_next = jnp.where(do_accept, longdydt_new_val, s.longdydt)
    where_varies_most_next = jnp.where(do_accept, where_varies_most_new, s.where_varies_most)

    # Certificate candidate: `_convergence_ok` on this step's longdy. The refresh is forced on it, and geom_ok records
    # whether mu/g/Hp/dzi/Hpi moved less than geom_conv_tol. Hp here is the pre-refresh value; the certificate re-reads
    # the refreshed one next iteration.
    s_cand = s._replace(longdy=longdy_next, longdydt=longdydt_next, t=t_next, accept_count=accept_count_next)
    is_conv_cand = _convergence_ok(s_cand, statics)
    ready_after = (t_next > jnp.float64(statics.trun_min)) & (accept_count_next > s.count_min_dyn)
    candidate = do_accept & is_conv_cand & ready_after
    s_geom = _refresh_geometry(env, s, ymix_after_conden, y_after_conden, do_accept, candidate, it, lane_axis)

    # Cumulative element budget: loss_eps misses a slow drain. Accumulate each step's change in the operator-weighted
    # atom column of the returned state (after renormalisation and bottom pins) on that step's grid (s.dz), over the
    # fixed t=0 column; a rejected step adds zero. The gate reads it relative to H: the renormalisation scales every
    # atom column alike. Tolerance: element_budget_tol.
    budget_ref_safe = jnp.where(s.budget_ref == 0.0, 1.0, s.budget_ref)
    budget_dcol = column_atoms(y_next, s.dz, statics.compo_arr) - column_atoms(s.y_prev, s.dz, statics.compo_arr)
    budget_err_next = s.budget_err + jnp.where(s.budget_ref == 0.0, 0.0, budget_dcol / budget_ref_safe)
    budget_drift_next = (1.0 + budget_err_next) / (1.0 + budget_err_next[statics.budget_ref_atom]) - 1.0
    budget_ok_next = candidate & (jnp.max(jnp.abs(budget_drift_next)) < jnp.float64(statics.element_budget_tol))

    # Hybrid vm_mol phase flip: when phase 0 (upwind) ends by convergence, runtime or step count, switch to central
    # difference, reset the convergence trackers and extend the budget (_HYBRID_FLIP_*). Priority as vm_branch stop():
    # convergence > runtime > count.
    hybrid_use_vm_next = s.hybrid_use_vm
    count_min_dyn_next = s.count_min_dyn
    count_max_dyn_next = s.count_max_dyn
    runtime_dyn_next = s.runtime_dyn
    if statics.hybrid_vm_mol:
        is_conv_after = _convergence_ok(s_cand._replace(Hp=s_geom.Hp), statics)
        in_phase0 = s.hybrid_use_vm > jnp.float64(0.5)
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

    rtol_next, loss_criteria_next = _adapt_rtol(statics, s, atom_loss_new, do_accept, trigger_fix)
    dt_next = _next_dt(statics, s, delta, rtol_next, d.accept, d.force_accept, d.next_dt_if_reject)

    # Photolysis frequency ini->final switch. Upstream evaluates `conv()` (which writes longdy/longdydt) only once
    # `t > trun_min and count > count_min` (op.py:1069); before that both sit at their initial 1.0 and the switch cannot
    # fire. Mirror that gate instead of the always-updated ring-buffer longdy.
    ready_next = (t_next > jnp.float64(statics.trun_min)) & (accept_count_next > count_min_dyn_next)
    switch_to_final = (
        jnp.bool_(env.photo_branch is not None)
        & ~s.is_final_photo_frq
        & ready_next
        & (longdy_next < jnp.float64(statics.photo_switch_longdy_thresh))
        & (longdydt_next < jnp.float64(statics.photo_switch_longdydt_thresh))
    )
    update_photo_frq_next = jnp.where(switch_to_final, jnp.int32(statics.final_update_photo_frq), s.update_photo_frq)
    is_final_next = s.is_final_photo_frq | switch_to_final

    s_next = s_geom._replace(
        y=y_next,
        y_prev=y_prev_next,
        ymix=ymix_next,
        dt=dt_next,
        t=t_next,
        delta=delta,
        accept_count=accept_count_next,
        retry_count=retry_count_next,
        atom_loss=atom_loss_next,
        atom_loss_prev=atom_loss_prev_next,
        nega_count=s.nega_count + d.nega_count_inc,
        loss_count=s.loss_count + d.loss_count_inc,
        delta_count=s.delta_count + d.delta_count_inc,
        small_y=s.small_y + small_y_inc,
        nega_y=s.nega_y + nega_y_inc,
        k_arr=s_post.k_arr,
        fix_species_started=s_post.fix_species_started,
        fix_y=s_post.fix_y,
        fix_mask=s_post.fix_mask,
        fix_pfix_idx=s_post.fix_pfix_idx,
        vs=s_post.vs,
        y_time_ring=y_time_ring_new,
        t_time_ring=t_time_ring_new,
        y_evo=y_evo_new,
        t_evo=t_evo_new,
        evo_idx=evo_idx_new,
        longdy=longdy_next,
        longdydt=longdydt_next,
        where_varies_most=where_varies_most_next,
        rtol=rtol_next,
        loss_criteria=loss_criteria_next,
        h2he_pinned=h2he_pinned_next,
        h2he_mix=h2he_mix_next,
        budget_err=budget_err_next,
        budget_drift=budget_drift_next,
        budget_ok=budget_ok_next,
        hybrid_use_vm=hybrid_use_vm_next,
        count_min_dyn=count_min_dyn_next,
        count_max_dyn=count_max_dyn_next,
        runtime_dyn=runtime_dyn_next,
        update_photo_frq=update_photo_frq_next,
        is_final_photo_frq=is_final_next,
    )
    return s_next, ring_idx, y_slot
