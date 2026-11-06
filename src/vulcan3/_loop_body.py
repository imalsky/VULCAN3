"""One step of the outer loop: the single-profile body, in named parts.

Body order per iteration (matches `op.Integration.__call__`):
  1. photo       (when accept_count % update_photo_frq == 0)
  2. Ros2 step + clip + atom_loss + delta + accept/reject
  3. conden      (on accept, once t >= start_conden_time, until the fix_species freeze fires)
  4. atm_refresh (on accept, when accept_count % update_frq == 0 or on a certificate candidate; batched runs key both
                  cadences to the loop's iteration tick)
  5. hydrostatic balance (reads post-conden ymix)
  6. ring-buffer append (on accept)
  7. recompute (longdy, longdydt) against the ring
  8. adaptive rtol
  9. photo-frequency ini->final switch
Moving an array operation moves the program.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp

from ._carry import _PHOTO_FIELDS, JaxIntegState, _Statics
from ._convergence import _convergence_ok
from ._loop_after_step import _adapt_rtol, _element_budget, _hybrid_flip, _next_dt, _photo_switch, _refresh_geometry
from ._loop_branches import _splice_step_atm
from ._termination import conv_jax, ring_slot
from .jax_step import jax_ros2_step
from .phy_const import UNDERFLOW_DENOM

if TYPE_CHECKING:
    from ._loop_env import RunnerEnv

# --- batched freeze ------------------------------------------------------
# The batched body advances every lane and then keeps the pre-step carry on
# lanes that are done. Selecting every carry leaf whole costs a full copy of
# the carry per iteration, so `_freeze_leaf` skips the leaves the step passed
# through, and the ring buffers below are never selected: `body_fn` writes
# their one slot with the freeze already applied (its `hold`). A freeze that
# read the pre-step ring after that write would keep two rings live, and XLA
# would copy the whole (conv_step, nz, ni) leaf twice per iteration.


def activate_fix_species(statics: _Statics, s_in: JaxIntegState) -> JaxIntegState:
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


def _photo_update(env: RunnerEnv, s: JaxIntegState, lane_axis, photo_take) -> JaxIntegState:
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


def _ros2_solve(env: RunnerEnv, s: JaxIntegState, atm_step, matrix_free):
    """Run the Ros2 step with the row pins the config asks for; return `(sol, delta_arr)`."""
    statics, net = env.statics, env.net
    # use_ion: master pins the electron rows inside both Ros2 stages
    # (op.py:2908-2911, 2925-2926) so sol[e]=y[e], delta[e]=0; 'e' is then
    # recomputed by the post-step charge balance below. The fix_mask
    # row-pin implements that; unlike fix_species, the e column
    # must not be overwritten with fix_y (the pinned step already returns
    # sol[e] == y[e]).
    if statics.use_ion:
        e_mask = jnp.zeros_like(s.fix_mask).at[:, statics.e_idx].set(True)
        step_mask = (s.fix_mask | e_mask) if statics.use_fix_species else e_mask
        sol, delta_arr = jax_ros2_step(
            s.y,
            s.k_arr,
            s.dt,
            atm_step,
            net,
            fix_mask=step_mask,
            matrix_free=matrix_free,
        )
        delta_arr = jnp.where(step_mask, 0.0, delta_arr)
        if statics.use_fix_species:
            sol = jnp.where(s.fix_mask, s.fix_y, sol)
    elif statics.use_fix_species:
        sol, delta_arr = jax_ros2_step(
            s.y,
            s.k_arr,
            s.dt,
            atm_step,
            net,
            fix_mask=s.fix_mask,
            matrix_free=matrix_free,
        )
        sol = jnp.where(s.fix_mask, s.fix_y, sol)
        delta_arr = jnp.where(s.fix_mask, 0.0, delta_arr)
    else:
        sol, delta_arr = jax_ros2_step(s.y, s.k_arr, s.dt, atm_step, net, matrix_free=matrix_free)
    return sol, delta_arr


def _bottom_pins(statics: _Statics, s: JaxIntegState, sol):
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


def _loss_and_delta(env: RunnerEnv, s: JaxIntegState, sol, sol_clip, delta_arr, zero_bot_dyn):
    """Return this trial's per-atom relative conservation residual and its aggregate truncation error."""
    statics, gas_indx_mask, non_gas_present, agg_delta_fn = (
        env.statics,
        env.gas_indx_mask,
        env.non_gas_present,
        env.agg_delta_fn,
    )
    # Per-atom relative conservation residual, (n_atoms,).
    atom_sum = jnp.einsum("zi,ia->a", sol_clip, statics.compo_arr)
    atom_loss_new = (atom_sum - s.pv.atom_ini) / s.pv.atom_ini
    # delta uses the pre-clip sol (master computes it before clip):
    # sol_clip would erase the truncation error of cells about to clip to
    # zero and let overly aggressive steps through (HD209 exercises this).
    # The trial mixing ratio of the mask is pre-clip too: the clip zeroes a
    # species the step creates below pos_cut (notes "Parity ledger"). Its
    # denominator is the clip's: the gas species only when condensates exist.
    gas_sum = jnp.sum(jnp.where(gas_indx_mask[None, :], sol, 0.0) if non_gas_present else sol, axis=1, keepdims=True)
    ymix_trial = sol / jnp.maximum(gas_sum, UNDERFLOW_DENOM)
    delta = agg_delta_fn(sol, delta_arr, s.ymix, zero_bot_dyn, ymix_trial)
    return atom_loss_new, delta


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


def _decide_step(statics: _Statics, s: JaxIntegState, sol, sol_clip, ymix_new, atom_loss_new, delta) -> _StepDecision:
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


def _conden_step(env: RunnerEnv, s: JaxIntegState, do_accept, y_post_clip, ymix_new):
    """Run condensation and the fix_species activation on an accepted step.

    Returns `(sol_clip, ymix_new, trigger_fix, carry updates)`.
    """
    statics, conden_branch = env.statics, env.conden_branch
    start_conden_time, stop_conden_time = env.start_conden_time, env.stop_conden_time
    # Conden gates on the pre-step `s.t` (not `t_next`); master's
    # save_step advances var.t after conden runs, so this matches.
    if conden_branch is not None:
        s_post = s._replace(y=y_post_clip, ymix=ymix_new)
        in_conden_window = s.t >= jnp.float64(start_conden_time)
        fire_conden = do_accept & in_conden_window & ~s.fix_species_started
        # The pin activates before the relax, so `fix_y` freezes the
        # post-solve reservoir master freezes: op.py:871-873 snapshots
        # `fix_y` (and :882 reads the cold-trap level) from the same y,
        # then op.py:898-902 relaxes inside the same block. Order is
        # free on every other step: `trigger_fix` implies `fire_conden`,
        # and `conden_branch` reads none of the fields the activation sets
        # (it touches k_arr/y/ymix from y, ymix, dt).
        if statics.use_fix_species:
            trigger_fix = do_accept & ~s.fix_species_started & in_conden_window & (s.t > jnp.float64(stop_conden_time))
            s_post = jax.lax.cond(
                trigger_fix,
                functools.partial(activate_fix_species, statics),
                lambda ss: ss,
                s_post,
            )
        else:
            trigger_fix = jnp.bool_(False)
        s_post = jax.lax.cond(
            fire_conden,
            conden_branch,
            lambda ss: ss,
            s_post,
        )
        sol_clip = s_post.y
        ymix_new = s_post.ymix
        k_arr_next = s_post.k_arr
        fix_species_started_next = s_post.fix_species_started
        fix_y_next = s_post.fix_y
        fix_mask_next = s_post.fix_mask
        fix_pfix_idx_next = s_post.fix_pfix_idx
        vs_next = s_post.vs
    else:
        sol_clip = y_post_clip
        k_arr_next = s.k_arr
        fix_species_started_next = s.fix_species_started
        fix_y_next = s.fix_y
        fix_mask_next = s.fix_mask
        fix_pfix_idx_next = s.fix_pfix_idx
        vs_next = s.vs
        trigger_fix = jnp.bool_(False)
    updates = {
        "k_arr": k_arr_next,
        "fix_species_started": fix_species_started_next,
        "fix_y": fix_y_next,
        "fix_mask": fix_mask_next,
        "fix_pfix_idx": fix_pfix_idx_next,
        "vs": vs_next,
    }
    return sol_clip, ymix_new, trigger_fix, updates


def _rebalance(env: RunnerEnv, s: JaxIntegState, atm_step, sol_clip, ymix_new, do_accept, accept, atom_loss_new):
    """Rebalance the hydrostatic column and advance y, ymix, atom_loss and the counters.

    Returns `(y_next, ymix_next, carry updates)`.
    """
    statics, conden_branch, gas_indx_mask = env.statics, env.conden_branch, env.gas_indx_mask
    n_0 = atm_step.M[:, None]
    sol_balanced_full = n_0 * ymix_new
    # With condensation on, only gas columns are rebalanced after Ros2.
    if conden_branch is not None:
        sol_balanced = jnp.where(gas_indx_mask[None, :], sol_balanced_full, sol_clip)
    else:
        sol_balanced = sol_balanced_full

    # Charge balance: with `charge_arr[i] = compo[i]['e']` for charged
    # species (and 0 for 'e' itself), `e[:] = -dot(y, charge_arr)`
    # enforces zero net charge per layer. Trace-time skip when off.
    if statics.use_ion:
        e_density = -jnp.einsum("zi,i->z", sol_balanced, statics.charge_arr)
        sol_balanced = sol_balanced.at[:, statics.e_idx].set(e_density)

    # y / ymix / atom_loss for the next iteration.
    y_next = jnp.where(do_accept, sol_balanced, s.y_prev)
    # A retry starts from the previous state: master restores var.y only
    # (op.py:2524-2531), so its retry masks with the rejected ymix (notes "Parity ledger").
    ymix_next = jnp.where(do_accept, ymix_new, s.ymix)
    atom_loss_next = jnp.where(accept, atom_loss_new, s.atom_loss)

    # y_prev is the revert target: on accept, the fresh state; on reject,
    # keep the carry's (still retrying this step).
    y_prev_next = jnp.where(do_accept, y_next, s.y_prev)
    atom_loss_prev_next = jnp.where(do_accept, atom_loss_next, s.atom_loss_prev)

    accept_count_next = s.accept_count + jnp.where(do_accept, jnp.int32(1), jnp.int32(0))
    retry_count_next = jnp.where(do_accept, jnp.int32(0), s.retry_count + jnp.int32(1))
    updates = {
        "y_prev": y_prev_next,
        "atom_loss": atom_loss_next,
        "atom_loss_prev": atom_loss_prev_next,
        "accept_count": accept_count_next,
        "retry_count": retry_count_next,
    }
    return y_next, ymix_next, updates


def _ring_write(statics: _Statics, s: JaxIntegState, y_next, t_next, accept_count_next, hold, defer_ring):
    """Append to the convergence ring; return `(ring_idx, y_slot, y_time_ring_new, t_time_ring_new, keep)`."""
    keep = None
    # Ring append (op.save_step). On reject this rewrites the same slot
    # with unchanged values. A batched lane the tick keeps (`hold`, or
    # this step went non-finite: `body_fn_batch`'s freeze rule) rewrites
    # its pre-step slot instead, and the freeze passes the ring through.
    # Batched (`defer_ring`), the y slot is returned, not written:
    # `body_fn_batch` writes it after its per-tick photolysis cond, since
    # a cond returning the written (conv_step, nz, ni) ring made XLA:GPU
    # rebuild and copy the whole ring every tick (notes "Efficiency
    # program").
    ring_idx = ring_slot(statics, accept_count_next)
    y_slot, t_slot = y_next, t_next
    if hold is not None:
        keep = hold | jnp.logical_not(jnp.all(jnp.isfinite(y_next)))
        y_slot = jnp.where(keep, s.y_time_ring[ring_idx], y_next)
        t_slot = jnp.where(keep, s.t_time_ring[ring_idx], t_next)
    y_time_ring_new = s.y_time_ring if defer_ring else s.y_time_ring.at[ring_idx].set(y_slot)
    t_time_ring_new = s.t_time_ring.at[ring_idx].set(t_slot)
    return ring_idx, y_slot, y_time_ring_new, t_time_ring_new, keep


def _evolution_append(statics: _Statics, s: JaxIntegState, y_next, t_next, do_accept, accept_count_next, hold, keep):
    """Capture this step for save_evolution; return the carry updates."""
    # save_evolution capture. Master appends every accepted step and
    # post-slices [::save_evo_frq], keeping accepted steps 1, K+1, 2K+1...
    # Gating on `(accept_count_next - 1) % K == 0` reproduces that (first
    # accepted step always captured). Compiles out when off.
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
    else:
        y_evo_new = s.y_evo
        t_evo_new = s.t_evo
        evo_idx_new = s.evo_idx
    return {"y_evo": y_evo_new, "t_evo": t_evo_new, "evo_idx": evo_idx_new}


def _update_longdy(
    statics: _Statics,
    s: JaxIntegState,
    y_next,
    ymix_next,
    t_next,
    t_time_ring_new,
    y_time_ring_new,
    accept_count_next,
    ring_idx,
    y_slot,
    do_accept,
    defer_ring,
):
    """Return `(longdy_next, longdydt_next, where_varies_most_next)` against the ring."""
    # Snapshot state with (y, t, time ring) updated, fed to _conv_jax,
    # with the y ring and the slot just written. Solo, that is the written
    # ring: a read of the pre-write ring after the in-place write made
    # XLA:CPU copy the whole (conv_step, nz, ni) ring twice an iteration
    # (notes "Efficiency program"). Batched (`defer_ring`) it is the
    # pre-write ring; the overlay gives the same values either way.
    s_for_conv = s._replace(
        y=y_next,
        ymix=ymix_next,
        t=t_next,
        t_time_ring=t_time_ring_new,
    )
    longdy_new_val, longdydt_new_val, where_varies_most_new = conv_jax(
        statics,
        s_for_conv,
        accept_count_next,
        s.y_time_ring if defer_ring else y_time_ring_new,
        ring_idx,
        y_slot,
    )
    # Only refresh longdy/longdydt on accepted steps; rejected steps
    # don't change the ring contents in any meaningful way.
    longdy_next = jnp.where(do_accept, longdy_new_val, s.longdy)
    longdydt_next = jnp.where(do_accept, longdydt_new_val, s.longdydt)
    where_varies_most_next = jnp.where(
        do_accept,
        where_varies_most_new,
        s.where_varies_most,
    )
    return longdy_next, longdydt_next, where_varies_most_next


def _certificate_candidate(
    statics: _Statics, s: JaxIntegState, longdy_next, longdydt_next, t_next, accept_count_next, do_accept
):
    """Return `(ready_after, candidate)`: whether this step may certify the run."""
    # Certificate candidate: `_convergence_ok` on this step's longdy. The
    # refresh below is forced on it, and geom_ok records whether
    # mu/g/Hp/dzi/Hpi moved less than geom_conv_tol. Hp here is the
    # pre-refresh value; the certificate re-reads the refreshed one next
    # iteration.
    s_cand = s._replace(longdy=longdy_next, longdydt=longdydt_next, t=t_next, accept_count=accept_count_next)
    is_conv_cand = _convergence_ok(s_cand, statics)
    ready_after = (t_next > jnp.float64(statics.trun_min)) & (accept_count_next > s.count_min_dyn)
    candidate = do_accept & is_conv_cand & ready_after
    return ready_after, candidate


# size-limit-ok: one call per part, in program order; the carry update names every field
def body_fn(
    env: RunnerEnv,
    s: JaxIntegState,
    atm_static_,
    lane_axis=None,
    it=None,
    hold=None,
    photo_take=None,
    defer_ring=False,
    matrix_free=True,
):
    """Advance the carry by one trial step.

    `lane_axis=None` is the single-profile trace. A string means the body runs under
    `jax.vmap(..., axis_name=lane_axis)` from `body_fn_batch`: the refresh cond then takes a lane-uniform predicate (a
    psum over the lanes, which vmap returns unmapped) so it lowers to a real `case` instead of both sides running on
    every lane, and each lane keeps its own gate as a `where` inside the taken branch. `it` is the batched loop's
    iteration tick (unmapped), which keys that cadence. Batched only: `hold` is the lanes the tick keeps whatever this
    step does (the ring writes fold it in), and `photo_take` the lanes that take photolysis now, None on a tick without
    any (`body_fn_batch` owns that cond).
    """
    statics = env.statics
    s = _photo_update(env, s, lane_axis, photo_take)
    atm_step = _splice_step_atm(atm_static_, s, statics.use_vm_mol, env.refresh_static.kb, env.refresh_static.Navo)
    sol, delta_arr = _ros2_solve(env, s, atm_step, matrix_free)
    sol, h2he_pinned_next, h2he_mix_next, zero_bot_dyn = _bottom_pins(statics, s, sol)
    sol_clip, ymix_new, small_y_inc, nega_y_inc = env.clip_fn(sol)
    atom_loss_new, delta = _loss_and_delta(env, s, sol, sol_clip, delta_arr, zero_bot_dyn)
    d = _decide_step(statics, s, sol, sol_clip, ymix_new, atom_loss_new, delta)
    sol_clip, ymix_new, trigger_fix, conden_updates = _conden_step(env, s, d.do_accept, d.y_post_clip, d.ymix_new)
    y_next, ymix_next, advance_updates = _rebalance(
        env, s, atm_step, sol_clip, ymix_new, d.do_accept, d.accept, atom_loss_new
    )
    accept_count_next = advance_updates["accept_count"]
    ring_idx, y_slot, y_time_ring_new, t_time_ring_new, keep = _ring_write(
        statics, s, y_next, d.t_next, accept_count_next, hold, defer_ring
    )
    evo_updates = _evolution_append(statics, s, y_next, d.t_next, d.do_accept, accept_count_next, hold, keep)
    longdy_next, longdydt_next, where_varies_most_next = _update_longdy(
        statics,
        s,
        y_next,
        ymix_next,
        d.t_next,
        t_time_ring_new,
        y_time_ring_new,
        accept_count_next,
        ring_idx,
        y_slot,
        d.do_accept,
        defer_ring,
    )
    ready_after, candidate = _certificate_candidate(
        statics, s, longdy_next, longdydt_next, d.t_next, accept_count_next, d.do_accept
    )
    geom_updates = _refresh_geometry(env, s, ymix_new, sol_clip, d.do_accept, candidate, it, lane_axis)
    budget_updates = _element_budget(statics, s, y_next, candidate)
    longdy_next, longdydt_next, hybrid_updates = _hybrid_flip(
        statics,
        s,
        longdy_next,
        longdydt_next,
        d.t_next,
        accept_count_next,
        geom_updates["Hp"],
        d.do_accept,
        ready_after,
    )
    rtol_next, loss_criteria = _adapt_rtol(statics, s, atom_loss_new, d.do_accept, trigger_fix)
    dt_next = _next_dt(statics, s, delta, rtol_next, d.accept, d.force_accept, d.next_dt_if_reject)
    photo_updates = _photo_switch(
        env, s, d.t_next, accept_count_next, hybrid_updates["count_min_dyn"], longdy_next, longdydt_next
    )
    s_next = s._replace(
        y=y_next,
        ymix=ymix_next,
        dt=dt_next,
        t=d.t_next,
        delta=delta,
        nega_count=s.nega_count + d.nega_count_inc,
        loss_count=s.loss_count + d.loss_count_inc,
        delta_count=s.delta_count + d.delta_count_inc,
        small_y=s.small_y + small_y_inc,
        nega_y=s.nega_y + nega_y_inc,
        y_time_ring=y_time_ring_new,
        t_time_ring=t_time_ring_new,
        longdy=longdy_next,
        longdydt=longdydt_next,
        where_varies_most=where_varies_most_next,
        rtol=rtol_next,
        loss_criteria=loss_criteria,
        h2he_pinned=h2he_pinned_next,
        h2he_mix=h2he_mix_next,
        **conden_updates,
        **advance_updates,
        **evo_updates,
        **geom_updates,
        **budget_updates,
        **hybrid_updates,
        **photo_updates,
    )
    return (s_next, ring_idx, y_slot) if defer_ring else s_next
