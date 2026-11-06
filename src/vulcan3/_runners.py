"""Build the runners: solo, chunked and batched loops over the step, plus the forward-mode and queue entry points."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

from ._carry import (
    _ATM_STATIC_BATCH_AXES,
    _CONV_RING_FIELDS,
    _EVO_RING_FIELDS,
    _LANE_AXIS,
    JaxIntegState,
    _PhotoStatic,
    _Statics,
)
from ._loop_body import body_fn
from ._loop_env import RunnerEnv, make_env
from ._runner_jvp import make_runner_jvp
from ._runner_queue import runner_queue
from ._termination import cond_fn, real_terminate
from .state import TERM_NONFINITE, TERM_RUNNING

if TYPE_CHECKING:
    from . import atm_refresh as _atm_refresh_mod, conden as _conden_mod
    from .jax_step import AtmStatic

# --- batched freeze ------------------------------------------------------
# The batched body advances every lane and then keeps the pre-step carry on
# lanes that are done. Selecting every carry leaf whole costs a full copy of
# the carry per iteration, so `_freeze_leaf` skips the leaves the step passed
# through, and the ring buffers below are never selected: `body_fn` writes
# their one slot with the freeze already applied (its `hold`). A freeze that
# read the pre-step ring after that write would keep two rings live, and XLA
# would copy the whole (conv_step, nz, ni) leaf twice per iteration.


def _freeze_leaf(old: Any, new: Any, keep_old: Any) -> Any:
    """`where(keep_old, old, new)`.

    Skipped when the step passed the leaf through unchanged (the same object, so the select is a copy for nothing).
    """
    return old if old is new else jnp.where(keep_old, old, new)


def _freeze_state(s: JaxIntegState, s_adv: JaxIntegState, keep_old: Any) -> Any:
    """Keep the pre-step carry `s` where `keep_old`, the advanced carry elsewhere.

    Leaf by leaf, so untouched leaves are not re-selected. The ring buffers come back from `body_fn` already frozen
    (its `hold`) and pass through.
    """
    out = {}
    for f in JaxIntegState._fields:
        old, new = getattr(s, f), getattr(s_adv, f)
        if f in _CONV_RING_FIELDS + _EVO_RING_FIELDS:
            out[f] = new
        elif f == "pv":
            out[f] = jax.tree_util.tree_map(lambda o, n: _freeze_leaf(o, n, keep_old), old, new)
        else:
            out[f] = _freeze_leaf(old, new, keep_old)
    return s._replace(**out)


def body_fn_batch(env: RunnerEnv, s: JaxIntegState, atm_static_: Any, lane_axis: Any = None, it: Any = None) -> Any:
    """Advance every lane one tick and freeze the ones that are done."""
    statics = env.statics
    # vmap applies the body to every lane each iteration until the slowest
    # finishes, so finished lanes must be frozen: advance all lanes, then
    # keep the pre-step carry `s` for lanes that are done / terminate now /
    # went non-finite. Freezing on `s` keeps a finished lane's carry
    # unchanged while the slowest lane runs.
    real_term, reason = real_terminate(statics, s)
    already_done = s.is_done
    hold = already_done | real_term

    def tick(photo_take: Any) -> Any:
        def advance(operand: Any) -> Any:
            ss, y_ring = operand
            ss = ss._replace(y_time_ring=y_ring)
            s_adv, ring_idx, y_slot = body_fn(
                env, ss, atm_static_, lane_axis=lane_axis, it=it, hold=hold, photo_take=photo_take, defer_ring=True
            )
            nan_now = jnp.logical_not(jnp.all(jnp.isfinite(s_adv.y)))
            # Non-finite lanes freeze at the last good carry `ss` (reason
            # 5); the bad `s_adv` is discarded.
            became_nan = nan_now & ~hold
            keep_old = hold | became_nan
            frozen = _freeze_state(ss, s_adv, keep_old)
            reason_next = jnp.where(
                already_done,
                ss.termination_reason,
                jnp.where(
                    real_term,
                    reason,
                    jnp.where(became_nan, jnp.int32(TERM_NONFINITE), jnp.int32(TERM_RUNNING)),
                ),
            )
            frozen = frozen._replace(is_done=keep_old, termination_reason=reason_next, y_time_ring=None)
            return frozen, ring_idx, y_slot

        return advance

    # The y ring enters the step read-only and is written after it
    # (body_fn's `defer_ring`).
    operand = (s._replace(y_time_ring=None), s.y_time_ring)

    def write_ring(out: Any) -> Any:
        s_out, ring_idx, y_slot = out
        return s_out._replace(y_time_ring=s.y_time_ring.at[ring_idx].set(y_slot))

    if env.photo_branch is None:
        return write_ring(tick(None)(operand))
    # Batched runs key the photolysis cadence to the shared tick (per-lane
    # counts dephase and would fire every iteration); the retry gate keeps
    # updates on accepted states and `live` keeps frozen lanes out. The
    # cond spans the step and its freeze, not the photolysis update alone:
    # a lane that goes non-finite falls back to its pre-step photolysis
    # arrays, and a cond output may not overwrite an operand that is still
    # read after it, so a cond ending before the step would copy those
    # arrays on every tick without photolysis. Here they are read only
    # inside the branch that changed them; the other branch passes them
    # through untouched.
    live = jnp.logical_not(already_done)
    photo_take = (jnp.mod(it, s.update_photo_frq) == jnp.int32(0)) & (s.retry_count == jnp.int32(0)) & live
    any_photo = jax.lax.psum(photo_take.astype(jnp.int32), lane_axis) > 0
    return write_ring(jax.lax.cond(any_photo, tick(photo_take), tick(None), operand))


# size-limit-ok: the runner closures share one environment
def make_runner(
    net: Any,
    statics: _Statics,
    non_gas_present: bool,
    gas_indx_mask: jnp.ndarray,
    zero_bot_row: bool,
    condense_zero_mask: jnp.ndarray,
    start_conden_time: float,
    stop_conden_time: float,
    refresh_static: _atm_refresh_mod.AtmRefreshStatic,
    photo_static: _PhotoStatic | None = None,
    conden_static: _conden_mod.CondenStatic | None = None,
) -> Any:
    """Build `(runner, runner_chunk, runner_batch, runner_queue, make_runner_jvp)`.

    The JIT'd `runner(state, atm_static) -> state` runs to convergence, `count_max`, or `runtime`, and
    `runner_chunk(state, atm_static, target)` also stops when `accept_count` reaches `target`. `cond_fn` checks
    `(t > runtime_dyn) | (count > count_max_dyn) | (ready & certified)` (phase 1 only on hybrid runs), or a non-finite
    `y`.
    The termination test reads count_min, count_max and runtime from the carry's `*_dyn` fields (the hybrid flip extends
    them), never from `statics`.
    """
    env = make_env(
        net,
        statics,
        non_gas_present,
        gas_indx_mask,
        zero_bot_row,
        condense_zero_mask,
        start_conden_time,
        stop_conden_time,
        refresh_static,
        photo_static,
        conden_static,
    )

    @jax.jit
    def runner(state: JaxIntegState, atm_static: AtmStatic) -> Any:
        # One-shot run to convergence / count_max / runtime.
        final = jax.lax.while_loop(
            functools.partial(cond_fn, statics),
            lambda s: body_fn(env, s, atm_static),
            state,
        )
        # The single-profile body never writes termination_reason; evaluate
        # it once on the terminal state.
        _real_term, reason = real_terminate(statics, final)
        return final._replace(termination_reason=reason)

    @jax.jit
    def runner_chunk(state: JaxIntegState, atm_static: AtmStatic, target: Any) -> Any:
        # `runner` cut off once `accept_count` reaches `target`, so a host
        # loop can print progress and take Ctrl+C between chunks. Same body
        # and carry: resuming from the returned state continues the run.
        # Never differentiated (`__call__` is not an AD entry point), so the
        # step takes the dense stage operator: the matrix-free one's RHS jvp
        # costs seconds of tracing and nothing in the primal.
        final = jax.lax.while_loop(
            lambda s: cond_fn(statics, s) & (s.accept_count < target),
            lambda s: body_fn(env, s, atm_static, matrix_free=False),
            state,
        )
        _real_term, reason = real_terminate(statics, final)
        return final._replace(termination_reason=reason)

    # One batched iteration over the lanes at the shared tick `it`.
    lane_step = jax.vmap(
        lambda it, s, a: body_fn_batch(env, s, a, lane_axis=_LANE_AXIS, it=it),
        in_axes=(None, 0, _ATM_STATIC_BATCH_AXES),
        axis_name=_LANE_AXIS,
    )

    def runner_batch(state_b: JaxIntegState, atm_b: AtmStatic) -> Any:
        # Freeze-on-done loop on stacked inputs; the caller jits it. The
        # while loop sits above the vmap, so photo/refresh conds key on the
        # shared iteration tick: batched and solo runs agree at the
        # convergence scale, not bitwise. The loop gates only on the carry's
        # is_done, never re-derives `real_term`: the body must run one final
        # (frozen) iteration on a terminal lane to record is_done /
        # termination_reason.
        _, final_b = jax.lax.while_loop(
            lambda c: jnp.any(jnp.logical_not(c[1].is_done)),
            lambda c: (c[0] + jnp.int32(1), lane_step(c[0], c[1], atm_b)),
            (jnp.int32(0), state_b),
        )
        return final_b

    return (
        runner,
        runner_chunk,
        runner_batch,
        functools.partial(runner_queue, env, lane_step),
        functools.partial(make_runner_jvp, env),
    )
