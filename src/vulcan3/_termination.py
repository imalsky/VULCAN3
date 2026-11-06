"""The loop's termination predicate and the longdy statistics it reads from the convergence ring."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

from ._convergence import _branches, _certified, _longdy_reduce
from .phy_const import UNDERFLOW_DENOM
from .state import TERM_CONVERGED, TERM_NONFINITE, TERM_RUNNING, TERM_RUNTIME, TERM_STEP_COUNT

if TYPE_CHECKING:
    from ._carry import JaxIntegState, _Statics


def ring_slot(statics: _Statics, accept_count: jnp.ndarray) -> Any:
    """Return the ring slot the accepted step number `accept_count` is written to (op.save_step).

    `(accept_count - 1) % conv_step`.
    """
    return jnp.mod(
        jnp.maximum(accept_count - jnp.int32(1), jnp.int32(0)),
        jnp.int32(statics.conv_step),
    )


def lookback_index(statics: _Statics, s: JaxIntegState, accept_count_after: jnp.ndarray) -> Any:
    """Ring slot closest in time to `t * st_factor`, never the newest."""
    target_t = s.t * statics.st_factor
    diffs = jnp.abs(s.t_time_ring - target_t)
    # Bump the most-recent slot to +inf so argmin can never pick it,
    # matching VULCAN's `if indx == count-1: indx-=1` guard.
    last_idx = ring_slot(statics, accept_count_after)
    big = jnp.float64(jnp.inf)
    diffs_guarded = diffs.at[last_idx].set(big)
    return jnp.argmin(diffs_guarded)


def conv_jax(
    statics: _Statics,
    s: JaxIntegState,
    accept_count_after: jnp.ndarray,
    y_ring: jnp.ndarray,
    ring_idx: jnp.ndarray,
    y_slot: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """(longdy, longdydt, ratio) against the `_lookback_index` slot.

    `s` carries this step's time ring. The y lookback reads `y_ring` with
    this step's slot overlaid: the written ring's values, also when
    `y_ring` is the pre-write ring (`body_fn_batch` writes it after its
    cond).

    On the first iteration the ring is all-zero, which yields
    longdy ≈ |y/n_0| ~ O(1); the `ready` gate prevents acting on it.
    """
    indx = lookback_index(statics, s, accept_count_after)

    longdy_new, ratio = _longdy_reduce(
        s.y,
        s.ymix,
        jnp.where(indx == ring_idx, y_slot, y_ring[indx]),
        s.pv.n_0,
        atol=statics.atol,
        mtol_conv=statics.mtol_conv,
        ignore_mask=statics.conver_ignore_mask[None, :],
        condense_mask=statics.condense_zero_conv_mask,
    )
    dt_lookback = jnp.maximum(s.t - s.t_time_ring[indx], UNDERFLOW_DENOM)
    longdydt_new = longdy_new / dt_lookback
    return longdy_new, longdydt_new, ratio


def two_branch(statics: _Statics, longdy: jnp.ndarray, longdydt: jnp.ndarray, slope_min: Any) -> Any:
    """Tight (yconv_cri, slope_cri) OR loose (yconv_min, slope_min)."""
    tight, loose = _branches(longdy, longdydt, slope_min, statics)
    return tight | loose


def real_terminate(statics: _Statics, s: JaxIntegState, tangent_ok: bool = True) -> Any:
    """Termination predicate + reason code.

    Reason priority matches master's stop() (op.py:1065-1085): converged
    over runtime over step-count, so a step that is both converged and at
    a cap reports success. Codes: 0 running, 1 converged, 2 runtime
    exceeded, 3 step-count exceeded, 5 non-finite (which outranks the
    rest); 4 is unassigned.

    `tangent_ok` is the sensitivity certificate of `runner_jvp`; the
    primal runner passes the default, which folds away at trace time.
    """
    # Live budget (equals the static count_max/runtime for non-hybrid runs;
    # the hybrid phase flip resets it for phase 1).
    too_long = s.t > s.runtime_dyn
    too_many = s.accept_count > s.count_max_dyn

    ready = (s.t > jnp.float64(statics.trun_min)) & (s.accept_count > s.count_min_dyn)
    conv_term = ready & _certified(s, statics) & tangent_ok
    real_term = too_long | too_many | conv_term
    if statics.hybrid_vm_mol:
        # Phase 0 (upwind) never terminates here: the body flips to phase 1
        # (central difference) and extends the budget instead (vm_branch
        # stop()). A run stopping through this predicate is in phase 1 --
        # a central-difference fixed point only if phase 1 converged
        # (reason 1). The non-finite exit
        # below is a bypass and can still return in phase 0.
        real_term = real_term & (s.hybrid_use_vm < jnp.float64(0.5))
    # A non-finite state can never recover, so stop at once with the
    # batched path's reason 5 (`body_fn_batch`) instead of burning the
    # step budget. A bypass exit: OR'd after the hybrid gate, and it
    # outranks every other code (nothing NaN converged or timed out).
    non_finite = jnp.logical_not(jnp.all(jnp.isfinite(s.y)))
    real_term = real_term | non_finite
    reason = jnp.where(
        non_finite,
        jnp.int32(TERM_NONFINITE),
        jnp.where(
            conv_term,
            jnp.int32(TERM_CONVERGED),
            jnp.where(
                too_long,
                jnp.int32(TERM_RUNTIME),
                jnp.where(too_many, jnp.int32(TERM_STEP_COUNT), jnp.int32(TERM_RUNNING)),
            ),
        ),
    )
    return real_term, reason


def cond_fn(statics: _Statics, s: JaxIntegState) -> Any:
    """Continue while the run has not terminated."""
    real_term, _ = real_terminate(statics, s)
    return jnp.logical_not(real_term)
