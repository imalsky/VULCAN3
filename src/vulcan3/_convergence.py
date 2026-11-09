"""The convergence test: the ring lookback, the longdy reduction, the two-branch test, the certificate, the stop test.

Public names: `conv_normal`, `certificate_failures`. Rules: notes "Certificates".
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from .phy_const import UNDERFLOW_DENOM
from .state import (
    _FROZEN_WHY,
    _NEARER_BALANCE,
    TERM_CONVERGED,
    TERM_DT_MIN_FROZEN,
    TERM_NONFINITE,
    TERM_RUNNING,
    TERM_RUNTIME,
    TERM_STEP_COUNT,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ._carry import JaxIntegState, _Statics
    from .config import Config

__all__ = ["certificate_failures", "conv_normal"]

# The loose convergence branch's slope threshold (s^-1), master's conv()
# (exoclime@80f75b9 op.py:1028-1029): min over the interfaces of
# Kzz / (0.1 Hp)^2, capped at 1e-8 and floored at 1e-10.
_SLOPE_MIN_HP_FRAC = 0.1

_SLOPE_MIN_CAP = 1e-8

_SLOPE_MIN_FLOOR = 1e-10

# Force-accepts at dt_min that keep the state before a run stops: the second repeats the first's trial.
_FROZEN_REPEATS = 2


def _frozen(s: JaxIntegState) -> jnp.ndarray:
    """Test whether the trial has failed at `dt_min`, keeping the state, `_FROZEN_REPEATS` times since a real accept."""
    frozen: jnp.ndarray = s.keep_prev_run >= jnp.int32(_FROZEN_REPEATS)
    return frozen


# In the certificate functions below, `c` is the runner's `_Statics` or a config: both carry yconv_cri, yconv_min,
# slope_cri and flux_cri under these names.
def _slope_min(s: JaxIntegState) -> jnp.ndarray:
    slope_min = jnp.minimum(
        jnp.min(s.pv.Kzz / (_SLOPE_MIN_HP_FRAC * s.Hp[:-1]) ** 2),
        jnp.float64(_SLOPE_MIN_CAP),
    )
    return jnp.maximum(slope_min, jnp.float64(_SLOPE_MIN_FLOOR))


def _branches(
    longdy: jnp.ndarray, longdydt: jnp.ndarray, slope_min: jnp.ndarray, c: _Statics | Config
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """(tight, loose): (yconv_cri, slope_cri) and (yconv_min, slope_min)."""
    tight = (longdy < jnp.float64(c.yconv_cri)) & (longdydt < jnp.float64(c.slope_cri))
    loose = (longdy < jnp.float64(c.yconv_min)) & (longdydt < slope_min)
    return tight, loose


def _convergence_ok(s: JaxIntegState, c: _Statics | Config) -> jnp.ndarray:
    """Convergence predicate shared by `real_terminate`, the certificate candidate and the hybrid phase-flip.

    `slope_min` is recomputed from the live Hp (atm refresh moves it). Same
    two-branch predicate as upstream (op.py:1060); shipped configs exit on
    the loose branch.
    """
    tight, loose = _branches(s.longdy, s.longdydt, _slope_min(s), c)
    ok: jnp.ndarray = (tight | loose) & (s.aflux_change < jnp.float64(c.flux_cri))
    return ok


def _certified(s: JaxIntegState, c: _Statics | Config) -> jnp.ndarray:
    """`_convergence_ok` AND the geometry and element-budget terms.

    `geom_ok` / `budget_ok` are written by the body on the candidate step (see there): the refreshed geometry agreed
    with the composition, and the column kept its elements since t=0. A column idling at `dt_min` is not one, though its
    state does not move and its longdy can read 0.
    """
    return _convergence_ok(s, c) & s.geom_ok & s.budget_ok & jnp.logical_not(_frozen(s))


def conv_normal(final: JaxIntegState, cfg: Config) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return the runner's convergence certificate on a carry, e.g. its exit state.

    The terms `real_terminate` ends a run on, read with `cfg`'s thresholds:
    tight OR loose branch, the photo-flux gate, and the geometry and
    element-budget terms, and not frozen at `dt_min`. Not its ready gate,
    hybrid-phase gate or non-finite exit; a forward-mode caller ANDs in `run_jvp`'s tangent
    certificate itself. On a finished non-hybrid run `ok` is
    `termination_reason == 1`: the body sets `geom_ok` / `budget_ok` only on
    a step past the ready gate.

    Parameters
    ----------
    final : JaxIntegState
        One column's carry, e.g. the runner's exit state.
    cfg : Config
        The run's config, for the thresholds.

    Returns
    -------
    ok : jnp.ndarray []
        bool, True when the certificate holds.
    branch : jnp.ndarray []
        int32: 1 when the tight branch certified, 2 the loose one, 0 when `ok` is False.
    """
    ok = _certified(final, cfg)
    tight, _ = _branches(final.longdy, final.longdydt, _slope_min(final), cfg)
    branch = jnp.where(ok, jnp.where(tight, jnp.int32(1), jnp.int32(2)), jnp.int32(0))
    return ok, branch


def certificate_failures(
    final: JaxIntegState, cfg: Config, atom_order: Sequence[str]
) -> list[tuple[str, float, float]]:
    """`(term, value, threshold)` for each certificate term the exit carry fails; empty when `conv_normal` holds.

    `budget_ok` / `geom_ok` are written only on a candidate step, so the
    budget is read from `budget_drift` (updated every accepted step), and
    geometry, which keeps no residual, is named (value nan) only when every
    other term passes.

    Parameters
    ----------
    final : JaxIntegState
        One column's exit carry, concrete.
    cfg : Config
        The run's config, for the thresholds.
    atom_order : sequence of str [n_atoms]
        Element names in `final.budget_drift` order, for the budget term's label.

    Returns
    -------
    list of tuple of (str, float, float)
        `(term, value, threshold)` per failed term.
    """
    slope_min = _slope_min(final)
    tight, loose = _branches(final.longdy, final.longdydt, slope_min, cfg)
    out = []
    if not bool(tight | loose):
        if not float(final.longdy) < float(cfg.yconv_min):
            out.append(("long dy", float(final.longdy), float(cfg.yconv_min)))
        else:
            out.append(("long dy/dt", float(final.longdydt), float(slope_min)))
    if not float(final.aflux_change) < float(cfg.flux_cri):
        out.append(("actinic flux change", float(final.aflux_change), float(cfg.flux_cri)))
    drift = np.abs(np.asarray(final.budget_drift))
    i = int(np.argmax(drift))
    if not drift[i] < float(cfg.element_budget_tol):
        out.append((f"element budget {atom_order[i]}/H", float(drift[i]), float(cfg.element_budget_tol)))
    if bool(_frozen(final)):
        out.append(("trial repeats at dt_min", float(final.keep_prev_run), float(_FROZEN_REPEATS)))
    if not out and not bool(final.geom_ok):
        out.append(("geometry", float("nan"), float(cfg.geom_conv_tol)))
    return out


def _longdy_reduce(
    y: jnp.ndarray,
    ymix: jnp.ndarray,
    y_old: jnp.ndarray,
    n_0: jnp.ndarray,
    *,
    atol: float,
    mtol_conv: float,
    ignore_mask: jnp.ndarray | None = None,
    condense_mask: jnp.ndarray | None = None,
    diff: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return the `longdy` convergence reduction as (longdy, ratio).

    `diff` replaces `y - y_old` while every mask still reads the primal
    (`y`, `ymix`): the tangent certificate passes `dy - dy_old` here, so the
    sensitivity is held to the same per-cell tolerance as the state.
    """
    longdy_arr = jnp.abs(((y - y_old) if diff is None else diff) / n_0[:, None])
    longdy_arr = jnp.where(ymix < mtol_conv, 0.0, longdy_arr)
    longdy_arr = jnp.where(y < atol, 0.0, longdy_arr)
    if ignore_mask is not None:
        longdy_arr = jnp.where(ignore_mask, 0.0, longdy_arr)
    if condense_mask is not None:
        longdy_arr = jnp.where(condense_mask, 0.0, longdy_arr)

    # Zeroed numerator -> denominator 1, for the same reason as in
    # `_make_delta_ratio_fn`: a sub-mtol_conv ymix would make the tangent
    # `0 * inf` and the max JVP does not discard it. Primal unchanged.
    den = jnp.where(longdy_arr == 0.0, 1.0, jnp.maximum(ymix, UNDERFLOW_DENOM))
    ratio = jnp.where(ymix > 0, longdy_arr / den, 0.0)
    longdy = jnp.max(ratio)
    # NaN guard: the masks above are all False for NaN, so an all-NaN state
    # would read longdy == 0.0 ("converged"). Force +inf so a poisoned run can
    # never converge, matching master's raise on an empty amax (op.py:1055).
    # `real_terminate` stops a non-finite `y` at once (reason 5); a
    # non-finite tangent runs to the count/runtime cap.
    state_is_bad = ~jnp.all(jnp.isfinite(y)) | ~jnp.all(jnp.isfinite(ymix))
    if diff is not None:
        state_is_bad = state_is_bad | ~jnp.all(jnp.isfinite(diff))
    return jnp.where(state_is_bad, jnp.inf, longdy), ratio


def ring_slot(statics: _Statics, accept_count: jnp.ndarray) -> jnp.ndarray:
    """Return the ring slot the accepted step number `accept_count` is written to (op.save_step).

    `(accept_count - 1) % conv_step`.
    """
    return jnp.mod(
        jnp.maximum(accept_count - jnp.int32(1), jnp.int32(0)),
        jnp.int32(statics.conv_step),
    )


def lookback_index(statics: _Statics, s: JaxIntegState, accept_count_after: jnp.ndarray) -> jnp.ndarray:
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
    """(longdy, longdydt, ratio) against the `lookback_index` slot.

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


def two_branch(statics: _Statics, longdy: jnp.ndarray, longdydt: jnp.ndarray, slope_min: jnp.ndarray) -> jnp.ndarray:
    """Tight (yconv_cri, slope_cri) OR loose (yconv_min, slope_min)."""
    tight, loose = _branches(longdy, longdydt, slope_min, statics)
    return tight | loose


def real_terminate(
    statics: _Statics, s: JaxIntegState, tangent_ok: bool | jnp.ndarray = True
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Termination predicate + reason code.

    Reason priority matches master's stop() (op.py:1065-1085): converged
    over runtime over step-count, so a step that is both converged and at
    a cap reports success. Codes: 0 running, 1 converged, 2 runtime
    exceeded, 3 step-count exceeded, 4 frozen at `dt_min` (which outranks
    those), 5 non-finite (which outranks the rest).

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
        # (reason 1). The non-finite exit below is a bypass and can still
        # return in phase 0.
        real_term = real_term & (s.hybrid_use_vm < jnp.float64(0.5))
    # A non-finite state can never recover, so stop at once with the
    # batched path's reason 5 (`body_fn_batch`) instead of burning the
    # step budget. A bypass exit: OR'd after the hybrid gate, and it
    # outranks every other code (nothing NaN converged or timed out).
    non_finite = jnp.logical_not(jnp.all(jnp.isfinite(s.y)))
    # A frozen column repeats one failing trial, and is the same kind of exit.
    frozen = _frozen(s)
    real_term = real_term | non_finite | frozen
    reason = jnp.where(
        non_finite,
        jnp.int32(TERM_NONFINITE),
        jnp.where(
            frozen,
            jnp.int32(TERM_DT_MIN_FROZEN),
            jnp.where(
                conv_term,
                jnp.int32(TERM_CONVERGED),
                jnp.where(
                    too_long,
                    jnp.int32(TERM_RUNTIME),
                    jnp.where(too_many, jnp.int32(TERM_STEP_COUNT), jnp.int32(TERM_RUNNING)),
                ),
            ),
        ),
    )
    return real_term, reason


def frozen_column_message(
    ratio: np.ndarray,
    sol_clip: np.ndarray,
    loss_change: np.ndarray,
    statics: _Statics,
    species: Sequence[str],
    atom_order: Sequence[str],
) -> str:
    """State why the repeated trial of a frozen column fails: its error estimate, a negative cell or an atom's loss."""
    z, i = (int(v) for v in np.unravel_index(np.argmax(ratio), ratio.shape))
    if ratio[z, i] > statics.rtol_accept:
        why = f"{species[i]} at layer {z} has error ratio {ratio[z, i]:.3g} against rtol {statics.rtol_accept:g}"
        remedy = f"raise `rtol` above the ratio, lower `dt_min`, or {_NEARER_BALANCE}"
    elif np.any(sol_clip < 0):
        z, i = (int(v) for v in np.unravel_index(np.argmin(sol_clip), sol_clip.shape))
        why = f"{species[i]} at layer {z} goes to {sol_clip[z, i]:.3g} cm^-3, below nega_cut {statics.nega_cut:g}"
        remedy = f"lower `nega_cut`, or {_NEARER_BALANCE}"
    else:
        a = int(np.argmax(loss_change))
        why = f"{atom_order[a]}'s atom loss moves {loss_change[a]:.3g} against loss_eps {statics.loss_eps:g}"
        remedy = f"raise `loss_eps`, or {_NEARER_BALANCE}"
    return f"Column frozen at dt_min = {statics.dt_min:g} s: {why}. {_FROZEN_WHY} To try: {remedy}."


def cond_fn(statics: _Statics, s: JaxIntegState) -> jnp.ndarray:
    """Continue while the run has not terminated."""
    real_term, _ = real_terminate(statics, s)
    return jnp.logical_not(real_term)
