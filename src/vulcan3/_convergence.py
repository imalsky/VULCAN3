"""The convergence test: the longdy reduction, the two-branch test and the run's certificate.

Public names: `conv_normal`, `certificate_failures`. Rules: notes "Certificates".
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from .phy_const import UNDERFLOW_DENOM

if TYPE_CHECKING:
    from ._carry import JaxIntegState

__all__ = ["certificate_failures", "conv_normal"]

# The loose convergence branch's slope threshold (s^-1), master's conv()
# (exoclime@80f75b9 op.py:1028-1029): min over the interfaces of
# Kzz / (0.1 Hp)^2, capped at 1e-8 and floored at 1e-10.
_SLOPE_MIN_HP_FRAC = 0.1

_SLOPE_MIN_CAP = 1e-8

_SLOPE_MIN_FLOOR = 1e-10


# The convergence certificate. `c` is the runner's `_Statics` or a config:
# both carry yconv_cri, yconv_min, slope_cri and flux_cri under these names.
def _slope_min(s: JaxIntegState):
    slope_min = jnp.minimum(
        jnp.min(s.pv.Kzz / (_SLOPE_MIN_HP_FRAC * s.Hp[:-1]) ** 2),
        jnp.float64(_SLOPE_MIN_CAP),
    )
    return jnp.maximum(slope_min, jnp.float64(_SLOPE_MIN_FLOOR))


def _branches(longdy, longdydt, slope_min, c):
    """(tight, loose): (yconv_cri, slope_cri) and (yconv_min, slope_min)."""
    tight = (longdy < jnp.float64(c.yconv_cri)) & (longdydt < jnp.float64(c.slope_cri))
    loose = (longdy < jnp.float64(c.yconv_min)) & (longdydt < slope_min)
    return tight, loose


def _convergence_ok(s: JaxIntegState, c):
    """Convergence predicate shared by `_real_terminate`, the certificate candidate and the hybrid phase-flip.

    `slope_min` is recomputed from the live Hp (atm refresh moves it). Same
    two-branch predicate as upstream (op.py:1060); shipped configs exit on
    the loose branch.
    """
    tight, loose = _branches(s.longdy, s.longdydt, _slope_min(s), c)
    return (tight | loose) & (s.aflux_change < jnp.float64(c.flux_cri))


def _certified(s: JaxIntegState, c):
    """`_convergence_ok` AND the geometry and element-budget terms.

    `geom_ok` / `budget_ok` are written by the body on the candidate step (see there): the refreshed geometry agreed
    with the composition, and the column kept its elements since t=0.
    """
    return _convergence_ok(s, c) & s.geom_ok & s.budget_ok


def conv_normal(final: JaxIntegState, cfg):
    """Return the runner's convergence certificate on a carry, e.g. its exit state.

    The terms `_real_terminate` ends a run on, read with `cfg`'s thresholds:
    tight OR loose branch, the photo-flux gate, and the geometry and
    element-budget terms. Not its ready gate, hybrid-phase gate or
    non-finite exit; a forward-mode caller ANDs in `run_jvp`'s tangent
    certificate itself. On a finished non-hybrid run `ok` is
    `termination_reason == 1`: the body sets `geom_ok` / `budget_ok` only on
    a step past the ready gate.

    Returns `(ok, branch)`: a () bool, and a () int32 that is 1 when the
    tight branch certified, 2 the loose one, 0 when `ok` is False.
    """
    ok = _certified(final, cfg)
    tight, _ = _branches(final.longdy, final.longdydt, _slope_min(final), cfg)
    branch = jnp.where(ok, jnp.where(tight, jnp.int32(1), jnp.int32(2)), jnp.int32(0))
    return ok, branch


def certificate_failures(final: JaxIntegState, cfg, atom_order):
    """`(term, value, threshold)` for each certificate term the exit carry fails; empty when `conv_normal` holds.

    `budget_ok` / `geom_ok` are written only on a candidate step, so the
    budget is read from `budget_drift` (updated every accepted step), and
    geometry, which keeps no residual, is named (value nan) only when every
    other term passes.
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
    if not out and not bool(final.geom_ok):
        out.append(("geometry", float("nan"), float(cfg.geom_conv_tol)))
    return out


def _longdy_reduce(
    y,
    ymix,
    y_old,
    n_0,
    *,
    atol,
    mtol_conv,
    ignore_mask=None,
    condense_mask=None,
    diff=None,
):
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
    # `_make_aggregate_delta_fn`: a sub-mtol_conv ymix would make the tangent
    # `0 * inf` and the max JVP does not discard it. Primal unchanged.
    den = jnp.where(longdy_arr == 0.0, 1.0, jnp.maximum(ymix, UNDERFLOW_DENOM))
    ratio = jnp.where(ymix > 0, longdy_arr / den, 0.0)
    longdy = jnp.max(ratio)
    # NaN guard: the masks above are all False for NaN, so an all-NaN state
    # would read longdy == 0.0 ("converged"). Force +inf so a poisoned run can
    # never converge, matching master's raise on an empty amax (op.py:1055).
    # `_real_terminate` stops a non-finite `y` at once (reason 5); a
    # non-finite tangent runs to the count/runtime cap.
    state_is_bad = ~jnp.all(jnp.isfinite(y)) | ~jnp.all(jnp.isfinite(ymix))
    if diff is not None:
        state_is_bad = state_is_bad | ~jnp.all(jnp.isfinite(diff))
    return jnp.where(state_is_bad, jnp.inf, longdy), ratio
