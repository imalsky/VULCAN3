"""Step control: the step size, the per-step clip and the aggregate truncation error."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from .phy_const import UNDERFLOW_DENOM


def _step_size(
    dt: jnp.ndarray,
    delta: jnp.ndarray,
    rtol: float,
    dt_var_min: float,
    dt_var_max: float,
    dt_min: float,
    dt_max: float,
    safety: float,
    zero_delta_frac: float,
) -> jnp.ndarray:
    """Adaptive Ros2 dt update. Returns the next dt (scalar, seconds).

    I-control (master-faithful, the only dt control):
    `h_factor = clip(safety * (rtol/delta)^0.5, dt_var_min, dt_var_max)`,
    `h_new = clip(dt * h_factor, dt_min, dt_max)`; `delta == 0` substitutes
    `zero_delta_frac * rtol` (cfg.step_size_safety /
    cfg.step_size_zero_delta_frac).
    """
    delta_eff = jnp.where(delta < UNDERFLOW_DENOM, zero_delta_frac * rtol, delta)
    h_factor = safety * (rtol / delta_eff) ** 0.5
    h_factor = jnp.clip(h_factor, dt_var_min, dt_var_max)
    return jnp.clip(dt * h_factor, dt_min, dt_max)


def _clip_prologue(y_in: Any, pos_cut: float, nega_cut: float) -> Any:
    """Return `(y_clip, small_y_inc, nega_y_inc)`: the per-step clip of both `_make_clip_fn` branches.

    Values in [nega_cut, pos_cut) go to 0. A value below nega_cut stays
    negative, so `all_nonneg` rejects the step (notes "Parity ledger"). Master's
    second rule (op.py:2459 on the post-solve ymix, op.py:2991-2993) zeroes
    every negative, so its negativity test can never fire.
    """
    small_y_inc = jnp.sum(jnp.where((y_in < pos_cut) & (y_in >= 0), jnp.abs(y_in), 0.0))
    nega_y_inc = jnp.sum(jnp.where((y_in > nega_cut) & (y_in <= 0), jnp.abs(y_in), 0.0))
    y_clip = jnp.where((y_in < pos_cut) & (y_in >= nega_cut), 0.0, y_in)
    return y_clip, small_y_inc, nega_y_inc


def _make_clip_fn(
    non_gas_present: bool,
    gas_indx_mask: jnp.ndarray,
    pos_cut: float,
    nega_cut: float,
) -> Any:
    """Build a `clip(y_in) -> (y_clip, ymix_new, small_inc, nega_inc)` closure.

    `non_gas_present` selects between the gas-only / total `ysum` denominators
    at closure time so the traced body keeps a single branch. The two
    branches use different divide guards; see each.
    """
    if non_gas_present:
        gas_mask_2d = gas_indx_mask  # (ni,) bool

        def clip_fn(y_in: Any) -> Any:
            y_clip, small_y_inc, nega_y_inc = _clip_prologue(y_in, pos_cut, nega_cut)
            ysum = jnp.sum(jnp.where(gas_mask_2d[None, :], y_clip, 0.0), axis=1, keepdims=True)
            # ysum==0 on an all-clipped gas layer (master divides unguarded):
            # return 0 there. `where`, since a condensate over a floored ysum
            # would overflow. A non-finite ysum stays non-finite, so a NaN
            # solve still ends the run.
            ymix_new = jnp.where(ysum > 0, y_clip / ysum, jnp.where(jnp.isfinite(ysum), 0.0, jnp.nan))
            return y_clip, ymix_new, small_y_inc, nega_y_inc
    else:

        def clip_fn(y_in: Any) -> Any:
            y_clip, small_y_inc, nega_y_inc = _clip_prologue(y_in, pos_cut, nega_cut)
            # No condensate column exists here, so the numerator cannot
            # overflow a tiny denominator: flooring an all-zero layer's ysum
            # is enough (0/0 guard), and it is a no-op on the physical path
            # where ysum ~ n_0 >> 1e-300.
            ysum = jnp.sum(y_clip, axis=1, keepdims=True)
            ysum = jnp.maximum(ysum, UNDERFLOW_DENOM)
            return y_clip, y_clip / ysum, small_y_inc, nega_y_inc

    return clip_fn


def _make_aggregate_delta_fn(mtol: float, atol: float, zero_bot_row: bool, condense_zero_mask: jnp.ndarray) -> Any:
    """Build the scalar `delta(sol, delta_arr, ymix_old, zero_bot, ymix_trial)` reducer.

    Zeros entries where both the pre-step and the trial (pre-clip) mixing ratio
    are below mtol (master masks on the pre-step one alone, op.py:2949, so a
    species the step creates escapes the estimate: notes "Parity ledger"),
    sol < atol, optionally the bottom row (use_botflux / use_fix_sp_bot),
    and condensed species, then returns the max of (delta_arr / sol) over
    (sol > 0) cells.
    """
    cond_zero = jnp.asarray(condense_zero_mask, dtype=jnp.bool_)

    def agg(
        sol: jnp.ndarray, delta_arr: jnp.ndarray, ymix_old: Any, zero_bot: Any = None, ymix_trial: Any = None
    ) -> Any:
        dead = ymix_old < mtol
        if ymix_trial is not None:
            dead = dead & (ymix_trial < mtol)
        masked = jnp.where(dead, 0.0, delta_arr)
        masked = jnp.where(sol < atol, 0.0, masked)
        # `zero_bot` is the runtime extension of the static flag: upstream's
        # use_fix_H2He trip turns `use_fix_sp_bot` non-empty mid-run, which
        # activates its `delta[0] = 0` for the rest of the run.
        zb = jnp.bool_(zero_bot_row)
        if zero_bot is not None:
            zb = zb | zero_bot
        masked = jnp.where(zb, masked.at[0].set(0.0), masked)
        masked = jnp.where(cond_zero, 0.0, masked)
        # A zeroed numerator takes denominator 1: dividing it by a sub-atol
        # density gives a `0 * den**-2 = 0 * inf` NaN tangent that the
        # `jnp.max` JVP (a multiply by 0/1, not a select) propagates. Primal
        # unchanged (0/x == 0/1); a live numerator implies den >= atol.
        den = jnp.where(masked == 0.0, 1.0, jnp.maximum(jnp.abs(sol), UNDERFLOW_DENOM))
        ratio = jnp.where(sol > 0, masked / den, 0.0)
        return jnp.max(ratio)

    return agg
