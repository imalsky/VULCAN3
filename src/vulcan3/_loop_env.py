"""What the step closes over: the network, statics, clip, delta and branches of one runner."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

from ._loop_branches import _make_conden_branch, _make_photo_branch
from ._step_control import _make_aggregate_delta_fn, _make_clip_fn

if TYPE_CHECKING:
    import jax.numpy as jnp

    from . import atm_refresh as _atm_refresh_mod, conden as _conden_mod
    from ._carry import _PhotoStatic, _Statics


class RunnerEnv(NamedTuple):
    """What every part of the step closes over: built once per runner, read at trace time."""

    net: Any
    statics: _Statics
    non_gas_present: bool
    gas_indx_mask: jnp.ndarray
    start_conden_time: float
    stop_conden_time: float
    refresh_static: _atm_refresh_mod.AtmRefreshStatic
    clip_fn: Any
    agg_delta_fn: Any
    photo_branch: Any
    conden_branch: Any


def make_env(
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
) -> RunnerEnv:
    """Build the step's closure inputs; a branch is None when its process is off."""
    return RunnerEnv(
        net,
        statics,
        non_gas_present,
        gas_indx_mask,
        start_conden_time,
        stop_conden_time,
        refresh_static,
        _make_clip_fn(non_gas_present, gas_indx_mask, statics.pos_cut, statics.nega_cut),
        _make_aggregate_delta_fn(statics.mtol, statics.atol, zero_bot_row, condense_zero_mask),
        _make_photo_branch(photo_static) if photo_static is not None else None,
        _make_conden_branch(conden_static) if conden_static is not None else None,
    )
