"""The forward-mode runner: integrates a state and its tangents together and certifies the tangent."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

from ._convergence import _longdy_reduce, _slope_min, lookback_index, real_terminate, two_branch
from ._loop_body import body_fn
from .phy_const import UNDERFLOW_DENOM

if TYPE_CHECKING:
    from collections.abc import Callable

    from ._carry import JaxIntegState, RunnerEnv, _Statics


def _tangent_conv(statics: _Statics, s_next: JaxIntegState, ds_next: JaxIntegState) -> Any:
    """Return the primal's longdy reduction on the tangent.

    `dy - dy_lookback` over `y`, on the cells the primal certifies on. Per cell and absolute, so the tangent
    direction is read in the units of a finite-difference step and 1% of y means what it means for the
    state. A zero tangent reads 0 (settled).

    The tangent leaves carry a leading direction axis, so this returns
    (tangent_longdy, tangent_longdydt) of shape (D,): one certificate per direction, over one
    primal.
    """
    indx = lookback_index(statics, s_next, s_next.accept_count)
    dt_lookback = jnp.maximum(s_next.t - s_next.t_time_ring[indx], UNDERFLOW_DENOM)

    def one_dir(dy: Any, dy_ring: Any) -> Any:
        tangent_longdy, _ = _longdy_reduce(
            s_next.y,
            s_next.ymix,
            s_next.y_time_ring[indx],
            s_next.pv.n_0,
            atol=statics.atol,
            mtol_conv=statics.mtol_conv,
            ignore_mask=statics.conver_ignore_mask[None, :],
            condense_mask=statics.condense_zero_conv_mask,
            diff=dy - dy_ring[indx],
        )
        return tangent_longdy, tangent_longdy / dt_lookback

    return jax.vmap(one_dir)(ds_next.y, ds_next.y_time_ring)


def _make_jvp_step(env: RunnerEnv, active_state: tuple[bool, ...], active_atm: tuple[bool, ...]) -> Callable:
    """One `jax.jvp` of `body_fn`, vmapped over a leading direction axis D.

    `active_*` flag, per flattened leaf of the state and AtmStatic, the
    leaves with a float tangent; only those go through the jvp. Every
    other leaf (ints, bools, flags) rides as `has_aux` output, so the loop
    holds no float0 tangents. The primal and aux outputs are unmapped
    (`out_axes=None`): the state, block assembly and factorisation are
    computed once for all D directions.
    """

    def _merge(flat: list, leaves: list, active: tuple[bool, ...]) -> list:
        nxt = iter(flat)
        return [next(nxt) if a else x for x, a in zip(leaves, active, strict=False)]

    def _step(s: JaxIntegState, atm_static: Any, ds_active: Any, datm_active: Any) -> Any:
        s_def = jax.tree_util.tree_structure(s)
        a_leaves, a_def = jax.tree_util.tree_flatten(atm_static)

        def f(s_act: Any, a_act: Any) -> Any:
            s_in = jax.tree_util.tree_unflatten(s_def, _merge(s_act, jax.tree_util.tree_leaves(s), active_state))
            a_in = jax.tree_util.tree_unflatten(a_def, _merge(a_act, a_leaves, active_atm))
            o_leaves = jax.tree_util.tree_leaves(body_fn(env, s_in, a_in)[0])
            act = [x for x, a in zip(o_leaves, active_state, strict=False) if a]
            rest = [x for x, a in zip(o_leaves, active_state, strict=False) if not a]
            return act, rest

        s_act = [x for x, a in zip(jax.tree_util.tree_leaves(s), active_state, strict=False) if a]
        a_act = [x for x, a in zip(a_leaves, active_atm, strict=False) if a]
        act_out, dact_out, rest_out = jax.vmap(
            lambda t_s, t_a: jax.jvp(f, (s_act, a_act), (t_s, t_a), has_aux=True),
            out_axes=(None, 0, None),
        )(list(ds_active), list(datm_active))
        it_a, it_r = iter(act_out), iter(rest_out)
        s_next = jax.tree_util.tree_unflatten(s_def, [next(it_a) if a else next(it_r) for a in active_state])
        # Tangent tree for the reduction: inactive leaves borrow the
        # primal (never read there).
        ds_next = jax.tree_util.tree_unflatten(s_def, _merge(dact_out, jax.tree_util.tree_leaves(s_next), active_state))
        return s_next, dact_out, ds_next

    return _step


def make_runner_jvp(env: RunnerEnv, active_state: tuple[bool, ...], active_atm: tuple[bool, ...]) -> Callable:
    """Build the forward-mode runner: integrates (state, tangent) together.

    Stops only when the tangent has settled by the primal's own rule.

    Carry: (state, active tangent leaves, tangent_longdy,
    tangent_longdydt); the last two are (D,) -- one per tangent
    direction -- and are refreshed on accepted steps as `longdy` is. The
    run stops when every direction has settled, so the predicate
    never depends on a direction axis and the primal carry never grows
    one.
    """
    statics = env.statics
    _step = _make_jvp_step(env, active_state, active_atm)

    @jax.jit
    def runner_jvp(state: Any, atm_static: Any, dstate_active: Any, datm_active: Any) -> Any:
        def body(carry: Any) -> Any:
            s, ds_active, tangent_longdy, tangent_longdydt = carry
            s_next, dact_out, ds_next = _step(s, atm_static, ds_active, datm_active)
            tangent_longdy_new, tangent_longdydt_new = _tangent_conv(statics, s_next, ds_next)
            accepted = s_next.retry_count == jnp.int32(0)
            return (
                s_next,
                dact_out,
                jnp.where(accepted, tangent_longdy_new, tangent_longdy),
                jnp.where(accepted, tangent_longdydt_new, tangent_longdydt),
            )

        def cond(carry: Any) -> Any:
            s, _ds, tangent_longdy, tangent_longdydt = carry
            real_term, _ = real_terminate(
                statics, s, jnp.all(two_branch(statics, tangent_longdy, tangent_longdydt, _slope_min(s)))
            )
            return jnp.logical_not(real_term)

        inf = jnp.full(dstate_active[0].shape[:1], jnp.float64(jnp.inf))
        final, dfinal_active, tangent_longdy, tangent_longdydt = jax.lax.while_loop(
            cond, body, (state, list(dstate_active), inf, inf)
        )
        tangent_ok = jnp.all(two_branch(statics, tangent_longdy, tangent_longdydt, _slope_min(final)))
        _real_term, reason = real_terminate(statics, final, tangent_ok)
        return (
            final._replace(termination_reason=reason),
            dfinal_active,
            tangent_longdy,
            tangent_ok,
        )

    return runner_jvp
