"""Step-control corrections of notes "Parity ledger", one unit check each.

Master (op.py:2949) masks the error estimate on the pre-step mixing ratio alone, so a species the step creates escapes
it; its clip (op.py:2459) zeroes every negative value, so the negativity test of step_ok can never fire. VULCAN-JAX
masks a species only when both its pre-step and its pre-clip trial mixing ratio are below mtol, zeroes every finite
negative as master does but rejects a step that drives a populated gas negative (the estimate cannot see it), and keeps
a non-finite gas sum non-finite.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def test_estimate_counts_a_species_the_step_creates():
    from vulcan3._step_control import _make_aggregate_delta_fn

    agg = _make_aggregate_delta_fn(1e-22, 1e-30, False, jnp.zeros((1, 2), dtype=bool))
    ymix_old = jnp.asarray([[1.0, 0.0]])  # species 1 absent before the step
    sol = jnp.asarray([[1.0, 1e-6]])  # ... and created by it
    delta_arr = jnp.asarray([[0.0, 1e-6]])  # with a 100 % error
    trial = sol / jnp.sum(sol, axis=1, keepdims=True)
    assert float(agg(sol, delta_arr, ymix_old, None, trial)) == 1.0
    # absent before and after: masked
    sol0 = jnp.asarray([[1.0, 1e-25]])
    trial0 = sol0 / jnp.sum(sol0, axis=1, keepdims=True)
    assert float(agg(sol0, jnp.asarray([[0.0, 1e-25]]), ymix_old, None, trial0)) == 0.0


def test_clip_zeroes_every_finite_negative_and_counts_it():
    from vulcan3._step_control import _clip_prologue

    y = jnp.asarray([[5.0, 0.5, -0.5, -2.0e8, -jnp.inf, jnp.nan]])
    y_clip, small, nega = _clip_prologue(y, pos_cut=1.0)
    y_clip = np.asarray(y_clip)
    np.testing.assert_array_equal(y_clip[0, :4], [5.0, 0.0, 0.0, 0.0])
    assert y_clip[0, 4] == -np.inf  # a non-finite solve is never zeroed
    assert np.isnan(y_clip[0, 5])
    assert float(small) == 0.5
    assert float(nega) == 0.5 + 2.0e8
    # The adjoint audit's mask of the cells the clip owns follows the same rule.
    from types import SimpleNamespace

    from vulcan3._adjoint_core import _clip_dead_mask

    cfg = SimpleNamespace(pos_cut=1.0, mtol=1e-16, atol=1e-1)
    y_old = np.asarray([[5.0, 0.5, 0.0, 0.0, 1.0, 1.0]])  # an empty cell at index 3, so its undershoot is the clip's
    dead = _clip_dead_mask(np.asarray(y), y_old, np.ones(6, dtype=bool), cfg)
    np.testing.assert_array_equal(dead, [[False, True, True, True, False, False]])
    # A populated gas driven negative is rejected by the runner, not zeroed: it stays in the audit's scan.
    y_pop = np.asarray([[5.0, 0.5, 0.0, 3.0, 1.0, 1.0]])
    dead_pop = _clip_dead_mask(np.asarray(y), y_pop, np.ones(6, dtype=bool), cfg)
    np.testing.assert_array_equal(dead_pop, [[False, True, True, False, False, False]])


def test_only_a_populated_gas_going_negative_rejects_the_step():
    """Only a populated gas driven negative, or a -inf, rejects the step.

    The K2-18b cloud-edge undershoot and an empty gas's are zeroed and the step stands; a populated gas driven through
    zero, which the error estimate skips, rejects it at any size.
    """
    from types import SimpleNamespace

    from vulcan3._loop_body import _decide_step
    from vulcan3._step_control import _make_clip_fn
    from vulcan3.config import load_config

    cfg = load_config("default")
    statics = SimpleNamespace(
        loss_eps=cfg.loss_eps,
        rtol_accept=cfg.rtol,
        dt_var_min=cfg.dt_var_min,
        dt_min=cfg.dt_min,
        batch_max_retries=110,
        mtol=cfg.mtol,
        atol=cfg.atol,
    )
    gas = jnp.asarray([True, True, False])  # the third species is a condensate
    clip = _make_clip_fn(True, gas, cfg.pos_cut)
    y_old = jnp.asarray([[1.0e18, 1.0e6, 1.0e10]])  # a bulk gas, a populated trace gas, a condensate
    base = {
        "atom_loss_prev": jnp.zeros(1),
        "dt": jnp.float64(1.0),
        "retry_count": jnp.int32(0),
        "keep_prev_run": jnp.int32(0),
        "t": jnp.float64(0.0),
        "y_prev": y_old,
        "y": y_old,
        "ymix": y_old / 1.0e18,
    }
    s = SimpleNamespace(**base)

    def decide(trial, s=s):
        sol = jnp.asarray([trial])
        sol_clip, ymix, _, _ = clip(sol)
        return _decide_step(statics, s, sol, sol_clip, ymix, jnp.zeros(1), jnp.float64(0.0), gas)

    assert bool(decide([1.0e18, 1.0e6, -4.96e4]).accept)  # the condensate's undershoot: zeroed
    empty = SimpleNamespace(
        **{**base, "y": jnp.asarray([[1.0e18, 0.0, 1.0e10]]), "ymix": jnp.asarray([[1.0, 0.0, 1.0e-8]])}
    )
    assert bool(decide([1.0e18, -2.0e8, 1.0e10], empty).accept)  # an empty gas's undershoot, however large
    for trial in ([1.0e18, -1.0e-3, 1.0e10], [1.0e18, 1.0e6, -jnp.inf]):  # a populated gas through zero; a -inf
        d = decide(trial)
        assert not bool(d.accept)
        assert int(d.nega_count_inc) == 1


def test_condensing_clip_keeps_a_nan_gas_sum_nan():
    from vulcan3._step_control import _make_clip_fn

    clip = _make_clip_fn(True, jnp.asarray([True, True, False]), 0.0)
    _, ymix, _, _ = clip(jnp.asarray([[jnp.nan, 1.0, 2.0]]))
    assert not np.all(np.isfinite(np.asarray(ymix)))
    _, ymix0, _, _ = clip(jnp.asarray([[0.0, 0.0, 2.0]]))  # an all-zero gas layer still gives 0
    np.testing.assert_array_equal(np.asarray(ymix0), [[0.0, 0.0, 0.0]])


def test_a_pinned_species_that_carries_no_conserved_atom_does_not_switch_the_repair_off():
    """Under use_ion the electron is pinned in every layer; that pin moves no atom, so the stage repair still acts."""
    from vulcan3 import chem_funs
    from vulcan3.jax_step import _CHEM_CARRIES_ATOM, _repair_stage

    ni, nz = chem_funs.ni, 6
    free = int(np.flatnonzero(~np.asarray(_CHEM_CARRIES_ATOM))[0])  # He on the default network: no H, O, C, N or S
    rng = np.random.default_rng(0)
    k = jnp.asarray(rng.normal(size=(nz, ni)) * 1e6)  # a stage vector with an element defect in every layer
    y = jnp.full((nz, ni), 1e12)
    zeros = jnp.zeros((nz, ni))
    args = (k, zeros, jnp.float64(1e-8), zeros, zeros[:-1], zeros[:-1])
    n_tot, pin_free = jnp.sum(y, axis=1, keepdims=True), jnp.zeros((nz, ni), dtype=bool).at[:, free].set(True)
    repaired = _repair_stage(*args, None, n_tot, y)
    assert not np.array_equal(np.asarray(repaired), np.asarray(k))  # the repair acts on this input
    np.testing.assert_array_equal(np.asarray(_repair_stage(*args, pin_free, n_tot, y)), np.asarray(repaired))
