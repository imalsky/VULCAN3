"""Step-control corrections of notes "Parity ledger", one unit check each.

Master (op.py:2949) masks the error estimate on the pre-step mixing ratio alone, so a species the step creates escapes
it; its clip (op.py:2459) zeroes every negative value, so the negativity test of step_ok can never fire. VULCAN-JAX
masks a species only when both its pre-step and its pre-clip trial mixing ratio are below mtol, keeps values below
nega_cut negative (the step is then rejected), and keeps a non-finite gas sum non-finite.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def test_estimate_counts_a_species_the_step_creates():
    from vulcan3.outer_loop import _make_aggregate_delta_fn

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


def test_clip_keeps_values_below_nega_cut_negative():
    from vulcan3.outer_loop import _clip_prologue

    y = jnp.asarray([[5.0, 0.5, -0.5, -2.0]])
    y_clip, _, _ = _clip_prologue(y, pos_cut=1.0, nega_cut=-1.0)
    np.testing.assert_array_equal(np.asarray(y_clip), [[5.0, 0.0, 0.0, -2.0]])


def test_shipped_cut_rejects_a_too_negative_step():
    from types import SimpleNamespace

    from vulcan3._loop_body import _decide_step
    from vulcan3.config import load_config
    from vulcan3.outer_loop import _clip_prologue

    cfg = load_config("default")
    statics = SimpleNamespace(
        loss_eps=cfg.loss_eps, rtol_accept=cfg.rtol, dt_var_min=cfg.dt_var_min, dt_min=cfg.dt_min, batch_max_retries=110
    )
    s = SimpleNamespace(
        atom_loss_prev=jnp.zeros(1),
        dt=jnp.float64(1.0),
        retry_count=jnp.int32(0),
        t=jnp.float64(0.0),
        y_prev=jnp.ones((1, 2)),
        ymix=jnp.ones((1, 2)),
    )

    def decide(trial):
        y = jnp.asarray([[1.0e18, trial]])
        y_clip, _, _ = _clip_prologue(y, cfg.pos_cut, cfg.nega_cut)
        return _decide_step(statics, s, y, y_clip, y_clip, jnp.zeros(1), jnp.float64(0.0))

    # Past the cut, and the 0-D atomic-H start's CH4 trial that froze its error (notes "Parity ledger"): rejected.
    for trial in (2.0 * cfg.nega_cut, -1.56e5):
        d = decide(trial)
        assert not bool(d.accept)
        assert int(d.nega_count_inc) == 1
    # Within the cut: zeroed and accepted.
    d = decide(0.5 * cfg.nega_cut)
    assert bool(d.accept)
    assert float(d.y_post_clip[0, 1]) == 0.0


def test_condensing_clip_keeps_a_nan_gas_sum_nan():
    from vulcan3.outer_loop import _make_clip_fn

    clip = _make_clip_fn(True, jnp.asarray([True, True, False]), 0.0, -1.0)
    _, ymix, _, _ = clip(jnp.asarray([[jnp.nan, 1.0, 2.0]]))
    assert not np.all(np.isfinite(np.asarray(ymix)))
    _, ymix0, _, _ = clip(jnp.asarray([[0.0, 0.0, 2.0]]))  # an all-zero gas layer still gives 0
    np.testing.assert_array_equal(np.asarray(ymix0), [[0.0, 0.0, 0.0]])
