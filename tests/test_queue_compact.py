"""`OuterLoop.run_queue` tail compaction (`compact=` / config `queue_compact`)."""

# Once no job is pending and the live lanes fit a rung of the ladder, the queue
# continues at that narrower width from the same tick with no refill. A done
# lane is frozen and outside every lane-uniform predicate, so dropping it
# changes no job's step, cadence or certificate. Whether the narrower program
# computes each lane's arithmetic to the same bits is up to XLA: at the
# production widths (792 -> 264 -> 132 lanes and a 256-lane retry pass -> 132,
# WASP-39 b on a GH200) every job came back bitwise, but at the tiny widths of
# these tests the CPU backend can round differently from the first step on, and
# the GPU the column-sum accumulators (atom_loss, the element budget). So the
# pins below are exact on everything the queue decides (ticks, step counts,
# termination, done flags, every integer leaf) and roundoff-close on the
# floats: mixing ratios above 1e-12 within 1e-6 in log10, the dimensionless
# accumulators within 1e-10, every other float leaf within 1e-4 of its own
# scale (the step-control leaves, dt and the error proxy delta, amplify a
# last-bit difference: 2e-5 on the CPU with the grouped RHS).
# Pins (CPU and GPU):
#
#   1. Ladder parsing: descending, deduplicated, rungs at or above the lane
#      count dropped; bad entries raise. The config key exists and is off.
#   2. Off is the stock program: `compact=None` (config default), `()` and a
#      ladder whose every rung is dropped trace the same jaxpr; a ladder adds
#      one while loop per rung.
#   3. Photo off, refill + compaction 3 -> 2 -> 1 lanes: every job's final
#      state matches the uncompacted queue's (as above) and the tick count is
#      the same.
#   4. Photo on, refill + compaction 3 -> 2 -> 1 lanes, photolysis every 4
#      ticks and geometry refresh every 3: as 3.
#   5. More lanes than jobs (the queue compacts at tick 0): matches the
#      uncompacted queue and `run_batch`.
#   6. The config key drives the default: `queue_compact` set on the runner's
#      config gives the explicit-ladder program and result (bitwise: the same
#      program).
#   7. (slow) jvp through a compacted queue matches the lockstep batch's jvp,
#      primal and tangent (both within 1e-6 of their scale).
#   8. Every queue loop (traced, not compiled) ends in the refill cond, whose idle
#      branch passes the carry through, and no other cond returns the y ring: a
#      tail loop without the refill cond copied every lane leaf each tick, and a
#      photolysis cond returning the written ring made XLA:GPU rebuild and copy
#      it every tick (VULCAN3 notes "Efficiency program").

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _helpers import stack_atm_statics, stack_integ_states
from test_vmap_while_loop import _build_integ, _build_rs, _pin_cfg

from vulcan3._runner_queue import _queue_compact_rungs

TISO = (900.0, 1000.0, 1100.0, 1200.0, 1300.0)
# Per-job step caps (the carry's count_max_dyn) so the lanes end at different
# ticks whatever the physics does; all below the photo-off COUNT_MAX of 40.
CAPS_OFF = (10, 30, 18, 39, 5)


def _ident(job):
    return job


def _whole(final):
    """Keep every leaf of a job's final state."""
    return final


def _with_caps(st, caps):
    return st._replace(count_max_dyn=jnp.asarray(caps, dtype=st.count_max_dyn.dtype))


def _assert_trees_bitwise(a, b, what):
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    assert len(la) == len(lb), what
    names = [jax.tree_util.keystr(p) for p, _ in jax.tree_util.tree_flatten_with_path(a)[0]]
    bad = []
    for name, x, y in zip(names, la, lb, strict=False):
        x, y = np.asarray(x), np.asarray(y)
        en = bool(np.issubdtype(x.dtype, np.floating))
        if x.shape != y.shape or x.dtype != y.dtype or not np.array_equal(x, y, equal_nan=en):
            bad.append(name)
    assert not bad, f"{what}: leaves differ: {bad}"


# Roundoff-closeness of two final states (see the module docstring).
REL = 1e-6  # log10 mixing ratios; the jvp primal and tangent, by scale
LEAF_REL = 1e-4  # any other float leaf, by its own scale
ACC_ABS = 1e-10
ACCUMULATORS = (".atom_loss", ".atom_loss_prev", ".budget_err", ".budget_drift")
YMIX_FLOOR = 1e-12


def _assert_trees_close(a, b, what):
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    assert len(la) == len(lb), what
    names = [jax.tree_util.keystr(p) for p, _ in jax.tree_util.tree_flatten_with_path(a)[0]]
    bad, n_bitwise, worst = [], 0, {}
    for name, x, y in zip(names, la, lb, strict=False):
        x, y = np.asarray(x), np.asarray(y)
        if x.shape != y.shape or x.dtype != y.dtype:
            bad.append(f"{name} shape/dtype")
            continue
        if not np.issubdtype(x.dtype, np.floating):
            if not np.array_equal(x, y):
                bad.append(f"{name} (exact)")
            else:
                n_bitwise += 1
            continue
        if np.array_equal(x, y, equal_nan=True):
            n_bitwise += 1
            continue
        fx, fy = np.isfinite(x), np.isfinite(y)
        if not (np.array_equal(fx, fy) and np.array_equal(x[~fx], y[~fy], equal_nan=True)):
            bad.append(f"{name} (non-finite cells)")
            continue
        d = float(np.max(np.abs(x[fx] - y[fy]))) if fx.any() else 0.0
        if name in ACCUMULATORS:
            worst[name] = d
            ok = d <= ACC_ABS
        else:
            scale = max(float(np.max(np.abs(x[fx]))), float(np.max(np.abs(y[fy]))))
            worst[name] = d / scale
            ok = d <= LEAF_REL * scale
        if not ok:
            bad.append(name)
    assert not bad, f"{what}: {bad}; worst {worst}"
    ya, yb = np.asarray(a.y), np.asarray(b.y)
    ma = ya / ya.sum(axis=-1, keepdims=True)
    mb = yb / yb.sum(axis=-1, keepdims=True)
    m = (ma > YMIX_FLOOR) & (mb > YMIX_FLOOR)
    dl = float(np.max(np.abs(np.log10(ma[m]) - np.log10(mb[m])))) if m.any() else 0.0
    assert dl <= REL, f"{what}: max |dlog10 ymix| {dl:.3g}"
    print(f"{what}: {n_bitwise}/{len(la)} leaves bitwise; max |dlog10 ymix| {dl:.3g}; worst float leaves {worst}")


@pytest.fixture(scope="module")
def photo_off():
    cfg = _pin_cfg()
    integ = _build_integ()
    pairs = [integ.prepare_runstate(_build_rs(cfg, Tiso=T)) for T in TISO]
    st = stack_integ_states([s for s, _ in pairs])
    atm = stack_atm_statics([a for _, a in pairs])
    return integ, (_with_caps(st, CAPS_OFF), atm)


def test_ladder_parsing_and_config_key():
    from vulcan3.config import load_config, validate_overrides

    assert _queue_compact_rungs(None, 792) == ()
    assert _queue_compact_rungs([], 792) == ()
    assert _queue_compact_rungs([132, 264], 792) == (264, 132)
    assert _queue_compact_rungs((264, 132, 264), 792) == (264, 132)
    assert _queue_compact_rungs([np.int64(264), 1000, 792], 792) == (264,)
    assert _queue_compact_rungs([264, 132], 132) == ()
    for bad in ([0], [-4], [2.5], [True], "264", 264, [None]):
        with pytest.raises(ValueError):
            _queue_compact_rungs(bad, 792)
    assert list(load_config("W39b").queue_compact) == []
    assert list(load_config("W39b", queue_compact=[264, 132]).queue_compact) == [264, 132]
    validate_overrides({"queue_compact": [264, 132]})


def _jaxpr(integ, jobs, **kw):
    return jax.make_jaxpr(lambda j: integ.run_queue(_ident, j, n_lanes=3, out_fn=_whole, chunk=1, **kw))(jobs)


def _top_level_whiles(closed):
    """While loops at the top level of the queue program (inside its jit)."""
    (eqn,) = [e for e in closed.jaxpr.eqns if "jaxpr" in e.params]
    return sum(e.primitive.name == "while" for e in eqn.params["jaxpr"].jaxpr.eqns)


def test_off_traces_the_stock_program(photo_off):
    integ, jobs = photo_off
    ref = _jaxpr(integ, jobs)  # the config default: []
    assert str(ref) == str(_jaxpr(integ, jobs, compact=()))
    assert str(ref) == str(_jaxpr(integ, jobs, compact=[3, 7]))  # every rung dropped
    assert _top_level_whiles(ref) == 1
    assert _top_level_whiles(_jaxpr(integ, jobs, compact=(2, 1))) == 3


def test_compaction_matches_photo_off(photo_off):
    integ, jobs = photo_off
    off, n_off = integ.run_queue(_ident, jobs, n_lanes=3, out_fn=_whole, chunk=1)
    on, n_on = integ.run_queue(_ident, jobs, n_lanes=3, out_fn=_whole, chunk=1, compact=(1, 2))
    assert int(n_on) == int(n_off)
    assert bool(np.all(np.asarray(off.is_done)))
    # a lane stops once accept_count exceeds its cap
    assert np.all(np.asarray(off.accept_count) <= np.asarray(CAPS_OFF) + 1)
    _assert_trees_close(off, on, "photo off, 3 -> 2 -> 1 lanes")


def test_compaction_matches_photo_on():
    from test_vmap_photo_batch import _pin_cfg as _photo_cfg

    cfg = _photo_cfg(ini_update_photo_frq=4, final_update_photo_frq=4, update_frq=3)
    integ = _build_integ()
    sA, aA = integ.prepare_runstate(_build_rs(cfg, Tiso=900.0))
    sB, aB = integ.prepare_runstate(_build_rs(cfg, Tiso=1200.0))
    st = stack_integ_states([sA, sB, sA, sB])
    atm = stack_atm_statics([aA, aB, aA, aB])
    jobs = (_with_caps(st, (7, 15, 24, 30)), atm)
    off, n_off = integ.run_queue(_ident, jobs, n_lanes=3, out_fn=_whole, chunk=1)
    on, n_on = integ.run_queue(_ident, jobs, n_lanes=3, out_fn=_whole, chunk=1, compact=(2, 1))
    assert int(n_on) == int(n_off)
    # Setup fills the flux too: only the loop's refresh can have moved it.
    assert not np.array_equal(np.asarray(off.aflux[0]), np.asarray(sA.aflux))
    _assert_trees_close(off, on, "photo on, 3 -> 2 -> 1 lanes")


def test_idle_lanes_compact_at_tick_zero_and_match_the_batch(photo_off):
    integ, jobs = photo_off
    L = len(TISO) + 2
    off, _ = integ.run_queue(_ident, jobs, n_lanes=L, out_fn=_whole)
    out, _ = integ.run_queue(_ident, jobs, n_lanes=L, out_fn=_whole, compact=(len(TISO), 2))
    _assert_trees_close(off, out, "idle lanes, compacted at tick 0, vs uncompacted")
    _assert_trees_close(integ.run_batch(*jobs), out, "idle lanes, compacted at tick 0, vs run_batch")


def test_config_key_drives_the_default(photo_off):
    integ, jobs = photo_off
    explicit, _ = integ.run_queue(_ident, jobs, n_lanes=3, out_fn=_whole, chunk=1, compact=(2, 1))
    integ._cfg.queue_compact = [1, 2]  # restored by conftest's _cfg_guard
    n_cached = len(integ._vrunner_queue)
    via_cfg, _ = integ.run_queue(_ident, jobs, n_lanes=3, out_fn=_whole, chunk=1)
    assert len(integ._vrunner_queue) == n_cached, "config ladder must hit the explicit key"
    _assert_trees_bitwise(explicit, via_cfg, "config queue_compact vs explicit ladder")


@pytest.mark.slow
def test_compacted_queue_jvp_matches_the_batch_jvp():
    from _helpers import fast_cfg

    cfg = fast_cfg(count_max=8, nz=12, use_vm_mol=False, use_hybrid_vm_mol=False)
    integ = _build_integ()
    pairs = [integ.prepare_runstate(_build_rs(cfg, Tiso=T)) for T in (900.0, 1200.0)]
    s0 = _with_caps(stack_integ_states([s for s, _ in pairs]), (4, 8))
    a0 = stack_atm_statics([a for _, a in pairs])

    def jobs(x):
        f = jnp.exp(x)
        return (s0._replace(pv=s0.pv._replace(Kzz=s0.pv.Kzz * f)), a0._replace(Kzz=a0.Kzz * f))

    def batch_y(x):
        return integ.run_batch(*jobs(x)).y

    def queue_y(x):
        out, _n = integ.run_queue(_ident, jobs(x), n_lanes=2, out_fn=lambda f: f.y, compact=(1,))
        return out

    x0, dx = jnp.float64(0.0), jnp.float64(1.0)
    y_b, dy_b = jax.jvp(batch_y, (x0,), (dx,))
    y_q, dy_q = jax.jvp(queue_y, (x0,), (dx,))
    scale = float(jnp.max(jnp.abs(dy_b)))
    assert scale > 0.0, "vacuous: zero tangent"
    sy = float(jnp.max(jnp.abs(y_b)))
    assert float(jnp.max(jnp.abs(y_q - y_b))) <= REL * sy
    assert float(jnp.max(jnp.abs(dy_q - dy_b))) <= REL * scale


def _eqns(jaxpr):
    """Every equation of `jaxpr` and of the jaxprs it calls (loop bodies, cond branches, jit bodies), depth first."""
    for e in jaxpr.eqns:
        yield e
        for v in e.params.values():
            for x in v if isinstance(v, (tuple, list)) else (v,):
                sub = getattr(x, "jaxpr", x)
                if hasattr(sub, "eqns"):
                    yield from _eqns(sub)


def test_queue_loops_keep_the_ring_out_of_the_photolysis_cond():
    from test_vmap_photo_batch import _pin_cfg as _photo_cfg

    cfg = _photo_cfg(ini_update_photo_frq=4, final_update_photo_frq=4, update_frq=3)
    integ = _build_integ()
    sA, aA = integ.prepare_runstate(_build_rs(cfg, Tiso=900.0))
    sB, aB = integ.prepare_runstate(_build_rs(cfg, Tiso=1200.0))
    st = stack_integ_states([sA, sB, sA, sB])
    atm = stack_atm_statics([aA, aB, aA, aB])
    jobs = (_with_caps(st, (7, 15, 24, 30)), atm)

    def is_ring(v, w):
        return v.aval.shape == (w, *st.y_time_ring.shape[1:]) and v.aval.dtype == st.y_time_ring.dtype

    widths = []
    for loop in _eqns(_jaxpr(integ, jobs, compact=(2, 1)).jaxpr):
        if loop.primitive.name != "while":
            continue
        body = loop.params["body_jaxpr"].jaxpr
        (w,) = [w for w in (3, 2, 1) if any(is_ring(v, w) for v in body.outvars)] or (None,)
        if w is None:
            continue
        widths.append(w)
        (ring_out,) = [v for v in body.outvars if is_ring(v, w)]
        refill = [e for e in body.eqns if e.primitive.name == "cond" and ring_out in e.outvars]
        assert len(refill) == 1, f"width {w}: the carry's y ring does not come out of the refill cond"
        assert any(
            not b.jaxpr.eqns and all(v in b.jaxpr.invars for v in b.jaxpr.outvars) for b in refill[0].params["branches"]
        ), f"width {w}: no pass-through branch"
        bad = [
            e
            for e in _eqns(body)
            if e.primitive.name == "cond" and e is not refill[0] and any(is_ring(v, w) for v in e.outvars)
        ]
        assert not bad, f"width {w}: {len(bad)} cond(s) besides the refill return the y ring"
    assert sorted(widths) == [1, 2, 3], widths
