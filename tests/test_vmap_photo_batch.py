"""Batched photochemistry: `run_batch` with photolysis on.

The T-interpolated cross sections (`absp_T_cross`, `cross_J_T`) ride
`ProfileVars` per lane; the rest of the photo statics is closure-baked.
Pins: (1) profile B solo on its own runner matches lane 1 of a batch baked
from A, so no per-profile photo field stays in the closure; (2) vacuity: the
lanes' cross sections, photo firing and results differ; (3)
`prepare_runstate` rejects a different TOA stellar flux; (4) a lane that goes
non-finite on a photolysis tick keeps its whole pre-step carry, photolysis
arrays and convergence ring included.
"""

from __future__ import annotations

import numpy as np
import pytest
from _helpers import fast_cfg, stack_atm_statics, stack_integ_states, unstack_integ_states
from test_vmap_while_loop import _build_integ, _build_rs, _max_rel_diff

COUNT_MAX = 30
# Batched runs evaluate photolysis on the iteration tick (accepted states
# only), solo runs on the accepted-step count, so the stored RT arrays are a
# different iterate of the same fixed point: they are judged by SCALE against
# their own peak, at RTOL. The chemistry agrees at the convergence scale and
# the vacuity checks below catch a closure leak.
RTOL = 5e-2
# ymix is judged by scale, |lane - own solo| / |soloA - soloB|: a
# closure-baked photo field makes lane 1 equal soloA (ratio ~1). Bit
# identity depends on the XLA version. The photolysis cadence differs (tick
# vs accepted count), so rejected steps move the lanes apart: 1.6e-4 with the
# Step control (8 rejections of 39 attempts on lane A).
SCALE_TOL = 1e-3


def _pin_cfg(**extra):
    """Small photo-on batched config: isothermal, const_mix init.

    Fixed diffusion scheme (the hybrid flips schemes per lane), photo cadence low enough to fire within COUNT_MAX.
    `T_cross_sp=["H2O"]` gives the two Tiso lanes different cross sections.
    """
    return fast_cfg(
        **{
            "count_max": COUNT_MAX,
            "use_photo": True,
            "ini_mix": "const_mix",
            "use_vm_mol": False,
            "use_hybrid_vm_mol": False,
            "nz": 40,
            "ini_update_photo_frq": 5,
            "T_cross_sp": ["H2O"],
            **extra,
        }
    )


def _peak_ratio(bat, sol):
    """Max absolute difference from the lane's own solo, against the solo peak."""
    bat, sol = (np.asarray(x, dtype=np.float64) for x in (bat, sol))
    return float(np.max(np.abs(bat - sol))) / float(np.max(np.abs(sol)))


def _scale_ratio(bat, sol, other):
    bat, sol, other = (np.asarray(x, dtype=np.float64) for x in (bat, sol, other))
    return float(np.max(np.abs(bat - sol))) / float(np.max(np.abs(sol - other)))


@pytest.mark.strict_isolation
def test_photo_lane_matches_its_solo_run():

    vulcan_cfg = _pin_cfg()

    rsA = _build_rs(vulcan_cfg, Tiso=900.0)
    integA = _build_integ()
    sA, atmA = integA.prepare_runstate(rsA)

    rsB = _build_rs(vulcan_cfg, Tiso=1600.0)
    # Route B through integA too (the batch-driver pattern; exercises the
    # same-star guard's accept path) AND through its own integB for the solo.
    sB, atmB = integA.prepare_runstate(rsB)
    integB = _build_integ()
    sB_own, atmB_own = integB.prepare_runstate(rsB)

    # --- vacuity: per-lane T-dep cross sections differ ------------------
    n_absp_T = sA.pv.p_absp_T_cross.shape[0]
    tdiff = _max_rel_diff(sB.pv.p_absp_T_cross, sA.pv.p_absp_T_cross, floor=0.0)
    jdiff = _max_rel_diff(sB.pv.p_cross_J_T, sA.pv.p_cross_J_T, floor=0.0)
    assert n_absp_T != 0 and tdiff != 0.0 and jdiff != 0.0, (
        "T-dep cross sections do not differ between lanes",
        n_absp_T,
        tdiff,
        jdiff,
    )

    # --- solo runs --------------------------------------------------------
    soloA = integA._runner(sA, atmA)
    soloB = integB._runner(sB_own, atmB_own)

    # photo really fired: actinic flux nonzero, photo branch wrote k rows.
    assert np.any(np.asarray(soloA.aflux) > 0) and float(soloA.aflux_change) != 0.0, "photo branch never fired in soloA"

    # --- batch on A's runner ----------------------------------------------
    batched = integA.run_batch(
        stack_integ_states([sA, sB]),
        stack_atm_statics([atmA, atmB]),
    )
    out = unstack_integ_states(batched, 2)

    for name in ("ymix", "k_arr", "aflux", "tau"):
        bat0, bat1 = getattr(out[0], name), getattr(out[1], name)
        refA, refB = getattr(soloA, name), getattr(soloB, name)
        if name == "ymix":
            relA = _scale_ratio(bat0, refA, refB)
            relB = _scale_ratio(bat1, refB, refA)
            tol = SCALE_TOL
        else:
            relA = _peak_ratio(bat0, refA)
            relB = _peak_ratio(bat1, refB)
            tol = RTOL
        assert relA <= tol and relB <= tol, ("solo-vs-batch", name, relA, relB)

    profiles_differ = _max_rel_diff(soloB.ymix, soloA.ymix)
    assert profiles_differ >= 1e-6, ("profiles identical", profiles_differ)

    # --- same-star guard ----------------------------------------------------
    rsC = rsB._replace(photo=rsB.photo._replace(sflux_top=rsB.photo.sflux_top * 2.0))
    with pytest.raises(ValueError):
        integA.prepare_runstate(rsC)


@pytest.mark.strict_isolation
def test_lane_gone_nonfinite_on_a_photo_tick_keeps_its_whole_carry():
    """A lane whose step goes non-finite freezes at its last good carry, leaf for leaf and bitwise.

    Also when that step's tick updated its photolysis arrays. The batched freeze never re-selects the convergence
    ring (the step writes its slot already frozen) and reads the pre-step photolysis arrays only inside the
    photolysis tick's cond, so a regression in either shows up here as a frozen lane that kept part of the failed
    tick.

    Lane 1 is lane 0 with one thermal rate row non-finite and dt at dt_min:
    tick 0 is on the photolysis cadence, the lane takes the update, its solve
    is non-finite, and the rejected attempt cannot shrink dt below dt_min, so
    it is force-accepted and y goes non-finite on that tick. Its actinic flux
    and optical depth start at zero, so the update it took changed them.
    """
    import jax
    import jax.numpy as jnp

    from vulcan3 import outer_loop

    vulcan_cfg = _pin_cfg()
    integ = _build_integ()
    s0, atm0 = integ.prepare_runstate(_build_rs(vulcan_cfg, Tiso=900.0))
    row = 1  # a thermal reaction: no photolysis or ion branch writes it
    photo_rows = set(np.asarray(integ._photo_static.branch_re_idx).tolist())
    photo_rows |= set(np.asarray(integ._photo_static.ion_branch_re_idx).tolist())
    assert row not in photo_rows
    bad = s0._replace(
        k_arr=s0.k_arr.at[row].set(jnp.nan),
        dt=jnp.float64(float(vulcan_cfg.dt_min)),
        aflux=jnp.zeros_like(s0.aflux),
        tau=jnp.zeros_like(s0.tau),
    )
    good, frozen = unstack_integ_states(
        integ.run_batch(
            stack_integ_states([s0, bad]),
            stack_atm_statics([atm0, atm0]),
        ),
        2,
    )
    assert int(frozen.termination_reason) == outer_loop.TERM_NONFINITE
    assert int(good.termination_reason) != outer_loop.TERM_NONFINITE
    # vacuity: the update the frozen lane took would have made these nonzero
    assert np.any(np.asarray(good.aflux) != 0.0)
    assert np.any(np.asarray(good.tau) != 0.0)

    done_flags = ("is_done", "termination_reason")
    leaves_in = jax.tree_util.tree_flatten_with_path(bad)[0]
    leaves_out = jax.tree_util.tree_leaves(frozen)
    for (path, a), b in zip(leaves_in, leaves_out, strict=False):
        name = jax.tree_util.keystr(path)
        if name.lstrip(".") in done_flags:
            continue
        a, b = np.asarray(a), np.asarray(b)
        equal_nan = bool(np.issubdtype(a.dtype, np.floating))
        assert a.dtype == b.dtype and np.array_equal(a, b, equal_nan=equal_nan), name


@pytest.mark.strict_isolation
def test_queue_on_photo_lanes_matches_the_batch():
    """`run_queue` against `run_batch` on the same two photo-on profiles.

    (1) One lane per job: bitwise `run_batch`. With photo on, the initial fill
    can hit the photo branch twice before the first chemistry step and move
    aflux_change, which gates the certificate. Photo-off twin: test_run_queue.

    (2) ONE lane: the second profile is refilled into a lane the first profile
    just left, and must start on its own photolysis fields the way a
    plain-batch lane does at tick 0. It reaches the same termination reason
    and the same mixing ratios as the pair run in a single `run_batch`; a
    refilled lane left on its predecessor's RT state would integrate a
    different column.
    """
    vulcan_cfg = _pin_cfg()
    integ = _build_integ()
    sA, atmA = integ.prepare_runstate(_build_rs(vulcan_cfg, Tiso=900.0))
    sB, atmB = integ.prepare_runstate(_build_rs(vulcan_cfg, Tiso=1600.0))
    init_b = stack_integ_states([sA, sB])
    atm_b = stack_atm_statics([atmA, atmB])
    ref = integ.run_batch(init_b, atm_b)

    (y, t, acc, reason), n_iter = integ.run_queue(
        lambda job: job,
        (init_b, atm_b),
        n_lanes=2,
        out_fn=lambda f: (f.y, f.t, f.accept_count, f.termination_reason),
    )
    for name, a, b in (
        ("y", y, ref.y),
        ("t", t, ref.t),
        ("accept_count", acc, ref.accept_count),
        ("termination_reason", reason, ref.termination_reason),
    ):
        assert np.array_equal(np.asarray(a), np.asarray(b)), name

    (y, reason), _n_iter = integ.run_queue(
        lambda job: job,
        (init_b, atm_b),
        n_lanes=1,
        out_fn=lambda f: (f.y, f.termination_reason),
        chunk=1,
    )
    assert np.array_equal(np.asarray(reason), np.asarray(ref.termination_reason))
    for k in (0, 1):
        yk, yr = np.asarray(y[k]), np.asarray(ref.y[k])
        rel = _max_rel_diff(yk / yk.sum(axis=1, keepdims=True), yr / yr.sum(axis=1, keepdims=True))
        assert rel < RTOL, (k, rel)


@pytest.mark.strict_isolation
def test_refill_on_a_photo_tick_applies_photolysis_once():
    """A lane refilled at a tick on its photolysis cadence takes photolysis once, from its first step's own gate.

    As a plain-batch lane does at tick 0. With the photo and geometry cadences at 1 every tick is on cadence, so the
    refilled job sees the ticks it would see alone and the two results are bitwise equal. A second application at
    the refill moves aflux_change, which gates the convergence certificate.
    """
    vulcan_cfg = _pin_cfg(ini_update_photo_frq=1, final_update_photo_frq=1, update_frq=1)
    integ = _build_integ()
    sA, atmA = integ.prepare_runstate(_build_rs(vulcan_cfg, Tiso=900.0))
    sB, atmB = integ.prepare_runstate(_build_rs(vulcan_cfg, Tiso=1600.0))
    fields = ("y", "t", "accept_count", "termination_reason", "aflux_change")

    def out_fn(f):
        return tuple(getattr(f, name) for name in fields)

    pair, _ = integ.run_queue(
        lambda job: job,
        (stack_integ_states([sA, sB]), stack_atm_statics([atmA, atmB])),
        n_lanes=1,
        out_fn=out_fn,
        chunk=1,
    )
    alone, _ = integ.run_queue(
        lambda job: job,
        (stack_integ_states([sB]), stack_atm_statics([atmB])),
        n_lanes=1,
        out_fn=out_fn,
    )
    for name, refilled, own in zip(fields, pair, alone, strict=False):
        assert np.array_equal(np.asarray(refilled[1]), np.asarray(own[0])), name
