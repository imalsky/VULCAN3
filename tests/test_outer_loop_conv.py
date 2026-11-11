"""The runner is one-shot and its in-loop convergence check ends the run.

A count_max=50 HD189 run must:
(1) leave the conv ring with 51 strictly increasing times, read from slot
    (accept_count - L) % conv_step; the first is a post-step state (as
    op.save_step) and the last is the final t;
(2) leave longdy and longdydt finite and positive;
(3) end with count == count_max + 1 and end_case 3;
(4) keep every atom_loss under MAX_ATOM_LOSS and end with finite, positive
    dt and t.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

# Roundoff the conservation projection leaves over 50 HD189 steps (~3.5e-10); a 10x regression still trips 1e-8.
MAX_ATOM_LOSS = 1.0e-8


@pytest.mark.strict_isolation
def test_count_capped_run_conv_ring_and_termination():
    import vulcan3.legacy_io as op
    from vulcan3 import outer_loop

    vulcan_cfg = op.default_config()

    vulcan_cfg.count_max = 50
    vulcan_cfg.count_min = 1
    # A count-capped run needs the hybrid vm_mol budget extension OFF: phase-0
    # budget exhaustion flips to phase 1 and extends count_max by +1000, which
    # breaks this test's exact count == count_max + 1 contract.
    vulcan_cfg.use_vm_mol = False
    vulcan_cfg.use_hybrid_vm_mol = False
    vulcan_cfg.use_print_prog = False

    from vulcan3 import op_jax
    from vulcan3.state import RunState

    rs = RunState.with_pre_loop_setup(vulcan_cfg)
    integ = outer_loop.OuterLoop(op_jax.Ros2JAX(), op.Output())
    rs_out = integ(rs)
    # The same run on the raw runner, for the ring the RunState does not carry.
    final = integ._runner(*integ.prepare_runstate(rs))

    assert float(final.t) == float(rs_out.step.t), "runner t != integ(rs) t"

    # 1. Ring buffer + chronological reconstruction
    count = int(final.accept_count)
    conv_step = int(vulcan_cfg.conv_step)
    n_kept = min(count, conv_step)
    start = (count - n_kept) % conv_step
    order = [(start + i) % conv_step for i in range(n_kept)]
    t_arr = np.asarray(final.t_time_ring)[order]
    assert np.all(np.diff(t_arr) > 0), "ring reconstruction is in the wrong order"
    assert t_arr[-1] == float(final.t), "last ring t != final t"

    # 2. longdy / longdydt populated
    longdy, longdydt = rs_out.step.longdy, rs_out.step.longdydt
    assert np.isfinite(longdy), longdy
    assert longdy > 0, longdy
    assert np.isfinite(longdydt), longdydt
    assert longdydt > 0, longdydt

    # 3. Single-shot termination via count_max
    assert int(rs_out.params.count) == vulcan_cfg.count_max + 1
    assert int(rs_out.params.end_case) == 3, "expected end_case 3 (count_max exceeded)"

    # 4. Conservation and a finite clock
    for atom, loss in zip(rs_out.atoms.atom_order, np.asarray(rs_out.atoms.atom_loss), strict=False):
        assert abs(loss) <= MAX_ATOM_LOSS, f"atom_loss[{atom}] = {loss:.3e}"
    t, dt = float(rs_out.step.t), float(rs_out.step.dt)
    assert dt > 0, dt
    assert math.isfinite(dt), dt
    assert t > 0, t
    assert math.isfinite(t), t


# (key, value, Kzz cm^2/s, longdydt s^-1): a carry whose loose-branch verdict flips when that key moves. Hp is
# 1e7 cm, so Kzz / (0.1 Hp)^2 is 5e-9 (between floor and cap), 1e-7 (above the cap) or 1e-12 (below the floor).
_SLOPE_CASES = [
    ("slope_min_hp_frac", 0.2, 5.0e3, 3.0e-9),
    ("slope_min_cap", 1.0e-7, 1.0e5, 3.0e-8),
    ("slope_min_floor", 1.0e-11, 1.0, 5.0e-11),
]


@pytest.mark.parametrize(("key", "value", "kzz", "longdydt"), _SLOPE_CASES, ids=[c[0] for c in _SLOPE_CASES])
def test_slope_min_keys_set_the_loose_certificate(key, value, kzz, longdydt):
    """Each `slope_min_*` key moves `conv_normal`'s loose-branch verdict on the same carry."""
    from types import SimpleNamespace

    import jax.numpy as jnp

    from vulcan3 import conv_normal, load_config

    nz = 5
    s = SimpleNamespace(
        longdy=jnp.float64(0.05),  # under yconv_min 0.1, over yconv_cri 0.01: only the loose branch can pass
        longdydt=jnp.float64(longdydt),
        aflux_change=jnp.float64(0.0),
        pv=SimpleNamespace(Kzz=jnp.full(nz - 1, kzz)),
        Hp=jnp.full(nz, 1.0e7),
        geom_ok=jnp.bool_(True),
        budget_ok=jnp.bool_(True),
        keep_prev_run=jnp.int32(0),
    )
    default = bool(conv_normal(s, load_config("default"))[0])
    moved = bool(conv_normal(s, load_config("default", **{key: value}))[0])
    assert default != moved, (key, default, moved)


@pytest.mark.strict_isolation
def test_a_looser_slope_threshold_ends_the_run_sooner():
    """The runner reads the slope keys: a looser loose-branch threshold certifies where the default runs to the cap.

    `yconv_min` is opened so that the slope test alone decides the loose branch on this small photo-off column.
    """
    import vulcan3
    import vulcan3.legacy_io as op
    from vulcan3 import op_jax, outer_loop
    from vulcan3.state import RunState

    counts = {}
    for name, loose in (("default", {}), ("loose", {"slope_min_cap": 1.0e-2, "slope_min_floor": 1.0e-2})):
        cfg = vulcan3.make_config(
            nz=12,
            P_b=1e5,
            P_t=1e-2,
            count_max=250,
            yconv_min=10.0,
            use_photo=False,
            use_print_prog=False,
            ini_mix="const_mix",
            const_mix={"H2": 0.9, "He": 0.089, "H2O": 5e-3, "CO": 5e-3, "N2": 1e-3},
            **loose,
        )
        integ = outer_loop.OuterLoop(op_jax.Ros2JAX(), op.Output(cfg=cfg), cfg=cfg)
        rs = integ(RunState.with_pre_loop_setup(cfg))
        counts[name] = (int(rs.params.count), int(rs.params.end_case))
    assert counts["default"] == (251, 3), counts
    assert counts["loose"][1] == 1, counts
    assert counts["loose"][0] < 251, counts
