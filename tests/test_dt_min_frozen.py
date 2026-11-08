"""A trial that keeps failing at `dt_min` ends the run loudly, on every runner, and leaves a healthy column alone.

Force-accepted trials keep the previous state (notes "Parity ledger"), so without this a frozen column idles to its
step cap and reads as a plain rejected run. Two constructed failures: a step floor too long for the chemistry, on a
whole run, and an atom-loss record no trial can match, which freezes one lane of a batch.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _helpers import fast_cfg, run_child, stack_atm_statics, stack_integ_states

from vulcan3 import DtMinFrozenError, MissingAtomError, RunState, legacy_io, op_jax, raise_if_frozen
from vulcan3.outer_loop import ATM_STATIC_BATCH_AXES, OuterLoop, certificate_failures, conv_normal
from vulcan3.runtime_validation import refuse_missing_atoms
from vulcan3.state import TERM_DT_MIN_FROZEN, TERM_STEP_COUNT

_COUNT_MAX = 12
# A step floor far too long for the chemistry, so the error estimate fails at dt_min. A tiny rtol alone would not do
# it: at 1e-14 s the estimate falls to zero in round-off and the step is accepted.
_FLOOR_S = 1e6
_TIGHT_RTOL = 1e-12


def _column(**overrides):
    # count_min at the cap: an idling column's longdy reads 0, and only the ready gate stands between it and a
    # certificate at the first force-accept.
    cfg = fast_cfg(count_max=_COUNT_MAX, nz=12, Tiso=1000.0, use_vm_mol=False, use_hybrid_vm_mol=False, **overrides)
    cfg.count_min = _COUNT_MAX
    integ = OuterLoop(op_jax.Ros2JAX(), legacy_io.Output(), cfg=cfg)
    return cfg, integ, RunState.with_pre_loop_setup(cfg)


def _frozen_lane(state):
    """Record an atom loss far from any the trial can reach; only an accept could rewrite it, and none can come."""
    far = jnp.ones_like(state.atom_loss)
    return state._replace(atom_loss=far, atom_loss_prev=far)


def _ident(job):
    return job


def _out_fn(final):
    return final.y, final.accept_count, final.termination_reason, final.delta


@pytest.mark.strict_isolation
def test_a_floor_too_long_stops_a_solo_run_and_names_the_cell(tmp_path, caplog):
    from vulcan3 import cli
    from vulcan3._loop_branches import _splice_step_atm
    from vulcan3.chem_funs import NET_JAX, NETWORK
    from vulcan3.config import dump_config
    from vulcan3.jax_step import jax_ros2_step

    cfg, integ, rs = _column(rtol=_TIGHT_RTOL, dt_min=_FLOOR_S)
    state, atm_static = integ.prepare_runstate(rs)
    final = integ._runner(state, atm_static)

    # Stopped on the second force-accept at dt_min, not on the cap.
    assert int(final.termination_reason) == TERM_DT_MIN_FROZEN
    assert int(final.keep_prev_run) == 2
    assert int(final.accept_count) < _COUNT_MAX
    assert float(final.dt) == _FLOOR_S
    # The state did not move, which is what could certify a column: the certificate has to say no itself.
    idle = final._replace(
        geom_ok=jnp.bool_(True), budget_ok=jnp.bool_(True), longdy=jnp.float64(0.0), longdydt=jnp.float64(0.0)
    )
    assert not bool(conv_normal(idle, cfg)[0])
    assert bool(conv_normal(idle._replace(keep_prev_run=jnp.int32(0)), cfg)[0])  # the frozen term alone said no
    assert any("dt_min" in term for term, _, _ in certificate_failures(final, cfg, integ._atom_order))

    # The message's cell and ratio, against the trial computed here from the exit carry.
    refresh = integ._refresh_static
    atm_step = _splice_step_atm(atm_static, final, False, refresh.kb, refresh.Navo)
    sol, delta_arr = jax_ros2_step(final.y, final.k_arr, final.dt, atm_step, NET_JAX, fix_mask=None, matrix_free=False)
    trial_mix = sol / jnp.sum(sol, axis=1, keepdims=True)
    live = np.asarray(((final.ymix >= cfg.mtol) | (trial_mix >= cfg.mtol)) & (sol >= cfg.atol) & (sol > 0))
    ratio = np.where(live, np.abs(np.asarray(delta_arr) / np.asarray(sol)), 0.0)
    z, i = np.unravel_index(np.argmax(ratio), ratio.shape)
    assert ratio[z, i] > _TIGHT_RTOL

    for run in (lambda: integ(rs), lambda: integ.runner(state, atm_static)):
        with pytest.raises(DtMinFrozenError) as err:
            run()
        message = str(err.value)
        assert f"{NETWORK.species[i]} at layer {z} has error ratio" in message
        assert float(re.search(r"error ratio (\S+) against", message).group(1)) == pytest.approx(ratio[z, i], rel=1e-2)
        assert f"rtol {_TIGHT_RTOL:g}" in message
        assert "To try: raise `rtol` above the ratio" in message

    # Under an outer jit nothing can raise and no callback sits in the program; the reason comes out in the carry.
    in_jit = jax.jit(integ.runner)
    with caplog.at_level(logging.WARNING, logger="vulcan3.outer_loop"):
        assert "callback" not in in_jit.lower(state, atm_static).as_text()
    assert "raise_if_frozen" in caplog.text
    frozen = in_jit(state, atm_static)
    with pytest.raises(DtMinFrozenError, match=r"1 of 1 columns froze at dt_min, at index \[0\].*rtol 1e-12"):
        raise_if_frozen(frozen.termination_reason, frozen.delta, rtol=_TIGHT_RTOL)

    # The CLI path writes no result for it.
    cfg_path = tmp_path / "frozen.yaml"
    dump_config(cfg, cfg_path)
    cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        with pytest.raises(DtMinFrozenError):
            cli.cli_main(["--config", str(cfg_path)])
    finally:
        os.chdir(cwd)
    assert not list(tmp_path.rglob("*.vul"))


@pytest.mark.strict_isolation
def test_the_message_repeats_the_trial_that_failed():
    """A failure outside the error estimate leaves the next dt at twice the floor; the repeat has to use the floor."""
    _, integ, rs = _column(dt_min=_FLOOR_S, rtol=1e30)
    state, atm_static = integ.prepare_runstate(rs)
    final = integ._runner(_frozen_lane(state), atm_static)
    assert int(final.termination_reason) == TERM_DT_MIN_FROZEN
    assert float(final.dt) == 2.0 * _FLOOR_S
    ratio, _, _ = integ._failed_trial(final, atm_static)
    # Dense repeat against the loop's matrix-free step, so not bitwise everywhere (todo, arm64). Twice the dt is far.
    assert float(final.delta) > 0.0
    assert float(jnp.max(ratio)) == pytest.approx(float(final.delta), rel=1e-12)


@pytest.mark.strict_isolation
def test_frozen_outranks_the_step_cap():
    """Accepted step, then two force-accepts: the freeze lands on the cap step and must not read as a plain cap exit."""
    cfg, integ, rs = _column(rtol=_TIGHT_RTOL, dt_min=_FLOOR_S)
    cfg.count_max = 2
    state, atm_static = integ.prepare_runstate(rs)
    final = integ._runner(state, atm_static)
    assert int(final.accept_count) == 3
    assert int(final.termination_reason) == TERM_DT_MIN_FROZEN


@pytest.mark.strict_isolation
def test_a_real_accept_clears_the_count():
    _, integ, rs = _column()
    state, atm_static = integ.prepare_runstate(rs)
    final = integ._runner(state._replace(keep_prev_run=jnp.int32(1)), atm_static)
    assert int(final.termination_reason) == TERM_STEP_COUNT
    assert int(final.keep_prev_run) == 0


@pytest.mark.strict_isolation
def test_a_frozen_lane_stops_alone_and_the_batch_raises(caplog):
    _, integ, rs = _column()
    state, atm_static = integ.prepare_runstate(rs)
    healthy_b = stack_integ_states([state, state])
    frozen_b = stack_integ_states([_frozen_lane(state), state])
    atm_b = stack_atm_statics([atm_static, atm_static])

    # run_batch names the lane and the failure.
    with pytest.raises(DtMinFrozenError, match=r"1 of 2 lanes froze, at \[0\]\. Lane 0: .*atom loss moves"):
        integ.run_batch(frozen_b, atm_b)
    raw = integ._vrunner(frozen_b, atm_b)
    np.testing.assert_array_equal(np.asarray(raw.termination_reason), [TERM_DT_MIN_FROZEN, TERM_STEP_COUNT])
    assert 2 <= int(raw.accept_count[0]) < _COUNT_MAX
    # The healthy lane does not feel its neighbour.
    healthy = integ.run_batch(healthy_b, atm_b)
    np.testing.assert_array_equal(np.asarray(raw.y[1]), np.asarray(healthy.y[1]))

    # run_queue: the frozen job gives up its lane to the next, and the raise names its index. Job 2 waits for a lane.
    jobs3 = (
        stack_integ_states([_frozen_lane(state), state, state]),
        stack_atm_statics([atm_static, atm_static, atm_static]),
    )
    with pytest.raises(DtMinFrozenError, match=r"1 of 3 columns froze .* at index \[0\]"):
        integ.run_queue(_ident, jobs3, n_lanes=2, out_fn=_out_fn, chunk=1)
    (out, reason, *_), _ = next(iter(integ._vrunner_queue.values()))(jobs3)
    np.testing.assert_array_equal(np.asarray(reason), [TERM_DT_MIN_FROZEN, TERM_STEP_COUNT, TERM_STEP_COUNT])
    assert 2 <= int(out[1][0]) < _COUNT_MAX

    # Under a vmap the values come back concrete, but the runner saw tracers: the helper again.
    vmapped = jax.vmap(integ.runner, in_axes=(0, ATM_STATIC_BATCH_AXES))(frozen_b, atm_b)
    with pytest.raises(DtMinFrozenError, match=r"1 of 2 columns froze .* at index \[0\]"):
        raise_if_frozen(vmapped.termination_reason)

    # Under an outer jit, as a retrieval calls it: no callback, and the helper raises right after the jitted call.
    in_jit = jax.jit(lambda jobs: integ.run_queue(_ident, jobs, n_lanes=2, out_fn=_out_fn, chunk=1))
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="vulcan3.outer_loop"):
        assert "callback" not in in_jit.lower(jobs3).as_text()
    assert "raise_if_frozen" in caplog.text
    (_, _, jit_reason, jit_delta), _ = in_jit(jobs3)
    with pytest.raises(DtMinFrozenError, match=r"1 of 3 columns froze .* at index \[0\]"):
        raise_if_frozen(jit_reason, jit_delta)
    # A healthy queue passes the helper, bitwise the same as without the jit.
    jobs2 = (healthy_b, atm_b)
    (y_jit, _, healthy_reason, _), _ = in_jit(jobs2)
    raise_if_frozen(healthy_reason)
    (y_plain, *_), _ = integ.run_queue(_ident, jobs2, n_lanes=2, out_fn=_out_fn, chunk=1)
    np.testing.assert_array_equal(np.asarray(y_jit), np.asarray(y_plain))


@pytest.mark.slow
@pytest.mark.strict_isolation
def test_forward_mode_raises_on_a_frozen_column():
    _, integ, rs = _column(rtol=_TIGHT_RTOL, dt_min=_FLOOR_S)
    state, atm_static = integ.prepare_runstate(rs)

    def zero(x):
        if jnp.issubdtype(jnp.asarray(x).dtype, jnp.floating):
            return jnp.zeros_like(x)
        return np.zeros(np.shape(x), dtype=jax.dtypes.float0)

    dstate, datm = jax.tree_util.tree_map(zero, state), jax.tree_util.tree_map(zero, atm_static)
    with pytest.raises(DtMinFrozenError, match="has error ratio"):
        integ.run_jvp(state, atm_static, dstate, datm)
    with pytest.raises(DtMinFrozenError, match="has error ratio"):
        jax.jvp(integ.runner, (state, atm_static), (dstate, datm))


@pytest.mark.strict_isolation
def test_a_hybrid_flip_gives_the_central_scheme_its_own_two_trials():
    cfg = fast_cfg(count_max=2, nz=12, Tiso=1000.0, use_vm_mol=True, use_hybrid_vm_mol=True)
    cfg.rtol, cfg.dt_min, cfg.count_min = _TIGHT_RTOL, _FLOOR_S, 1000
    integ = OuterLoop(op_jax.Ros2JAX(), legacy_io.Output(), cfg=cfg)
    state, atm_static = integ.prepare_runstate(RunState.with_pre_loop_setup(cfg))
    final = integ._runner(state, atm_static)
    # One accepted step at dt_ini, then force-accepts at dt_min: the second flips the scheme on the count cap, and two
    # more follow on central differences.
    assert int(final.termination_reason) == TERM_DT_MIN_FROZEN
    assert float(final.hybrid_use_vm) == 0.0
    assert int(final.keep_prev_run) == 2
    assert int(final.accept_count) == 5


_NO_NITROGEN = {"H2": 0.9, "He": 0.09, "H2O": 5e-3, "CO": 5e-3}


@pytest.mark.strict_isolation
def test_a_column_missing_an_atom_is_refused_at_setup_and_runs_with_loss_ex(tmp_path):
    from vulcan3 import cli
    from vulcan3.config import dump_config

    cfg = fast_cfg(count_max=_COUNT_MAX, nz=12, Tiso=1000.0, use_vm_mol=False, use_hybrid_vm_mol=False)
    cfg.ini_mix, cfg.const_mix = "const_mix", _NO_NITROGEN
    with pytest.raises(MissingAtomError, match=r"holds no N: .*Add it to `loss_ex`, or give the column some"):
        RunState.with_pre_loop_setup(cfg)

    # The CLI refuses it the same way, and writes nothing.
    cfg_path = tmp_path / "no_nitrogen.yaml"
    dump_config(cfg, cfg_path)
    cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        with pytest.raises(MissingAtomError, match="holds no N"):
            cli.cli_main(["--config", str(cfg_path)])
    finally:
        os.chdir(cwd)
    assert not list(tmp_path.rglob("*.vul"))

    # loss_ex lets it run, and it moves.
    _, integ, rs = _column(ini_mix="const_mix", const_mix=_NO_NITROGEN, loss_ex=["N"])
    final = integ(rs)
    assert int(final.params.termination_reason) == TERM_STEP_COUNT
    assert float(final.step.t) > float(integ.config.dt_min) * _COUNT_MAX


def _without_atom(integ, state, atom="N"):
    i = list(integ._atom_order).index(atom)
    return state._replace(pv=state.pv._replace(atom_ini=state.pv.atom_ini.at[i].set(0.0)))


@pytest.mark.strict_isolation
def test_a_changed_state_missing_an_atom_is_refused_by_each_runner():
    _, integ, rs = _column()
    state, atm_static = integ.prepare_runstate(rs)
    bad = _without_atom(integ, state)

    with pytest.raises(MissingAtomError, match=r"holds no N: "):
        integ.runner(bad, atm_static)
    i = list(integ._atom_order).index("N")
    no_n = rs._replace(atoms=rs.atoms._replace(atom_ini=rs.atoms.atom_ini.at[i].set(0.0)))
    with pytest.raises(MissingAtomError, match=r"holds no N: "):
        integ(no_n)
    atm_b = stack_atm_statics([atm_static, atm_static])
    with pytest.raises(MissingAtomError, match=r"holds no N in lane \[1\]"):
        integ.run_batch(stack_integ_states([state, bad]), atm_b)
    assert integ._vrunner is None, "the batch's program was built before the refusal"

    def zero(x):
        if jnp.issubdtype(jnp.asarray(x).dtype, jnp.floating):
            return jnp.zeros_like(x)
        return np.zeros(np.shape(x), dtype=jax.dtypes.float0)

    with pytest.raises(MissingAtomError, match=r"holds no N: "):
        integ.run_jvp(bad, atm_static, jax.tree_util.tree_map(zero, bad), jax.tree_util.tree_map(zero, atm_static))
    assert not integ._runner_jvp_cache, "the tangent program was built before the refusal"

    # run_queue builds its columns inside the program: the job freezes and the refusal names it after the loop.
    jobs = (stack_integ_states([state, bad]), atm_b)
    with pytest.raises(MissingAtomError, match=r"holds no N in lane \[1\]"):
        integ.run_queue(_ident, jobs, n_lanes=2, out_fn=_out_fn, chunk=1)
    # ... and whatever else stopped it first.
    capped = bad._replace(runtime_dyn=jnp.float64(1e-16))
    with pytest.raises(MissingAtomError, match=r"holds no N in lane \[1\]"):
        integ.run_queue(_ident, (stack_integ_states([state, capped]), atm_b), n_lanes=2, out_fn=_out_fn, chunk=1)

    # Under an outer jit: a freeze the helper reports, with no refusal in the program.
    (_, _, reason, delta), _ = jax.jit(lambda j: integ.run_queue(_ident, j, n_lanes=2, out_fn=_out_fn, chunk=1))(jobs)
    assert np.asarray(reason).tolist() == [TERM_STEP_COUNT, TERM_DT_MIN_FROZEN]
    with pytest.raises(DtMinFrozenError, match=r"at index \[1\]"):
        raise_if_frozen(reason, delta)


def test_refuse_missing_atoms_reads_only_a_zero_total():
    refuse_missing_atoms(np.array([1.0, 1e-300]), ["H", "N"])
    refuse_missing_atoms(np.array([[1.0, 2.0], [3.0, 4.0]]), ["H", "N"])
    with pytest.raises(MissingAtomError, match=r"holds no C, H in lane \[0, 2\]"):
        refuse_missing_atoms(np.array([[0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 0.0, 1.0]]), ["C", "H", "N"])


def test_the_count_is_kept_at_the_floor_only():
    from types import SimpleNamespace

    from vulcan3._loop_body import _decide_step

    statics = SimpleNamespace(loss_eps=1.0, rtol_accept=0.05, dt_var_min=0.5, dt_min=1e-14, batch_max_retries=3)

    def run_after(run, *, dt=1e-14, retry=0, delta=0.08, sol=1.0):
        s = SimpleNamespace(
            atom_loss_prev=jnp.zeros(1),
            dt=jnp.float64(dt),
            retry_count=jnp.int32(retry),
            keep_prev_run=jnp.int32(run),
            t=jnp.float64(0.0),
            y_prev=jnp.ones((1, 1)),
            ymix=jnp.ones((1, 1)),
        )
        y = jnp.full((1, 1), sol)
        return int(_decide_step(statics, s, y, y, y, jnp.zeros(1), jnp.float64(delta)).keep_prev_run)

    assert run_after(0) == 1  # a kept force-accept at dt_min
    assert run_after(1) == 2  # and its repeat
    assert run_after(1, delta=0.01) == 0  # a real accept clears it
    assert run_after(1, dt=1e-12) == 1  # a rejection above the floor leaves it
    assert run_after(0, dt=1.5e-14) == 0  # a force-accept just above dt_min: the next dt is another trial
    assert run_after(0, dt=1e-12, retry=3) == 0  # the retry budget, above the floor: another dt follows
    assert run_after(0, sol=jnp.nan) == 0  # a non-finite solve goes through and ends as non-finite


# The network is import-frozen, so this column needs its own process. Isothermal, const_mix, no photolysis; atol is
# above the species this start creates from nothing, so that S4 alone sets the step.
_FAST_SULFUR_CHILD = r"""
import os, sys
from pathlib import Path

repo = Path(sys.argv[1])
sys.path.insert(0, str(repo / "src"))
os.chdir(repo / "src" / "vulcan3")

from vulcan3.config import default_config

cfg = default_config()
cfg.atom_list = ["H", "O", "C", "N", "S"]
cfg.atm_type = "isothermal"; cfg.Tiso = 2900.0
cfg.atm_base = "H2"; cfg.use_moldiff = False
cfg.use_Kzz = True; cfg.Kzz_prof = "const"; cfg.const_Kzz = 1e9; cfg.use_vz = False
cfg.use_photo = False; cfg.use_ion = False; cfg.use_condense = False
cfg.ini_mix = "const_mix"
cfg.const_mix = {"H2": 0.85, "He": 0.149, "H2O": 5e-4, "CO": 2e-4, "CH4": 1e-5, "N2": 1e-4, "H2S": 3e-4, "S2": 3e-4,
                 "S4": 1e-14}
cfg.nz = 12; cfg.P_b = 7.6e6; cfg.P_t = 1e6
cfg.use_topflux = False; cfg.use_botflux = False
cfg.use_fix_sp_bot = {}; cfg.use_ini_cold_trap = False; cfg.use_print_prog = False
cfg.rtol = 0.05; cfg.atol = 1e4; cfg.count_min = 6; cfg.count_max = 6

from vulcan3 import RunState, legacy_io, op_jax
from vulcan3.chem_funs import spec_list
from vulcan3.outer_loop import OuterLoop
from vulcan3.state import TERM_STEP_COUNT

i_s4 = spec_list.index("S4")


def solve():
    integ = OuterLoop(op_jax.Ros2JAX(), legacy_io.Output(), cfg=cfg)
    state, atm_static = integ.prepare_runstate(RunState.with_pre_loop_setup(cfg))
    return integ, state, atm_static, integ._runner(state, atm_static)


_, state, _, final = solve()
assert int(final.termination_reason) == TERM_STEP_COUNT, (int(final.termination_reason), float(final.delta), cfg.dt_min)
assert float(final.delta) <= cfg.rtol
assert float(final.y[0, i_s4]) > 10.0 * float(state.y[0, i_s4])

cfg.dt_min = 1e-14
_, state, _, final = solve()
assert int(final.termination_reason) == TERM_STEP_COUNT, (int(final.termination_reason), float(final.delta), cfg.dt_min)
assert float(final.y[0, i_s4]) > 10.0 * float(state.y[0, i_s4])
"""


def test_a_short_sulfur_lifetime_can_step_at_the_default_floor():
    """S4 + M -> S2 + S2 + M at 2900 K and 7.6 bar. S4 starts 50 times below its balance with S2.

    From K_eq alone the S4 lifetime is 4e-14 s, and the column froze on S4 at master's floor 1e-14 s. The collision
    limit holds it to 1/(K_COLL M), 5e-11 s, so the column steps at the default floor and at 1e-14 s (notes
    "Differences from VULCAN master").
    """
    run_child(_FAST_SULFUR_CHILD, network="thermo/SNCHO_photo_network_2025.txt", label="fast sulfur lifetime")
