"""Typed pre-loop pytree schema.

The conversion between a legacy (var, atm, para) and a RunState, its refusals, and the `.vul` writer's parameter keys.
"""

from __future__ import annotations

import numpy as np


def test_store_inputs_carry_the_source_values(hd189_state):
    """Each field holds its own source's values, so a cross-wired field fails."""
    from vulcan3.state import AtmInputs, _inputs_from_store

    var, atm = hd189_state.var, hd189_state.atm
    atm_inputs, rate, photo = _inputs_from_store(var, atm)

    for name in AtmInputs._fields:
        np.testing.assert_array_equal(
            np.asarray(getattr(atm_inputs, name)), np.asarray(getattr(atm, name)), err_msg=name
        )
    np.testing.assert_array_equal(np.asarray(rate.k), np.asarray(var.k_arr))
    np.testing.assert_array_equal(np.asarray(photo.sflux_top), np.asarray(var.sflux_top).reshape(-1))
    assert photo.sflux_top.size > 0


def test_runstate_round_trips_through_the_legacy_view(hd189_state):
    """`legacy_view` and `runstate_from_store` are inverses on the slots a run starts from."""
    import jax

    from vulcan3.state import legacy_view, runstate_from_store

    rs = runstate_from_store(hd189_state.var, hd189_state.atm, hd189_state.para)
    back = runstate_from_store(*legacy_view(rs))
    for slot in ("atm", "step", "params", "atoms", "fix_species"):
        for a, b in zip(
            jax.tree_util.tree_leaves(getattr(rs, slot)), jax.tree_util.tree_leaves(getattr(back, slot)), strict=True
        ):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=slot)


def test_conversion_and_entry_points_refuse_what_they_cannot_honour(hd189_state, monkeypatch):
    """Each refusal stands where a wrong input would become a silent zero, a wrong network or a NoneType call."""
    import copy

    import pytest

    from vulcan3 import legacy_io, op_jax, rates_jax
    from vulcan3._pack_state import _initial_photo_carry
    from vulcan3.config import default_config, load_config
    from vulcan3.outer_loop import OuterLoop
    from vulcan3.runtime_validation import RuntimeConfigError, validate_runtime_config
    from vulcan3.state import runstate_from_store

    var, atm, para = hd189_state.var, hd189_state.atm, hd189_state.para
    rs = runstate_from_store(var, atm, para)

    # A missing element total reads as a refusal, not 0, which `atom_loss` would then divide by.
    short = copy.copy(var)
    short.atom_ini = {k: v for k, v in var.atom_ini.items() if k != "H"}
    with pytest.raises(ValueError, match=r"var.atom_ini has no value for \['H'\]"):
        runstate_from_store(short, atm, para)

    # The legacy view carries no fix_species pin, so a frozen run cannot come back through it.
    monkeypatch.setattr(default_config(), "fix_species", ["H2O"])
    frozen = copy.copy(para)
    frozen.fix_species_start = True
    with pytest.raises(ValueError, match="has already frozen"):
        runstate_from_store(var, atm, frozen)
    monkeypatch.setattr(default_config(), "fix_species", [])

    # Rates of the frozen network must not be built with another network's thermo.
    with pytest.raises(ValueError, match="import-locked"):
        rates_jax.setup_var_k(load_config("default", network="thermo/CHO_photo_network.txt"), var, atm)

    integ = OuterLoop(op_jax.Ros2JAX(), legacy_io.Output())
    with pytest.raises(RuntimeError, match="statics are not built"):
        integ.pack_state(rs)
    with pytest.raises(RuntimeError, match="run_jvp called before"):
        integ.run_jvp(None, None, None, None)
    integ._photo_static = rs.photo_static or object()
    with pytest.raises(ValueError, match="no photo_runtime"):
        _initial_photo_carry(integ, rs._replace(photo_runtime=None))

    with pytest.raises(RuntimeConfigError, match="fix_species"):
        validate_runtime_config(load_config("default", fix_species=None))


def test_runstate_output_parameter_schema(hd189_state):
    """RunState-backed `.vul` output exposes VULCAN-master parameter keys."""
    from vulcan3 import legacy_io
    from vulcan3.config import default_config

    vulcan_cfg = default_config()
    from vulcan3.state import runstate_from_store

    rs = runstate_from_store(hd189_state.var, hd189_state.atm, hd189_state.para)
    _, _, param = legacy_io._synthesize_save_dicts(
        rs._replace(photo_static=getattr(hd189_state.solver, "_photo_static", None)),
        vulcan_cfg,
    )

    expected = {
        "nega_y",
        "small_y",
        "delta",
        "count",
        "nega_count",
        "loss_count",
        "delta_count",
        "end_case",
        "solver_str",
        "switch_final_photo_frq",
        "where_varies_most",
        "pic_count",
        "fix_species_start",
        "tableau20",
        "start_time",
    }
    assert expected <= set(param), f"missing parameter keys: {expected - set(param)}"
    assert param["solver_str"] == "solver"
    assert np.asarray(param["where_varies_most"]).shape == hd189_state.var.y.shape
    assert len(param["tableau20"]) == 20
