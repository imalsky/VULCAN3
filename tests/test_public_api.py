"""Every name the consumers (vulcan-forward, vulcan-retrieval, jwst-transit-authority) use must resolve."""

from __future__ import annotations

import importlib

import pytest

# module -> names read by a consumer, deprecated private ones included until the consumers have moved.
_CONSUMED = {
    "vulcan3": ["__file__", "__version__", "conv_normal", "load_config", "validate_overrides", "RunState"],
    "vulcan3.paths": ["resolve_data_path"],
    "vulcan3._paths": ["resolve_data_path"],
    "vulcan3.atm_jax": ["make_physical_inputs", "mol_diff", "_mol_diff"],
    "vulcan3.atm_refresh": ["update_mu_dz_jax"],
    "vulcan3.atm_setup": [
        "VISCOSITY_TABLE",
        "_VISCOSITY_TABLE",
        "compute_mu_dz_g",
        "compute_pico",
        "sat_p_jax",
        "settling_velocity_jax",
        "kzz_profile_jax",
        "surface_gravity",
    ],
    "vulcan3.chem_funs": ["NETWORK", "NET_JAX", "_NETWORK", "_NET_JAX"],
    "vulcan3.composition": ["atom_list", "compo", "compo_array", "compo_row"],
    "vulcan3.conden": ["build_conden_profile", "make_conden_spec"],
    "vulcan3.gibbs": ["load_nasa9"],
    "vulcan3.ini_abun": ["column_atoms", "element_vector", "eq_seed", "ratio_indices"],
    "vulcan3.jax_step": ["make_atm_static"],
    "vulcan3.legacy_io": ["Output"],
    "vulcan3.network": ["parse_network"],
    "vulcan3.op_jax": ["Ros2JAX"],
    "vulcan3.outer_loop": [
        "OuterLoop",
        "ATM_STATIC_BATCH_AXES",
        "_ATM_STATIC_BATCH_AXES",
        "conv_normal",
        "certificate_failures",
    ],
    "vulcan3.phy_const": ["G_grav", "kb"],
    "vulcan3.rates_jax": ["build_rate_array", "thermo_dir", "_thermo_dir"],
    "vulcan3.runtime_validation": ["rate_temp_range_exposure"],
    "vulcan3.state": ["RunState", "legacy_view"],
    "vulcan3.steady_state_grad": [
        "BODY_MAP_DT_CANDIDATES",
        "audit_adjoint_scope",
        "make_body_terms",
        "make_photo_recompute_k",
        "steady_state_input_sensitivity",
        "steady_state_reaction_sensitivity",
    ],
}
_OUTER_LOOP_ATTRS = [
    "run_batch",
    "run_queue",
    "run_jvp",
    "prepare_runstate",
    "runner",
    "config",
    "ensure_runner",
    "pack_state",
    "build_refresh_static",
    "compo_arr",
    "photo_static",
    "statics",
    "_ensure_runner",
    "_pack_state_from_runstate",
    "_build_refresh_static",
    "_build_statics",
]


@pytest.mark.parametrize(("module", "names"), _CONSUMED.items(), ids=list(_CONSUMED))
def test_consumed_names_resolve(module, names):
    mod = importlib.import_module(module)
    missing = [name for name in names if not hasattr(mod, name)]
    assert not missing, f"{module} lost {missing}: move the consumers before removing a name (docs/api.md)"


@pytest.mark.parametrize("name", [n for names in _CONSUMED.values() for n in names if not n.startswith("_")])
def test_public_names_are_listed(name):
    module = next(m for m, names in _CONSUMED.items() if name in names and not m.endswith("._paths"))
    assert name in importlib.import_module(module).__all__ or name in {"__file__", "RunState"}


def test_outer_loop_and_solver_attributes():
    from vulcan3.op_jax import Ros2JAX
    from vulcan3.outer_loop import OuterLoop

    missing = [name for name in _OUTER_LOOP_ATTRS if not hasattr(OuterLoop, name)]
    assert not missing, f"OuterLoop lost {missing}"
    solver = Ros2JAX()
    assert solver.photo_static is None
    solver.photo_static = 1
    assert solver._photo_static == 1
    assert hasattr(solver, "naming_solver")
    assert not hasattr(OuterLoop, "does_not_exist")


def test_private_aliases_are_the_public_objects():
    from vulcan3 import atm_jax, atm_setup, chem_funs, outer_loop, rates_jax

    assert chem_funs.NETWORK is chem_funs._NETWORK
    assert chem_funs.NET_JAX is chem_funs._NET_JAX
    assert outer_loop.ATM_STATIC_BATCH_AXES is outer_loop._ATM_STATIC_BATCH_AXES
    assert atm_setup.VISCOSITY_TABLE is atm_setup._VISCOSITY_TABLE
    assert atm_jax.mol_diff is atm_jax._mol_diff
    assert rates_jax.thermo_dir is rates_jax._thermo_dir
