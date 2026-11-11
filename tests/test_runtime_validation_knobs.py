"""`runtime_validation`'s bound checks refuse their silently wrong values."""

from __future__ import annotations

import pytest

_VALID = {
    "atom_list": [],
    "loss_ex": [],
    "nz": 10,
    "P_b": 1e9,
    "P_t": 1e-2,
    "Mp": 1.9e30,
    "Rp": 8e9,
    "rtol": 0.25,
    "geom_conv_tol": 1e-3,
    "element_budget_tol": 1e-2,
    "slope_min_hp_frac": 0.1,
    "slope_min_cap": 1e-8,
    "slope_min_floor": 1e-10,
    "photo_lambda_min": 2.0,
    "photo_lambda_max": 700.0,
    "dt_min": 1e-14,
    "dt_max": 1e15,
    "sl_angle": 1.0,
    "humidity": 0.5,
    "count_min": 120,
    "count_max": 10000,
    "update_frq": 100,
    "conv_step": 500,
    "batch_max_retries": 110,
    "dt_var_min": 0.5,
    "dttry": 1.0e-10,
    "ini_update_photo_frq": 100,
    "final_update_photo_frq": 5,
}
_NAN = float("nan")


@pytest.mark.parametrize(
    "bad",
    [
        {"nz": 2},
        {"P_b": 1e-3},
        {"P_t": 0.0},
        {"Mp": _NAN},
        {"Rp": -1.0},
        {"rtol": float("inf")},
        {"geom_conv_tol": 0.0},
        {"element_budget_tol": _NAN},
        {"element_budget_tol": -float("inf")},
        {"slope_min_hp_frac": 0.0},
        {"slope_min_cap": 1e-11},
        # each alone fails only the finite-and-positive check, so that check is what refuses it
        {"slope_min_cap": float("inf")},
        {"slope_min_floor": -1e-10},
        # each above its 10x guard, the floor still under the cap
        {"slope_min_cap": 2e-7},
        {"slope_min_floor": 5e-9},
        {"photo_lambda_max": 1.0},
        {"photo_lambda_min": -2.0},
        {"photo_lambda_max": float("inf")},
        {"dt_min": 1e16},
        {"sl_angle": 58.0},
        {"batch_max_retries": 64},
        {"batch_max_retries": 111},
        {"dt_var_min": 0.9},
        {"dt_var_min": 1.0},
        {"dt_var_min": 0.0},
        {"dt_var_min": _NAN},
        {"dt_min": 1.0e-30},
        {"dt_min": 0.0},
        {"humidity": 50.0},
        *(
            dict(**{k: v})
            for k in ("count_min", "batch_max_retries", "ini_update_photo_frq", "final_update_photo_frq")
            for v in (0, 1.5, _NAN, 2**31)
        ),
    ],
    ids=repr,
)
def test_silently_wrong_values_are_refused(bad):
    """Each value refused alone; the valid baseline passes clean."""
    from types import SimpleNamespace

    from vulcan3.runtime_validation import _validate_numerical_bounds

    assert _validate_numerical_bounds(SimpleNamespace(**_VALID)) == []
    # .inf is the budget term's documented off switch.
    off = SimpleNamespace(**{**_VALID, "element_budget_tol": float("inf")})
    assert _validate_numerical_bounds(off) == []
    assert len(_validate_numerical_bounds(SimpleNamespace(**{**_VALID, **bad}))) == 1


@pytest.mark.parametrize(
    ("flags", "expect"),
    [
        ({}, None),
        ({"use_vm_mol": True, "use_hybrid_vm_mol": True}, None),
        ({"use_moldiff": False}, None),
        ({"use_moldiff": False, "use_vm_mol": True}, "use_vm_mol=True needs use_moldiff"),
        ({"use_moldiff": False, "use_settling": True}, "use_settling=True needs use_moldiff"),
        ({"use_hybrid_vm_mol": True}, "use_hybrid_vm_mol=True needs use_vm_mol"),
    ],
)
def test_a_diffusion_option_the_run_would_ignore_is_refused(flags, expect):
    from types import SimpleNamespace

    from vulcan3.runtime_validation import _validate_diffusion_flags

    cfg = SimpleNamespace(
        **{"use_moldiff": True, "use_vm_mol": False, "use_settling": False, "use_hybrid_vm_mol": False, **flags}
    )
    errors = _validate_diffusion_flags(cfg)
    assert errors == [] if expect is None else any(expect in e for e in errors), errors
