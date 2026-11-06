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
    "dt_min": 1e-14,
    "dt_max": 1e15,
    "sl_angle": 1.0,
    "humidity": 0.5,
    "count_min": 120,
    "count_max": 10000,
    "update_frq": 100,
    "conv_step": 500,
    "batch_max_retries": 110,
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
        {"dt_min": 1e16},
        {"sl_angle": 58.0},
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
