"""`tools/compare_vul.py` reports a difference and refuses what it cannot compare."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from tools.compare_vul import main

_PCO_BAR = np.array([1.0, 1e-3, 1e-9])
_YMIX = np.array([[1e-3], [1e-4], [1e-5]])


def _vul(path, ymix=_YMIX, pco_bar=_PCO_BAR):
    data = {"atm": {"pco": pco_bar * 1e6}, "variable": {"species": ["H2O"], "ymix": ymix}}
    path.write_bytes(pickle.dumps(data))
    return str(path)


def _compare(tmp_path, **test):
    return main(
        [_vul(tmp_path / "ref.vul"), _vul(tmp_path / "test.vul", **test), "--species", "H2O", "--max-dex", "0.01"]
    )


def test_agreement_and_a_difference(tmp_path):
    assert _compare(tmp_path) == 0
    assert _compare(tmp_path, ymix=_YMIX * 2.0) == 1


@pytest.mark.parametrize(
    ("test", "match"),
    [
        ({"ymix": _YMIX * np.nan}, "non-finite"),
        ({"ymix": _YMIX * 1e-20}, "no cell above"),
        ({"pco_bar": _PCO_BAR * np.array([1.0, 1.0, 5.0])}, "pressure grids"),
        ({"pco_bar": _PCO_BAR[:2], "ymix": _YMIX[:2]}, "pressure grids"),
    ],
)
def test_what_cannot_be_compared_is_refused(tmp_path, test, match):
    with pytest.raises(SystemExit, match=match):
        _compare(tmp_path, **test)
