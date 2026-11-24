"""The cold-trap level the `fix_species` pin takes (`_loop_body.activate_fix_species`)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest

from vulcan3._loop_body import activate_fix_species

NZ = 8
TOP = 5  # the saturation-minimum layer
SAT_MIX = np.array([1e-2, 1e-3, 1e-4, 1e-5, 2e-6, 1e-6, 3e-6, 1e-4])  # minimum at TOP


class _State(NamedTuple):
    y: jnp.ndarray
    pv: SimpleNamespace
    fix_y: jnp.ndarray
    fix_mask: jnp.ndarray
    vs: jnp.ndarray
    fix_species_started: jnp.ndarray = jnp.bool_(False)
    fix_pfix_idx: jnp.ndarray = jnp.zeros((1,), dtype=jnp.int32)
    rtol: jnp.ndarray = jnp.float64(0.05)


def _level(ratio_at_top: float, relax_top: bool, ratio_below: float = 1.01) -> int:
    """Pin level for one gas: `ratio_below` of saturation under TOP, `ratio_at_top` at TOP, dry above."""
    n_0 = np.full(NZ, 1e18)
    ratio = np.where(np.arange(NZ) < TOP, ratio_below, 0.1)
    ratio[TOP] = ratio_at_top
    y = np.zeros((NZ, 1))
    y[:, 0] = ratio * SAT_MIX * n_0
    statics = SimpleNamespace(
        fix_species_idx=jnp.asarray([0], dtype=jnp.int32),
        fix_species_from_coldtrap_lev=True,
        fix_species_wholecol=jnp.asarray([False]),
        fix_species_use_relax_top=jnp.asarray([relax_top]),
        post_conden_rtol=0.15,
    )
    pv = SimpleNamespace(
        n_0=jnp.asarray(n_0), fix_species_sat_mix=jnp.asarray(SAT_MIX[None, :]), c_nh3_conden_top=jnp.int32(TOP)
    )
    s = _State(
        y=jnp.asarray(y),
        pv=pv,
        fix_y=jnp.zeros((NZ, 1)),
        fix_mask=jnp.zeros((NZ, 1), dtype=bool),
        vs=jnp.zeros((NZ - 1, 1)),
    )
    out = activate_fix_species(statics, s)  # type: ignore[arg-type]  # duck-typed stand-ins for _Statics and the carry
    level = int(out.fix_pfix_idx[0])
    assert np.array_equal(np.asarray(out.fix_mask[:, 0]), np.arange(NZ) < level)
    return level


@pytest.mark.parametrize("ratio_at_top", [0.998, 1.0, 1.026])
def test_relaxed_nh3_level_is_the_relaxation_top(ratio_at_top: float) -> None:
    """The relaxation holds the top layer within a few percent of saturation; the level must not depend on its side."""
    assert _level(ratio_at_top, relax_top=True) == TOP


def test_other_gases_keep_the_snapshot_test() -> None:
    """VULCAN 2's rule (op.py:882-889): the saturation minimum among the layers at or above saturation."""
    assert _level(1.026, relax_top=False) == TOP
    assert _level(0.998, relax_top=False) == TOP - 1


def test_a_gas_that_condenses_nowhere_is_not_pinned() -> None:
    """VULCAN 2's level 0 (op.py:891-893), for relaxed NH3 too."""
    assert _level(0.9, relax_top=True, ratio_below=0.9) == 0
    assert _level(0.9, relax_top=False, ratio_below=0.9) == 0
