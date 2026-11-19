"""`diagnose.reaction_rates`: production minus loss over all reactions is the chemistry RHS of the species.

The saved `.vul` is built from the shipped HD209 state fixture, in the layout `legacy_io` writes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
RTOL = 1e-10


@pytest.fixture(scope="module")
def vul() -> dict:
    """Return a `.vul` dict (variable, atm) built from `tests/data/state_hd209.npz`."""
    from vulcan3.config import default_config
    from vulcan3.network import parse_network

    state = np.load(ROOT / "tests" / "data" / "state_hd209.npz")
    net = parse_network(default_config().network)
    if list(state["species"]) != list(net.species):
        pytest.fail("state_hd209.npz does not match the default network; rebuild it with gen_fixtures.py --force")
    k_arr = state["k_arr"]
    return {
        "variable": {
            "species": list(net.species),
            "y": state["y"],
            "Rf": dict(net.Rf),
            "k": {i: k_arr[i] for i in range(1, net.nr + 1)},
        },
        "atm": {"n_0": state["M"]},
    }


@pytest.mark.parametrize("species", ["CH4", "H2O", "OH"])
def test_net_flux_is_the_chemistry_rhs(vul, species):
    """Sum of (production - loss) flux over every reaction equals `chem_rhs_codegen` at all layers."""
    from vulcan3.chem_funs import chem_rhs_codegen, nr
    from vulcan3.diagnose import reaction_rates

    var = vul["variable"]
    k = np.stack([np.zeros_like(var["k"][1])] + [var["k"][i] for i in range(1, nr + 1)])
    rhs = np.asarray(chem_rhs_codegen(var["y"], vul["atm"]["n_0"], k))[:, var["species"].index(species)]

    r = reaction_rates(vul, species, top=None)
    net_flux = r.production.flux.sum(axis=0) - r.loss.flux.sum(axis=0)

    scale = np.abs(r.production.flux).sum(axis=0) + np.abs(r.loss.flux).sum(axis=0)
    assert np.all(scale > 0)
    # Production and loss nearly cancel in places, so the tolerance is relative to each layer's gross flux.
    err = np.abs(net_flux - rhs) / scale
    assert err.max() < RTOL, f"max |net - rhs| / gross flux = {err.max():.2e} at layer {err.argmax()}"


def test_ranking_and_layers(vul):
    """`top` keeps the strongest reactions, by maximum flux over the requested layers."""
    from vulcan3.diagnose import reaction_rates

    r = reaction_rates(vul, "CH4", slice(2, 6), top=3)
    assert list(r.layers) == [2, 3, 4, 5]
    for kind in (r.production, r.loss):
        assert kind.rate.shape == (3, 4)
        peak = kind.flux.max(axis=1)
        assert np.all(np.diff(peak) <= 0)
    with pytest.raises(ValueError, match="not a species"):
        reaction_rates(vul, "NOT_A_SPECIES")
