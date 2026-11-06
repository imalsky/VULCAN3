"""The collision limit on thermal rate pairs (rates_jax.apply_collision_limit; notes "Parity ledger").

The reverse of CH + M -> C + H + M, as a bimolecular-equivalent rate k_r M, is about 6e-5 cm3/s at 300 K and 1 bar
(the dissociation fit extrapolated far below its 2500-3800 K range): the limit scales the pair down to 1e-9 and keeps
k_f / k_r. At 1500 K and 1 bar no rate of the network reaches the limit; at 3000 K the forward fit of NCO + NH2 does.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def _rates(T):
    from vulcan3 import rates_jax
    from vulcan3._paths import resolve_data_path
    from vulcan3.gibbs import load_nasa9
    from vulcan3.network import parse_network

    f = "thermo/NCHO_photo_network.txt"
    net = parse_network(resolve_data_path(f))
    nasa9, _ = load_nasa9(net.species, rates_jax._thermo_dir(f))
    Tj = jnp.asarray([T])
    M = jnp.asarray([1e6 / (1.380649e-16 * T)])
    k_fwd = rates_jax.compute_forward_k(net, Tj, M)
    K = rates_jax.K_eq_array(net, rates_jax.gibbs_sp_vector(nasa9, Tj), Tj)
    raw = rates_jax.fill_reverse_k(net, k_fwd, K)
    return net, np.asarray(raw), np.asarray(rates_jax.apply_collision_limit(net, raw, M)), float(M[0])


def test_collision_limit_caps_the_ch_pair_and_keeps_its_ratio():
    net, raw, cap, M = _rates(300.0)
    i = next(j for j, eq in net.Rf.items() if eq.replace(" ", "") == "CH+M->C+H+M")
    assert raw[i + 1, 0] * M > 1e-6  # super-collisional before
    np.testing.assert_allclose(cap[i + 1, 0] * M, 1e-9, rtol=1e-12)
    np.testing.assert_allclose(cap[i, 0] / cap[i + 1, 0], raw[i, 0] / raw[i + 1, 0], rtol=1e-12)


def test_collision_limit_leaves_ordinary_rates_alone():
    net, raw, cap, _M = _rates(1500.0)
    changed = np.flatnonzero(np.any(cap != raw, axis=1))
    assert len(changed) == 0, [net.Rf.get(j, j) for j in changed]


def test_collision_limit_also_caps_a_forward_rate():
    net, raw, cap, _M = _rates(3000.0)  # NCO + NH2 -> HNCO + NH: its fit gives 2e-9 at 3000 K
    i = next(j for j, eq in net.Rf.items() if eq.replace(" ", "") == "NCO+NH2->HNCO+NH")
    assert raw[i, 0] > 1e-9
    np.testing.assert_allclose(cap[i, 0], 1e-9, rtol=1e-12)
    np.testing.assert_allclose(cap[i, 0] / cap[i + 1, 0], raw[i, 0] / raw[i + 1, 0], rtol=1e-12)
