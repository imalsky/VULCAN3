"""The collision limit on thermal rate pairs (rates_jax.apply_collision_limit; notes "Parity ledger").

The reverse of CH + M -> C + H + M, as a bimolecular-equivalent rate k_r M, is about 6e-5 cm3/s at 300 K and 1 bar
(the dissociation fit extrapolated far below its 2500-3800 K range): the limit scales the pair down to 1e-9 and keeps
k_f / k_r. At 1500 K and 1 bar no rate of the network reaches the limit; at 3000 K the forward fit of NCO + NH2 does.
M counts as a collision partner, so the dissociation S4 + M -> S2 + S2 + M (k from K_eq, about 9e-7 cm3/s at 2620 K) is
capped too.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def _rates(T, f="thermo/NCHO_photo_network.txt"):
    from vulcan3 import rates_jax
    from vulcan3._paths import resolve_data_path
    from vulcan3.gibbs import load_nasa9
    from vulcan3.network import parse_network

    net = parse_network(resolve_data_path(f))
    nasa9, _ = load_nasa9(net.species, rates_jax.thermo_dir(f))
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


def test_collision_limit_caps_a_dissociation_with_m():
    net, raw, cap, _M = _rates(2000.0, "thermo/SNCHO_photo_network_2025.txt")
    i = next(j for j, eq in net.Rf.items() if eq.replace(" ", "") == "S2+S2+M->S4+M")
    assert raw[i + 1, 0] > 1e-8  # S4 + M: more than ten times the limit per collision pair
    np.testing.assert_allclose(cap[i + 1, 0], 1e-9, rtol=1e-12)
    np.testing.assert_allclose(cap[i, 0] / cap[i + 1, 0], raw[i, 0] / raw[i + 1, 0], rtol=1e-12)


def _s4_pair_rates(remove_list=None):
    """Return the S2 + S2 + M slot index and the built rates at one 1-bar layer."""
    from vulcan3 import rates_jax
    from vulcan3._paths import resolve_data_path
    from vulcan3.gibbs import load_nasa9
    from vulcan3.network import parse_network

    f = "thermo/SNCHO_photo_network_2025.txt"
    net = parse_network(resolve_data_path(f))
    nasa9, _ = load_nasa9(net.species, rates_jax.thermo_dir(f))
    i = next(j for j, eq in net.Rf.items() if eq.replace(" ", "") == "S2+S2+M->S4+M")
    M = jnp.asarray([1e6 / (1.380649e-16 * 2000.0)])

    def build(T):
        return rates_jax.build_rate_array(net, jnp.reshape(T, (1,)), M, nasa9, remove_list)

    return net, nasa9, i, build


def test_collision_limit_tangent_keeps_the_pair_ratio_under_the_cap():
    # Capped, k_r is the constant K_COLL, so d ln k_r / dT = 0 and d ln k_f / dT = d ln K_eq / dT.
    import jax

    from vulcan3 import rates_jax

    net, nasa9, i, build = _s4_pair_rates()
    T0 = jnp.asarray(2000.0)
    k, dk = jax.jvp(build, (T0,), (jnp.asarray(1.0),))
    dlnk = np.asarray(dk[:, 0] / k[:, 0])

    def ln_K(T):
        T1 = jnp.reshape(T, (1,))
        return jnp.log(rates_jax.K_eq_array(net, rates_jax.gibbs_sp_vector(nasa9, T1), T1)[i, 0])

    assert abs(dlnk[i + 1]) < 1e-12 * abs(dlnk[i])  # zero up to rounding in k_r * (K_COLL / q)
    np.testing.assert_allclose(dlnk[i], float(jax.grad(ln_K)(T0)), rtol=1e-10)
    h = 1e-3
    fd = (np.log(float(build(T0 + h)[i, 0])) - np.log(float(build(T0 - h)[i, 0]))) / (2 * h)
    np.testing.assert_allclose(dlnk[i], fd, rtol=1e-6)


def test_a_removed_slot_does_not_scale_its_partner():
    # Removing the super-collisional S4 + M slot leaves its association at the uncapped value, as master's
    # rev_rate zeroes a removed reverse before the oracle copy applies the limit.
    _net, _nasa9, i, build = _s4_pair_rates()
    _net, _nasa9, _i, build_rm = _s4_pair_rates(remove_list=[i + 1])
    full = np.asarray(build(jnp.asarray(2000.0)))
    rm = np.asarray(build_rm(jnp.asarray(2000.0)))
    assert full[i, 0] < rm[i, 0]  # with the slot kept, the pair is scaled down
    assert rm[i + 1, 0] == 0.0
    raw_fwd = np.asarray(_rates(2000.0, "thermo/SNCHO_photo_network_2025.txt")[1])[i, 0]
    assert rm[i, 0] == raw_fwd
