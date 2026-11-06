"""`rates_jax.build_rate_array` must stay differentiable.

The rate build is one implementation shared by setup and gradients (parity
against master is `test_gibbs.py`, and against the legacy
chain `test_read_rate.py`). What is only testable here is that the tangents
are right: a non-finite jvp w.r.t. T or w.r.t. an Arrhenius coefficient would
silently wreck every T-profile / rate-uncertainty sensitivity, and a severed
one would read as zero.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

RTOL = 1e-12  # the same rate expression evaluated two ways
FD_RTOL = 1e-5  # a forward-mode tangent against a central difference
FD_ROUNDOFF = 1e-9  # the difference's own roundoff, relative to the rate
# JAX's tangent of `kf / K` holds K**-2, which underflows above K ~ 1e154: such a reverse rate (k < 1e-150 cm3/s, nil
# against any density) has a zero tangent. Measured on the cool column below, at k 1e-166 to 1e-232.
NIL_RATE = 1e-150


def _check_tangent(fn, x, dx, eps):
    """`jvp(fn)(x)[dx]` matches a central difference cell by cell, and is not zero everywhere."""
    k, dk = jax.jvp(fn, (x,), (dx,))
    fd = (np.asarray(fn(x + eps * dx)) - np.asarray(fn(x - eps * dx))) / (2.0 * eps)
    k, dk = np.asarray(k), np.asarray(dk)
    assert np.all(np.isfinite(dk))
    assert np.any(dk != 0.0), "every tangent is zero: the graph is cut"
    bad = (np.abs(dk - fd) > FD_RTOL * np.abs(dk) + FD_ROUNDOFF * np.abs(k)) & (np.abs(k) > NIL_RATE)
    assert not bad.any(), f"{int(bad.sum())} cells off, first {np.argwhere(bad)[0]}"


def test_rate_build_tangents_match_finite_differences(hd189_state):
    from vulcan3.config import default_config

    vulcan_cfg = default_config()
    from vulcan3 import network as net_mod, rates_jax
    from vulcan3._paths import resolve_data_path
    from vulcan3.gibbs import load_nasa9

    atm = hd189_state.atm
    network = net_mod.parse_network(str(resolve_data_path(vulcan_cfg.network)))

    T = jnp.asarray(atm.Tco, dtype=jnp.float64)
    M = jnp.asarray(atm.M, dtype=jnp.float64)
    nasa9, _ = load_nasa9(network.species, rates_jax.thermo_dir(vulcan_cfg.network))
    remove_list = vulcan_cfg.remove_list

    def k_of_T(Tj):
        return rates_jax.build_rate_array(network, Tj, M, nasa9, remove_list=remove_list)

    _check_tangent(k_of_T, T, jnp.ones_like(T), 1e-3)

    # The same on a cool column where the three Moses+2005 caps fire: the cap branch is a `where`, so its tangent is
    # checked separately. The cap formulas themselves are pinned in test_read_rate.py.
    T_cool = jnp.full_like(T, 250.0)

    def k_capped_of_T(Tj):
        return rates_jax.build_rate_array(network, Tj, M, nasa9, remove_list=remove_list, use_lowT_caps=True)

    kf = rates_jax.compute_forward_k(network, T_cool, M)
    kf_capped = rates_jax.apply_lowT_caps(network, kf, T_cool, M)
    changed = np.nonzero(np.any(np.asarray(kf_capped) != np.asarray(kf), axis=1))[0]
    assert changed.size >= 1, "lowT caps did not fire at 250 K"
    _check_tangent(k_capped_of_T, T_cool, jnp.ones_like(T_cool), 1e-3)

    # Arrhenius coefficient overrides: passing the network's own `a` matches the default build, and the rate-uncertainty
    # tangent matches a difference.
    a0 = jnp.asarray(np.asarray(network.a, dtype=np.float64))
    k_ovr = np.asarray(rates_jax.compute_forward_k(network, T, M, a=a0))
    k_def = np.asarray(rates_jax.compute_forward_k(network, T, M))
    onz = np.abs(k_def) > 0
    assert np.abs(k_ovr[onz] - k_def[onz]).max() / np.abs(k_def[onz]).max() < RTOL

    def k_of_a(av):
        return rates_jax.build_rate_array(network, T, M, nasa9, remove_list=remove_list, rate_coeffs={"a": av})

    _check_tangent(k_of_a, a0, a0, 1e-6)


@pytest.mark.parametrize(("T", "Pr"), [(1000.0, 0.1), (1500.0, 0.1), (1000.0, 1.0), (1500.0, 10.0)])
def test_troe_oh_ch3_is_visscher_moses_eq14(T, Pr):
    """The OH+CH3+M row follows Visscher & Moses 2011 eqs 13-14, with the eq 24-26 fits.

    The Troe width is in log10; upstream op.py:204-205 uses ln. The only guard against a
    transcription error: the oracle comparison carries the same declared correction on its side.
    """
    from vulcan3 import rates_jax

    k0 = 1.932e3 * T**-9.88 * np.exp(-7544.0 / T) + 5.109e-11 * T**-6.25 * np.exp(-1433.0 / T)
    kinf = 1.031e-10 * T**-0.018 * np.exp(16.74 / T)
    Fc = 0.1855 * np.exp(-T / 155.8) + 0.8145 * np.exp(-T / 1675.0) + np.exp(-4531.0 / T)
    beta = 1.0 / (1.0 + (np.log10(Pr) / (0.75 - 1.27 * np.log10(Fc))) ** 2)
    want = k0 / (1.0 + Pr) * 10.0 ** (beta * np.log10(Fc))
    Tz, M = np.array([T]), np.array([Pr * kinf / k0])
    got = float(rates_jax._troe_OH_CH3(jnp.asarray(Tz), jnp.asarray(M))[0])
    assert abs(got - want) <= RTOL * want


def test_lindemann_with_zero_k_inf_is_the_zero_rate_limit():
    """`k_inf == 0` on a falloff row is the zero-rate limit: the row reads exactly 0, with a finite tangent.

    The guard value's low-pressure `arr / (1 + arr*M)` would not. A mis-shaped M raises instead of broadcasting the
    bottom layer's density over the column.
    """
    from vulcan3 import network as net_mod, rates_jax
    from vulcan3._paths import resolve_data_path
    from vulcan3.config import default_config

    net = net_mod.parse_network(str(resolve_data_path(default_config().network)))
    falloff = np.logical_and(
        np.asarray(net.has_kinf, dtype=bool),
        np.logical_not(np.asarray(net.is_special, dtype=bool)),
    )
    rows = np.flatnonzero(falloff)
    T = jnp.linspace(500.0, 2500.0, 6)
    M = jnp.full(6, 1e18)
    k_ref = rates_jax.compute_forward_k(net, T, M)
    assert bool(jnp.any(k_ref[rows] > 0.0)), "no live falloff row to test"

    a_inf = jnp.asarray(net.a_inf, dtype=jnp.float64).at[rows].set(0.0)
    k, dk = jax.jvp(
        lambda a: rates_jax.compute_forward_k(net, T, M, a_inf=a),
        (a_inf,),
        (jnp.ones_like(a_inf),),
    )
    assert bool(jnp.all(k[rows] == 0.0))
    assert bool(jnp.all(jnp.isfinite(k)))
    assert bool(jnp.all(jnp.isfinite(dk)))

    with pytest.raises(ValueError, match="same shape"):
        rates_jax.compute_forward_k(net, T, M[:1])
