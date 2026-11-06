"""Saturation vapour pressures of the condensable species.

Units: CGS throughout: T in K, pressure in dyn cm^-2, number density in cm^-3, length in cm, time in s.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from .phy_const import ATM_BAR, ATM_CGS, BAR_CGS

__all__ = ["compute_sat_p", "sat_p_jax"]

# Phase breaks in the saturation fits (build_atm.py:829, :840, :856).
_S_ALLOTROPE_BREAK_K = 413.0  # S2 and S8

_H2S_ICE_LIQUID_K = 187.6

# bar per cmHg. Upstream's 0.001333 (build_atm.py:857) is mmHg.
_CMHG_TO_BAR = 0.01333

# Valid temperature range of the NASA-9 Gibbs fits (build_atm.py:618).
_GIBBS_T_MIN_K = 200.0

_GIBBS_T_MAX_K = 6000.0

_SUPPORTED_CONDENSABLES: tuple[str, ...] = (
    "H2O",
    "NH3",
    "H2SO4",
    "S2",
    "S4",
    "S8",
    "C",
    "H2S",
)


def sat_p_jax(sp: str, T: jnp.ndarray) -> jnp.ndarray:
    """JAX core of one condensable's saturation vapour pressure (dyne/cm^2).

    Single source of truth for :func:`compute_sat_p`, expressed in ``jnp`` so
    the saturation curve is differentiable w.r.t. temperature. The only
    non-smooth points are the phase-boundary jumps (ice/liquid for H2O, the
    413 K break for S2/S8, the 187.6 K break for H2S). Fits and coefficients
    from build_atm.py:804-857.
    """
    T = jnp.asarray(T, dtype=jnp.float64)
    if sp == "H2O":
        T_C = T - 273.0
        c0, c1, c2, c3 = 6111.5, 23.036, -333.7, 279.82  # ice constants
        w0, w1, w2, w3 = 6112.1, 18.729, -227.3, 257.87  # liquid constants
        # Ackerman & Marley (2001): ice for T < 0 C, liquid water for T >= 0 C.
        # Upstream's (T<0)*ice + (T>0)*water is 0 at 273 K; one `where` has
        # no zero there (the two fits still differ by 0.6 dyne/cm^2 at 0 C).
        ice = c0 * jnp.exp((c1 * T_C + T_C**2 / c2) / (T_C + c3))
        liquid = w0 * jnp.exp((w1 * T_C + T_C**2 / w2) / (T_C + w3))
        return jnp.where(T_C < 0, ice, liquid)
    if sp == "NH3":
        c0, c1, c2 = 10.53, -2161.0, -86596.0
        return jnp.exp(c0 + c1 / T + c2 / T**2) * BAR_CGS
    if sp == "H2SO4":
        return jnp.exp(-10156.0 / T + 16.259) * ATM_BAR * BAR_CGS
    if sp == "S2":
        return jnp.where(
            T < _S_ALLOTROPE_BREAK_K,
            jnp.exp(27.0 - 18500.0 / T) * BAR_CGS,
            jnp.exp(16.1 - 14000.0 / T) * BAR_CGS,
        )
    if sp == "S4":
        return 10 ** (6.0028 - 6047.5 / T) * ATM_CGS
    if sp == "S8":
        return jnp.where(
            T < _S_ALLOTROPE_BREAK_K,
            jnp.exp(20.0 - 11800.0 / T) * BAR_CGS,
            jnp.exp(9.6 - 7510.0 / T) * BAR_CGS,
        )
    if sp == "C":
        a, b, c = 3.27860e1, -8.65139e4, 4.80395e-1
        return jnp.exp(a + b / (T + c))
    if sp == "H2S":
        # Giauque & Blue (1936) Antoine fits; output is in cmHg.
        ice_log10 = -1329.0 / T + 9.28588 - 0.0051263 * T
        l_log10 = -1145.0 / T + 7.94746 - 0.00322 * T
        sat_p = 10 ** jnp.where(T <= _H2S_ICE_LIQUID_K, ice_log10, l_log10)
        # Check: the H2S boiling point, 212.8 K -> 76.1 cmHg = 1.015 bar.
        return sat_p * _CMHG_TO_BAR * BAR_CGS
    raise OSError(f"No saturation vapor data for {sp}. Check `sat_p_jax` in atm_setup.py")


def compute_sat_p(condense_sp: list[str], Tco: np.ndarray) -> dict[str, np.ndarray]:
    """Saturation vapour pressure (dyne/cm^2) per condensable species."""
    Tco_j = jnp.asarray(Tco, dtype=jnp.float64)
    out: dict[str, np.ndarray] = {}
    for sp in condense_sp:
        out[sp] = np.asarray(sat_p_jax(sp, Tco_j), dtype=np.float64)
    return out
