"""Column structure and transport at setup: mean mass, hydrostatic scans, settling, moldiff, boundary fluxes.

The scans stay apart from `atm_refresh`: folding them moved bits (notes "Do not retry", Bit-moving refactors).

Units: CGS throughout: T in K, pressure in dyn cm^-2, number density in cm^-3, length in cm, time in s.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from ._paths import resolve_data_path
from ._tp_kzz import surface_gravity
from .atm_refresh import hydrostatic_step, recompute_vm_jax
from .phy_const import BAR_CGS, Navo, kb

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from .config import Config

__all__ = [
    "VISCOSITY_TABLE",
    "compute_mean_mass",
    "compute_mol_diff",
    "compute_mu_dz_g",
    "compute_settling_velocity",
    "mol_diff_jax",
    "mu_dz_g_jax",
    "read_bc_flux",
    "settling_coeff_array",
    "settling_velocity_jax",
]


logger = logging.getLogger(__name__)

# Pressure (dyne/cm^2, 1 bar) where `Rp` and `G Mp / Rp^2` apply on a gas giant, VULCAN 2's anchor
# (build_atm.py:533-536). Fixed, not a config key: jwst-transit-authority restates this rule
# (`chem_build.chem_radius_anchor_bar`), so moving it needs a change there first.
_RP_REF_PRESSURE = BAR_CGS


def compute_mean_mass(ymix: jnp.ndarray, ms_arr: jnp.ndarray) -> jnp.ndarray:
    """Mean molecular weight per layer.

    Parameters
    ----------
    ymix : jnp.ndarray [nz, ni]
        Mixing ratios.
    ms_arr : jnp.ndarray [ni]
        Molar mass per species, g/mol.

    Returns
    -------
    jnp.ndarray [nz]
        Mean molar mass, g/mol.
    """
    return jnp.einsum("zi,i->z", jnp.asarray(ymix), jnp.asarray(ms_arr))


def _scan_up_mu_dz_g(
    pref_indx: int,
    gs: float | jnp.ndarray,
    Rp: float | jnp.ndarray,
    Tco: jnp.ndarray,
    mu: jnp.ndarray,
    pico: jnp.ndarray,
    nz: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Sequential upward height integration from `pref_indx` to the top."""
    Tco_up = Tco[pref_indx:nz]
    mu_up = mu[pref_indx:nz]
    pico_up_lo = pico[pref_indx:nz]
    pico_up_hi = pico[pref_indx + 1 : nz + 1]

    def body(
        carry: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        scan_in: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        z_prev, _gz_prev, _Hp_prev = carry
        i, T_i, mu_i, p_lo, p_hi = scan_in
        is_first = i == 0
        gz_i = jnp.where(is_first, gs, gs * (Rp / (Rp + z_prev)) ** 2)
        Hp_i, dz_i = hydrostatic_step(T_i, mu_i, gz_i, p_lo, p_hi, kb, Navo)
        z_next = z_prev + dz_i
        return (z_next, gz_i, Hp_i), (gz_i, Hp_i, dz_i, z_next)

    # jnp.asarray (not jnp.float64) keeps the carry tracer-safe for a traced
    # `gs`.
    init = (jnp.float64(0.0), jnp.asarray(gs, dtype=jnp.float64), jnp.float64(0.0))
    n_up = nz - pref_indx
    idx = jnp.arange(n_up)
    _, (gz_seq, Hp_seq, dz_seq, z_after_seq) = jax.lax.scan(body, init, (idx, Tco_up, mu_up, pico_up_lo, pico_up_hi))
    return gz_seq, Hp_seq, dz_seq, z_after_seq


def _scan_down_mu_dz_g(
    pref_indx: int,
    gs: float | jnp.ndarray,
    Rp: float | jnp.ndarray,
    Tco: jnp.ndarray,
    mu: jnp.ndarray,
    pico: jnp.ndarray,
    z_at_pref: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Sequential downward height integration from `pref_indx-1` to 0.

    Requires `pref_indx >= 1`. Returns arrays in low-to-high index order.
    """
    Tco_dn_rev = Tco[:pref_indx][::-1]
    mu_dn_rev = mu[:pref_indx][::-1]
    pico_lo_rev = pico[:pref_indx][::-1]
    pico_hi_rev = pico[1 : pref_indx + 1][::-1]

    def body(
        carry: jnp.ndarray, scan_in: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
    ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        z_above = carry
        T_i, mu_i, p_lo, p_hi = scan_in
        gz_i = gs * (Rp / (Rp + z_above)) ** 2
        Hp_i, dz_i = hydrostatic_step(T_i, mu_i, gz_i, p_lo, p_hi, kb, Navo)
        z_here = z_above - dz_i
        return z_here, (gz_i, Hp_i, dz_i, z_here)

    init = jnp.float64(z_at_pref)
    _, (gz_rev, Hp_rev, dz_rev, z_rev) = jax.lax.scan(body, init, (Tco_dn_rev, mu_dn_rev, pico_lo_rev, pico_hi_rev))
    return gz_rev[::-1], Hp_rev[::-1], dz_rev[::-1], z_rev[::-1]


def mu_dz_g_jax(
    pref_indx: int,
    gs: float | jnp.ndarray,
    Rp: float | jnp.ndarray,
    Tco: jnp.ndarray,
    mu: jnp.ndarray,
    pico: jnp.ndarray,
    nz: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Hydrostatic height integration on the JAX graph, anchored at `pref_indx`.

    The host setup path (:func:`compute_mu_dz_g`) runs it on the graph too, so
    a traced caller gets the same float64 bits. `pref_indx` is a discrete
    host-side choice and is passed in.

    Parameters
    ----------
    pref_indx : int
        Reference index: ``zco[pref_indx] = 0`` and ``g[pref_indx] = gs``.
    gs : float or jnp.ndarray
        Gravity at the reference level, cm/s^2.
    Rp : float or jnp.ndarray
        Planet radius, cm.
    Tco : jnp.ndarray [nz]
        Layer temperature, K.
    mu : jnp.ndarray [nz]
        Mean molar mass, g/mol.
    pico : jnp.ndarray [nz + 1]
        Interface pressure, dyne/cm^2, bottom first.
    nz : int
        Number of layers.

    Returns
    -------
    g : jnp.ndarray [nz]
        Gravity, cm/s^2.
    Hp : jnp.ndarray [nz]
        Pressure scale height, cm.
    dz : jnp.ndarray [nz]
        Layer thickness, cm.
    zco : jnp.ndarray [nz + 1]
        Interface height above `pref_indx`, cm.
    zmco : jnp.ndarray [nz]
        Layer-centre height, cm.
    dzi : jnp.ndarray [nz - 1]
        Spacing of layer centres across each interior interface, cm.
    Ti : jnp.ndarray [nz - 1]
        Interior-interface temperature, K.
    Hpi : jnp.ndarray [nz - 1]
        Interior-interface scale height, cm.
    """
    gz_up, Hp_up, dz_up, z_after_up = _scan_up_mu_dz_g(pref_indx, gs, Rp, Tco, mu, pico, nz)
    gz = jnp.zeros(nz, dtype=jnp.float64).at[pref_indx:nz].set(gz_up)
    Hp = jnp.zeros(nz, dtype=jnp.float64).at[pref_indx:nz].set(Hp_up)
    dz = jnp.zeros(nz, dtype=jnp.float64).at[pref_indx:nz].set(dz_up)
    # zco[pref_indx] = 0 by definition.
    zco = jnp.zeros(nz + 1, dtype=jnp.float64).at[pref_indx + 1 : nz + 1].set(z_after_up)

    if pref_indx > 0:
        gz_dn, Hp_dn, dz_dn, z_dn = _scan_down_mu_dz_g(pref_indx, gs, Rp, Tco, mu, pico, z_at_pref=0.0)
        gz = gz.at[:pref_indx].set(gz_dn)
        Hp = Hp.at[:pref_indx].set(Hp_dn)
        dz = dz.at[:pref_indx].set(dz_dn)
        zco = zco.at[:pref_indx].set(z_dn)

    zmco = 0.5 * (zco[:-1] + jnp.roll(zco, -1)[:-1])
    dzi = 0.5 * (dz[1:] + jnp.roll(dz, 1)[1:])
    Ti = 0.5 * (Tco[:-1] + jnp.roll(Tco, -1)[:-1])
    Hpi = 0.5 * (Hp[:-1] + jnp.roll(Hp, -1)[:-1])
    return gz, Hp, dz, zco, zmco, dzi, Ti, Hpi


def compute_mu_dz_g(
    cfg: Config,
    ymix: np.ndarray,
    ms_arr: np.ndarray,
    pico: np.ndarray,
    Tco: np.ndarray,
) -> dict[str, np.ndarray | int | float]:
    """Build the hydrostatic column structure.

    Host wrapper over :func:`mu_dz_g_jax`: it picks the discrete `pref_indx`
    and freezes the result to NumPy.

    Parameters
    ----------
    cfg : Config
        The run's config: `Mp`, `Rp`, `rocky`, `P_b`, `use_moldiff`.
    ymix : np.ndarray [nz, ni]
        Mixing ratios.
    ms_arr : np.ndarray [ni]
        Molar mass per species, g/mol.
    pico : np.ndarray [nz + 1]
        Interface pressure, dyne/cm^2, bottom first.
    Tco : np.ndarray [nz]
        Layer temperature, K.

    Returns
    -------
    dict
        `mu` (g/mol), `g`, `Hp`, `dz`, `dzi`, `zco`, `zmco` as :func:`mu_dz_g_jax` gives them, `gs` (float, cm/s^2),
        `pref_indx` (int), and `Ti`, `Hpi` only when `use_moldiff` is on.
    """
    nz = int(Tco.shape[0])
    pico_j = jnp.asarray(pico, dtype=jnp.float64)
    Tco_j = jnp.asarray(Tco, dtype=jnp.float64)
    mu_j = compute_mean_mass(jnp.asarray(ymix, dtype=jnp.float64), jnp.asarray(ms_arr, dtype=jnp.float64))

    gs = surface_gravity(cfg)

    # `pref_indx` anchors g(z)=gs: gas giants with P_b >= _RP_REF_PRESSURE anchor at the
    # interface nearest it; rocky planets at index 0 (build_atm.py:533-536).
    pico_host = np.asarray(pico_j)
    if (not bool(cfg.rocky)) and float(cfg.P_b) >= _RP_REF_PRESSURE:
        pref_indx = int(min(range(nz + 1), key=lambda i: abs(np.log10(pico_host[i]) - np.log10(_RP_REF_PRESSURE))))
    else:
        pref_indx = 0

    gz, Hp, dz, zco, zmco, dzi, Ti, Hpi = mu_dz_g_jax(pref_indx, gs, float(cfg.Rp), Tco_j, mu_j, pico_j, nz)

    out: dict[str, np.ndarray | int | float] = {
        "mu": np.asarray(mu_j),
        "g": np.asarray(gz),
        "gs": gs,
        "Hp": np.asarray(Hp),
        "dz": np.asarray(dz),
        "dzi": np.asarray(dzi),
        "zco": np.asarray(zco),
        "zmco": np.asarray(zmco),
        "pref_indx": pref_indx,
    }
    if bool(cfg.use_moldiff):
        out["Ti"] = np.asarray(Ti)
        out["Hpi"] = np.asarray(Hpi)
    return out


# Cloutman dynamic-viscosity polynomial (na, a, b) per atm_base
# (build_atm.py:589-599). CO2 is not tabulated; falls back to N2.
VISCOSITY_TABLE: Mapping[str, tuple[float, float, float]] = {
    "N2": (1.52, 1.186e-5, 86.54),
    "H2": (1.67, 1.936e-6, 2.187),
    "CO2": (1.52, 1.186e-5, 86.54),
    "H2O": (1.5, 1.6e-5, 0.0),
    "O2": (1.46, 2.294e-5, 164.4),
}


def settling_velocity_jax(
    na: float,
    a: float,
    b: float,
    Tco: jnp.ndarray,
    g: jnp.ndarray,
    settle_coeff: jnp.ndarray,
) -> jnp.ndarray:
    """JAX core of the Stokes-regime settling velocity.

    Single source for :func:`compute_settling_velocity`, in ``jnp`` so the fall speed is differentiable w.r.t.
    temperature, gravity and particle size/density.

    Parameters
    ----------
    na : float
        Exponent of the background gas's Cloutman dynamic viscosity ``a * T**na / (b + T)``, g/(cm s).
    a : float
        Its coefficient, g cm^-1 s^-1 K^(1-na).
    b : float
        Its temperature offset, K.
    Tco : jnp.ndarray [nz]
        Layer temperature, K.
    g : jnp.ndarray [nz]
        Gravity, cm/s^2.
    settle_coeff : jnp.ndarray [ni]
        Per-species ``rho_p * r_p**2``, g/cm, zero for gas species.

    Returns
    -------
    jnp.ndarray [nz - 1, ni]
        Settling velocity at the interior interfaces, cm/s, negative (downward).
    """
    Tco = jnp.asarray(Tco, dtype=jnp.float64)
    g = jnp.asarray(g, dtype=jnp.float64)
    dmu = a * Tco**na / (b + Tco)  # dynamic viscosity, (nz,) g/(cm s)
    gi = 0.5 * (g[:-1] + g[1:])  # (nz-1,)
    factor = -(2.0 / 9.0) * gi / dmu[1:]  # (nz-1,)
    return factor[:, None] * jnp.asarray(settle_coeff, dtype=jnp.float64)[None, :]


def compute_settling_velocity(
    cfg: Config,
    Tco: np.ndarray,
    g: np.ndarray,
    species_list: list[str],
    rho_p: Mapping[str, float],
    r_p: Mapping[str, float],
) -> np.ndarray:
    """Per-species terminal-fall velocity (Stokes regime).

    Parameters
    ----------
    cfg : Config
        The run's config: `use_settling`, `atm_base`, `non_gas_sp`.
    Tco : np.ndarray [nz]
        Layer temperature, K.
    g : np.ndarray [nz]
        Gravity, cm/s^2.
    species_list : list of str [ni]
        Network species.
    rho_p : mapping of str to float
        Particle density per condensate, g/cm^3.
    r_p : mapping of str to float
        Particle radius per condensate, cm.

    Returns
    -------
    np.ndarray [nz - 1, ni]
        Settling velocity at the interior interfaces, cm/s, negative (downward); non-zero only for `non_gas_sp`, and
        all zero when `use_settling` is off.

    Raises
    ------
    OSError
        If `atm_base` has no viscosity polynomial.
    """
    nz = int(Tco.shape[0])
    ni = len(species_list)
    if not bool(cfg.use_settling):
        return np.zeros((nz - 1, ni), dtype=np.float64)
    atm_base = cfg.atm_base
    if atm_base == "CO2":
        warnings.warn("NO CO2 viscosity yet! (using N2 instead)", stacklevel=2)
    if atm_base not in VISCOSITY_TABLE:
        raise OSError(f"No viscosity polynomial for atm_base={atm_base!r}")
    na, a, b = VISCOSITY_TABLE[atm_base]
    settle_coeff = settling_coeff_array(cfg, species_list, rho_p, r_p)
    vs = settling_velocity_jax(na, a, b, jnp.asarray(Tco), jnp.asarray(g), jnp.asarray(settle_coeff))
    return np.asarray(vs, dtype=np.float64)


def settling_coeff_array(
    cfg: Config,
    species_list: list[str],
    rho_p: Mapping[str, float],
    r_p: Mapping[str, float],
) -> np.ndarray:
    """Per-species ``rho_p * r_p**2``, zero for gas species.

    The static, host-side half of the settling velocity: it validates that
    every condensible in ``non_gas_sp`` has a prescribed size and density.
    Split out so the differentiable builder can reuse it without re-running
    the validation each step.

    Parameters
    ----------
    cfg : Config
        The run's config: `non_gas_sp`.
    species_list : list of str [ni]
        Network species.
    rho_p : mapping of str to float
        Particle density per condensate, g/cm^3.
    r_p : mapping of str to float
        Particle radius per condensate, cm.

    Returns
    -------
    np.ndarray [ni]
        ``rho_p * r_p**2``, g/cm.

    Raises
    ------
    OSError
        If a network species in `non_gas_sp` lacks a size or density.
    """
    ni = len(species_list)
    settle_coeff = np.zeros(ni, dtype=np.float64)
    for sp in cfg.non_gas_sp:
        if sp not in species_list:
            continue
        if sp not in rho_p or sp not in r_p:
            raise OSError(f"{sp} has not been prescribed size and density!")
        settle_coeff[species_list.index(sp)] = float(rho_p[sp]) * float(r_p[sp]) ** 2
    return settle_coeff


def _Dzz_gen_for_base(atm_base: str) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Return a JAX lambda `(T, n_tot, mi) -> Dzz` for an atm_base.

    Published pair fits, fixed with the data: CH4-H2 from Moses 2000a (H2 base); CH4-N2 and CH4-O2 from Banks &
    Kockarts 1973 and H2-CO2 from Hu & Seager, each scaled by molecular mass (VULCAN 2 build_atm.py:684-714 and its
    comments there).
    """
    if atm_base == "H2":
        return lambda T, n_tot, mi: 2.2965e17 * T**0.765 / n_tot * (16.04 / mi * (mi + 2.016) / 18.059) ** 0.5
    if atm_base == "N2":
        return lambda T, n_tot, mi: 7.34e16 * T**0.75 / n_tot * (16.04 / mi * (mi + 28.014) / 44.054) ** 0.5
    if atm_base == "O2":
        return lambda T, n_tot, mi: 7.51e16 * T**0.759 / n_tot * (16.04 / mi * (mi + 32.0) / 48.04) ** 0.5
    if atm_base == "CO2":
        return lambda T, n_tot, mi: 2.15e17 * T**0.750 / n_tot * (2.016 / mi * (mi + 44.001) / 46.017) ** 0.5
    raise OSError(f"\n Unknown atm_base={atm_base!r}!")


def _alpha_array_for_base(atm_base: str, species_list: list[str], mol_mass: Callable[[str], float]) -> np.ndarray:
    """Thermal-diffusion factor per species, atm_base-dependent. Returns (ni,).

    Defaults to zero. H2 base: H = -0.1, He = 0.145, then every species over
    4 amu (He included) gets 0.25, as master (build_atm.py:690-693). N2/O2/CO2
    base: H/H2/He = -0.25, Ar = 0.17. Other atm_base raises. Fixed with the data:
    upstream calls the H2-base H value "simplified from Moses 2000a" (VULCAN 2
    build_atm.py:690) and gives no source for the others (691-693, and the
    N2/O2/CO2 tables at 699-702, 708-711 and 717-720). The overwritten He value
    is in notes "Known bugs, all repos".
    """
    ni = len(species_list)
    alpha = np.zeros(ni, dtype=np.float64)
    if atm_base == "H2":
        if "H" in species_list:
            alpha[species_list.index("H")] = -0.1
        if "He" in species_list:
            alpha[species_list.index("He")] = 0.145
        for sp in species_list:
            if mol_mass(sp) > 4.0:
                alpha[species_list.index(sp)] = 0.25
    elif atm_base in ("N2", "O2", "CO2"):
        for sp in ("H", "H2", "He"):
            if sp in species_list:
                alpha[species_list.index(sp)] = -0.25
        if "Ar" in species_list:
            alpha[species_list.index("Ar")] = 0.17
    else:
        raise OSError(f"\n Unknown atm_base={atm_base!r}!")
    return alpha


def mol_diff_jax(
    atm_base: str,
    Tco: jnp.ndarray,
    n_0: jnp.ndarray,
    g: jnp.ndarray,
    Hp: jnp.ndarray,
    dz: jnp.ndarray,
    ms: jnp.ndarray,
    alpha: jnp.ndarray,
    nongas_mask: jnp.ndarray,
    *,
    use_vm_mol: bool,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Molecular diffusion on the JAX graph: (Dzz, Dzz_cen, vm).

    Shared by the host setup path (:func:`compute_mol_diff`) and the
    on-graph `atm_jax.mol_diff`. Callers gate `use_moldiff` themselves.

    Parameters
    ----------
    atm_base : str
        Background gas: ``H2``, ``N2``, ``O2`` or ``CO2``.
    Tco : jnp.ndarray [nz]
        Layer temperature, K.
    n_0 : jnp.ndarray [nz]
        Total number density, cm^-3.
    g : jnp.ndarray [nz]
        Gravity, cm/s^2.
    Hp : jnp.ndarray [nz]
        Pressure scale height, cm.
    dz : jnp.ndarray [nz]
        Layer thickness, cm.
    ms : jnp.ndarray [ni]
        Molar mass per species, g/mol.
    alpha : jnp.ndarray [ni]
        Thermal-diffusion factor per species.
    nongas_mask : jnp.ndarray [ni]
        bool, True for condensates; their interface `Dzz` (and so `vm`) is zero.
    use_vm_mol : bool
        Compute the drift `vm`; zero otherwise.

    Returns
    -------
    Dzz : jnp.ndarray [nz - 1, ni]
        Molecular diffusion coefficient at the interior interfaces, cm^2/s.
    Dzz_cen : jnp.ndarray [nz, ni]
        The same at layer centres, cm^2/s, kept for the `.vul` diagnostic surface.
    vm : jnp.ndarray [nz - 1, ni]
        Interface drift (advective) velocity of molecular diffusion, cm/s.
    """
    Dzz_gen = _Dzz_gen_for_base(atm_base)

    # Interface values for the (nz-1, ni) Dzz array.
    Tco_i = (Tco[1:] + Tco[:-1]) * 0.5
    n0_i = (n_0[1:] + n_0[:-1]) * 0.5
    Dzz = jax.vmap(lambda mi: Dzz_gen(Tco_i, n0_i, mi), out_axes=1)(ms)
    Dzz_cen = jax.vmap(lambda mi: Dzz_gen(Tco, n_0, mi), out_axes=1)(ms)

    # Non-gaseous species (e.g. H2O_l_s) get zero diffusion on the interfaces;
    # because vm below reuses this interface Dzz, the zeroing carries straight
    # into vm with no separate condensation gate.
    Dzz = jnp.where(nongas_mask[None, :], 0.0, Dzz)

    if not use_vm_mol:
        return Dzz, Dzz_cen, jnp.zeros_like(Dzz)

    # Interface-centered upwind drift velocity (vm_branch op.update_mu_dz):
    # vm = -Dzz * (1/H_i - 1/Hp + thermal) on cell interfaces, using the same
    # arithmetic means mu_dz_g_jax forms for Ti/Hpi/dzi. Shared with the
    # in-loop refresh (atm_refresh.recompute_vm_jax) so both stay bit-identical.
    Hpi = 0.5 * (Hp[:-1] + Hp[1:])  # (nz-1,)
    dzi = 0.5 * (dz[1:] + dz[:-1])  # (nz-1,)
    vm = recompute_vm_jax(g, Hpi, dzi, Dzz, ms, alpha, Tco, kb, Navo)
    return Dzz, Dzz_cen, vm


def compute_mol_diff(
    cfg: Config,
    Tco: np.ndarray,
    n_0: np.ndarray,
    g: np.ndarray,
    Hp: np.ndarray,
    dz: np.ndarray,
    ms_arr: np.ndarray,
    alpha_arr: np.ndarray,
    species_list: list[str],
) -> dict[str, np.ndarray]:
    """Build (Dzz, Dzz_cen, vm) for the cfg's `atm_base`.

    Host wrapper over :func:`mol_diff_jax`; all three are zeros when
    `use_moldiff` is off.

    Parameters
    ----------
    cfg : Config
        The run's config: `use_moldiff`, `atm_base`, `non_gas_sp`, `use_vm_mol`.
    Tco : np.ndarray [nz]
        Layer temperature, K.
    n_0 : np.ndarray [nz]
        Total number density, cm^-3.
    g : np.ndarray [nz]
        Gravity, cm/s^2.
    Hp : np.ndarray [nz]
        Pressure scale height, cm.
    dz : np.ndarray [nz]
        Layer thickness, cm.
    ms_arr : np.ndarray [ni]
        Molar mass per species, g/mol.
    alpha_arr : np.ndarray [ni]
        Thermal-diffusion factor per species.
    species_list : list of str [ni]
        Network species.

    Returns
    -------
    dict of str to np.ndarray
        `Dzz` [nz - 1, ni] and `Dzz_cen` [nz, ni] (cm^2/s), `vm` [nz - 1, ni] (cm/s).
    """
    nz = int(Tco.shape[0])
    ni = len(species_list)
    if not bool(cfg.use_moldiff):
        return {
            "Dzz": np.zeros((nz - 1, ni), dtype=np.float64),
            "Dzz_cen": np.zeros((nz, ni), dtype=np.float64),
            "vm": np.zeros((nz - 1, ni), dtype=np.float64),
        }

    nongas = np.array([sp in set(cfg.non_gas_sp) for sp in species_list], dtype=bool)
    Dzz, Dzz_cen, vm = mol_diff_jax(
        cfg.atm_base,
        jnp.asarray(Tco, dtype=jnp.float64),
        jnp.asarray(n_0, dtype=jnp.float64),
        jnp.asarray(g, dtype=jnp.float64),
        jnp.asarray(Hp, dtype=jnp.float64),
        jnp.asarray(dz, dtype=jnp.float64),
        jnp.asarray(ms_arr, dtype=jnp.float64),
        jnp.asarray(alpha_arr, dtype=jnp.float64),
        jnp.asarray(nongas),
        use_vm_mol=bool(cfg.use_vm_mol),
    )
    return {
        "Dzz": np.asarray(Dzz),
        "Dzz_cen": np.asarray(Dzz_cen),
        "vm": np.asarray(vm),
    }


def _parse_bc_file(path: str) -> list[list[str]]:
    """Read a BC text file into token rows; skip comments and blank lines."""
    rows: list[list[str]] = []
    with Path(resolve_data_path(path)).open() as f:
        for line in f:
            if not line.startswith("#") and line.strip():
                rows.append(line.split())
    return rows


def _bc_species_index(sp: str, species_list: list[str], path: str) -> int:
    """Return a boundary-flux row's species index, refusing a species outside the network, as upstream does."""
    if sp not in species_list:
        raise ValueError(f"{path}: boundary-flux row names {sp!r}, which is not in the network; remove the row.")
    return species_list.index(sp)


def read_bc_flux(cfg: Config, species_list: list[str]) -> dict[str, np.ndarray]:
    """Build (top_flux, bot_flux, bot_vdep, bot_fix_sp).

    use_topflux and use_botflux each fill their section from a file. bot_fix_sp
    stays zero: OuterLoop pins the `use_fix_sp_bot` dict entries itself.

    Parameters
    ----------
    cfg : Config
        The run's config: `use_topflux`, `use_botflux` and their files.
    species_list : list of str [ni]
        Network species.

    Returns
    -------
    dict of str to np.ndarray [ni]
        `top_flux` and `bot_flux` (cm^-2 s^-1), `bot_vdep` (deposition velocity, cm/s) and `bot_fix_sp` (zero).
    """
    ni = len(species_list)
    out = {
        "top_flux": np.zeros(ni, dtype=np.float64),
        "bot_flux": np.zeros(ni, dtype=np.float64),
        "bot_vdep": np.zeros(ni, dtype=np.float64),
        "bot_fix_sp": np.zeros(ni, dtype=np.float64),
    }
    if bool(cfg.use_topflux):
        logger.info("Using the prescribed constant top flux.")
        for tokens in _parse_bc_file(cfg.top_BC_flux_file):
            out["top_flux"][_bc_species_index(tokens[0], species_list, cfg.top_BC_flux_file)] = float(tokens[1])
    if bool(cfg.use_botflux):
        logger.info("Using the prescribed constant bottom flux.")
        for tokens in _parse_bc_file(cfg.bot_BC_flux_file):
            i = _bc_species_index(tokens[0], species_list, cfg.bot_BC_flux_file)
            out["bot_flux"][i] = float(tokens[1])
            out["bot_vdep"][i] = float(tokens[2])
    return out
