"""JAX-native atmosphere setup.

Pure-function kernels (cfg + arrays in, arrays out) plus a thin `Atm`
facade that mutates a legacy `data_atm` container for callers that still
read attributes directly (rate setup, initial abundances, photolysis setup, the RunState snapshot).
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp_special
import numpy as np

from ._paths import resolve_data_path
from .atm_refresh import hydrostatic_step, recompute_vm_jax
from .config import default_config
from .phy_const import ATM_BAR, ATM_CGS, BAR_CGS, G_grav, Navo, au, kb, r_sun

if TYPE_CHECKING:
    from collections.abc import Mapping

from ._saturation import (
    _CMHG_TO_BAR,
    _GIBBS_T_MAX_K,
    _GIBBS_T_MIN_K,
    _H2S_ICE_LIQUID_K,
    _S_ALLOTROPE_BREAK_K,
    _SUPPORTED_CONDENSABLES,
    compute_sat_p,
    sat_p_jax,
)
from ._stellar_flux import (
    _parse_bc_file,
    read_bc_flux,
    read_sflux_binned,
)
from ._tp_kzz import (
    _interp_descending_or_ascending,
    _read_atm_table,
    analytical_TP_H14,
    compute_pico,
    high_temp_cut_regrid,
    kzz_profile_jax,
    load_TPK,
    surface_gravity,
)
from ._transport_setup import (
    _VISCOSITY_TABLE,
    VISCOSITY_TABLE,
    _alpha_array_for_base,
    _Dzz_gen_for_base,
    _scan_down_mu_dz_g,
    _scan_up_mu_dz_g,
    compute_mean_mass,
    compute_mol_diff,
    compute_mu_dz_g,
    compute_settling_velocity,
    mol_diff_jax,
    mu_dz_g_jax,
    settling_coeff_array,
    settling_velocity_jax,
)

__all__ = [
    "VISCOSITY_TABLE",
    "Atm",
    "analytical_TP_H14",
    "compute_mean_mass",
    "compute_mol_diff",
    "compute_mu_dz_g",
    "compute_pico",
    "compute_sat_p",
    "compute_settling_velocity",
    "high_temp_cut_regrid",
    "kzz_profile_jax",
    "load_TPK",
    "mol_diff_jax",
    "mu_dz_g_jax",
    "read_bc_flux",
    "read_sflux_binned",
    "sat_p_jax",
    "settling_coeff_array",
    "settling_velocity_jax",
    "surface_gravity",
]

logger = logging.getLogger(__name__)

_CFG = default_config()


# Load TPK (T, Kzz, vz from cfg / file) -- host-side I/O


class Atm:
    """Thin mutable facade over the pure functions above.

    Routes `make_atm.X(data_atm)` calls into them and writes results back to the legacy `data_atm` / `data_var`
    containers.
    """

    def f_pico(self, data_atm: Any) -> Any:
        """Stagger pressure to interfaces and write to `data_atm.pico`."""
        data_atm.pico = np.asarray(compute_pico(jnp.asarray(data_atm.pco)))
        return data_atm

    def load_TPK(self, data_atm: Any) -> Any:
        """Populate `Tco / Kzz / vz / M / n_0` per `atm_type` (isothermal, analytical, file, vulcan_ini, or table)."""
        out = load_TPK(_CFG, np.asarray(data_atm.pco), pico=np.asarray(data_atm.pico))
        # `atm_type='table'` rewrites pco from a saved file.
        if "pco" in out:
            data_atm.pco = out["pco"]
        for k in ("Tco", "Kzz", "vz", "M", "n_0"):
            setattr(data_atm, k, out[k])
        return data_atm

    def apply_high_temp_cut(self, data_atm: Any) -> Any:
        """Lower the bottom pressure so deep T does not exceed ``high_temp_cut_K``.

        Matches vm_branch ``build_atm.apply_high_temp_cut``.

        Delegates grid selection to :func:`high_temp_cut_regrid`, then reloads
        T/Kzz on the re-gridded column. No-op when the deep column is already
        below ``T_max``.
        """
        T_max = float(_CFG.high_temp_cut_K)
        P_min = float(_CFG.high_temp_cut_P)
        nz = int(_CFG.nz)

        new_pco = high_temp_cut_regrid(data_atm.pco, data_atm.Tco, T_max=T_max, P_min=P_min, P_t=_CFG.P_t, nz=nz)
        if new_pco is None:
            return data_atm

        logger.warning(
            "high_temp_cut: capping deep T at %.0f K (P >= %.2e bar) for numerical stability.", T_max, P_min / BAR_CGS
        )
        logger.warning(
            "  effective P_b %.2e -> %.2e bar (nz = %s)", float(_CFG.P_b) / BAR_CGS, float(new_pco[0]) / BAR_CGS, nz
        )

        data_atm.pco = new_pco
        data_atm = self.f_pico(data_atm)
        data_atm = self.load_TPK(data_atm)

        if np.any(np.asarray(data_atm.Tco) > T_max):
            warnings.warn(
                f"After high_temp_cut: max Tco = {float(np.max(np.asarray(data_atm.Tco))):.1f} K still > "
                f"{T_max:.0f} K.",
                stacklevel=2,
            )
        return data_atm

    def mol_mass(self, sp: str) -> Any:
        """Molecular mass (amu) for species `sp` from the composition table."""
        from .composition import compo, compo_row

        return compo["mass"][compo_row.index(sp)]

    def mean_mass(self, var: Any, atm: Any, ni: int) -> Any:
        """Compute per-layer mean molecular mass (amu, (nz,)) and write `atm.mu`.

        No in-repo call site, but a LIVE cross-repo contract: master's
        `op.Integration.update_mu_dz` (op.py:951) duck-types this method on
        our `Atm` (tests/test_outer_loop_atm_refresh.py). Do not delete.
        """
        from .composition import species_mass

        atm.mu = np.asarray(
            compute_mean_mass(
                jnp.asarray(var.ymix),
                jnp.asarray(species_mass[:ni]),
            )
        )
        return atm

    def f_mu_dz(self, data_var: Any, data_atm: Any, output: Any) -> Any:
        """Hydrostatic-balance refresh: rebuild `mu, g, Hp, dz, zco, vs` and friends."""
        del output  # build_atm.py:527 signature; master uses it only for plot_TP (:616)
        from .composition import species, species_mass

        out = compute_mu_dz_g(
            _CFG,
            np.asarray(data_var.ymix),
            species_mass,
            np.asarray(data_atm.pico),
            np.asarray(data_atm.Tco),
        )
        data_atm.mu = out["mu"]
        data_atm.g = out["g"]
        data_atm.gs = out["gs"]
        data_atm.Hp = out["Hp"]
        data_atm.dz = out["dz"]
        data_atm.dzi = out["dzi"]
        data_atm.zco = out["zco"]
        data_atm.zmco = out["zmco"]
        data_atm.pref_indx = out["pref_indx"]
        if "Ti" in out:
            data_atm.Ti = out["Ti"]
            data_atm.Hpi = out["Hpi"]
        if _CFG.use_settling:
            data_atm.vs = compute_settling_velocity(
                _CFG,
                data_atm.Tco,
                data_atm.g,
                list(species),
                rho_p=getattr(data_atm, "rho_p", {}),
                r_p=getattr(data_atm, "r_p", {}),
            )
        if np.any(np.logical_or(data_atm.Tco < _GIBBS_T_MIN_K, data_atm.Tco > _GIBBS_T_MAX_K)):
            warnings.warn("Temperatures exceed the valid range of Gibbs free energy.", stacklevel=2)
        return data_atm

    def mol_diff(self, atm: Any) -> None:
        """Compute molecular-diffusion coefficients and write `atm.Dzz / Dzz_cen / vm`."""
        from .composition import species, species_mass

        ms_arr = species_mass.copy()
        atm.ms = ms_arr
        alpha_arr = _alpha_array_for_base(
            _CFG.atm_base,
            list(species),
            self.mol_mass,
        )
        atm.alpha = alpha_arr
        out = compute_mol_diff(
            _CFG,
            np.asarray(atm.Tco),
            np.asarray(atm.n_0),
            np.asarray(atm.g),
            np.asarray(atm.Hp),
            np.asarray(atm.dz),
            ms_arr,
            alpha_arr,
            list(species),
        )
        atm.Dzz = out["Dzz"]
        atm.Dzz_cen = out["Dzz_cen"]
        atm.vm = out["vm"]

    def BC_flux(self, atm: Any) -> None:
        """Read top/bottom boundary-condition fluxes from the configured files."""
        from .composition import species

        out = read_bc_flux(_CFG, list(species))
        atm.top_flux = out["top_flux"]
        atm.bot_flux = out["bot_flux"]
        atm.bot_vdep = out["bot_vdep"]
        atm.bot_fix_sp = out["bot_fix_sp"]

    def sp_sat(self, atm: Any) -> None:
        """Populate `atm.sat_p[sp]` for each condensable species."""
        out = compute_sat_p(list(_CFG.condense_sp), np.asarray(atm.Tco))
        for sp, sp_arr in out.items():
            atm.sat_p[sp] = sp_arr

    def read_sflux(self, var: Any, atm: Any) -> None:  # noqa: ARG002 (master's signature; legacy callers pass atm)
        """Read the stellar flux file, rebin onto `var.bins`, and write to var/atm."""
        out = read_sflux_binned(_CFG, np.asarray(var.bins))
        var.sflux_top = out["sflux_top"]
        var.sflux_din12_indx = out["sflux_din12_indx"]
