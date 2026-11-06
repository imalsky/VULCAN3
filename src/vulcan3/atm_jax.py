"""On-graph molecular-diffusion inputs.

`make_physical_inputs` bridges a host setup (cfg, var, atm) into the
differentiable `PhysicalInputs` leaves and the static `AtmSpec`; `_mol_diff`
rebuilds (Dzz, Dzz_cen, vm) from them with `atm_setup.mol_diff_jax`, the body
the host setup runs. The hydrostatic column on the graph is
`atm_setup.mu_dz_g_jax` (`atm_refresh.update_mu_dz_jax` in the loop).
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from .atm_setup import mol_diff_jax, settling_coeff_array, surface_gravity

__all__ = ["AtmSpec", "PhysicalInputs", "make_physical_inputs", "mol_diff"]


class PhysicalInputs(NamedTuple):
    """Differentiable atmosphere inputs (the leaves AD flows through).

    Every field is a JAX array. `pco`/`Tco`/`ymix`/`Kzz`/`vz` are per-layer or
    per-interface profiles; `gs`/`Rp` are scalars. To differentiate w.r.t. a
    *parameterisation* (T_irr, K_deep), compose the matching on-graph
    front-end (`analytical_TP_H14`, `atm_setup.kzz_profile_jax`) in front of
    the leaf it feeds.
    """

    pco: jnp.ndarray  # (nz,) cell-centre pressures, dyne/cm^2
    Tco: jnp.ndarray  # (nz,) temperature, K
    ymix: jnp.ndarray  # (nz, ni) mixing ratios (feeds mean molecular weight)
    Kzz: jnp.ndarray  # (nz-1,) eddy diffusion at interfaces
    vz: jnp.ndarray  # (nz-1,) vertical advection at interfaces
    gs: jnp.ndarray  # scalar surface gravity, cm/s^2
    Rp: jnp.ndarray  # scalar planet radius, cm


class AtmSpec(NamedTuple):
    """Static, non-differentiated configuration for `_mol_diff` and the settling velocity.

    Discrete host-side choices (species identities, background gas, transport toggles) plus the per-species constant
    arrays. Close over it; do not trace through it.
    """

    nz: int
    ni: int
    atm_base: str  # background gas -> Dzz fit + viscosity polynomial
    ms: jnp.ndarray  # (ni,) species molar masses, g/mol
    alpha: jnp.ndarray  # (ni,) thermal-diffusion factors (static lookup)
    nongas_mask: jnp.ndarray  # (ni,) bool, condensed/non-gas species
    settle_coeff: jnp.ndarray  # (ni,) rho_p * r_p**2, zero for gas species
    use_moldiff: bool
    use_vm_mol: bool
    use_settling: bool


def _mol_diff(phys: PhysicalInputs, spec: AtmSpec, n_0, gz, Hp, dz):
    """Molecular diffusion on the graph: (Dzz, Dzz_cen, vm).

    The body is `atm_setup.mol_diff_jax`, the same one the host setup path
    runs. All three are zero when `use_moldiff` is off.
    """
    nz, ni = spec.nz, spec.ni
    if not spec.use_moldiff:
        return (
            jnp.zeros((nz - 1, ni), dtype=jnp.float64),  # Dzz (interface)
            jnp.zeros((nz, ni), dtype=jnp.float64),  # Dzz_cen (cell)
            jnp.zeros((nz - 1, ni), dtype=jnp.float64),  # vm (interface)
        )
    return mol_diff_jax(
        spec.atm_base,
        phys.Tco,
        n_0,
        gz,
        Hp,
        dz,
        spec.ms,
        spec.alpha,
        spec.nongas_mask,
        use_vm_mol=spec.use_vm_mol,
    )


mol_diff = _mol_diff  # the public name; `_mol_diff` is deprecated (docs/api.md)


def make_physical_inputs(cfg, var, atm, species_list: list[str]) -> tuple[PhysicalInputs, AtmSpec]:
    """Bridge a legacy (cfg, var, atm) setup into (PhysicalInputs, AtmSpec).

    Pulls the differentiable profiles into `PhysicalInputs` and the discrete
    configuration into `AtmSpec`. Call after the host setup has populated `atm`
    (Tco, Kzz, ms, alpha, ...).
    """
    nz = int(np.asarray(atm.Tco).shape[0])
    ni = len(species_list)

    phys = PhysicalInputs(
        pco=jnp.asarray(atm.pco, dtype=jnp.float64),
        Tco=jnp.asarray(atm.Tco, dtype=jnp.float64),
        ymix=jnp.asarray(var.ymix, dtype=jnp.float64),
        Kzz=jnp.asarray(atm.Kzz, dtype=jnp.float64),
        vz=jnp.asarray(atm.vz, dtype=jnp.float64),
        gs=jnp.asarray(surface_gravity(cfg), dtype=jnp.float64),
        Rp=jnp.asarray(float(cfg.Rp), dtype=jnp.float64),
    )

    nongas = np.zeros(ni, dtype=bool)
    for sp in cfg.non_gas_sp:
        if sp in species_list:
            nongas[species_list.index(sp)] = True

    if bool(cfg.use_settling):
        settle_coeff = settling_coeff_array(
            cfg,
            list(species_list),
            getattr(atm, "rho_p", {}),
            getattr(atm, "r_p", {}),
        )
    else:
        settle_coeff = np.zeros(ni, dtype=np.float64)

    spec = AtmSpec(
        nz=nz,
        ni=ni,
        atm_base=str(cfg.atm_base),
        ms=jnp.asarray(atm.ms, dtype=jnp.float64),
        alpha=jnp.asarray(atm.alpha, dtype=jnp.float64),
        nongas_mask=jnp.asarray(nongas),
        settle_coeff=jnp.asarray(settle_coeff, dtype=jnp.float64),
        use_moldiff=bool(cfg.use_moldiff),
        use_vm_mol=bool(cfg.use_vm_mol),
        use_settling=bool(cfg.use_settling),
    )
    return phys, spec
