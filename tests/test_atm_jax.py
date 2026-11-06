"""Tests for the on-graph atmosphere pieces (`atm_setup.mu_dz_g_jax`, `atm_jax.mol_diff`).

Pins two things:
  1. The vm expression against the verbatim vm_branch port.
  2. Differentiability: forward-mode tangents match finite differences for gs
     and Tco through the hydrostatic column and molecular diffusion, and the
     Kzz-profile and Heng+14 T(P) front-ends are differentiable.
"""

from __future__ import annotations

import os
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vulcan3 import atm_setup
from vulcan3.atm_jax import AtmSpec, PhysicalInputs, make_physical_inputs, mol_diff
from vulcan3.atm_setup import (
    analytical_TP_H14,
    compute_mean_mass,
    compute_pico,
    mu_dz_g_jax,
)
from vulcan3.composition import species as _SPECIES
from vulcan3.phy_const import UNDERFLOW_DENOM, kb

os.environ.setdefault("OMP_NUM_THREADS", "1")

RTOL = 1e-12  # two builds of the same quantity agree to roundoff
FD_RTOL = 1e-5  # forward-mode tangent vs a central difference


def _rel(a, b):
    """Max relative error, with an absolute floor tied to the field's own magnitude.

    So all-zero reference fields compare cleanly (0 vs 0).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0:
        return 0.0
    floor = UNDERFLOW_DENOM + 1e-30 * float(np.abs(b).max())
    denom = np.maximum(np.abs(b), floor)
    return float(np.max(np.abs(a - b) / denom))


def _synthetic_mol_diff_setup():
    """Small H2-base column with a condensible, for the vm/settling branches."""
    species_list = ["H2", "He", "CH4", "H2O_l_s"]
    masses = {"H2": 2.016, "He": 4.0026, "CH4": 16.04, "H2O_l_s": 18.015}
    ms_arr = np.array([masses[s] for s in species_list], dtype=np.float64)
    nz, ni = 16, len(species_list)
    rng = np.random.default_rng(0)
    Tco = np.linspace(2200.0, 700.0, nz)
    n_0 = np.logspace(18, 10, nz)
    g = np.full(nz, 2140.0)
    Hp = np.full(nz, 5e6) + rng.uniform(0, 1e5, nz)
    dz = np.full(nz, 4e5) + rng.uniform(0, 1e4, nz)
    alpha = atm_setup._alpha_array_for_base("H2", species_list, lambda s: masses[s])
    return species_list, ms_arr, nz, ni, Tco, n_0, g, Hp, dz, alpha


def _reference_vm_branch_vm(Tco, n_0, g, Hp, dz, ms, alpha, species_list, non_gas_sp, atm_base):
    """Transparent NumPy port of the canonical vm_branch interface vm.

    Mirrors `op.update_mu_dz` from shami-EEG/VULCAN vm_branch VERBATIM,
    including its `np.roll(species_Hi, -1, axis=0)` layer-averaging. Do not
    port the build_atm.py copy instead: it drops `axis=0`, a latent
    species-mixing bug that op.update_mu_dz overrides during the run.
    """
    from vulcan3._transport_setup import _Dzz_gen_for_base
    from vulcan3.phy_const import Navo, kb

    Tco = np.asarray(Tco, np.float64)
    n_0 = np.asarray(n_0, np.float64)
    g = np.asarray(g, np.float64)
    Hp = np.asarray(Hp, np.float64)
    dz = np.asarray(dz, np.float64)
    ms = np.asarray(ms, np.float64)
    alpha = np.asarray(alpha, np.float64)
    nz, ni = Tco.shape[0], len(species_list)

    Dzz_gen = _Dzz_gen_for_base(atm_base)
    Tco_i = 0.5 * (Tco[:-1] + Tco[1:])
    n0_i = 0.5 * (n_0[:-1] + n_0[1:])
    Dzz = np.zeros((nz - 1, ni), dtype=np.float64)
    for i in range(ni):
        Dzz[:, i] = np.asarray(Dzz_gen(Tco_i, n0_i, ms[i]))
    for sp in non_gas_sp:
        if sp in species_list:
            Dzz[:, species_list.index(sp)] = 0.0

    Ti = 0.5 * (Tco[:-1] + Tco[1:])
    Hpi = 0.5 * (Hp[:-1] + Hp[1:])
    dzi = 0.5 * (dz[1:] + dz[:-1])
    delta_Ti = (np.roll(Tco, -1) - Tco)[:-1]
    species_Hi = ms[None, :] * g[:, None] / (Navo * kb * Tco[:, None])
    Hi_interf = 1.0 / (0.5 * (1.0 / species_Hi + 1.0 / np.roll(species_Hi, -1, axis=0)))
    Hi_interf = Hi_interf[:-1, :]
    return -Dzz * (Hi_interf - 1.0 / Hpi[:, None] + alpha[None, :] / Ti[:, None] * delta_Ti[:, None] / dzi[:, None])


def test_compute_mol_diff_vm_matches_vm_branch_reference():
    """compute_mol_diff's upwind vm equals the canonical vm_branch expression.

    The production kernel's discretization is test_diffusion_production_kernel.
    """
    species_list, ms_arr, nz, ni, Tco, n_0, g, Hp, dz, alpha = _synthetic_mol_diff_setup()
    cfg = types.SimpleNamespace(
        use_moldiff=True,
        atm_base="H2",
        non_gas_sp=["H2O_l_s"],
        use_vm_mol=True,
    )
    out = atm_setup.compute_mol_diff(cfg, Tco, n_0, g, Hp, dz, ms_arr, alpha, species_list)
    vm_ref = _reference_vm_branch_vm(Tco, n_0, g, Hp, dz, ms_arr, alpha, species_list, ["H2O_l_s"], "H2")
    assert out["vm"].shape == (nz - 1, ni), out["vm"].shape
    assert _rel(out["vm"], vm_ref) < RTOL
    assert np.any(np.asarray(vm_ref) != 0.0)  # the comparison is not vacuous


def test_vm_branch_differentiates_wrt_Tco():
    """Forward-mode tangent of the interface vm w.r.t. a Tco scale is finite and matches a central difference."""
    species_list, ms_arr, nz, ni, Tco, n_0, g, Hp, dz, alpha = _synthetic_mol_diff_setup()
    nongas = np.array([s == "H2O_l_s" for s in species_list])
    phys = PhysicalInputs(
        pco=jnp.ones(nz),
        Tco=jnp.asarray(Tco),
        ymix=jnp.ones((nz, ni)),
        Kzz=jnp.ones(nz - 1),
        vz=jnp.zeros(nz - 1),
        gs=jnp.float64(2140.0),
        Rp=jnp.float64(8e9),
    )
    spec = AtmSpec(
        nz=nz,
        ni=ni,
        atm_base="H2",
        ms=jnp.asarray(ms_arr),
        alpha=jnp.asarray(alpha),
        nongas_mask=jnp.asarray(nongas),
        settle_coeff=jnp.zeros(ni),
        use_moldiff=True,
        use_vm_mol=True,
        use_settling=False,
    )
    Tco_j = jnp.asarray(Tco)
    n0_j, g_j, Hp_j, dz_j = (jnp.asarray(a) for a in (n_0, g, Hp, dz))

    def vm_sum(scale):
        _, _, vm = mol_diff(phys._replace(Tco=Tco_j * scale), spec, n0_j, g_j, Hp_j, dz_j)
        return jnp.sum(vm)

    _, tangent = jax.jvp(vm_sum, (jnp.float64(1.0),), (jnp.float64(1.0),))
    eps = 1e-6
    fd = (vm_sum(jnp.float64(1.0 + eps)) - vm_sum(jnp.float64(1.0 - eps))) / (2 * eps)
    assert np.isfinite(float(tangent))
    assert float(tangent) != 0.0
    assert abs(float(tangent) - float(fd)) / (abs(float(fd)) + UNDERFLOW_DENOM) < FD_RTOL


# 2. Differentiability -- forward-mode tangents vs finite differences.


def _column(hd189_state, gs_scale, T_scale):
    """(dzi, Dzz) of the HD189 column through the live on-graph pieces.

    The hydrostatic integrator and molecular diffusion, at scaled gs and Tco. `M` is the ideal-gas law a T-profile
    gradient writes itself, as vulcan-forward does: the package has no on-graph M to test, and its tangent reaches Dzz.
    """
    from vulcan3.config import default_config

    atm = hd189_state.atm
    phys, spec = make_physical_inputs(default_config(), hd189_state.var, atm, list(_SPECIES))
    Tco = phys.Tco * T_scale
    M = phys.pco / (kb * Tco)
    g, Hp, dz, _zco, _zmco, dzi, _Ti, _Hpi = mu_dz_g_jax(
        int(atm.pref_indx),
        phys.gs * gs_scale,
        phys.Rp,
        Tco,
        compute_mean_mass(phys.ymix, spec.ms),
        compute_pico(phys.pco),
        spec.nz,
    )
    Dzz, _, _ = mol_diff(phys._replace(Tco=Tco), spec, M, g, Hp, dz)
    return dzi, Dzz


@pytest.mark.parametrize(
    ("knob", "field"),
    [("gs", "dzi"), ("Tco", "dzi"), ("Tco", "Dzz")],
)
def test_column_jvp_matches_fd(hd189_state, knob, field):
    """The tangent of each field under a unit scale of gs or Tco matches a central difference in every cell.

    A column bar would be set by the deepest cell: Dzz spans 11 orders, so a zeroed upper atmosphere would pass it.
    """
    index = ("dzi", "Dzz").index(field)

    def column(x):
        return _column(hd189_state, x, 1.0)[index] if knob == "gs" else _column(hd189_state, 1.0, x)[index]

    _, tangent = jax.jvp(column, (jnp.float64(1.0),), (jnp.float64(1.0),))
    eps = 1e-6
    fd = (np.asarray(column(jnp.float64(1.0 + eps))) - np.asarray(column(jnp.float64(1.0 - eps)))) / (2 * eps)
    tangent = np.asarray(tangent)
    assert np.any(fd != 0.0)
    bad = np.abs(tangent - fd) > FD_RTOL * np.abs(fd)  # where fd is zero, the tangent must be too
    assert not bad.any(), f"{int(bad.sum())} of {bad.size} cells off, first {np.argwhere(bad)[0]}"


def test_analytical_TP_front_end_differentiates():
    """The Heng+14 T(P) front-end composes onto the Tco leaf: dTco/dT_irr is finite.

    The pressure range stays narrow: `jax.scipy.special.expn`'s forward mode is slow over many decades.
    """
    pco = jnp.asarray(np.logspace(5.0, 4.0, 8))  # narrow -> small expn argument
    base = jnp.asarray([500.0, 1200.0, 0.1, 0.02, 0.7, 0.7], dtype=jnp.float64)

    def tprofile(T_irr):
        return jnp.sum(analytical_TP_H14(pco, base.at[1].set(T_irr), gs=2140.0, Pb=1e9))

    _, tangent = jax.jvp(tprofile, (jnp.float64(1200.0),), (jnp.float64(1.0),))
    assert np.isfinite(float(tangent))
    assert float(tangent) != 0.0


def test_kzz_profile_jax_differentiates():
    """dKzz/dK_deep is finite and nonzero where the K_deep floor binds.

    The JM16 floor is max(K_deep, 1e5*sqrt(300/(P*1e-3))); on logspace(7,-1)
    the profile term bottoms out at ~1.7e4 (P=1e7), so K_deep=1e6 binds on
    the deep layers and the tangent must be nonzero there.
    """
    pico_int = jnp.asarray(np.logspace(7, -1, 30))

    def total(K_deep):
        return jnp.sum(
            atm_setup.kzz_profile_jax(
                "JM16",
                pico_int,
                K_deep=K_deep,
                K_max=1e10,
                K_p_lev=0.1,
                const_Kzz=1e9,
            )
        )

    tangent = jax.jvp(total, (jnp.float64(1e6),), (jnp.float64(1.0),))[1]
    assert np.isfinite(float(tangent))
    assert float(tangent) != 0.0
