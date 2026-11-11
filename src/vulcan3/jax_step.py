# size-limit-ok: numpydoc sections on the public functions; the code is under the limit
"""Vmap-compatible JAX Ros2 step.

`jax_ros2_step(y, k_arr, dt, atm, net)` operates on a static atmosphere
snapshot (Kzz, Dzz, Hpi, ... plus the third-body density `atm.M`); ysum(y) is
handled inside the step. For batch parallelism over multiple atmospheres, vmap
over the leading axis of (y, k_arr, atm).

Units: CGS throughout: T in K, pressure in dyn cm^-2, number density in cm^-3, length in cm, time in s.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import _settings, composition as _composition
from .chem import NetworkArrays, chem_jac_analytical
from .chem_funs import chem_rhs_codegen as _chem_rhs, spec_list as _spec_list
from .config import REPAIR_ABS_FLOOR, default_config
from .phy_const import UNDERFLOW_DENOM, Navo, kb

if TYPE_CHECKING:
    import types

    from .config import Config

__all__ = ["IMPORT_ATOM_LIST", "AtmStatic", "DiffGrav", "compute_diff_grav", "jax_ros2_step", "make_atm_static"]

_SOLVER = _settings.solver()  # import-frozen
if _SOLVER == "reference":
    from .solver import (
        factor_block_thomas_diag_offdiag,
        solve_block_thomas_diag_offdiag,
    )
else:  # see solver_fast.py
    from .solver_fast import (  # type: ignore[assignment]  # the two kernels have one call shape
        factor as factor_block_thomas_diag_offdiag,
        solve as solve_block_thomas_diag_offdiag,
    )

_CFG = default_config()

# (atom, reservoir) pairs for the atom-conservation projection: each atom's
# per-layer production residual is distributed onto its abundant reservoir
# carrier. Only pairs the active network supports are used (C-H-N-O networks
# conserve H/O/C/N; SNCHO adds S via H2S). One reservoir per atom keeps the
# reservoir count matrix square, so one linear solve zeros every atom residual
# at once, including the H that H2S also carries. Each is its element's main
# carrier in an H2-dominated column; where a carrier is a trace, the stage
# repair drops a correction over `_REPAIR_MAX_CELL_FRAC` of it and the
# element-budget term sees what is left. Fixed: the tables are built at import.
_ATOM_RESERVOIRS = (
    ("H", "H2"),
    ("O", "H2O"),
    ("C", "CO"),
    ("N", "N2"),
    ("S", "H2S"),
)


def _build_chem_projection_tables() -> tuple[bool, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build static reservoir-projection tables for chemistry conservation.

    Conserves every atom in ``cfg.atom_list`` that has a composition column and
    a tracked reservoir species, so SNCHO also conserves S. ``reservoir_counts`` is invertible for
    every atom subset: the C, N and S columns each have a single nonzero entry,
    so the solve always reduces to the invertible H/O block.
    """
    compo_names: tuple[str, ...] = _composition.compo.dtype.names  # type: ignore[assignment]  # a structured array has names
    cfg_atoms = _CFG.atom_list
    pairs = tuple(
        (atom, reservoir)
        for atom, reservoir in _ATOM_RESERVOIRS
        if atom in cfg_atoms and atom in compo_names and reservoir in _spec_list
    )
    if not pairs:
        return (
            False,
            jnp.zeros((len(_spec_list), 0), dtype=jnp.float64),
            jnp.zeros((0,), dtype=jnp.int32),
            jnp.zeros((0, 0), dtype=jnp.float64),
        )

    project_atoms = tuple(atom for atom, _ in pairs)
    reservoir_species = tuple(reservoir for _, reservoir in pairs)

    row_by_species = {str(row["species"]): row for row in _composition.compo}
    atom_counts = np.asarray(
        [[float(row_by_species[sp][atom]) for atom in project_atoms] for sp in _spec_list],
        dtype=np.float64,
    )
    reservoir_idx = np.asarray(
        [_spec_list.index(sp) for sp in reservoir_species],
        dtype=np.int32,
    )
    reservoir_counts = atom_counts[reservoir_idx]
    inv_reservoir_counts = np.linalg.inv(reservoir_counts)
    return (
        True,
        jnp.asarray(atom_counts, dtype=jnp.float64),
        jnp.asarray(reservoir_idx, dtype=jnp.int32),
        jnp.asarray(inv_reservoir_counts, dtype=jnp.float64),
    )


(
    _CHEM_PROJECTION_ENABLED,
    _CHEM_ATOM_COUNTS,
    _CHEM_RESERVOIR_IDX,
    _CHEM_INV_RESERVOIR_COUNTS,
) = _build_chem_projection_tables()

# atom_list baked into the projection tables above; import-frozen (the tables
# are module constants). state._assert_frozen_knobs_match_import fails fast
# if a later make_config passes a different atom_list.
IMPORT_ATOM_LIST = tuple(_CFG.atom_list)


def _project_chem_rhs(rhs: jnp.ndarray) -> jnp.ndarray:
    """Project chemistry RHS onto per-atom conservation via abundant reservoirs."""
    if not _CHEM_PROJECTION_ENABLED:
        return rhs
    residual = rhs @ _CHEM_ATOM_COUNTS  # shape: (nz, n_atoms)
    correction = -residual @ _CHEM_INV_RESERVOIR_COUNTS
    # correction shape: (nz, n_reservoir)
    return rhs.at[:, _CHEM_RESERVOIR_IDX].add(correction)


def _project_chem_jac(chem_jac: jnp.ndarray) -> jnp.ndarray:
    """Apply the same reservoir projection to chemistry Jacobian rows."""
    if not _CHEM_PROJECTION_ENABLED:
        return chem_jac
    residual = jnp.einsum("ia,zij->zaj", _CHEM_ATOM_COUNTS, chem_jac)
    # residual shape: (nz, n_atoms, ni)
    correction = -jnp.einsum("ar,zaj->zrj", _CHEM_INV_RESERVOIR_COUNTS, residual)
    # correction shape: (nz, n_reservoir, ni)
    return chem_jac.at[:, _CHEM_RESERVOIR_IDX, :].add(correction)


def _projected_chem_rhs(y: jnp.ndarray, M: jnp.ndarray, k_arr: jnp.ndarray) -> jnp.ndarray:
    """Evaluate and conservatively project the generated chemistry RHS."""
    return _project_chem_rhs(_chem_rhs(y, M, k_arr))


def _stage_defect(
    k: jnp.ndarray,
    b_tr: jnp.ndarray,
    c0: jnp.ndarray,
    diag_d: jnp.ndarray,
    sup_d: jnp.ndarray,
    sub_d: jnp.ndarray,
    with_scale: bool = False,
) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
    """Per-layer element defect `c0 a^T k - a^T (T k) - a^T b_tr` of a Ros2 stage vector, shape (nz, n_atoms).

    Zero for the exact solution of the stage system because the projected chemistry terms carry no element residual.
    `T` is the species-diagonal transport tridiagonal of the matrix (bands `diag_d`, `sup_d`, `sub_d`). With
    `with_scale` also returns the atom-weighted sum of the terms' absolute sizes, the scale against which the
    defect's roundoff floor is set.
    """
    Tk = diag_d * k
    Tk = Tk.at[:-1].add(sup_d * k[1:])
    Tk = Tk.at[1:].add(sub_d * k[:-1])
    defect = (c0 * k - Tk - b_tr) @ _CHEM_ATOM_COUNTS
    if not with_scale:
        return defect
    parts = jnp.abs(c0 * k) + jnp.abs(diag_d * k) + jnp.abs(b_tr)
    parts = parts.at[:-1].add(jnp.abs(sup_d * k[1:]))
    parts = parts.at[1:].add(jnp.abs(sub_d * k[:-1]))
    return defect, parts @ _CHEM_ATOM_COUNTS


# Stage-defect roundoff floor relative to the size of its terms (~500 float64
# ulps).
_DEFECT_FLOOR = 1e-13

# Largest repair correction on a carrier cell, as a fraction of max(cell,
# |raw stage change|); above it the reservoir is a trace there and the
# correction would damage the cell.
_REPAIR_MAX_CELL_FRAC = 1.0

# Layers per scan iteration of the repair sweep: fewest kernels at an
# acceptable jvp compile time.
_REPAIR_SWEEP_UNROLL = 8


def _tridiagonal_solve(dl: jnp.ndarray, d: jnp.ndarray, du: jnp.ndarray, g: jnp.ndarray) -> jnp.ndarray:
    """Solve `dl x[i-1] + d x[i] + du x[i+1] = g` along axis 0.

    One system per trailing index (`dl[0]`, `du[-1]` unused).

    LAPACK dgtsv partial-pivoting elimination as two rolled `lax.scan`s of
    elementwise ops, so lanes, reservoirs and tangent directions share
    kernels; pivoted because central-difference drift can zero an unpivoted
    pivot. Replaces `lax.linalg.tridiagonal_solve`, a per-system cuSPARSE
    call on GPU.
    """
    du = du.at[-1].set(0.0)  # the last swap reads it as the fill-in

    def fwd(
        row: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        below: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        d_i, du_i, b_i = row  # row i as the sweep has left it
        a, d_n, du_n, b_n = below  # row i+1 as given; `a` is its sub-diagonal
        swap = jnp.abs(d_i) < jnp.abs(a)
        fact = jnp.where(swap, d_i, a) / jnp.where(swap, a, d_i)
        done = (
            jnp.where(swap, a, d_i),
            jnp.where(swap, d_n, du_i),
            jnp.where(swap, du_n, 0.0),
            jnp.where(swap, b_n, b_i),
        )
        nxt = (
            jnp.where(swap, du_i - fact * d_n, d_n - fact * du_i),
            jnp.where(swap, -fact * du_n, du_n),
            jnp.where(swap, b_i - fact * b_n, b_n - fact * b_i),
        )
        return nxt, done

    (d_last, _, b_last), (dd, du1, du2, bb) = jax.lax.scan(
        fwd,
        (d[0], du[0], g[0]),
        (dl[1:], d[1:], du[1:], g[1:]),
        unroll=_REPAIR_SWEEP_UNROLL,
    )
    x_last = b_last / d_last

    def bwd(
        carry: tuple[jnp.ndarray, jnp.ndarray], row: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        x1, x2 = carry  # x[i+1], x[i+2]
        d_i, du_i, du2_i, b_i = row
        x = (b_i - du_i * x1 - du2_i * x2) / d_i
        return (x, x1), x

    _, xs = jax.lax.scan(
        bwd,
        (x_last, jnp.zeros_like(x_last)),
        (dd, du1, du2, bb),
        unroll=_REPAIR_SWEEP_UNROLL,
        reverse=True,
    )
    return jnp.concatenate([xs, x_last[None]])


def _repair_stage(
    k: jnp.ndarray,
    b_tr: jnp.ndarray,
    c0: jnp.ndarray,
    diag_d: jnp.ndarray,
    sup_d: jnp.ndarray,
    sub_d: jnp.ndarray,
    fix_mask: jnp.ndarray | None,
    n_tot: jnp.ndarray,
    y: jnp.ndarray,
) -> jnp.ndarray:
    """Move a Ros2 stage vector's per-layer element defect (`_stage_defect`) onto the reservoir species.

    One scalar tridiagonal solve `(c0 - T_rho) c = g` per reservoir. `n_tot` is the (nz, 1) layer density;
    the reservoir cells of `y` (the step's start state) bound the correction.

    At small c0 = 1/(gamma dt) the pivoted LU leaks elements. A defect is
    corrected only above both `_DEFECT_FLOOR` of its terms and
    `REPAIR_ABS_FLOOR` of the layer density. Pinned layers are skipped. A
    correction over `_REPAIR_MAX_CELL_FRAC` of its carrier is dropped. Not in
    VULCAN 2.0 (op.py:2914, :2929).
    """
    if not _CHEM_PROJECTION_ENABLED:
        return k
    ridx = _CHEM_RESERVOIR_IDX
    defect, scale = _stage_defect(k, b_tr, c0, diag_d, sup_d, sub_d, with_scale=True)
    floor = jnp.maximum(_DEFECT_FLOOR * scale, REPAIR_ABS_FLOOR * c0 * n_tot)
    defect = jnp.where(jnp.abs(defect) > floor, defect, 0.0)
    g = -(defect @ _CHEM_INV_RESERVOIR_COUNTS)  # (nz, n_reservoir)
    pad = jnp.zeros((1, ridx.shape[0]))
    d = c0 - diag_d[:, ridx]
    du = jnp.concatenate([-sup_d[:, ridx], pad])
    dl = jnp.concatenate([pad, -sub_d[:, ridx]])
    if fix_mask is not None:
        pinned = jnp.any(fix_mask, axis=1, keepdims=True)
        d = jnp.where(pinned, c0, d)
        du = jnp.where(pinned, 0.0, du)
        dl = jnp.where(pinned, 0.0, dl)
        g = jnp.where(pinned, 0.0, g)
    c = _tridiagonal_solve(dl, d, du, g)
    # Per layer and atom: drop a correction the carrier cell cannot carry --
    # one larger than both the cell's own content and the carrier's raw stage
    # change -- and leave the raw solve there. That layer's element budget
    # stays open, which the certificate's cumulative element-budget term sees.
    cap = _REPAIR_MAX_CELL_FRAC * jnp.maximum(y[:, ridx], jnp.abs(k[:, ridx]))
    c = jnp.where(jnp.abs(c) > cap, 0.0, c)
    return k.at[:, ridx].add(c)


class AtmStatic(NamedTuple):
    """Atmosphere parameters held constant within a Ros2 step.

    nz comes from Tco's leading axis, ni from ms's.
    """

    Kzz: jnp.ndarray  # (nz-1,), cm^2 s^-1
    Dzz: jnp.ndarray  # (nz-1, ni), cm^2 s^-1
    dzi: jnp.ndarray  # (nz-1,), cm
    vz: jnp.ndarray  # (nz-1,), cm s^-1
    Hpi: jnp.ndarray  # (nz-1,), cm
    Ti: jnp.ndarray  # (nz-1,), K
    Tco: jnp.ndarray  # (nz,), K
    g: jnp.ndarray  # (nz,), cm s^-2
    ms: jnp.ndarray  # (ni,), g mol^-1
    alpha: jnp.ndarray  # (ni,), thermal diffusion factor
    M: jnp.ndarray  # (nz,), cm^-3
    vm: jnp.ndarray  # (nz-1, ni), cm s^-1, interface molecular-diffusion drift velocity
    vs: jnp.ndarray  # (nz-1, ni), cm s^-1
    top_flux: jnp.ndarray  # (ni,), cm^-2 s^-1
    bot_flux: jnp.ndarray  # (ni,), cm^-2 s^-1
    bot_vdep: jnp.ndarray  # (ni,), cm s^-1
    gas_indx_mask: jnp.ndarray  # (ni,) bool
    diff_esc_mask: jnp.ndarray  # (ni,) bool, species in cfg.diff_esc
    # Blend weight (1.0 upwind, 0.0 central): a bool at build, float64 when
    # the hybrid runner splices it in.
    use_vm_mol: bool | float | jnp.ndarray
    use_settling: bool
    use_topflux: bool
    use_botflux: bool


class DiffGrav(NamedTuple):
    """Pre-baked y-independent transport contributions to the mol-diff blocks.

    Computed once per Ros2 step (atm refresh recomputes); reused for both
    stages of the Rosenbrock step. `dz_ave = 0.5*(dzi[j-1] + dzi[j])` is
    also consumed by the ysum-dependent eddy/mol terms.
    """

    A_grav_int: jnp.ndarray
    B_grav_int: jnp.ndarray
    C_grav_int: jnp.ndarray
    bdry0_grav: jnp.ndarray
    bdry_top_grav: jnp.ndarray
    dz_ave: jnp.ndarray


def compute_diff_grav(atm: AtmStatic) -> DiffGrav:
    """y-independent transport piece of the molecular-diffusion blocks.

    Gravity-mode or upwind-`vm` molecular diffusion, blended by `atm.use_vm_mol`, plus the settling terms when
    `atm.use_settling` is on.

    Parameters
    ----------
    atm : AtmStatic
        Atmosphere for this step.

    Returns
    -------
    DiffGrav
        Interior block coefficients `A_grav_int`, `B_grav_int`, `C_grav_int` [nz - 2, ni] (s^-1), boundary terms
        `bdry0_grav`, `bdry_top_grav` [2, ni] (s^-1), and `dz_ave` [nz - 2] (cm).
    """
    dzi, Hpi, Ti, Tco, g, ms, alpha, Dzz = (
        atm.dzi,
        atm.Hpi,
        atm.Ti,
        atm.Tco,
        atm.g,
        atm.ms,
        atm.alpha,
        atm.Dzz,
    )
    vm = atm.vm
    vs = atm.vs
    nz = atm.Tco.shape[0]
    j_int = jnp.arange(1, nz - 1)
    dz_ave = 0.5 * (dzi[j_int - 1] + dzi[j_int])  # (nz-2,)
    use_vm = jnp.asarray(atm.use_vm_mol, dtype=jnp.float64)
    use_set = jnp.asarray(atm.use_settling, dtype=jnp.float64)
    mol_active = jnp.any(Dzz != 0.0)
    Hpi = jnp.where(mol_active, Hpi, jnp.ones_like(Hpi))

    grav_j = (
        -1.0 / Hpi[j_int][:, None]
        + ms[None, :] * g[j_int][:, None] / (Navo * kb * Ti[j_int][:, None])
        + alpha[None, :] / Ti[j_int][:, None] * (Tco[j_int + 1][:, None] - Tco[j_int][:, None]) / dzi[j_int][:, None]
    )
    grav_jm = (
        -1.0 / Hpi[j_int - 1][:, None]
        + ms[None, :] * g[j_int][:, None] / (Navo * kb * Ti[j_int - 1][:, None])
        + alpha[None, :]
        / Ti[j_int - 1][:, None]
        * (Tco[j_int][:, None] - Tco[j_int - 1][:, None])
        / dzi[j_int - 1][:, None]
    )
    grav_jp_b = (
        -1.0 / Hpi[j_int][:, None]
        + ms[None, :] * g[j_int + 1][:, None] / (Navo * kb * Ti[j_int][:, None])
        + alpha[None, :] / Ti[j_int][:, None] * (Tco[j_int + 1][:, None] - Tco[j_int][:, None]) / dzi[j_int][:, None]
    )
    grav_jm_c = (
        -1.0 / Hpi[j_int - 1][:, None]
        + ms[None, :] * g[j_int - 1][:, None] / (Navo * kb * Ti[j_int - 1][:, None])
        + alpha[None, :]
        / Ti[j_int - 1][:, None]
        * (Tco[j_int][:, None] - Tco[j_int - 1][:, None])
        / dzi[j_int - 1][:, None]
    )

    inv_2dz_ave = 1.0 / (2.0 * dz_ave[:, None])
    A_grav_int = inv_2dz_ave * (Dzz[j_int] * grav_j - Dzz[j_int - 1] * grav_jm)
    B_grav_int = inv_2dz_ave * Dzz[j_int] * grav_jp_b
    C_grav_int = -inv_2dz_ave * Dzz[j_int - 1] * grav_jm_c

    bdry0_grav = (
        1.0
        / dzi[0]
        * Dzz[0]
        / 2.0
        * (-1.0 / Hpi[0] + ms * g[0] / (Navo * kb * Ti[0]) + alpha / Ti[0] * (Tco[1] - Tco[0]) / dzi[0])
    )
    bdry_top_grav = (
        -1.0
        / dzi[-1]
        * Dzz[-1]
        / 2.0
        * (-1.0 / Hpi[-1] + ms * g[-1] / (Navo * kb * Ti[-1]) + alpha / Ti[-1] * (Tco[-1] - Tco[-2]) / dzi[-1])
    )

    # Upwind molecular-diffusion advection variant (`use_vm_mol`). `vm` is the
    # interface drift velocity (nz-1, ni): vm[k] acts on the interface between
    # cells k and k+1, so vm[-1] is the top interface (matches op.diffdf_vm).
    A_vm_int = -((vm[j_int] > 0) * vm[j_int] - (vm[j_int - 1] < 0) * vm[j_int - 1]) / dz_ave[:, None]
    B_vm_int = -((vm[j_int] < 0) * vm[j_int]) / dz_ave[:, None]
    C_vm_int = ((vm[j_int - 1] > 0) * vm[j_int - 1]) / dz_ave[:, None]
    bdry0_vm = -((vm[0] > 0) * vm[0]) / dzi[0]
    bdry0_vm_B = -((vm[0] < 0) * vm[0]) / dzi[0]
    bdry_top_vm = ((vm[-1] < 0) * vm[-1]) / dzi[-1]
    bdry_top_vm_C = ((vm[-1] > 0) * vm[-1]) / dzi[-1]

    # Settling velocity is additive to either gravity-mode or vm-mode mol-diff.
    A_vs_int = -((vs[j_int] > 0) * vs[j_int] - (vs[j_int - 1] < 0) * vs[j_int - 1]) / dz_ave[:, None]
    B_vs_int = -((vs[j_int] < 0) * vs[j_int]) / dz_ave[:, None]
    C_vs_int = ((vs[j_int - 1] > 0) * vs[j_int - 1]) / dz_ave[:, None]
    bdry0_vs = -((vs[0] > 0) * vs[0]) / dzi[0]
    bdry0_vs_B = -((vs[0] < 0) * vs[0]) / dzi[0]
    bdry_top_vs = ((vs[-1] < 0) * vs[-1]) / dzi[-1]
    bdry_top_vs_C = ((vs[-1] > 0) * vs[-1]) / dzi[-1]

    A_base = (1.0 - use_vm) * A_grav_int + use_vm * A_vm_int
    B_base = (1.0 - use_vm) * B_grav_int + use_vm * B_vm_int
    C_base = (1.0 - use_vm) * C_grav_int + use_vm * C_vm_int

    bdry0_A_base = (1.0 - use_vm) * bdry0_grav + use_vm * bdry0_vm
    bdry0_B_base = (1.0 - use_vm) * bdry0_grav + use_vm * bdry0_vm_B
    bdry_top_A_base = (1.0 - use_vm) * bdry_top_grav + use_vm * bdry_top_vm
    bdry_top_C_base = (1.0 - use_vm) * bdry_top_grav + use_vm * bdry_top_vm_C

    A_extra = A_base + use_set * A_vs_int
    B_extra = B_base + use_set * B_vs_int
    C_extra = C_base + use_set * C_vs_int
    bdry0_A = bdry0_A_base + use_set * bdry0_vs
    bdry0_B = bdry0_B_base + use_set * bdry0_vs_B
    bdry_top_A = bdry_top_A_base + use_set * bdry_top_vs
    bdry_top_C = bdry_top_C_base + use_set * bdry_top_vs_C

    return DiffGrav(
        A_grav_int=A_extra,
        B_grav_int=B_extra,
        C_grav_int=C_extra,
        bdry0_grav=jnp.stack([bdry0_A, bdry0_B], axis=0),
        bdry_top_grav=jnp.stack([bdry_top_A, bdry_top_C], axis=0),
        dz_ave=dz_ave,
    )


def _build_diff_coeffs_jax(
    y: jnp.ndarray, atm: AtmStatic, grav: DiffGrav
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Diffusion (eddy + molecular) coefficient blocks for one Ros2 stage.

    `grav` carries the y-independent gravity contribution; this function
    layers in the y-dependent part. Returns
    (A_eddy, B_eddy, C_eddy, A_mol, B_mol, C_mol, ysum).
    """
    Kzz, Dzz, dzi, vz = atm.Kzz, atm.Dzz, atm.dzi, atm.vz
    ni = atm.ms.shape[0]
    nz = atm.Tco.shape[0]

    ysum = jnp.sum(jnp.where(atm.gas_indx_mask[None, :], y, 0.0), axis=1)
    ysum = jnp.maximum(ysum, UNDERFLOW_DENOM)

    # Interior rows (nz-2) first, then concatenate the boundary rows.
    j_int = jnp.arange(1, nz - 1)
    dz_ave = grav.dz_ave  # (nz-2,)

    A_eddy_int = (
        -1.0
        / dz_ave
        * (
            Kzz[j_int] / dzi[j_int] * (ysum[j_int + 1] + ysum[j_int]) / 2.0
            + Kzz[j_int - 1] / dzi[j_int - 1] * (ysum[j_int] + ysum[j_int - 1]) / 2.0
        )
        / ysum[j_int]
    )
    B_eddy_int = 1.0 / dz_ave * Kzz[j_int] / dzi[j_int] * (ysum[j_int + 1] + ysum[j_int]) / 2.0 / ysum[j_int + 1]
    C_eddy_int = (
        1.0 / dz_ave * Kzz[j_int - 1] / dzi[j_int - 1] * (ysum[j_int] + ysum[j_int - 1]) / 2.0 / ysum[j_int - 1]
    )

    # Vertical advection (bool*value gives the upwind switch).
    A_eddy_int = A_eddy_int - ((vz[j_int] > 0) * vz[j_int] - (vz[j_int - 1] < 0) * vz[j_int - 1]) / dz_ave
    B_eddy_int = B_eddy_int - ((vz[j_int] < 0) * vz[j_int]) / dz_ave
    C_eddy_int = C_eddy_int + ((vz[j_int - 1] > 0) * vz[j_int - 1]) / dz_ave

    A_eddy_0 = -1.0 / dzi[0] * (Kzz[0] / dzi[0]) * (ysum[1] + ysum[0]) / 2.0 / ysum[0]
    A_eddy_0 = A_eddy_0 - ((vz[0] > 0) * vz[0]) / dzi[0]
    B_eddy_0 = 1.0 / dzi[0] * (Kzz[0] / dzi[0]) * (ysum[1] + ysum[0]) / 2.0 / ysum[1]
    B_eddy_0 = B_eddy_0 - ((vz[0] < 0) * vz[0]) / dzi[0]
    C_eddy_0 = 0.0

    A_eddy_top = -1.0 / dzi[-1] * (Kzz[-1] / dzi[-1]) * (ysum[-1] + ysum[-2]) / 2.0 / ysum[-1]
    A_eddy_top = A_eddy_top + ((vz[-1] < 0) * vz[-1]) / dzi[-1]
    C_eddy_top = 1.0 / dzi[-1] * (Kzz[-1] / dzi[-1]) * (ysum[-1] + ysum[-2]) / 2.0 / ysum[-2]
    C_eddy_top = C_eddy_top + ((vz[-1] > 0) * vz[-1]) / dzi[-1]
    B_eddy_top = 0.0

    A_eddy = jnp.concatenate([jnp.array([A_eddy_0]), A_eddy_int, jnp.array([A_eddy_top])])
    B_eddy = jnp.concatenate([jnp.array([B_eddy_0]), B_eddy_int, jnp.array([B_eddy_top])])
    C_eddy = jnp.concatenate([jnp.array([C_eddy_0]), C_eddy_int, jnp.array([C_eddy_top])])

    # Molecular-diffusion per-species blocks: y-dependent part + pre-baked grav.
    Ai_int = (
        -1.0
        / dz_ave[:, None]
        * (
            Dzz[j_int] / dzi[j_int][:, None] * (ysum[j_int + 1][:, None] + ysum[j_int][:, None]) / 2.0
            + Dzz[j_int - 1] / dzi[j_int - 1][:, None] * (ysum[j_int][:, None] + ysum[j_int - 1][:, None]) / 2.0
        )
        / ysum[j_int][:, None]
    )
    Ai_int = Ai_int + grav.A_grav_int

    Bi_int = (
        1.0
        / dz_ave[:, None]
        * Dzz[j_int]
        / dzi[j_int][:, None]
        * (ysum[j_int + 1][:, None] + ysum[j_int][:, None])
        / 2.0
        / ysum[j_int + 1][:, None]
    )
    Bi_int = Bi_int + grav.B_grav_int

    Ci_int = (
        1.0
        / dz_ave[:, None]
        * Dzz[j_int - 1]
        / dzi[j_int - 1][:, None]
        * (ysum[j_int][:, None] + ysum[j_int - 1][:, None])
        / 2.0
        / ysum[j_int - 1][:, None]
    )
    Ci_int = Ci_int + grav.C_grav_int

    Ai_0 = -1.0 / dzi[0] * (Dzz[0] / dzi[0]) * (ysum[1] + ysum[0]) / 2.0 / ysum[0] + grav.bdry0_grav[0]
    Bi_0 = 1.0 / dzi[0] * (Dzz[0] / dzi[0]) * (ysum[1] + ysum[0]) / 2.0 / ysum[1] + grav.bdry0_grav[1]
    Ci_0 = jnp.zeros(ni)

    Ai_top = -1.0 / dzi[-1] * (Dzz[-1] / dzi[-1]) * (ysum[-1] + ysum[-2]) / 2.0 / ysum[-1] + grav.bdry_top_grav[0]
    Ci_top = 1.0 / dzi[-1] * (Dzz[-1] / dzi[-1]) * (ysum[-1] + ysum[-2]) / 2.0 / ysum[-2] + grav.bdry_top_grav[1]
    Bi_top = jnp.zeros(ni)

    A_mol = jnp.concatenate([Ai_0[None], Ai_int, Ai_top[None]], axis=0)
    B_mol = jnp.concatenate([Bi_0[None], Bi_int, Bi_top[None]], axis=0)
    C_mol = jnp.concatenate([Ci_0[None], Ci_int, Ci_top[None]], axis=0)

    return A_eddy, B_eddy, C_eddy, A_mol, B_mol, C_mol, ysum


def _apply_diffusion_jax(
    y: jnp.ndarray,
    A_eddy: jnp.ndarray,
    B_eddy: jnp.ndarray,
    C_eddy: jnp.ndarray,
    A_mol: jnp.ndarray,
    B_mol: jnp.ndarray,
    C_mol: jnp.ndarray,
    atm: AtmStatic,
) -> jnp.ndarray:
    """diff[j] = (A+Ai)*y[j] + (B+Bi)*y[j+1] + (C+Ci)*y[j-1], plus BC fluxes."""
    A_total = A_eddy[:, None] + A_mol
    B_total = B_eddy[:, None] + B_mol
    C_total = C_eddy[:, None] + C_mol
    diff_0 = A_total[0] * y[0] + B_total[0] * y[1]
    diff_top = A_total[-1] * y[-1] + C_total[-1] * y[-2]
    diff_int = A_total[1:-1] * y[1:-1] + B_total[1:-1] * y[2:] + C_total[1:-1] * y[:-2]
    diff = jnp.concatenate([diff_0[None], diff_int, diff_top[None]], axis=0)
    diff = diff.at[-1].add(jnp.where(atm.use_topflux, atm.top_flux / atm.dzi[-1], jnp.zeros_like(atm.top_flux)))
    return diff.at[0].add(
        jnp.where(
            atm.use_botflux,
            (atm.bot_flux - y[0] * atm.bot_vdep) / atm.dzi[0],
            jnp.zeros_like(atm.bot_flux),
        )
    )


# gamma = 1 + 1/sqrt(2) (Verwer et al. 1999; op.py). The transport Jacobian
# omits the ysum coupling, so this is a second-order W-method, as in both
# upstreams.
_ROS2_GAMMA = 1.0 + 2.0**-0.5


def _ros2_stages(
    y: jnp.ndarray,
    k_arr: jnp.ndarray,
    dt: jnp.ndarray,
    atm: AtmStatic,
    net: NetworkArrays,
    fix_mask: jnp.ndarray | None,
    matrix_free: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, tuple[jnp.ndarray, ...]]:
    """Return the two Ros2 stage solves.

    Returns (k1, k2, yk2, ident) with `ident` = (c0, diag_d, sup_d, sub_d, b_tr1, b_tr2): the matrix's transport
    bands and the transport part of each stage RHS, which is what the per-layer element identity of a stage vector
    needs (`_stage_defect`).
    """
    r = _ROS2_GAMMA
    c0 = 1.0 / (r * dt)
    ni = atm.ms.shape[0]
    M = atm.M

    # y-independent gravity terms; reused for the y and yk2 evaluations.
    grav = compute_diff_grav(atm)

    A_eddy, B_eddy, C_eddy, A_mol, B_mol, C_mol, _ = _build_diff_coeffs_jax(y, atm, grav)

    diff_at_y = _apply_diffusion_jax(y, A_eddy, B_eddy, C_eddy, A_mol, B_mol, C_mol, atm)
    rhs_y = _projected_chem_rhs(y, M, k_arr) + diff_at_y
    # Analytical Jacobian: <= 1e-13 vs the AD (jacrev) oracle, a gather along
    # the network's static tables.
    chem_J = _project_chem_jac(chem_jac_analytical(y, M, k_arr, net))

    # Diffusion blocks are diagonal-in-species: pass off-diagonals as (nz-1, ni)
    # vectors to skip the O(ni^3) C @ invA_B matmul in forward elimination.
    diag_d = A_eddy[:, None] + A_mol
    sup_d = B_eddy[:-1, None] + B_mol[:-1]
    sub_d = C_eddy[1:, None] + C_mol[1:]
    bot_vdep_term = jnp.where(
        atm.use_botflux,
        -atm.bot_vdep / atm.dzi[0],
        jnp.zeros_like(atm.bot_vdep),
    )
    diag_d = diag_d.at[0].add(bot_vdep_term)
    # Diffusion-limited escape at TOA (`top_flux / y[-1]` on the top-layer
    # diagonal). Of upstream's live Jacobians, only the upwind ones carry it
    # (exoclime@80f75b9 op.py:2044-2121, 2366+; vm_branch@84d010d
    # op.py:2123-2200, 2445+), so the gate is `use_vm_mol`, not "diff_esc
    # non-empty". The inner `where` mirrors upstream's `y > 0` guard and keeps
    # the division and its derivative finite at y = 0. The entry exceeds the
    # true derivative by dzi[-1] (op.py:2106-2107, vm_branch op.py:2185-2186);
    # kept for bit-parity, LHS only.
    y_top_pos = y[-1] > 0.0
    diff_lim = jnp.where(
        atm.diff_esc_mask & y_top_pos,
        atm.top_flux / jnp.where(y_top_pos, y[-1], 1.0),
        0.0,
    )
    diag_d = diag_d.at[-1].add(diff_lim * jnp.asarray(atm.use_vm_mol, dtype=jnp.float64))

    eye = jnp.eye(ni)
    on_diag = eye[None] != 0.0  # (1, ni, ni), the block diagonal
    # Diagonal and pins go in the elementwise build; a `.at` scatter makes XLA
    # copy the whole (nz, ni, ni) block. The result is bitwise equal to the
    # scatter.
    diag = c0 * eye[None] - chem_J
    diag = jnp.where(on_diag, diag - diag_d[:, :, None], diag)
    sup_neg = -sup_d  # (nz-1, ni)
    sub_neg = -sub_d  # (nz-1, ni)

    if fix_mask is not None:
        # A pinned row: zero off the diagonal, c0 on it.
        diag = jnp.where(fix_mask[:, :, None], jnp.where(on_diag, c0, 0.0), diag)
        rhs_y = jnp.where(fix_mask, 0.0, rhs_y)
        sup_neg = jnp.where(fix_mask[:-1], 0.0, sup_neg)
        sub_neg = jnp.where(fix_mask[1:], 0.0, sub_neg)

    def matvec(x: jnp.ndarray) -> jnp.ndarray:
        # The same operator without its dense block, for the tangent's dA x.
        # The chemistry part is the jvp of the RHS that `chem_J`
        # differentiates analytically (they agree to ~1e-13), so a tangent
        # costs a second-order jvp of the RHS instead of a dense dJ per
        # direction. Pinned rows as the dense build pins them.
        jx = jax.jvp(lambda yy: _chem_rhs(yy, M, k_arr), (y,), (x,))[1]
        out = c0 * x - _project_chem_rhs(jx) - diag_d * x
        if fix_mask is not None:
            out = jnp.where(fix_mask, c0 * x, out)
        out = out.at[:-1].add(sup_neg * x[1:])
        return out.at[1:].add(sub_neg * x[:-1])

    # The reference pair differentiates through the LU and takes no operator.
    # `matrix_free=False` keeps the dense one: reverse-over-forward of the RHS
    # is slower in reverse mode, so the adjoint keeps it.
    solve_kw = {"matvec": matvec} if matrix_free and _SOLVER != "reference" else {}
    factors = factor_block_thomas_diag_offdiag(diag, sup_neg, sub_neg)
    k1 = solve_block_thomas_diag_offdiag(factors, rhs_y, **solve_kw)
    n_tot = jnp.sum(y, axis=1, keepdims=True)
    k1 = _repair_stage(k1, diff_at_y, c0, diag_d, sup_d, sub_d, fix_mask, n_tot, y)

    yk2 = y + k1 / r
    A_eddy2, B_eddy2, C_eddy2, A_mol2, B_mol2, C_mol2, _ = _build_diff_coeffs_jax(yk2, atm, grav)
    diff_at_yk2 = _apply_diffusion_jax(yk2, A_eddy2, B_eddy2, C_eddy2, A_mol2, B_mol2, C_mol2, atm)
    rhs_yk2 = _projected_chem_rhs(yk2, M, k_arr) + diff_at_yk2
    if fix_mask is not None:
        rhs_yk2 = jnp.where(fix_mask, 0.0, rhs_yk2)

    rhs2 = rhs_yk2 - (2.0 / (r * dt)) * k1
    k2 = solve_block_thomas_diag_offdiag(factors, rhs2, **solve_kw)
    # Transport part of the stage-2 RHS (the projected chemistry term carries
    # no element content; the k1 term does).
    b_tr2 = diff_at_yk2 - (2.0 / (r * dt)) * k1
    k2 = _repair_stage(k2, b_tr2, c0, diag_d, sup_d, sub_d, fix_mask, n_tot, y)
    return k1, k2, yk2, (c0, diag_d, sup_d, sub_d, diff_at_y, b_tr2)


@functools.partial(jax.jit, static_argnames=("matrix_free",))
def jax_ros2_step(
    y: jnp.ndarray,
    k_arr: jnp.ndarray,
    dt: jnp.ndarray,
    atm: AtmStatic,
    net: NetworkArrays,
    fix_mask: jnp.ndarray | None = None,
    matrix_free: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """One 2nd-order Rosenbrock step.

    Parameters
    ----------
    y : jnp.ndarray [nz, ni]
        Number densities, cm^-3.
    k_arr : jnp.ndarray [nr + 1, nz]
        Rate constants in the units of `rates_jax.build_rate_array`, row 0 unused.
    dt : jnp.ndarray []
        Time step, s.
    atm : AtmStatic
        Atmosphere for this step.
    net : NetworkArrays
        Active network.
    fix_mask : jnp.ndarray [nz, ni] or None, optional
        bool; pins the True (layer, species) entries by zeroing their LHS rows off the diagonal (c0 on it) and their
        RHS entries.
    matrix_free : bool, optional
        The stage operator the solve's AD rules use (the primal is the same either way): True for forward mode,
        False for reverse mode (`_ros2_stages`).

    Returns
    -------
    sol : jnp.ndarray [nz, ni]
        Number densities after the step, cm^-3.
    delta_arr : jnp.ndarray [nz, ni]
        Absolute error estimate, cm^-3.
    """
    r = _ROS2_GAMMA
    k1, k2, yk2, _ = _ros2_stages(y, k_arr, dt, atm, net, fix_mask, matrix_free)
    sol = y + (3.0 / (2.0 * r)) * k1 + (1.0 / (2.0 * r)) * k2
    delta_arr = jnp.abs(sol - yk2)
    return sol, delta_arr


def make_atm_static(atm: types.SimpleNamespace, ni: int, nz: int, cfg: Config | None = None) -> AtmStatic:
    """Build an AtmStatic from a legacy AtmData container.

    Parameters
    ----------
    atm : types.SimpleNamespace
        Legacy atmosphere container after setup.
    ni : int
        Number of species.
    nz : int
        Number of layers.
    cfg : Config or None, optional
        Transport toggles. None takes the process default; OuterLoop passes its own cfg so the toggles honor a
        load_config() cfg (this runs at integration time, after state._cfg_overlay has restored the default).

    Returns
    -------
    AtmStatic
        `Dzz`, `vm` and `vs` are zero where their toggles are off.
    """
    if cfg is None:
        cfg = default_config()
    use_vm = bool(cfg.use_vm_mol and cfg.use_moldiff)
    use_set = bool(cfg.use_settling and cfg.use_moldiff)
    use_topflux = bool(cfg.use_topflux)
    use_botflux = bool(cfg.use_botflux)
    gas_mask = jnp.zeros((ni,), dtype=jnp.bool_)
    gas_mask = gas_mask.at[jnp.asarray(atm.gas_indx, dtype=jnp.int32)].set(True)
    # Independent of the toggles above (see the diff_esc note at the Jacobian assembly). Nothing checks these names
    # against the network first, so a species not in the network raises ValueError here.
    diff_esc_np = np.zeros((ni,), dtype=bool)
    for _sp in cfg.diff_esc:
        diff_esc_np[_spec_list.index(_sp)] = True
    vm = atm.vm if use_vm else jnp.zeros((nz - 1, ni), dtype=jnp.float64)
    vs = atm.vs if use_set else jnp.zeros((nz - 1, ni), dtype=jnp.float64)
    Dzz = atm.Dzz if cfg.use_moldiff else jnp.zeros((nz - 1, ni), dtype=jnp.float64)
    return AtmStatic(
        Kzz=jnp.asarray(atm.Kzz),
        Dzz=jnp.asarray(Dzz),
        dzi=jnp.asarray(atm.dzi),
        vz=jnp.asarray(atm.vz),
        Hpi=jnp.asarray(atm.Hpi),
        Ti=jnp.asarray(atm.Ti),
        Tco=jnp.asarray(atm.Tco),
        g=jnp.asarray(atm.g),
        ms=jnp.asarray(atm.ms),
        alpha=jnp.asarray(atm.alpha),
        M=jnp.asarray(atm.M),
        vm=jnp.asarray(vm),
        vs=jnp.asarray(vs),
        top_flux=jnp.asarray(atm.top_flux),
        bot_flux=jnp.asarray(atm.bot_flux),
        bot_vdep=jnp.asarray(atm.bot_vdep),
        gas_indx_mask=gas_mask,
        diff_esc_mask=jnp.asarray(diff_esc_np),
        use_vm_mol=use_vm,
        use_settling=use_set,
        use_topflux=use_topflux,
        use_botflux=use_botflux,
    )
