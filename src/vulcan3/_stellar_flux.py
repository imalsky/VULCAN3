"""Read the stellar flux onto the wavelength bins, and the boundary-flux files."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from ._paths import resolve_data_path
from .phy_const import au, r_sun

__all__ = ["read_bc_flux", "read_sflux_binned"]

logger = logging.getLogger(__name__)


# size-limit-ok: one binning pass, in the order the bit-parity tests pin
def read_sflux_binned(
    cfg,
    bins: np.ndarray,
    sflux_raw: np.ndarray | None = None,
) -> dict[str, Any]:
    """Interpolate the stellar flux onto VULCAN's uniform photo bins grid.

    Returns sflux_top and the dbin1 -> dbin2 transition index. `sflux_raw`
    may be supplied for tests.
    """
    if sflux_raw is None:
        sflux_raw = np.genfromtxt(
            resolve_data_path(cfg.sflux_file),
            dtype=float,
            skip_header=1,
            names=["lambda", "flux"],
        )
    bins_np = np.asarray(bins, dtype=np.float64)
    dbin1 = float(cfg.dbin1)
    dbin2 = float(cfg.dbin2)
    dbin_12 = float(cfg.dbin_12trans)
    raw_lambda = np.asarray(sflux_raw["lambda"], dtype=np.float64)
    # Scale the surface flux to the planet's orbit by (R_star / r_orbit)^2.
    geom = (float(cfg.r_star) * r_sun / (au * float(cfg.orbit_radius))) ** 2
    raw_flux = np.asarray(sflux_raw["flux"], dtype=np.float64) * geom
    # np.interp needs non-decreasing wavelengths; duplicates are allowed
    # (sflux-epseri.txt has 20).
    # bin_min comes from wavelength[0], as upstream (op.py:575).
    if (
        raw_lambda.ndim != 1
        or raw_lambda.shape != raw_flux.shape
        or raw_lambda.size < 2
        or not (np.all(np.isfinite(raw_lambda)) and np.all(np.isfinite(raw_flux)))
    ):
        raise ValueError(
            f"stellar flux {cfg.sflux_file}: need equal-length finite 1-D wavelength/flux "
            f"columns with >= 2 rows (got shapes {raw_lambda.shape}, {raw_flux.shape})"
        )
    if np.any(raw_lambda <= 0.0) or np.any(np.diff(raw_lambda) < 0.0) or np.any(raw_flux < 0.0):
        raise ValueError(
            f"stellar flux {cfg.sflux_file}: wavelength must be positive and "
            f"non-decreasing and flux non-negative (min lambda {raw_lambda.min():g}, "
            f"min diff {np.diff(raw_lambda).min():g}, min flux {raw_flux.min():g})"
        )

    # Bins and the star file are independent; warn when bins outside the file
    # are zero-filled.
    if bins_np[0] < raw_lambda[0] or bins_np[-1] > raw_lambda[-1]:
        warnings.warn(
            f"stellar flux {cfg.sflux_file} spans "
            f"[{raw_lambda[0]:g}, {raw_lambda[-1]:g}] nm but the photolysis bins "
            f"span [{bins_np[0]:g}, {bins_np[-1]:g}] nm; bins outside the file are "
            "zero-filled, so photolysis is switched off there.",
            RuntimeWarning,
            stacklevel=2,
        )

    sflux_top = np.asarray(
        jnp.interp(
            jnp.asarray(bins_np),
            jnp.asarray(raw_lambda),
            jnp.asarray(raw_flux),
            left=0.0,
            right=0.0,
        ),
        dtype=np.float64,
    )

    # Upstream (build_atm.py:635, :652) leaves sflux_din12_indx = -1 when the node is
    # absent and compute_J then integrates bins[:-1] at dbin1 (dropping the
    # last bin); refused here instead.
    transition = np.flatnonzero(bins_np == dbin_12)
    if transition.size != 1 or transition[0] == 0 or transition[0] == bins_np.size - 1:
        raise ValueError(
            f"photolysis grid must contain one internal dbin_12trans={dbin_12:g} nm node "
            f"(grid [{bins_np[0]:g}, {bins_np[-1]:g}] nm, {transition.size} matches)"
        )
    sflux_din12_indx = int(transition[0])

    # Energy-conservation diagnostic: raw trapezoidal integral vs. binned sum.
    raw_left_indx = int(np.searchsorted(raw_lambda, bins_np[0], side="right"))
    raw_right_indx = int(np.searchsorted(raw_lambda, bins_np[-1], side="right")) - 1
    sum_orgin = 0.0
    for n in range(raw_left_indx, raw_right_indx):
        sum_orgin += 0.5 * (raw_flux[n] + raw_flux[n + 1]) * (raw_lambda[n + 1] - raw_lambda[n])
    sum_orgin += (
        0.5
        * (float(np.interp(bins_np[0], raw_lambda, raw_flux)) + raw_flux[raw_left_indx])
        * (raw_lambda[raw_left_indx] - bins_np[0])
    )
    sum_orgin += (
        0.5
        * (float(np.interp(bins_np[-1], raw_lambda, raw_flux)) + raw_flux[raw_right_indx])
        * (bins_np[-1] - raw_lambda[raw_right_indx])
    )

    sum_bin = dbin1 * np.sum(sflux_top[:sflux_din12_indx])
    sum_bin -= dbin1 * 0.5 * (sflux_top[0] + sflux_top[sflux_din12_indx - 1])
    sum_bin += dbin2 * np.sum(sflux_top[sflux_din12_indx:])
    sum_bin -= dbin2 * 0.5 * (sflux_top[sflux_din12_indx] + sflux_top[-1])

    logger.info(
        "The stellar flux is interpolated onto uniform grid of %s (<%s nm) and %s (>=%s nm) "
        "and conserving %.2f %% energy.",
        dbin1,
        dbin_12,
        dbin2,
        dbin_12,
        100 * sum_bin / sum_orgin,
    )
    return {
        "sflux_top": sflux_top,
        "sflux_din12_indx": sflux_din12_indx,
    }


def _parse_bc_file(path: str) -> list[list[str]]:
    """Read a BC text file into token rows; skip comments and blank lines."""
    rows: list[list[str]] = []
    with Path(resolve_data_path(path)).open() as f:
        for line in f:
            if not line.startswith("#") and line.strip():
                rows.append(line.split())
    return rows


def _bc_species_index(sp: str, species_list: list[str], path) -> int:
    """Return a boundary-flux row's species index, refusing a species outside the network, as upstream does."""
    if sp not in species_list:
        raise ValueError(f"{path}: boundary-flux row names {sp!r}, which is not in the network; remove the row.")
    return species_list.index(sp)


def read_bc_flux(cfg, species_list: list[str]) -> dict[str, np.ndarray]:
    """Build (top_flux, bot_flux, bot_vdep, bot_fix_sp).

    use_topflux and use_botflux each fill their section from a file. bot_fix_sp
    stays zero: OuterLoop pins the `use_fix_sp_bot` dict entries itself.
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
