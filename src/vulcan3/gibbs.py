"""NASA-9 polynomial thermo table: file I/O and the shared constants.

The Gibbs energy and reverse-rate math itself lives in `rates_jax.py` (one
implementation, on the AD graph); this module owns the reading of
`thermo/NASA9/<sp>.txt` and the constants both sides pin.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import numpy as np

from .phy_const import BAR_CGS, kb

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["CORR", "load_nasa9"]

# Valid temperature range of the NASA-9 Gibbs fits (build_atm.py:618).
_GIBBS_T_MIN_K = 200.0
_GIBBS_T_MAX_K = 6000.0

# `CORR * T` is the (kB T / P0) factor K_eq carries per unit change in mole
# number, P0 = 1 bar the standard-state pressure.
CORR = kb / BAR_CGS

# NASA-9 polynomials are fit in two temperature segments; the low-T coefficient
# set applies below this breakpoint and the high-T set above it (NASA/TP-2002-211556).
_NASA9_BRANCH_T = 1000.0
# Coefficients per temperature segment in a thermo/NASA9/<sp>.txt file.
_NASA9_NCOEF = 10


@functools.cache
def load_nasa9(species: tuple[str, ...], thermo_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load NASA-9 polynomial coefficients per species.

    Cached per (species, directory): the rate build and the equilibrium seed
    ask for the same table, and one call reads one file per species. The
    arrays are shared between callers -- read them, never write into them.

    Parameters
    ----------
    species : tuple of str [ni]
        Species names.
    thermo_dir : str or Path
        Directory holding ``NASA9/<sp>.txt``, resolved by `resolve_data_path`.

    Returns
    -------
    coeffs : np.ndarray [ni, 2, 10]
        Coefficients; index 0 of axis 1 is low-T (T < 1000 K), index 1 high-T. Zero for a species without a file.
    present : np.ndarray [ni]
        bool, True where a file was read.

    Raises
    ------
    ValueError
        If a file has fewer than 20 entries.
    """
    from ._paths import resolve_data_path

    thermo_dir = resolve_data_path(str(thermo_dir))
    ni = len(species)
    coeffs = np.zeros((ni, 2, _NASA9_NCOEF), dtype=np.float64)
    present = np.zeros(ni, dtype=bool)
    for j, sp in enumerate(species):
        fp = thermo_dir / "NASA9" / f"{sp}.txt"
        if not fp.exists():
            continue
        flat = np.loadtxt(fp).flatten()
        if flat.size < 2 * _NASA9_NCOEF:
            raise ValueError(f"NASA-9 file {fp} has only {flat.size} entries; need at least {2 * _NASA9_NCOEF}")
        coeffs[j, 0] = flat[:_NASA9_NCOEF]
        coeffs[j, 1] = flat[_NASA9_NCOEF : 2 * _NASA9_NCOEF]
        present[j] = True
    return coeffs, present
