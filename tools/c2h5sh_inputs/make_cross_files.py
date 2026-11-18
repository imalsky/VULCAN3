"""Build scratch cross-section inputs for C2H5SH from two MPI-Mainz atlas tables, in the layout of CH3SCH3's files.

Assumptions, none of them checked against upstream's own (unseen) file:
  - photodissociation = photoabsorption (quantum yield 1) and no photoionisation, as the packaged CH3SCH3 file has;
  - Clark & Simpson (1965) below the first Bridges (1972) point, Bridges from there to 270 nm;
  - one branch, C2H5SH -> C2H6 + S, the only photolysis row in the collaborator's network;
  - threshold 270 nm, the value in upstream's thresholds.txt.
Usage: python make_cross_files.py <atlas folder> <out folder>
"""

import sys
from pathlib import Path

import numpy as np

atlas, out = Path(sys.argv[1]), Path(sys.argv[2])
bridges = np.loadtxt(atlas / "C2H5SH_Bridges(1972)_298K_185-270nm.txt")
clark = np.loadtxt(atlas / "C2H5SH_ClarkSimpson(1965)_298K_162.0-249.0nm.txt")
for name, table in (("Bridges", bridges), ("Clark & Simpson", clark)):
    if not (np.all(np.diff(table[:, 0]) > 0) and np.all(table[:, 1] > 0)):
        raise ValueError(f"{name}: wavelengths must increase and cross sections be positive")
grid = np.array([190.0, 200.0, 210.0, 220.0, 230.0, 240.0, 248.0])
ratio = np.interp(grid, clark[:, 0], clark[:, 1]) / np.interp(grid, bridges[:, 0], bridges[:, 1])
print("Clark & Simpson / Bridges in the overlap:", dict(zip(grid.tolist(), np.round(ratio, 2).tolist(), strict=True)))
merged = np.vstack([clark[clark[:, 0] < bridges[0, 0]], bridges])
(out / "C2H5SH").mkdir(parents=True, exist_ok=True)
lines = ["# wavelength (nm) photoabsorption  photodissociation  photoionisation (cm^2)"]
lines += [f"{lam:.6e},    {sig:.6e},    {sig:.6e},    0." for lam, sig in merged]
(out / "C2H5SH" / "C2H5SH_cross.csv").write_text("\n".join(lines) + "\n")
branch = [
    "# Branching ratios for C2H5SH -> (1)C2H6 + S",
    "# lambda , br_ratio_1",
    f"{merged[0, 0]:.6e},    1.0",
    f"{merged[-1, 0]:.6e},    1.0",
]
(out / "C2H5SH" / "C2H5SH_branch.csv").write_text("\n".join(branch) + "\n")
print(
    f"wrote {len(merged)} points, {merged[0, 0]} to {merged[-1, 0]} nm; "
    f"{int((clark[:, 0] < bridges[0, 0]).sum())} from Clark & Simpson, {len(bridges)} from Bridges"
)
