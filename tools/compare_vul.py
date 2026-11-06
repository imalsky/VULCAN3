"""Compare two converged VULCAN outputs (.vul pickles): per species.

The worst |dlog10 VMR| over the cells where both are above a floor, and the pressure it sits at. The end-to-end check of
a solver path (the CUDA kernel, the grouped RHS, a GPU) against VULCAN 2 on matched inputs, or against another vulcan3
path; the CPU port's own agreement with VULCAN 2 on the same case is the envelope to read it against.

    python tools/compare_vul.py REF.vul TEST.vul [--species H2O,CO,CO2,CH4,SO2,H2S] [--floor 1e-12]

Both files must share the pressure grid. Exit 1 when any species exceeds --max-dex.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

_PRESSURE_RTOL = 1e-12  # the same grid written by two runs, so only roundoff apart


def _load(path):
    with Path(path).open("rb") as f:
        d = pickle.load(f)
    return (np.asarray(d["atm"]["pco"]) / 1e6, list(d["variable"]["species"]), np.asarray(d["variable"]["ymix"]))


def main(argv=None) -> int:
    """Run the comparison; return 1 when a species exceeds --max-dex."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref")
    ap.add_argument("test")
    ap.add_argument("--species", default="H2O,CO,CO2,CH4,SO2,H2S")
    ap.add_argument("--floor", type=float, default=1e-12)
    ap.add_argument("--max-dex", type=float, default=float("inf"))
    args = ap.parse_args(argv)
    p, sp_r, y_r = _load(args.ref)
    p_t, sp_t, y_t = _load(args.test)
    if p.shape != p_t.shape or not np.allclose(p, p_t, rtol=_PRESSURE_RTOL, atol=0.0):
        raise SystemExit(f"the two outputs are on different pressure grids: {p} bar against {p_t} bar")
    worst = 0.0
    print(f"{'species':8s} {'max|dlog10|':>12s} {'at bar':>10s} {'cells':>6s}")
    for s in args.species.split(","):
        if s not in sp_r or s not in sp_t:
            print(f"{s:8s} {'not in both networks':>30s}")
            continue
        a, b = y_r[:, sp_r.index(s)], y_t[:, sp_t.index(s)]
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            raise SystemExit(
                f"{s}: non-finite mixing ratio, {np.isfinite(a).sum()} and {np.isfinite(b).sum()} finite cells"
            )
        m = (a > args.floor) & (b > args.floor)
        if not m.any():
            raise SystemExit(f"{s}: no cell above --floor {args.floor} in both outputs, so nothing to compare")
        d = np.where(m, np.abs(np.log10(np.where(m, a, 1.0)) - np.log10(np.where(m, b, 1.0))), 0.0)
        k = int(np.argmax(d))
        worst = max(worst, float(d[k]))
        print(f"{s:8s} {d[k]:12.3g} {p[k]:10.3g} {int(m.sum()):6d}")
    print(f"worst {worst:.3g} dex")
    return int(worst > args.max_dex)


if __name__ == "__main__":
    sys.exit(main())
