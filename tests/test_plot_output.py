"""The VULCAN 2 end-of-run plots (`use_plot_end`, `use_plot_evo`) write their figures from a run's state."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


def test_plot_end_and_evo_write_their_figures(tmp_path):
    pytest.importorskip("matplotlib", reason="the plots need the `plot` extra")
    from vulcan3 import make_config
    from vulcan3.legacy_io import Output, species

    nz, n_snap = 12, 5
    cfg = make_config(plot_dir=f"{tmp_path}/plot/", output_dir=f"{tmp_path}/out/")
    ymix = np.logspace(-8, -3, nz)[:, None] * np.ones(len(species))
    n_0 = np.full(nz, 1.0e18)
    rs = SimpleNamespace(
        atm=SimpleNamespace(pco=np.logspace(9, 1, nz), n_0=n_0),  # 1e3 to 1e-5 bar, dyne/cm^2
        step=SimpleNamespace(
            ymix=ymix,
            t_evo=np.logspace(0, 4, n_snap),
            y_evo=np.logspace(0, 2, n_snap)[:, None, None] * (ymix * n_0[:, None]),  # each species grows in time
        ),
    )
    out = Output(cfg=cfg)
    out.plot_end(rs)
    out.plot_evo(rs)
    for name in ("mix.png", "evo.png"):
        assert (tmp_path / "plot" / name).stat().st_size > 0
