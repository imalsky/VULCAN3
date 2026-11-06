"""Legacy I/O surface matching VULCAN-master/op.py's entry points.

`ReadRate` keeps upstream's call signature (`ReadRate().read_rate(var,
atm)`) but is not a parser: `network.parse_network` is the one
network parser, and this copies the host-side metadata the runtime reads
off `var` (`var.Rf`, `var.pho_rate_index`, ...) from it. Rate *values*
come from `rates_jax.build_rate_array`.

`Output` writes the `.vul` pickle with the same public schema upstream
plotting tools read; photo cross-section dicts, the per-reaction `var.k`
dict, and parameter fields are synthesised at pickle time from the typed
`RunState`.
"""

from __future__ import annotations

import logging
import pickle
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from . import chem_funs, photo as _photo_mod
from .chem_funs import ni, nr
from .config import default_config
from .op_jax import Ros2JAX
from .state import TERM_NONFINITE, TERM_RUNTIME, TERM_STEP_COUNT, legacy_view

__all__ = ["Output", "ReadRate", "species"]

logger = logging.getLogger(__name__)

_CFG = default_config()
species = chem_funs.spec_list

# VULCAN-master's Tableau 20 palette (0-255). Published 0-1 normalized in the
# .vul `parameter` dict (`para.tableau20`) as master publishes it
# (store.py:203-209), so the values are a compatibility contract.
_TABLEAU20_RGB255 = [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
    (174, 199, 232),
    (255, 187, 120),
    (152, 223, 138),
    (255, 152, 150),
    (197, 176, 213),
    (196, 156, 148),
    (247, 182, 210),
    (199, 199, 199),
    (219, 219, 141),
    (158, 218, 229),
]


def _warn_stale_reaction_ids(network_path: str, stale: list[tuple[int, int, str]]) -> None:
    """Warn when written photo/ion ids differ from parse positions (the file was not renumbered upstream).

    Rates are indexed by position, but a ``cfg.remove_list`` taken from the id column picks the wrong reactions.
    """
    if not stale:
        return
    shown = ", ".join(f"{txt.strip()!r} written {fid} but at {pos}" for pos, fid, txt in stale[:3])
    more = f" (+{len(stale) - 3} more)" if len(stale) > 3 else ""
    warnings.warn(
        f"Network {network_path} has {len(stale)} photo/ion reaction(s) whose "
        f"written id disagrees with its position: {shown}{more}. The file has not "
        "been renumbered by VULCAN's make_chem_funs.py. Rates are indexed by "
        "position, so this run is correct, but any cfg.remove_list entry taken "
        "from this file's id column will select the WRONG reaction. Renumber the "
        "file, or write remove_list using positions (1-based, forward reactions "
        "odd).",
        RuntimeWarning,
        stacklevel=2,
    )


class ReadRate:
    """Upstream's rate-setup entry point, backed by `network.parse_network`.

    Upstream's ``op.ReadRate`` re-parsed the network file and built rate
    coefficients; both are done elsewhere here (``network.parse_network``
    owns the parse, ``rates_jax.build_rate_array`` owns ``var.k_arr``). What is
    left is publishing the host-side metadata the runtime reads off the
    legacy ``var``.
    """

    def read_rate(self, var: Any, atm: Any) -> Any:
        """Copy the parsed network's host-side metadata onto `var`.

        Sets the attributes something reads: `Rf` (reaction text by
        parser position), the photo/ion branch indices, and
        `conden_re_list`. The Arrhenius columns and the section markers are
        read off the `Network` everywhere, never off `var`, so they are not
        republished here.
        """
        del atm  # upstream signature; the metadata is temperature-free
        net = chem_funs._NETWORK
        var.Rf = dict(net.Rf)
        var.pho_rate_index = dict(net.pho_rate_index)
        var.ion_rate_index = dict(net.ion_rate_index)
        var.n_branch = dict(net.n_branch)
        var.ion_branch = dict(net.ion_branch)
        var.photo_sp = set(net.photo_sp)
        if _CFG.use_ion:
            var.ion_sp = set(net.ion_sp)
        var.conden_re_list = [int(i) for i in np.flatnonzero(net.is_conden & net.is_forward)]
        _warn_stale_reaction_ids(net.network_path, list(net.stale_ids))
        return var


def _synthesize_cross_dicts(static: Any) -> dict:
    """Build the legacy `var.cross*` dict views from a `PhotoStaticInputs`.

    The .vul writer publishes up to these five dicts, the keys the upstream
    `plot_py/` scripts index (`d['variable']['cross'][sp]`,
    `d['variable']['cross_J'][(sp,i)]`, etc.); the dicts are rebuilt from
    the dense pytree at pickle time. Every value is wrapped in
    `np.asarray(..., dtype=np.float64)` so downstream consumers see plain
    ndarrays (not jax arrays).
    """
    cross = {sp: np.asarray(static.absp_cross[i], dtype=np.float64) for i, sp in enumerate(static.absp_sp)}
    cross_T = {sp: np.asarray(static.absp_T_cross[i], dtype=np.float64) for i, sp in enumerate(static.absp_T_sp)}
    cross_J = {k: np.asarray(static.cross_J[i], dtype=np.float64) for i, k in enumerate(static.branch_keys)}
    cross_scat = {sp: np.asarray(static.scat_cross[i], dtype=np.float64) for i, sp in enumerate(static.scat_sp)}
    cross_Jion = {k: np.asarray(static.cross_Jion[i], dtype=np.float64) for i, k in enumerate(static.ion_branch_keys)}
    return {
        "cross": cross,
        "cross_T": cross_T,
        "cross_J": cross_J,
        "cross_scat": cross_scat,
        "cross_Jion": cross_Jion,
    }


def _synthesize_save_dicts(runstate: Any, cfg: Any) -> Any:
    """Build the three .vul top-level dicts from a `RunState`.

    Returns `(variable_dict, atm_dict, parameter_dict)` matching the
    legacy `(vars(data_var) filtered by var_save, vars(atm), vars(para))`
    public shape so `plot_py/` and downstream consumers see the same keys,
    shapes, and dtypes as upstream.
    """
    use_photo = bool(cfg.use_photo)
    use_ion = bool(cfg.use_ion)
    var, atm, para = legacy_view(runstate)

    # 1. Variable dict: upstream's var.var_save filter (store.py:83-88).
    keys = [
        "y",
        "ymix",
        "t",
        "dt",
        "longdy",
        "longdydt",
        "y_ini",
        "atom_ini",
        "atom_sum",
        "atom_loss",
        "atom_conden",
        "Rf",
    ]
    if use_photo:
        keys += ["tau", "sflux", "aflux", "aflux_change", "nbin", "bins", "dbin1", "dbin2", "n_branch"]
    if use_ion:
        keys += ["charge_list", "ion_sp", "ion_branch", "ion_br_ratio"]
    var_save = {"species": species, "nr": nr}
    var_save.update({k: getattr(var, k) for k in keys if hasattr(var, k)})
    var_save["k"] = {i: var.k_arr[i].copy() for i in range(1, var.k_arr.shape[0])}

    static = runstate.photo_static
    if use_photo and static is not None:
        photo_dicts = _synthesize_cross_dicts(static)
        var_save["cross"] = photo_dicts["cross"]
        var_save["cross_scat"] = photo_dicts["cross_scat"]
        var_save["cross_J"] = photo_dicts["cross_J"]
        if list(cfg.T_cross_sp):
            var_save["cross_T"] = photo_dicts["cross_T"]
        if use_ion:
            var_save["cross_Jion"] = photo_dicts["cross_Jion"]
        # Master writes J for every branch, removed ones included (op.py:2764,
        # 2785); an empty rate index makes `_fill_J` leave k_arr alone.
        if hasattr(var, "aflux"):
            var_save["J_sp"] = Ros2JAX._fill_J(
                var, _photo_mod.photo_J_data_from_static(static), var.photo_sp, var.n_branch, {}
            )
            if use_ion:
                var_save["Jion_sp"] = Ros2JAX._fill_J(
                    var, _photo_mod.photo_ion_data_from_static(static), var.ion_sp, var.ion_branch, {}
                )
    if use_ion:
        var_save["ion_wavelen"] = {}

    # Evolution buffer under the legacy y_time / t_time keys.
    if bool(cfg.save_evolution) and runstate.step is not None:
        var_save["y_time"] = np.asarray(runstate.step.y_evo, dtype=np.float64)
        var_save["t_time"] = np.asarray(runstate.step.t_evo, dtype=np.float64)

    # 2-3. `vars(data_atm)` and `vars(data_para)` (store.py:182); solver_str
    # and the plotting-only tableau20 keep the public .vul schema.
    para_save = dict(
        vars(para), solver_str="solver", tableau20=[(r / 255.0, g / 255.0, b / 255.0) for r, g, b in _TABLEAU20_RGB255]
    )
    return var_save, dict(vars(atm)), para_save


class Output:
    """Per-run output: cfg copy, .vul writer, progress report."""

    def __init__(self, cfg: Any = None) -> None:
        """Set up the `.vul` writer for one run.

        Warn if the target file already exists. save_cfg and save_out create their own directories.
        """
        # cfg defaults to the process default; load_config() users pass their
        # namespace so output honors the same cfg as setup and the runner.
        # Pair with OuterLoop(cfg=cfg).
        self._cfg = cfg if cfg is not None else default_config()

        output_dir, out_name = self._cfg.output_dir, self._cfg.out_name
        if Path(output_dir + out_name).is_file():
            warnings.warn("The output file: " + str(out_name) + " already exists.", stacklevel=2)

    def print_prog(self, var: Any, para: Any) -> None:
        """Log one progress block (the config asked for it.

        `use_print_prog`): elapsed model time, step count vs `count_max`, longdy / longdy_dt / dt, and the
        most-varying (level, species).
        """
        indx_max = np.nanargmax(para.where_varies_most)
        logger.info("Elapsed time: %.2e || Step number: %s/%s", var.t, para.count, self._cfg.count_max)
        logger.info("longdy = %.2e      || longdy/dt = %.2e  || dt = %.2e", var.longdy, var.longdydt, var.dt)
        logger.info("from nz = %s and %s", int(indx_max / ni), species[indx_max % ni])
        logger.info("------------------------------------------------------------------------")

    def print_end_msg(self, var: Any, para: Any) -> None:
        """Log the steady-state success summary.

        CPU wall time, step count, final model time, long dy / dy_dt, per-atom loss (skipping `cfg.loss_ex`),
        and the negative/loss/delta rejection counters.
        """
        logger.info("%s s CPU time", "After ------- %s seconds -------" % (time.time() - para.start_time))
        logger.info(
            "%s has successfully run to steady-state with %s steps and %s s",
            self._cfg.out_name[:-4],
            para.count,
            f"{var.t:.2e}",
        )
        logger.info("long dy = %.6e and long dy/dt = %.6e", var.longdy, var.longdydt)
        self._log_losses(var, para)
        logger.info("------ Live long and prosper \\V/ ------")

    def print_unconverged_msg(self, var: Any, para: Any, case: Any) -> None:
        """Log the non-converged summary for termination `case`.

        2 = runtime budget, 3 = max steps, 5 = stopped without converging and without hitting a cap; any other case
        raises RuntimeError. Also logs per-atom loss and rejection counters.
        """
        why = {
            TERM_RUNTIME: f"Maximal allowed runtime exceeded ({self._cfg.runtime:.1e} sec)",
            TERM_STEP_COUNT: f"Maximal allowed steps exceeded ({self._cfg.count_max} steps)",
            TERM_NONFINITE: "Stopped without converging and without hitting a cap "
            f"(termination_reason {para.termination_reason}); "
            "the state may be non-finite",
        }.get(case)
        if why is None:
            raise RuntimeError(f"Unconverged case undefined (case={case})")

        logger.info("%s s CPU time", "After ------- %s seconds -------" % (time.time() - para.start_time))
        logger.warning("%s did not reach steady-state:", self._cfg.out_name[:-4])
        logger.info("long dy = %s and long dy/dt = %s", var.longdy, var.longdydt)
        logger.warning("Integration stopped before converged...\n%s", why)
        self._log_losses(var, para)

    def _log_losses(self, var: Any, para: Any) -> None:
        """Log per-atom loss (skipping `cfg.loss_ex`) and the negative/loss/delta rejection counters."""
        logger.info("total atom loss:")
        for atom in self._cfg.atom_list:
            if atom not in self._cfg.loss_ex:
                logger.info("%s: %.4e ", atom, var.atom_loss[atom])
        logger.info("negative solution counter:")
        logger.info(str(para.nega_count))
        logger.info("loss rejected counter:")
        logger.info(str(para.loss_count))
        logger.info("delta rejected counter:")
        logger.info(str(para.delta_count))

    def save_cfg(self, dname: str) -> None:
        """Write a repr snapshot of the active cfg, make_config overrides included.

        Goes to `cfg_<out_name stem>.txt` under `dname/output_dir`, skipping private, callable, type and module
        attributes so it re-reads as Python.
        """
        # Create the directory where the file is written, not under the cwd.
        output_dir, out_name = self._cfg.output_dir, self._cfg.out_name
        target_dir = Path(dname) / output_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        lines = []
        for key in sorted(vars(self._cfg)):
            if key.startswith("_"):
                continue
            val = getattr(self._cfg, key)
            if callable(val) or isinstance(val, type) or hasattr(val, "__loader__"):
                continue
            lines.append(f"{key} = {val!r}")
        out_path = target_dir / ("cfg_" + out_name[:-3] + "txt")
        with out_path.open("w") as f:
            f.write("\n".join(lines) + "\n")

    def save_out(self, runstate: Any, dname: str) -> None:
        """Write the `.vul` pickle output; reads everything from `runstate`."""
        var_save, atm_save, para_save = _synthesize_save_dicts(runstate, self._cfg)

        output_dir, out_name = self._cfg.output_dir, self._cfg.out_name
        target_dir = Path(dname) / output_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        output_file = target_dir / out_name

        with output_file.open("wb") as outfile:
            pickle.dump(
                {"variable": var_save, "atm": atm_save, "parameter": para_save},
                outfile,
                protocol=4,
            )

    # Post-run plotters (master's Output.plot_end / plot_evo, restyled): PNGs
    # in `plot_dir`. matplotlib is the optional `plot` extra, imported here so
    # a run without plots never needs it.

    def plot_end(self, rs: Any) -> None:
        """Plot final mixing ratios of `cfg.plot_spec` against pressure, saved to `plot_dir/mix.png`."""
        p_bar = np.asarray(rs.atm.pco) / 1.0e6
        ymix = np.asarray(rs.step.ymix)
        self._plot_species(
            lambda i: (ymix[:, i], p_bar),
            "mix.png",
            "mixing ratio",
            "pressure (bar)",
            mix_axis=0,
            other_lim=(p_bar.max(), p_bar.min()),
        )

    def plot_evo(self, rs: Any) -> None:
        """Mixing ratios of `cfg.plot_spec` against model time in the layer nearest `_EVO_P_BAR`.

        From the `save_evo_frq` snapshots, saved to `plot_dir/evo.png`.
        """
        t_evo = np.asarray(rs.step.t_evo)
        if t_evo.size == 0:
            raise ValueError(
                "plot_evo needs the evolution snapshots; set save_evolution or use_plot_evo before the run."
            )
        p_bar = np.asarray(rs.atm.pco) / 1.0e6
        j = int(np.argmin(np.abs(np.log(p_bar / _EVO_P_BAR))))
        ymix = np.asarray(rs.step.y_evo)[:, j, :] / float(np.asarray(rs.atm.n_0)[j])
        self._plot_species(
            lambda i: (t_evo, ymix[:, i]),
            "evo.png",
            "time (s)",
            f"mixing ratio at {p_bar[j]:.1e} bar",
            mix_axis=1,
            other_lim=(t_evo.min(), t_evo.max()),
        )

    def _plot_species(self, xy: Any, name: str, xlabel: Any, ylabel: Any, mix_axis: Any, other_lim: Any) -> None:
        """One log-log line per `cfg.plot_spec` species, `xy(index) -> (x, y)`.

        The mixing-ratio axis (`mix_axis` 0 = x, 1 = y) spans the plotted values, floored at `_PLOT_YMIN`.
        """
        plt = _import_plt()
        curves = [(sp, xy(species.index(sp))) for sp in self._cfg.plot_spec]
        mix = np.concatenate([c[mix_axis] for _, c in curves])
        mix = mix[mix > _PLOT_YMIN]
        mix_lim = (mix.min(), mix.max()) if mix.size else (_PLOT_YMIN, 1.0)
        with plt.rc_context(_RC):
            fig, ax = plt.subplots(figsize=(6.5, 6.5))
            for k, (sp, c) in enumerate(curves):
                ax.plot(*c, label=sp, color=_CYCLE[k % len(_CYCLE)], ls=_STYLES[k // len(_CYCLE) % len(_STYLES)])
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_box_aspect(1)
            ax.set_xlim(*(mix_lim if mix_axis == 0 else other_lim))
            ax.set_ylim(*(other_lim if mix_axis == 0 else mix_lim))
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.legend(frameon=False, fontsize=10, loc="best")
            Path(self._cfg.plot_dir).mkdir(parents=True, exist_ok=True)
            path = Path(self._cfg.plot_dir) / name
            fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Plot saved to %s", path)


# Okabe-Ito colours, then matplotlib's own dash and dot for more species.
_CYCLE = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9"]
_STYLES = ["-", "--", ":"]
# Mixing-ratio axis floor (master's plot_end xlim).
_PLOT_YMIN = 1.0e-20
# plot_evo's layer: the one nearest 1 mbar, where transmission spectra form.
_EVO_P_BAR = 1.0e-3
# Serif text, ticks inward on all four sides; applied to these figures only.
_RC = {
    "font.family": "serif",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "axes.grid": False,
}


def _import_plt() -> Any:
    """Matplotlib's pyplot, or a pointer to the `plot` extra."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("use_plot_end / use_plot_evo need matplotlib: pip install 'vulcan3[plot]'") from exc
    return plt
