"""YAML-backed configuration for VULCAN3.

The config surface is authored as YAML. Canonical configs ship inside the
package (``vulcan3/configs/*.yaml``) and are overridable by a ``./configs/``
directory in the current working directory (CWD-first), so a run can be tuned
locally without editing packaged files.

``load_config(name_or_path, **overrides)`` returns a ``Config`` read over the
packaged ``default.yaml``; ``default_config()`` is the cached process-wide
default, read at the first ``import vulcan3``.
"""

from __future__ import annotations

import copy
import functools
import re
import types
import warnings
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from . import _settings

if TYPE_CHECKING:
    import os
    from collections.abc import Callable

__all__ = [
    "DT_MAX_S",
    "REPAIR_ABS_FLOOR",
    "Config",
    "default_config",
    "dump_config",
    "load_config",
    "validate_overrides",
]


class _StrictLoader(yaml.SafeLoader):
    """Strict YAML loader for config files.

    Two departures from ``yaml.safe_load``, both to fail loudly instead of
    silently doing the wrong thing:

    1. Scientific notation with no dot or no exponent sign (``1e22``,
       ``5.0e21``, ``1e-14``)
       parses as a float. Stock PyYAML (YAML 1.1 core schema) treats these as
       *strings*, which would silently poison numeric knobs a user hand-authors
       in a ``./configs`` override.
    2. Duplicate keys raise instead of last-wins.
    """

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        seen: set = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise yaml.constructor.ConstructorError(
                    None, None, f"duplicate config key {key!r}", key_node.start_mark
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


# Float regex extended with the `1e22` (no dot / unsigned exponent) form.
_FLOAT_RE = re.compile(
    r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?\.[0-9][0-9_]+(?:[eE][-+]?[0-9]+)?
        |[-+]?[0-9][0-9_]*[eE][-+]?[0-9]+
        |[-+]?\.(?:inf|Inf|INF)
        |\.(?:nan|NaN|NAN))$""",
    re.VERBOSE,
)
_StrictLoader.add_implicit_resolver("tag:yaml.org,2002:float", _FLOAT_RE, list("-+0123456789."))

# Import-frozen knobs (read at the first `import vulcan3`): each honors an
# environment override so a non-default network / atom set / composition can be
# selected before import. `com_file` is overridable for symmetry.
_FROZEN_ENV = {
    "network": "VULCAN3_NETWORK",
    "atom_list": "VULCAN3_ATOM_LIST",
    "com_file": "VULCAN3_COM_FILE",
}

_settings.refuse_old_prefix()

# Largest step the Ros2 stage repair (jax_step._repair_stage) can resolve;
# its correction system is as singular as the stage system by 1e17 s.
DT_MAX_S = 1.0e15
# Smallest stage element defect the repair corrects, as a fraction of the
# layer number density (times c0); real leaks sit >= 1e-7.
REPAIR_ABS_FLOOR = 1.0e-11
# dt_max = runtime * 1e-5 (vulcan_cfg.py:136); the photo-frequency switch
# threshold is 10 * yconv_min (op.py:819).
_DT_MAX_RUNTIME_FRAC = 1e-5
_PHOTO_SWITCH_LONGDY_FACTOR = 10.0
# Values that are functions of other knobs. Authored YAML omits them; an
# explicit YAML/override value always wins (they are only filled when absent).
_DERIVED: tuple[tuple[str, Callable[[dict[str, Any]], Any]], ...] = (
    ("dt_max", lambda d: min(d["runtime"] * _DT_MAX_RUNTIME_FRAC, DT_MAX_S)),
    ("photo_switch_longdy_thresh", lambda d: d["yconv_min"] * _PHOTO_SWITCH_LONGDY_FACTOR),
    ("para_anaTP", lambda d: copy.deepcopy(d["para_warm"])),
)

# Retired keys: refused with a remedy instead of merged as inert attributes
# (tools/audit_master_parity.py reads the names).
_REMOVED_KEYS: dict[str, str] = {
    "gs": (
        "gravity is derived as G*Mp/Rp**2; set `Mp` (planet mass, g) and `Rp` (planet radius, cm) and remove the key"
    ),
    "fix_species_time": ("the fix_species pin starts at `stop_conden_time`; set that and remove the key"),
    "use_print_delta": ("no per-step delta print; use `use_print_prog` and remove the key"),
    "fastchem_met_scale": (
        "the equilibrium seed carries only the network's own elements; set "
        "`<X>_H` for the elements of `atom_list` or pick another "
        "`fastchem_solar_abundance_file`, and remove the key"
    ),
    **dict.fromkeys(
        ("use_conv_stall", "conv_stall_window"),
        "a run ends on the convergence certificate or a cap; remove the key",
    ),
    "report_column_atom_loss": (
        "the element-budget term (`element_budget_tol`) checks the "
        "operator-weighted column on every run and `ini_abun.column_atoms` "
        "computes it; remove the key"
    ),
    **dict.fromkeys(
        ("use_pi_controller", "pi_controller_alpha", "pi_controller_beta"),
        "the master-faithful I-controller is the only dt control; remove the key",
    ),
    "use_allFC_ele": ("the equilibrium seed carries only the network's own elements; remove the key"),
    "wall_clock_max": ("a run ends on the convergence certificate, `count_max` or `runtime`; remove the key"),
    "use_chunked_runner": (
        "the CLI always runs in chunks of `print_prog_num` steps; from Python call `integ(rs, chunk=N)`; remove the key"
    ),
    "ode_solver": "Ros2 is the only solver; remove the key",
    "use_fix_all_bot": (
        "VULCAN 2 never selects the all-species bottom pin (op.py:3080-3088); "
        "pin species with `use_fix_sp_bot` and remove the key"
    ),
    "gibbs_text": ("reverse rates read thermo/NASA9/<species>.txt (gibbs.py); remove the key"),
}


# Master's display-only keys (live plot, movies, text output). They change
# nothing in a run, so a master-style config warns and drops them instead of
# failing.
_IGNORED_KEYS = frozenset(
    (
        "movie_dir",
        "plot_TP",
        "use_live_plot",
        "use_live_flux",
        "use_save_movie",
        "use_flux_movie",
        "plot_height",
        "use_PIL",
        "live_plot_frq",
        "save_movie_rate",
        "y_time_freq",
        "output_humanread",
        "use_shark",
    )
)


# `<X>_H` elemental ratios: default.yaml declares the shipped elements; a
# network carrying another element (P, Na, ...) reads its own `<X>_H`.
# runtime_validation refuses a symbol the composition table does not have.
_ELEMENT_RATIO_KEY = re.compile(r"[A-Z][a-z]?_H")


@functools.cache
def _known_config_keys() -> frozenset[str]:
    """Canonical set of accepted config keys.

    Sourced from the *packaged* ``default.yaml`` (the maintained superset, since
    every shipped config is a subset of its keys) plus the derived keys that
    authored YAML omits but a resolved/dumped config carries. Read raw (not via
    ``load_config``) so building the schema cannot recurse through validation.
    """
    return frozenset(_packaged_defaults()) | {name for name, _ in _DERIVED}


@functools.cache
def _packaged_defaults() -> dict[str, Any]:
    """Return the packaged ``default.yaml``, read raw once: the value of every knob a config file omits."""
    pkg = resources.files("vulcan3").joinpath("configs", "default.yaml")
    defaults: dict[str, Any] = yaml.load(pkg.read_text(), Loader=_StrictLoader)
    return defaults


def _validate_keys(d: dict[str, Any], source: str) -> list[str]:
    """Reject removed or unknown config keys before they silently do nothing.

    Warn about display-only keys and return them for the caller to drop.
    """
    removed = sorted(k for k in d if k in _REMOVED_KEYS)
    if removed:
        detail = "; ".join(f"`{k}` -- {_REMOVED_KEYS[k]}" for k in removed)
        raise ValueError(f"{source}: uses removed config key(s): {detail}.")
    ignored = sorted(k for k in d if k in _IGNORED_KEYS)
    if ignored:
        warnings.warn(
            f"{source}: ignoring display-only key(s) {ignored}; VULCAN3 has no live plot, movie or text output.",
            stacklevel=3,
        )
    unknown = sorted(k for k in set(d) - _known_config_keys() - _IGNORED_KEYS if not _ELEMENT_RATIO_KEY.fullmatch(k))
    if unknown:
        raise ValueError(
            f"{source}: unknown config key(s) {unknown}. Every knob is declared in "
            "vulcan3/configs/default.yaml -- check for a typo or a renamed knob."
        )
    return ignored


def validate_overrides(overrides: dict[str, Any]) -> None:
    """Refuse removed or unknown keys in an override dict, as ``load_config(**overrides)`` does.

    For a caller that applies the dict to a loaded Config with ``setattr`` (which checks nothing). Display-only keys
    are refused here: setattr would apply them, and code has no reason to pass one.
    """
    ignored = sorted(set(overrides) & _IGNORED_KEYS)
    if ignored:
        raise ValueError(f"config overrides: display-only key(s) {ignored} do nothing in VULCAN3; remove them.")
    _validate_keys(overrides, "config overrides")


class Config(types.SimpleNamespace):
    """Free-attribute config namespace (attribute access like master's ``vulcan_cfg``)."""


def _read_text(name_or_path: str | os.PathLike) -> str:
    """Resolve a config reference to YAML text.

    Resolution order: an explicit path or ``*.yaml``/``*.yml`` file; else a bare
    name resolved CWD-first (``./configs/<name>.yaml``) then from the packaged
    ``vulcan3/configs/<name>.yaml``.
    """
    p = Path(name_or_path)
    if p.suffix in (".yaml", ".yml") or p.exists():
        if not p.is_file():
            raise FileNotFoundError(f"config file not found: {name_or_path}")
        return p.read_text()

    fname = f"{name_or_path}.yaml"
    cwd = Path.cwd() / "configs" / fname
    if cwd.is_file():
        return cwd.read_text()

    pkg = resources.files("vulcan3").joinpath("configs", fname)
    if pkg.is_file():
        return pkg.read_text()

    raise FileNotFoundError(
        f"no config named {name_or_path!r}: looked for ./configs/{fname} and the packaged vulcan3/configs/{fname}."
    )


def _apply_frozen_env(d: dict[str, Any]) -> None:
    """Overlay the import-frozen knobs from ``$VULCAN3_*`` (env wins)."""
    for key, env in _FROZEN_ENV.items():
        val = _settings.frozen_override(env)
        if val is None:
            continue
        if key == "atom_list":
            d[key] = [s for s in val.split(",") if s]
        else:
            d[key] = val


def _resolve_derived(d: dict[str, Any]) -> None:
    """Fill derived values (only when absent) and normalize a couple of types."""
    for key, fn in _DERIVED:
        if d.get(key) is None:
            d[key] = fn(d)
    if d.get("count_max") is not None:
        # Accept scientific-notation literals (e.g. 1e4) as an integer step cap.
        d["count_max"] = int(d["count_max"])


def load_config(name_or_path: str | os.PathLike = "default", /, **overrides: Any) -> Config:
    """Load a YAML config, apply env + caller overrides, resolve derived values.

    Parameters
    ----------
    name_or_path : str or os.PathLike
        A bare name (``"default"``, ``"W39b"``; resolved CWD-first then
        packaged) or an explicit path to a ``.yaml`` file.
    **overrides : Any
        Attribute overrides applied after the YAML (and after the frozen-env
        fold), before derived resolution.

    Returns
    -------
    Config
        The full runtime attribute surface.
    """
    raw = yaml.load(_read_text(name_or_path), Loader=_StrictLoader)
    if not isinstance(raw, dict):
        raise ValueError(  # noqa: TRY004 (a bad file, not a bad argument)
            f"config {name_or_path!r} did not parse to a mapping."
        )
    # A config file overlays default.yaml: a knob it omits takes that value.
    d: dict[str, Any] = {**copy.deepcopy(_packaged_defaults()), **raw}
    _apply_frozen_env(d)
    d.update(overrides)
    for key in _validate_keys(d, f"config {name_or_path!r}"):
        del d[key]
    _resolve_derived(d)
    return Config(**d)


def dump_config(cfg: Config, path: str | os.PathLike) -> None:
    """Write a resolved Config back to YAML so a run can be reproduced exactly.

    Dumps the full attribute set (including the derived values), so re-loading
    the file with ``load_config`` reconstructs the same run inputs verbatim.
    """
    d = {k: v for k, v in vars(cfg).items() if not k.startswith("_") and not callable(v)}
    Path(path).write_text(yaml.safe_dump(d, sort_keys=False, default_flow_style=False, width=100))


@functools.cache
def default_config() -> Config:
    """Return the process-wide default Config (loaded once from ``configs/default.yaml``).

    Resolves the import-frozen knobs, folding any ``$VULCAN3_*`` overrides,
    at the first ``import vulcan3``. Cached: the
    same object is returned every call, so ``state._cfg_overlay`` can identity-
    compare against it for its no-op fast path.
    """
    return load_config("default")
