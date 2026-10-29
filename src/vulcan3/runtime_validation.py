"""Pre-run configuration validation for the standalone VULCAN3 runtime."""

from __future__ import annotations

import importlib.util
import math

import numpy as np

from . import chem_funs
from ._paths import resolve_data_path
from .config import _ELEMENT_RATIO_KEY


def _validate_abundance_preset(cfg) -> list[str]:
    """The EQ seed's abundance preset must parse and cover the network.

    The file is read as `SYMBOL log10(n_X/n_H)+12` rows. The seed needs a
    finite, positive ratio for hydrogen and for every element the loaded
    network contains; a truncated or hand-mangled file otherwise drops an
    element silently (its species then cannot form at all).
    """
    if getattr(cfg, "ini_mix", None) != "EQ":
        return []
    from . import ini_abun

    rel = getattr(cfg, "fastchem_solar_abundance_file", None)
    if not rel or not resolve_data_path(rel).exists():
        return []  # unset or missing is reported by the caller
    try:
        ratios = ini_abun.read_abundances(resolve_data_path(rel))
    except (OSError, ValueError) as exc:
        return [f"abundance preset {rel!r} does not parse: {exc}"]
    bad = [
        sp
        for sp in ini_abun.seed_elements()
        if not math.isfinite(ratios.get(sp, math.nan)) or ratios.get(sp, 0.0) <= 0.0
    ]
    if bad:
        message = (
            f"abundance preset {rel!r} has no finite, positive abundance for "
            f"{', '.join(bad)}, which the {getattr(cfg, 'network', '?')!r} "
            "network contains. Every element of the loaded network must have "
            "a row in the preset."
        )
        return [message]
    return []


def _validate_network_assets(cfg) -> list[str]:
    """Return a list of human-readable errors for missing network-linked assets.

    Checks what produces cryptic downstream errors if wrong (a network species
    missing from the composition table already fails `import
    vulcan3.composition`):
    1. Every photodissociation species has a cross-section file
       (`<cross_folder><sp>/<sp>_cross.csv`).
    2. cfg.atom_list entries are recognised column headers in all_compose.txt.
    3. Every atom_list element outside loss_ex is carried by some network
       species: otherwise its initial column is 0 and the per-step atom_loss
       is 0/0 = NaN, which rejects every step down to dt_min and the run never
       converges (loss_ex elements are left out of that check).
    """
    errors: list[str] = []

    com_path = resolve_data_path(getattr(cfg, "com_file", ""))
    if not com_path.exists():
        return errors  # file-existence error already captured by caller

    # Read species column from composition table.
    with open(com_path) as f:
        header = f.readline().split()
        rows = {r[0]: r[1:] for r in (line.split() for line in f) if r}
    compose_atoms = set(header[1:])  # every column after 'species' (elements, 'e', 'mass')
    net_name = getattr(cfg, "network", "")

    # 1. Photo species need a cross-section file.
    if bool(getattr(cfg, "use_photo", False)):
        cross_folder = resolve_data_path(getattr(cfg, "cross_folder", ""))
        for sp in chem_funs._NETWORK.photo_sp:
            cross_csv = cross_folder / sp / f"{sp}_cross.csv"
            if not cross_csv.exists():
                errors.append(
                    f"Photo species {sp!r} has no cross-section file at "
                    f"{cfg.cross_folder}{sp}/{sp}_cross.csv. "
                    f"Add the file or remove {sp!r} from photo reactions."
                )

    # An `<X>_H` key for a symbol the table lacks is a typo (config.py
    # accepts any `<X>_H` so a new network's element can carry one).
    for key in vars(cfg):
        if _ELEMENT_RATIO_KEY.fullmatch(key) and key[:-2] not in compose_atoms:
            errors.append(f"{key!r}: {key[:-2]!r} is not an element of {cfg.com_file}.")

    # 2. Refuse atom_list entries not in the composition table header.
    for atom in getattr(cfg, "atom_list", []):
        if atom not in compose_atoms:
            errors.append(
                f"atom_list entry {atom!r} is not a column in {cfg.com_file} "
                f"(available: {sorted(compose_atoms)}). "
                f"Update atom_list or add a column to the composition table."
            )
            continue
        # 3. ... and must occur in at least one network species.
        if atom in getattr(cfg, "loss_ex", []):
            continue
        col = header.index(atom) - 1
        carried = any(
            float(rows[sp][col]) > 0 for sp in chem_funs.spec_list if sp in rows
        )
        # Without use_solar the seed reads `<X>_H` for every carried element
        # (ini_abun._base_vector).
        if (carried and atom != "H" and getattr(cfg, "use_solar", True) is not True
                and not hasattr(cfg, f"{atom}_H")):
            errors.append(f"use_solar is off and {atom!r} has no `{atom}_H` "
                          f"(n_{atom}/n_H) in the config.")
        if not carried:
            errors.append(
                f"atom_list entry {atom!r} has no species in network {net_name!r}, "
                "so its initial column is zero and every step fails the "
                f"atom-conservation check. Remove {atom!r} from atom_list or "
                f"use a network with {atom!r} species."
            )

    return errors


# Knobs `_validate_numerical_bounds` checks.
_FINITE_POSITIVE = ("Mp", "Rp", "rtol", "geom_conv_tol", "element_budget_tol")
_COUNTERS = ("count_min", "count_max", "update_frq", "conv_step",
             "batch_max_retries", "ini_update_photo_frq", "final_update_photo_frq")


def _validate_numerical_bounds(cfg) -> list[str]:
    """Errors for knob values that would run to a silently wrong answer.

    A value that raises later or ends in a visibly failed run (no
    convergence, a non-finite state) is left to do so.
    """
    errors: list[str] = []
    nz = int(cfg.nz)
    if nz < 3:
        errors.append(
            f"nz={nz} must be >= 3: with nz <= 2 the interface grid is "
            "degenerate and the run completes on a meaningless column."
        )
    P_b, P_t = float(cfg.P_b), float(cfg.P_t)
    if not P_b > P_t > 0.0:
        errors.append(
            f"P_b={P_b} and P_t={P_t} (bottom and top pressure, dyne/cm^2) "
            "must be P_b > P_t > 0: inverted, the column has negative "
            "thickness and still runs."
        )
    # A NaN fails no `<= 0` test; gravity is G*Mp/Rp**2.
    for key in _FINITE_POSITIVE:
        v = float(getattr(cfg, key))
        # element_budget_tol .inf switches the budget term off (open columns).
        off = key == "element_budget_tol" and v == math.inf
        if not ((math.isfinite(v) or off) and v > 0.0):
            errors.append(f"{key}={v} must be finite and > 0.")
    dt_min, dt_max = float(cfg.dt_min), float(cfg.dt_max)
    if not dt_min < dt_max:
        errors.append(f"dt_min={dt_min:g} must be < dt_max={dt_max:g}.")
    # int32 in the runner; the cadences are modulo divisors there (x % 0
    # does not raise in XLA), and batch_max_retries 0 force-accepts every
    # rejected step.
    for key in _COUNTERS:
        v = float(getattr(cfg, key))
        if not (math.isfinite(v) and v.is_integer() and 1 <= v < 2**31):
            errors.append(f"{key}={v:g} must be an integer in [1, 2**31).")
    # A unit slip (degrees, percent) runs on to a plausible wrong answer.
    sl_angle = float(cfg.sl_angle)
    if not 0.0 <= sl_angle < math.pi / 2.0:
        errors.append(f"sl_angle={sl_angle:g} must be in [0, pi/2) radians.")
    humidity = float(cfg.humidity)
    if not 0.0 <= humidity <= 1.0:
        errors.append(f"humidity={humidity:g} must be a fraction in [0, 1].")
    # The element budget is measured relative to H, and a loss_ex entry outside
    # atom_list would be ignored without a word.
    atom_list = list(getattr(cfg, "atom_list", []))
    loss_ex = list(getattr(cfg, "loss_ex", []))
    if "H" in loss_ex or not set(loss_ex) <= set(atom_list):
        errors.append(f"loss_ex={loss_ex} must be a subset of atom_list={atom_list} "
                      "without H (the element budget is measured relative to H).")
    return errors


def _validate_condensation(cfg) -> list[str]:
    """Return errors for an unsupported or silently-inert condensation config.

    Forward-run validation only, applied to every consumer identically.

    NOTE: the completed *pinned* condensation state is not
    differentiable-through (transient snapshot, discrete phase switches); that
    contract is enforced at the autodiff entry points in
    `steady_state_grad.py`, not here (see README.md, Limits).
    """
    if not bool(getattr(cfg, "use_condense", False)):
        return []

    from .atm_setup import _SUPPORTED_CONDENSABLES
    from .conden import SUPPORTED_CONDEN_KINETICS

    errors: list[str] = []
    condense_sp = list(getattr(cfg, "condense_sp", []) or [])

    if not condense_sp:
        errors.append(
            "use_condense=True with an empty condense_sp: nothing would "
            "condense. List the condensable gas species, or set "
            "use_condense=False."
        )

    # The condensation growth coefficient Dg IS the molecular-diffusion Dzz,
    # which setup zeros when use_moldiff=False, so every condensation rate
    # would silently be zero. Refuse rather than run inert.
    if not bool(getattr(cfg, "use_moldiff", True)):
        errors.append(
            "use_condense=True requires use_moldiff=True: the condensation "
            "growth coefficient Dg is the molecular-diffusion coefficient Dzz, "
            "which is zeroed when use_moldiff=False, so every condensation rate "
            "would silently be zero."
        )

    start_ct = float(cfg.start_conden_time)
    stop_ct = float(cfg.stop_conden_time)
    if stop_ct < start_ct:
        errors.append(
            f"stop_conden_time={stop_ct} < start_conden_time={start_ct}: the "
            "condensation window would never open."
        )

    # Support tier: kinetics-ported condensates vs saturation-only (H2S).
    sat_only = sorted(set(_SUPPORTED_CONDENSABLES) - set(SUPPORTED_CONDEN_KINETICS))
    for sp in condense_sp:
        if sp not in _SUPPORTED_CONDENSABLES:
            errors.append(
                f"condense_sp entry {sp!r} is not a supported condensate. "
                f"Condensation kinetics are ported for "
                f"{sorted(SUPPORTED_CONDEN_KINETICS)}; {sat_only} have "
                "saturation data only (capping/cold-trap, no conden reactions, "
                "as in VULCAN-master). New condensates must be added "
                "explicitly with physical constants and tests."
            )
    return errors


def validate_runtime_config(cfg) -> None:
    """Raise RuntimeError if cfg is unsupported or required files are missing.

    Aggregates every configuration error (flag consistency, required
    files, network assets, abundance preset, numerical bounds) and raises once
    so the user sees all problems at once; returns None on success. Relative
    asset paths resolve as the run resolves them (`resolve_data_path`: the
    current directory first, then the package data).
    """
    errors: list[str] = []
    species = set(chem_funs.spec_list)

    if bool(getattr(cfg, "use_ion", False)) and not bool(
        getattr(cfg, "use_photo", False)
    ):
        errors.append("use_ion=True requires use_photo=True in VULCAN3.")

    fix_sp = list(getattr(cfg, "fix_species", []) or [])
    if fix_sp and not bool(getattr(cfg, "use_condense", False)):
        errors.append(
            "fix_species is set but use_condense=False; this configuration is inconsistent."
        )
    if fix_sp:
        # Otherwise an entry absent from the import-locked network dies later
        # on a bare `.index()` ValueError naming neither species nor remedy.
        missing = [sp for sp in fix_sp if sp not in species]
        if missing:
            errors.append(
                f"fix_species entries not in the loaded network: {missing}. "
                f"The network is import-locked, so this cannot be fixed after "
                f"`import vulcan3`: either drop these entries or set "
                f"$VULCAN3_NETWORK to a network containing them before the "
                f"first import."
            )

    if bool(getattr(cfg, "use_fix_H2He", False)):
        for sp in ("H2", "He"):
            if sp not in species:
                errors.append(
                    f"use_fix_H2He=True requires {sp!r} in the network; "
                    f"got species list without it."
                )

    if getattr(cfg, "ini_mix", None) == "const_mix":
        for sp in getattr(cfg, "const_mix", {}):
            if sp not in species:
                errors.append(
                    f"const_mix key {sp!r} is not a species of the loaded "
                    "network. Inert/background gases without network "
                    "reactions are not supported; VULCAN-master fails on "
                    "the same configuration (build_atm.ini_y calls "
                    "species.index(sp) unconditionally, e.g. its shipped "
                    "Earth example crashes on 'Ar'). Remove the key or use "
                    "a network that contains the species."
                )

    errors.extend(_validate_condensation(cfg))

    required_paths = [
        ("network", getattr(cfg, "network", None)),
        ("com_file", getattr(cfg, "com_file", None)),
        ("atm_file", getattr(cfg, "atm_file", None)),
    ]
    if bool(getattr(cfg, "use_photo", False)):
        required_paths.extend(
            [
                ("cross_folder", getattr(cfg, "cross_folder", None)),
                ("sflux_file", getattr(cfg, "sflux_file", None)),
            ]
        )
    if bool(getattr(cfg, "use_topflux", False)):
        required_paths.append(
            ("top_BC_flux_file", getattr(cfg, "top_BC_flux_file", None))
        )
    if bool(getattr(cfg, "use_botflux", False)):
        required_paths.append(
            ("bot_BC_flux_file", getattr(cfg, "bot_BC_flux_file", None))
        )
    if getattr(cfg, "ini_mix", None) == "EQ":
        required_paths.append(
            (
                "fastchem_solar_abundance_file",
                getattr(cfg, "fastchem_solar_abundance_file", None),
            )
        )

    for label, rel_path in required_paths:
        if not rel_path:
            errors.append(f"{label} is unset.")
            continue
        if not resolve_data_path(rel_path).exists():
            errors.append(
                f"{label}={rel_path!r} does not exist in the current directory "
                "or the package data."
            )

    errors.extend(_validate_network_assets(cfg))
    errors.extend(_validate_abundance_preset(cfg))
    errors.extend(_validate_numerical_bounds(cfg))

    # Plots run after the solve; refuse a plot request that would fail there.
    if getattr(cfg, "use_plot_end", False) or getattr(cfg, "use_plot_evo", False):
        if importlib.util.find_spec("matplotlib") is None:
            errors.append("use_plot_end / use_plot_evo need matplotlib: "
                          "pip install 'vulcan3[plot]'.")
        plot_spec = list(getattr(cfg, "plot_spec", []))
        missing = [sp for sp in plot_spec if sp not in species]
        if not plot_spec or missing:
            errors.append(f"plot_spec {plot_spec} must list species of the "
                          f"network; not in it: {missing}.")

    if errors:
        raise RuntimeError(
            "Unsupported or invalid VULCAN3 runtime configuration:\n- "
            + "\n- ".join(errors)
        )


def rate_temp_range_exposure(net, Tco) -> dict:
    """Count thermal rows evaluated outside their documented T ranges on `Tco`.

    Advisory only (no rate is altered). jwst-transit-authority's
    `forward.run_model` calls it; nothing in this package does, so vulture
    and a dead-code sweep miss that caller.

    Coverage per row is the UNION of its documented ranges (lenient: a
    Lindemann row's k0 and k_inf windows are OR'd, so this undercounts).
    A bare measurement temperature parses as a degenerate (T, T) range, so
    room-temperature-only rates count as out-of-range on hot profiles
    rather than hiding in the no-range bucket.
    """
    T = np.asarray(Tco, dtype=np.float64)
    no_range = any_outside = all_outside = 0
    for ranges in (net.temp_ranges or {}).values():
        if not ranges:
            no_range += 1
            continue
        covered = np.zeros(T.shape, dtype=bool)
        for lo, hi in ranges:
            covered |= (T >= lo) & (T <= hi)
        n_out = int(np.count_nonzero(~covered))
        if n_out:
            any_outside += 1
            if n_out == T.size:
                all_outside += 1
    return {
        "thermal_rows": len(net.temp_ranges or {}),
        "no_range": no_range,
        "any_outside": any_outside,
        "all_outside": all_outside,
    }
