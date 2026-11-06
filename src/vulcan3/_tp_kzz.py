"""The T-P grid and the Kzz profile: pressure levels, gravity, the analytical T-P and the atmosphere table.

Units: CGS throughout: T in K, pressure in dyn cm^-2, number density in cm^-3, length in cm, time in s.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import jax.scipy.special as jsp_special
import numpy as np

from ._paths import resolve_data_path
from .phy_const import BAR_CGS, G_grav, kb

__all__ = [
    "analytical_TP_H14",
    "compute_pico",
    "high_temp_cut_regrid",
    "kzz_profile_jax",
    "load_TPK",
    "surface_gravity",
]

logger = logging.getLogger(__name__)


def compute_pico(pco: jnp.ndarray) -> jnp.ndarray:
    """Pressure grid at cell interfaces from cell centres. Returns shape (nz+1,).

    Interior interfaces are the geometric mean of adjacent cells; outer
    boundaries are log-extrapolated from the nearest pair.
    """
    pco = jnp.asarray(pco)
    pco_up1 = jnp.roll(pco, 1)
    interior = jnp.sqrt(pco * pco_up1)
    bottom = pco[0] ** 1.5 * pco[1] ** -0.5
    top = pco[-1] ** 1.5 * pco[-2] ** -0.5
    return jnp.concatenate(
        [
            jnp.array([bottom]),
            interior[1:],
            jnp.array([top]),
        ]
    )


def surface_gravity(cfg: Any) -> float:
    """Surface gravity at the reference level (cm/s^2): ``g = G*Mp/Rp**2``.

    Single source of gravity (upstream vm_branch); there is no ``gs`` knob.
    Callers supply ``Mp`` (planet mass, g) and ``Rp`` (planet radius, cm).
    """
    Mp, Rp = cfg.Mp, cfg.Rp
    if Mp is None or Rp is None or float(Mp) <= 0.0 or float(Rp) <= 0.0:
        raise RuntimeError(
            "Surface gravity is undetermined: set `Mp` (planet mass, g) and "
            f"`Rp` (planet radius, cm), both > 0. Got Mp={Mp!r}, Rp={Rp!r}."
        )
    return G_grav * float(Mp) / float(Rp) ** 2


def high_temp_cut_regrid(
    pco: np.ndarray | jnp.ndarray, Tco: np.ndarray | jnp.ndarray, *, T_max: float, P_min: float, P_t: float, nz: int
) -> np.ndarray | None:
    """Re-gridded pressure column for the high-temperature bottom cut.

    ``pco[0]`` is the deepest (highest-pressure) level. If any layer at
    ``P >= P_min`` is hotter than ``T_max``, the bottom pressure is lowered to
    the first deep level that is cool enough (floored at ``P_min``) and the
    column is re-gridded onto ``nz`` logspaced levels down to ``P_t``.
    Returns ``None`` when no cut is needed. NumPy port of vm_branch
    ``build_atm.apply_high_temp_cut``; host-side setup only.
    """
    pco = np.asarray(pco, dtype=np.float64)
    Tco = np.asarray(Tco, dtype=np.float64)
    deep = pco >= P_min
    if not np.any(deep) or np.all(Tco[deep] <= T_max):
        return None
    ok_deep = np.where(deep & (Tco <= T_max))[0]
    new_P_b = float(P_min) if ok_deep.size == 0 else max(float(pco[ok_deep[0]]), float(P_min))
    if new_P_b >= pco[0]:
        return None
    return np.logspace(np.log10(new_P_b), np.log10(P_t), int(nz))


def analytical_TP_H14(pco: jnp.ndarray, params: Any, *, gs: float, Pb: float) -> jnp.ndarray:
    """Heng et al. 2014 analytical T(P) profile.

    `params` is the 6-tuple `(T_int, T_irr, ka_0, ka_s, beta_s, beta_l)`.
    Vectorised over `pco`.
    """
    T_int, T_irr, ka_0, ka_s, beta_s, beta_l = params
    pco = jnp.asarray(pco)

    albedo = (1.0 - beta_s) / (1.0 + beta_s)
    T_irr_eff = T_irr * (1.0 - albedo) ** 0.25
    eps_L = 3.0 / 8.0
    eps_L3 = 1.0 / 3.0
    ka_CIA = 0.0
    m = pco / gs
    m_0 = Pb / gs
    ka_l = ka_0 + ka_CIA * m / m_0
    term1 = T_int**4 / 4.0 * (1.0 / eps_L + m / (eps_L3 * beta_l**2) * (ka_0 + ka_CIA * m / (2.0 * m_0)))
    arg = ka_s * m / beta_s
    expn2 = jsp_special.expn(2, arg)
    expn4 = jsp_special.expn(4, arg)
    term2 = 1.0 / (2.0 * eps_L) + expn2 * (
        ka_s / (ka_l * beta_s) - ka_CIA * m * beta_s / (eps_L3 * ka_s * m_0 * beta_l**2)
    )
    term3 = ka_0 * beta_s / (eps_L3 * ka_s * beta_l**2) * (1.0 / 3.0 - expn4)
    term4 = 0.0
    return (term1 + T_irr_eff**4 / 8.0 * (term2 + term3 + term4)) ** 0.25


def kzz_profile_jax(
    kzz_prof: str,
    pico_int: np.ndarray | jnp.ndarray,
    *,
    K_deep: float | jnp.ndarray = 0.0,
    K_max: float | jnp.ndarray = 0.0,
    K_p_lev: float | jnp.ndarray = 0.0,
    const_Kzz: float | jnp.ndarray = 0.0,
) -> jnp.ndarray:
    """Analytic eddy-diffusion profile Kzz(P) at interior interfaces (cm^2/s).

    `pico_int` is ``pico[1:-1]`` (interface pressures, shape ``(nz-1,)``,
    dyne/cm^2). Single source for the ``const``/``JM16``/``Pfunc`` branches of
    :func:`load_TPK`, in ``jnp`` so the profile is differentiable w.r.t. its
    parameters and the grid (the ``file`` profile has no closed form and lives
    in :func:`load_TPK`). Each branch reads only its own parameter; a missing
    required knob must fail loud at the call site, not be defaulted here.
    """
    # jnp.asarray (not float()) keeps the parameters tracer-safe for AD;
    # bit-identical for the production Python-float call path.
    pico_int = jnp.asarray(pico_int, dtype=jnp.float64)
    K_deep = jnp.asarray(K_deep, dtype=jnp.float64)
    K_max = jnp.asarray(K_max, dtype=jnp.float64)
    K_p_lev = jnp.asarray(K_p_lev, dtype=jnp.float64)
    const_Kzz = jnp.asarray(const_Kzz, dtype=jnp.float64)
    if kzz_prof == "const":
        return jnp.zeros_like(pico_int) + const_Kzz
    if kzz_prof == "JM16":
        # Moses+2016 fit, build_atm.py:414 (1e-3 converts dyne/cm^2 to mbar).
        return jnp.maximum(K_deep, 1e5 * (300.0 / (pico_int * 1e-3)) ** 0.5)
    if kzz_prof == "Pfunc":
        # Tsai+2020 fit, build_atm.py:417.
        return jnp.maximum(K_max, K_max * (K_p_lev * BAR_CGS / pico_int) ** 0.4)
    raise OSError(
        f'\n"Kzz_prof"={kzz_prof!r} cannot be recognized.\n'
        f'Assign it as "file", "const", "JM16" or "Pfunc" in the config.'
    )


def _interp_descending_or_ascending(
    x_query: np.ndarray,
    xp_raw: np.ndarray,
    fp_raw: np.ndarray,
    fill_left: float,
    fill_right: float,
) -> np.ndarray:
    """`scipy.interp1d(bounds_error=False, fill_value=(left, right))` via `jnp.interp`.

    Sorts `xp` ascending on the host so `jnp.interp` is happy.
    """
    xp_raw = np.asarray(xp_raw, dtype=np.float64)
    fp_raw = np.asarray(fp_raw, dtype=np.float64)
    order = np.argsort(xp_raw)
    xp_sorted = xp_raw[order]
    fp_sorted = fp_raw[order]
    return np.asarray(
        jnp.interp(
            jnp.asarray(x_query, dtype=jnp.float64),
            jnp.asarray(xp_sorted),
            jnp.asarray(fp_sorted),
            left=float(fill_left),
            right=float(fill_right),
        )
    )


def _read_atm_table(atm_file: str) -> dict[str, np.ndarray]:
    """Read an atm/atm_*.txt file.

    Returns dict with `Pressure`/`Temp` and optional `Kzz`/`vz` columns.
    """
    table = np.genfromtxt(resolve_data_path(atm_file), names=True, dtype=None, skip_header=1)
    out: dict[str, np.ndarray] = {
        "Pressure": np.asarray(table["Pressure"], dtype=np.float64),
        "Temp": np.asarray(table["Temp"], dtype=np.float64),
    }
    if "Kzz" in table.dtype.names:
        out["Kzz"] = np.asarray(table["Kzz"], dtype=np.float64)
    if "vz" in table.dtype.names:
        out["vz"] = np.asarray(table["vz"], dtype=np.float64)
    nrow = out["Pressure"].size
    if out["Pressure"].ndim != 1 or nrow < 2:
        raise ValueError(f"atmosphere table {atm_file}: need at least two rows")
    if any(values.ndim != 1 or values.size != nrow for values in out.values()):
        raise ValueError(f"atmosphere table {atm_file}: columns have inconsistent lengths")
    if any(not np.all(np.isfinite(values)) for values in out.values()):
        raise ValueError(f"atmosphere table {atm_file}: all values must be finite")
    # Upstream (build_atm.load_TPK, interp1d with assume_sorted=False) sorts
    # the pressure column and accepts Kzz = 0; only sign/finiteness are refused.
    if np.any(out["Pressure"] <= 0.0) or np.any(out["Temp"] <= 0.0):
        raise ValueError(
            f"atmosphere table {atm_file}: Pressure and Temp must be positive "
            f"(min Pressure {out['Pressure'].min():g}, min Temp {out['Temp'].min():g})"
        )
    if "Kzz" in out and np.any(out["Kzz"] < 0.0):
        raise ValueError(f"atmosphere table {atm_file}: Kzz must be >= 0 (min {out['Kzz'].min():g})")
    return out


# size-limit-ok: one branch per atm_type and Kzz_prof, as master
def load_TPK(cfg: Any, pco: np.ndarray, *, pico: np.ndarray) -> dict[str, np.ndarray]:  # noqa: C901 (one branch per atm_type and Kzz_prof, as master)
    """Build (Tco, Kzz, vz, M, n_0) for the configured atm/Kzz/vz modes.

    `pco` and `pico` are inputs because the `table` mode
    overwrites `pco` from a saved file.
    """
    nz = int(pco.shape[0])
    atm_type = cfg.atm_type
    Kzz_prof = cfg.Kzz_prof
    vz_prof = cfg.vz_prof
    use_Kzz = bool(cfg.use_Kzz)
    use_vz = bool(cfg.use_vz)

    out: dict[str, np.ndarray] = {}
    table_for_vz: dict[str, np.ndarray] | None = None

    if atm_type == "isothermal":
        Tco = np.full(nz, float(cfg.Tiso), dtype=np.float64)
    elif atm_type == "analytical":
        Tco = np.asarray(
            analytical_TP_H14(
                jnp.asarray(pco),
                cfg.para_anaTP,
                gs=surface_gravity(cfg),
                # pco[0] (deepest level) keeps the profile self-consistent
                # after a high_temp_cut re-grid lowers the bottom pressure.
                Pb=float(np.asarray(pco)[0]),
            )
        )
    elif atm_type == "file":
        table = _read_atm_table(cfg.atm_file)
        table_for_vz = table
        p_file = table["Pressure"]
        T_file = table["Temp"]
        if max(p_file) < pco[0] or min(p_file) > pco[-1]:
            logger.warning(
                "P_b and P_t assigned in the config are out of range of the input. Constant extension is used."
            )
        Tco = _interp_descending_or_ascending(
            pco,
            p_file,
            T_file,
            fill_left=float(T_file[np.argmin(p_file)]),
            fill_right=float(T_file[np.argmax(p_file)]),
        )
        if use_Kzz and Kzz_prof == "file":
            Kzz_file = table["Kzz"]
            out["Kzz"] = _interp_descending_or_ascending(
                pico[1:-1],
                p_file,
                Kzz_file,
                fill_left=float(Kzz_file[np.argmin(p_file)]),
                fill_right=float(Kzz_file[np.argmax(p_file)]),
            )
    elif atm_type == "vulcan_ini":
        import pickle

        logger.info("Initializing PT from the previous run %s", cfg.vul_ini)
        with Path(resolve_data_path(cfg.vul_ini)).open("rb") as handle:
            vul_data = pickle.load(handle)
        Tco = np.asarray(vul_data["atm"]["Tco"], dtype=np.float64)
    elif atm_type == "table":
        logger.info("Initializing PT from the previous run %s", cfg.vul_ini)
        prev = np.genfromtxt(resolve_data_path(cfg.vul_ini), names=True, dtype=None, skip_header=1)
        if not nz == len(prev["Pressure"]):
            raise OSError(f"{cfg.vul_ini} has {len(prev['Pressure'])} layers, the config's nz is {nz}.")
        out["pco"] = np.asarray(prev["Pressure"], dtype=np.float64)
        Tco = np.asarray(prev["Temp"], dtype=np.float64)
    else:
        raise OSError(f'\n"atm_type"={atm_type!r} cannot be recognized.\nPlease reassign it in the config.')
    out["Tco"] = Tco

    # Read each profile's knob directly so a missing one raises, as in master.
    if Kzz_prof == "const":
        out["Kzz"] = np.asarray(
            kzz_profile_jax("const", pico[1:-1], const_Kzz=cfg.const_Kzz),
            dtype=np.float64,
        )
    elif Kzz_prof == "JM16":
        out["Kzz"] = np.asarray(
            kzz_profile_jax("JM16", pico[1:-1], K_deep=cfg.K_deep),
            dtype=np.float64,
        )
    elif Kzz_prof == "Pfunc":
        out["Kzz"] = np.asarray(
            kzz_profile_jax("Pfunc", pico[1:-1], K_max=cfg.K_max, K_p_lev=cfg.K_p_lev),
            dtype=np.float64,
        )
    elif Kzz_prof == "file":
        if "Kzz" not in out:
            raise OSError('Kzz_prof="file" requires atm_type="file" with a Kzz column.')
    else:
        raise OSError(
            f'\n"Kzz_prof"={Kzz_prof!r} cannot be recognized.\n'
            f'Assign it as "file", "const", "JM16" or "Pfunc" in the config.'
        )

    if vz_prof == "const":
        out["vz"] = np.full(nz - 1, float(cfg.const_vz), dtype=np.float64)
    elif vz_prof == "file":
        if table_for_vz is None or "vz" not in table_for_vz:
            raise OSError('vz_prof="file" requires atm_type="file" with a vz column.')
        p_file = table_for_vz["Pressure"]
        vz_file = table_for_vz["vz"]
        out["vz"] = _interp_descending_or_ascending(
            pico[1:-1],
            p_file,
            vz_file,
            fill_left=0.0,
            fill_right=0.0,
        )
    else:
        raise OSError(f'\n"vz_prof"={vz_prof!r} cannot be recognized.\nAssign it as "file" or "const" in the config.')

    # Force-zero when the corresponding `use_*` switch is off, regardless of cfg.
    if not use_Kzz:
        out["Kzz"] = np.zeros(nz - 1, dtype=np.float64)
    if not use_vz:
        out["vz"] = np.zeros(nz - 1, dtype=np.float64)

    pco_for_M = out.get("pco", pco)
    M = jnp.asarray(pco_for_M, dtype=jnp.float64) / (kb * jnp.asarray(out["Tco"], dtype=jnp.float64))
    out["M"] = np.asarray(M)
    out["n_0"] = np.asarray(M)

    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}
