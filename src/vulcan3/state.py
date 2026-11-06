"""Typed pre-loop input pytrees and runner state.

`RunState.with_pre_loop_setup(cfg)` runs the full pre-loop pipeline and
returns a populated pytree. The private `_Variables` / `_AtmData`
containers are scratch inside that constructor only.
"""

from __future__ import annotations

import contextlib
import copy
import types
import warnings
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from . import chem_funs
from ._legacy_containers import (
    _AtmData,
    _Variables,
    legacy_view,
)
from ._paths import resolve_data_path
from .config import _FROZEN_ENV, default_config

__all__ = [
    "TERM_CONVERGED",
    "TERM_NONFINITE",
    "TERM_RUNNING",
    "TERM_RUNTIME",
    "TERM_STEP_COUNT",
    "AtmInputs",
    "AtomInputs",
    "FixSpeciesInputs",
    "IniAbunOutputs",
    "ParamInputs",
    "PhotoInputs",
    "PhotoRuntimeInputs",
    "PhotoStaticInputs",
    "RateInputs",
    "RunMetadata",
    "RunState",
    "StepInputs",
    "legacy_view",
    "load_stellar_flux",
    "pytree_from_store",
    "runstate_from_store",
]


class AtmInputs(NamedTuple):
    """Atmosphere structure. nz layers, ni species; (nz-1) arrays are interface-staggered."""

    pco: jnp.ndarray  # (nz,)    pressure on cells (dyne/cm^2)
    pico: jnp.ndarray  # (nz+1,)  pressure on interfaces
    Tco: jnp.ndarray  # (nz,)    temperature (K)
    Kzz: jnp.ndarray  # (nz-1,)  eddy diffusion (cm^2/s)
    vz: jnp.ndarray  # (nz-1,)  vertical advection (cm/s)
    M: jnp.ndarray  # (nz,)    third-body number density
    n_0: jnp.ndarray  # (nz,)    total number density
    mu: jnp.ndarray  # (nz,)    mean molecular weight
    g: jnp.ndarray  # (nz,)    local gravity
    Hp: jnp.ndarray  # (nz,)    pressure scale height
    Hpi: jnp.ndarray  # (nz-1,)  interface scale height
    dz: jnp.ndarray  # (nz,)    layer thickness
    dzi: jnp.ndarray  # (nz-1,)  interface thickness
    zco: jnp.ndarray  # (nz+1,)  height at interfaces
    zmco: jnp.ndarray  # (nz,)    height at cell centers
    ms: jnp.ndarray  # (ni,)    molecular weight per species
    alpha: jnp.ndarray  # (ni,)    thermal diffusion factor
    Dzz: jnp.ndarray  # (nz-1, ni) molecular diffusion (interface)
    Dzz_cen: jnp.ndarray  # (nz, ni)   molecular diffusion (cell)
    vm: jnp.ndarray  # (nz-1, ni) molecular-diffusion drift velocity (interface)
    vs: jnp.ndarray  # (nz-1, ni) settling velocity
    top_flux: jnp.ndarray  # (ni,) top BC flux (#/cm^2/s)
    bot_flux: jnp.ndarray  # (ni,) bottom BC flux
    bot_vdep: jnp.ndarray  # (ni,) bottom deposition velocity
    bot_fix_sp: jnp.ndarray  # (ni,) bottom fixed mixing ratio


class RateInputs(NamedTuple):
    """Rate constants. `k[0]` is unused: reactions are 1-based throughout VULCAN."""

    k: jnp.ndarray  # (nr+1, nz) rate constants of every reaction


class IniAbunOutputs(NamedTuple):
    """Initial-abundance solver output. nz layers, ni species."""

    y: jnp.ndarray  # (nz, ni)    number density
    ymix: jnp.ndarray  # (nz, ni)    mixing ratio
    y_ini: jnp.ndarray  # (nz, ni)    initial-state snapshot
    charge_list: tuple[str, ...]  # charged species (empty when use_ion=False)


class PhotoInputs(NamedTuple):
    """TOA stellar flux + wavelength extents. `sflux_top` shape (0,) when use_photo=False."""

    sflux_top: jnp.ndarray
    def_bin_min: float
    def_bin_max: float


class PhotoStaticInputs(NamedTuple):
    """Dense photo cross-section pytree.

    `absp_cross` carries a row for every species in `absp_sp` union
    `absp_T_sp`; T-dep species also keep their 1D room-T cross.
    `din12_indx` is `-1` until the post-photo `read_sflux` write fires; use
    `with_din12_indx(idx)` to attach it.
    """

    bins: jnp.ndarray  # (nbin,)  float64 wavelength grid (nm)
    nbin: int
    dbin1: float
    dbin2: float
    din12_indx: int  # -1 sentinel pre-read_sflux

    absp_sp: tuple  # (str, ...)  ordered absorbers (photo + ion)
    absp_T_sp: tuple  # (str, ...)  T-dep subset of absp_sp
    scat_sp: tuple  # (str, ...)  Rayleigh scatterers
    branch_keys: tuple  # ((sp, branch_idx), ...) non-T dissociation
    branch_T_keys: tuple  # ((sp, branch_idx), ...) T-dep counterparts
    ion_branch_keys: tuple  # ((sp, branch_idx), ...) ion branches

    absp_cross: jnp.ndarray  # (n_absp, nbin)        room-T cross per absorber
    absp_T_cross: jnp.ndarray  # (n_absp_T, nz, nbin)  T-dep total cross
    scat_cross: jnp.ndarray  # (n_scat, nbin)
    cross_J: jnp.ndarray  # (n_br, nbin)
    cross_J_T: jnp.ndarray  # (n_br_T, nz, nbin)
    cross_Jion: jnp.ndarray  # (n_ion_br, nbin)  -- (0, nbin) when use_ion=False

    def with_din12_indx(self, idx: int) -> PhotoStaticInputs:
        """Return a new pytree with `din12_indx` set; rejects negative values."""
        if int(idx) < 0:
            raise ValueError(f"PhotoStaticInputs.din12_indx must be >= 0; got {idx}")
        return self._replace(din12_indx=int(idx))


class StepInputs(NamedTuple):
    """Time-stepping state. `y_evo`/`t_evo` are zero-length when save_evolution=False."""

    y: jnp.ndarray  # (nz, ni)  current number density
    y_prev: jnp.ndarray  # (nz, ni)  last accepted state
    ymix: jnp.ndarray  # (nz, ni)  mixing ratios
    t: float  # integration time
    dt: float  # current step size
    longdy: float  # long-window dy (Tsai+17 eq 10)
    longdydt: float  # long-window dy/dt (eq 11)
    y_evo: jnp.ndarray = jnp.zeros((0, 0, 0), dtype=jnp.float64)
    t_evo: jnp.ndarray = jnp.zeros((0,), dtype=jnp.float64)


# Termination codes (JaxIntegState.termination_reason, ParamInputs.end_case):
# 1-3 are master's end_case (op.py:1072-1084); 5 (non-finite, or stopped
# without converging and without a cap) is JAX-only; 0 means still running.
TERM_RUNNING = 0
TERM_CONVERGED = 1
TERM_RUNTIME = 2
TERM_STEP_COUNT = 3
TERM_NONFINITE = 5


class ParamInputs(NamedTuple):
    """Convergence + retry counters that cross the runner boundary."""

    count: int
    nega_count: int
    loss_count: int
    delta_count: int
    delta: float
    small_y: float
    nega_y: float
    end_case: int
    switch_final_photo_frq: bool
    pic_count: int
    where_varies_most: jnp.ndarray  # shape: (nz, ni)
    fix_species_start: bool
    # Why the integration stopped (a TERM_* code, the runner's
    # `JaxIntegState.termination_reason`); `end_case` is the coarser master
    # code. Defaulted so a positional constructor may omit it.
    termination_reason: int = TERM_RUNNING


class AtomInputs(NamedTuple):
    """Atom-conservation diagnostics. `atom_order` is the static ordering for the four arrays."""

    atom_order: tuple  # cfg.atom_list minus cfg.loss_ex, e.g. ('H','O','C','N')
    atom_ini: jnp.ndarray  # (n_atoms,)
    atom_loss: jnp.ndarray  # (n_atoms,)
    atom_loss_prev: jnp.ndarray  # (n_atoms,)
    atom_sum: jnp.ndarray  # (n_atoms,)


class PhotoRuntimeInputs(NamedTuple):
    """Per-step photo state. `None` on RunState when use_photo=False."""

    tau: jnp.ndarray  # (nz+1, nbin)
    aflux: jnp.ndarray  # (nz, nbin)
    sflux: jnp.ndarray  # (nz+1, nbin)
    dflux_u: jnp.ndarray  # (nz+1, nbin)
    aflux_change: float


class FixSpeciesInputs(NamedTuple):
    """Fixed-species snapshot.

    `conden_min_lev` is read only when fix_species_from_coldtrap_lev=True.
    """

    fix_species: tuple  # static ordering of species names
    fix_y: jnp.ndarray  # (n_fix_sp, nz)
    conden_min_lev: jnp.ndarray  # (n_fix_sp,) int


class RunMetadata(NamedTuple):
    """Host-side static metadata. Treat as immutable post-construction."""

    Rf: dict  # 1-based reaction id -> equation string
    n_branch: dict  # species -> int (#photodissoc branches)
    ion_branch: dict  # species -> int (#photoionisation branches)
    photo_sp: frozenset  # species with photodissociation
    ion_sp: frozenset  # species with photoionisation
    pho_rate_index: dict  # (sp, br) -> 1-based reaction index
    ion_rate_index: dict  # (sp, br) -> 1-based reaction index
    ion_br_ratio: dict  # (sp, br) -> branching ratio
    charge_list: tuple  # ion species (with non-zero charge)
    conden_re_list: tuple  # 1-based reaction ids that condense
    start_time: float  # wall-clock start (for the end-of-run report)
    Ti: jnp.ndarray  # (nz-1,) interface temperature, 0.5*(Tco[:-1]+Tco[1:]); Tco (nz,) when use_moldiff=False
    gas_indx: tuple  # gas-only species indices
    pref_indx: int  # reference-layer index (height integ)
    gs: float  # surface gravity (cm/s^2)
    sat_p: dict  # species -> (nz,) saturation pressure
    sat_mix: dict  # species -> (nz,) saturation mixing ratio
    r_p: dict  # condensate -> particle radius (cm)
    rho_p: dict  # condensate -> particle density (g/cm^3)
    fix_sp_indx: dict  # species -> banded-Jacobian indices
    y_ini: jnp.ndarray  # (nz, ni) initial-state snapshot from ini_y


class RunState(NamedTuple):
    """Umbrella pytree carrying every input the runner reads."""

    atm: AtmInputs
    rate: RateInputs
    photo: PhotoInputs
    # Runner-state slots. Default `None`: `pytree_from_store` fills only
    # atm/rate/photo.
    step: StepInputs | None = None
    params: ParamInputs | None = None
    atoms: AtomInputs | None = None
    photo_runtime: PhotoRuntimeInputs | None = None
    fix_species: FixSpeciesInputs | None = None
    # Host-side static metadata + photo cross-section pytree. Both default
    # to None for partial RunStates; `with_pre_loop_setup(cfg)` fills `metadata`, and
    # `photo_static` when use_photo
    # (`runstate_from_store` fills `metadata` only).
    metadata: RunMetadata | None = None
    photo_static: PhotoStaticInputs | None = None

    @staticmethod
    def with_pre_loop_setup(cfg) -> RunState:
        """Build a fully-initialised `RunState` from a config.

        Runs the full pre-loop pipeline (atmosphere structure, rate constants
        + low-T caps + reverse + remove, initial abundance solver, photo cross
        sections, photo runtime arrays + remove pass) and snapshots the result
        into the typed pytree. The canonical pre-loop entry point:
        ``rs = RunState.with_pre_loop_setup(cfg); rs = integ(rs)``.

        Fails fast on network/com_file/atom_list import-lock mismatches and on
        an invalid runtime configuration, then runs the pipeline with ``cfg``
        overlaid onto the process default so ``make_config()`` overrides are
        honored with zero leakage.
        """
        _assert_frozen_knobs_match_import(cfg)
        from .runtime_validation import validate_runtime_config

        validate_runtime_config(cfg)
        with _cfg_overlay(cfg):
            return _build_pre_loop_runstate_impl(cfg)


# Photolysis wavelength window (nm): the stellar-flux bin range is clamped to
# it, as master's store.py:79-80.
_SFLUX_BIN_MIN_NM = 2.0
_SFLUX_BIN_MAX_NM = 700.0


def load_stellar_flux(cfg) -> tuple[float, float]:
    """Read the stellar flux file and return the spectral-bin extents.

    Returns `(def_bin_min, def_bin_max)`, clamped to [2, 700] nm, or
    `(0.0, 0.0)` when `cfg.use_photo` is False so callers can call
    unconditionally.
    """
    if not bool(cfg.use_photo):
        return 0.0, 0.0
    sflux_data = np.genfromtxt(
        resolve_data_path(cfg.sflux_file),
        dtype=float,
        skip_header=1,
        names=["lambda", "flux"],
    )
    wavelength = np.asarray(sflux_data["lambda"], dtype=np.float64)
    return (
        float(max(wavelength[0], _SFLUX_BIN_MIN_NM)),
        float(min(wavelength[-1], _SFLUX_BIN_MAX_NM)),
    )


def pytree_from_store(var, atm) -> RunState:
    """Snapshot atm + rate + photo fields off a legacy (var, atm) into a typed RunState."""
    nr = int(chem_funs.nr)
    nz = int(atm.Tco.shape[0])

    atm_inputs = AtmInputs(
        pco=jnp.asarray(atm.pco),
        pico=jnp.asarray(atm.pico),
        Tco=jnp.asarray(atm.Tco),
        Kzz=jnp.asarray(atm.Kzz),
        vz=jnp.asarray(atm.vz),
        M=jnp.asarray(atm.M),
        n_0=jnp.asarray(atm.n_0),
        mu=jnp.asarray(atm.mu),
        g=jnp.asarray(atm.g),
        Hp=jnp.asarray(atm.Hp),
        Hpi=jnp.asarray(getattr(atm, "Hpi", np.zeros_like(np.asarray(atm.dzi)))),
        dz=jnp.asarray(atm.dz),
        dzi=jnp.asarray(atm.dzi),
        zco=jnp.asarray(atm.zco),
        zmco=jnp.asarray(atm.zmco),
        ms=jnp.asarray(atm.ms),
        alpha=jnp.asarray(atm.alpha),
        Dzz=jnp.asarray(atm.Dzz),
        Dzz_cen=jnp.asarray(atm.Dzz_cen),
        vm=jnp.asarray(atm.vm),
        vs=jnp.asarray(atm.vs),
        top_flux=jnp.asarray(atm.top_flux),
        bot_flux=jnp.asarray(atm.bot_flux),
        bot_vdep=jnp.asarray(atm.bot_vdep),
        bot_fix_sp=jnp.asarray(atm.bot_fix_sp),
    )

    k_dense = np.asarray(var.k_arr, dtype=np.float64)
    if k_dense.shape != (nr + 1, nz):
        raise ValueError(f"var.k_arr shape {k_dense.shape} != expected ({nr + 1}, {nz})")
    rate_inputs = RateInputs(k=jnp.asarray(k_dense))

    sflux_top = getattr(var, "sflux_top", None)
    if sflux_top is None:
        sflux_top_arr = jnp.zeros((0,), dtype=jnp.float64)
    else:
        sflux_top_arr = jnp.asarray(sflux_top, dtype=jnp.float64).reshape(-1)
    photo_inputs = PhotoInputs(
        sflux_top=sflux_top_arr,
        def_bin_min=float(getattr(var, "def_bin_min", 0.0)),
        def_bin_max=float(getattr(var, "def_bin_max", 0.0)),
    )

    return RunState(atm=atm_inputs, rate=rate_inputs, photo=photo_inputs)


def _atom_dict_to_arr(d, atom_order) -> np.ndarray:
    return np.asarray(
        [float(d.get(a, 0.0)) for a in atom_order],
        dtype=np.float64,
    )


def _atom_arr_to_dict(arr, atom_order) -> dict:
    return {a: float(arr[i]) for i, a in enumerate(atom_order)}


# size-limit-ok: one RunState field per line
def runstate_from_store(var, atm, para) -> RunState:
    """Snapshot the full runner-input state of `(var, atm, para)`.

    Carries pre-loop fields plus runner-state slots (step/params/atoms/
    photo_runtime/fix_species). Anything the runner needs at entry comes
    from here.
    """
    _cfg = default_config()

    base = pytree_from_store(var, atm)

    nz = int(atm.Tco.shape[0])
    ni = int(np.asarray(var.y).shape[1])

    step = StepInputs(
        y=jnp.asarray(var.y, dtype=jnp.float64),
        y_prev=jnp.asarray(
            var.y if getattr(var, "y_prev", None) is None else var.y_prev,
            dtype=jnp.float64,
        ),
        ymix=jnp.asarray(var.ymix, dtype=jnp.float64),
        t=float(var.t),
        dt=float(var.dt),
        longdy=float(getattr(var, "longdy", 1.0)),
        longdydt=float(getattr(var, "longdydt", 1.0)),
    )

    params = ParamInputs(
        count=int(getattr(para, "count", 0)),
        nega_count=int(getattr(para, "nega_count", 0)),
        loss_count=int(getattr(para, "loss_count", 0)),
        delta_count=int(getattr(para, "delta_count", 0)),
        delta=float(getattr(para, "delta", 0.0)),
        small_y=float(getattr(para, "small_y", 0.0)),
        nega_y=float(getattr(para, "nega_y", 0.0)),
        end_case=int(getattr(para, "end_case", 0)),
        switch_final_photo_frq=bool(getattr(para, "switch_final_photo_frq", False)),
        pic_count=int(getattr(para, "pic_count", 0)),
        where_varies_most=jnp.asarray(
            getattr(para, "where_varies_most", np.zeros((nz, ni))),
            dtype=jnp.float64,
        ),
        fix_species_start=bool(getattr(para, "fix_species_start", False)),
        termination_reason=int(getattr(para, "termination_reason", 0)),
    )

    atom_order = tuple(a for a in _cfg.atom_list if a not in _cfg.loss_ex)
    atoms = AtomInputs(
        atom_order=atom_order,
        atom_ini=jnp.asarray(_atom_dict_to_arr(getattr(var, "atom_ini", {}), atom_order)),
        atom_loss=jnp.asarray(_atom_dict_to_arr(getattr(var, "atom_loss", {}), atom_order)),
        atom_loss_prev=jnp.asarray(_atom_dict_to_arr(getattr(var, "atom_loss_prev", {}), atom_order)),
        atom_sum=jnp.asarray(_atom_dict_to_arr(getattr(var, "atom_sum", {}), atom_order)),
    )

    if bool(_cfg.use_photo) and hasattr(var, "tau"):
        photo_runtime = PhotoRuntimeInputs(
            tau=jnp.asarray(var.tau, dtype=jnp.float64),
            aflux=jnp.asarray(var.aflux, dtype=jnp.float64),
            sflux=jnp.asarray(var.sflux, dtype=jnp.float64),
            dflux_u=jnp.asarray(var.dflux_u, dtype=jnp.float64),
            aflux_change=float(getattr(var, "aflux_change", 0.0)),
        )
    else:
        photo_runtime = None

    fix_species_cfg = list(_cfg.fix_species)
    if fix_species_cfg:
        fix_y_arr = np.zeros((len(fix_species_cfg), nz), dtype=np.float64)
        coldtrap = np.zeros((len(fix_species_cfg),), dtype=np.int32)
        var_fix_y = getattr(var, "fix_y", {}) or {}
        var_coldtrap = getattr(atm, "conden_min_lev", {}) or {}
        for i, sp in enumerate(fix_species_cfg):
            row = var_fix_y.get(sp)
            if row is not None:
                fix_y_arr[i] = np.asarray(row, dtype=np.float64)
            coldtrap[i] = int(var_coldtrap.get(sp, 0))
        fix_species = FixSpeciesInputs(
            fix_species=tuple(fix_species_cfg),
            fix_y=jnp.asarray(fix_y_arr),
            conden_min_lev=jnp.asarray(coldtrap),
        )
    else:
        fix_species = FixSpeciesInputs(
            fix_species=(),
            fix_y=jnp.zeros((0, nz), dtype=jnp.float64),
            conden_min_lev=jnp.zeros((0,), dtype=jnp.int32),
        )

    return base._replace(
        step=step,
        params=params,
        atoms=atoms,
        photo_runtime=photo_runtime,
        fix_species=fix_species,
        metadata=_runmetadata_from_legacy(var, atm, para),
    )


def _atm_metadata_from_atm(atm) -> dict:
    return {
        "Ti": jnp.asarray(getattr(atm, "Ti", atm.Tco), dtype=jnp.float64),
        "gas_indx": tuple(getattr(atm, "gas_indx", []) or []),
        "pref_indx": int(getattr(atm, "pref_indx", 0)),
        "gs": float(getattr(atm, "gs", 0.0)),
        "sat_p": dict(getattr(atm, "sat_p", {}) or {}),
        "sat_mix": dict(getattr(atm, "sat_mix", {}) or {}),
        "r_p": dict(getattr(atm, "r_p", {}) or {}),
        "rho_p": dict(getattr(atm, "rho_p", {}) or {}),
        "fix_sp_indx": dict(getattr(atm, "fix_sp_indx", {}) or {}),
    }


def _runmetadata_from_legacy(var, atm, para) -> RunMetadata:
    """Snapshot host-side static metadata that doesn't fit the JAX pytrees."""
    atm_meta = _atm_metadata_from_atm(atm)
    y_ini_src = getattr(var, "y_ini", None)
    if y_ini_src is None:
        y_ini_src = getattr(var, "y", None)
    if y_ini_src is None:
        nz = int(np.asarray(atm.Tco).shape[0])
        y_ini_arr = jnp.zeros((nz, int(chem_funs.ni)), dtype=jnp.float64)
    else:
        y_ini_arr = jnp.asarray(np.asarray(y_ini_src), dtype=jnp.float64)
    return RunMetadata(
        Rf=dict(getattr(var, "Rf", {}) or {}),
        n_branch=dict(getattr(var, "n_branch", {}) or {}),
        ion_branch=dict(getattr(var, "ion_branch", {}) or {}),
        photo_sp=frozenset(getattr(var, "photo_sp", set()) or set()),
        ion_sp=frozenset(getattr(var, "ion_sp", set()) or set()),
        pho_rate_index=dict(getattr(var, "pho_rate_index", {}) or {}),
        ion_rate_index=dict(getattr(var, "ion_rate_index", {}) or {}),
        ion_br_ratio=dict(getattr(var, "ion_br_ratio", {}) or {}),
        charge_list=tuple(getattr(var, "charge_list", []) or []),
        conden_re_list=tuple(getattr(var, "conden_re_list", []) or []),
        start_time=float(getattr(para, "start_time", 0.0)),
        y_ini=y_ini_arr,
        **atm_meta,
    )


def _same_file(want: str, have: str, what: str) -> bool:
    """Return True when `want` is `have` or a byte-identical copy of it.

    An unreadable pair warns and counts as a match: the skip announces itself.
    """
    if want == have:
        return True
    try:
        with Path(want).open("rb") as _w, Path(have).open("rb") as _h:
            return _w.read() == _h.read()
    except FileNotFoundError:
        return False  # the requested file is missing -> a mismatch
    except OSError as exc:
        warnings.warn(
            f"{what} import-lock guard could not compare {want!r} with the "
            f"import-time {have!r} ({type(exc).__name__}: {exc}); the check "
            "was SKIPPED, not passed.",
            RuntimeWarning,
            stacklevel=3,
        )
        return True


def _frozen_mismatch(cfg) -> dict[str, tuple]:
    """Return the import-frozen knobs (`config._FROZEN_ENV`) on which `cfg` disagrees with the first `import vulcan3`.

    Returned as ``{knob: (requested, import-time)}``. A file knob matches a byte-identical copy at another path.

    The network (``ni``/``nr``/``spec_list`` and the codegen RHS), the
    composition table (``composition.py``) and the reservoir-projection
    tables (``jax_step``) are built from these once, at import.
    """
    from . import composition as _composition, jax_step as _jax_step

    loaded = {
        "network": chem_funs._NETWORK.network_path,
        "com_file": _composition.COM_FILE_PATH,
        "atom_list": _jax_step.IMPORT_ATOM_LIST,
    }
    out = {}
    for knob in _FROZEN_ENV:
        if knob == "atom_list":
            want, have = list(cfg.atom_list), list(loaded[knob])
            same = want == have
        else:
            want = str(Path(resolve_data_path(getattr(cfg, knob))).resolve())
            have = str(Path(loaded[knob]).resolve())
            same = _same_file(want, have, knob)
        if not same:
            out[knob] = (want, have)
    return out


def _assert_frozen_knobs_match_import(cfg) -> None:
    """Fail fast if `cfg` requests another network, com_file or atom_list.

    It must match the import-frozen one (`_frozen_mismatch`). ``make_config(...)`` cannot change them
    in-process, and running on would mix the import-time tables with cfg-time accounting.
    """
    bad = _frozen_mismatch(cfg)
    if not bad:
        return
    lines = "".join(f"  {knob}: import-time {have}, cfg requests {want}\n" for knob, (want, have) in bad.items())
    envs = ", ".join(f"${_FROZEN_ENV[k]}" for k in bad)
    raise ValueError(
        f"cfg changes the import-locked {', '.join(bad)}: VULCAN3 reads them "
        f"once, at the first `import vulcan3`.\n{lines}"
        f"Set {envs} (or the config used at that import) BEFORE the first "
        "`import vulcan3`, or use the subprocess driver."
    )


@contextlib.contextmanager
def _cfg_overlay(cfg):
    """Overlay cfg's public attributes onto the process default config for setup.

    The pre-loop scratch containers, the ``atm_setup``/``ini_abun`` facades,
    and ``legacy_io.ReadRate`` read knobs off ``config.default_config()``, not
    a threaded argument. When ``cfg`` IS that default this is a no-op; when it
    is a separate namespace, overlay its public attributes for the duration of
    setup and restore on exit, so overrides take effect with zero leakage.
    The runner-side counterpart is ``OuterLoop(cfg=...)``.
    """
    base = default_config()

    if cfg is base:
        yield
        return

    before: dict = {}
    added: list = []
    for name, val in vars(cfg).items():
        if name.startswith("_") or callable(val) or hasattr(val, "__loader__"):
            continue
        if hasattr(base, name):
            before[name] = copy.deepcopy(getattr(base, name))
        else:
            added.append(name)
        # Copy mutable containers rather than aliasing, so an in-place write
        # through `base` can never reach the caller's cfg. Setup must not
        # write config knobs: such a write is lost when the overlay ends.
        if isinstance(val, (list, dict, set, bytearray)):
            val = copy.deepcopy(val)
        setattr(base, name, val)
    try:
        yield
    finally:
        for name, val in before.items():
            setattr(base, name, val)
        for name in added:
            delattr(base, name)


def _build_pre_loop_runstate_impl(cfg) -> RunState:
    """Run the full pre-loop pipeline and return a populated RunState."""
    import time

    from . import legacy_io as _io, op_jax as _op_jax, photo_setup as _photo_setup, rates_jax as _rates_mod
    from .atm_setup import Atm
    from .ini_abun import InitialAbun

    var = _Variables()
    atm = _AtmData()
    para = types.SimpleNamespace(start_time=time.time())

    make_atm = Atm()

    atm = make_atm.f_pico(atm)
    atm = make_atm.load_TPK(atm)
    if bool(cfg.high_temp_cut):
        atm = make_atm.apply_high_temp_cut(atm)
    if bool(cfg.use_condense):
        make_atm.sp_sat(atm)

    rate = _io.ReadRate()
    var = rate.read_rate(var, atm)
    network = _rates_mod.setup_var_k(cfg, var, atm)
    ini = InitialAbun()
    var = ini.ini_y(var, atm)
    var = ini.ele_sum(var)

    atm = make_atm.f_mu_dz(var, atm, None)
    make_atm.mol_diff(atm)
    make_atm.BC_flux(atm)

    photo_static_pytree = None
    if bool(cfg.use_photo):
        photo_static_pytree = _photo_setup.populate_photo(var, atm)
        make_atm.read_sflux(var, atm)
        photo_static_pytree = photo_static_pytree.with_din12_indx(int(var.sflux_din12_indx))
        solver = _op_jax.Ros2JAX()
        solver._photo_static = photo_static_pytree
        solver.compute_tau(var, atm)
        solver.compute_flux(var, atm)
        solver.compute_J(var, atm)
        if bool(cfg.use_ion):
            solver.compute_Jion(var, atm)
        _rates_mod.apply_photo_remove(cfg, var, network)

    # Mirror VULCAN master's initial Integration.backup() so the first
    # reject/force-accept path can revert to the real pre-step state.
    var.y_prev = np.asarray(var.y, dtype=np.float64).copy()
    var.atom_loss_prev = var.atom_loss.copy()

    rs = runstate_from_store(var, atm, para)
    return rs._replace(photo_static=photo_static_pytree)
