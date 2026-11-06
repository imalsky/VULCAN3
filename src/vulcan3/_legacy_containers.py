"""The legacy `var`, `atm` and `para` views of a `RunState`, for master-facing code."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .state import RunState

import types

import numpy as np

from .config import default_config

__all__ = ["legacy_view"]


# size-limit-ok: one legacy field per line
def legacy_view(rs: RunState) -> tuple[types.SimpleNamespace, types.SimpleNamespace, types.SimpleNamespace]:
    """Return a `(var, atm, para)` SimpleNamespace shim built from `rs`.

    Mutations to the shim do not round-trip back to `rs`.
    """
    md = rs.metadata
    var = types.SimpleNamespace()
    atm = types.SimpleNamespace()
    para = types.SimpleNamespace()

    a = rs.atm
    for f in a._fields:
        setattr(atm, f, np.asarray(getattr(a, f)))
    atm.Ti = np.asarray(md.Ti)
    atm.gas_indx = list(md.gas_indx)
    atm.pref_indx = int(md.pref_indx)
    atm.gs = float(md.gs)
    atm.sat_p = dict(md.sat_p)
    atm.sat_mix = dict(md.sat_mix)
    atm.r_p = dict(md.r_p)
    atm.rho_p = dict(md.rho_p)
    atm.fix_sp_indx = dict(md.fix_sp_indx)
    atm.conden_min_lev = {}

    var.k_arr = np.asarray(rs.rate.k, dtype=np.float64)
    var.def_bin_min = float(rs.photo.def_bin_min)
    var.def_bin_max = float(rs.photo.def_bin_max)
    var.aflux_change = float(rs.photo_runtime.aflux_change) if rs.photo_runtime is not None else 0.0
    if int(rs.photo.sflux_top.shape[0]) > 0:
        var.sflux_top = np.asarray(rs.photo.sflux_top)
    var.y = np.asarray(rs.step.y)
    var.y_prev = np.asarray(rs.step.y_prev)
    var.ymix = np.asarray(rs.step.ymix)
    var.t = float(rs.step.t)
    var.dt = float(rs.step.dt)
    var.longdy = float(rs.step.longdy)
    var.longdydt = float(rs.step.longdydt)
    var.y_ini = np.asarray(md.y_ini)

    from .state import _atom_arr_to_dict  # state imports this module

    var.atom_ini = _atom_arr_to_dict(np.asarray(rs.atoms.atom_ini), rs.atoms.atom_order)
    var.atom_loss = _atom_arr_to_dict(np.asarray(rs.atoms.atom_loss), rs.atoms.atom_order)
    var.atom_loss_prev = _atom_arr_to_dict(np.asarray(rs.atoms.atom_loss_prev), rs.atoms.atom_order)
    var.atom_sum = _atom_arr_to_dict(np.asarray(rs.atoms.atom_sum), rs.atoms.atom_order)
    var.atom_conden = dict.fromkeys(rs.atoms.atom_order, 0.0)
    if rs.photo_runtime is not None:
        var.tau = np.asarray(rs.photo_runtime.tau)
        var.aflux = np.asarray(rs.photo_runtime.aflux)
        var.sflux = np.asarray(rs.photo_runtime.sflux)
        var.dflux_u = np.asarray(rs.photo_runtime.dflux_u)
    if rs.photo_static is not None:
        var.nbin = int(rs.photo_static.nbin)
        var.bins = np.asarray(rs.photo_static.bins)
        var.dbin1 = float(rs.photo_static.dbin1)
        var.dbin2 = float(rs.photo_static.dbin2)
        var.J_sp = {}
        if int(rs.photo_static.cross_Jion.shape[0]) > 0:
            var.Jion_sp = {}
            var.ion_wavelen = {}
    var.Rf = dict(md.Rf)
    var.n_branch = dict(md.n_branch)
    var.ion_branch = dict(md.ion_branch)
    var.photo_sp = set(md.photo_sp)
    var.ion_sp = set(md.ion_sp)
    var.pho_rate_index = dict(md.pho_rate_index)
    var.ion_rate_index = dict(md.ion_rate_index)
    var.ion_br_ratio = dict(md.ion_br_ratio)
    var.charge_list = list(md.charge_list)
    var.conden_re_list = list(md.conden_re_list)

    para.count = int(rs.params.count)
    para.nega_count = int(rs.params.nega_count)
    para.loss_count = int(rs.params.loss_count)
    para.delta_count = int(rs.params.delta_count)
    para.delta = float(rs.params.delta)
    para.small_y = float(rs.params.small_y)
    para.nega_y = float(rs.params.nega_y)
    para.end_case = int(rs.params.end_case)
    para.termination_reason = int(rs.params.termination_reason)
    para.switch_final_photo_frq = bool(rs.params.switch_final_photo_frq)
    para.pic_count = int(rs.params.pic_count)
    para.where_varies_most = np.asarray(rs.params.where_varies_most, dtype=np.float64)
    para.fix_species_start = bool(rs.params.fix_species_start)
    para.start_time = float(md.start_time)

    return var, atm, para


class _Variables:
    """Private mutable scratch container used by the pre-loop pipeline."""

    def __init__(self) -> None:
        from .chem_funs import ni as _ni

        _vcfg = default_config()
        _nz = _vcfg.nz

        self.k: dict[int, np.ndarray] = {}
        self.k_arr: np.ndarray | None = None
        self.y = np.zeros((_nz, _ni))
        self.y_prev = np.zeros((_nz, _ni))
        self.ymix = np.zeros((_nz, _ni))
        self.y_ini = np.zeros((_nz, _ni))
        self.t = 0
        self.dt = _vcfg.dttry
        self.longdy = 1.0
        self.longdydt = 1.0

        self.atom_ini: dict[str, float] = {}
        self.atom_sum: dict[str, float] = {}
        self.atom_loss: dict[str, float] = {}
        self.atom_loss_prev: dict[str, float] = {}
        self.atom_conden: dict[str, float] = {}

        self.Rf: dict[int, str] = {}
        # Master's `ReadRate.read_rate` fills these when the oracle tests run it on this container (op.py:66); no code
        # here reads them.
        self.Rindx: dict = {}
        self.a: dict = {}
        self.n: dict = {}
        self.E: dict = {}
        self.a_inf: dict = {}
        self.n_inf: dict = {}
        self.E_inf: dict = {}
        self.k_fun: dict = {}
        self.k_inf: dict = {}
        self.kinf_fun: dict = {}
        self.k_fun_new: dict = {}
        self.photo_sp: set[str] = set()
        self.pho_rate_index: dict[tuple[str, int], int] = {}
        self.n_branch: dict[str, int] = {}
        self.ion_rate_index: dict[tuple[str, int], int] = {}
        self.ion_branch: dict[str, int] = {}
        self.ion_wavelen: dict[str, np.ndarray] = {}
        self.ion_br_ratio: dict[tuple[str, int], np.ndarray] = {}
        self.charge_list: list[str] = []
        self.ion_sp: set[str] = set()

        self.aflux_change = 0.0

        from .state import load_stellar_flux  # state imports this module

        self.def_bin_min, self.def_bin_max = load_stellar_flux(_vcfg)
        self.conden_re_list: list[int] = []


class _AtmData:
    """Private mutable scratch container for atmosphere-structure arrays."""

    def __init__(self) -> None:
        from .chem_funs import ni as _ni, spec_list as _spec_list

        _vcfg = default_config()
        _nz = _vcfg.nz
        from .atm_setup import surface_gravity as _surface_gravity

        self.pco = np.logspace(np.log10(_vcfg.P_b), np.log10(_vcfg.P_t), _nz)
        self.pico = np.empty(_nz + 1)
        self.dz = np.zeros(_nz)
        self.dzi = np.zeros(_nz - 1)
        self.zco = np.zeros(_nz + 1)
        self.zmco = np.empty(_nz)
        self.Tco = np.empty(_nz)
        self.Kzz = np.zeros(_nz - 1)
        self.vz = np.zeros(_nz - 1)
        self.M = np.empty(_nz)
        self.n_0 = np.empty(_nz)
        self.Hp = np.empty(_nz)
        self.mu = np.empty(_nz)
        self.ms = np.empty(_ni)
        self.Dzz = np.zeros((_nz - 1, _ni))
        self.Dzz_cen = np.zeros((_nz, _ni))
        self.vm = np.zeros((_nz - 1, _ni))
        self.vs = np.zeros((_nz - 1, _ni))
        self.alpha = np.zeros(_ni)
        self.gs = _surface_gravity(_vcfg)
        self.g = np.zeros(_nz)

        self.top_flux = np.zeros(_ni)
        self.bot_flux = np.zeros(_ni)
        self.bot_vdep = np.zeros(_ni)
        self.bot_fix_sp = np.zeros(_ni)

        self.sat_p: dict[str, np.ndarray] = {}
        self.sat_mix: dict[str, np.ndarray] = {}
        self.conden_min_lev: dict[str, int] = {}

        self.gas_indx = [_ for _ in range(_ni) if _spec_list[_] not in _vcfg.non_gas_sp]

        self.fix_sp_indx = {}
        for sp in _vcfg.fix_species:
            self.fix_sp_indx[sp] = np.arange(
                _spec_list.index(sp),
                _spec_list.index(sp) + _ni * _nz,
                _ni,
            )

        if _vcfg.use_ion:
            self.fix_e_indx = np.arange(
                _spec_list.index("e"),
                _spec_list.index("e") + _ni * _nz,
                _ni,
            )

        self.r_p, self.rho_p = {}, {}
        if _vcfg.use_condense:
            for sp in _vcfg.r_p:
                self.r_p[sp] = _vcfg.r_p[sp]
            for sp in _vcfg.rho_p:
                self.rho_p[sp] = _vcfg.rho_p[sp]
