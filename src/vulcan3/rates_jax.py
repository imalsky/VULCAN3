"""Rate-coefficient build: `T -> k_arr`, on the AD graph.

Setup and gradients run the same code: the forward forms (modified Arrhenius, Lindemann falloff with k_inf, bare 3-body,
and the one hardcoded Troe expression for `OH + CH3 + M -> CH3OH + M`), the Moses+2005 low-T caps, and the NASA-9 Gibbs
reverse-rate path. Photo, conden and ion slots are zero here and filled at runtime; the 3-body [M] factor is applied in
the chemistry RHS (it depends on the time-evolving sum(y)).

Output is `k[nr+1, nz]` with 1-based reaction indexing (row 0 unused).
:func:`setup_var_k` freezes it to NumPy on `var.k_arr`; a gradient calls
:func:`build_rate_array` directly, where the Arrhenius coefficients
(`rate_coeffs=`) and the NASA-9 table are differentiable inputs.

Units: CGS throughout: T in K, pressure in dyn cm^-2, number density in cm^-3, length in cm, time in s.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np

from ._paths import resolve_data_path
from .gibbs import _NASA9_BRANCH_T, CORR, load_nasa9
from .network import Network, parse_network

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "K_COLL",
    "K_eq_array",
    "apply_collision_limit",
    "apply_lowT_caps",
    "apply_photo_remove",
    "apply_remove_list",
    "build_rate_array",
    "compute_forward_k",
    "fill_reverse_k",
    "gibbs_sp_vector",
    "setup_var_k",
    "thermo_dir",
]

_SPECIAL_OH_CH3 = "OH + CH3 + M -> CH3OH + M"

# Clip on the Gibbs exponent before exp() in K_eq_array, just under the
# largest finite float64 exp() argument (log(finfo.max) = 709.78). A clipped K
# is finite, so k_rev = k_fwd/K is tiny but positive where an overflow would
# give 0; the clip keeps the jvp of k_fwd/K off inf/inf = NaN on cold columns.
_EXP_ARG_MAX = 709.0

# Moses+2005 low-T rate caps, master's lim_lowT_rates (exoclime@80f75b9
# op.py:320-340): at T <= T_max (K) the post-Lindemann forward slot takes the
# cap. CH3's cap is the Lindemann form k0 / (1 + k0 M / kinf) with
# kinf = 2.06e-10 T^-0.4 (op.py:322-328); the other two are constants
# (op.py:330-340).
_LOWT_CAP_CH3 = ("H + CH3 + M -> CH4 + M", 277.5, 6.0e-29)  # (rxn, T_max, k0)
_LOWT_CAP_CONST = (  # (rxn, T_max, k_cap)
    ("H + C2H4 + M -> C2H5 + M", 300.0, 3.7e-30),
    ("H + C2H5 + M -> C2H6 + M", 200.0, 2.49e-27),
)


def _thermo_dir(network_file: Any) -> Path:
    """`thermo/` beside the network file, else the packaged one.

    The rate build and the equilibrium seed read one NASA-9 table from one place.
    """
    beside = resolve_data_path(network_file).parent
    if (beside / "NASA9").exists():
        return beside
    return Path(__file__).resolve().parent / "thermo"


thermo_dir = _thermo_dir  # the public name; `_thermo_dir` is deprecated (docs/api.md)


def _arrhenius(a: Any, n: Any, E: Any, T: jnp.ndarray) -> Any:
    return a * T**n * jnp.exp(-E / T)


def _troe_OH_CH3(T: jnp.ndarray, M: jnp.ndarray) -> Any:
    """Hardcoded Troe form for OH + CH3 + M -> CH3OH + M.

    Visscher & Moses 2011 eqs 13-14 (log10 width; upstream op.py:204-205 uses ln) with the eq 24-26 fits of Jasper
    et al. 2007.
    """
    k0 = 1.932e3 * T**-9.88 * jnp.exp(-7544.0 / T) + 5.109e-11 * T**-6.25 * jnp.exp(-1433.0 / T)
    kinf = 1.031e-10 * T**-0.018 * jnp.exp(16.74 / T)
    Fc = 0.1855 * jnp.exp(-T / 155.8) + 0.8145 * jnp.exp(-T / 1675.0) + jnp.exp(-4531.0 / T)
    nn = 0.75 - 1.27 * jnp.log10(Fc)
    ff = Fc ** (1.0 / (1.0 + (jnp.log10(k0 * M / kinf) / nn) ** 2))
    return k0 / (1.0 + k0 * M / kinf) * ff


def _col(x: Any) -> Any:
    """(nr+1,) network array -> (nr+1, 1) jnp float column for broadcasting."""
    return jnp.asarray(np.asarray(x, dtype=np.float64))[:, None]


def compute_forward_k(
    net: Network,
    T: jnp.ndarray,
    M: jnp.ndarray,
    *,
    a: Any = None,
    n: int | None = None,
    E: Any = None,
    a_inf: Any = None,
    n_inf: Any = None,
    E_inf: Any = None,
) -> jnp.ndarray:
    """Vectorized forward rates. T, M are (nz,) arrays -> (nr+1, nz).

    Reverse slots (even indices) are zero here and filled by
    :func:`fill_reverse_k`; photo / ion / conden slots are
    zero and filled at runtime.

    The six Arrhenius/Lindemann coefficient arrays default to the network's
    static values; pass any as a (nr+1,) JAX array to differentiate the rates
    w.r.t. rate-coefficient uncertainty (`jvp`/`grad`). The one hardcoded Troe
    row (OH+CH3) is not overridable (its constants stay fixed).
    """
    if jnp.shape(M) != jnp.shape(T):
        raise ValueError(
            f"T and M must be the same shape; got T {jnp.shape(T)} and M "
            f"{jnp.shape(M)}. A length-1 M would broadcast silently over the "
            "column and give every layer the bottom layer's density."
        )
    T = jnp.asarray(T)[None, :]  # (1, nz)
    M = jnp.asarray(M)[None, :]

    def _ovr(val: Any, default: Any) -> Any:
        return _col(default) if val is None else jnp.asarray(val)[:, None]

    a, n, E = _ovr(a, net.a), _ovr(n, net.n), _ovr(E, net.E)
    a_inf, n_inf, E_inf = (
        _ovr(a_inf, net.a_inf),
        _ovr(n_inf, net.n_inf),
        _ovr(E_inf, net.E_inf),
    )

    arr = _arrhenius(a, n, E, T)  # plain Arrhenius (a==0 -> 0)
    kinf = _arrhenius(a_inf, n_inf, E_inf, T)
    # k_inf == 0 is the zero-rate limit, not a low-pressure rate: with the
    # guard value alone the falloff row would read `arr / (1 + arr*M)`.
    kinf_safe = jnp.where(kinf > 0, kinf, 1.0)
    lindemann = jnp.where(kinf > 0, arr / (1.0 + arr * M / kinf_safe), 0.0)
    has_kinf = jnp.asarray(np.asarray(net.has_kinf, dtype=bool))[:, None]
    general = jnp.where(has_kinf, lindemann, arr)

    # Special handling: the one OH+CH3 Troe row; other "special" rows fall back to
    # plain Arrhenius (a==0 specials give 0 automatically).
    is_special = jnp.asarray(np.asarray(net.is_special, dtype=bool))[:, None]
    oh = np.zeros(net.nr + 1, dtype=bool)
    for i in range(1, net.nr + 1):
        if net.is_special[i] and net.Rf.get(i, "").strip() == _SPECIAL_OH_CH3:
            oh[i] = True
    oh_mask = jnp.asarray(oh)[:, None]
    troe = _troe_OH_CH3(T, M)
    special_val = jnp.where(oh_mask, troe, arr)
    k = jnp.where(is_special, special_val, general)

    # Keep only thermal forward slots; zero photo/ion/conden/non-forward.
    thermal_fwd = net.is_forward & ~net.is_photo & ~net.is_ion & ~net.is_conden
    keep = jnp.asarray(np.asarray(thermal_fwd, dtype=bool))[:, None]
    return jnp.where(keep, k, 0.0)


def apply_lowT_caps(net: Network, k_fwd: jnp.ndarray, T: jnp.ndarray, M: jnp.ndarray) -> jnp.ndarray:
    """Apply the three Moses+2005 low-T recombination caps.

    Callers gate on `cfg.use_lowT_limit_rates`, used for cool-atmosphere networks.

    Caps the three forward rows when present in `net`, gated by a temperature
    threshold via `jnp.where`. The cap branch is a T-independent constant for
    C2H4/C2H5 (so dk/dT = 0 below threshold) and a Lindemann form for CH3; the
    tangent is non-smooth only exactly at the threshold (a measure-zero kink),
    smooth on either side. `k_fwd` (nr+1, nz); `T`, `M` (nz,).
    """
    Tz = jnp.asarray(T)
    Mz = jnp.asarray(M)
    # Locate the cap rows on the host (Rf is static); first forward slot each.
    rows: dict[str, int] = {}
    for i in range(1, net.nr + 1, 2):
        if net.is_forward[i]:
            rows.setdefault(net.Rf.get(i, ""), i)

    k = k_fwd
    rxn, t_max, k0 = _LOWT_CAP_CH3
    i = rows.get(rxn)
    if i is not None:
        kinf = 2.06e-10 * Tz**-0.4
        cap = k0 / (1.0 + k0 * Mz / kinf)
        k = k.at[i].set(jnp.where(Tz <= t_max, cap, k[i]))
    for rxn, t_max, k_cap in _LOWT_CAP_CONST:
        i = rows.get(rxn)
        if i is not None:
            k = k.at[i].set(jnp.where(Tz <= t_max, k_cap, k[i]))
    return k


def gibbs_sp_vector(coeffs: Any, T: jnp.ndarray) -> jnp.ndarray:
    """g_sp/(RT) per species per layer. coeffs (ni,2,10), T (nz,) -> (ni,nz).

    `coeffs` may be a NumPy array (the static thermo) or a JAX array/tracer:
    `jnp.asarray` (not `np.asarray`) keeps it on the graph so reverse rates are
    differentiable w.r.t. the NASA-9 coefficients.
    """
    c = jnp.asarray(coeffs, dtype=jnp.float64)
    a_low, a_high = c[:, 0, :], c[:, 1, :]
    Tg = jnp.asarray(T)[None, :]

    def _g(a: Any) -> Any:
        return (
            -a[:, 0:1] / Tg**2
            + a[:, 1:2] * jnp.log(Tg) / Tg
            + a[:, 2:3]
            + a[:, 3:4] * Tg / 2.0
            + a[:, 4:5] * Tg**2 / 3.0
            + a[:, 5:6] * Tg**3 / 4.0
            + a[:, 6:7] * Tg**4 / 5.0
            + a[:, 8:9] / Tg
        ) - (
            -a[:, 0:1] / Tg**2 / 2.0
            - a[:, 1:2] / Tg
            + a[:, 2:3] * jnp.log(Tg)
            + a[:, 3:4] * Tg
            + a[:, 4:5] * Tg**2 / 2.0
            + a[:, 5:6] * Tg**3 / 3.0
            + a[:, 6:7] * Tg**4 / 4.0
            + a[:, 9:10]
        )

    mask_low = (jnp.asarray(T) < _NASA9_BRANCH_T)[None, :]
    return jnp.where(mask_low, _g(a_low), _g(a_high))


def K_eq_array(net: Network, gibbs_sp: jnp.ndarray, T: jnp.ndarray) -> jnp.ndarray:
    """Equilibrium constants per forward reaction (nr+1, nz). Vectorized.

    Slots without a thermal reverse (photo/ion/conden/radiative, beyond
    `stop_rev_indx`) are 1 -- a sentinel so the reverse-fill divide is safe.
    """
    nz = gibbs_sp.shape[1]
    g_pad = jnp.concatenate([gibbs_sp, jnp.zeros((1, nz))], axis=0)  # (ni+1, nz)
    r_idx = jnp.asarray(np.asarray(net.reactant_idx))  # (nr+1, S)
    p_idx = jnp.asarray(np.asarray(net.product_idx))
    r_st = jnp.asarray(np.asarray(net.reactant_stoich, dtype=np.float64))
    p_st = jnp.asarray(np.asarray(net.product_stoich, dtype=np.float64))

    reac = jnp.einsum("rs,rsz->rz", r_st, g_pad[r_idx])  # (nr+1, nz)
    prod = jnp.einsum("rs,rsz->rz", p_st, g_pad[p_idx])
    delta_n = (r_st.sum(axis=1) - p_st.sum(axis=1))[:, None]  # (nr+1, 1)
    Tg = jnp.asarray(T)[None, :]
    # Overflow guard; see _EXP_ARG_MAX.
    K_raw = jnp.exp(jnp.minimum(reac - prod, _EXP_ARG_MAX)) * (CORR * Tg) ** delta_n

    # Valid forward slots get a real K; everything else stays 1 (sentinel).
    nr = net.nr
    idx = np.arange(nr + 1)
    valid = net.is_forward & (idx + 1 < net.stop_rev_indx) & ~net.is_photo & ~net.is_ion & ~net.is_conden
    valid_m = jnp.asarray(np.asarray(valid, dtype=bool))[:, None]
    return jnp.where(valid_m, K_raw, 1.0)


def fill_reverse_k(net: Network, k_fwd: jnp.ndarray, K_eq: jnp.ndarray) -> jnp.ndarray:
    """Fill even (reverse) slots from forward rates / K_eq. (nr+1, nz).

    Reverse slots beyond `stop_rev_indx` stay as they come in (zero out of
    :func:`compute_forward_k`): photo/conden/ion/radiative have no thermal
    reverse. Their forward slots are left untouched.
    """
    nr = net.nr
    idx = np.arange(nr + 1)
    even_rev = (idx % 2 == 0) & (idx >= 2) & (idx < net.stop_rev_indx)
    partner = np.where(even_rev, idx - 1, 0)  # forward partner index
    even_rev_m = jnp.asarray(even_rev)[:, None]
    partner_j = jnp.asarray(partner)

    k_partner = k_fwd[partner_j]  # (nr+1, nz)
    K_partner = K_eq[partner_j]
    k_rev = jnp.where(K_partner > 0, k_partner / jnp.where(K_partner > 0, K_partner, 1.0), 0.0)
    return jnp.where(even_rev_m, k_rev, k_fwd)


K_COLL = 1e-9  # cm3 s^-1: the collision limit of apply_collision_limit (notes "Parity ledger")


def apply_collision_limit(net: Network, k: Any, M: jnp.ndarray) -> jnp.ndarray:
    """Scale each thermal forward / reverse pair by c = min(1, K_COLL / q) (notes "Parity ledger"; not in master).

    q is the larger of the two directions' bimolecular-equivalent rates, k M^(three-body) M^(m - 2) with m the slot's
    reactant molecules (stoichiometry, M excluded); a direction with m < 2 does not count. Both directions take the same
    c, so their ratio (K_eq) and the equilibrium are unchanged. It only acts where a rate extrapolated from its fitted
    range exceeds collision rates (the reverse of CH + M -> C + H + M and C2H + M -> C2 + H + M at low T).
    """
    m = np.asarray(net.reactant_stoich).sum(axis=1)
    expo = np.asarray(net.is_three_body, float) + m - 2.0  # power of M in q
    idx = np.arange(net.nr + 1)
    fwd = (idx % 2 == 1) & (idx + 1 < net.stop_rev_indx)  # forward slots with a thermal reverse
    elig = jnp.asarray(m >= 2)[:, None]
    q = k * M[None, :] ** jnp.asarray(expo)[:, None]
    r = jnp.where(elig, q / K_COLL, 0.0)
    fi = np.flatnonzero(fwd)
    rp = jnp.maximum(r[fi], r[fi + 1])  # the pair's larger ratio
    c = 1.0 / jnp.maximum(rp, 1.0)
    scale = jnp.ones_like(k).at[fi].set(c).at[fi + 1].set(c)
    return k * scale


def apply_remove_list(net: Network, k: Any, remove_list: Iterable[int] | None) -> Any:
    """Zero the rows in `remove_list`.

    No auto-pairing: passing a lone forward leaves its reverse intact.

    Accepts a NumPy or a JAX `k` and returns the same kind, so the on-graph
    build and the host post-photolysis pass share one implementation.
    """
    if not remove_list:
        return k
    rm = np.zeros(net.nr + 1, dtype=bool)
    for i in remove_list:
        idx = int(i)
        if 0 <= idx <= net.nr:
            rm[idx] = True
    if isinstance(k, np.ndarray):
        return np.where(rm[:, None], 0.0, k)
    return jnp.where(jnp.asarray(rm)[:, None], 0.0, k)


def build_rate_array(
    net: Network,
    T: jnp.ndarray,
    M: jnp.ndarray,
    nasa9_coeffs: np.ndarray,
    remove_list: Any = None,
    use_lowT_caps: bool = False,
    rate_coeffs: dict | None = None,
) -> jnp.ndarray:
    """End-to-end: forward(T) -> (lowT caps) -> reverse via Gibbs(T) -> collision limit -> remove.

    `T`, `M` are (nz,) arrays; `nasa9_coeffs` is the (ni,2,10) thermo table
    (differentiable -- it flows through the Gibbs reverse path); `remove_list`
    zeros those rows. Pass `use_lowT_caps=True` to match
    `cfg.use_lowT_limit_rates` on cool networks (off by default -- the hot
    benchmarks never trigger the caps). Pass `rate_coeffs` (a dict of any of
    `a/n/E/a_inf/n_inf/E_inf` as (nr+1,) JAX arrays) to differentiate w.r.t.
    Arrhenius rate-coefficient uncertainty; defaults to the network's values.

    Index 0 is unused (1-based reactions); reverse slots beyond
    `net.stop_rev_indx` are zero.
    """
    k_fwd = compute_forward_k(net, T, M, **(rate_coeffs or {}))
    if use_lowT_caps:
        k_fwd = apply_lowT_caps(net, k_fwd, T, M)
    g_sp = gibbs_sp_vector(nasa9_coeffs, T)
    K_eq = K_eq_array(net, g_sp, T)
    # `remove_list` is applied in its own pass, after the reverse fill, to
    # match master (no auto fwd/rev pairing).
    k = fill_reverse_k(net, k_fwd, K_eq)
    k = apply_collision_limit(net, k, M)
    return apply_remove_list(net, k, remove_list)


def _assert_reversible_thermo_present(net: Network, present: np.ndarray) -> None:
    """Raise if a species in a reversible reaction lacks NASA-9 thermo.

    `load_nasa9` leaves a missing species' coefficients zero, which corrupts
    `K_eq` and every reverse rate it enters. Master also raises
    `FileNotFoundError`: its generated chem_funs.py:2985 loads NASA-9 for every
    species at import, before `remove_list` is read (op.py:300, :314). Only
    species in a reaction below `stop_rev_indx` need thermo; condensate-,
    photo- and ion-only species may have no NASA-9 file.
    """
    needed: set[int] = set()
    last_rev = min(net.stop_rev_indx, net.nr + 1)
    for i in range(1, last_rev):
        for idx_arr, st_arr in (
            (net.reactant_idx, net.reactant_stoich),
            (net.product_idx, net.product_stoich),
        ):
            for slot in range(idx_arr.shape[1]):
                if st_arr[i, slot] == 0.0:
                    continue
                sp = int(idx_arr[i, slot])
                if 0 <= sp < net.ni:
                    needed.add(sp)
    missing = [net.species[j] for j in sorted(needed) if not bool(present[j])]
    if missing:
        raise FileNotFoundError(
            "Missing NASA-9 thermo file(s) for species used in reversible "
            f"reactions: {', '.join(missing)}. Each needs thermo/NASA9/<sp>.txt "
            "(a missing file silently zeros its Gibbs energy and corrupts the "
            "reverse rates). Add the file or delete the reactions from the "
            "network file; remove_list cannot help, the check runs before it."
        )


def setup_var_k(cfg: Any, var: Any, atm: Any) -> Network:
    """Parse network, load NASA-9 coeffs, freeze `var.k_arr`. Returns the Network.

    The host half of the setup contract: it resolves the files and pins the
    result to NumPy for the runner. The physics is :func:`build_rate_array`,
    the same call a temperature gradient makes. `np.array` (not `asarray`):
    `op_jax.compute_J` writes the photolysis rows into `var.k_arr` in place,
    and a view of a JAX buffer is read-only.
    """
    network = parse_network(str(resolve_data_path(cfg.network)))
    nasa9_coeffs, present = load_nasa9(network.species, _thermo_dir(cfg.network))
    _assert_reversible_thermo_present(network, present)
    var.k_arr = np.array(
        build_rate_array(
            network,
            jnp.asarray(np.asarray(atm.Tco, dtype=np.float64)),
            jnp.asarray(np.asarray(atm.M, dtype=np.float64)),
            nasa9_coeffs,
            remove_list=cfg.remove_list,
            use_lowT_caps=bool(cfg.use_lowT_limit_rates),
        ),
        dtype=np.float64,
    )
    return network


def apply_photo_remove(cfg: Any, var: Any, network: Network) -> None:
    """Re-apply `cfg.remove_list` after the photolysis rows of `var.k_arr` were overwritten.

    `compute_J` and `compute_Jion` write those rows.
    """
    var.k_arr = np.array(apply_remove_list(network, var.k_arr, cfg.remove_list), dtype=np.float64)
