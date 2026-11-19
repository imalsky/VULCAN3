# size-limit-ok: numpydoc sections on the public functions; the code is under the limit
"""JAX-native initial-abundance setup.

`compute_initial_abundance` returns a typed `IniAbunOutputs`; `InitialAbun` is the facade that mutates `data_var` and
`data_atm`. `eq_seed` is the end-to-end JAX equilibrium seed. The five `ini_mix` modes are in docs/method.md.

Units: CGS throughout: T in K, pressure in dyn cm^-2, number density in cm^-3, length in cm, time in s.
"""

from __future__ import annotations

import functools
import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    import types
    from collections.abc import Callable, Sequence

if TYPE_CHECKING:  # the exogibbs imports are function-local (pandas at import)
    from exogibbs.api.gas import EquilibriumOptions
    from exogibbs.thermo.models import ChemicalSetup

    from .config import Config

from . import chem_funs, gibbs, rates_jax
from ._paths import resolve_data_path
from .composition import (
    atom_list as _compo_atoms,
    compo,
    compo_array,
    compo_row,
    species,
)
from .config import default_config
from .phy_const import BAR_CGS
from .state import IniAbunOutputs

__all__ = [
    "InitialAbun",
    "column_atoms",
    "compute_atom_ini",
    "compute_initial_abundance",
    "element_vector",
    "eq_column",
    "eq_seed",
    "operator_column_weights",
    "ratio_indices",
    "read_abundances",
    "seed_elements",
]

logger = logging.getLogger(__name__)

_CFG = default_config()

# --- the equilibrium seed ---------------------------------------------------
# Network-frozen inputs (species, elements, NASA-9 coefficients) are resolved
# once, on first use (`_seed`). Everything the config owns is read from the
# config each function is given: a consumer traces the seed after setup has
# restored `_CFG` to the process default, so it passes the run's own config.


def _abundance_path(cfg: Config) -> Path:
    """Return the configured elemental-abundance preset file."""
    return resolve_data_path(str(cfg.fastchem_solar_abundance_file))


def _charged_species() -> list[str]:
    """Species with a non-zero electron count, in network order."""
    return [sp for sp in species if compo[compo_row.index(sp)]["e"] != 0]


def _condensate_species() -> set[str]:
    """Species produced by a condensation reaction.

    They have no gas-phase entry in the minimizer: the seed solves the gas
    equilibrium and the cold-trap clip (`_apply_condense`) runs after it.
    """
    net = chem_funs.NETWORK
    out: set[str] = set()
    for i in range(1, net.nr + 1):
        if not (net.is_conden[i] and net.is_forward[i]):
            continue
        for slot, sp_idx in enumerate(net.product_idx[i]):
            if net.product_stoich[i, slot] != 0.0:
                out.add(net.species[sp_idx])
    return out


def _photo_only_species() -> set[str]:
    """Species that appear only in photolysis or ionisation rows.

    With `use_photo` off VULCAN 2 drops those rows and these species (`make_chem_funs.py:74`). Left out of the
    seed, they start at 0, and no remaining row produces them.
    """
    net = chem_funs.NETWORK
    out = set(net.species)
    for i in np.flatnonzero(net.is_forward & ~net.is_photo & ~net.is_ion):
        for idx, stoich in ((net.reactant_idx, net.reactant_stoich), (net.product_idx, net.product_stoich)):
            out -= {net.species[sp_idx] for sp_idx, s in zip(idx[i], stoich[i], strict=True) if s != 0.0}
    return out


def _make_hvector(coeffs: np.ndarray) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """`T -> mu0(T)/(RT)` (K,) at the 1 bar standard state.

    That is the convention ExoGibbs's `gk = h + ln n_k - ln n_tot +
    ln(P/Pref)` expects, so the seed takes P in bar with `Pref=1.0`.
    `gibbs_sp_vector` is the same function the reverse rates use, evaluated
    on a one-layer grid because the minimizer calls `h` per layer. `coeffs`
    stays NumPy: the closure is cached in `_SEED`, so a `jnp.asarray` here
    would be a tracer whenever the first seed call happens inside a trace.
    """
    coeffs = np.asarray(coeffs, dtype=np.float64)

    def hvector(T: jnp.ndarray) -> jnp.ndarray:
        return rates_jax.gibbs_sp_vector(coeffs, jnp.atleast_1d(T))[:, 0]

    return hvector


def _build_seed_setup(use_photo: bool) -> tuple[ChemicalSetup, np.ndarray, tuple[str, ...]]:
    """Build the ExoGibbs setup for the loaded network's gas species.

    With `use_photo` off the photolysis-only species are left out, as VULCAN 2 leaves them out of the run.

    Every array in the returned setup is NumPy. The result is cached for the
    process, and the first seed call may happen inside a trace (a consumer
    calls `eq_seed` from a jitted forward model); a JAX array built there is a
    tracer that escapes into every later trace.
    """
    from exogibbs.thermo.models import ChemicalSetup

    excluded = _condensate_species() | set(_charged_species()) | (set() if use_photo else _photo_only_species())
    seed_idx = np.array([i for i, sp in enumerate(species) if sp not in excluded], dtype=np.int64)
    coeffs, present = gibbs.load_nasa9(tuple(species), rates_jax.thermo_dir(_CFG.network))
    missing = [species[i] for i in seed_idx if not present[i]]
    if missing:
        raise RuntimeError(
            f"the equilibrium seed needs NASA-9 data for every gas species of "
            f"{_CFG.network!r}, but thermo/NASA9/ has no file for: "
            f"{', '.join(missing)}. Add them, or initialize with "
            "ini_mix='const_mix'."
        )
    counts = np.asarray(compo_array)[seed_idx]
    elem_cols = [j for j, atom in enumerate(_compo_atoms) if atom != "e" and counts[:, j].sum() > 0.0]
    elements = tuple(_compo_atoms[j] for j in elem_cols)
    setup = ChemicalSetup(
        formula_matrix=np.ascontiguousarray(counts[:, elem_cols].T),
        hvector_func=_make_hvector(coeffs[seed_idx]),
        elements=elements,
        species=tuple(species[i] for i in seed_idx),
    )
    return setup, seed_idx, elements


# The built seed setup, keyed on the config's `use_photo`.
_SEED: dict[bool, tuple[ChemicalSetup, np.ndarray, tuple[str, ...]]] = {}


def _seed(use_photo: bool) -> tuple[ChemicalSetup, np.ndarray, tuple[str, ...]]:
    """Return `(setup, seed species indices, element names)`, built once per `use_photo` on first use.

    Lazy: it reads one `thermo/NASA9/<sp>.txt` per species, and a
    run with any other `ini_mix` must not pay for that at import.
    """
    if use_photo not in _SEED:
        _SEED[use_photo] = _build_seed_setup(use_photo)
    return _SEED[use_photo]


def seed_elements(cfg: Config) -> tuple[str, ...]:
    """Elements the seed of a run with config `cfg` solves for, in the order `b` is indexed."""
    return _seed(bool(cfg.use_photo))[2]


# Sequential warm start, layer by layer. ExoGibbs's name: it flips the
# arrays, so on VULCAN's grid (index 0 = bottom) the scan starts at the cold
# top. `scan_hot_from_top` shifts the seed past the HD189_vulcan3 gate's step
# cap.
_SEED_METHOD = "scan_hot_from_bottom"


def _seed_key(cfg: Config) -> tuple[float, int, bool]:
    """Return the run's solver controls and `use_photo`, the key the seed functions are built on."""
    return (float(cfg.fastchem_newton_tol), int(cfg.fastchem_newton_max_iter), bool(cfg.use_photo))


def eq_seed(Tco: jnp.ndarray, p_bar: jnp.ndarray, b: jnp.ndarray, cfg: Config) -> jnp.ndarray:
    """Equilibrium mixing ratios of one column.

    Species outside the seed (condensates, ions, and with `use_photo` off the
    photolysis-only species) are exactly zero. A column with an unconverged
    layer is NaN in every seed species, so no caller can use half a solution.
    The tangent is zero: the seed is an initial condition.

    The inputs are cast to float64: a float32 caller would otherwise give the
    minimizer a mixed-dtype loop carry.

    Parameters
    ----------
    Tco : jnp.ndarray [nz]
        Layer temperature, K.
    p_bar : jnp.ndarray [nz]
        Layer pressure, bar.
    b : jnp.ndarray [n_elements]
        Elemental abundances in `seed_elements(cfg)` order; only their ratios matter.
    cfg : Config
        The run's config: `use_photo`, `fastchem_newton_tol` and `fastchem_newton_max_iter`, read at trace time.

    Returns
    -------
    jnp.ndarray [nz, ni]
        Mixing ratios.
    """
    y: jnp.ndarray = _seed_fns(*_seed_key(cfg))[1](Tco, p_bar, b)
    return y


@functools.cache
def _seed_fns(
    tol: float, max_iter: int, use_photo: bool
) -> tuple[EquilibriumOptions, Callable[..., jnp.ndarray], Callable[..., jnp.ndarray]]:
    """`(options, seed, seed jitted)` for one set of solver controls and one `use_photo`, interned.

    Each function closes over its key, so a new key is a new function and a fresh trace: JAX keys its trace cache on
    the function and the argument shapes, not on the key. ExoGibbs keys its scan-body cache on `id(options)`, so the
    options are built once per key. Without the jit the host path re-traces the scan on every call.
    """
    from exogibbs.api.gas import EquilibriumOptions

    options = EquilibriumOptions(epsilon_crit=tol, max_iter=max_iter, method=_SEED_METHOD)

    @jax.custom_batching.custom_vmap
    def column(Tco: jnp.ndarray, p_bar: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        res, diag = _solve_profile(options, use_photo, Tco, p_bar, b)
        seed_idx = _seed(use_photo)[1]
        ok = jnp.all(diag["converged"]) & jnp.all(jnp.isfinite(res.x))
        y = jnp.zeros((Tco.shape[0], chem_funs.ni), dtype=jnp.float64)
        return y.at[:, jnp.asarray(seed_idx)].set(jnp.where(ok, res.x, jnp.nan))

    @column.def_vmap
    def lanes(axis_size: int, in_batched: list[bool], Tco: jnp.ndarray, p_bar: jnp.ndarray, b: jnp.ndarray) -> Any:
        """Batched seed: the unbatched program once per lane, in a `lax.map`.

        A vmapped minimizer computes a lane in another order than the solo one,
        so a lane's seed would depend on how many lanes are seeded with it, and
        the solver amplifies that to the convergence scale. The map calls the wrapped
        function, so a nested `vmap` also runs lane by lane.
        """
        args = tuple(
            x if batched else jnp.broadcast_to(x, (axis_size, *jnp.shape(x)))
            for x, batched in zip((Tco, p_bar, b), in_batched, strict=False)
        )
        return jax.lax.map(lambda a: column(*a), args), True

    @jax.custom_jvp
    def seed(Tco: jnp.ndarray, p_bar: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        y: jnp.ndarray = column(Tco, p_bar, b)
        return y

    @seed.defjvp
    def seed_jvp(
        primals: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], tangents: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Zero tangent: the seed is an initial condition, not a model output.

        ExoGibbs's kernel carries a `custom_vjp` and no forward rule, so a jvp
        must never reach it.
        """
        del tangents
        primal = seed(*primals)
        return primal, jnp.zeros_like(primal)

    return options, seed, jax.jit(seed)


def _solve_profile(
    options: EquilibriumOptions,
    use_photo: bool,
    Tco: np.ndarray | jnp.ndarray,
    p_bar: np.ndarray | jnp.ndarray,
    b: np.ndarray | jnp.ndarray,
) -> Any:
    """ExoGibbs `solve_profile` on one column: `(result, per-layer diagnostics)`."""
    from exogibbs.api.gas import solve_profile

    return solve_profile(
        _seed(use_photo)[0],
        jnp.asarray(Tco, dtype=jnp.float64),
        jnp.asarray(p_bar, dtype=jnp.float64),
        jnp.asarray(b, dtype=jnp.float64),
        Pref=1.0,
        options=options,
        return_diagnostics=True,
    )


def read_abundances(path: Path) -> dict[str, float]:
    """Parse a `SYMBOL log10(n_X/n_H)+12` preset into n_X/n_H, with H == 1.

    Parameters
    ----------
    path : Path
        The preset file; ``#`` starts a comment.

    Returns
    -------
    dict of str to float
        Element to its abundance relative to H.

    Raises
    ------
    ValueError
        If the file has no H row.
    """
    dex: dict[str, float] = {}
    with Path(path).open() as fh:
        for line in fh:
            row = line.split("#", 1)[0].split()
            if len(row) >= 2:
                dex[row[0]] = float(row[1])
    if "H" not in dex:
        raise ValueError(
            f"{path}: no H row. The preset is read as log10(n_X/n_H)+12, "
            "so hydrogen sets the scale and must be present."
        )
    return {sp: 10.0 ** (value - dex["H"]) for sp, value in dex.items()}


def _preset_vector(cfg: Config) -> np.ndarray:
    """`b` (E,) straight from the configured preset file."""
    path = _abundance_path(cfg)
    preset = read_abundances(path)
    elements = seed_elements(cfg)
    absent = [sp for sp in elements if sp not in preset]
    if absent:
        raise RuntimeError(
            f"{path} has no row for {', '.join(absent)}, which the "
            f"{cfg.network!r} network needs. Every element of the loaded "
            "network must be in the abundance preset."
        )
    return np.array([preset[sp] for sp in elements], dtype=np.float64)


def _base_vector(cfg: Config) -> np.ndarray:
    """`b` (E,) with the config's `<X>_H` applied on top of the preset.

    Only elements of `cfg.atom_list` other than H that the network carries
    are overridden. Helium comes from the file for every shipped atom list
    (`He_H` is a `const_lowT` knob).
    """
    b = _preset_vector(cfg)
    elements = seed_elements(cfg)
    for atom in cfg.atom_list:
        if atom != "H" and atom in elements:
            b[elements.index(atom)] = float(getattr(cfg, atom + "_H"))
    return b


def ratio_indices(names: Sequence[str], cfg: Config) -> jnp.ndarray:
    """Positions of `names` in the seed's element vector, resolved once.

    The array path (`element_vector`) takes these so a batched caller never
    resolves element names inside a traced function.

    Parameters
    ----------
    names : sequence of str [n]
        Element symbols.
    cfg : Config
        The run's config (`use_photo` selects the seed species).

    Returns
    -------
    jnp.ndarray [n]
        int32 positions in `seed_elements(cfg)`.

    Raises
    ------
    KeyError
        If a name is not an element of the loaded network.
    """
    elements = seed_elements(cfg)
    unknown = [sp for sp in names if sp not in elements]
    if unknown:
        raise KeyError(
            f"{unknown} are not elements of the loaded network ({', '.join(elements)}); setting them would do nothing."
        )
    return jnp.asarray([elements.index(sp) for sp in names], dtype=jnp.int32)


def element_vector(ratios: jnp.ndarray, idx: jnp.ndarray, cfg: Config) -> jnp.ndarray:
    """Return the elemental abundance vector with the elements at `idx` replaced by `ratios`. Traceable.

    Starts from `_element_vector(cfg)`, the host path's own no-override vector,
    so `use_solar` and the `cfg.<X>_H` overrides have the same precedence on
    both paths, and it is bit-identical to `_element_vector(cfg, {name: value,
    ...})` for the same elements: a replaced element also loses its `sequestered_frac`.

    Parameters
    ----------
    ratios : jnp.ndarray [n]
        Abundances relative to H.
    idx : jnp.ndarray [n]
        Their positions, from `ratio_indices`.
    cfg : Config
        The run's config: the preset, `use_solar`, `<X>_H`, `sequestered_frac` and `use_photo`.

    Returns
    -------
    jnp.ndarray [n_elements]
        `b` in `seed_elements(cfg)` order, relative to H.
    """
    scaled = jnp.asarray(ratios, dtype=jnp.float64) * jnp.asarray(_sequestered_scale(cfg))[idx]
    return jnp.asarray(_element_vector(cfg)).at[idx].set(scaled)


def _element_vector(cfg: Config, ratios: dict | None = None) -> np.ndarray:
    """`b` (E,) for the config `cfg`, host-side.

    The base is the same on both paths -- the preset when `use_solar` is
    true, else the preset with `cfg.<X>_H` applied -- and `ratios` replaces
    entries on top of it, exactly as `element_vector` does with `idx`. Last,
    each element X is scaled by 1 - `sequestered_frac[X]`.
    """
    b = _preset_vector(cfg) if cfg.use_solar is True else _base_vector(cfg)
    if ratios:
        elements = seed_elements(cfg)
        for sp, value in ratios.items():
            if sp not in elements:
                raise KeyError(f"ratios[{sp!r}] is not an element of the loaded network ({', '.join(elements)}).")
            b[elements.index(sp)] = float(value)
    sequestered: np.ndarray = b * _sequestered_scale(cfg)
    return sequestered


def _sequestered_fracs(cfg: Config) -> dict[str, float]:
    """Return `cfg.sequestered_frac` checked against `cfg.atom_list`: element to fraction of its bulk abundance.

    Builds no seed, so a run with another `ini_mix` is refused without paying for one.
    """
    fracs = cfg.sequestered_frac
    if not isinstance(fracs, dict):
        raise ValueError(  # noqa: TRY004 (a bad config value, as the other checks here)
            f"sequestered_frac={fracs!r} is not a mapping. Give a fraction per element, e.g. {{O: 0.2}}, or {{}}."
        )
    checked = {}
    for atom, value in fracs.items():
        if atom == "H":
            raise ValueError(
                f"sequestered_frac[{atom!r}]={value!r}: H is the reference element the abundances are relative to. "
                "Remove the key."
            )
        if atom not in cfg.atom_list:
            raise ValueError(
                f"sequestered_frac[{atom!r}]={value!r}: {atom!r} is not in atom_list {list(cfg.atom_list)}. Remove "
                "the key or run a network that carries the element."
            )
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value < 1.0:
            raise ValueError(
                f"sequestered_frac[{atom!r}]={value!r} is not a number in [0, 1): it is the fraction of the bulk "
                f"{atom} removed from the gas. Set a value in [0, 1)."
            )
        checked[atom] = float(value)
    return checked


def _sequestered_scale(cfg: Config) -> np.ndarray:
    """`(E,)` factor on the bulk elemental vector: 1 - `sequestered_frac[X]` on each element X named, 1 elsewhere."""
    elements = seed_elements(cfg)
    scale = np.ones(len(elements), dtype=np.float64)
    for atom, frac in _sequestered_fracs(cfg).items():
        if atom not in elements:
            raise ValueError(
                f"sequestered_frac[{atom!r}]={frac!r} has no effect: the {cfg.network!r} seed carries no {atom}. "
                "Remove the key."
            )
        scale[elements.index(atom)] = 1.0 - frac
    return scale


def _abun_lowT_residual(
    x: jnp.ndarray, O_H: jnp.ndarray, C_H: jnp.ndarray, He_H: jnp.ndarray, N_H: jnp.ndarray
) -> jnp.ndarray:
    """5-mol residual: H2 / H2O / CH4 / He / NH3.

    The sum constraint (f1) only includes x1..x4; NH3 (x5) participates
    in the H/C/O/He/N balance equations only.
    """
    x1, x2, x3, x4, x5 = x[0], x[1], x[2], x[3], x[4]
    coupling = 2.0 * x1 + 2.0 * x2 + 4.0 * x3 + 3.0 * x5
    f1 = x1 + x2 + x3 + x4 - 1.0
    f2 = x2 - coupling * O_H
    f3 = x3 - coupling * C_H
    f4 = x4 - coupling * He_H
    f5 = x5 - coupling * N_H
    return jnp.stack([f1, f2, f3, f4, f5])


def _jax_newton(
    residual_fn: Callable[..., jnp.ndarray],
    m0: jnp.ndarray,
    args: tuple[float | jnp.ndarray, ...],
    max_iter: int,
    tol: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Small dense Newton via `lax.while_loop` on residual norm; returns the iterate and its residual norm.

    Solves the 5-element `_abun_lowT_residual` system. The Jacobian is built with
    `jax.jacrev`; the linear solve is `jnp.linalg.solve` (5x5 dense). Production callers pass
    `max_iter` / `tol` from `_CFG.fastchem_newton_max_iter` and
    `_CFG.fastchem_newton_tol`. The loop also ends at `max_iter` or on a non-finite norm, so the caller compares the
    returned norm with `tol`: it is the loop's own exit value (infinite when no iteration ran).
    """
    jac_fn = jax.jacrev(residual_fn)

    def cond_fn(state: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
        _, residual_norm, iter_count = state
        return jnp.logical_and(residual_norm > tol, iter_count < max_iter)

    def body_fn(state: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        m, _, iter_count = state
        r = residual_fn(m, *args)
        J = jac_fn(m, *args)
        delta = jnp.linalg.solve(J, r)
        m_new = m - delta
        r_new = residual_fn(m_new, *args)
        return (m_new, jnp.linalg.norm(r_new), iter_count + 1)

    initial_state = (
        jnp.asarray(m0, dtype=jnp.float64),
        jnp.asarray(jnp.inf, dtype=jnp.float64),
        jnp.asarray(0, dtype=jnp.int32),
    )
    final_m, residual_norm, _ = jax.lax.while_loop(cond_fn, body_fn, initial_state)
    return final_m, residual_norm


def compute_atom_ini(y: jnp.ndarray, compo_arr: jnp.ndarray = compo_array) -> jnp.ndarray:
    """`atom_ini[a] = Σ_i compo[i,a] * Σ_z y[z,i]`, the unweighted atom sum over the column.

    Parameters
    ----------
    y : jnp.ndarray [nz, ni]
        Number densities, cm^-3.
    compo_arr : jnp.ndarray [ni, n_atoms], optional
        Atom counts per species.

    Returns
    -------
    jnp.ndarray [n_atoms]
        Atom sum per element, in the units of `y`.
    """
    return jnp.einsum("zi,ia->a", y, compo_arr)


def operator_column_weights(dz: jnp.ndarray) -> jnp.ndarray:
    """Cell weights (nz,) that the flux-form transport operator conserves.

    The diffusion divergence divides by the interface-centered spacing
    `dz_ave = 0.5*(dzi[j-1]+dzi[j])` (`jax_step`, mirroring master op.py), so
    with zero-flux boundaries the discrete transport invariant is
    `Σ_j n_j * w_j` with `w_0 = dzi[0]`, `w_j = 0.5*(dzi[j-1]+dzi[j])`,
    `w_{nz-1} = dzi[-1]` (not `Σ_j n_j * dz_j`). On a uniform grid w == dz.

    Parameters
    ----------
    dz : jnp.ndarray [nz]
        Layer thickness, cm.

    Returns
    -------
    jnp.ndarray [nz]
        Cell weights, cm.
    """
    dz = jnp.asarray(dz, dtype=jnp.float64)
    dzi = 0.5 * (dz[1:] + dz[:-1])
    return jnp.concatenate([dzi[:1], 0.5 * (dzi[:-1] + dzi[1:]), dzi[-1:]])


def column_atoms(y: jnp.ndarray, dz: jnp.ndarray, compo_arr: jnp.ndarray = compo_array) -> jnp.ndarray:
    """Operator-weighted atom column.

    `Σ_z w_z * y[z,i] * compo[i,a]` with `w = operator_column_weights(dz)`:
    the invariant the discrete transport conserves on the grid `dz`.

    Parameters
    ----------
    y : jnp.ndarray [nz, ni]
        Number densities, cm^-3.
    dz : jnp.ndarray [nz]
        Layer thickness, cm.
    compo_arr : jnp.ndarray [ni, n_atoms], optional
        Atom counts per species.

    Returns
    -------
    jnp.ndarray [n_atoms]
        Atom column per element, cm^-2.
    """
    w = operator_column_weights(dz)
    return jnp.einsum("z,zi,ia->a", w, jnp.asarray(y, dtype=jnp.float64), compo_arr)


def eq_column(pco: np.ndarray | jnp.ndarray, Tco: np.ndarray | jnp.ndarray, M: np.ndarray | jnp.ndarray) -> np.ndarray:
    """Equilibrium column in absolute number densities.

    Species outside the seed stay zero. Host-side: a batched caller drives
    `eq_seed` under `vmap` with `element_vector` instead.

    Parameters
    ----------
    pco : np.ndarray or jnp.ndarray [nz]
        Layer pressure, dyne/cm^2.
    Tco : np.ndarray or jnp.ndarray [nz]
        Layer temperature, K.
    M : np.ndarray or jnp.ndarray [nz]
        Total number density, cm^-3.

    Returns
    -------
    np.ndarray [nz, ni]
        Number densities, cm^-3.

    Raises
    ------
    RuntimeError
        If `use_ion` is on, or the seed does not converge in some layer.
    """
    if _CFG.use_ion is True:
        raise RuntimeError(
            "ini_mix='EQ' does not support use_ion=True: the equilibrium seed "
            "is gas-phase and neutral, with no electron balance. Initialize an "
            "ionized run with ini_mix='const_mix' or 'vulcan_ini'."
        )
    b = _element_vector(_CFG)
    elements = seed_elements(_CFG)
    Tco = np.asarray(Tco, dtype=np.float64)
    p_bar = np.asarray(pco, dtype=np.float64) / BAR_CGS
    logger.info(
        "Equilibrium seed from %s: %s",
        _abundance_path(_CFG),
        ", ".join(f"{sp}/H={b[elements.index(sp)]:.8g}" for sp in ("He", "C", "N", "O", "S") if sp in elements),
    )
    key = _seed_key(_CFG)
    options, _, seed_jit = _seed_fns(*key)
    ymix = np.asarray(seed_jit(jnp.asarray(Tco), jnp.asarray(p_bar), jnp.asarray(b)))
    if not np.isfinite(ymix).all():
        diag = _solve_profile(options, key[2], Tco, p_bar, b)[1]
        conv = np.asarray(diag["converged"])
        bad = np.flatnonzero(~conv)
        # A layer that stopped before the cap broke down (NaN residual), which
        # more iterations do not cure; seen below ~225 K on the shipped networks.
        capped = bool(np.asarray(diag["hit_max_iter"])[bad].all())
        remedy = (
            "Raise fastchem_newton_max_iter."
            if capped
            else "The minimizer broke down before the cap (a NaN residual, seen on "
            "columns colder than ~225 K); seed it with ini_mix='const_lowT' or "
            "'const_mix'."
        )
        raise RuntimeError(
            f"the equilibrium seed did not converge in "
            f"{_CFG.fastchem_newton_max_iter} iterations at "
            f"{bad.size} of {Tco.size} layers, tol={_CFG.fastchem_newton_tol:g}: "
            + "; ".join(f"layer {i} (T={Tco[i]:.1f} K, p={p_bar[i]:.3e} bar)" for i in bad[:5])
            + ". "
            + remedy
        )
    return ymix * np.asarray(M, dtype=np.float64)[:, None]


def _load_eq_y(data_atm: types.SimpleNamespace) -> tuple[np.ndarray, list[str]]:
    """Seed the atmosphere's own T-P at the config's elemental abundances."""
    return eq_column(data_atm.pco, data_atm.Tco, data_atm.M), []


def _load_vulcan_ini_y(data_atm: types.SimpleNamespace) -> tuple[np.ndarray, list[str]]:
    """Load `y` from a previous `.vul` file via pickle."""
    logger.info("Initializing with compositions from the previous run %s", _CFG.vul_ini)
    with Path(resolve_data_path(_CFG.vul_ini)).open("rb") as handle:
        vul_data = pickle.load(handle)
    nz_ = len(data_atm.pco)
    y = np.zeros((nz_, chem_funs.ni), dtype=np.float64)
    prev_species = vul_data["variable"]["species"]
    prev_y = vul_data["variable"]["y"]
    for sp in species:
        if sp in prev_species:
            y[:, species.index(sp)] = prev_y[:, prev_species.index(sp)]
        else:
            logger.warning("%s not included in the previous run.", sp)
    return y, (_charged_species() if _CFG.use_ion is True else [])


def _load_table_y(data_atm: types.SimpleNamespace) -> tuple[np.ndarray, list[str]]:
    """Load `y` from a per-layer mixing-ratio text table."""
    table = np.genfromtxt(resolve_data_path(_CFG.vul_ini), names=True, dtype=None, skip_header=1)
    if not len(data_atm.pco) == len(table["Pressure"]):
        raise ValueError(f"{_CFG.vul_ini} has {len(table['Pressure'])} layers, the config's nz is {len(data_atm.pco)}.")
    nz_ = len(data_atm.pco)
    y = np.zeros((nz_, chem_funs.ni), dtype=np.float64)
    n_0 = np.asarray(data_atm.n_0)
    for sp in species:
        y[:, species.index(sp)] = n_0 * table[sp]
    return y, []


def _load_const_mix_y(data_atm: types.SimpleNamespace) -> tuple[np.ndarray, list[str]]:
    """Load `y` from `_CFG.const_mix` (a per-species mixing dict)."""
    logger.info("Initializing with constant (well-mixed): %s", _CFG.const_mix)
    nz_ = len(data_atm.pco)
    y = np.zeros((nz_, chem_funs.ni), dtype=np.float64)
    gas_tot = np.asarray(data_atm.M)
    for sp in _CFG.const_mix:
        y[:, species.index(sp)] = gas_tot * _CFG.const_mix[sp]
    return y, (_charged_species() if _CFG.use_ion is True else [])


# Newton initial guess for the H2/H2O/CH4/He/NH3 system (build_atm.py:35).
_LOWT_M0 = (0.9, 0.1, 0.0, 0.0, 0.0)


def _load_const_lowT_y(data_atm: types.SimpleNamespace) -> tuple[np.ndarray, list[str]]:
    """Solve the 5-mol H2/H2O/CH4/He/NH3 system via JAX Newton."""
    O_H = float(_CFG.O_H)
    C_H = float(_CFG.C_H)
    He_H = float(_CFG.He_H)
    N_H = float(_CFG.N_H)
    m0 = jnp.array(_LOWT_M0, dtype=jnp.float64)
    max_iter = int(_CFG.fastchem_newton_max_iter)
    tol = float(_CFG.fastchem_newton_tol)
    root, norm = _jax_newton(_abun_lowT_residual, m0, (O_H, C_H, He_H, N_H), max_iter=max_iter, tol=tol)
    ini_mol, residual = np.asarray(root), float(norm)
    # The loop also ends at the iteration cap or on a non-finite iterate; neither is a solution.
    if not (np.isfinite(residual) and residual <= tol):
        raise RuntimeError(
            f"ini_mix='const_lowT': the H2/H2O/CH4/He/NH3 Newton solve did not converge: residual norm "
            f"{residual:.3e} against fastchem_newton_tol={tol:g} after at most {max_iter} iterations "
            f"(O_H={O_H:g}, C_H={C_H:g}, He_H={He_H:g}, N_H={N_H:g}). Raise fastchem_newton_max_iter, or check "
            "the element ratios."
        )

    nz_ = _CFG.nz
    y = np.zeros((nz_, chem_funs.ni), dtype=np.float64)
    gas_tot = np.asarray(data_atm.M)
    h2_idx = species.index("H2")
    h2o_idx = species.index("H2O")
    ch4_idx = species.index("CH4")
    nh3_idx = species.index("NH3")
    he_idx = species.index("He")
    for i in range(nz_):
        y[i, h2_idx] = ini_mol[0] * gas_tot[i]
        y[i, h2o_idx] = ini_mol[1] * gas_tot[i]
        y[i, ch4_idx] = ini_mol[2] * gas_tot[i]
        y[i, nh3_idx] = ini_mol[4] * gas_tot[i]
        y[i, he_idx] = gas_tot[i] - np.sum(y[i, :])
    return y, []


_MODE_DISPATCH = {
    "EQ": _load_eq_y,
    "vulcan_ini": _load_vulcan_ini_y,
    "table": _load_table_y,
    "const_mix": _load_const_mix_y,
    "const_lowT": _load_const_lowT_y,
}


def _apply_condense(y: np.ndarray, data_atm: types.SimpleNamespace) -> np.ndarray:
    """Apply the `use_condense` initial-cold-trap clip.

    Mutates `data_atm.sat_mix` / `data_atm.conden_min_lev`. Returns the
    clipped `y` so the caller can recompute ymix. The `use_sat_surfaceH2O`
    bottom pin is built by the runner from `sat_mix`, not written here.
    """
    if _CFG.use_condense is not True:
        return y

    for sp in _CFG.condense_sp:
        sp_idx = species.index(sp)
        data_atm.sat_mix[sp] = data_atm.sat_p[sp] / data_atm.pco
        data_atm.sat_mix[sp] = np.minimum(1.0, data_atm.sat_mix[sp])

        if sp == "H2O":
            data_atm.sat_mix[sp] *= _CFG.humidity
            if _CFG.use_sat_surfaceH2O is True:
                logger.info(
                    "The fixed surface water is now reset by condensation and humidity to %s", data_atm.sat_mix[sp][0]
                )
                # The ymix write is overwritten by the final renormalisation;
                # only the y replacement survives, so just update y here.
                y[:, sp_idx] = data_atm.sat_mix[sp][0] * data_atm.n_0

        if _CFG.use_ini_cold_trap is True and _CFG.ini_mix != "table":
            if _CFG.use_sat_surfaceH2O is True:
                conden_bot = 0
            else:
                conden_bot = int(np.argmax(data_atm.n_0 * data_atm.sat_mix[sp] <= y[:, sp_idx]))
            sat_rho = data_atm.n_0 * data_atm.sat_mix[sp]
            conden_status = y[:, sp_idx] >= sat_rho
            y[:, sp_idx] = np.minimum(
                data_atm.n_0 * data_atm.sat_mix[sp],
                y[:, sp_idx],
            )
            if list(y[conden_status, sp_idx]):
                min_sat = np.amin(data_atm.sat_mix[sp][conden_status])
                conden_min_lev = np.where(data_atm.sat_mix[sp] == min_sat)[0][0]
                data_atm.conden_min_lev[sp] = conden_min_lev
                logger.info(
                    "%s condensed from nz = %s to the minimum level nz = %s (cold trap)", sp, conden_bot, conden_min_lev
                )
                y[conden_min_lev:, sp_idx] = data_atm.sat_mix[sp][conden_min_lev] * data_atm.n_0[conden_min_lev:]
    return y


def _compute_ymix(y: np.ndarray) -> np.ndarray:
    """Per-layer normalisation.

    Excludes condensed-out species when `use_condense=True` (matches master's `non_gas_sp` carve-out).
    """
    if _CFG.use_condense is True:
        exc_conden = [i for i in range(chem_funs.ni) if species[i] not in _CFG.non_gas_sp]
        ysum = np.sum(y[:, exc_conden], axis=1).reshape((-1, 1))
    else:
        ysum = np.sum(y, axis=1).reshape((-1, 1))
    return y / ysum


def compute_initial_abundance(data_atm: types.SimpleNamespace) -> IniAbunOutputs:
    """Run the configured `ini_mix` mode and return a typed pytree.

    Side effect: when `use_condense=True`, the legacy `data_atm` container
    is mutated (saturation profiles, cold-trap min level). The pytree carries
    the gas-phase composition only.

    Parameters
    ----------
    data_atm : types.SimpleNamespace
        Legacy atmosphere container after the T-P setup (and the saturation profiles, with `use_condense`).

    Returns
    -------
    IniAbunOutputs
        `y` and `y_ini` (cm^-3) and `ymix`, each [nz, ni], and the charged species.

    Raises
    ------
    ValueError
        If `ini_mix` is not a mode, `use_ion` is on and the network has no ions, or `sequestered_frac` names H or an
        element outside `atom_list`, holds a value outside [0, 1), or is not empty for a mode other than EQ.
    """
    mix = _CFG.ini_mix
    if mix not in _MODE_DISPATCH:
        raise ValueError(f"ini_mix={mix!r} is not a mode; choose one of {sorted(_MODE_DISPATCH)}.")
    if _sequestered_fracs(_CFG) and mix != "EQ":
        raise ValueError(
            f"sequestered_frac={_CFG.sequestered_frac!r} acts only on the elemental abundances of ini_mix='EQ', "
            f"and ini_mix={mix!r} does not read them. Set it to {{}} or use ini_mix='EQ'."
        )
    y, charge_list = _MODE_DISPATCH[mix](data_atm)
    y = _apply_condense(y, data_atm)
    ymix = _compute_ymix(y)

    if _CFG.use_ion is True:
        if not charge_list:
            raise ValueError("use_ion = True but the network with ions is not supplied.\n")
        if "e" in charge_list:
            charge_list = [c for c in charge_list if c != "e"]

    return IniAbunOutputs(
        y=jnp.asarray(y),
        ymix=jnp.asarray(ymix),
        y_ini=jnp.asarray(y),
        charge_list=tuple(charge_list),
    )


class InitialAbun:
    """Legacy-mutation facade matching `atm_setup.Atm`'s pattern.

    It captures `cfg.atom_list` (cfg's possibly reordered or subset atom set) for `ele_sum`.
    """

    def __init__(self) -> None:
        self.atom_list = _CFG.atom_list

    def ini_y(self, data_var: types.SimpleNamespace, data_atm: types.SimpleNamespace) -> types.SimpleNamespace:
        """Compute the initial abundance in place and return `data_var`.

        Writes `data_var.y`, `ymix` and `y_ini`, plus `charge_list` when `use_ion`.
        """
        outputs = compute_initial_abundance(data_atm)
        data_var.y = np.asarray(outputs.y)
        data_var.ymix = np.asarray(outputs.ymix)
        data_var.y_ini = np.asarray(outputs.y_ini)
        if _CFG.use_ion is True:
            data_var.charge_list = list(outputs.charge_list)
        return data_var

    def ele_sum(self, data_var: types.SimpleNamespace) -> types.SimpleNamespace:
        """Write per-atom initial totals onto `data_var.atom_ini` and return `data_var`.

        Zeroes `atom_sum`, `atom_loss` and `atom_conden`; reads `data_var.y` and looks up each atom's column by name,
        skipping those in `cfg.loss_ex`.
        """
        atoms_jax = compute_atom_ini(jnp.asarray(data_var.y))
        atoms_np = np.asarray(atoms_jax)
        loss_ex = list(_CFG.loss_ex)
        for atom in self.atom_list:
            if atom in loss_ex:
                continue
            col = _compo_atoms.index(atom)
            data_var.atom_ini[atom] = float(atoms_np[col])
            data_var.atom_sum[atom] = 0.0  # written each step
            data_var.atom_loss[atom] = 0.0
            data_var.atom_conden[atom] = 0.0
        return data_var
