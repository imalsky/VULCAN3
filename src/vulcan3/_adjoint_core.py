"""The fixed-point adjoint: body map, deflation and LGMRES. Method: docs/method.md."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .conden import CondenStatic, apply_h2o_relax_jax, apply_nh3_relax_jax, update_conden_rates
from .jax_step import _SOLVER, AtmStatic, jax_ros2_step
from .phy_const import UNDERFLOW_DENOM as _UNDERFLOW_DENOM

if TYPE_CHECKING:
    from .chem import NetworkArrays

__all__ = [
    "BODY_MAP_DT",
    "LGMRES_CYCLES",
    "LGMRES_INNER_M",
    "LGMRES_MAXITER",
    "LGMRES_OUTER_K",
    "LGMRES_RTOL",
    "N_SOLVES_DEFAULT",
    "SOLVER_MAP",
    "BodyTerms",
    "PhotoRecomputeArg",
]

SOLVER_MAP = "renorm"

PhotoRecomputeArg = Callable[[jnp.ndarray], jnp.ndarray] | None

BODY_MAP_DT = 1e7

# Hard guard: above this the implicit step goes near-singular; refuse rather than return a silently divergent gradient.
_BODY_MAP_DT_MAX = 1e10

# Ensemble size: the returned gradient is the mean over this many solves with ulp-perturbed RHS twins (deterministic,
# seeded); twin disagreement is reported as info["ensemble_spread"], the magnitude error bar. n_solves=1 is one
# unperturbed solve with no spread estimate.
N_SOLVES_DEFAULT = 3

# Relative RHS perturbation for ensemble twins: large enough to decorrelate the Krylov trajectories of a semi-converged
# solve, ~13 orders below the gradient signal so the exact solution is unchanged.
_TWIN_PERTURB = 1e-13

# Entries in the ranked reports: the ensemble spread and pair sums (top reactions by |mean|), the audit's worst cells.
_TOP_K = 10

# Warn above this ensemble spread (max over the top reactions by |mean| of (max-min)/|mean|): the twins disagree on the
# reactions one would report; treat magnitudes as ranking weights.
_SPREAD_WARN = 0.15

# scipy.sparse.linalg.lgmres inner Krylov dimension (the HD189 calibration in
# tests/test_steady_state_reaction_sensitivity.py).
LGMRES_INNER_M = 250

# Augmentation vectors carried across restarts; the LGMRES knob that fixes restarted-GMRES oscillation on this
# indefinite operator.
LGMRES_OUTER_K = 40

# Outer LGMRES iterations (scipy `maxiter`, each up to `inner_m` inner steps) per warm-start cycle; the solve is chunked
# into cycles with per-cycle x0 warm-start (the validated configuration).
LGMRES_MAXITER = 4

# Number of warm-start cycles (the HD189 calibration).
LGMRES_CYCLES = 8

# Relative-residual target; tighter buys nothing once finite-tolerance state mismatch dominates.
LGMRES_RTOL = 1e-12

# Warn above this median relative LGMRES residual across the ensemble (the median ignores a single wandering twin).
_ADJOINT_RESID_WARN = 0.2

# Warn above this body-map fixed-point error: y_star is off the steady-state manifold of the chosen map.
_FP_ERR_WARN = 1e-2

# Minimum |R_jj| of the unit-column QR in _conserved_null_basis; below it the atom columns are (near-)dependent and the
# basis is refused.
_NULL_BASIS_RANK_TOL = 1e-10

# Production sits ~3 orders below this; O(1) means a deflated direction is not null (e.g. open boundary fluxes) and the
# deflation is corrupting the solve.
_NULL_QUALITY_WARN = 1e-3


def _warn_poor_convergence(resid: float, fp_err: float, spread: float = 0.0, null_quality: float = 0.0) -> None:
    """Warn on a poor adjoint solve, even when the caller ignores `info`.

    An under-converged solve, a loose fixed point, twin disagreement or a non-null deflation basis.
    """
    if null_quality > _NULL_QUALITY_WARN:
        warnings.warn(
            f"steady_state_reaction_sensitivity: null_quality {null_quality:.2e} "
            f"exceeds {_NULL_QUALITY_WARN:.0e}: a deflated direction is not "
            "actually null, so the deflation is corrupting the adjoint solve. "
            "Check the boundary conditions (open fluxes break the conservation "
            "null space) before trusting the magnitudes.",
            stacklevel=3,
        )
    if fp_err > _FP_ERR_WARN:
        warnings.warn(
            "steady_state_reaction_sensitivity: y_star is not a tight fixed point "
            f"of the body map (fp_err={fp_err:.2e} > {_FP_ERR_WARN:.0e}); the adjoint "
            "is evaluated off the steady-state manifold and the gradient may be "
            "unreliable. Converge y_star tighter (or lower body_dt).",
            stacklevel=3,
        )
    if resid > _ADJOINT_RESID_WARN:
        warnings.warn(
            f"steady_state_reaction_sensitivity: median LGMRES relative residual {resid:.2e} "
            f"exceeds {_ADJOINT_RESID_WARN:.0e}: the solve is in the stagnation "
            "regime observed on closed columns (dominant-reaction magnitudes "
            "bounce ~+/-25% around FD there; sign and ranking remain robust). "
            "Treat magnitudes as "
            "ranking weights only. More cycles do not reliably reduce the "
            "residual; scan body_dt for a lower-residual regime (see "
            "BODY_MAP_DT) and check null_quality.",
            stacklevel=3,
        )
    if spread > _SPREAD_WARN:
        warnings.warn(
            f"steady_state_reaction_sensitivity: ensemble spread {spread:.2e} "
            f"exceeds {_SPREAD_WARN:.0e}: the perturbation twins disagree on the "
            "dominant-reaction magnitudes, so the solve is trajectory-sensitive "
            "at this body_dt. The MEAN is still the best estimate and the "
            "ranking is robust; for tighter magnitudes scan body_dt for a "
            "lower-residual regime.",
            stacklevel=3,
        )


def _check_body_dt(body_dt: float) -> None:
    """Refuse body_dt outside (0, _BODY_MAP_DT_MAX].

    0 makes the implicit step the identity (NaN); < 0 integrates backwards and
    returns a finite but wrong sensitivity; above the max the step goes
    near-singular. NaN and inf fail the range test too.
    """
    if not 0.0 < body_dt <= _BODY_MAP_DT_MAX:
        raise ValueError(
            f"body_dt={body_dt!r} is outside (0, {_BODY_MAP_DT_MAX:.0e}]: <= 0 or "
            "non-finite has no valid probe step; above the max is the "
            f"near-singular danger zone. Default {BODY_MAP_DT:.0e}."
        )


class BodyTerms(NamedTuple):
    """Optional per-step runner processes for the adjoint body map, beyond `ros2_step (+ renorm) (+ photo)`.

    Built by `make_body_terms` from a finished runner; all fields default-inactive (None, or False for
    `hydro_partial`).

    `conden_static` enables the in-window condensation composite (conden/evap
    k-rows recomputed from y each application -- the dk/dy feedback -- plus
    the H2O/NH3 relax kernels at the probe body_dt). `gas_mask`/`hydro_partial`
    switch the hydrostatic rebalance to the runner's gas-only-denominator,
    non-gas-passthrough form. `fix_mask`/`fix_y` reproduce the fix_species
    regime (pinned rows become constants of the map). `bot_idx`/`bot_val` are
    the layer-0 Dirichlet pins, applied after the balance; the runner sets
    `use_fix_sp_bot` and the H2/He pin on the step output, before the clip and
    the balance.
    """

    conden_static: CondenStatic | None = None
    gas_mask: jnp.ndarray | None = None  # (ni,) bool
    hydro_partial: bool = False
    fix_mask: jnp.ndarray | None = None  # (nz, ni) bool
    fix_y: jnp.ndarray | None = None  # (nz, ni)
    bot_idx: jnp.ndarray | None = None  # (m,) int32
    bot_val: jnp.ndarray | None = None  # (m,)


def _clip_dead_mask(G: Any, cfg: Any) -> np.ndarray:
    """Cells where the runner's per-step zero-clip would not be the identity.

    Applied to a candidate post-step state `G` (number density, cm^-3), the
    same rule as `_step_control._clip_prologue`:

        y < pos_cut and y >= nega_cut          -> 0     (small/negative cut)

    A value below nega_cut is not clipped there (the step is rejected), so it
    stays in the defect scan.
    """
    G = np.asarray(G)
    pos_cut = float(cfg.pos_cut)
    nega_cut = float(cfg.nega_cut)
    return (G < pos_cut) & (G >= nega_cut)


# size-limit-ok: the body map closes over one set of adjoint inputs
def _make_body_map(
    y_star: Any, k_arr: jnp.ndarray, atm: Any, net: Any, body_dt: Any, photo_recompute_k: Any, body_terms: Any = None
) -> Any:
    """Build the one-step body map `G(y)` the adjoint linearizes (unjitted).

    Returns `(apply_post_map, body_map, body_map_k, step_fn)`:
    `apply_post_map(sol, M_col=None)` is the shared post-step (plain renorm or
    the conden-aware composite; the `M_col` override is for parameter maps
    where M depends on the input), `body_map_k` is the k-linearization at
    `y_star`, `step_fn(y, k, atm)` the pinned Ros2 step for parameter maps
    that rebuild `atm`.

    SINGLE definition of the map: both sensitivity entry points and
    `audit_adjoint_scope` build from here, so the audited map is the solved
    one.
    """
    t = body_terms
    has_terms = t is not None and (
        t.conden_static is not None or t.gas_mask is not None or t.fix_mask is not None or t.bot_idx is not None
    )
    # Per-layer total density for the runner's post-step hydrostatic renorm.
    M_col_default = atm.M[:, None]
    dt64 = jnp.float64(body_dt)

    def apply_post_map(sol: jnp.ndarray, M_col: Any = None) -> Any:
        # The runner's rebalance, so y_star is a tight fixed point (SOLVER_MAP).
        M_c = M_col_default if M_col is None else M_col
        if not has_terms:
            return M_c * sol / jnp.sum(sol, axis=1, keepdims=True)
        # Runner composite (body_fn, clip omitted as identity-a.e.): gas-only
        # ymix denominator, relax kernels on the post-step state, partial
        # balance (non-gas species bypass the rebalance), layer-0 pins (body_fn
        # sets the fix_sp_bot and H2/He pins before the clip).
        gas = t.gas_mask
        if gas is None:
            ymix = sol / jnp.sum(sol, axis=1, keepdims=True)
        else:
            gsum = jnp.sum(jnp.where(gas[None, :], sol, 0.0), axis=1, keepdims=True)
            ymix = jnp.where(gsum > 0, sol / gsum, 0.0)
        y2 = sol
        if t.conden_static is not None:
            y2, ymix = apply_h2o_relax_jax(y2, ymix, dt64, t.conden_static)
            y2, ymix = apply_nh3_relax_jax(y2, ymix, dt64, t.conden_static)
        balanced = M_c * ymix
        if t.hydro_partial and gas is not None:
            balanced = jnp.where(gas[None, :], balanced, y2)
        if t.bot_idx is not None:
            balanced = balanced.at[0, t.bot_idx].set(t.bot_val)
        return balanced

    def step_fn(y: jnp.ndarray, k_use: Any, atm_use: Any) -> Any:
        # Reverse mode through the step: the dense stage operator
        # (`matrix_free=False`, jax_step._ros2_stages).
        # fix_species regime: pin inside the step (pinned rows zeroed off the diagonal, RHS zeroed)
        # then
        # overwrite with the pinned values, so pinned rows are constants of
        # the map (identity rows of I - dG/dy), not singular pass-throughs.
        if t is not None and t.fix_mask is not None:
            sol, _ = jax_ros2_step(y, k_use, dt64, atm_use, net, fix_mask=t.fix_mask, matrix_free=False)
            return jnp.where(t.fix_mask, t.fix_y, sol)
        sol, _ = jax_ros2_step(y, k_use, dt64, atm_use, net, matrix_free=False)
        return sol

    # With photo_recompute_k the photolysis rows are rebuilt from y each
    # application (the runner's own two-stream RT), so the y-VJP carries
    # dJ/dy; update_conden_rates does the same for the conden rows
    # (rate ~ y - y_sat). Folding the recomputes in from the incoming y is
    # exact for the fixed-point adjoint (incoming == post-step at the fixed
    # point; the one-step lag drops out of the coupled-system algebra).
    def body_map(y: jnp.ndarray) -> Any:
        k_use = photo_recompute_k(y) if photo_recompute_k is not None else k_arr
        if t is not None and t.conden_static is not None:
            k_use = update_conden_rates(k_use, y, t.conden_static)
        return apply_post_map(step_fn(y, k_use, atm))

    # k-linearization at y_star: k stays free (photo/conden rows not
    # recomputed -- a row entry means "perturb this rate"; the y-operator
    # above carries the state feedback).
    def body_map_k(k: Any) -> Any:
        return apply_post_map(step_fn(y_star, k, atm))

    return apply_post_map, body_map, body_map_k, step_fn


def _safe_inv_y(y_star: jnp.ndarray) -> jnp.ndarray:
    """Elementwise 1/y* with exact zeros mapped to 0, not inf.

    Closed columns clip trace species to 0.0; unmasked,
    the 1/y* log-scaling would poison the whole adjoint with NaN. A zeroed
    species becomes an identity row of the log operator (cotangent left
    untouched), the correct leading-order behavior.
    """
    pos = y_star > 0.0
    return jnp.where(pos, 1.0 / jnp.where(pos, y_star, 1.0), 0.0)


def _conserved_null_basis(y_star: jnp.ndarray, compo_array: jnp.ndarray, dz: jnp.ndarray) -> jnp.ndarray:
    """Orthonormal basis Q of the log-space conserved-mass null space.

    For each tracked element e, `sum_i compo[i,e]*dz*y_i` is conserved; its
    log-space gradient is `c_e[z,i] = compo[i,e]*dz[z]*y*[z,i]`. Stack the
    active elements, unit-normalize, QR -> the deflation projector basis.
    A (near-)zero `|R_jj|` raises: unpivoted QR maps a rank-deficient stack
    to arbitrary directions outside span(C), silently corrupting the deflation.

    Shapes: y_star (nz, ni), compo_array (ni, n_atoms), dz (nz,) -> Q (nz*ni, n_e).
    Host-side, one-shot, off the hot path.
    """
    y_np = np.asarray(y_star)
    compo_np = np.asarray(compo_array)
    dz_np = np.asarray(dz)
    atom_cols = np.where(compo_np.sum(axis=0) > 0)[0]
    if atom_cols.size == 0:
        raise ValueError(
            "compo_array has no populated atom columns; cannot build the "
            "conserved-mass null space. Pass composition.compo_array[:ni]."
        )
    cols = [(y_np * (compo_np[:, e][None, :] * dz_np[:, None])).ravel() for e in atom_cols]
    C = np.stack(cols, axis=1)
    norms = np.linalg.norm(C, axis=0)
    if np.any(norms == 0.0):
        dead = [int(atom_cols[j]) for j in np.where(norms == 0.0)[0]]
        raise ValueError(
            f"conserved-mass null basis: atom columns {dead} are all-zero "
            "(every carrier of that atom has y* == 0), so their conservation "
            "directions are degenerate. Drop those atoms from compo_array."
        )
    # Unit columns so |diag(R)| below directly measures column independence.
    C = C / norms
    Q, R = np.linalg.qr(C)
    rdiag = np.abs(np.diag(R))
    if np.any(rdiag < _NULL_BASIS_RANK_TOL):
        dep = [int(atom_cols[j]) for j in np.where(rdiag < _NULL_BASIS_RANK_TOL)[0]]
        raise ValueError(
            f"conserved-mass null basis is rank-deficient: atom columns {dep} "
            f"are (near-)linearly dependent on the others (|R_jj| < "
            f"{_NULL_BASIS_RANK_TOL:.0e}). Unpivoted QR would emit arbitrary "
            "directions outside span(C) and the deflation would corrupt the "
            "adjoint solve. Drop the dependent atom(s) from compo_array."
        )
    return jnp.asarray(Q)


def _lgmres_solve(matvec: Callable[[np.ndarray], np.ndarray], bvec: np.ndarray) -> np.ndarray:
    """Solve A x = b with scipy LGMRES (the LGMRES_* constants), chunked over warm-start cycles.

    Host-side scipy because JAX exposes no LGMRES (its restarted `gmres`
    oscillates on this indefinite operator); each matvec is one host<->device
    round trip, run once post-convergence, off the hot path.

    Returns the best-residual iterate across cycles, not the last: the
    warm-restart trajectory is not monotone on this operator (one extra
    matvec per cycle).
    """
    import scipy.sparse.linalg as spla

    n = bvec.shape[0]
    A_op = spla.LinearOperator((n, n), matvec=matvec, dtype=np.float64)
    x = np.zeros(n)
    x_best = x
    r_best = np.inf
    bnorm = max(float(np.linalg.norm(bvec)), _UNDERFLOW_DENOM)
    for _ in range(LGMRES_CYCLES):
        x, info = spla.lgmres(
            A_op,
            bvec,
            x0=x,
            rtol=LGMRES_RTOL,
            atol=0.0,
            inner_m=LGMRES_INNER_M,
            outer_k=LGMRES_OUTER_K,
            maxiter=LGMRES_MAXITER,
        )
        if info < 0:
            raise RuntimeError(
                f"scipy lgmres reported breakdown/illegal input (info={info}); "
                "the adjoint solve did not produce a usable solution. Check "
                "body_dt and the deflation basis (the null_quality diagnostic)."
            )
        r = float(np.linalg.norm(matvec(x) - bvec)) / bnorm
        if r < r_best:
            r_best = r
            x_best = x
        if info == 0:
            # Converged; cycles exist to continue an unconverged solve.
            break
    if not np.isfinite(r_best):
        raise RuntimeError(
            f"the adjoint solve produced no finite residual in {LGMRES_CYCLES} LGMRES cycles (last {r:g}, "
            f"|b| {bnorm:g}): the operator or right-hand side is non-finite, so there is no solution to return. "
            "Check y_star, k_arr and the loss cotangent for NaN."
        )
    return x_best


def _topk_ensemble_spread(g_stack: np.ndarray) -> float:
    """Twin disagreement on the components one would report.

    Max over the top `_TOP_K` by |mean| of (max-min)/|mean|. Weaker components carry larger relative bounce and
    would alarm on noise that never enters a ranking.
    """
    if g_stack.shape[0] <= 1:
        return 0.0
    g_mean = g_stack.mean(axis=0)
    order = np.argsort(np.abs(g_mean))[::-1]
    top = order[: min(_TOP_K, order.size)]
    top = top[np.abs(g_mean[top]) > 0.0]
    if not top.size:
        return 0.0
    width = g_stack[:, top].max(axis=0) - g_stack[:, top].min(axis=0)
    return float(np.max(width / np.abs(g_mean[top])))


def _host_network(net: Any) -> Any:
    """Return the import-locked parsed network, after checking that `net` has its reaction count.

    The body map's Ros2 step runs the import-time codegen RHS.
    """
    from . import chem_funs

    host = chem_funs.NETWORK
    if net.nr != host.nr:
        raise ValueError(
            f"net has {net.nr} reactions but the import-locked network "
            f"{host.network_path} has {host.nr}. The adjoint linearizes the "
            "import-time codegen RHS: pass chem_funs.NET_JAX, and select "
            "another network with $VULCAN3_NETWORK before the first import."
        )
    return host


def _active_photolysis_rows(k_arr: jnp.ndarray, net: Any) -> bool:
    """Whether `k_arr` contains nonzero photodissociation rows."""
    photo_rows = np.asarray(_host_network(net).is_photo, dtype=bool)
    if not photo_rows.any():
        return False
    return bool(np.any(np.asarray(k_arr)[photo_rows] != 0.0))


def _resolve_photo_recompute_k(
    photo_recompute_k: PhotoRecomputeArg, k_arr: jnp.ndarray, net: Any
) -> Callable[[jnp.ndarray], jnp.ndarray] | None:
    """Resolve the public photo-feedback argument for sensitivity calls.

    A callable is used as given. None means no photolysis feedback, which is exact only when `k_arr` has no active
    photolysis rows; with active rows, refuse rather than return a frozen-photolysis adjoint.
    """
    if callable(photo_recompute_k):
        return photo_recompute_k
    if photo_recompute_k is not None:
        raise ValueError(f"photo_recompute_k must be a callable or None; got {photo_recompute_k!r}.")
    if not _active_photolysis_rows(k_arr, net):
        return None
    raise ValueError(
        "photo_recompute_k=None on a column with active "
        "photolysis rows would freeze J and omit dJ/dy (11% on W39b OH+H2). "
        "Pass photo_recompute_k=make_photo_recompute_k(integ.photo_static, "
        "final_state)."
    )


def _guard_unmodeled_processes(y_star: Any, k_arr: jnp.ndarray, net: Any, body_terms: Any) -> None:
    """Refuse processes the body map would silently mistreat.

    Raises on nonzero ion rows (charge balance is in no body map) and on
    condensation fingerprints (nonzero conden rate rows, or a populated
    condensate) without conden/fix-species body terms. Row masks and species
    names come from the import-locked network (`_host_network`).
    """
    from . import chem_funs

    network = _host_network(net)
    k_np = np.asarray(k_arr)
    terms_conden = body_terms is not None and body_terms.conden_static is not None
    terms_pins = body_terms is not None and body_terms.fix_mask is not None

    ion_rows = np.asarray(network.is_ion, dtype=bool)
    if ion_rows.any() and bool(np.any(k_np[ion_rows] != 0.0)):
        raise NotImplementedError(
            "ion rows are active in k_arr: the runner pins the electron "
            "rows inside both Ros2 stages and applies a post-step charge "
            "balance, neither of which is in the adjoint body map, so "
            "ion-coupled sensitivities would be silently wrong. Ion "
            "columns are not supported by the steady-state adjoint; use "
            "forward-mode."
        )
    conden_rows = np.asarray(network.is_conden, dtype=bool)
    conden_rate_active = conden_rows.any() and bool(np.any(k_np[conden_rows] != 0.0))

    condensate_active = False
    y_np = np.asarray(y_star)
    for i, sp in enumerate(chem_funs.spec_list):
        # Condensate names in `conden.GAS_TO_CONDENSATE`: `_l_s`
        # (H2O/NH3/S2/S4/S8), `_l` (H2SO4), `_s` (C_s); testing only
        # `_l_s` would silently miss H2SO4_l and C_s.
        if sp.endswith(("_l_s", "_l", "_s")) and float(y_np[:, i].max()) > 0.0:
            condensate_active = True
            break

    if (conden_rate_active or condensate_active) and not (terms_conden or terms_pins):
        raise ValueError(
            "this state was converged with condensation active (nonzero "
            "conden rate rows and/or populated condensate species), but the "
            "adjoint body map carries no condensation terms, so the gradient "
            "would be silently wrong on conden-coupled rows/layers. Pass "
            "body_terms from make_body_terms(integ, converged_state, "
            "atm_static) (conden rate-recompute + relax kernels + partial "
            "balance in the conden window, or the fix_species pins after "
            "it), or use forward-mode."
        )


# size-limit-ok: one solve; its locals are the info dict
def _adjoint_solve_core(
    loss_fn: Callable[[jnp.ndarray], jnp.ndarray],
    y_star: jnp.ndarray,
    k_arr: jnp.ndarray,
    atm: AtmStatic,
    net: NetworkArrays,
    *,
    compo_array: jnp.ndarray,
    dz: jnp.ndarray,
    body_dt: float,
    photo_recompute_k: PhotoRecomputeArg,
    body_terms: BodyTerms | None,
    n_solves: int,
) -> Any:
    """Shared deflated-LGMRES adjoint solve on the body map.

    Resolves the photo feedback, refuses unmodeled processes, builds the body
    map (`_make_body_map`) and solves. Returns
    `(lams, info, (apply_post_map, body_map_k, step_fn))`: per-twin unscaled
    cotangents `lambda = z ./ y*` (contract against any parameter map as
    `dL/dp = lambda^T dG/dp`), the shared diagnostics dict, and the map pieces
    for the final contraction, which is the only part the two sensitivity
    entry points do differently.
    """
    if _SOLVER == "ffi":
        raise ValueError(
            "the reverse-mode adjoint does not run under VULCAN3_SOLVER=ffi: the kernel has no transpose. "
            "Run with VULCAN3_SOLVER=fast."
        )
    photo_recompute_k = _resolve_photo_recompute_k(photo_recompute_k, k_arr, net)
    _guard_unmodeled_processes(y_star, k_arr, net, body_terms)
    n_solves = max(1, int(n_solves))
    apply_post, body_map_raw, body_map_k, step_fn = _make_body_map(
        y_star, k_arr, atm, net, body_dt, photo_recompute_k, body_terms
    )

    nz, ni = y_star.shape
    inv_y = _safe_inv_y(y_star)
    body_map = jax.jit(body_map_raw)

    fp_err = float(
        jnp.max(jnp.abs(body_map(y_star) - y_star)) / jnp.maximum(jnp.max(jnp.abs(y_star)), _UNDERFLOW_DENOM)
    )
    _, vjp_Gy = jax.vjp(body_map, y_star)

    def a_eta(z: Any) -> Any:  # (I - dG/deta)^T in log-abundance coordinates
        return z - y_star * vjp_Gy(z * inv_y)[0]

    # Jit once, shared by the null-quality diagnostic and the LGMRES matvec,
    # so the expensive step-VJP XLA compile is paid once. `proj` stays
    # outside the jit (two small matmuls per matvec, negligible).
    a_eta_j = jax.jit(a_eta)

    # Conserved-mass deflation projector.
    Q = _conserved_null_basis(y_star, compo_array, dz)

    def proj(z: Any) -> Any:
        zf = z.reshape(-1)
        return (zf - Q @ (Q.T @ zf)).reshape(nz, ni)

    def deflated(z: Any) -> Any:
        return proj(a_eta_j(proj(z)))

    # How null the deflated directions are: max_e ||A_eta^T q_e|| (unit-norm
    # columns) relative to the operator's action on a fixed-seed random unit
    # direction (threshold: _NULL_QUALITY_WARN).
    null_defect = max(float(jnp.linalg.norm(a_eta_j(Q[:, e].reshape(nz, ni)))) for e in range(Q.shape[1]))
    rng = np.random.default_rng(0)  # fixed seed: the diagnostic is deterministic
    r = rng.normal(size=(nz, ni))
    r /= np.linalg.norm(r)
    op_scale = float(jnp.linalg.norm(a_eta_j(jnp.asarray(r))))
    null_quality = null_defect / max(op_scale, _UNDERFLOW_DENOM)

    # RHS: log-space cotangent of the loss, deflated.
    v = jax.grad(loss_fn)(y_star)
    b = proj(y_star * v)
    bvec = np.asarray(b).ravel()

    n_matvec = [0]

    def matvec(x: Any) -> Any:
        n_matvec[0] += 1
        return np.asarray(deflated(jnp.asarray(x.reshape(nz, ni)))).ravel()

    # Twin ensemble: a semi-converged Krylov endpoint is bit-level
    # trajectory-sensitive, so solving against seeded ulp-perturbed copies of
    # b turns that lottery into a measurable mean + spread. Twin 0 is the
    # unperturbed b.
    twin_rng = np.random.default_rng(0)
    lams = []
    resids = []
    for i in range(n_solves):
        b_i = bvec if i == 0 else bvec * (1.0 + _TWIN_PERTURB * twin_rng.standard_normal(bvec.shape[0]))
        x = _lgmres_solve(matvec, b_i)
        z = proj(jnp.asarray(x.reshape(nz, ni)))
        resids.append(
            float(
                jnp.linalg.norm(proj(a_eta_j(z)).reshape(-1) - jnp.asarray(b_i))
                / max(float(np.linalg.norm(b_i)), _UNDERFLOW_DENOM)
            )
        )
        lams.append(z * inv_y)  # unscaled cotangent lambda

    info = {
        "fp_err": fp_err,
        "null_quality": null_quality,
        "resid": max(resids),
        "resid_median": float(np.median(resids)),
        "resids": resids,
        "n_matvec": int(n_matvec[0]),
        "n_null": int(Q.shape[1]),
        "n_solves": n_solves,
        "body_dt": float(body_dt),
        "solver_map": SOLVER_MAP,
        "photo_feedback": photo_recompute_k is not None,
    }
    return lams, info, (apply_post, body_map_k, step_fn)
