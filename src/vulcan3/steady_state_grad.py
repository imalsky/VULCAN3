"""Reverse-mode sensitivities at the converged photochemical state.

Public names: `steady_state_reaction_sensitivity` (`dL/d(ln k_r)`), `steady_state_input_sensitivity` (`dL/dp`),
`audit_adjoint_scope`, `make_body_terms`, `make_photo_recompute_k`, `BODY_MAP_DT_CANDIDATES`, `SOLVER_MAP`.

Run `audit_adjoint_scope` on the converged run before trusting a gradient. Method and rules: docs/method.md. Recipe:
examples/grad_reverse_example.py.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from .phy_const import UNDERFLOW_DENOM as _UNDERFLOW_DENOM

if TYPE_CHECKING:
    from collections.abc import Callable

    from .chem import NetworkArrays
    from .jax_step import AtmStatic

from ._adjoint_audit import audit_adjoint_scope
from ._adjoint_core import (
    _TOP_K,
    BODY_MAP_DT,
    LGMRES_CYCLES,
    LGMRES_INNER_M,
    LGMRES_MAXITER,
    LGMRES_OUTER_K,
    LGMRES_RTOL,
    N_SOLVES_DEFAULT,
    SOLVER_MAP,
    BodyTerms,
    PhotoRecomputeArg,
    _adjoint_solve_core,
    _check_body_dt,
    _topk_ensemble_spread,
    _warn_poor_convergence,
)

__all__ = [
    "BODY_MAP_DT",
    "BODY_MAP_DT_CANDIDATES",
    "LGMRES_CYCLES",
    "LGMRES_INNER_M",
    "LGMRES_MAXITER",
    "LGMRES_OUTER_K",
    "LGMRES_RTOL",
    "N_SOLVES_DEFAULT",
    "SOLVER_MAP",
    "BodyTerms",
    "PhotoRecomputeArg",
    "audit_adjoint_scope",
    "make_body_terms",
    "make_photo_recompute_k",
    "steady_state_input_sensitivity",
    "steady_state_reaction_sensitivity",
]

# One-step map the adjoint linearizes (info["solver_map"]):
# G(y) = M * ros2_step(y, k, dt) / sum_i ros2_step, the hydrostatic-
# renormalized Ros2 step the runner iterates, so y_star is a tight fixed
# point of it.


# Probe step (s) for the adjoint body map. Adjoint-only: never touches the
# forward model; it sets how one probe step weights each chemical mode
# (weight ~ dt/tau for slow modes; FP-noise amplification grows with dt).
# The usable window is column/network-dependent: scan a few values and keep
# the lowest-residual, low-spread solution.
BODY_MAP_DT_CANDIDATES = (3e6, 1e7, 3e7, 1e8)

# steady_state_input_sensitivity: warn when rebuild(p0) reproduces the converged (k_arr, atm) only to this relative
# level (rates_jax matches the host build to ~5e-14; anything above float noise is a subtly different map).
_REBUILD_CONSISTENCY_WARN = 1e-8
# ...and refuse above this: the adjoint would be linearized against a visibly different map.
_REBUILD_CONSISTENCY_ERR = 1e-3


# size-limit-ok: parameter docs, the input checks, then one adjoint solve and its report
def steady_state_reaction_sensitivity(
    loss_fn: Callable[[jnp.ndarray], jnp.ndarray],
    y_star: jnp.ndarray,
    k_arr: jnp.ndarray,
    atm: AtmStatic,
    net: NetworkArrays,
    *,
    compo_array: jnp.ndarray,
    dz: jnp.ndarray,
    body_dt: float = BODY_MAP_DT,
    photo_recompute_k: PhotoRecomputeArg = None,
    body_terms: BodyTerms | None = None,
    n_solves: int = N_SOLVES_DEFAULT,
    return_info: bool = False,
) -> Any:
    """Reverse-mode `dL/d(ln k_r)` at a converged steady state.

    Sensitivity of a scalar loss of the converged composition to every
    reaction-rate constant -- the reaction-ranking use case. The result is the
    mean over `n_solves` adjoint solves with deterministic ulp-perturbed
    right-hand sides; `info["ensemble_spread"]` (twin disagreement) is the
    magnitude error bar.

    Accuracy is limited by the linear solve and by the finite-tolerance
    mismatch between the exact fixed point and the state the forward run
    stopped at. The usable `body_dt` window is column-dependent: for
    publication-grade magnitudes loop over `BODY_MAP_DT_CANDIDATES` and
    inspect `info["resid_median"]` / `info["ensemble_spread"]` per candidate. For `dL/dT`, `dL/dKzz`, ... use
    `steady_state_input_sensitivity`; forward-mode `jvp` is exact for a
    handful of directions.

    Parameters
    ----------
    loss_fn
        `loss_fn(y) -> scalar` on the full `(nz, ni)` number-density state,
        e.g. `lambda y: jnp.log10(y[L, so2] / y[L].sum())`.
    y_star : (nz, ni)
        Converged state (number density, cm^-3) -- a tight fixed point of the
        body map (`info["fp_err"]`; ~1e-9). Do not iterate the map to tighten
        `fp_err`: it trades the deflation basis for the fixed point and
        degrades `info["null_quality"]`.
        Clip and charge balance are outside the body map; condensation,
        fix-species and bottom pins enter it only through `body_terms`: run
        `audit_adjoint_scope(...)` first (its per-cell defect
        scan also catches what the global max-norm `fp_err` masks).
    k_arr : (nr+1, nz)
        Converged rate-constant table.
    atm : AtmStatic
        Atmosphere with the converged refresh fields (g, dzi, Hpi, ...), as
        fed to the runner's body map.
    net : NetworkArrays
        Active network.
    compo_array : (ni, n_atoms)
        Per-species atom counts; pass `composition.compo_array[:ni]`.
    dz : (nz,)
        Layer thickness (cm). Required: `AtmStatic` carries only the interface
        average `dzi`, which is not invertible to `dz`. Use `AtmInputs.dz`.
    body_dt
        Adjoint-only probe step in s (see `BODY_MAP_DT`); scan on a new column.
    photo_recompute_k
        On photo-on columns pass `make_photo_recompute_k(integ.photo_static,
        final_state)`, a `k(y) -> k_arr` recompute so the state operator
        carries `dJ/dy` (WASP-39b OH+H2 11% -> 0.2%); it costs an RT solve per
        Krylov matvec. `None` (default) means no photolysis
        feedback and raise on a column with active photolysis rows (no
        frozen-photolysis adjoint).
    body_terms
        Optional `BodyTerms` (condensation composite, fix_species pins,
        layer-0 boundary pins). Required when the state converged with
        condensation active; build with `make_body_terms(integ,
        converged_state, atm_static)`, which also returns the spliced `atm`.
    n_solves
        Twin-ensemble size (`_TWIN_PERTURB`, seeded); `n_solves=1` is one
        solve with no spread estimate. Each extra solve costs one LGMRES
        budget; the operator compile is shared.
    return_info
        If True, also return a diagnostics dict.

    Returns
    -------
    dL_dlnk : (nr+1,)
        Ensemble-mean `dL/d(ln k_r)` per directional rate-table row (index 0
        is the unused 1-based pad). For a physical detailed-balance
        perturbation of a reversible reaction use the pair sum
        `g[fwd] + g[rev]`.
    info : dict, optional
        `fp_err`, `null_quality` (O(1) = a deflated direction is not null and
        the deflation is suspect), `resid`/`resid_median`/`resids`,
        `ensemble_spread` (top-10 (max-min)/|mean|; 0.0 when n_solves=1),
        `pair_antisym` (diagnostic only; see the module docstring),
        `n_matvec`, `n_null`, `n_solves`, `body_dt`, and the mode flags.
    """
    _check_body_dt(body_dt)

    # The body map holds the condensate reservoir and saturation tables
    # fixed. Rates do not move the saturation curve directly, so a
    # condensation-active gradient is labeled conditional, not refused.
    _conden_pinned = body_terms is not None and body_terms.fix_mask is not None
    _conden_in_window = body_terms is not None and body_terms.conden_static is not None
    if _conden_pinned or _conden_in_window:
        warnings.warn(
            "steady_state_reaction_sensitivity: condensation is active at this "
            "converged state, so the returned dL/d ln k is CONDITIONAL on the "
            + ("frozen (pinned) condensate reservoir" if _conden_pinned else "frozen saturation tables")
            + "; it does not include how the rate changes what condenses. It is "
            "a valid conditional ranking, not the total rate sensitivity "
            "(info['conditional_on_fixed_reservoir']).",
            stacklevel=2,
        )

    lams, info, (_, body_map_k, _) = _adjoint_solve_core(
        loss_fn,
        y_star,
        k_arr,
        atm,
        net,
        compo_array=compo_array,
        dz=dz,
        body_dt=body_dt,
        photo_recompute_k=photo_recompute_k,
        body_terms=body_terms,
        n_solves=n_solves,
    )

    # Reaction cotangent per twin: dL/d(ln k_r) = (k .* G_k^T lambda)_r.
    _, vjp_Gk = jax.vjp(body_map_k, k_arr)
    grads = [np.asarray((k_arr * vjp_Gk(lam)[0]).sum(axis=1)) for lam in lams]

    g_stack = np.stack(grads, axis=0)  # (n_solves, nr+1)
    dL_dlnk = jnp.asarray(g_stack.mean(axis=0))

    # Twin-to-twin disagreement on the reactions one would actually report.
    ensemble_spread = _topk_ensemble_spread(g_stack)

    # Forward/reverse pair antisymmetry over the top `_TOP_K`: near partial
    # equilibrium dL/dln k_f ~ -dL/dln k_r, so |g_f+g_r|/max(|g_f|,|g_r|)
    # should be small. Diagnostic only (see module docstring); photolysis /
    # irreversible rows (all-zero reverse k) are skipped.
    g_mean_full = np.asarray(dL_dlnk)
    k_np = np.asarray(k_arr)
    pair_antisym = 0.0
    for r_np in np.argsort(np.abs(g_mean_full))[::-1][:_TOP_K]:
        r = int(r_np)
        f = r if r % 2 == 1 else r - 1
        rev = f + 1
        if f < 1 or rev >= g_mean_full.shape[0]:
            continue
        if not (np.any(k_np[f] != 0.0) and np.any(k_np[rev] != 0.0)):
            continue  # photo/conden/irreversible slot, no genuine pair
        denom = max(abs(g_mean_full[f]), abs(g_mean_full[rev]), _UNDERFLOW_DENOM)
        pair_antisym = max(pair_antisym, abs(g_mean_full[f] + g_mean_full[rev]) / denom)

    # Warn on the median twin residual; info["resid"] reports the max.
    _warn_poor_convergence(info["resid_median"], info["fp_err"], ensemble_spread, info["null_quality"])

    if not bool(jnp.all(jnp.isfinite(dL_dlnk))):
        raise ValueError(
            "Reaction sensitivity is non-finite. Common causes: a body_dt in "
            "the danger zone, or a y_star that is not a fixed point of the body "
            f"map (fp_err={info['fp_err']:.2e})."
        )

    if not return_info:
        return dL_dlnk
    info.update(
        ensemble_spread=ensemble_spread,
        pair_antisym=pair_antisym,
        conditional_on_fixed_reservoir=bool(_conden_pinned),
    )
    return dL_dlnk, info


def _worst_rel(a: Any, b: Any) -> float:
    """Return the largest cell-wise relative difference; inf for mismatched shapes or a non-finite entry."""
    a_np = np.asarray(a, dtype=np.float64)
    b_np = np.asarray(b, dtype=np.float64)
    if a_np.shape != b_np.shape:
        return float("inf")
    denom = np.maximum(np.maximum(np.abs(a_np), np.abs(b_np)), _UNDERFLOW_DENOM)
    rel = np.abs(a_np - b_np) / denom
    return float(np.max(rel)) if np.isfinite(rel).all() else float("inf")


# size-limit-ok: parameter docs, the rebuild consistency check, then one adjoint solve and its map
def steady_state_input_sensitivity(
    loss_fn: Callable[[jnp.ndarray], jnp.ndarray],
    y_star: jnp.ndarray,
    k_arr: jnp.ndarray,
    atm: AtmStatic,
    net: NetworkArrays,
    p0: Any,
    rebuild: Callable,
    *,
    compo_array: jnp.ndarray,
    dz: jnp.ndarray,
    body_dt: float = BODY_MAP_DT,
    photo_recompute_k: PhotoRecomputeArg = None,
    body_terms: BodyTerms | None = None,
    n_solves: int = N_SOLVES_DEFAULT,
    return_info: bool = False,
) -> Any:
    """Reverse-mode `dL/dp` at a converged steady state.

    For an arbitrary physical input pytree `p` (a (nz,) temperature profile, ln Kzz, ...).

    One adjoint solve (identical to `steady_state_reaction_sensitivity`'s)
    yields the cotangent `lambda`; the input gradient is one VJP of the
    parameter map at the fixed point:

        dL/dp = lambda^T dG/dp ,   G(p) = post(ros2_step(y*, k(p), dt, atm(p)))

    so all components of `p` cost one solve plus one VJP. The renormalization
    uses `atm(p).M`, so an input that moves the total density (temperature:
    M = pco/(kb T)) is differentiated through the rebalance too.

    Parameters (beyond the shared ones)
    -----------------------------------
    p0
        The input value the state was converged at (array or pytree).
    rebuild
        `rebuild(p) -> (k_arr_p, atm_p)`: a JAX-differentiable rebuild of the
        full rate table and `AtmStatic` at input `p`. Must reproduce the
        converged inputs at `p0` (warn above `_REBUILD_CONSISTENCY_WARN`,
        refuse above `_REBUILD_CONSISTENCY_ERR`); in particular non-thermal
        rows (photolysis J, conden) must be spliced in frozen from the
        converged `k_arr`. Example, a temperature profile on a photo-off
        column::

            pco = atm.M * kb * atm.Tco          # fixed hydrostatic pressures

            def rebuild(T):
                M = pco / (kb * T)
                k = rates_jax.build_rate_array(network, T, M, nasa9, remove_list)
                # photo-on columns: k = jnp.where(photo_rows[:, None], k_arr, k)
                return k, atm._replace(Tco=T, Ti=0.5 * (T[:-1] + T[1:]), M=M)

    Returns
    -------
    dL_dp
        Same pytree structure as `p0`; ensemble mean over `n_solves` twins.
    info : dict, optional
        The shared adjoint diagnostics plus `rebuild_consistency` (worst
        relative mismatch of `rebuild(p0)` per field).

    Scope and accuracy notes
    ------------------------
    * Chemistry path only: `(dL/dy*) . (dy*/dp)`. A loss with a direct `p`
      dependence (e.g. T in the RT opacities of a spectrum chi-square) needs
      that term added separately (`jax.grad` w.r.t. p at fixed `y_star`).
    * Condensation-active states are refused: the saturation tables are frozen
      in dG/dp and, post-pin, the captured reservoir is held fixed, so the
      result would be O(1)-unreliable vs FD.
    * Also frozen by design (p-derivative omitted): the photolysis
      T-cross-section interpolation and the atm-refresh geometry cascade
      (dz/Hp/g, second-order; rebuild what you need on-graph in `atm_p`).
    * Accuracy class matches the reaction sensitivities. The deflation is
      exact only for atom-conserving rate knobs; spot-validate a new input
      type against a forward-mode `jvp` in one or two directions before
      production use.
    """
    _conden_in_window = body_terms is not None and body_terms.conden_static is not None
    _conden_pinned = body_terms is not None and body_terms.fix_mask is not None
    if _conden_in_window or _conden_pinned:
        _regime = "post-pin fix_species" if _conden_pinned else "in-window"
        raise ValueError(
            "steady_state_input_sensitivity: condensation is active at this "
            f"converged state ({_regime} regime); an input (T/Kzz/...) "
            "gradient through condensation is NOT reliably differentiable. "
            "The saturation tables are frozen in dG/dp (d(sat)/dT dropped) "
            "and, once fix_species has pinned, the captured condensate "
            "reservoir is held fixed, so the parameter's effect on what "
            "condensed is entirely absent. The pinned-species tangent "
            "disagrees with re-converged finite differences at O(1) (0.91 "
            "relative) -- the same reason Fisher / retrieval-inference "
            "through condensation is refused project-wide (README.md, "
            "Limits). Use forward-mode jvp on a single switch-free direction "
            "and validate it against FD, or disable condensation."
        )
    _check_body_dt(body_dt)

    # rebuild(p0) must reproduce the map the state converged under.
    k0, atm0 = rebuild(p0)

    consistency = {"k_arr": _worst_rel(k0, k_arr)}
    for name, ref in atm._asdict().items():
        cand = getattr(atm0, name)
        if ref is None or isinstance(ref, bool):
            consistency[name] = 0.0 if cand == ref else float("inf")
        else:
            consistency[name] = _worst_rel(cand, ref)
    worst_name = max(consistency, key=consistency.__getitem__)
    worst = consistency[worst_name]
    if worst > _REBUILD_CONSISTENCY_ERR:
        raise ValueError(
            f"rebuild(p0) does not reproduce the converged step inputs "
            f"(worst mismatch {worst:.2e} on {worst_name!r}); the adjoint "
            "would be linearized against a different map and the gradient "
            "silently wrong. Common causes: missing frozen photolysis/conden "
            "row splice into k(p), lowT-cap or remove_list mismatch, wrong "
            "pressure grid or un-spliced refresh geometry in atm(p)."
        )
    if worst > _REBUILD_CONSISTENCY_WARN:
        warnings.warn(
            f"rebuild(p0) reproduces the converged inputs only to {worst:.2e} "
            f"(field {worst_name!r}); the gradient carries that inconsistency.",
            stacklevel=2,
        )

    lams, info, (apply_post, _, step_fn) = _adjoint_solve_core(
        loss_fn,
        y_star,
        k_arr,
        atm,
        net,
        compo_array=compo_array,
        dz=dz,
        body_dt=body_dt,
        photo_recompute_k=photo_recompute_k,
        body_terms=body_terms,
        n_solves=n_solves,
    )

    # Parameter map at the fixed point; photolysis rows ride k(p) frozen.
    def G_p(p: Any) -> Any:
        k_p, atm_p = rebuild(p)
        return apply_post(step_fn(y_star, k_p, atm_p), M_col=atm_p.M[:, None])

    _, pullback = jax.vjp(G_p, p0)
    twin_grads = [pullback(lam)[0] for lam in lams]
    dL_dp = jax.tree_util.tree_map(lambda *leaves: jnp.mean(jnp.stack(leaves), axis=0), *twin_grads)

    flat = [np.concatenate([np.ravel(np.asarray(leaf)) for leaf in jax.tree_util.tree_leaves(g)]) for g in twin_grads]
    ensemble_spread = _topk_ensemble_spread(np.stack(flat, axis=0))

    _warn_poor_convergence(info["resid_median"], info["fp_err"], ensemble_spread, info["null_quality"])

    finite = all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree_util.tree_leaves(dL_dp))
    if not finite:
        raise ValueError(
            "Input sensitivity is non-finite. Common causes: a body_dt in "
            "the danger zone, a y_star that is not a fixed point of the body "
            f"map (fp_err={info['fp_err']:.2e}), or a non-differentiable rebuild."
        )

    if not return_info:
        return dL_dp
    info.update(ensemble_spread=ensemble_spread, rebuild_consistency=consistency)
    return dL_dp, info


def make_photo_recompute_k(runner_photo_static: Any, converged_state: Any) -> Any:
    """Build a differentiable `k(y)` that rebuilds the photolysis rows from y.

    Reuses the runner's own in-loop photo branch so the recompute is
    bit-identical to the forward model's photolysis (optical depth ->
    two-stream RT -> J-rates -> photolysis rows of `k_arr`). The RT sweeps
    are `lax.associative_scan`s, hence reverse-mode differentiable, so passing the result
    as `photo_recompute_k` makes the state operator carry `dJ/dy`.

    Parameters
    ----------
    runner_photo_static
        The runner's internal `_PhotoStatic` (`OuterLoop.photo_static` after
        `ensure_runner`), not the public `PhotoStaticInputs` pytree.
    converged_state
        A converged `JaxIntegState`; supplies the frozen geometry (`dz`, `pv`
        T-cross sections, prior `dflux_u` -- its second-order self-recursion
        is not differentiated) read alongside the perturbed `y`.

    Returns
    -------
    recompute_k : Callable[[y], k_arr]
        `y (nz, ni) -> k_arr (nr+1, nz)` with photolysis rows = J(y).
    """
    from ._loop_branches import _make_photo_branch  # lazy: keep the module acyclic

    photo_branch = _make_photo_branch(runner_photo_static)

    def recompute_k(y: jnp.ndarray) -> Any:
        ymix = y / jnp.sum(y, axis=1, keepdims=True)
        return photo_branch(converged_state._replace(y=y, ymix=ymix)).k_arr

    return recompute_k


# size-limit-ok: one branch per process, with its doc
def make_body_terms(integ: Any, converged_state: Any, atm_static: Any) -> Any:
    """Build `(atm_step, BodyTerms)` for the adjoint from a finished runner.

    Packs every supported per-step process the runner's configuration turns
    on:

    * atm splice: `g`/`dzi`/`Hpi`/`top_flux`/`vs` from the converged carry,
      plus a live `vm` recompute when `use_vm_mol` (the setup-time
      `atm_static.vm` is stale).
    * condensation, in-window regime (fix_species not tripped): the runner's
      `CondenStatic` spliced with the converged `ProfileVars`, enabling the
      conden-row recompute + relax kernels. Sat tables are T-baked constants
      (d(sat)/dT missing for T-gradients). Free-running conden states are
      typically pseudo-steady; read `audit_adjoint_scope`'s per-cell defect
      for how tight the state actually is.
    * fix_species regime (`fix_species_started`): pin mask + pinned values
      from the carry -- the regime real converged conden runs end in.
    * layer-0 pins: `use_fix_sp_bot` and the hycean H2/He pin once tripped.
    * gas-only / partial hydrostatic balance (condensate-aware form).

    Raises NotImplementedError for `use_ion`; warns when `diff_esc` is active
    (d(phi_esc)/dy stays frozen, leading-order only).

    Parameters
    ----------
    integ
        The `OuterLoop` whose runner produced `converged_state`.
    converged_state
        The converged `JaxIntegState` (the `_runner` output).
    atm_static
        The `AtmStatic` fed to the runner; returned re-spliced.
    """
    st = integ._statics
    if st is None:
        raise ValueError(
            "make_body_terms: the runner has not been built/run; integrate "
            "first (integ(rs) or integ._runner(state, atm_static))."
        )
    cfg = integ._cfg
    s = converged_state

    if bool(st.use_ion):
        raise NotImplementedError(
            "use_ion=True: the electron-row pin and post-step charge balance "
            "are not implemented in the adjoint body map; ion columns are "
            "not supported. Use forward-mode."
        )

    # --- the runner's own atm splice: converged geometry, live vm, and the
    # hybrid vm_mol phase it converged on (a hybrid run ends in phase 1) ---
    from ._loop_branches import _splice_conden, _splice_step_atm  # lazy: acyclic

    atm_step = _splice_step_atm(
        atm_static, s, bool(st.use_vm_mol), integ._refresh_static.kb, integ._refresh_static.Navo
    )

    if int(np.asarray(integ._refresh_static.diff_esc_idx).size) > 0:
        warnings.warn(
            "diff_esc is active: the runner recomputes the diffusion-"
            "limited escape flux from the top-layer densities; the "
            "adjoint freezes top_flux at its converged value, so "
            "d(phi_esc)/dy is dropped and TOA-coupled sensitivities are "
            "leading-order only.",
            stacklevel=2,
        )

    started = bool(np.asarray(s.fix_species_started))

    # --- condensation regimes ---
    conden_static = None
    fix_mask = None
    fix_y = None
    cs = integ._conden_static
    if cs is not None:
        if started:
            # Post-window: species pinned via fix_mask inside the step;
            # conden no longer fires.
            fix_mask = s.fix_mask
            fix_y = s.fix_y
        elif float(np.asarray(s.t)) >= float(cfg.start_conden_time):
            # In-window: _make_conden_branch's per-lane splice.
            conden_static = _splice_conden(cs, s.pv)

    gas_mask = cs.gas_indx_mask if cs is not None else None
    if gas_mask is not None and bool(jnp.all(gas_mask)):
        gas_mask = None  # all-gas network: plain renorm denominator

    # --- layer-0 Dirichlet pins (set after the balance here; the runner
    # sets them before the clip) ---
    n0_bot = s.pv.n_0[0]
    idx_parts = []
    val_parts = []
    if bool(st.use_fix_sp_bot) and int(np.asarray(st.fix_sp_bot_idx).size) > 0:
        idx_parts.append(st.fix_sp_bot_idx.astype(jnp.int32))
        val_parts.append(st.fix_sp_bot_mix * n0_bot)
    if bool(st.use_fix_H2He):
        if bool(np.asarray(s.h2he_pinned)):
            idx_parts.append(jnp.asarray([int(st.h2_idx), int(st.he_idx)], dtype=jnp.int32))
            val_parts.append(s.h2he_mix * n0_bot)
        else:
            warnings.warn(
                "use_fix_H2He=True but the pin had not tripped at this state "
                "(t <= hycean_pin_time); the body map omits it, consistent "
                "with the converged state.",
                stacklevel=2,
            )
    bot_idx = jnp.concatenate(idx_parts) if idx_parts else None
    bot_val = jnp.concatenate(val_parts) if val_parts else None

    terms = BodyTerms(
        conden_static=conden_static,
        gas_mask=gas_mask,
        hydro_partial=bool(cfg.use_condense),
        fix_mask=fix_mask,
        fix_y=fix_y,
        bot_idx=bot_idx,
        bot_val=bot_val,
    )
    return atm_step, terms
