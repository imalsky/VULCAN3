"""Scan a converged run for physics the adjoint's body map drops: public name `audit_adjoint_scope`."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from ._adjoint_core import _FP_ERR_WARN, _TOP_K, BODY_MAP_DT, BodyTerms, _check_body_dt, _clip_dead_mask, _make_body_map
from .phy_const import UNDERFLOW_DENOM as _UNDERFLOW_DENOM

if TYPE_CHECKING:
    from collections.abc import Callable

    from ._adjoint_core import PhotoRecomputeArg
    from ._carry import JaxIntegState
    from .chem import NetworkArrays
    from .config import Config
    from .jax_step import AtmStatic

__all__ = ["audit_adjoint_scope"]

# The per-cell defect scan ignores cells below this mixing ratio: numerical dust with meaningless relative defect, far
# below any loss species (trace radicals sit ~1e-12..1e-8).
_AUDIT_MIN_YMIX = 1e-16

# Per-cell defect above which the finding is an ERROR: an O(1) move under one probe step means the iterated map includes
# a process (pin, conden clamp, charge balance) the body map lacks. Between _FP_ERR_WARN and this is a WARNING (slow
# trace cells of a healthy column land here).
_AUDIT_DEFECT_ERROR = 0.3

# The "loss footprint": cells whose log-space cotangent magnitude |y* dL/dy| is within this fraction of the maximum, the
# cells the loss actually reads, where a fixed-point defect directly biases the gradient.
_AUDIT_LOSS_FOOTPRINT_FRAC = 1e-3

# A spliced refresh field (g, dzi, Hpi, top_flux, vs) must match the converged carry to float noise.
_STALE_GEOMETRY_RTOL = 1e-12

logger = logging.getLogger(__name__)


# size-limit-ok: one check per process
def _adjoint_scope_findings(  # noqa: C901 (one check per process)
    cfg: Config,
    final_state: JaxIntegState | None = None,
    photo_recompute_k: PhotoRecomputeArg = None,
    body_terms: BodyTerms | None = None,
) -> list:
    """Run static scope checks: which runner processes are OUTSIDE the adjoint's body map for this configuration.

    Also says what that does to the gradient.

    Severity: "error" = a dropped process is active at the fixed point, so
    sensitivities touching it are wrong; "warning" = a leading-order-only
    omission; "info" = a second-order truncation or a diagnostic to watch.
    Pure host logic; `audit_adjoint_scope` adds the per-cell defect scan.
    """
    findings: list = []

    def add(code: str, severity: str, message: str) -> None:
        findings.append({"code": code, "severity": severity, "message": message})

    def state_bool(name: str) -> bool | None:
        if final_state is None:
            return None
        return bool(np.asarray(getattr(final_state, name)))

    pass_terms = "Pass body_terms from make_body_terms(integ, final, atm_static)."
    if bool(cfg.use_ion):
        add(
            "ion_charge_balance",
            "error",
            "use_ion=True: the electron-row pin and the post-step charge "
            "balance are not in the body map; use forward mode for ion columns.",
        )

    terms_conden = body_terms is not None and body_terms.conden_static is not None
    terms_fix = body_terms is not None and body_terms.fix_mask is not None
    terms_bot = body_terms is not None and body_terms.bot_idx is not None

    if bool(cfg.use_condense):
        started = state_bool("fix_species_started")
        if terms_conden or (started and terms_fix):
            add(
                "condensation",
                "info",
                "condensation is carried by body_terms; the saturation tables "
                "are T-baked, so T-input gradients miss d(sat)/dT.",
            )
        else:
            add(
                "condensation",
                "error",
                "use_condense=True: the conden k-row recompute, the relax "
                "kernels and the gas-only hydrostatic balance are not in the "
                "body map. " + pass_terms,
            )
        if started and not terms_fix:
            add(
                "fix_species_pins",
                "error",
                "fix_species is active at this state (fix_species_started) but "
                "the body map has no fix_mask. " + pass_terms,
            )
        elif started is None and not terms_fix and not terms_conden and len(cfg.fix_species) > 0:
            add(
                "fix_species_pins",
                "error",
                "cfg.fix_species is non-empty and the body map has no fix_mask; "
                "pass the converged state to check whether the pin is active.",
            )

    bot_pins = len(cfg.use_fix_sp_bot) > 0
    if bot_pins:
        if terms_bot:
            add("bottom_boundary_pins", "info", "the bottom-layer pins are carried by body_terms.")
        else:
            add(
                "bottom_boundary_pins",
                "error",
                "use_fix_sp_bot: the bottom-layer pin is not in the body map, so "
                "pinned bottom rows are wrong while the global fp_err can read "
                "~1e-9; trust the per-cell defect scan. " + pass_terms,
            )

    if bool(cfg.use_fix_H2He):
        pinned = state_bool("h2he_pinned")
        if pinned and terms_bot:
            add("hycean_h2he_pin", "info", "the tripped H2/He bottom pin is carried by body_terms.")
        elif pinned is False:
            add(
                "hycean_h2he_pin",
                "warning",
                "the H2/He bottom pin has not tripped yet (t <= hycean_pin_time); "
                "once it trips, the bottom H2/He rows are wrong.",
            )
        else:
            add(
                "hycean_h2he_pin",
                "error",
                "use_fix_H2He=True: the H2/He bottom pin is not in the body map. " + pass_terms,
            )

    if bool(cfg.use_photo):
        if photo_recompute_k is None:
            add(
                "photolysis_feedback",
                "error",
                "use_photo=True with photo_recompute_k=None freezes J (dJ/dy "
                "omitted, 11% on W39b OH+H2). Pass make_photo_recompute_k("
                "integ.photo_static, final).",
            )
        else:
            add(
                "photo_dflux_recursion",
                "info",
                "dflux_u is held at its converged value inside the photo recompute (second order).",
            )

    if len(cfg.diff_esc) > 0:
        add(
            "escape_flux_feedback",
            "warning",
            "diff_esc: top_flux is frozen, so d(phi_esc)/dy is dropped and TOA-sensitive losses are biased.",
        )

    if bool(cfg.use_vm_mol) and not bool(cfg.use_hybrid_vm_mol):
        add(
            "vm_mol_feedback",
            "warning",
            "use_vm_mol=True: vm is frozen at its converged value, dropping d(vm)/dy (matters at low Kzz).",
        )
    elif bool(cfg.use_vm_mol) and bool(cfg.use_hybrid_vm_mol):
        add(
            "vm_mol_hybrid",
            "info",
            "use_hybrid_vm_mol=True: the run ends on the central operator "
            "(hybrid_use_vm=0), so there is no d(vm)/dy to drop.",
        )

    add(
        "atm_refresh_feedback",
        "info",
        "the mu -> (g, dz, Hp, Hpi) refresh feedback is not in dG/dy (second "
        "order; included in the 0.1-0.7% FD totals on HD189/W39b).",
    )

    if bool(cfg.use_topflux) or bool(cfg.use_botflux) or bot_pins or len(cfg.diff_esc) > 0:
        add(
            "open_boundary_deflation",
            "info",
            "open or pinned boundaries break column atom conservation; check "
            "info['null_quality'] (O(1) means the deflation corrupts the solve).",
        )

    return findings


# size-limit-ok: four checks in order, with the doc of each
def audit_adjoint_scope(  # noqa: C901 (four checks in order)
    y_star: jnp.ndarray,
    k_arr: jnp.ndarray,
    atm: AtmStatic,
    net: NetworkArrays,
    *,
    cfg: Config | None = None,
    final_state: JaxIntegState | None = None,
    loss_fn: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    photo_recompute_k: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    body_terms: BodyTerms | None = None,
    body_dt: float = BODY_MAP_DT,
    print_report: bool = True,
) -> dict[str, Any]:
    """Scan a run for physics the adjoint's body map drops, before trusting it.

    The adjoint linearizes `G = ros2_step (+ renorm) (+ photo) (+ body_terms)` at `y_star`;
    every other per-step runner process is outside that map. Dropping a
    process is exact when it is inactive/identity at the fixed point and
    physically wrong when it still shapes it. Four checks:

    1. Static config/state checks (`_adjoint_scope_findings`), classified
       error / warning / info.
    2. Per-cell fixed-point defect `|G(y*) - y*| / y*` on cells with
       `ymix >= _AUDIT_MIN_YMIX`, built from the same `_make_body_map` the solver
       uses -- any unmodeled active process shows up as a localized defect.
       This catches what the global max-norm `fp_err` cannot: a pinned bottom
       row is invisible next to the deep-column H2 density.
       Cells the runner's zero-clip owns are excluded and reported separately
       (`n_clip_dead_excluded`); that exclusion is lifted inside the loss
       footprint, where such a cell is a hard error.
    3. Stale-geometry check (with `final_state`): verifies the converged
       refresh fields (g, dzi, Hpi, top_flux, vs) were spliced into `atm`.
    4. Loss footprint (with `loss_fn`): the worst defect among the cells the
       loss actually reads (|y* dL/dy| within `_AUDIT_LOSS_FOOTPRINT_FRAC` of
       max) -- a defect there biases the answer directly.

    Parameters mirror `steady_state_reaction_sensitivity` where shared,
    except `photo_recompute_k`: the callable from `make_photo_recompute_k`
    on a photo-on column, None on a photo-off one (None with `use_photo` is
    an error finding). `cfg` defaults to the process default config (pass the run's cfg for
    `make_config`-driven runs); `final_state` is the converged `JaxIntegState`
    (sharpens the conden/pin checks, enables the stale-geometry check).

    Returns a dict: `findings` ({code, severity, message} list),
    `max_rel_defect`, `worst_cells`, `fp_err_global`, `loss_footprint_defect`
    (None without `loss_fn`), and `ok` (True when no error-severity findings;
    warnings do not clear it to False).
    """
    _check_body_dt(body_dt)
    if cfg is None:
        from .config import default_config

        cfg = default_config()  # the runner's own default cfg surface

    findings = _adjoint_scope_findings(cfg, final_state, photo_recompute_k, body_terms)

    # Stale-geometry check: the body map must see the same refreshed fields
    # the runner converged with, or the linearization is taken off-manifold.
    if final_state is not None:
        stale = []
        for name in ("g", "dzi", "Hpi", "top_flux", "vs"):
            a_np, b_np = np.asarray(getattr(atm, name)), np.asarray(getattr(final_state, name))
            if a_np.shape != b_np.shape or not np.allclose(a_np, b_np, rtol=_STALE_GEOMETRY_RTOL, atol=0.0):
                stale.append(name)
        if stale:
            findings.append(
                {
                    "code": "stale_geometry",
                    "severity": "error",
                    "message": f"atm fields {stale} differ from the converged "
                    "carry: splice the refreshed geometry before the adjoint "
                    "with atm_static._replace(g=final.g, dzi=final.dzi, "
                    "Hpi=final.Hpi, top_flux=final.top_flux, vs=final.vs).",
                }
            )

    # Per-cell fixed-point defect of the exact map the solver would use.
    _, body_map, _, _ = _make_body_map(y_star, k_arr, atm, net, body_dt, photo_recompute_k, body_terms)
    G = jax.jit(body_map)(y_star)
    y_np = np.asarray(y_star)
    G_np = np.asarray(G)
    non_finite = ~(np.isfinite(y_np) & np.isfinite(G_np))
    if non_finite.any():
        z, i = np.argwhere(non_finite)[0]
        findings.append(
            {
                "code": "probe_non_finite",
                "severity": "error",
                "message": f"{int(non_finite.sum())} cell(s) of y_star or of its one-step probe are NaN/Inf, first at "
                f"layer {int(z)}, species index {int(i)} (y {y_np[z, i]:g}, probe {G_np[z, i]:g}): every defect "
                "comparison below reads False against a NaN, so nothing below certifies this state.",
            }
        )
    defect = np.abs(G_np - y_np)
    fp_err_global = float(defect.max() / max(float(np.abs(y_np).max()), _UNDERFLOW_DENOM))
    ymix = y_np / np.maximum(y_np.sum(axis=1, keepdims=True), _UNDERFLOW_DENOM)

    # Exclude cells the zero-clip owns: they have no fixed point, and
    # _AUDIT_MIN_YMIX (a mixing ratio) cannot catch the absolute cm^-3 clip window.
    clip_dead = _clip_dead_mask(G_np, cfg)
    scanned = (y_np > 0.0) & (ymix >= _AUDIT_MIN_YMIX)
    rel_full = np.where(scanned, defect / np.maximum(y_np, _UNDERFLOW_DENOM), 0.0)
    rel = np.where(clip_dead, 0.0, rel_full)
    max_rel_defect = float(rel.max())

    # Skipped cells are not passed cells: report the exclusion instead of
    # silently shrinking the scan.
    dead_scanned = clip_dead & scanned
    n_clip_dead = int(dead_scanned.sum())
    clip_dead_worst = float(rel_full[dead_scanned].max()) if n_clip_dead else 0.0

    from . import chem_funs  # label with the import-locked network when the width matches

    species = chem_funs.spec_list if len(chem_funs.spec_list) == y_np.shape[1] else None

    order = np.argsort(rel.ravel())[::-1][:_TOP_K]
    worst_cells = []
    for flat_idx in order:
        z, i = np.unravel_index(int(flat_idx), rel.shape)
        if rel[z, i] <= 0.0:
            break
        worst_cells.append(
            {
                "layer": int(z),
                "species_index": int(i),
                "species": species[i] if species is not None else f"sp{i}",
                "rel_defect": float(rel[z, i]),
                "ymix": float(ymix[z, i]),
            }
        )

    if max_rel_defect > _FP_ERR_WARN:
        w0 = worst_cells[0]
        where = f"at {w0['species']} layer {w0['layer']}"
        if max_rel_defect > _AUDIT_DEFECT_ERROR:
            findings.append(
                {
                    "code": "fixed_point_defect",
                    "severity": "error",
                    "message": f"max per-cell fixed-point defect "
                    f"{max_rel_defect:.2e} {where} (global fp_err "
                    f"{fp_err_global:.2e}): an O(1) move under one probe step "
                    "cannot be convergence creep: the map the runner "
                    "actually iterates includes a process (pin, conden "
                    "clamp, charge balance) this body map does not. See the "
                    "findings above and worst_cells.",
                }
            )
        else:
            findings.append(
                {
                    "code": "fixed_point_defect",
                    "severity": "warning",
                    "message": f"per-cell fixed-point defect up to "
                    f"{max_rel_defect:.2e} {where} (global fp_err "
                    f"{fp_err_global:.2e}): either a weak unmodeled per-step "
                    "process acts there, or those cells are slow species "
                    "still creeping at the forward convergence tolerance "
                    "(the state-definition mismatch; typical for upper-"
                    "atmosphere trace species). Both bias the adjoint in "
                    "those cells; converge/polish y_star tighter if they "
                    "matter to your loss; check worst_cells against the "
                    "loss footprint.",
                }
            )

    if n_clip_dead:
        findings.append(
            {
                "code": "clip_dead_cells_excluded",
                "severity": "info",
                "message": f"{n_clip_dead} cell(s) excluded from the per-cell "
                f"defect scan: the runner's zero-clip ([{float(cfg.nega_cut):g}, "
                f"{float(cfg.pos_cut):g}) cm^-3) fires there, so they have no fixed "
                f"point and their relative defect (up to "
                f"{clip_dead_worst:.2e}) measures the clip rather than a "
                "dropped process. They are solver noise -- densities far "
                "below one particle per cm^3 -- but their adjoint rows are "
                "linearized as identity, so a loss that READS one is still "
                "refused (see loss_footprint_defect).",
            }
        )

    # Loss footprint: does the loss read any defective cell directly?
    loss_footprint_defect = None
    if loss_fn is not None:
        v = np.asarray(jax.grad(loss_fn)(y_star))
        w = np.abs(v * y_np)  # log-space cotangent magnitude, the adjoint RHS
        # Report a non-finite cotangent, never swallow it: every `>` against
        # NaN is False, so a NaN w.max() would leave `foot` empty and declare
        # a clean footprint on a poisoned loss gradient.
        n_bad_cot = int((~np.isfinite(v)).sum())
        if n_bad_cot:
            findings.append(
                {
                    "code": "loss_cotangent_non_finite",
                    "severity": "error",
                    "message": f"{n_bad_cot} cell(s) of jax.grad(loss_fn)(y_star) "
                    "are NaN/Inf, so the adjoint RHS is poison before the scope "
                    "audit even begins. Every downstream footprint comparison "
                    "silently reads False against a NaN, which would look like a "
                    "clean footprint. Fix the loss (a log of a zero-clipped cell "
                    "is the usual cause) before trusting any sensitivity.",
                }
            )
        w_finite = w[np.isfinite(w)]
        w_max = float(w_finite.max()) if w_finite.size else 0.0
        foot = np.isfinite(w) & (w > _AUDIT_LOSS_FOOTPRINT_FRAC * max(w_max, _UNDERFLOW_DENOM))
        # Score the footprint on the defect before the clip-dead exclusion.
        loss_footprint_defect = float(rel_full[foot].max()) if foot.any() else 0.0
        if foot.any() and (clip_dead & foot).any():
            findings.append(
                {
                    "code": "loss_reads_clip_dead_cell",
                    "severity": "error",
                    "message": f"{int((clip_dead & foot).sum())} cell(s) in "
                    "the loss footprint sit in the runner's zero-clip dead "
                    "zone: those cells have no fixed point and the adjoint "
                    "would linearize a clip as identity, so THIS loss's "
                    "gradient is wrong at first order. Choose a loss on "
                    "species/layers with real abundance.",
                }
            )
        if loss_footprint_defect > _FP_ERR_WARN:
            findings.append(
                {
                    "code": "loss_footprint_defect",
                    "severity": "error",
                    "message": f"the loss directly reads cells with fixed-"
                    f"point defect up to {loss_footprint_defect:.2e}: the "
                    "gradient of THIS loss is biased at that level "
                    "regardless of cause (unmodeled process or cells not "
                    "fully converged); fix the scope or converge tighter "
                    "before using it.",
                }
            )

    severity_rank = {"error": 0, "warning": 1, "info": 2}
    findings.sort(key=lambda f: severity_rank.get(f["severity"], 3))
    ok = all(f["severity"] != "error" for f in findings)

    if print_report:
        logger.info("audit_adjoint_scope report")
        footprint = "" if loss_footprint_defect is None else f"  loss_footprint_defect={loss_footprint_defect:.3e}"
        logger.info("  ok=%s  max_rel_defect=%.3e  fp_err_global=%.3e%s", ok, max_rel_defect, fp_err_global, footprint)
        for f in findings:
            logger.info("  [%-7s] %s: %s", f["severity"].upper(), f["code"], f["message"])
        if worst_cells:
            logger.info("  worst fixed-point-defect cells (rel |G(y*)-y*|/y*):")
            for c in worst_cells:
                logger.info(
                    "    %12s  layer %3d  defect %.3e  ymix %.3e", c["species"], c["layer"], c["rel_defect"], c["ymix"]
                )

    return {
        "findings": findings,
        "ok": ok,
        "max_rel_defect": max_rel_defect,
        "fp_err_global": fp_err_global,
        "worst_cells": worst_cells,
        "loss_footprint_defect": loss_footprint_defect,
        # how much the scan skipped, so a caller can see the leniency it got
        "n_clip_dead_excluded": n_clip_dead,
        "clip_dead_worst_defect": clip_dead_worst,
    }
