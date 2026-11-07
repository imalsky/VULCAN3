"""Single-JIT outer integration loop.

The full VULCAN integration runs inside one `jax.lax.while_loop`: per-step
kernel (chem RHS, analytical Jacobian, diffusion, block-Thomas) composed
with photo update, atm refresh, condensation, ion charge balance,
fix-all-bot, adaptive rtol, photo-frequency ini->final switch, and a
ring-buffered convergence check. No Python loop, no NumPy on the hot path.

The accept/reject decision and dt formula match VULCAN-master's `op.Ros2`
(including the forced-accept fallback when `dt < dt_min`).
"""

from __future__ import annotations

import functools
import logging
import math
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from . import _build_statics, _pack_state, state as _state_mod
from ._carry import _ATM_ARRAY_FIELDS, ATM_STATIC_BATCH_AXES, JaxIntegState, ProfileVars, _PhotoStatic, _Statics
from ._convergence import certificate_failures, conv_normal, frozen_column_message
from ._runner_queue import _QUEUE_REFILL_CHUNK, _queue_compact_rungs
from ._runners import make_runner
from .chem_funs import NET_JAX, NETWORK
from .config import DT_MAX_S, default_config
from .jax_step import AtmStatic, make_atm_static
from .runtime_validation import refuse_missing_atoms, validate_runtime_config
from .state import TERM_CONVERGED, TERM_DT_MIN_FROZEN, DtMinFrozenError, raise_if_frozen

if TYPE_CHECKING:
    import types
    from collections.abc import Callable, Sequence

    from . import atm_refresh as _atm_refresh_mod, conden as _conden_mod
    from ._carry import PyTree
    from .config import Config
    from .legacy_io import Output
    from .op_jax import Ros2JAX
    from .state import RunState

__all__ = [
    "ATM_STATIC_BATCH_AXES",
    "DtMinFrozenError",
    "JaxIntegState",
    "OuterLoop",
    "ProfileVars",
    "certificate_failures",
    "conv_normal",
    "raise_if_frozen",
]

logger = logging.getLogger(__name__)

_ATM_STATIC_BATCH_AXES = ATM_STATIC_BATCH_AXES  # deprecated (docs/api.md)

_TRACED_CHECK = (
    "A runner is traced under an outer jit or vmap, so it cannot raise DtMinFrozenError. Unless the caller already "
    "does, call vulcan3.raise_if_frozen on termination_reason right after that call."
)


class OuterLoop:
    """Standalone outer-integration driver.

    One JIT'd `lax.while_loop` runs every accepted step with internal retries, photo / atm-refresh / conden updates,
    ring-buffered convergence, and adaptive rtol.
    """

    def __init__(self, odesolver: Ros2JAX, output: Output, cfg: Config | None = None) -> None:
        # cfg defaults to the process default (the CLI);
        # load_config() users pass their own namespace so every runtime knob
        # reads from the cfg the RunState was built with (setup counterpart:
        # state._cfg_overlay). The import-locked knobs (network, com_file,
        # atom_list) are the ones cfg cannot change here: `NETWORK` is the
        # network chem_funs parsed at import, so a changed `network` needs a
        # new Python process.
        self._cfg = cfg if cfg is not None else default_config()
        if float(self._cfg.dt_max) > DT_MAX_S:
            raise ValueError(
                f"dt_max={float(self._cfg.dt_max):g} s exceeds config.DT_MAX_S="
                f"{DT_MAX_S:g} s, the largest step the Ros2 stage repair resolves; "
                "lower it or drop the key so it is derived."
            )
        self.output = output
        self.odesolver = odesolver

        self._species = list(NETWORK.species)

        # Atom ordering captured once at init; dict<->array conversion relies on it.
        self._atom_order = [a for a in self._cfg.atom_list if a not in self._cfg.loss_ex]

        # compo_arr (ni, n_atoms): rows = species, cols = atom_order.
        from . import composition

        compo = composition.compo
        compo_row = composition.compo_row
        ni_ = NETWORK.ni
        compo_np = np.zeros((ni_, len(self._atom_order)), dtype=np.float64)
        for i, sp in enumerate(self._species):
            r_i = compo_row.index(sp)
            for a, atom in enumerate(self._atom_order):
                compo_np[i, a] = float(compo[r_i][atom])
        self._compo_arr = compo_np

        # Full charge column from compo. Per-run `charge_arr` (zeroed for
        # species outside var.charge_list) is built in _build_statics once
        # `var` is available.
        self._compo_charge = np.array(
            [float(compo[compo_row.index(sp)]["e"]) for sp in self._species],
            dtype=np.float64,
        )

        # Lazy runner cache; populated on first call and reused thereafter.
        # The closure pins the config read at build time; there is no reset --
        # a changed config means a new runner object.
        self._runner: Callable[[JaxIntegState, AtmStatic], JaxIntegState] | None = None
        self._runner_chunk: Callable[[JaxIntegState, AtmStatic, jnp.ndarray], JaxIntegState] | None = None
        self._make_runner_jvp: Callable[[tuple[bool, ...], tuple[bool, ...]], Callable] | None = None
        self._failed_trial: Callable[[JaxIntegState, AtmStatic], tuple[jnp.ndarray, ...]] | None = None
        # `integ(rs, chunk=N)` stores the last finished chunk here as a
        # RunState when Ctrl+C stops the run.
        self.interrupted_runstate: RunState | None = None
        self._runner_jvp_cache: dict[tuple, Callable] = {}
        # Un-jitted freeze-on-done while_loop over a vmapped step (the
        # batched path), and its jax.jit wrapper (`run_batch`). Both ride the
        # same (nz, toggle-combo) closure as `_runner`.
        self._runner_batch: Callable[[JaxIntegState, AtmStatic], JaxIntegState] | None = None
        self._vrunner: Callable[[JaxIntegState, AtmStatic], JaxIntegState] | None = None
        # Same closure with a job queue above the lanes (`run_queue`), and
        # its jit cache, keyed in `run_queue`.
        self._runner_queue: Callable | None = None
        self._vrunner_queue: dict[tuple, Callable] = {}
        self._statics: _Statics | None = None
        self._photo_static: _PhotoStatic | None = None
        self._refresh_static: _atm_refresh_mod.AtmRefreshStatic | None = None
        self._conden_static: _conden_mod.CondenStatic | None = None
        # First batched photo profile's TOA stellar flux; prepare_runstate
        # rejects later profiles with a different star (see the guard there).
        self._sflux_top_ref: np.ndarray | None = None

    def ensure_runner(self, var: types.SimpleNamespace, atm: types.SimpleNamespace) -> None:
        """Build the JIT'd runner on the first call; cached for subsequent."""
        if self._runner is not None:
            return

        nz = atm.Tco.shape[0]
        ni = NETWORK.ni

        gas_mask_jnp = jnp.asarray(_build_statics.gas_mask(self, atm))

        # condense_zero_mask (nz, ni) -- True where delta is zeroed; all False
        # unless use_condense (condense_sp + non_gas_sp).
        cond_mask_np = np.zeros((nz, ni), dtype=bool)
        if self._cfg.use_condense:
            for sp in self._cfg.condense_sp + self._cfg.non_gas_sp:
                if sp in NETWORK.species_idx:
                    cond_mask_np[:, NETWORK.species_idx[sp]] = True

        self._statics = _build_statics.build_statics(self, var, atm)
        self._photo_static = _build_statics.build_photo_static(self, var, atm)
        self._refresh_static = _build_statics.build_refresh_static(self, atm)
        self._conden_static = _build_statics.build_conden_static(self, var, atm, gas_mask_jnp)
        (
            self._runner,
            self._runner_chunk,
            self._runner_batch,
            self._runner_queue,
            self._make_runner_jvp,
            self._failed_trial,
        ) = make_runner(
            NET_JAX,
            self._statics,
            bool(self._cfg.non_gas_sp),
            gas_mask_jnp,
            bool(self._cfg.use_botflux) or bool(self._statics.use_fix_sp_bot),
            jnp.asarray(cond_mask_np),
            float(self._cfg.start_conden_time),
            float(self._cfg.stop_conden_time),
            refresh_static=self._refresh_static,
            photo_static=self._photo_static,
            conden_static=self._conden_static,
        )

    def run_jvp(
        self, state: JaxIntegState, atm_static: AtmStatic, dstate: PyTree, datm: PyTree
    ) -> tuple[JaxIntegState, PyTree, jnp.ndarray, jnp.ndarray]:
        """Forward-mode integration certified on the tangent.

        `dstate` / `datm` mirror `state` / `atm_static` leaf for leaf, as
        `jax.jvp` hands them back: a leaf with a float tangent is
        differentiated, a non-float leaf (ints, flags; float0 tangent) is
        held constant, and a float leaf without a float tangent raises
        TypeError. Returns `(final, dfinal, tangent_longdy, tangent_ok)`:
        `dfinal` mirrors `final` the same way (float0 on constant leaves),
        so downstream maps continue with `jax.jvp(g, (final.y,), (dfinal.y,))`.

        The run stops when the state certifies AND the tangent's change over
        the same lookback, per cell over `y`, passes the same two-branch
        test (`_tangent_conv`); the tangent direction is therefore read in
        the units of a finite-difference step. `termination_reason` 1 means
        both certified; 3 means the count cap hit before they did.

        Several directions at once: give every tangent leaf a leading axis of
        length D (the primal keeps its own shape). The primal is then
        integrated once for all D, `tangent_longdy` comes back as (D,),
        `tangent_ok` stays a scalar (every direction settled) and `dfinal`
        carries the D axis. Do not vmap this call over directions: that
        integrates the primal D times.
        """

        def _mask(dtree: PyTree, tree: PyTree) -> tuple[bool, ...]:
            leaves = jax.tree_util.tree_leaves(tree)
            dleaves = jax.tree_util.tree_leaves(dtree)
            if len(leaves) != len(dleaves):
                raise ValueError(f"tangent tree has {len(dleaves)} leaves, primal {len(leaves)}")

            def _is_float(x: PyTree) -> bool:
                return (
                    getattr(x, "dtype", None) is not None
                    and x.dtype != jax.dtypes.float0
                    and jnp.issubdtype(x.dtype, jnp.floating)
                )

            act = tuple(_is_float(d) for d in dleaves)
            # Every float leaf integrates a tangent (a zero seed grows one),
            # so a float primal without a float tangent would be silently
            # held constant and read as primal history by the certificate.
            bad = [i for i, (x, a) in enumerate(zip(leaves, act, strict=False)) if _is_float(x) and not a]
            if bad:
                raise TypeError(
                    f"float primal leaves {bad} carry no float tangent; pass "
                    "zeros (as jax.jvp does), never float0 or ints"
                )
            return act

        if self._make_runner_jvp is None:
            raise RuntimeError(
                "run_jvp called before the runner was built; call prepare_runstate on at least one profile first."
            )
        self._refuse_missing_atoms(state)
        act_s, act_a = _mask(dstate, state), _mask(datm, atm_static)
        key = (act_s, act_a)
        if key not in self._runner_jvp_cache:
            self._runner_jvp_cache[key] = self._make_runner_jvp(act_s, act_a)

        def pick(dtree: PyTree, act: tuple[bool, ...]) -> list[jnp.ndarray]:
            return [d for d, a in zip(jax.tree_util.tree_leaves(dtree), act, strict=False) if a]

        ds_act, da_act = pick(dstate, act_s), pick(datm, act_a)
        # One direction at the primal's rank (what `jax.jvp` hands out) rides
        # in as D=1 and is squeezed back out, so a single-tangent caller gets
        # tangents at the primal's rank.
        one_dir = jnp.ndim(ds_act[0]) == jnp.ndim(pick(state, act_s)[0])
        if one_dir:
            ds_act = [jnp.expand_dims(d, 0) for d in ds_act]
            da_act = [jnp.expand_dims(d, 0) for d in da_act]
        final, dfinal_active, tangent_longdy, tangent_ok = self._runner_jvp_cache[key](
            state, atm_static, ds_act, da_act
        )
        self._raise_if_frozen(final, atm_static)
        if one_dir:
            dfinal_active = [jnp.squeeze(d, 0) for d in dfinal_active]
            tangent_longdy = jnp.squeeze(tangent_longdy, 0)
        leaves, tdef = jax.tree_util.tree_flatten(final)
        it = iter(dfinal_active)
        # float0 placeholders keep the primal's shape; nothing reads them.
        dfinal = jax.tree_util.tree_unflatten(
            tdef,
            [
                next(it) if a else np.zeros(np.shape(x), dtype=jax.dtypes.float0)
                for x, a in zip(leaves, act_s, strict=False)
            ],
        )
        return final, dfinal, tangent_longdy, tangent_ok

    def _refuse_missing_atoms(self, state: JaxIntegState) -> None:
        """Refuse a column with none of a conserved atom before its loop; a traced column is checked by its freeze."""
        atom_ini = jax.lax.stop_gradient(state.pv.atom_ini)
        if not isinstance(atom_ini, jax.core.Tracer):
            refuse_missing_atoms(np.asarray(atom_ini), self._atom_order)

    def _check_queue_exit(self, reason: jnp.ndarray, delta: jnp.ndarray, atom_ini: jnp.ndarray) -> None:
        """Refuse or raise from the jobs' returned arrays; traced, say who has to.

        A job's column exists only inside the program, so its atom totals are read here, after the loop.
        """
        if isinstance(reason, jax.core.Tracer):
            logger.warning(_TRACED_CHECK)
            return
        refuse_missing_atoms(np.asarray(atom_ini), self._atom_order)
        delta = jax.lax.stop_gradient(delta)
        raise_if_frozen(np.asarray(reason), np.asarray(delta), rtol=_pack_state._built_statics(self).rtol_accept)

    def _raise_if_frozen(self, final: JaxIntegState, atm_static: AtmStatic) -> None:
        """Raise `DtMinFrozenError` if the exit carry, one column or a stack of lanes, froze at `dt_min`.

        No host callback: a program holding one stays out of JAX's persistent compile cache. Under an outer jit the
        values do not exist yet, so the caller checks them after the jitted call (`raise_if_frozen`).
        """
        reason = final.termination_reason
        if isinstance(reason, jax.core.Tracer):
            logger.warning(_TRACED_CHECK)
            return
        frozen = np.flatnonzero(np.ravel(reason) == TERM_DT_MIN_FROZEN)
        if not frozen.size:
            return
        statics = _pack_state._built_statics(self)
        message = ""
        if jnp.ndim(reason) > 0:
            lane = int(frozen[0])
            message = f"{frozen.size} of {jnp.size(reason)} lanes froze, at {frozen[:8].tolist()}. Lane {lane}: "
            final = jax.tree_util.tree_map(lambda x: x[lane], final)
            atm_static = atm_static._replace(**{f: getattr(atm_static, f)[lane] for f in _ATM_ARRAY_FIELDS})
        # Under a plain jax.jvp the integer reason comes back concrete while the arrays still carry tangents.
        final, atm_static = jax.lax.stop_gradient((final, atm_static))
        failed = self._failed_trial(final, atm_static)  # type: ignore[misc]  # the exit carry came from a built runner
        ratio, sol_clip, loss_change = (np.asarray(x) for x in failed)
        raise DtMinFrozenError(
            message + frozen_column_message(ratio, sol_clip, loss_change, statics, self._species, self._atom_order)
        )

    def _checked_runner(self, state: JaxIntegState, atm_static: AtmStatic) -> JaxIntegState:
        self._refuse_missing_atoms(state)
        final = self._runner(state, atm_static)  # type: ignore[misc]  # the caller holds this through `runner`, built
        self._raise_if_frozen(final, atm_static)
        return final

    def build_refresh_static(self, atm: types.SimpleNamespace) -> _atm_refresh_mod.AtmRefreshStatic:
        """Pack the static inputs to `atm_refresh.update_mu_dz_jax`."""
        return _build_statics.build_refresh_static(self, atm)

    def pack_state(self, rs: RunState) -> JaxIntegState:
        """Build the initial JaxIntegState from a fully-populated RunState."""
        return _pack_state.pack_state(self, rs)

    def _report_end(self, rs: RunState, final: JaxIntegState) -> None:
        """End-of-run report (op.py:1069-1085 and op.stop).

        print_prog, then print_end_msg (end_case 1) or print_unconverged_msg, which states the reason (2 / 3 / 5),
        followed
        by the certificate terms the exit carry `final` fails.
        """
        var, _, para = _state_mod.legacy_view(rs)
        end_case = para.end_case
        if end_case == TERM_CONVERGED:
            logger.info(
                "Integration successful with %s steps and long dy, long dydt = %s, %s\nActinic flux change: %.2E",
                para.count,
                var.longdy,
                var.longdydt,
                var.aflux_change,
            )
        if self._cfg.use_print_prog:
            self.output.print_prog(var, para)
        if end_case == TERM_CONVERGED:
            self.output.print_end_msg(var, para)
            return
        self.output.print_unconverged_msg(var, para, end_case)
        failing = certificate_failures(final, self._cfg, self._atom_order)
        if failing:
            logger.warning(
                "Convergence terms not met at exit: %s",
                "; ".join(
                    f"{term} {value:.3g} (threshold {threshold:.3g})"
                    if math.isfinite(value)
                    else f"{term} (threshold {threshold:.3g})"
                    for term, value, threshold in failing
                ),
            )
        if any(term.startswith("element budget") for term, _, _ in failing):
            logger.warning(
                "The element budget assumes a closed column: boundary fluxes and "
                "bottom or fix_species pins count as drift. For such a column set "
                "element_budget_tol: .inf to stop on VULCAN 2's criteria."
            )

    # Deprecated (docs/api.md).
    _ensure_runner = ensure_runner
    _pack_state_from_runstate = pack_state
    _build_refresh_static = build_refresh_static

    @property
    def runner(self) -> Callable[[JaxIntegState, AtmStatic], JaxIntegState] | None:
        """The compiled `runner(state, atm_static) -> state`; None before `prepare_runstate`.

        It raises `DtMinFrozenError` (see `_raise_if_frozen`); `_runner` is the bare program.
        """
        return None if self._runner is None else self._checked_runner

    @property
    def config(self) -> Config:
        """The `Config` this runner was built with."""
        return self._cfg

    @property
    def compo_arr(self) -> np.ndarray:
        """Per-species atom counts, shape (ni, n_atoms)."""
        return self._compo_arr

    @property
    def photo_static(self) -> _PhotoStatic | None:
        """The runner's baked photolysis tables, or None with photolysis off."""
        return self._photo_static

    @property
    def statics(self) -> _Statics | None:
        """The runner's closure inputs, or None before `ensure_runner`."""
        return self._statics

    def __call__(self, rs: RunState, chunk: int | None = None) -> RunState:
        """Integrate a fresh `RunState` to convergence or a cap and return a new `RunState`.

        Raise `ValueError` if `rs` has already run (`params.count > 0`), and `DtMinFrozenError`, with no result, if the
        column's trial keeps failing at `dt_min`. The caps are runtime and step count.

        `chunk=None` runs one compiled loop. `chunk=N` runs the same loop in
        pieces of N accepted steps, prints a progress block after each (when
        `use_print_prog`), and on Ctrl+C stores the last finished piece as
        `self.interrupted_runstate` before re-raising. One loop cannot be
        interrupted: Ctrl+C reaches Python only when it returns.
        """
        count = int(rs.params.count)
        if count > 0:
            raise ValueError(
                f"OuterLoop needs a fresh RunState (params.count == 0); this one "
                f"has already run {count} steps. Start again from "
                "RunState.with_pre_loop_setup(cfg): a RunState does not carry "
                "the certificate ring, the cumulative element budget "
                "(budget_err / budget_drift) or the hybrid vm_mol phase, so a "
                "resumed run would restart all three from zero."
            )
        self.interrupted_runstate = None
        init_state, atm_static = self.prepare_runstate(rs)
        self._refuse_missing_atoms(init_state)
        if chunk is None:
            final = self._runner(init_state, atm_static)  # type: ignore[misc]  # prepare_runstate built it
        else:
            final = self._run_chunked(init_state, atm_static, rs, int(chunk))
        self._raise_if_frozen(final, atm_static)
        rs_out = _pack_state.unpack_state(self, final, rs)
        self._report_end(rs_out, final)
        return rs_out

    def _run_chunked(self, state: JaxIntegState, atm_static: AtmStatic, rs: RunState, chunk: int) -> JaxIntegState:
        """Drive `runner_chunk` until a chunk ends short of its target.

        Only a real termination does that (a run that ends exactly on a target returns from the next call after
        zero iterations). Returns the final carry.
        """
        import types

        if chunk < 1:
            raise ValueError(f"chunk must be a positive step count, got {chunk}")
        try:
            while True:
                target = int(state.accept_count) + chunk
                target_ = jnp.asarray(target, state.accept_count.dtype)
                nxt = self._runner_chunk(state, atm_static, target_)  # type: ignore[misc]  # prepare_runstate built it
                done = int(nxt.accept_count) < target
                state = nxt
                if done:
                    return state
                if self._cfg.use_print_prog:
                    # print_prog reads only these fields.
                    self.output.print_prog(
                        types.SimpleNamespace(
                            t=float(state.t),
                            dt=float(state.dt),
                            longdy=float(state.longdy),
                            longdydt=float(state.longdydt),
                        ),
                        types.SimpleNamespace(
                            count=int(state.accept_count), where_varies_most=np.asarray(state.where_varies_most)
                        ),
                    )
        except KeyboardInterrupt:
            self.interrupted_runstate = _pack_state.unpack_state(self, state, rs)
            raise

    def prepare_runstate(self, rs: RunState) -> tuple[JaxIntegState, AtmStatic]:
        """Build `(init_state, atm_static)` for one RunState.

        Builds the runner closure on the first call (later calls reuse it; nz is not checked).

        The batched GPU driver calls this per profile, then stacks the
        results along a leading axis into one batch for `run_batch`. All profiles in a single `run_batch` call must
        share
        the same nz / toggle-combo / `pref_indx` so the closure and array
        shapes match; bucket profiles accordingly. `__call__` runs the
        same setup, then the runner.
        """
        validate_runtime_config(self._cfg)
        var, atm, _ = _state_mod.legacy_view(rs)
        # Wire a pre-built PhotoStaticInputs onto the solver so
        # build_photo_static doesn't rebuild from the legacy_view shim
        # (which does not carry the var.cross* dict surface).
        ps, baked = rs.photo_static, self.odesolver._photo_static
        if ps is not None and baked is None:
            self.odesolver._photo_static = ps
        elif ps is not None and baked is not None:
            # The closure bakes the first profile's star + wavelength grid;
            # only T-dependent cross sections ride per lane (ProfileVars).
            # Reject any other difference instead of using lane 0's star.
            same = (
                int(ps.nbin) == int(baked.nbin)
                and int(ps.din12_indx) == int(baked.din12_indx)
                and np.array_equal(np.asarray(ps.bins), np.asarray(baked.bins))
                and (self._sflux_top_ref is None or np.array_equal(np.asarray(rs.photo.sflux_top), self._sflux_top_ref))
            )
            if not same:
                raise ValueError(
                    "run_batch photo lanes must share the star, wavelength "
                    "grid, and network (only the T-P profile may differ); "
                    "this profile's photo statics do not match the first "
                    "profile's."
                )
        self.ensure_runner(var, atm)
        # The refresh closure bakes the first profile's `pref_indx` (it sizes
        # a `jnp.arange`, so it cannot ride ProfileVars); reject a lane that
        # differs instead of silently anchoring it at lane 0's layer.
        baked_pref = int(self._refresh_static.pref_indx)  # type: ignore[union-attr]  # ensure_runner just built it
        if int(atm.pref_indx) != baked_pref:
            raise ValueError(
                "run_batch lanes must share pref_indx: this profile has "
                f"{int(atm.pref_indx)}, the runner was built with {baked_pref}; "
                "bucket profiles by pref_indx before batching."
            )
        ni = NETWORK.ni
        nz = int(rs.atm.Tco.shape[0])
        atm_static = make_atm_static(atm, ni, nz, cfg=self._cfg)
        init_state = self.pack_state(rs)
        return init_state, atm_static

    def run_batch(self, states_batched: JaxIntegState, atm_static_batched: AtmStatic) -> JaxIntegState:
        """Integrate a whole batch of profiles in one vmapped device call.

        `states_batched` / `atm_static_batched` are the stacked outputs of
        `prepare_runstate` (leading batch axis on every array leaf; the four
        AtmStatic toggle flags broadcast). Returns the batched final `JaxIntegState`; read per-lane
        `termination_reason` / `ymix` by indexing the lane axis. Requires `ensure_runner` to have run (via
        `prepare_runstate`) for this batch's (nz, toggle-combo).

        Lanes run with freeze-on-done: each profile's converged result
        matches running it alone at the convergence scale (the photo /
        geometry cadences follow the loop's iteration tick here, not the
        lane's accept count), and the call returns once the slowest
        lane finishes. Lanes are never compacted or refilled here; feed a
        queue of profiles through `run_queue` for that. A lane frozen at `dt_min`
        stops with `termination_reason` 4, and `DtMinFrozenError` follows the loop. Under an outer jit the caller
        checks the returned carry (`raise_if_frozen`).
        """
        if self._runner_batch is None:
            raise RuntimeError(
                "run_batch called before the runner was built; call prepare_runstate on at least one profile first."
            )
        self._refuse_missing_atoms(states_batched)
        if self._vrunner is None:
            self._vrunner = jax.jit(self._runner_batch)
        final_b = self._vrunner(states_batched, atm_static_batched)
        self._raise_if_frozen(final_b, atm_static_batched)
        return final_b

    def run_queue(
        self,
        init_fn: Callable[[PyTree], tuple[JaxIntegState, AtmStatic]],
        jobs: PyTree,
        n_lanes: int,
        out_fn: Callable[[JaxIntegState], PyTree],
        *,
        chunk: int = _QUEUE_REFILL_CHUNK,
        compact: Sequence[int] | None = None,
    ) -> tuple[PyTree, jnp.ndarray]:
        """Integrate `n_jobs` independent profiles on `n_lanes` lanes with refill.

        `init_fn(job) -> (JaxIntegState, AtmStatic)` builds one job's initial
        state (traceable); `jobs` is a pytree with a leading job axis on every
        array leaf; `out_fn(final_lane_state) -> pytree` is what is kept per
        job. Returns `(out, n_iter)`: `out_fn`'s pytree stacked in job order
        and the loop's iteration count. A job whose lane went non-finite comes
        back with `termination_reason` 5. A job frozen at `dt_min` gives up its lane
        like a finished one, and `DtMinFrozenError` follows the loop, naming its index. Under an outer jit,
        `out_fn` keeps `termination_reason` and the caller checks it (`raise_if_frozen`).

        A finished lane is written out and refilled with the next pending job
        once `chunk` lanes are free, when every lane still holding a job is
        free, or every `_QUEUE_REFILL_EVERY` iterations; with `n_lanes >= n_jobs`
        nothing is refilled and the result is bitwise `run_batch`'s. A fresh
        lane starts as a plain-batch lane does at tick 0 (its own photolysis
        fields), so a refilled job matches the plain batch at the convergence
        scale, not bitwise: it enters the loop at a tick that moves the photo
        and geometry cadences. A job stopped by `count_max` is `is_done` like
        a converged one and is refilled. Requires `prepare_runstate` to have built the runner for this nz /
        toggle-combo, like `run_batch`.

        Tail compaction: `compact` (default: the config's `queue_compact`, off
        when empty) is a ladder of narrower lane counts, e.g. `(264, 132)` on
        792 lanes. Once no job is pending and the live lanes fit a rung, they
        continue in a program of that width from the same tick (no refill), so
        the tail of a few slow jobs stops paying for the full width. Freeze-on-
        done keeps a done lane out of every lane-uniform photolysis / refresh
        predicate, so dropping it changes no live lane's step: each job's
        result is the uncompacted one, bitwise wherever the compiled step's
        per-lane arithmetic does not depend on the width (the GPU validation
        checks it). Rungs at or above `n_lanes` are dropped; each rung used
        adds one compiled loop to the program.
        """
        if self._runner_queue is None:
            raise RuntimeError(
                "run_queue called before the runner was built; call prepare_runstate on at least one profile first."
            )
        # The callables are part of the key, not their `id`s: holding them
        # keeps a dead closure's id from being reused by a new one.
        if compact is None:
            compact = self._cfg.queue_compact
        rungs = _queue_compact_rungs(compact, n_lanes)
        key = (init_fn, out_fn, int(n_lanes), int(chunk), rungs)
        fn = self._vrunner_queue.get(key)
        if fn is None:
            if rungs:
                logger.info(
                    "run_queue: tail compaction %s -> %s lanes", int(n_lanes), " -> ".join(str(r) for r in rungs)
                )

            def out_and_stop(final: JaxIntegState) -> tuple[PyTree, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
                delta, atom_ini = jax.lax.stop_gradient((final.delta, final.pv.atom_ini))
                return out_fn(final), final.termination_reason, delta, atom_ini

            fn = jax.jit(
                functools.partial(
                    self._runner_queue,
                    n_lanes=int(n_lanes),
                    chunk=int(chunk),
                    init_fn=init_fn,
                    out_fn=out_and_stop,
                    compact=rungs,
                )
            )
            self._vrunner_queue[key] = fn
        (out, reason, delta, atom_ini), n_iter = fn(jobs)
        self._check_queue_exit(reason, delta, atom_ini)
        return out, n_iter
