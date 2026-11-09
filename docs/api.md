# Public API

Consumers import the names in each module's `__all__`. `tests/test_public_api.py` checks a hand-kept list of the names and
`OuterLoop` attributes consumers read.

The top level exports `Config`, `DtMinFrozenError`, `MissingAtomError`, `RunState`, `__version__`, `conv_normal`, `default_config`,
`load_config`, `make_config`, `raise_if_frozen` and `validate_overrides`. A refused config raises `runtime_validation.RuntimeConfigError`, a
`ValueError` and a `RuntimeError`.

A column holding none of a conserved atom (in `atom_list`, not in `loss_ex`) raises `MissingAtomError`, a
`RuntimeConfigError`, naming the atom: at setup in `RunState.with_pre_loop_setup`, so in the CLI too, and for a changed state at
the entry of `OuterLoop.__call__`, `runner`, `run_jvp` and `run_batch`, by lane index. `run_queue` builds its columns inside the
program, so it names the job after the loop. Under an outer jit the lane freezes (below).

A column whose trial fails twice at exactly `dt_min` with no accept between keeps its state, so `OuterLoop.__call__`,
`OuterLoop.runner`, `run_jvp`, `run_batch` and `run_queue` raise `DtMinFrozenError` (a `RuntimeError`) after the loop, with no
result, naming species, layer, ratio and `rtol` where the exit carry is at hand (`run_queue`: the job index). The exit carry
reads `termination_reason` 4 (`state.TERM_DT_MIN_FROZEN`) and `conv_normal` False. No jitted program holds a host callback,
which would keep it out of JAX's persistent compile cache, so under an outer `jax.jit` or `jax.vmap` nothing can raise: keep
`termination_reason` in what it returns and call `raise_if_frozen(termination_reason, delta, rtol=...)` right after it. A
trace that skips the check logs a warning. `OuterLoop._runner` is the bare program, for a caller that wants the carry.

## Deprecated names

The packages at their main branches use the public names. Released versions of vulcan-forward and jwst-transit-authority,
which floor `vulcan3>=0.18.0`, read the old ones, so these aliases stay for one more release. No warning is raised.

| Deprecated alias | Use |
|---|---|
| `OuterLoop._ensure_runner` | `OuterLoop.ensure_runner` |
| `OuterLoop._pack_state_from_runstate` | `OuterLoop.pack_state` |
| `OuterLoop._build_refresh_static` | `OuterLoop.build_refresh_static` |
| `chem_funs._NETWORK`, `_NET_JAX` | `chem_funs.NETWORK`, `NET_JAX` |
| `outer_loop._ATM_STATIC_BATCH_AXES` | `outer_loop.ATM_STATIC_BATCH_AXES` |
| `atm_setup._VISCOSITY_TABLE` | `atm_setup.VISCOSITY_TABLE` |
| `atm_jax._mol_diff` | `atm_jax.mol_diff` |
| `rates_jax._thermo_dir` | `rates_jax.thermo_dir` |

## Internals behind a public name

These are the objects' own attributes and modules, not aliases, and may change. Read the public name.

| Internal | Public |
|---|---|
| `OuterLoop._runner` | `OuterLoop.runner`, which adds the `DtMinFrozenError` check |
| `OuterLoop._cfg` | `OuterLoop.config` |
| `OuterLoop._compo_arr` | `OuterLoop.compo_arr` |
| `OuterLoop._photo_static` | `OuterLoop.photo_static` |
| `OuterLoop._statics` | `OuterLoop.statics` |
| `Ros2JAX._photo_static` | `Ros2JAX.photo_static` |
| `vulcan3._paths.resolve_data_path` | `vulcan3.paths.resolve_data_path` |

## Removed names

`steady_state_grad.LGMRES_*` are removed: rebinding them never reached the solve, whose settings live in `_adjoint_core`.
`steady_state_grad.PHOTO_RECOMPUTE_AUTO` and the `photo_recompute_k="auto"` spelling are removed: they meant the same as
`None`, the default.
