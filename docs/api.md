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

## Entry points

One row per name in `vulcan3.__all__`.

| Name | Signature | Purpose |
|---|---|---|
| `Config` | `class Config(types.SimpleNamespace):` | Free-attribute config namespace (attribute access like master's `vulcan_cfg`). |
| `DtMinFrozenError` | `class DtMinFrozenError(RuntimeError):` | A column's trial keeps failing at `dt_min`, so its state does not move and more steps repeat the failure. |
| `MissingAtomError` | `class MissingAtomError(RuntimeConfigError):` | A column holds none of a conserved atom, so its relative loss is 0/0 and no trial could ever be accepted. |
| `RunState` | `class RunState(NamedTuple):` | Umbrella pytree carrying every input the runner reads. |
| `__version__` | `__version__ = "<version>"` in `_version.py`, a string | The package version. |
| `conv_normal` | `def conv_normal(final: JaxIntegState, cfg: Config) -> tuple[jnp.ndarray, jnp.ndarray]:` | Return the runner's convergence certificate on a carry, e.g. its exit state. |
| `default_config` | `def default_config() -> Config:` | Return the process-wide default Config (loaded once from `configs/default.yaml`). |
| `load_config` | `def load_config(name_or_path: str \| os.PathLike = "default", /, **overrides: Any) -> Config:` | Load a YAML config, apply env + caller overrides, resolve derived values. |
| `make_config` | `def make_config(**overrides: Any) -> Config:` | Load the default config with user overrides (alias for `load_config`). |
| `raise_if_frozen` | `def raise_if_frozen(termination_reason: np.ndarray \| jnp.ndarray, error_ratio: np.ndarray \| jnp.ndarray \| None = None, *, rtol: float \| None = None) -> None:` | Raise `DtMinFrozenError` if any column stopped frozen at `dt_min`. |
| `validate_overrides` | `def validate_overrides(overrides: dict[str, Any]) -> None:` | Refuse removed or unknown keys in an override dict, as `load_config(**overrides)` does. |

`conv_normal` is defined in `_convergence.py` and `make_config` in `__init__.py`. The source splits the parameters of `raise_if_frozen` over several lines.

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
