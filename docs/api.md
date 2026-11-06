# Public API

Consumers import the names in each module's `__all__`. `tests/test_public_api.py` checks a hand-kept list of the names and
`OuterLoop` attributes consumers read.

The top level exports `Config`, `RunState`, `__version__`, `conv_normal`, `default_config`, `load_config`, `make_config` and
`validate_overrides`. A refused config raises `runtime_validation.RuntimeConfigError`, a `ValueError` and a `RuntimeError`.

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
| `OuterLoop._runner` | `OuterLoop.runner` |
| `OuterLoop._cfg` | `OuterLoop.config` |
| `OuterLoop._compo_arr` | `OuterLoop.compo_arr` |
| `OuterLoop._photo_static` | `OuterLoop.photo_static` |
| `OuterLoop._statics` | `OuterLoop.statics` |
| `Ros2JAX._photo_static` | `Ros2JAX.photo_static` |
| `vulcan3._paths.resolve_data_path` | `vulcan3.paths.resolve_data_path` |

`steady_state_grad.PHOTO_RECOMPUTE_AUTO` and the `photo_recompute_k="auto"` spelling are removed: they meant the same as
`None`, now the default.
