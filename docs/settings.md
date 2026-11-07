# Settings

Run settings are config keys: `configs/default.yaml` lists them, and `config.py` and `runtime_validation.py` hold the refusals.
These environment variables are read once in the process, and a bad value is refused. The first three must be set before the
first `import vulcan3`; `cli_main` re-execs once to match a config, the Python API does not.

| Variable | Values | Default | Meaning |
|---|---|---|---|
| `VULCAN3_NETWORK` | path under the package or absolute | the config's `network` | reaction network |
| `VULCAN3_ATOM_LIST` | comma-separated elements | the config's `atom_list` | elements of the network |
| `VULCAN3_COM_FILE` | path | the config's `com_file` | composition table |
| `VULCAN3_SOLVER` | `fast`, `ffi`, `reference` | `fast` | block solve, read when `jax_step` first loads. `ffi` is the kernel, `reference` the plain pair |
| `VULCAN3_CHEM_RHS` | `codegen`, `grouped` | `codegen` | chemistry RHS |
| `VULCAN3_BT_BUFFERS` | `1`, `2` | per call | the CUDA solve's one- or two-buffer choice, read once by the kernel library |

Build tools read `CXX`, `NVCC` and `NVCC_ARCH` in `solver_fast.build`. The CLI uses `JAX_COMPILATION_CACHE_DIR`, by default
`~/.cache/vulcan3`. A variable with the old `VULCAN_JAX_` prefix raises at import.

`nega_cut` (cm^-3, default -1e4) is an absolute number density: a trial value from it up to 0 clips to 0, a value below it rejects
the step. It is one threshold for the whole column, not a share of the layer: -1e4 is -7e-7 of HD189's top layer (n 1.45e10) and
-3e-18 of its bottom. A share of the layer would be far looser at depth. notes "Forward mode" has the reason for the value.

`use_botflux` reads `bot_BC_flux_file`, and a row naming a species outside the network is refused. The shipped
`atm/BC_bot_Earth.txt` names H2SO4, which no shipped network holds, and NO3 and HNO3, which the default network does not, so
copy the file and delete those rows.

`dt_min` (s, default 1e-14) is the shortest step. A trial that fails there is force-accepted and keeps the previous state, so
a second such failure stops the run with `DtMinFrozenError`, which a caller that jits its solve raises itself (docs/api.md).
