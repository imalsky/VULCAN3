# Settings

Run settings are config keys: `configs/default.yaml` lists them, and `config.py` and `runtime_validation.py` hold the refusals.
A flag (a key whose `default.yaml` value is true or false) also takes 0 or 1, as VULCAN 2's `== True` reads it, and loads as a
bool; any other value is refused.
These environment variables are read once in the process, and a bad value is refused. The first three must be set before the
first `import vulcan3`; `cli_main` re-execs once to match a config, the Python API does not. When one of them
replaces a value that the loaded config file names (not the packaged `default.yaml`), `load_config` logs a WARNING
naming both values; the run uses the variable's.

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
The command sets `VULCAN3_CLI_RELAUNCHED` on its own re-exec, so that a second mismatch raises and does not loop. It is not a user setting.

Negative trial values: every finite negative is zeroed and counted in `nega_y`, as VULCAN 2 does. A trial that drives a populated
gas (pre-step mixing ratio at least `mtol` and density at least `atol`) negative is rejected, because the error estimate skips
negative cells; -inf and NaN fail the step (notes "Parity ledger", C29).
The former `nega_cut` key is refused with this remedy.

`use_botflux` reads `bot_BC_flux_file`, and a row naming a species outside the network is refused. The shipped
`atm/BC_bot_Earth.txt` names H2SO4, NO3 and HNO3, which the default network lacks: `Earth.yaml` runs it on
`SNCHO_full_photo_network.txt`, which holds them. With another network, copy the file and delete those rows.

`dt_min` (s, default 1e-15, master's is 1e-14) is the shortest step. A trial that fails there is force-accepted and keeps the previous state, so
a second such failure stops the run with `DtMinFrozenError`, which a caller that jits its solve raises itself (docs/api.md).
Each force-accept is counted, and a run that recovers from one ends with a WARNING giving the count, where VULCAN 2
prints a line per force-accept.
