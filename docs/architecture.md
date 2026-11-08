# Architecture

## Layout

`OuterLoop` (one JIT-compiled `lax.while_loop`) drives `jax_step.jax_ros2_step`, which calls the generated chemistry RHS, the
analytical Jacobian (`chem.chem_jac_analytical`) and the block solve: `solver_fast.factor` / `solve` by default, or the
reference pair `solver.factor_block_thomas_diag_offdiag` / `solve_block_thomas_diag_offdiag`. The `photo.py` kernels fire
through `lax.cond`. `state.RunState.with_pre_loop_setup(cfg)` runs setup, and `legacy_io.Output.save_out(runstate, dname)`
writes a `.vul` pickle from a `RunState` only.

| Module | Holds |
|---|---|
| `outer_loop` | `OuterLoop` |
| `_carry` | the carry and closure pytrees, what the step closes over, field lists, batch axes |
| `_convergence` | the ring lookback, longdy, the convergence test and certificate, the stop test |
| `_step_control` | clip, aggregate truncation error, adaptive rtol, next step size |
| `_loop_body`, `_loop_branches` | one trial step, in program order; its photolysis and condensation branches |
| `_runners`, `_runner_jvp`, `_runner_queue` | solo, chunk and batch; forward mode; queue and compaction |
| `_build_statics`, `_pack_state` | setup into the runner's inputs; `RunState` to carry and back |
| `steady_state_grad`, `_adjoint_core`, `_adjoint_audit` | public sensitivities; the solve; the scope audit |
| `atm_setup`, `_tp_kzz`, `_transport_setup`, `_stellar_flux`, `_saturation` | `Atm` facade and setup pieces |
| `state`, `_legacy_containers` | pytrees and the pre-loop pipeline; the legacy views |
| `_settings` | the one reader of `VULCAN3_*` |

## Import order

`import vulcan3` enables x64 and freezes the network: `chem_funs` parses it, `composition` reads the tables and
`make_chem_funs` emits the RHS. `jax_step` and `solver_fast` load later, so `$VULCAN3_SOLVER` stays open after the package
import; `tests/test_import_laziness.py` pins this. Consumers that need another network set the variables before the first
import, and `cli_main` re-execs once to match a config. `docs/settings.md` lists the variables.

## Generated RHS

`make_chem_funs.emit_chem_rhs_source` writes Python and compiles it. A data-driven RHS lets XLA rewrite `H*H` as
`exp(2*log(H))`, which moves bits. `VULCAN3_CHEM_RHS=grouped` is the data form, opt-in until it is bitwise in full CPU runs.

## Where this code departs from the general standards

Each row is deliberate.

| Standard | What stays | Why |
|---|---|---|
| Names say what a thing is | Master's symbols and classes (`glossary.md`) | A faithful port: master's `plot_py/` and the facades read them |
| Modern file formats | `.vul` pickle with master's three keys | Master's plotting code loads it |
| No code from strings | `emit_chem_rhs_source` and its `exec` | See "Generated RHS" |
| Import does no work | x64 and the frozen network and tables | notes "Do not retry": a pre-import launcher, an in-process hot-swap and conftest reordering all failed. vulcan-forward owns the order |
| No hidden environment variables | `VULCAN3_NETWORK`, `_ATOM_LIST`, `_COM_FILE`, `_SOLVER`, `_CHEM_RHS`, `_BT_BUFFERS` | A contract with vulcan-forward, vulcan-retrieval and jwst-transit-authority; documented in `settings.md` |
| No history in comments | `op.py:NNN`, `exoclime@80f75b9` | Provenance citations to the pinned clone are required by the parity rules |
| No duplicated logic | Scans and kernels listed in notes "Do not retry", Bit-moving refactors | Folding them moved bits; each has an agreement test |
| Fix a finding in the code | Items in notes "Parity ledger" and "Decided by the maintainer, not to be re-opened" | Settled parity decisions |
| Size limits | Kernels, with `size-limit-ok` | notes "Do not retry", kernel restructurings; a logic edit needs a GPU run |
| pytest style | `main()` plus `run_oracle_subprocess` | Master must run in a subprocess, on a copy |
| A cleanup proof replaces checks | The normal tier still runs before a push | The repository's Tests rule |
| `dev` as a dependency group | The `dev` extra stays, beside the groups | The workspace and `tests.yml` install `vulcan3[dev,plot]` |

## Known exceptions

- `state._cfg_overlay` writes a caller's config over the process default during setup, and `_CFG = default_config()` is read in
  eight modules. Threading `cfg` through setup is a separate job.
- `tests/conftest.py` must snapshot that default around every test for the same reason.
