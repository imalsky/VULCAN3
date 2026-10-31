# VULCAN3 notes

What is true about the solver (package `vulcan3`, folder `VULCAN-JAX`) and why: the
results of earlier experiments and the ideas already tried, by topic, so that work is
not repeated; the evidence behind each CLAUDE.md rule; design records; the parity and
upstream-error ledger; superseded designs and the do-not-retry register. A run is
recorded by its result and its name, not its date; git history is the log. Open work
is in `todo.md`; retrieval-side studies are in vulcan-retrieval's `notes.md`.
[M] measured, [E] estimated, [I] inferred. Upstream citations (`op.py:NNN`,
`build_atm.py:NNN`) are the pinned clone exoclime@80f75b9 in
`VULCAN_Project/.oracles/vulcan2_ncho`, or vm_branch@84d010d where marked. A
reference to `analysis/<name>` means the `<name>` entry here or in vulcan-retrieval's
notes (the NAS `analysis/` tree no longer exists).

## Current state

- 0.18.0 on main; the last solver change is `dba59fd`. Tags `v0.17.0`, `v0.18.0`
  (`760d96c`, the rename plus the W39b baseline merge), `w39b-baseline-v2`. Package,
  distribution, CLI and env prefix are `vulcan3` / `VULCAN3_*` since 0.18.0 (GitHub
  `imalsky/VULCAN3`); earlier releases were `vulcan-jax`. 0.18.0 is on TestPyPI, not
  on PyPI [I].
- Consumers floor `vulcan3>=0.18.0`. Production: vulcan-retrieval's WASP-39b on NAS
  GH200s (2025 SNCHO file, ni 93; v5 ran the 2021 file, ni 89; `ffi`, grouped RHS).
- Step-count gates (`tests/test_step_count_gates.py`, full tier): HD189 1495, HD209
  1211, W39b 1260 (1202 on the 2021 file, as below) on arm64 and x86-64. HD189_vulcan3
  is not in that test: 2603 on arm64; on x86-64 it ends on its phase-1 budget at 4018.
- Tests [M], daw, `-n 16`, no GPU: normal tier (`python -m pytest tests`, budget 10 min)
  322 passed / 61 skipped in 5:47; full tier (`VULCAN3_RUN_SLOW=1`, oracle on) 339
  passed / 32 skipped in 18:01 before the T-range restore, plus the W39b gate re-pinned
  (3 of 3 pass); the skips are the CUDA kernel and compute-sanitizer tests. `oracle.yml`
  is `workflow_dispatch` only; last green dispatch 37173830150 at `b372e65`.
- Oracle clones: `vulcan2_ncho` 80f75b9, `vulcan2_fastchem_ps_order` 8970337
  (provenance only, no test), `vulcan3_vm_branch` 84d010d. On daw the clones are
  clean at those commits with `core.fileMode false`, so
  `VULCAN_MASTER_DIR=~/VULCAN_Project/.oracles/vulcan2_ncho` works.
- Fixtures: the tracked `FIXTURES.json` is the arm64 (laptop) manifest of a rebuild at
  `2441e6c`; daw's `tests/data/*.npz` are an x86-64 rebuild of the same solver code
  (`73fd25a`, which changes only docs and the manifest) and differ bitwise, so `gen_fixtures.py --verify` fails on daw by design (restore the tracked
  manifest after a daw rebuild). A rebuild takes about a minute.

## Efficiency program

GH200 rows are the v2 production shape (W39b SNCHO, nz 62, ni 89); other rows name
their host. Per-commit record: `git log`. Batched runs are not bitwise equal to
solo runs (maintainer decision); the parity gates are solo runs and every change
kept them.

| Mechanism | Default | Results | Measured gain |
|---|---|---|---|
| `fast` solver: `solver.py` scans in `lax.custom_linear_solve`; tangent `A dx = db - dA x` on the primal factors | on | primal bitwise vs `reference`; tangent 1.4e-9..1.3e-6 vs AD through LU | one step's jvp 1.8x (CPU) |
| `ffi` solver: C++ / CUDA block-Thomas through XLA FFI | opt-in; production GH200 | roundoff, residual-matched; gates kept; reverse mode unsupported | GPU rows below |
| Gauss-Jordan factor + reciprocal-pivot solve, `d992314` (the LU fallback is removed: CUDA `ffi` takes ni <= 96, else `fast`) | on under `ffi` | layer 0 bitwise, same permutation; dlogL median 9e-8 | 1.24x per step, 75.2 -> 60.7 ms/iter at 396 lanes |
| REDUX pivot search, `bt_div_by_rcp`, cp.async solve loader, `a5b1215` | on under `ffi` (sm_80+) | bitwise | factor 29.1 -> 23.1 ms, solve 5.3 -> 4.6 ms at 792 lanes; pass 1 343.8 -> 300.6 s |
| Solve guard: divide when a layer's pivot leaves [2^-1000, 2^1000], `2e947dc` | on under `ffi`; kept at its cost (the maintainer decided) | bitwise vs `bd97791` at 792 lanes (lu, perm, x, xs; W39b ni 89 and random blocks; NAS jobs 96222, 96238; the A/B script `tools/ab_cuda_solve.py` is in git at `770af2e`); a 2^-1030 pivot solves exactly (old kernel: nan, inf) [M] | solve +6.6% W39b (11.14 -> 11.87 ms), +8.1% random (4.52 -> 4.88 ms), two jobs within 0.1 ms [M]; ~1% of a v4 batch [E]. The trees also differ by `2216219` (NB 4 variant removed; NB 3 registers 76/79 -> 80/80, no spills) |
| One shared buffer per kernel, `a6e8498` (`VULCAN3_BT_BUFFERS`) | chosen per call | bitwise | shared memory at ni 89 127.7 -> 64.4 kB, so 3 blocks/SM on GH200; daw factor 1.22x, solve 1.83x lanes/ms |
| Solve substitution vector in registers, `0261b4a` | on under `ffi` | bitwise | daw primal lane-step 1.11-1.48x, tangent solve 2.59-2.64x |
| 792 lanes (vulcan-retrieval `FFI_BLOCKS_PER_SM = 3`) | run config | n/a | ms/iter 264:25.8 396:40.0 528:49.7 792:67.2 1188:104.4; queue throughput 2.03x stock-396 |
| Column-layout analytical Jacobian (`chem_jac_entries`), `3154d6d` | always | vs the per-layer form vmapped (since deleted; the measurement stands) [M, CPU]: HD189 bitwise; W39b 39 of 1.19M entries 1 ulp apart, all <= 7e-19 of their layer's peak (same with an 8-lane vmap) | with the grouped RHS 58.9 -> 54.6 ms at 396 lanes |
| Grouped chemistry RHS (`VULCAN3_CHEM_RHS=grouped`), `3154d6d` | opt-in; production uses it | bitwise on GH200 (job 93028). CPU [M]: the RHS alone is bitwise vs codegen at the HD189 and W39b pre-loop states, solo and 8-lane vmap; HD189 solo + 8-lane vmap is pinned by `test_chem_rhs_codegen.py::test_grouped_rhs_is_bitwise_codegen` (passes on x86-64: daw, CI). On Apple Silicon (arm64 macOS) the pin is an expected failure: roundoff only, 1.3e-15 of each species' peak on the perturbed lanes with both paths as close to the NumPy oracle, and at the unperturbed near-equilibrium state both 2e-2 of the net peak from it (cancellation); the commit's CPU last-bit differences (1.3e-14 per cell, two 1e-12 tests) come from inside full runs and were not re-checked | as above (GH200, from the commit) |
| Freeze without whole-carry ring copies; photolysis cond spans the step, `309cca9` | on | bitwise except `budget_drift` (<= 4e-5 relative; no certification changed) | 75.0 -> 58.9 ms/iter at 396 lanes |
| y ring written after that cond (`body_fn(defer_ring=True)`), `dba59fd`: a cond returning the written ring made XLA:GPU build the scatter out of place and copy it back, two 18.8 GB passes per 528-lane tick | on (batched) | bitwise: CPU A/B (refill, compaction, wraparound, evolution, hybrid), GH200 at 528 and 132 lanes, a 4-batch v5 run (per-draw steps, nautilus points and log L) | v5 shape, GH200: 110.9 -> 90.4 ms/tick at 528 lanes, 35.9 -> 30.8 at 132; v5 warm batch 848 -> 768 s |
| Tail compaction (`queue_compact`), `01f1558`, `d7021cb` | off (`queue_compact: []`) | bitwise per job in tests; the retry pass at width 320 is roundoff-level | pass 1 520 -> 344 s |
| Compacted tail loops run the main body, `eeb4873` | with compaction | bitwise (2376-job batch) | removes a 5.6 GB/tick ring copy at 264 lanes; with vulcan-retrieval's speed2 changes pass 1 344.0 -> 287.3 s |
| Matrix-free stage operator (`jax_ros2_step(matrix_free=True)`) | forward mode; reverse passes `False` | primal bitwise on CPU; tangent within 8.5e-5 of dense | CPU gradient step 4.49x -> 1.58x the primal; GH200 1.21-1.27x (job 79534); the VJP is 1.9-3.2x slower, hence off in reverse |
| Batched `while_loop` above `vmap(axis_name="lane")`, conds on a `psum` predicate | `run_batch`, `run_queue` | lanes bitwise independent of width; batch vs solo differs (HD189 1337 vs 1495 steps, composition 5.4e-5) | photolysis no longer runs on every lane each iteration (a vmapped `lax.cond` lowers to `select_n`) |
| `run_queue` lane refill | API | bitwise `run_batch` if `n_lanes >= n_jobs`; else 5.1e-13 photo off, 7.0e-4 on | see the lane-count row |
| `_QUEUE_STOP_FIELDS` (stop gradient of certificate-only lane fields) | on | bitwise | XLA:CPU temp 588 -> 330 MiB at 2 lanes |
| `run_jvp` direction axis | on | D=6 primal bitwise; stacked vs single 1.9e-16 | retrieval cold gradient 2.74x the primal at 132 lanes (job 79954; op-count floor ~1.5x) |
| Gather-table analytical Jacobian (`chem._jac_gather_tables`, `_left_to_right_sum_axis1`) | on | bitwise vs the old chunked scatter on every case measured (not a general identity) | assembly 2.3-2.5x, primal step 1.3x (CPU); GH200 primal 1.13x, gradient 1.03x (job 79666); temp buffers halved |
| Repair tridiagonal as rolled pivoted scans (`jax_step._tridiagonal_solve`, unroll 8) | on | LAPACK dgtsv to roundoff; bitwise on the gates | GH200 `ffi` gradient 343.0 -> 151.8 ms/step at 132 lanes (job 79954) |
| Two-stream sweeps as `lax.associative_scan` | on | 1.9e-16 vs the sequential sweep | not recorded |
| ExoGibbs EQ seed per lane (`custom_vmap` + `lax.map`) | on | lane bitwise independent of batch width | CPU batched seeding 1.0-1.5x slower (accepted) |

Kernel limit [M]: the factor is latency-bound on the serial pivot chain (89 steps
per layer x 62 layers), at ~7% of FP64 peak (2.4 TFLOP/s useful of 33.5),
occupancy 25%, barrier stalls 42%. One pivot step ~1500 cycles before `a5b1215`,
838 after; the column chain alone has a floor of 623. Blocked, panel and
tensor-core variants were not faster (bgj8sf 29.3 ms, bgj8s 33.7, bgjl8 76.5),
although DMMA reaches 64 TFLOP/s in isolation. ~60% of the remaining factor time
is the pivot chain [E]: further gains need fewer serial steps. At nz 100 (`a420570`,
GH200, nsys of vulcan-retrieval job 94946) the factor runs 3.0 TFLOP/s at 528 and 264
lanes, 2.5 at 396 (2 blocks/SM, so a half-empty 2nd wave), 1.8 at 132 [E].

Steps to certification (HD189) [M]: 1054 of 1495 steps are model time at the
3.775e4 s plateau, ~441 the ramp from dttry 1e-10 plus five dt collapses on
photolysis refreshes. The live levers are the photolysis cadence and the
certificate window (a science decision).

## Grid resolution

A real VULCAN discretization effect, not a port bug.

- At 62 layers (6.2/decade over 1e-9..7.6 bar) the grid smears a narrow SO2
  depletion below a sharp OH front (~2e-4 bar) and over-produces SO2 by up to ~1
  dex at 1e-4..3e-3 bar. Only SO2 and SO change; convergence order p = 2.09 [M].
- Unmodified numpy VULCAN master (exoclime c870b1a) reproduces it on three
  posterior draws: SO2 column (1e-4..1e-1 bar) master minus vulcan3 <= 0.002-0.005
  dex at every nz; 62 -> 300 layers moves draw 117 by -1.713 dex in both codes,
  draw 301 by -0.072 in both, draw 345 -0.878 (master) vs -0.914 (vulcan3).
  Master's own shipped W39b example is already converged (75 vs 600 layers: <=
  0.03 dex). (vulcan2_xcheck)
- Low-Kzz "bistability": two locally stable states that differ by where the
  photochemical front (H2 -> H, H2O photolysis, 1e-7..1e-6 bar) sits, by whole
  grid layers. At 100 layers 35/40 split lanes merge, at 150 layers 40/40. A
  62-layer grid artifact, not physics. (bistable_policy)
- At nz 150 the vertical scheme is sound [M]: a manufactured tracer on a
  15x-stretched grid gives eddy order 1.95-1.99, central moldiff 2.00 (upwind
  1.00); HD189 nz 150 vs 597: CH4 5.5e-2, NH3 1.7e-2, ~1% at p95 at observable
  pressures.
- Non-uniform pressure grids are not supported (no pco-file input; "Ideas not
  adopted").

## Differences from VULCAN master

Every place vulcan3 departs from the pinned VULCAN 2, with its recorded reason.
Upstream errors corrected in the port are in "Parity ledger". The provenance
audit covered only batch A (e46234a..7eda1af, 60 commits); the later 328 commits
(batches B-D) are not audited and not owed. The maintainer keeps every departure
below ("Decided by the maintainer, not to be re-opened"), including those marked
"provenance not recorded". Lines at `93896f6`.

Convergence and certificate
- Geometry term C21 (`geom_conv_tol` 1e-3; `default.yaml:176`, `W39b.yaml:170`;
  added in `001082e`, 0.7.0). Reason: upstream stops on chemistry and flux alone
  (`op.py:1060-1069`) with mu/dz refreshed every `update_frq` (100) steps
  (`op.py:904`); HD209 photo-off certified with mu 26.3% off ("Certificates").
- Element-budget term C23 (`element_budget_tol` 1e-2; `default.yaml:180`,
  `W39b.yaml:171`; added in `46313d1`, 0.9.0). Reason: `loss_eps` bounds only the
  per-step change of the atom sum (`op.py:2480-2495`), so a slow drain passes it; a
  W39b column certified having lost 21% of its sulfur. The stage repair drops
  oversized corrections and misses the moldiff boundary rows, so this is the one
  check on the finished column. Closed columns only ("Stage repair and element budget").
- Tangent term C22 (`OuterLoop.run_jvp` only). Reason: `jax.jvp(runner)` from a
  warm start stops at `count_min` with the tangent unrelaxed (AD/FD corr 0.33).
- `conver_ignore`: parity configs ship `[]`, as upstream's HD189 example
  (`cfg_examples/vulcan_cfg_HD189.py:148`); HD189_vulcan3 ships `['HC3N']`, as
  upstream's `vulcan_cfg.py:153`; equivalent on the gates. `mtol_conv` 1e-20
  equals that example (:139). Against master's W39b example (exoclime c870b1a,
  not on daw) the NAS audit lists "HC3N not ignored" and `mtol_conv` 1e-20 as
  stricter.
- Removed JAX-only rules ("Superseded designs"): the `use_conv_stall` stall rule,
  c957288's accept test on the adapted rtol (reverted), bf01cd6's W39b default with
  `wall_clock_max` 1800 s (reverted), 9c2bb75's `conver_ignore` (now `[]`),
  `use_vm_mol` true in W39b (now false), the Gustafsson PI controller.

Integrator
- Atom-conservation projection of the chemistry RHS and Jacobian onto fixed
  reservoirs (H2, H2O, CO, N2, H2S): `jax_step._project_chem_rhs` /
  `_project_chem_jac` (`jax_step.py:116-135`, applied at :141, :650, :707); on
  whenever the network carries the reservoirs (every shipped config). Reason: XLA
  FMA reordering leaves a per-layer atom residual (~5e-13 per step). Effect:
  HD189 median difference to upstream 7.3e-6 with it, 3.9e-4 without. From
  ccc40f7; provenance not recorded.
- Ros2 stage repair C19 (`jax_step._repair_stage`, `jax_step.py:226`, called at
  :720, :738; no knob) and `config.DT_MAX_S` 1e15 (upstream `runtime*1e-5` =
  1e17). Reason: stage vectors leak element content at large dt; pinned upstream
  fails the same column (S +32%). From later, unaudited commits.
- Retry cap with force-accept (`batch_max_retries` 110, C11;
  `outer_loop.py:1077-1080`). Inert: at `dt_var_min` 0.5 the 97th consecutive
  reject from `DT_MAX_S` 1e15 (the 84th from the retrieval's 1e11) would take dt
  below `dt_min` 1e-14, and that ends the chain first, as in master. Kept as the
  loop guard (a jitted `while_loop` cannot raise). Provenance not recorded.
- `default.yaml` rtol 0.2, use_adapt_rtol false, count_max 1e4 (from 3938cf9):
  [M] rtol and count_max equal upstream's HD189 example
  (`vulcan_cfg_HD189.py:151`, `:136`); that example has no `use_adapt_rtol` line
  and `op.py:836` reads it unconditionally.

Inputs and data
- `thermo/solar_element_abundances.dat` replaced in ccc40f7, still the default:
  He and S changed, P/Si/Ti/V/Cl/K/Na/Mg/F/Ca/Fe at -3.0: the networks have no Mg/Si/Fe
  species, and upstream FastChem's SiO and Fe(OH)2 hold 17.3% of the O (W39b, 10x, rocky
  10x) [M]; the ExoGibbs seed sees network elements only. Upstream's file is kept as
  `..._lodders2009.dat` with the P/S rows corrected (C12). For W39b the config's
  `<X>_H` override the file (`ini_abun._base_vector`), so only He comes from it.
- EQ seed: ExoGibbs on the port's NASA-9 data, not FastChem (one dataset with the
  reverse rates; no aarch64 FastChem). On identical inputs it matches upstream FastChem
  to ~1e-6 dex (HO2 aside); seeded from either, HD189 and W39b keep 1495 / 1202 steps
  and agree to <= 3e-3 dex, median 4e-5 [M]. The gap is upstream's input rounding. It
  converges 225-6000 K; below, it breaks down (NaN residual, not the cap) at scattered T
  on every network, where FastChem converges 150-300 K, fails at 100 K [M]; untracked.
- Data corrections: C1 (= e841d77, SNCHO reaction 949 k0 1e-20 -> 1e-29, which
  upstream also fixed), C4, C5, C6, C7, C12, C14 (networks renumbered), C20 (both
  sides), P18.
- Gravity is `G Mp / Rp^2`; no `gs` knob. `HD189*.yaml` ship the VULCAN 2
  gs-matched `Mp` (`2140.0*Rp**2/G`, Tsai et al. 2017 sec 3.1).

Refused or not ported (no physics change)
- Not ported (scope: master's live Ros2 physics): `chem_funs.symjac` /
  `neg_symjac`, exact `make_bins_read_cross` (replaced by
  `photo_setup._build_photo_static_dense`), non-Ros2 solvers, byte-identical
  pickles, gradients through raw readers, FastChem internals, in-process network
  hot-swap, `use_print_delta`, `fix_species_time`, `gs`, `wall_clock_max`.
- Display-only master keys (`config._IGNORED_KEYS`) warn and are dropped; keys
  that change a run are refused with a remedy (`config._REMOVED_KEYS`).
- Refused where upstream is silent or fails late: an unknown condensate (a silent
  no-op upstream), `dbin_12trans` outside the bin range (C15b), duplicate
  reactions, unparsable or radiative network rows, an `atom_list` element no
  species carries (C13), a non-network `const_mix` key.
- The network id column is output; photolysis is indexed by parser position
  (C14). `.vul` keeps master's keys, shapes and dtypes, not byte identity.

Retrieval per-run settings (vulcan-retrieval's W39b runs vs master's W39b
example): fixed rtol 0.05 (master 0.05 adaptive), cap-and-reject at 5000 accepted
steps (master stops at 1e4-2e4 and saves the unconverged state), dt_max 1e11
(1e17), photolysis keyed to ticks, nz 100 over 1e-9..7.6 bar (master: 150 over
5e-9..50 bar).

Referee verdicts. An outside review (at `93896f6`) and a second referee checked the
departures above; their numbers are their own [M]. Their measurements are optional.
- Atom-conservation projection: keep. On x86 the worst normalised defect went
  8.7e-14 -> 4.4e-16, but 126 of 360 individual residuals got worse; measure on/off
  on real columns.
- Ros2 stage repair: keep. A 50%-of-carrier correction is inside the designed cap
  (`_REPAIR_MAX_CELL_FRAC` 1.0). Both ask for on/off on real draws plus a
  |correction|/carrier histogram.
- `DT_MAX_S` 1e15: keep; a `dt_max` sweep 1e11..1e17 on difficult columns is optional.
- Geometry and element-budget terms: the outside review would demote both to
  diagnostics under the original-criteria rule. The second referee keeps both: each
  fixes a measured defect ("Certificates", "Stage repair and element budget"); every
  relaxation of VULCAN's gate was dropped while tightenings that fix a confirmed
  defect were kept; stopping on upstream's rule and reporting the two bits is worse
  than either. On W39b, relaxing the budget rescued nothing (vulcan-retrieval notes
  "Runs", failure_diag); geometry incidence is unmeasured (vulcan-forward's
  `ConvDiag` carries no geometry field). The outside reviewer, asked again at
  `92c3b1e` with the records in hand, also keeps both: the geometry term fixes a
  defect upstream shares (geometry refreshes only on the cadence, `op.py:904`, and
  `stop()` accepts a stale one), introduced in `001082e`; the budget term, from
  `46313d1`, polices the corrections the stage repair deliberately drops
  (`c6fdf4e`), and a certifying W39b photo-off column still drifts 2.16e-3, so it
  can bind. Both are extra stopping rules that can open likelihood holes, sized by
  their incidence at the first chemistry-qualified candidate on a saved W39b batch.
- Retry cap 110: keep (inert, above).
- rtol 0.2, `conver_ignore` [], `count_max` 1e4 equal upstream's HD189 example (no
  `use_adapt_rtol` line there, above), not its root `vulcan_cfg.py` (0.25, adaptive,
  `['HC3N']`, 3e4). `rtol_min`/`rtol_max` (0 / 1; upstream root 0.02 / 2.5) are
  dormant while `use_adapt_rtol` is false.
- Photolysis cadence: W39b 5/5; batched runs key it on the shared attempted tick and
  give a refilled lane one initial update. Effect unmeasured.
- Abundance file: keep. He is 13.5% below upstream's Lodders 2009 file, but upstream's
  root `vulcan_cfg.py:34` declares `He_H` 0.0838; this file makes the declared and the
  seeded He agree.
- Rate fixes: keep. Reaction 949's k0 1e-29 matches eight other networks in the
  80f75b9 clone; the Troe width in log10 is Visscher & Moses 2011 eq. 14.
- ExoGibbs seed (zero tangent by design), the `K_eq` exponent clip, gravity
  `G Mp / Rp^2`, the gather Jacobian: keep.

## Parity numbers

Default NCHO network, 69 species, 878 reactions.

| layer | max rel error | test |
|---|---|---|
| forward rates (439) / reverse rates (Gibbs) | 4.460e-16 / 5.705e-14 | `test_gibbs` |
| atmosphere (`pco`, `Tco`, `Kzz`, `M`) | bit-exact | `test_default_master_parity` |
| EQ seed | 7.9e-3 dex worst / 5.8e-5 median (HD189), 7.7e-3 / 9.1e-5 (W39b) | `test_eq_seed`, bar 2.4e-2 dex |
| chemistry RHS | ~2e-13 worst cell, ~1e-16 bulk | `test_chem_rhs_codegen` |
| analytical Jacobian vs `jacrev` | 2.8e-15 | `test_chem_jac_sparse` |
| diffusion operator / upwind `vm` operator | ~1e-5 (cancellation) / ~2.7e-6 | `test_diffusion` |
| block-Thomas: small case vs `np.linalg.solve` / VULCAN shape vs LAPACK banded (`_oracles.banded_solve`) / one Ros2 step | 1.8e-14 / 1.8e-15 [M] / 1.6e-9 | `test_block_thomas`, `test_ros2_step` |
| photo tau / flux / J | 7e-16 / 1.2e-11 / 6.8e-12 | `test_photo` |

- Matched steps: 20 steps 3e-9 over every cell, 200 steps 9e-8. Converged median
  fractional VMR difference: W39b 1.1e-9, HD189 3.3e-6, HD209 1.4e-4 (upstream
  against itself with one reaction changed: 7e-6). vm_branch vs `HD189_vulcan3`:
  100 steps 3e-9, 200 steps 2.4e-5 (vm_branch itself does not converge there).
- Static-vs-dynamic audit: the in-loop refresh (mu, g, Hp/dz/zco/dzi/Hpi, escape
  flux) matches to 4.4e-16 per refresh; statics (Kzz, Dzz, alpha, ms, vz, vs, M,
  Tco, pico, sat tables, cross sections) have no in-loop writer on either side.
  Not covered: non-empty `T_cross_sp` or `ion_sp` runs.
- Bars: the EQ seed bar is 3x the worst (both maxima HO2, whose FastChem row is older;
  otherwise one NASA-9 dataset; floor ~4.3e-5 dex FastChem convergence); W39b bulk species
  get a 1e-5 RHS bar because a ~1e-4 floor once let the converged state drift
  0.25-0.49 dex; the HD209 projection cases assert the residual within 1e-12 of
  the summed magnitude of the terms it cancels (the raw residual is a compiler
  property: 3e9-1.2e10 with FMA on arm64, 197.4 on x86-64). `fastchem_newton_tol`
  1e-12 (1e-14 does not converge); `fastchem_newton_max_iter` 450 is 2.1x the
  worst layer.
- Effective EQ He/H 0.08379150674 on all five presets.
- Rate-range extrapolation (P16): a one-sided `min(T, hi)` clamp moves W39b SO2
  +1.5% and HD189 NH3 up to +24.3%; the both-sided clamp's "-77% SO2" is an
  artifact. The manuscript quotes the one-sided W39b number, which has no recorded
  definition. Re-measured [M] (`rates_jax._arrhenius` patched, k0 and k_inf, 345 rates;
  W39b CLI, CPU): SO2 +0.72% at its peak (3.1e-5 bar), +0.22% in the column, 0.077 dex
  in one layer (1.5e-5 bar); CH4 up to 2.9 dex above 1e-6 bar.

## Parity ledger

"master" = the pinned VULCAN 2 clone; upstream lines are 80f75b9; "fixed
upstream" is against the VULCAN 3 line at a68d7151. "declared" = machine-checked
(`tests/science_sources.yaml intentional_divergences`,
`tools/audit_master_parity.py`, `tests/oracle.py::ORACLE_CODE_DELTAS`); "silent" =
notes only. None affects the default gas-only HD189 result unless stated. C8-C10
have no surviving record.

A. Upstream errors corrected in the port

| id | defect | upstream | port | size | fixed upstream |
|---|---|---|---|---|---|
| C1 | CH2CN + H + M k0 typo `1.00E-20` (DMS file swaps k0/k_inf) | `thermo/SNCHO_photo_network.txt:520` | `1.00E-29` (= e841d77, reaction 949); `KNOWN_THERMO_DIVERGENCES`, declared | unmeasured (trace nitrile) | bb9255b1 |
| C2 | S2 / S8 condensate molar masses 45.019 / 360.152 | `op.py:1203`, `:1249` | 64.12 / 256.48, `conden.GAS_MASS_G_PER_MOL`; silent | rates 0.702x / 1.404x; 0 shipped | no |
| C3 | H2O saturation = 0 at exactly 273 K | `build_atm.py:809` | `atm_setup.sat_p_jax` `where(T_C < 0, ice, liquid)`; silent | one grid point | no |
| C4 | eps Eri spectrum low by R*^4 (3.4265x) | `atm/make_spectra_in_nm.py:25` | rebuilt file, `KNOWN_SFLUX_RESCALES`; declared | 3.43x on the eps Eri drive (JWST tool's WASP-107 b) | no |
| C5 | duplicated `CH2_1` NASA-9 rows | `fastchem_vulcan/input/nasa9_logK_SNCHOTi.dat:19,421` | retired with the FastChem tree | 0 (rows identical) | 8970337 |
| C6 | H2S saturation constant 0.001333, 10x low | `build_atm.py:857` | `atm_setup._CMHG_TO_BAR` 0.01333; declared | 10x on H2S p_sat; 0 shipped | no |
| C7 | `NH3_l_s` mass is NH2's 16.023 | `thermo/all_compose.txt:167` | 17.031; declared | mean mass, moldiff only | no |
| C12 | FastChem P/S row order vs hard-coded C++ slots (S molecules used P's polynomial) | `mass_action_constant.cpp:388-389` + `.dat` rows | rows reordered, `test_abundance_presets_keep_fastchem_slot_order`; declared | atomic S 2.6-3.1 dex, molecules <= 0.004 dex (initial state only) | no (8970337 inert) |
| C13 | Earth example: `Ar` in `atom_list`, no Ar species | `cfg_examples/vulcan_cfg_Earth.py:6,39` | `runtime_validation` refuses carrier-less elements and non-network `const_mix` keys; silent | 0 shipped | 436cf3c6 |
| C14 | network id column rewritten by the code, then read back | `make_chem_funs.py:109`, `op.py:243` | positional indexing, `Network.stale_ids`, `legacy_io._warn_stale_reaction_ids`; silent | 5 of 11 vendored networks stale, 3 silently wrong | no |
| C15b | `make_bins_read_cross` drops a bin when `dbin_12trans` is out of range | `op.py:583-585` | `photo_setup._make_bins` raises; silent | unreachable shipped | no |
| C19 | Ros2 stage vectors leak element content at large dt (only guard `loss_eps`) | `op.py:2914,2929`; `:2490`; `vulcan_cfg.py:136` | `jax_step._repair_stage` + `config.DT_MAX_S` (a larger `dt_max` raises); cap declared, repair silent | upstream fails the same column (S +32%); gates unchanged | no |
| C20 | `OH + CH3 + M` Troe width in ln, not log10 | `op.py:204-205` | `rates_jax._troe_OH_CH3`; both sides via `ORACLE_CODE_DELTAS`; `test_troe_oh_ch3_is_visscher_moses_eq14` | rate 1.19-1.21x at Pr 0.1; max 0.111 ln VMR (HD189 CH4); gates unchanged | no |
| L1 | `diffdf_settling_vm` drops the `vm` drift at the bottom interface | `op.py:1845-1848` | consistent; `test_diffusion`; silent | one cell, `use_settling` only | no |
| L2 | vm_branch setup `vm` rolls a 2-D array without `axis=0` | `vulcan3_vm_branch/build_atm.py:803` | `axis=0`; silent | 5e-10 at step 2 | no |
| P18 | `sflux-GJ1214.txt` pads 133.75-181.55 nm with 464 zero rows | `atm/stellar_flux/sflux-GJ1214.txt` | upstream's regenerated file, `gj1214_upstream_regenerated`; declared | 0 shipped; regeneration lowers every flux 1.392x | bb9255b1 |

C12 was drafted as a report to upstream; whether it was sent is not recorded. C20
is not reported upstream (maintainer).

Port defects fixed back to parity, and JAX-only additions

| id | kind | upstream | port | size |
|---|---|---|---|---|
| C11 | JAX-only | give-up is a dt floor, `op.py:2515-2516` | `batch_max_retries` 110 + force-accept (`outer_loop.py:1077-1080`) | inert: dt < dt_min ends the reject chain before the cap ("Differences from VULCAN master"); redo if `dt_max`, `dt_min` or `dt_var_min` change |
| C15 | port defect | escape Jacobian term only under `use_vm_mol` (`op.py:2102-2107`, `:2426-2431`) | `diag_d.at[-1].add(diff_lim * use_vm_mol)` in `jax_ros2_step` | HD209 1206 -> 1211 |
| C16 | port defect (= P6) | `step_ok` binds `rtol` at import (`op.py:2489,2495`); writes reach only `step_size` (`:3111`) | `_Statics.rtol_accept` | 1058 of 2893 decisions on HD189_vulcan3 |
| C17 | port defect | bottom pins before delta and clip (`op.py:2945-2953`; `use_fix_H2He` `:2935-2941`) | pins before the rebalance; `test_use_fix_H2He` | 7.7e-3 on HD189 with `{'H2O': 0.5 EQ}` |
| C18 | port defect | photo-frequency switch waits for `trun_min` / `count_min` (`op.py:1069`, `:819`) | `switch_to_final` gate | HD189 switches after step 121 on both |
| - | port defects | all-NaN state raises (`op.py:1055`) | `longdy=+inf` (`test_nonfinite_never_converges.py`); clip on post-solve `ymix`; escape term on the Jacobian diagonal | NaN once scored converged |
| C21 | JAX-only | refresh only at `count % update_frq == 0` (`op.py:904`); `stop()` on chemistry (`op.py:1056-1085`) | "Certificate candidate" block + `_real_terminate` | see "Certificates" |
| C22 | JAX-only | none | `OuterLoop.run_jvp` (`_make_runner_jvp`); vulcan-forward `converged_y_jvp` | see "Certificates" |
| C23 | JAX-only | unweighted per-step loss only (`op.py:2485-2490`) | `budget_ok_next` + `_real_terminate`; `test_element_budget_certificate.py`; vulcan-forward seeds `budget_ref`/`budget_err` per theta | see "Stage repair and element budget" |
| C24 | port defect | `fix_y` snapshot before the relaxation (`op.py:871-873` before `:898-902`) | `_activate_fix_species` before the `conden_branch` cond; `test_conden_fix_species_pin_matches_master` | H2O_l_s column max cell relerr 1.0 -> 5.2e-15; 0 shipped |

B. Found in upstream, kept at parity

| id | defect | upstream | port | size | fixed upstream |
|---|---|---|---|---|---|
| P1 | unweighted atom-conservation diagnostic feeds `step_ok` and the adaptive rtol | `op.py:2472-2487` | `body_fn` `atom_loss_new` | unmeasured; a fix re-baselines | no |
| P2 | two-stream pole `ll = -w0/(1/mu^2 - (1-w0)/edd^2)`; the clip is dead code | `op.py:2652`, clip `:2666` | `photo.compute_flux_jax` | 0.1-0.5% of cells, diffuse flux only; a fix is the analytic limit | no |
| P3 | condensate mass in the gas mean molecular weight | `build_atm.py:520-525` | `atm_setup.compute_mean_mass` callers | condensation only; fix = restrict to `gas_indx` | no |
| P4 | `atm_type='table'` stale `pico` for `dz` | `vulcan.py:118` -> `build_atm.py:406,530,554` | same | `dzi` up to 12%; not live | no |
| P5 | `const_lowT` sum omits NH3, He as remainder | `build_atm.py:48`, `:213` | `ini_abun._load_const_lowT_y` | He/H 0.83% low at 10x N; not live | no |
| P6 | = C16 | | | | no |
| P7 | photolysis quadrature skips the bin-grid join interval | `op.py:2771-2780` | `photo.compute_J_jax_flat` | HO2 J 0.06% (W39b) / 0.32% (HD189), others <= 0.05% | no |
| P8 | `HNCO + M -> H + NCO` carries M on the reactant side only | `thermo/SNCHO_photo_network.txt:557` | parser tracks each side; `rates_jax` delta_n | 1.3e-5 / 5.0e-5 / 6.0e-4 ln VMR on the gates; HD189_vulcan3 2803 -> 2668 | partly, a68d7151 (in the vendored 2025 file; the 2021 SNCHO file still one-sided) |
| P9 | moldiff boundary rows do not conserve the column | `op.py:1549-1558` | `bdry0_grav` / `bdry_top_grav` in `jax_step` | +6.6e-19 to +7.9e-18 /s; C23 bounds it | no |
| P10 | N(2D) NASA-9 thermo is a molecule's | `thermo/NASA9/N_2D.txt` | same file | K_eq 1.6e3-1.5e4x low; HD209 1211 -> 1206 if fixed; hot-top N 1000x | no |
| P10 (second use of the ID) | escape Jacobian entry `top_flux/y[-1]` too large by `dzi[-1]` | `op.py:2106-2107` vs RHS `:1689` | comment at the `diff_lim` term in `jax_ros2_step` | LHS only; HD209's `diff_esc` inert | no |
| P11 | C2H4O, CH3NO2, HCS listed twice with different masses | `all_compose.txt:76/158, 118/161, 136/212` | `composition._PARITY_DUPLICATES` (first row kept) | moldiff 7e-5 | no |
| P12 | CS-family and H2SO4 masses inconsistent with the S row | `all_compose.txt:133-136,143` | same | ~0.2% on trace masses | no |
| P13 | `CH3SH_branch.csv` grid non-monotonic at the tail | `thermo/photo_cross/CH3SH/CH3SH_branch.csv:142-143` | same | unmeasured | no |
| P14 | TiSNCHO ships three reactions twice | `thermo/TiSNCHO_photo_network.txt:537-542` | `parse_network(duplicates_ok=True)` | 0 (unselected) | yes |
| P15 | SNCHO_C3 ships `C2H2 + C -> C3 + H2` twice, different rates | `thermo/SNCHO_photo_network_C3.txt:474,479` | same | 0 (unselectable) | file renamed away |
| P16 | rate fits evaluated outside their stated T range | `op.py:166,170` | same (`rates_jax`; no advisory) | see "Parity numbers" | no |
| P17 | `diffdf` and `lhs_jac_tot` build `ysum` from different predicates | `op.py:1505` vs `:1981` | matches the derivative of `diffdf`; `tests/test_diffusion.py` tolerance | LHS only, a few S8 diagonal cells | no |
| P19 | arithmetic layer-mean actinic flux | `op.py:2709` | `photo.compute_flux_jax` `ave_dir_flux` | see "Photolysis decisions" | no |

Master-only defects, already better here (do not "restore parity")
- Optical depth vs single-scattering albedo: master's `compute_tau` sums `photo_sp
  U ion_sp` with T-dependent cross sections while `compute_flux`'s `w0` uses the
  T-independent table (`op.py` ~2621-2672); the port builds both from one
  `PhotoData`. 0 shipped; with `T_cross_sp = [SO2, H2S, H2O, CO2]` column J moves
  up to 24.6%: quote no parity result for such a run.
- Not inherited: FastChem's shared I/O filenames (`build_atm.py:128,131,153,169`);
  `np.nanmax` on an empty selection for a dark column (`op.py:2737`); `S4` absent
  from `sat_sp_list` (`build_atm.py:787` vs `:832`); kinetic NH3 condensation
  looking for `NH3_l` (`op.py:1151`, a silent no-op); `check_conserv` byte-string
  names under NumPy < 2 (`make_chem_funs.py:719-725`); example configs that cannot
  start (`op.py:836,840,848`); the `fix_species` cold-trap tie (`op.py:888`
  raises); `use_relax` truthiness (`op.py:1121`, `:1153`; the port gates per
  species in `conden`).

## Known bugs, all repos

Do not trim this list in a cleanup; an entry changes only when its bug is fixed
or re-measured. Detail is in "Parity ledger" or the named repo's notes.

- vulcan3, inherited from upstream and kept: P1 unweighted atom-conservation
  diagnostic; P2 two-stream pole; P3 condensate mass in the gas mean molecular
  weight; P5 `const_lowT` initializer; P7
  photolysis bin-grid join interval; P8 HNCO + M one-sided M; P9 diffusion
  boundary rows leak the column; P10 N(2D) thermo is a molecule's polynomial
  (maintainer: not fixing, upstream data); P19 arithmetic layer-mean actinic flux.
- vulcan3, inherited from upstream, P4 (`atm_type='table'`): the table replaces the
  pressure centres after the interfaces were built, and the interfaces are not
  rebuilt (upstream the same order: `vulcan.py:118`, then `build_atm.py:406`); `dzi`
  up to 12%. Every shipped config, W39b included, reads `atm_type: file`. A fix is
  owed before a table route is used for science.
- vulcan3, both codes, no shipped effect: the He thermal-diffusion factor 0.145 is
  overwritten by the >4 amu loop's 0.25 (`atm_setup`); HNO3.txt's NASA-9 segments
  are ordered 50-200/200-1000/1000-6000 K, so the two-segment parser uses the
  50-200 K fit below 1000 K (HNO3 is in no shipped network).
- vulcan3, measured unreachable: `K_eq_array` clips only the exp() argument
  (`rates_jax._EXP_ARG_MAX` 709), not log K; the largest delta_n < 0 log K over
  100-6000 K is 433.6; NCHO and SNCHO overflow only below 58 K. Fix only if a
  config below ~60 K ships (clip the whole log K, re-run the gates).
- vulcan-forward: `ckd.overlap` random-overlap jumps at sort crossings
  (accepted); the photolysis-cadence switch ignores a profile's `yconv_min`
  override (timing only).
- vulcan-retrieval (unverified): the stage-2 elemental projection fixes carbon by
  scaling CO only, so cool CH4-dominated columns start with negative CO and get
  rejected (a theta-predictable cut of the prior's cold corner).
- jwst-transit-authority: the mixed-segment bin.

## Operational

- The JAX persistent compile cache key ignores `XLA_PYTHON_CLIENT_MEM_FRACTION`.
  Job 93577 loaded entries compiled at fraction 0.04 and produced wrong pass-1
  results (6114 ticks / 1679 unconverged vs 7658 / 237). Any process at a
  fraction other than 0.90 needs a private cache dir.
- The cache was write-only while `photo_sp` was a set (the hash seed permuted
  `cross_J` rows); `sorted(photo_sp)` / `sorted(ion_sp)` fixed it (`2fdc66b`; HD189
  `count_max=3` 30.0 -> 1.6 s warm). Localize a miss with
  `JAX_DEBUG_LOG_MODULES=jax._src.cache_key` and `PYTHONHASHSEED=0`. `dt_max` is
  not in the model cache key: compare two values in two processes.
- Node cgh10n04 has a broken GPU: fail fast when `jax.default_backend() != 'gpu'`.
- daw's RTX 6000 Ada (FP64 at 1/64) is a development device: bitwise
  reproducible, W39b 207-213 s against 101 s on one CPU core.
- The `use_print_prog` progress block (`legacy_io.Output.print_prog`) goes through
  `logging` at INFO. The CLI configures logging to stdout, so it still prints there;
  a Python-API caller sees it only with a handler at INFO.

## Evidence behind the numerical rules

Rules whose evidence is a section of its own point there from CLAUDE.md
("Solver conditioning", "Stage repair and element budget", "Certificates",
"Adjoint record", "Forward mode", "Parity ledger"). The rest:

- float64: the whole Ros2 step in float32 is non-finite at every dt; float32 block
  factors give forward error 1 to 4e2 already at dt 1e2.
- Analytical Jacobian vs jitted `jacrev`: agrees to 2.8e-15 (NCHO), 15.7x (NCHO
  ni 69) / 18.4x (SNCHO ni 89) faster with the old chunked scatter, more with the
  gather form. Jit both sides (un-jitted `jacrev` inflates NCHO to ~35x).
- `y**0`: NaN forward tangent at y=0; no separate measurement recorded.
- Where-zeroed numerator, denominator 1: a NaN batched tangent on TOI-7169 b
  (jwst-transit-authority 0.51.0) was `0 * inf` from sub-atol cells (S8 at
  6e-150 cm^-3) under `jnp.max`'s JVP, a rewrite XLA does only unbatched. Repro
  `max(stack([x[0], sqrt(x[1]) * 0]))` at [1, 0]: plain tangent 1.0, vmap NaN.
  Guard in `test_nonfinite_never_converges.py`.
- Fixture arrays as jit arguments: closed over, the HD189 tangent of
  `chem_jac_analytical(y) @ x` read 1e42 against 1e23 eager; as arguments it
  agrees to 1e-15.
- Clip on the post-solve `ymix`: fed the pre-step `ymix`, 267 / 204 / 204
  negatives survived at dt 1e4 / 1e8 / 1e12 and steps master accepts were
  rejected.
- SymPy-faithful codegen: XLA otherwise rewrites `H*H` as `exp(2*log(H))`; RHS vs
  NumPy oracle ~2e-13 worst cell.
- `conver_ignore`: before C20, HD189 `[]` and `['HC3N']` both took 1495 steps
  (atomic C controls); a 13-hydrocarbon list from a hand-patched master copy gave
  1296 by shrinking the metric. Every shipped run converges on the loose branch
  (`longdy < yconv_min` 0.1 with `longdydt < slope_min`, `op.py:1056`); realised
  longdy on daw 0.09098 / 0.03018 / 0.09919 (HD189 / HD209 / W39b, 1% under the
  threshold for W39b).
- `vm` in-loop: freezing it at setup put He 0.43 dex and CH4 1.7 dex off at
  `const_Kzz=1e7`, photo off (<= 0.16 dex after). The scheme matters only at low
  Kzz: HD189 (Kzz 1e10) central vs hybrid <= 0.008 dex, at `const_Kzz=1e7` up to
  2.7 dex in CO/H2O. Steps, central / upwind-then-central: HD189 vm_branch 1590 /
  2201, vulcan3 1495 / 2102; W39b both codes 1202 / 1301.
- Gravity from `Mp`/`Rp`: `default.yaml` once shipped a different `Mp` from
  `HD189.yaml` (gs 2139.7705 vs 2140, per-layer J up to 7%); a fix to the named
  configs only silently changes every API user's run.
- Condensation not differentiable: the pinned S8 tangent vs FD has relative error
  ~0.91.
- Abundance file before a cross-code number: HD189, both converged, floor 1e-12:
  upstream defaults 2.0e-1 median / 8.8e1 max; network AND composition matched
  3.8e-6 / 4.5e-2.
- Batched vs solo: at one step they differ only in `mu` (1 ulp; the einsum in
  `update_mu_dz_jax` sums in another order under vmap); the dt controller
  amplifies that to 1e-2 in y within 150 steps on a marginal column. The EQ seed,
  vmapped, summed in another order (4.1e-15), hence `custom_vmap` + `lax.map`.
- Wall-time ratios need `user+sys ~ real`: cpu/wall 0.29 and 0.085 came from a
  closed laptop lid and the 16.8x speedup derived from it was retracted.
  `tools/bench_table1.sh` enforces it, HD189 only (its VULCAN 2 has HD189's inputs).
- CI: unpinned, it resolved jax 0.11.0 with 3 failures of 258;
  `test_vmap_photo_batch`'s bit identity breaks from jax 0.10. The 16 GB runner
  was killed in two of six runs at four workers (~15.8 GB); conftest clears JAX
  caches per module and CI runs three workers (two if it is killed again). jax
  floor 0.5.3: older versions lack `optimization_barrier` batching or AD rules.
- `oracle.yml` runs one process per slow test: in one process the runner was
  shut down during the W39b tangent certificate (12.6 GB of a 16 GB runner) twice,
  before earlier failures reached the log. It rebuilds the fixtures at HEAD, so a
  daw check of a full-tier failure rebuilds them too (`gen_fixtures.py --all
  --force`): the tracked arm64 fixtures passed both tests that failed there.
- `python -m pytest`: the parity-audit tests fail under the console script (`No
  module named 'tools'`); `PYTHONSAFEPATH=1` fails `test_default_master_parity`.

## Solver conditioning

The factorized matrix is `I/(gamma*dt) - J`; on a converged HD189 state its blocks
reach cond ~1.4e18 at the run's dt (3.8e4 s) and ~6.7e23 at dt 1e11. Synthetic
diagonally dominant blocks (cond ~4e2) pass anything. Block-Thomas pivots only
within a block, so on imposed large dt its relative residual blows up where
banded LU does not (dt 1e11: 2.8e3 W39b / 8.5 HD189 vs 9.5e-13 / 7.4e-15). No run
reaches those dt (every solve instrumented):

| run | steps | max dt | max residual | solves > 1e-6 |
|---|---|---|---|---|
| HD189 / W39b CLI gates | 1495 / 1202 | 3.8e4 / 8.7e5 | 1.7e-11 / 8.4e-12 | 0 |
| 6 production prior draws (nz 62, cold) | 19161 | 5.7e6 | 9.8e-11 | 0 of 38322 |
| 9 production warm proposals | 41251 | 9.5e5 | 2.7e-10 | 0 of 82502 |

- Block-local pivoting can fail where banded LU does not: the outside review's
  generic 2x2 counterexample ([[0,1],[1,1]]) gives NaN. No physical W39b matrix has
  been shown to fail; real large-dt stage matrices against banded LU are not tested.
- Reverse mode through the block solve is componentwise backward stable
  (Oettli-Prager 3.8e-14..2.4e-12 at dt 1e11) although a normwise residual against
  a random rhs reaches 1e23. The cotangent is the transpose of the primal sweep on
  the same factors; a fresh factorisation of A^T for `ffi` failed the W39b
  sensitivity regression (top pair 876/875 instead of 1/2), so reverse mode
  through `ffi` is unsupported (documented in `solver_fast`, not refused).
- `test_solver_fast.py` bars: `cand <= max(2 x cur, 10 x gbsv)` against LAPACK
  `gbsv`; GPU batched cuSOLVER LU differs from unbatched by up to 1.3e-3, so the
  vmapped `fast` solve is judged by residual; `ffi` tangent bar 5e-2 (`fast` 1e-5;
  a wrong tangent reads ~0.66). A direction scaling every operand by one factor
  has an exact tangent of zero: test along a dt-shift-plus-noise direction.
- These numbers are 0.6.0-anchored; `tests/data/adj_state_*.npz` are conditioning
  inputs, not chemistry validation.

## The dt controller has no headroom

- `_step_size` is an I-controller with `h_factor = 1` at `delta = safety^2 *
  rtol`; measured median `delta/rtol` 0.81 on HD189 and HD209; the largest
  acceptable step is 1.23x the one taken. Upstream's is identical
  (`op.py:3105-3124`): any change breaks bit-parity with both oracles.
- One trace cell sets the step (C3H2 at layer 133, just above `atol`); masking to
  cells above 1e-6 VMR still gives delta 0.27-0.30: the ceiling is physics.
- dt -> inf IS Newton (k1 identical to 10 digits at dt 1e15 / 1e30). At the
  converged state `max|f(y*)| = 1.4e21` against production/loss ~1e28: a fixed
  point of the step map, not a zero of f. `_clip_prologue` zeroes negatives
  unconditionally, so `all_nonneg` is vacuous (inherited, `op.py:2459`); C23
  catches what it misses.
- The convergence window is not a step window: `_lookback_index` takes the ring
  slot closest in time to `t * st_factor` (0.5); `conv_step` 500 is only a floor.
  An uncapped dt band certifies cleanly and is 43% wrong on H2O.
- The one clean step-count lever is the photolysis cadence: 100/5 -> 1/1 takes
  HD189 1495 -> 940 steps, within 9.6e-5 of a long reference. Not a measured wall
  win; the parity configs keep 100/5 and 5/5.

## Stage repair and element budget

- The leak: on the JWST tool's WASP-39 b photo-off column (nz 100) at
  `dt_max=1e11` the operator-weighted O/H ratio runs 0.769 -> 0.543 and S/H 0.872
  -> 0.022 by the certified exit (5040 steps), no step rejected. With transport
  off the worst-layer |dS/S| per step is 5.6e-6 at dt 1e8, 5.8e-4 at 1e11, 0.44 at
  1e15: the conserved directions of `I/(gamma*dt) - J` are its near-null ones.
  Photo-on columns step at 1e5-1e7 s and conserve to 0.01%.
- The repair restores `c0 a^T k - a^T T k = a^T b_tr` per layer after each stage
  solve by adding the defect to a FIXED reservoir (a switching one would couple
  two rows); layers with a pinned cell are skipped. Per-step O change at dt 1e11 /
  1e13 / 1e15: +6.1e-5 / -1.3e-3 / -1.7e-3 unrepaired, +5.5e-7 / +5.5e-5 / +5.5e-3
  repaired; cost 26.4 -> 26.9 ms per step. At dt 1e17 the correction is as
  singular as the system, hence `DT_MAX_S` 1e15. The float64 step itself is
  unusable well below that (HD189 median species error 5.2e-4 at 1e11, 53 at
  1e15); consumers cap their own `dt_max` (largest shipped step 1.3e6 s).
- `_tridiagonal_solve` is LAPACK dgtsv's partial pivoting as two rolled scans:
  `c0 - T_rho` is not diagonally dominant (-0.37 by column on HD189) and 0.7% of
  its sub-diagonal entries have the wrong sign, so an unpivoted sweep can hit a
  zero pivot (3x3 counterexample: rows (2,-2,0), (-1,1,-3), (0,2,4)).
- Floors and guard, re-baseline knobs: `_DEFECT_FLOOR` 1e-13 of the identity's
  own terms (without it the repair moved roundoff onto reservoirs and broke
  20-step parity); `config.REPAIR_ABS_FLOOR` 1e-11 of the layer density, absolute
  because a relative floor never fires near steady state (it stalled the JWST
  tool's HD 189733 b Guillot preset at 30000 steps against 1506 unrepaired);
  `_REPAIR_MAX_CELL_FRAC` 1.0 drops a correction larger than its carrier cell
  (W39b negative H2S band 12 -> 0 at dt 1e11; inert on the gates). Real leaks sit
  at >= 1e-7 of the layer, roundoff at <= 1e-10. Stage identity after the repair:
  1e-13 at dt <= 1e11, up to 1e-11 at the cap, so its bar is 1e-10.
- Not fixed by the repair: the moldiff boundary rows (P9, +6.6e-19 to +7.9e-18 /s
  of column gain); they set C23's tolerance floor.
- C23 accumulates each accepted step's operator-weighted column change
  (`ini_abun.column_atoms`; `operator_column_weights`: `w_0 = dzi[0]`, `w_j =
  0.5*(dzi[j-1]+dzi[j])`, `w_last = dzi[-1]`, so a zero-flux update telescopes)
  on that step's own grid over the fixed t=0 column, RELATIVE TO H. The absolute
  column carries the renormalisation's common mode (-1.04e-2 on every element on
  the planner's WASP-107 b column; as X/H 2.1e-6). Tolerance 1e-2: 370x headroom on
  the worst photo-on gate, 4.6x on a photo-off column. Five knobs open the budget,
  none on in a shipped config (`use_topflux`, `use_botflux`/`v_dep`,
  `use_fix_sp_bot`, `use_fix_H2He`, the `fix_species` pin);
  a `loss_ex` containing H fails the runner build (H is the reference atom).
  An open column can fail to certify on exchange it is meant to have (measured on
  Earth, "Withdrawn configs"; [I] for the external user's K2-18b; the JWST S8 pin
  certified); its step-cap exit names the term and `.inf`. Precedent: KPP's
  invariant-preserving Rosenbrock, CVODE projection, Sandu 2001 (doi:10.1006/jcph.2001.6750).
- The "strong-Kzz top boundary collapse" of W39b domain extensions was this leak
  in a 3000-3500 K bottom: since 0.6.0 the 50 bar -> 5e-9 bar column converges in
  885 steps, every element within 5.4e-5; pinned upstream hits its 3000-step cap
  with S +32%, and its Kzz-capped twin "converges" at 5965 steps with S -21.5%
  (C23 rejects it 21x over). No shipped config has a >= 3000 K bottom cell.
- HD189_vulcan3 is a marginal count: its top moves with any bit-level change
  (2607 / 2702 / 2803 / 2603 / 3242 across such changes). With
  `photo_switch_longdy_thresh` pinned at 1.0 (it is derived as `yconv_min * 10`,
  so pin it in any photo-on convergence study) a tolerance ladder runs 2776 ->
  8715 steps and the shipped column sits 5.3e-3 (top-decile median |d ln VMR|)
  from the 8715-step reference. On x86-64 a 20000-step phase-1 budget certifies at
  7152 on the arm64 column: the same fixed point on a longer path; `reference`
  gives the same 4018. The end message then names `count_max`, not the
  `count_max_dyn` it hit (cosmetic).

## Certificates

C21 (geometry) and C22 (tangent); C23 is in "Stage repair and element budget".

- C21: HD209 photo-off certifies at 192 steps with mu 26.30% off what its ymix
  implies; tightening the chemistry alone leaves 6.6%. With the term: tol 1e-4 303
  steps / 5.0e-5 from a cadence-1 reference, 1e-3 297 / 3.1e-3, 1e-2 252 / 2.0e-2.
  On HD189_vulcan3 1e-4 never certifies. The norm is mu/g/Hp/dzi/Hpi only. Never
  below `count_min`, so matched-step parity is bit-identical.
- C22: `run_jvp` applies the primal's reduction to the tangent
  (`_longdy_reduce(..., diff=dy - dy_lookback)`), per cell and absolute, so the
  direction is read in FD-step units. HD209 photo-off d/d ln Kzz certifies at 297
  steps, 6.1e-4 from central FD; W39b photo-on at 1536 accepted steps (primal
  1202), 6.7e-3 from FD above 1e-4 VMR (corr 0.99982). `run_jvp`'s tangent is
  bit-identical to `jax.jvp(runner)`'s at equal step counts.
- The AD-build `dt_max` 1e5 s cap lives in jwst-transit-authority's
  `AD_BUILD_OVERRIDES`: on the planner's WASP-39 b eclipse column caps 1e4..3e6
  certify at corr 0.9976 against FD and agree to >= 0.999995; uncapped it does not
  certify. A cap on the cold solve (W39b 1363 -> 2487 steps, spectrum up to 6.23
  ppm) was decided against.

## Adjoint record

Reverse mode.

- Every route that linearizes `df/dy` or iterates the bare body map fails on
  conditioning (`||f(y*)|| = 1.56e21` against production/loss ~1e28; >= 8
  eigenvalues outside the unit circle). The shipped route needs log-abundance
  coordinates, QR deflation of the atom-count left-null vectors and LGMRES on the
  renormalised Ros2 solver map (`steady_state_grad.SOLVER_MAP = "renorm"`, the
  only map: HD189 CH4 6.6% -> 0.7%, HD209 r121 35% -> 1.1%).
- `body_dt` (HD189 CH4 r13 vs FD -0.565): 1e6 diverges; 3e6 has a 27.6%
  slow-mode bias; 1e7 (default) 0.3-6.0%, mean 3.5%; 1e8 0.5-20.6%; >= 3e8
  garbage.
- `photo_recompute_k`: W39b `OH + H2 -> H2O + H` frozen-J error 11.2-13.0% ->
  0.2%. `"auto"` and `None` mean no feedback and raise on active photolysis rows
  (`None` is accepted photo-off for the JWST tool's `adjoint_diag.py`).
- HD189 calibration (n_solves 3, inner_m 250, cycles 8): mean r13 5.0% from FD,
  top-6 exact, `null_quality` 2.2e-5, ~27 min warm. On the x86-64 fixture rebuilt
  at `2fef3c1` `null_quality` reads 7.2e-2 (r13 1.23% from FD), hence the slow
  regression's bar 0.2. `pair_antisym` ~1 does not mean wrong. W39b needs the
  exact-zero mask `where(y_star > 0, 1/y_star, 0)`.
- The JWST tool's tabulated W39b state is refused by `audit_adjoint_scope`
  (atomic N at layer 93 in the clamped top); the forward path is the route.

## Forward mode

- Full-integration HD189 (photo off, nz 150), uniform Kzz-scale JVP against
  re-converged central FD, top 1% response cells: median relative error 1.27e-4,
  max 1.71e-4 with the adaptive timestep differentiated; with its gradient
  stopped, median 0.388 with identical primals. `test_forward_jvp_integration`
  runs that map (4:16, 4.8 GB on one daw core); single-step and reverse-mode
  checks do not cover this tangent.
- End-to-end `jvp` on HD189 photo-on vs re-converged FD: better than 0.1% on
  responding levels, correlation > 0.9999.
- Photo-off W39b (retrieval `config.WIDE`) was a primal problem: a marginally
  stable fixed point from uncapped `dt_max=1e17`, the stage leak and hybrid
  moldiff. Converging recipe (121 steps): `use_vm_mol=False`, `dt_max=1e11`,
  `conver_ignore` += S/S2/S3/S4/S8/S2O/HS2/CS2, `mtol_conv=1e-15`.

## Withdrawn configs

Earth and K2-18b; an external user reports both converge upstream (their inputs
were not obtained).
- Earth on main [M] (`c61fd34`, CPU): vulcan3 hits the 20000-step cap at longdy
  0.040 (below `yconv_min` 0.1); with `element_budget_tol: 1e9` it certifies at
  11411 (longdy 0.0999): the budget term is the only blocker. Upstream (clone copy,
  C13 fix, four missing keys to start) certifies at 13331; the two columns agree to
  median 3.3e-4 dex, bulk species <= 2e-3 dex (worst CH3NH2 0.056). Inputs: the
  clone's network, atm file and four NASA-9 files. `atom_loss[H] = 2.98e3` is not
  a leak (`use_fix_sp_bot` pins H2O).
- K2-18b (nz 130, ni 93, condensation on): hybrid 31002 steps at longdy 25.8,
  central 30001 at 0.459, upstream 1.00 at 3000; the clock froze at 6.18e4 s
  from step 3001. Its `remove_list` resolved by position to `NH3 + CH` and `SH +
  NO2`, and it omitted `batch_max_retries`.
- K2-18b re-add (needs the external user's DMS-2025 network and atm file): convert
  their YAML, check `remove_list` by position (C14), hash-pin inputs, run to end_case 1.

## Design records

### Charter, configs, API

- Standalone port of exoclime VULCAN (from vm_branch @ 362cfa2): vendored data (only
  what a config, test or consumer reads, `25eca7d`), no sibling at runtime; one
  dependency beyond jax / numpy / scipy / PyYAML, `exogibbs>=0.6.0,<0.7`,
  installed `--no-deps` plus `pandas` (its declared pyfastchem has no aarch64
  wheel).
- The named configs stay inside the package (a pip install needs them there);
  `vulcan3 --config <path>.yaml` runs a config file from anywhere (the maintainer
  decided).
- `OuterLoop` reads its config at construction and at the first runner build; a
  changed config means a new `OuterLoop`. Cfg-derived toggles are read from
  `self._cfg` at their use sites, so the runner and
  `steady_state_grad.make_body_terms` cannot disagree. Nothing writes run results
  back into cfg. `integ(rs)` refuses a RunState with `params.count > 0`: a
  RunState does not carry the certificate ring, the element budget or the hybrid
  phase.
- `network` / `com_file` / `atom_list` are read at the first `import vulcan3`
  (`config._FROZEN_ENV`); on a cfg that differs (`state._frozen_mismatch`; a
  byte-identical file copy matches) the CLI re-execs once with matching
  `$VULCAN3_*` (guard `$VULCAN3_CLI_RELAUNCHED`), `with_pre_loop_setup` raises.
  A pre-import launcher cannot live in the package (any submodule import runs
  `__init__` -> `state` -> `chem_funs`). `VULCAN3_SOLVER`, `VULCAN3_CHEM_RHS` and
  `VULCAN3_BT_BUFFERS` are read once per process; `VULCAN_JAX_*` fails the import.
- A config file overlays `default.yaml`; a named config sets only the knobs that
  differ (`HD189.yaml` is `{}`). `<X>_H` is accepted for any symbol of the
  composition table. Validation resolves assets with `resolve_data_path` (cwd
  first, then package data), as the run does.
- `termination_reason`: 0 running, 1 converged, 2 runtime, 3 step count, 5
  non-finite; `end_case` follows master's 1/2/3, and 0/5 or a non-finite
  "converged" state read 5.
- `integ(rs, chunk=N)` drives `runner_chunk` (the runner's while_loop with `&
  (accept_count < target)`, target a jit argument, no carry leaf); chunked equals
  the one-loop run bit for bit (`tests/test_save_evolution.py`). The CLI always
  runs chunked at `print_prog_num`; Ctrl+C opens a shell on
  `integ.interrupted_runstate`, then exits 130. One loop cannot be interrupted:
  SIGINT reaches Python only when XLA returns.
- `vulcan3.conv_normal(final, cfg)` is the runner's certificate on a carry, from
  `_real_terminate`'s helpers; vulcan-forward's `ConvDiag` calls it.
- `release.sh` refuses a dirty tree or a branch other than `main`, runs the suite
  before the version bump and stages only `_version.py`.

### Code contracts not obvious from the modules

- `jax_step.py` is a W-method, not an L-stable Rosenbrock: the Jacobian models
  transport as a diagonal-in-species tridiagonal without the `ysum` coupling;
  contractivity comes from the post-step rebalance.
- Batched: `prepare_runstate` checks `nbin`/`din12_indx`/`bins`/`sflux_top`; a
  mixed `CondenStatic` batch uses the first profile's cfg fields; the NH3
  cold-trap `argmin` is frozen at setup. `CondenStatic` as a jit argument breaks:
  close over it and splice arrays with `_replace`. `run_queue` takes its array
  fields from `_ATM_STATIC_BATCH_AXES`.
- Condensation: a new quantity touches `CondenProfile`, the `ProfileVars c_*`
  threading and `_make_conden_branch`. `compute_sat_p` knows H2S but
  `conden.GAS_TO_CONDENSATE` does not, so `condense_sp=['H2S']` fails late.
- `high_temp_cut` is a host-side re-grid (raises `P_b`), not a T-clip.
- Setup never writes config knobs (`state._cfg_overlay` drops the write): the
  `use_sat_surfaceH2O` bottom pin is built in `OuterLoop._build_statics` from `sat_mix`.
- Codegen cache `__pycache__/chem_rhs_codegen_<hash>.py`: key = network content +
  `inspect.getsource` of both emitters; temp file + `os.replace`; never pruned.
- `parse_network` raises on within-section duplicates (`duplicates_ok=True`
  warns) and on any non-comment row that fails to parse; `composition.py` raises
  on a species with no `all_compose.txt` row and on disagreeing duplicates except
  `_PARITY_DUPLICATES` (P11).
- `fast` vs `ffi`: `solve` is `lax.custom_linear_solve(matvec, b, solve,
  transpose_solve)` because a hand-written `custom_jvp` cannot be transposed (jax
  0.6.2 `_scan_transpose` asserts); `transpose_solve` is, for both backends, the
  `jax.vjp` of the reference sweep on the backend's own factors. The `ffi` CPU
  kernel is the CUDA kernel's reference, not a fast CPU solver. Each built library
  embeds the sha256 of its sources; a stale one is refused.

### Oracle policy

- Never cite a hand-patched VULCAN-master copy for what upstream ships. The one
  that existed carried this port's stall detector, a 13-species `conver_ignore`,
  `wall_clock_max` and the C6/C7/C12 patches; citing it produced the paper's
  "HD189 1296 matches exactly" and W39b "1202 in both codes" (`eebc8a5`).
  `audit_master_parity.py` refuses a checkout with vulcan3 identifiers and has no
  default `--master`; a master child process imports nothing from the port.
- `exoclime/VULCAN` redirects to the VULCAN 3 line (its `vulcan_cfg.py` is
  byte-identical to `vulcan3_vm_branch`): the pinned clones are the only VULCAN 2
  reference.
- Declared code deltas (`tests/oracle.py::ORACLE_CODE_DELTAS`): an upstream defect
  corrected on both sides is one (OLD, NEW, tag) entry applied to the disposable
  oracle copy and to `bench_table1.sh`'s scratch copy; OLD must occur exactly once
  (`test_oracle_pinning`). Master's 20-step trajectory depends on the hash seed,
  so oracle subprocesses pin `PYTHONHASHSEED=0`.
- Upstream changed `NCHO_photo_network.txt`, `SNCHO_photo_network.txt`,
  `all_compose.txt` and the HD189 / HD209 Kzz files after 80f75b9; the port keeps
  the pinned versions. The pin can move only within VULCAN 2 history, and a move
  re-baselines every oracle family and gate. Exception (the maintainer's): the 2025
  SNCHO file is upstream's a68d715 one (`tests/science_sources.yaml`), `oracle: null`.

### Molecular diffusion and the hybrid scheme

- `use_vm_mol`: interface-centred drift recomputed in-loop every `update_frq`
  (`atm_refresh.recompute_vm_jax`); `compute_mol_diff` reproduces vm_branch's
  `update_mu_dz` bit for bit.
- Hybrid (`use_hybrid_vm_mol`, only `HD189_vulcan3.yaml`): carry `hybrid_use_vm`
  flips 1 -> 0 in-loop on `_convergence_ok` alone; phase 0 cannot end a run; flip
  priority convergence, runtime, step count (upstream's `stop()`); phase-1 budget
  `count + 2000` (convergence), `count + 1000` + `runtime * 1.1` (runtime),
  `count + 1000` (count).
- Zhang+2013 Eq. 1 benchmark (`git show
  fe17de5:benchmarks/zhang2013_moldiff_benchmark.py`): central goes negative
  between cell Peclet 1.55 and 1.88; upwind stays positive.

### Differentiability

- A quantity is differentiable when it reaches the runtime as a JAX array
  (`rates_jax` T -> k, `atm_setup.mu_dz_g_jax`, `atm_jax._mol_diff`,
  `kzz_profile_jax`, `sat_p_jax`, `settling_velocity_jax`; coefficients via
  `build_rate_array(..., rate_coeffs=, nasa9_coeffs=)`). [M/H] and C/O are `y0`
  tangents: the published `d ln SO2 / d ln Z = 2.6` for W39b.
- Blocked: `d/dT_irr` through Heng+14 (`expn` forward mode is pathologically
  slow; differentiate `Tco`), stellar flux (closure-baked), the cross-section T
  rebake, `alpha` and `pref_indx` (discrete), condensation `r_p` / `rho_p`. The
  EQ seed has a zero tangent.
- Reverse mode is the steady-state adjoint of the body map (`ros2_step` +
  renormalization + photochemistry + optional `make_body_terms`); clipping, ion
  charge balance and the geometry refresh (`atm_refresh_feedback`) are outside it.
  `_guard_unmodeled_processes` raises on condensation without terms and on ion
  states. `y_star` is a fixed point of the bare map; do not iterate the renorm map
  to tighten `fp_err`. No production Hessian; the recipe is in
  `jax_paper/scripts/hessian_demo/hessian_lib.py`.

### Condensation contract (F1-F5)

Upstream `master_pin` integrates a condensation window, snapshots the reservoir at
the first accepted step after `stop_conden_time` and pins it with `fix_species`.
Not differentiable as a whole (the snapshot rides the accepted-step sequence;
`max(0, y - y_sat)` and the cold trap switch discretely); the kernels are. Forward
runs, jit/vmap and jvp on a fixed smooth branch are supported; the reaction
adjoint after the pin is conditional on the frozen reservoir. F1
`steady_state_input_sensitivity` refuses `conden_static`, `fix_mask` or the
`*_l_s` fingerprint; F2 the `conditional_on_fixed_reservoir` label; F3
`validate_runtime_config` under `use_condense` refuses `use_moldiff=False`, an
empty `condense_sp` and `stop_conden_time < start_conden_time`; F4 docs; F5
vulcan-retrieval gates on the resolved `chem.conden_spec`.

### Photolysis decisions

Both kept for parity.

- P19, the layer-mean actinic flux: inside a layer the beam is `F_top *
  exp(-dtau/mu)`, whose exact mean is the logarithmic mean; the code (and
  upstream) uses the arithmetic mean, ratio `(x/2)coth(x/2)` (converged HD189:
  median 1.000000, p99 2.23, max 215). The exact mean on the direct beam only:
  1495 -> 1500 steps; |dlnVMR| CH4 1.5e-2, NH3 3.0e-2, H2O 4.1e-1 (top microbar
  layers). The diffuse streams need their own layer mean, not derived. Maintainer:
  kept for parity; the observable impact is unmeasured, not shown harmless.
- The >240 nm grid (`dbin2: 2.0`) point-samples ~0.1 nm data: TOA J errors OH
  +24.8% (HD189), W39b SH +47.6%; at 1e-5..1 bar the worst is NH3 0.6%. `dbin2`
  0.5 removes ~90% of it for +24% bins. Kept with P19.

### Decided by the maintainer, not to be re-opened

The atom-conservation projection, the Ros2 stage repair, `DT_MAX_S`, the solar
abundance file and the retry cap stay (both referees keep them); batches B-D of
the provenance audit are not owed. The geometry and element-budget certificate
terms stay (both referees recommended it): the rule "keep VULCAN's original
convergence criteria" keeps every VULCAN 2 test and allows extra tests that fix a
confirmed defect; a relaxation of VULCAN's gate is never adopted.

The parity audit's three ion-mode / low-T items stay at parity; P4, P5, P7, P8,
P9, P10 stay; NH3 and CH stay in the default network (removal re-baselines every
gate for no measured benefit); the C20 report is not filed upstream; the
tangent-stall early stop is not built; `test_vmap_photo_batch`'s ymix assertion
is the scale ratio `max|lane - solo| / max|soloA - soloB|` (`SCALE_TOL` 1e-4); the
photo-on batched deviation (13 of 87 carry leaves at the last bit) is accepted as
roundoff; the AD-build `dt_max` cap stays off the cold solve; `config.DT_MAX_S`
stays 1e15. The JAX paper quotes 0.6.0, the version its data came from. The CUDA
solve guard stays at its ~7% solve cost ("Efficiency program"). Upstream's rocky
seed (`use_other_ele` + `fastchem_met_scale`: FastChem holds O in species the
network drops) is not ported: the seed is ExoGibbs over the network's elements.

## Known and deliberately untracked

- Never run (no shipped claim rests on them): HPC chunked-Jacobian peak at 512
  lanes, untiled-vs-tiled throughput, GPU parity, exhaustive config-combination
  oracle sweeps, long oracles for every vendored example, the nonsense-value
  config space, whether the `update_frq` refresh cadence costs convergence.
- `tests/conftest.py`'s cfg snapshot drops attributes it cannot deepcopy;
  `photo_setup._discover_T_cross_files` hardcodes `thermo/photo_cross/`; the
  `run_queue` refill defaults (`_QUEUE_REFILL_CHUNK` 8, `_QUEUE_REFILL_EVERY` 100)
  are untuned; the `save_evolution` buffer is sized from the static `count_max`.
- Pin configs only: `steady_state_grad`'s body map sets the layer-0 pins after the
  rebalance where the runner pins before it (C17); `make_body_terms` does not
  dedupe pin indices.
- `analytical_TP_H14`'s `expn` is slow on arrays mixing arguments below and above
  1 (nz 50 `load_TPK` 4:14 on one core); no shipped config uses it.
- A BC-flux row naming a species outside the network is skipped silently
  (`atm_setup.read_bc_flux`); upstream raises (`build_atm.py:759,768`).
- `tools/bench_table1.sh` writes under the system temp dir unless `BENCH_OUT` is set.
- Undecided maintainer calls from the bloat audit: whether `steady_state_grad`
  stays as a research route, `HD189.yaml`, `VULCAN3_SOLVER=reference`. Left for a
  pass with a test run: script-style `main()` bodies in 10 test files.
- Reverse mode under `VULCAN3_SOLVER=ffi` is documented unsupported but not refused.
- No CPU test reaches the Gauss-Jordan kernel's division fallback since `2216219`
  removed the `tiny` case (row 3 scaled by 2^-975) from `tests/test_solver_fast.py`;
  the GPU A/B solved a subnormal pivot exactly ("Efficiency program").

## Superseded designs

- Jacobian assembly: `jacrev` -> flat scatter (OOM'd a 96 GB GH200 at batch 512)
  -> chunked scatter (`_JAC_CHUNK_REACTIONS` 128, 1.3-1.9x slower than flat) ->
  gather tables (`764c1ba`, 0.16.5). `git revert 764c1ba` undoes it alone (move
  vulcan-retrieval's HPC pin with it).
- FastChem EQ seed and tree: replaced by ExoGibbs in 0.15.0.
- Stall fallback (`use_conv_stall`, 89b5962/9c2bb75): removed `c009555`, before
  the fixedmass and v2 production runs; it never fired on a shipped case.
- Gustafsson PI controller: removed `cf993e2` (11-17% more steps; HD189 1296 ->
  1475, W39b 1202 -> 1444 at the time). If re-ported, sanitize
  `(delta_prev/delta)**(beta/2)` to 1.0 at the no-history sentinel.
- The 0.5.1 repair dt gate (repair only above 1e7 s): replaced by the absolute
  floor in 0.6.0 (HD209 carbon leaked 2.5e-4 under it, now 8e-6).
- `solver_map="bare"` and its W39b adjoint gate: removed `d966010`.
- Route B (opt-in smooth rainout: one-sided C1-hinge S8 sink, deep sulfur supply,
  flux ledger, implicit derivative): NO-GO at B0C (no steady state in 15000 cold
  steps; closure_frac 0.264; unrolled jvp off FD by 6-9 orders). Code in
  `~/VULCAN_Project/.backup/git_tags_2026-09-26/VULCAN-JAX-smooth-rainout-b0c-no-go-2026-07-14.bundle`
  (tag a092d9e deleted); Route A (exact-endpoint pin, ~1-1.5 weeks [E]) not asked for.
- `ffi_twisted` (two-ended block-Thomas): built, not merged (maintainer: no large
  architecture-specific code; net 649 lines; daw 1.9-2.0x up to 71 lanes,
  1.00-1.02x once the GPU is full). Patches in
  `~/VULCAN_Project/.backup/speed_studies_2026-09/gpu_ideas/twisted/`.

## Ideas not adopted

- Cross-step Jacobian / LU reuse (legal in a W-method): LU+J k=5 gives 3.2-4.3x
  fewer factorizations but +8.7 / +29.3 / +37.9% steps and W39b parity 1.1e-9 ->
  1e-3..6e-3. Rejected by the maintainer; re-open only with a measured net GPU
  gain at a parity cost the maintainer accepts.
- Kernel rewrites designed, not started: (a) the six-direction tangent solve in
  one kernel pass (tangent block solves are ~51% of the gradient step, job 79997)
  with a finished-lane mask for the straggler tail; (b) the chemistry RHS in the
  gather form (a new summation order moves step counts: needs a parity call).
  A fourth lane per SM at ni 89 needs ~10 rows in registers.
- A bin-averaged >240 nm grid with rectangle weights (segment median error 5.25%
  -> 0.068% at no extra bins).
- Non-uniform pressure grid support (a 284-layer non-uniform grid matched uniform
  600 to <= 0.06 sigma at ~55 h; vulcan-retrieval); a 12-line pco-file patch to
  `state.py` was prototyped and not kept.
- A whole-loop command buffer for `ffi` (GH200 host syncs are GPU waits, 82% of
  API time) and caching the transport coefficients (XLA fusions 12% of primal
  kernel time): declined by the maintainer.

## Do not retry

One line each, with the measured reason.

Adjoint and tangents
- Residual-IFT adjoint of `df/dy`: singular, cond ~1e28; gradient +876 vs FD -0.565 although every component was correct.
- LSQR pseudoinverse of `J^T` (`istop=7`, gradient ~0); a raw Neumann fixed-point adjoint (body map not a contraction, |lambda| 2.66).
- Restarted GMRES(300), un-restarted GMRES(2500), GCROT, LSMR: oscillate, break down, blow up, FP-dead.
- A left preconditioner or `reg` tuning instead of the solver map (cond(M) ~3e20); orthogonal deflation of slow modes (biases); one-sided eigendeflation (resid 21, non-normal).
- `body_dt` >= 3e8, <= 1e6 or 1e11 (resid 1e57); 3e6 only when determinism beats a 28% slow-mode bias.
- Tightening `yconv` for adjoint accuracy (7.0 / 6.3 / 8.4% at 1e-2 / 1e-3 / 1e-4; the bare map was the cause); `renorm_td` (0.7% -> 2.5%); reading `pair_antisym` ~1 as "gradient wrong".
- A tangent restarted from zero at the exit dt, or the fixed-point IFT solve, instead of path-following `run_jvp`: 50-500x off re-converged FD.
- Stop-gradient on the adaptive timestep: full-integration Kzz error 1.27e-4 -> 0.388 with identical primals.
- Reading the tangent at the chemistry certificate through a C21 veto continuation (eclipse lnZ corr -0.09), or a step floor on a differentiated map: certify the tangent (C22).
- A tangent-settling metric normalised by the largest sensitivity: hides small unsettled cells; a decaying tangent never certifies under a relative rule.
- Photolysis cadence as the eclipse tangent cure (a frozen field still grows; more refreshes is worse); certifying the observable instead of the state (depth-row rms 6.6e-5 -> 789); an implicit derivative at the uncapped eclipse state (never a fixed point); `aflux_change` in the dt controller (no threshold separates the regimes; HD189 freezes at dt 1e-10). The cap stays in the consumer.
- Photo-off Fisher/FD on the default W39b profile without the convergence recipe, or FD across non-comparable states (rows miss by 23-400x).
- Weakening `audit_adjoint_scope` to certify the JWST tool's tabulated W39b state.
- Reading a NaN batched tangent as a vmap or solver bug: it was `0 * inf` in the delta/longdy reductions.

Steady state and the dt controller
- Loosening the controller to cut steps (Gustafsson PI +10-17%, damped Shampine/RODAS estimate -26%, looser `mtol` -28%, upstream's `rtol=0.25` -26%): all "converge" 11-102% off the converged column from 8.7e-2 bar down; `rtol=0.5` and `safety=1.0` never converge.
- Raising `dt_var_max` 2 -> 10 or `update_frq` 100 -> 1: HD189 stays at exactly 1495 steps.
- Damped Newton or pseudo-transient continuation: dt -> inf already is Newton; relaxing rtol drives C/N/O drift to 0.3-2.1, then non-finite.
- March-then-Newton: diverges in 11-19 steps even from 0.1% off converged.
- Anderson acceleration on the step map: never beats Picard at equal evaluations; deeper is worse (5.2e-6 / 1.4e-5 / 1.2e-4).
- Cross-step LU+J reuse as a production policy (see "Ideas not adopted").
- Interpolated warm starts (|dlogL| up to 3 at rtol 0.2), tangent starts (1.13x), a dt kick (lost certifications).

Linear solver and precision
- Inverse carry in block-Thomas: 2.0x under jvp, 569x worse residual at dt 1e11.
- Equilibration or iterative refinement for the large-dt block-Thomas residual: no run reaches it; equilibration costs +16% steps on HD189 and does not reduce the stage element defect.
- An unpivoted Thomas sweep for the repair tridiagonal (no structural guarantee); parallel cyclic reduction (backward error 1.4e-5 at 1e15, cannot pivot); the sweep fully unrolled (jvp compile past 574 s at 7.9 GB).
- Parallel-in-layer block solvers (BCR 89 ms vs 39 ms at 396 lanes, OOM at 1188); explicit-inverse solves (forward error ~600x); a 4-rung compaction ladder (not bitwise; rungs 66 and 33 replay at 1.03x of a v5 batch, vulcan-retrieval notes "Pipeline speed").
- Other cures for the per-tick y-ring copy: the pre-write lookback alone (bitwise; one copy of 15 gone, tick unchanged), `--xla_gpu_memory_limit_slop_factor=200` (remats 6066 -> 3430; copies and time unchanged), `CONDITIONAL` in the command-buffer list (no change). A sequential `lax.scan` for the two-stream sweeps is not faster than the associative scan on GH200.
- Float32 in the step: float32 factors (forward error 1-4e2 at dt 1e2, refinement diverges), float32 tangent solves with float64 refinement (diverge at dt >= 1e11), the whole step (non-finite); float32 Jacobian terms with float64 sums are accurate but touch ~12% of GPU kernel time.
- AMF or block-Jacobi preconditioned GMRES/Richardson on the stage system: 14-81 iterations at the run's dt, > 300 at 1e6, cells O(1) off (`~/VULCAN_Project/.backup/speed_studies_2026-09/amf_tests/`).
- Kernel restructurings (one-barrier owner-warp LU 0.89-1.03x, batched rank-1 update 0.90-1.04x, register-resident inverse at 253 registers, DMMA not bitwise, TMA): the factor gains only from more lanes per SM.
- Summing the gather-table Jacobian with a `reduce`: XLA orders a reduce per program, so batch and queue disagreed by one accepted step (122 vs 121).
- Fusing the step's two sign reductions into one `jnp.min`: a NaN plus a negative reclassifies a rejected step.

Stage repair and certificates
- The repair at dt >= 1e17 or `dt_max` uncapped: the correction tridiagonal is as singular as the stage system.
- Judging the stage identity against `c0 |a^T y|` (roundoff of cancelling terms); a 1e-12 bar at dt 1e15 (conditioning, not roundoff).
- No defect floor (stalls the HD 189733 b Guillot column at 30000 steps), a dt gate, or a floor on the layer's largest element scale (never drops at steady state): only an absolute floor works.
- A cell-wise identity pin on the second Ros2 stage: two dropped stage-1 corrections reach 9421 of 10350 cells.
- Escape flux in the geometry norm (HD209 variants ran to a 3001-step cap); `count_min = 1000` or photolysis cadence 1 as the stale-geometry fix; a refresh-only geometry rewrite at return time.
- An element-budget tolerance of 1e-3 (a certifying W39b photo-off column sits at 2.16e-3 from the boundary rows); normalising C23 by a running reference (dilutes a drift through a refresh stretch).
- A `dz`-weighted `atom_loss` in step acceptance without a full re-baseline.
- Adding species to `conver_ignore` to make a case converge (that is what 1296 was; sulfur allotropes hide the large-dt leak): gate the element budget.

Parity and data
- Citing a hand-patched VULCAN-master copy for what upstream ships.
- HD189 (Kzz 1e10) as evidence the upwind scheme matters (0.000-0.008 dex); freezing `vm` at setup (1.7 dex).
- A both-sided rate-range clamp (the "-77% SO2" artifact).
- A strictly-increasing stellar-wavelength check (C4's file has duplicate wavelengths).
- "Fixing" the escape-flux Jacobian entry or the S8 diagonal cells to match upstream, or patching the TiSNCHO / SNCHO_C3 duplicates, `CH3SH_branch.csv` or the CS-family masses: breaks bit-parity or touches unconsumed data.
- Matched-step parity past ~step 145 at rtol >= 0.25 (FP amplification): compare converged states.
- A higher-order vertical transport scheme: the shipped eddy and central moldiff kernels measure second order (1.95-2.00) on a 15x-stretched grid, and nz 150 is converged to about 1% at observable pressures.
- Bit-moving refactors: folding atm_setup's mu/dz scans into `atm_refresh.hydrostatic_column` (1-2 ulp eager) or `update_mu_dz_jax` onto `atm_setup.mu_dz_g_jax` (`zco` 1 ulp on W39b under jit), one bin-weight contraction in `photo.compute_J_jax_flat`, the shared transport-kernel helpers (`c756201`, reverted `f3dfdf7`: vulcan-forward's `test_queue_is_differentiable` tangent 3.0e-4 -> 1.6e-2). Transport-kernel refactors need that full-tier test.
- ExoGibbs's bundled FastChem 3.1.3 tables as the seed's data (a second dataset, no excited states); exogibbs with its declared dependencies; `scan_hot_from_top` (cheaper, but moves the seed and HD189_vulcan3 misses its 20000-step cap).
- Deselecting the `oracle.yml` HD209 codegen cases or pinning their arm64 FMA residual: they assert the platform-independent invariant since `ebcb780`.

Process and environment
- `pytest.importorskip` / conftest tricks that reorder the import-frozen network; `CondenStatic` as a jit argument; a pre-import CLI launcher in the package; in-process network hot-swap.
- Route B smooth rainout or its middle paths, or a `custom_jvp` wrapper on `_runner` (see "Superseded designs").
- Lifting the `use_sat_surfaceH2O` refusal into core (retrieval-specific).
- Regenerating `.npz` fixtures when `AtmStatic` grows: splice a default in the loader.
- Trusting a `.vul` byte comparison across runs made before the branch-key ordering fix (`2fdc66b`).
- `vulture` or grep as a dead-code proof on master-facing facades (`op.py:951` calls `.mean_mass()`; vulcan-forward calls `naming_solver`).
- `k_arr = 1e-15` as "near-zero chemistry" in a synthetic test: three-body rates carry M = 1e15; use 1e-30.
- Quoting a wall time without cpu/wall.
- Earth and K2-18b cures already falsified: `use_lowT_limit_rates`, `P_t: 1.0`, conden/settling off, `high_temp_cut`, the stall fallback; the Gibbs T-range warning is a red herring.
