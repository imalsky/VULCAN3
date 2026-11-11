# oracle.md — what vulcan3 must achieve

This file states the goal of vulcan3 (this repo) and what counts as success or failure. It is the
yardstick for judging whether the repo is ready to ship. It holds targets and decisions, not
measurements: the evidence lives in notes.md, named here by section heading. Only the maintainer
changes this file. The goals of the other repos are in the project-root `oracle.md`.

Not to be confused with "the oracle" used everywhere else in this repo: the pinned VULCAN 2 clones
(`exoclime@80f75b9`, plus `vm_branch@84d010d` where a note says so) that vulcan3 is compared
against.

## Goal: vulcan3 is VULCAN 2, in JAX

vulcan3 is a working port of VULCAN 2 with almost all of it rewritten in JAX. It runs every live
VULCAN 2 Ros2 physics path and gives the same answer as the pinned VULCAN 2, while being jit-able,
batchable with vmap and differentiable. It uses the newest upstream networks. Every place it departs
from VULCAN 2 is a decision the maintainer made, with a recorded reason, usually an upstream bug
fixed.

"The same answer" does not mean bit-identical. JAX and XLA reorder floating-point work, so values
and step counts move by small amounts. A correct run is inside the tolerances set for it below.

### Success

- **Layer parity.** Rates, reverse rates, atmosphere setup, the EQ seed, the chemistry RHS and
  Jacobian, diffusion, the block solve, one Ros2 step and photolysis each match VULCAN 2 at the bars
  in notes "Parity numbers", each pinned by the test named there.
- **Matched steps.** Stopped after the same fixed number of steps on HD189 (the test sets the number;
  it is not a result), vulcan3 and VULCAN 2 stay inside the bars of `test_default_master_parity`,
  which tighten with fewer steps.
- **Converged columns.** Run to convergence, HD189, HD209 and W39b agree with VULCAN 2 at the level
  recorded in "Parity numbers" (measured with `tools/compare_vul.py`). A larger difference is either
  a declared departure or a failure.
- **Step counts.** Run freely to convergence, vulcan3 does not have to take VULCAN 2's number of
  steps: the two round differently, and VULCAN 2's own count moves by about 1 % from run to run.
  The count is within 10 % of VULCAN 2's on the same case. Against itself vulcan3 is exact: HD189,
  HD209 and W39b certify at their pinned counts (`tests/test_step_count_gates.py`, full tier),
  because the same code gives the same count on every machine. That pin is an alarm for an
  unintended change, not a comparison with VULCAN 2; a moved count is explained in notes or it is a
  failure.
- **Tolerances are fixed.** A bar is loosened only with a written reason in notes; never to make a
  run pass.
- **Settings are visible.** A number that changes a result is a named setting the user can read and
  change in the configuration, with its unit and its reason. It is not a value buried in the code.
  A physical constant or a numerical guard that is fixed on purpose is a named constant with its
  reason beside it.
- **Convergence is VULCAN's own.** Every VULCAN 2 stopping test is kept. An extra test is allowed only
  when it fixes a confirmed defect; a relaxation of VULCAN's gate is never adopted. A run ends on the
  certificate or on a cap, or it raises: a column frozen at its smallest step is an error, never a
  result.
- **Every departure is declared.** A departure is made only to fix a confirmed defect that changes
  results. Machine-checked in `tests/science_sources.yaml`
  (`intentional_divergences`), `tools/audit_master_parity.py` and `tests/oracle.py`
  (`ORACLE_CODE_DELTAS`, `ORACLE_DATA_DELTAS`), or listed in notes "Parity ledger" with its reason. The
  oracle copy receives the same corrected data and code, so the comparison measures the port, not
  the fixes.
- **Differentiable.** Derivatives are of the converged steady state, not of VULCAN 2's step-by-step
  path, so they are approximate. Each route is checked against re-converged central finite
  differences at its test's bar: forward mode (`test_forward_jvp_integration`), the tangent
  certificate (`test_tangent_certificate`) and the reverse-mode steady-state adjoint
  (`test_steady_state_reaction_sensitivity`). Inputs with no derivative (the EQ seed among them) are
  listed in notes "Differentiability".
- **Batchable.** Every shipped config runs under jit and vmap. A lane's result does not depend on the
  batch width. A batch lane matches the solo run within the bar of `test_vmap_photo_batch`; batch and
  solo step counts differ by design.
- **Networks.** The defaults are the newest NCHO network (`NCHO_photo_network.txt`) and the newest
  sulfur network (`SNCHO_photo_network_2025.txt`, upstream VULCAN's current sulfur chemistry). A sulfur
  result on the older `SNCHO_photo_network.txt` is superseded. The 2025 file has no VULCAN 2 oracle
  (`oracle: null`); it is upstream's file, adopted by the maintainer.
- **Default bars.** Where a test sets no bar of its own: agreement with VULCAN 2 at machine
  tolerance, a vmapped kernel against the single call at about 1e-12, and a gradient against finite
  differences at about 1e-6.
- **Standalone.** vulcan3 runs from its own package: its data is vendored, it needs no sibling repo at
  run time, and its runtime dependencies are jax, numpy, scipy, PyYAML and exogibbs.
- **Inputs are pinned.** Every vendored input file is hash-pinned (`tests/science_sources.yaml`), so a
  changed input is a declared change.
- **Output VULCAN 2's tools read.** A run writes a `.vul` pickle with VULCAN 2's three top-level keys
  and the same public keys, shapes and dtypes, so VULCAN 2's plotting scripts load it. Byte identity
  is not a goal.
- **The oracle is upstream as shipped.** Comparisons run against the pinned clones, never a
  hand-patched copy. The pin moves only within VULCAN 2 history, and a move re-baselines every oracle
  comparison and gate.

### Deliberate departures from VULCAN 2

Each has its reason and size in notes "Differences from VULCAN master" and "Parity ledger"; the ones
the maintainer closed are in "Decided by the maintainer, not to be re-opened".

- **Upstream errors corrected** (ledger table A). Data: condensate and species molar masses, the H2S
  saturation constant, the eps Eri spectrum scale, the FastChem P/S row order, duplicated or wrong
  NASA-9 sets (N(2D), HNO3), a reaction k0 typo, unbalanced or one-sided rows. Rates: the OH + CH3 + M
  Troe width in log10 (Visscher & Moses 2011), low-temperature reverse rates above the collision limit,
  and a collision limit on every thermal pair, applied to both directions so equilibrium is kept.
  Step control: VULCAN's Ros2 accept/reject rules that let a wrong step through, and the step floor
  that freezes a column once force-accepted.
- **Stability fixes VULCAN 2 lacks.** Ros2 stage repair with a lower step cap, `config.DT_MAX_S` (pinned
  upstream fails the same column); the atom-conservation projection onto fixed reservoirs (XLA
  reordering otherwise leaks atoms); a retry cap with force-accept that is inert on shipped settings.
- **Extra convergence terms.** Geometry and element-budget terms in the certificate (each fixes a
  measured false certification), and a tangent term on the differentiated path only.
- **Inputs.** The solar abundance file (matches the declared He/H; drops elements the networks cannot
  hold), gravity from `G Mp / Rp^2`, and the ExoGibbs equilibrium seed in place of FastChem. ExoGibbs
  is a JAX minimizer, so the whole run, seed included, stays in one jit-able and vmap-able JAX
  program with no external binary; it uses the same thermodynamic data as the reverse rates. Its
  derivative is zero by design: the seed is an initial condition, and the converged state does not
  depend on it. The seed differs slightly from FastChem's, within the `test_eq_seed` bar; the effect
  on converged runs is recorded in notes "Differences from VULCAN master".
- **Kept at parity on purpose.** Upstream defects the port reproduces rather than fixes are listed in
  notes "Known bugs, all repos". They stay unless the maintainer decides otherwise.
- **Not ported, by scope.** What VULCAN 2 ships outside its live Ros2 physics (symbolic Jacobians,
  other solvers, pickle byte-identity, FastChem internals, and the rest listed under "Refused or not
  ported"). Inputs upstream accepts silently and fails on later are refused at setup.

### Failure

- A difference from VULCAN 2 outside its bar that is not declared.
- A step-count gate moved or re-pinned without a recorded cause, or a converged count more than 10 %
  from VULCAN 2's with none.
- A number that changes a result, hidden in the code where the user cannot see or set it.
- A bar loosened, or a convergence gate relaxed, to make a run pass.
- A derivative used without a finite-difference check on its route, or presented as exact.
- A batch lane that depends on the batch width, or outside its bar against the solo run.
- A shipped result on a superseded network, or a parity claim made against a patched copy.
- A silent fallback or default where the code should refuse (project-root `CLAUDE.md`, "Error fast").
- A column frozen at its smallest step returned as a result.
- A vendored input changed without its pin, a new runtime dependency, or a sibling repo needed at run
  time.
- A change to a decided item without the maintainer.

## Ready to ship

The repo is ready when, at the commit to be released:

- the normal tier passes locally and in CI, and the full tier passes when the change touches what it
  covers;
- `oracle.yml` passes against the pinned clones, the step-count gates hold, and
  `tools/audit_master_parity.py` finds only declared departures;
- no `[owed]` item in todo.md backs a claim the release makes;
- notes "Known bugs, all repos" is current, and nothing above under Failure applies.
