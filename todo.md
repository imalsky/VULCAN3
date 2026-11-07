# todo

Evidence for these items is in `notes.md` (this repo's, unless another repo is named).

1. [decision] Claim `vulcan3` on PyPI: the README installs put TestPyPI first, so a TestPyPI package can shadow its PyPI namesake.
2. [unverified] The Jacobian power change (`chem._jac_pow_tables`) is bitwise only where measured, on CPU: check on a GH200 that it leaves a W39b batch unchanged.
3. [decision] The arm64 full tier fails `test_fast_on_real_w39b_blocks[fast]`: on the W39b 2025-network fixture the matrix-free step's primal under `jvp` is not bitwise the plain step's (bitwise on x86-64).
4. [owed] Re-baseline after the C25-C29 data and step-control fixes: the arm64 fixture manifest, the arm64 gate reading and the oracle comparisons (`oracle.yml`).
5. [owed] Rebuild the CUDA kernel on the NAS and check `ffi` on a GH200: the code cleanup changed the kernel sources, and its proof of `ffi` ran on CPU only.
6. [decision] Confirm or replace the collision limit C30 in the oracle copy: `oracle.yml` passes `test_gibbs` and the HD189 19-step case only because the copy now applies the limit too, so no oracle checks its effect.
7. [decision] Should the collision limit C30 cover a third-body dissociation? `S4 + M -> S2 + S2 + M` reads 860 times the limit at 2620 K.

## 1. vulcan3 on PyPI

The name was unclaimed on pypi.org on 2026-10-16 (404), as `vulcan-jax` was.
Every README installs with `-i https://test.pypi.org/simple/ --extra-index-url
https://pypi.org/simple/`, and the 2026-10-16 install tests took `psutil 8.0.0`,
`python-dateutil 3.9.0` and `radis 0.16.4` from TestPyPI. Claiming the name needs
the maintainer's pypi.org account. Commands: `uv build --out-dir <dir> .` in this
repo, then `uv run --with twine twine upload <dir>/*`. Publishing all four packages
on PyPI would let the READMEs drop the TestPyPI index. The 0.18.0 TestPyPI release
and its tag have no record on daw; vulcan-retrieval's todo lists confirming them. Done when the name is claimed (and, if decided, the
packages are published).

## 2. GH200 bitwise check of the Jacobian power change

Evidence: notes "Efficiency program" (the stoich >= 2 power row). The change drops
`pow(y, 1)` for `y` on every stoichiometry-1 slot; XLA:CPU returns `y` from
`pow(y, 1)`, so CPU runs are bitwise, but no GPU guarantee is documented (libdevice
pow, subnormals kept on GPU). Needs the maintainer's NAS login. Steps on a GH200:
(1) under `jax.jit`, compare the uint64 bits of `jnp.power(x, e)` with a runtime
exponent `e = 1.0` against `x` over signed binades, boundary mantissas, subnormals,
zeros, inf and NaN; (2) run one W39b production batch (vulcan-retrieval's v5 shape,
`VULCAN3_SOLVER=ffi`, grouped RHS) at the commit before the change and at the change,
and compare per-draw step counts and converged `y` bit for bit. Done when both match
(record it in the notes row), or the size of any mismatch is recorded for the
maintainer.

## 3. arm64: matrix-free primal under jvp

Evidence: notes "Efficiency program" (matrix-free row). `check_matrix_free`
(`tests/test_solver_fast.py`) asserts on CPU that the step's primal under `jax.jvp`
is the same with the matrix-free and the dense stage operator. On the laptop
(arm64) the W39b 2025-network fixture gives 1.1e-11 to 2.6e-9 over the six cases
(dense vs plain step: 0), identically before the speed changes; on daw (x86-64) all
six are 0, and HD189 is 0 on both machines. Choose: an arm64 bar (as
for the grouped RHS), an arm64 xfail, or keeping the operator's primal RHS from
fusing with the step's. Command: `VULCAN3_RUN_SLOW=1 python -m pytest
"tests/test_solver_fast.py::test_fast_on_real_w39b_blocks" -q`. Done when the arm64
full tier passes and the notes row states the CPU scope.

## 4. Re-baseline after C25-C29

Evidence: notes "Parity ledger" C25-C32. The step-count gates are re-pinned to x86-64's
counts. Left: the tracked `tests/data/FIXTURES.json` is the arm64 laptop manifest from before
these fixes. Steps: on the laptop `python tests/gen_fixtures.py --all --force` and commit the
manifest; run `VULCAN3_RUN_SLOW=1 python -m pytest tests -n 12 -q` and check that the gates
read HD189 1495, HD209 1206, W39b 1084 there (a different count means the record of
machine-independent gates fails: record it, do not re-pin); dispatch `oracle.yml`
(`VULCAN_MASTER_DIR` at 80f75b9: the oracle copy gets the corrected files through
`tests/oracle.py::ORACLE_DATA_DELTAS`). Done when the manifest is committed and the arm64 full
tier and the oracle workflow pass.

## 5. NAS rebuild and `ffi` check after the cleanup

Evidence: notes "Cleanup proofs". The kernel sources were reformatted, so a library built
before the cleanup is refused until it is rebuilt. Lowered `ffi` programs moved when reverse
mode's transpose was removed; the optimized CPU program stayed equal apart from names. The GPU
was not run. Needs the maintainer's NAS login. Steps, after the commits are pushed: `git pull`,
`python -m vulcan3.solver_fast --cuda`, then the kernel test job on a GH200. Done when
the rebuilt library loads and the job passes.

## 6. The collision limit in the oracle copy

Evidence: notes "Parity ledger" C30. Pinned VULCAN 2 has no limit. Without a delta on the oracle copy, `test_gibbs` reads
reverse k 0.90 against its 1e-8 bar (18 of 390 reverse rows on the test's HD189 column) and the 19-step case 1.07e-5
against its 5e-9 bar; both pass before C30 (`e91db66`). So this is a real difference from VULCAN, not stale wiring. The code
takes option (a) below: `tests/oracle.py::ORACLE_CODE_DELTAS` applies the limit in the copy's `rev_rate`, as for C20, no bar
changed. `tests/test_collision_limit.py` is then the only check of the limit's effect, and the delta's arithmetic is a
second copy of `rates_jax.apply_collision_limit`. Choose: (a) keep it; (b) hold those two tests another way, with no changed
bar; (c) change C30. No record says which the maintainer wants. Done when the choice is recorded in notes "Decided by the
maintainer, not to be re-opened".

## 7. The collision limit on a third-body dissociation

`rates_jax.apply_collision_limit` skips a slot with one reactant molecule, so `S4 + M -> S2 + S2 + M` keeps k 5.6e-7 cm^3/s at
2364 K and 8.6e-7 at 2620 K, 560 and 860 times `K_COLL`, a loss of 1.3e13 and 1.8e13 s^-1 at the bottom of a WASP-39 b column, against
`K_COLL` M 2e10 s^-1 and kT/h 5e13 s^-1 [M]. The forward rate is `S2 + S2 + M`, k0 4e-31 cm^6/s, an estimate in the network (Moses2002/Zahnle2016). This lifetime is why
`dt_min` is 1e-15 (notes "Differences from VULCAN master"), which does not depend on this choice. Measured with the slot's pair capped, in a scratch copy:
floor ratio gone on the frozen draw, which certifies in 542 steps at log L -1413.462; 4 certified free draws move at most 1.4e-4 in log
L and 3.1e-4 in ln y (vmr above 1e-12); the shipped W39b and HD189 `.vul` move at most 3.3e-6 and 5.9e-7 relative, on cells above 1 cm^-3, at equal step counts (1084, 1495). So it cannot be
called neutral, and it would change relaxation and quenching where such a rate is rate-limiting [I]. Other slots of this kind were not listed. Choose: extend C30 to these slots, or keep it. Done when the choice is recorded in notes "Decided by the maintainer, not to be re-opened".
