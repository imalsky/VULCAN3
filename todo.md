# todo

Evidence for these items is in `notes.md` (this repo's, unless another repo is named).

1. [decision] Claim `vulcan3` on PyPI: the README installs put TestPyPI first, so a TestPyPI package can shadow its PyPI namesake.
2. [unverified] The Jacobian power change (`chem._jac_pow_tables`) is bitwise only where measured, on CPU: check on a GH200 that it leaves a W39b batch unchanged.
3. [decision] The arm64 full tier fails `test_fast_on_real_w39b_blocks[fast]`: on the W39b 2025-network fixture the matrix-free step's primal under `jvp` is not bitwise the plain step's (bitwise on x86-64).
4. [owed] Re-baseline after the C25-C29 data and step-control fixes: the arm64 fixture manifest, the arm64 gate reading and the oracle comparisons (`oracle.yml`).
5. [owed] Rebuild the CUDA kernel on the NAS and check `ffi` on a GH200: the code cleanup changed the kernel sources, and its proof of `ffi` ran on CPU only.
6. [decision] Since C29's negativity rejection, HD189 photo off cannot certify at the shipped `nega_cut`: keep the rule or change the cut.

## 1. vulcan3 on PyPI

The name was unclaimed on pypi.org on 2026-10-16 (404), as `vulcan-jax` was.
Every README installs with `-i https://test.pypi.org/simple/ --extra-index-url
https://pypi.org/simple/`, and the 2026-10-16 install tests took `psutil 8.0.0`,
`python-dateutil 3.9.0` and `radis 0.16.4` from TestPyPI. Claiming the name needs
the maintainer's pypi.org account. Commands: `uv build --out-dir <dir> .` in this
repo, then `uv run --with twine twine upload <dir>/*`. Publishing all four packages
on PyPI would let the READMEs drop the TestPyPI index. The 0.18.0 TestPyPI release
and its tag were made in a session whose record is not on daw; vulcan-retrieval's
todo lists confirming them. Done when the name is claimed (and, if decided, the
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

## 6. HD189 photo off under C29

Evidence: notes "Forward mode". No shipped photo-on number moves: C29's code alone is bitwise
on the three gates. The column is a test and fixture setup, and any user's `default` with
`use_photo: false`: it ends on its step cap. Grepped `use_photo` false in vulcan-retrieval's
and jwst-transit-authority's `src`: the planner's GUI offers it, with moldiff on or off, and
its columns certify (notes). Choose: (a) keep the rule, photo-off HD189 needs `nega_cut: -inf` as the tests set it; (b) a
looser shipped cut (-1e6 removes every rejection on this column); (c) exempt or scale by cell,
which is not built: `-1e3` shows trace cells with VMR 1e-15 stop it next. (b) and (c) touch a
settled item and need maxichem's 0-D case re-run, the case that motivated C29. Done when decided
and recorded in the "Parity ledger" C29 row.
