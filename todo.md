# todo

Evidence for these items is in `notes.md` (this repo's, unless another repo is named).

1. [decision] Claim `vulcan3` on PyPI: the README installs put TestPyPI first, so a TestPyPI package can shadow its PyPI namesake.
2. [unverified] The Jacobian power change (`chem._jac_pow_tables`) is bitwise only where measured, on CPU: check on a GH200 that it leaves a W39b batch unchanged.
3. [decision] The arm64 full tier fails `test_fast_on_real_w39b_blocks[fast]`: on the W39b 2025-network fixture the matrix-free step's primal under `jvp` is not bitwise the plain step's (bitwise on x86-64).

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
