# todo

Evidence for these items is in `notes.md` (this repo's, unless another repo is named).

1. [owed] Re-baseline after the C25-C29 data and step-control fixes and the C30 extension to dissociations: the arm64 full tier.
2. [decision] The test of `oracle.md`'s 10 % step-count rule against VULCAN 2 does not run in `oracle.yml`.
3. [decision] `oracle.md` "Settings are visible": some result-changing numbers are fixed in code, and the environment overrides a config's network without a message.

## 1. Re-baseline after C25-C29

Evidence: notes "Parity ledger" C25-C32 and "Current state". The step-count gates read the
x86-64 counts on arm64 too, the tracked arm64 manifest is rebuilt, and `oracle.yml` is green at
this code. Left: on the laptop `VULCAN3_RUN_SLOW=1 python -m pytest tests -n 12 -q` (the whole
full tier; only the gates and `test_solver_fast.py` were run). Done when it passes.

## 2. Step counts against VULCAN 2 in the release gate

Evidence: notes "Evidence behind the numerical rules" (Step counts against VULCAN 2): all three planets are inside 10 %.
`tests/test_step_count_gates.py::test_step_count_within_10_percent_of_vulcan2` is `slow` and needs
`$VULCAN_MASTER_DIR`. In `oracle.yml` the "Full tier" step sets no `VULCAN_MASTER_DIR`, so the test skips there, and
the comparisons step does not set `VULCAN3_RUN_SLOW`. Setting the clone on the full-tier step adds about 21
test-minutes on daw (6-9 a planet) to a tier at 59 of its 90. The maintainer decides where it runs. Done when an
`oracle.yml` dispatch runs it green.

## 3. Settings outside the config file

Evidence: `oracle.md` "Settings are visible" and CLAUDE.md "Style" (magic numbers live in the config YAML). Fixed in
code and not in `configs/default.yaml`: the loose-branch slope threshold (`_convergence.py`: 0.1 of a scale height, cap
1e-8, floor 1e-10 s^-1; VULCAN 2 fixes it too), the scattering asymmetry `ag0` (`phy_const.py`), the 1 bar reference
level of `Rp` and gravity (`_transport_setup.py`), the 2-700 nm photolysis window (`state.py`) and the thermal-diffusion
factors (`_transport_setup.py`). `$VULCAN3_NETWORK`, `$VULCAN3_ATOM_LIST` and `$VULCAN3_COM_FILE` replace what a config
names with no message (vulcan-forward names the network in its profile and refuses a mismatch). The maintainer decides
which become config keys, which stay fixed with a recorded reason, and whether a config that names another network than
the environment's is refused. Done when each is a key or has its reason beside it.
