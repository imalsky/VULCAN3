# todo

Evidence for these items is in `notes.md` (this repo's, unless another repo is named).

1. [owed] Re-baseline after the C25-C29 data and step-control fixes and the C30 extension to dissociations: the arm64 full tier and the oracle comparisons (`oracle.yml`).

## 1. Re-baseline after C25-C29

Evidence: notes "Parity ledger" C25-C32 and "Current state". The step-count gates read the
x86-64 counts on arm64 too, and the tracked arm64 manifest is rebuilt. Left: on the laptop
`VULCAN3_RUN_SLOW=1 python -m pytest tests -n 12 -q` (the whole full tier; only the gates and
`test_solver_fast.py` were run), and a green `oracle.yml` dispatch at this code
(`VULCAN_MASTER_DIR` at 80f75b9: the oracle copy gets the corrected files through
`tests/oracle.py::ORACLE_DATA_DELTAS`). Done when both pass and "Current state" names the
dispatch.
