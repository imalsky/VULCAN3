# todo

Evidence for these items is in `notes.md` (this repo's, unless another repo is named).

1. [owed] Re-baseline after the C25-C29 data and step-control fixes and the C30 extension to dissociations: the arm64 full tier.

## 1. Re-baseline after C25-C29

Evidence: notes "Parity ledger" C25-C32 and "Current state". The step-count gates read the
x86-64 counts on arm64 too, the tracked arm64 manifest is rebuilt, and `oracle.yml` is green at
this code. Left: on the laptop `VULCAN3_RUN_SLOW=1 python -m pytest tests -n 12 -q` (the whole
full tier; only the gates and `test_solver_fast.py` were run). Done when it passes.
