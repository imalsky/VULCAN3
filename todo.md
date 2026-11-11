# todo

Evidence for these items is in `notes.md` (this repo's, unless another repo is named).

1. [owed] Re-baseline after the C25-C29 data and step-control fixes and the C30 extension to dissociations: the arm64 full tier.
2. [unverified] `oracle.md` asks for a converged step count within 10 % of VULCAN 2's: measured on HD189 only, and no test holds it.

## 1. Re-baseline after C25-C29

Evidence: notes "Parity ledger" C25-C32 and "Current state". The step-count gates read the
x86-64 counts on arm64 too, the tracked arm64 manifest is rebuilt, and `oracle.yml` is green at
this code. Left: on the laptop `VULCAN3_RUN_SLOW=1 python -m pytest tests -n 12 -q` (the whole
full tier; only the gates and `test_solver_fast.py` were run). Done when it passes.

## 2. Step counts against VULCAN 2

Evidence: jax_paper notes "Table 1" and this repo's notes "Evidence behind the numerical rules". On HD189, free
convergence, the pinned VULCAN 2 takes 1580, 1590 and 1602 steps in three repeats (1603 to 1611 with the declared
Troe delta) and vulcan3 1495 in every repeat: 5 to 7 % fewer; with the upwind-then-central scheme 2102 against
vm_branch's 2201, 4.5 % fewer. The 10 % in `oracle.md` "Step counts" is that spread with margin. HD209 and W39b have
no free-convergence VULCAN 2 count on record (the paper's rows for them are step-matched), `tools/bench_table1.sh`
runs HD189 only, and the gate test pins vulcan3's own counts. Done when HD209 and W39b are measured against the
pinned clone and the three counts are compared in a full-tier oracle test, or the maintainer changes the criterion.
