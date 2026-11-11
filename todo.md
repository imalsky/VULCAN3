# todo

Evidence for these items is in `notes.md` (this repo's, unless another repo is named).

1. [owed] Re-baseline after the C25-C29 data and step-control fixes and the C30 extension to dissociations: the arm64 full tier.
2. [decision] The test of `oracle.md`'s 10 % step-count rule against VULCAN 2 does not run in `oracle.yml`.
3. [science] The photolysis window's 2 nm lower edge cuts data on HD209 and W39b; the 700 nm upper edge binds on no
   shipped case.
4. [decision] The 1 bar level where `Rp` and gravity apply is a fixed constant; a config key needs a coordinated
   change in jwst-transit-authority first.

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

## 3. The photolysis window's lower edge

Evidence: notes "Photolysis decisions". With the window opened to 0.2-1000 nm the photolysis grid of HD189 does not
change (2.00-654 nm, 2588 bins), HD209 extends down to 0.5 nm and W39b to 0.2 nm (2606 bins): the 2 nm lower edge
(`photo_lambda_min`, VULCAN 2's value) cuts stellar and cross-section data on those two, and the 700 nm upper edge
binds on none, because the default networks' highest threshold is 654 nm. The effect of the lower edge on a column is
not measured. The maintainer decides whether `photo_lambda_min` moves (a change to the HD209 and W39b results, with
their step-count gates re-pinned). Done when the edge is kept with a measured reason, or moved and re-baselined.

## 4. The `Rp` anchor pressure

`_transport_setup._RP_REF_PRESSURE` (1 bar, VULCAN 2 `build_atm.py:532-535`) sets where `Rp` and `G Mp / Rp^2`
apply. jwst-transit-authority restates the rule (`chem_build.chem_radius_anchor_bar`) and its contract test calls
`compute_mu_dz_g` with a hand-built namespace, so a key read there breaks that repo until it changes too. Done when
the maintainer asks for the key (jwst change first) or closes this.
