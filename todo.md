# todo

Evidence for these items is in `notes.md` (this repo's, unless another repo is named).

1. [stale] The paper's GPU throughput figure and text come from VULCAN-JAX 0.6.0: rerun the GH200 batch sweep on the current release.
2. [owed] Re-baseline after the C25-C29 data and step-control fixes and the C30 extension to dissociations: the arm64 full tier and the oracle comparisons (`oracle.yml`).
3. [owed] Rebuild the CUDA kernel on the NAS and check `ffi` on a GH200: the code cleanup changed the kernel sources, and its proof of `ffi` ran on CPU only.

## 1. GH200 throughput rerun for the paper

Evidence: jax_paper/notes.md "Current state" (owed list). The paper's GPU
throughput figure and its Parallelism-section text (0.010 -> 0.76 profiles/s,
~75x, batch 256, "tens of GB") are hard-coded in
`jax_paper/scripts/fig_gpu_throughput.py` and `jax_paper/data/gpu_throughput.json`
from NAS job 61453 on 0.6.0. That job's log is not on disk; the json gives 76.3x,
and no peak memory or per-lane convergence is recorded. Needs the maintainer's NAS
login. Run the same sweep on the current release on one GH200: HD 189733 b-like,
nz 150, `ini_mix` EQ, `use_photo` False, `count_max` 2500, batch sizes 1, 2, 4, ...,
256, each batch converged in vmap'd compiled calls; record the total wall per batch
(compile included), each lane's end state and the peak device memory. The 0.6.0 wall
times sum to about 30 min of GPU time. Done when the figure, the json and the paper
text carry the new run's numbers and jax_paper/notes.md names the run.

## 2. Re-baseline after C25-C29

Evidence: notes "Parity ledger" C25-C32 and "Current state". The step-count gates read the
x86-64 counts on arm64 too, and the tracked arm64 manifest is rebuilt. Left: on the laptop
`VULCAN3_RUN_SLOW=1 python -m pytest tests -n 12 -q` (the whole full tier; only the gates and
`test_solver_fast.py` were run), and a green `oracle.yml` dispatch at this code
(`VULCAN_MASTER_DIR` at 80f75b9: the oracle copy gets the corrected files through
`tests/oracle.py::ORACLE_DATA_DELTAS`). Done when both pass and "Current state" names the
dispatch.

## 3. NAS rebuild and `ffi` check after the cleanup

Evidence: notes "Cleanup proofs". The kernel sources were reformatted, so a library built
before the cleanup is refused until it is rebuilt. Lowered `ffi` programs moved when reverse
mode's transpose was removed; the optimized CPU program stayed equal apart from names. The GPU
was not run. Needs the maintainer's NAS login. Steps, after the commits are pushed: `git pull`,
`python -m vulcan3.solver_fast --cuda`, then the kernel test job on a GH200. Done when
the rebuilt library loads and the job passes.
