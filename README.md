# VULCAN3

VULCAN3 is a JAX port of
[VULCAN](https://github.com/exoclime/VULCAN). It solves one-dimensional
chemical kinetics in planetary atmospheres. The goal for this project is to make a
(close to) fully auto-differentiable version of VULCAN, which will be VULCAN 3.0. A paper on
the code is currently in progress.

The package supports:

- thermochemistry and photochemistry;
- eddy and molecular diffusion;
- condensation and ion chemistry;
- CPU and GPU execution;
- batched models with `jax.vmap`;
- forward derivatives through the time integration; and
- steady-state reaction sensitivities.

VULCAN3 is designed to take the same API as VULCAN (with one change to use YAML files). It writes the standard
`.vul` output format. The test suite compares the port with pinned VULCAN
inputs and reference results.

## Install

VULCAN3 needs Python 3.10 or later.

```bash
python -m pip install \
  -i https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  vulcan3
```

For development:

```bash
git clone https://github.com/imalsky/VULCAN3.git
cd VULCAN3
python -m pip install -e ".[dev,plot]"
python tests/gen_fixtures.py --all   # the five .npz oracles are gitignored (~4.5 min)
python -m pytest tests -q
```

## Run a model

```bash
vulcan3 --config default
vulcan3 --config W39b
```

Results go to `output/`. Configuration files use YAML and are stored in
`src/vulcan3/configs/`. A file in `./configs/` with the same name overrides
the packaged file.

The reaction network is fixed when Python first imports the package. Set
`VULCAN3_NETWORK` before the first import if you need another network.
See [`examples/quickstart.ipynb`](examples/quickstart.ipynb) for the Python
interface.

On a GPU, `VULCAN3_SOLVER=ffi` runs the block-tridiagonal solve of each
Rosenbrock step as CUDA kernels. Build them once per host with
`python -m vulcan3.solver_fast --cuda` (needs `nvcc`). For networks of up
to 96 species the factorization keeps each block in registers (Gauss-Jordan);
a wider network is refused while the step is traced, so use `VULCAN3_SOLVER=fast`
there. On a GH200 with the WASP-39 b
network (89 species, 396 columns) the factorization and the two solves of a
step take 24.5 ms instead of 39.0 ms, and a whole step 60.7 ms instead of
75.2 ms.
The kernel needs sm_80 or newer: the Gauss-Jordan pivot search uses warp REDUX operations,
its multipliers are corrected products with the IEEE reciprocal, and the
solve stages each layer's factors with `cp.async`; the results are bitwise
the same. On a GH200 at 792 columns the factorization then takes 23.1 ms
instead of 29.1 ms and a solve 4.6 ms instead of 5.3 ms (6.4 and 1.6 ms
instead of 8.4 and 2.6 ms at 132 columns), and a 2376-job pass-1 batch of
the WASP-39 b retrieval (792 lanes, compacted to 264 and 132) runs in 301 s
instead of 344 s.

`VULCAN3_CHEM_RHS=grouped` (also read at first import) evaluates the
chemistry right-hand side as one vectorised gather per reaction signature
instead of one expression per reaction. The operation trees are the same, so
on a GH200 the batched WASP-39 b runner gives bitwise the same columns; it
cuts the transposes XLA inserts under the lane vmap. Other compiled programs
can differ in the last bits, from XLA's fused multiply-add choices; on cells
that nearly cancel that is a large relative difference, and two CPU
consistency tests with 1e-12 bars (`test_vmap_kernels` for the RHS,
`test_vmap_step` for `delta`) fail with it set. It is therefore opt-in.

`queue_compact` (config; `[]`, off, by default) shortens the tail of a
`run_queue` batch. Once no job is waiting and the live lanes fit the next
width in the list, they move to a narrower compiled program and continue from
the same tick, so the few slow columns no longer hold the full width. Each
column's arithmetic, cadence and certificate are unchanged. With
`[264, 132]` on 792 lanes, a 2376-column WASP-39 b batch on a GH200 gives
bitwise the same columns and diagnostics, and its first pass takes 344 s
instead of 520 s. Below about 64 lanes, XLA's per-lane arithmetic can change
in the last bits.

## Example

Run the default HD 189733 b model from Python and plot the converged
abundances. The run takes about a minute on a laptop CPU.

```python
import numpy as np
import matplotlib.pyplot as plt
import vulcan3
from vulcan3 import chem_funs, legacy_io, op_jax, outer_loop

# Build the initial state and integrate to convergence. Pass the same
# cfg to both the runner and the writer.
cfg = vulcan3.make_config()
rs = vulcan3.RunState.with_pre_loop_setup(cfg)
integ = outer_loop.OuterLoop(op_jax.Ros2JAX(), legacy_io.Output(cfg=cfg), cfg=cfg)
rs = integ(rs)

# Plot volume mixing ratios against pressure.
p_bar = np.asarray(rs.atm.pco) / 1e6
ymix = np.asarray(rs.step.ymix)
species = {"H2O": "H$_2$O", "CO": "CO", "CO2": "CO$_2$", "CH4": "CH$_4$",
           "NH3": "NH$_3$", "HCN": "HCN", "C2H2": "C$_2$H$_2$", "N2": "N$_2$"}
for sp, label in species.items():
    plt.loglog(ymix[:, chem_funs.spec_list.index(sp)], p_bar, label=label,
               lw=2.3, alpha=0.9)
plt.xlim(1e-14, 1e-2)
plt.gca().invert_yaxis()
plt.xlabel("volume mixing ratio")
plt.ylabel("pressure (bar)")
plt.legend(loc="lower left", ncol=2, fontsize="small")
plt.show()
```

![HD 189733 b converged abundances](assets/hd189_quickstart.png)

The scripts in [`examples/`](examples/) apply forward- and reverse-mode
derivatives to the same model.

## Limits

- Only the Rosenbrock-2 solver is ported.
- Reverse-mode differentiation does not pass through the time-integration
  loop. Use forward mode or the steady-state adjoint.
- The equilibrium seed (ExoGibbs) is not differentiable.
- Condensation can run in the forward model, but it is not validated for
  gradient inference.
- VULCAN3 includes documented corrections and deliberate differences from
  VULCAN. Check these before a strict parity study.

## Papers and citation

If you publish results from VULCAN3, cite the original VULCAN papers:

- [Tsai et al. (2017), ApJS 228, 20](https://doi.org/10.3847/1538-4365/228/2/20)
- [Tsai et al. (2021), ApJ 923, 264](https://doi.org/10.3847/1538-4357/ac29bc)

If the run uses the equilibrium seed (`ini_mix: EQ`), also cite ExoGibbs:
Kawahara, H. et al., <https://github.com/HajimeKawahara/exogibbs> (GPL-3.0).
Record the VULCAN3 version, configuration, network, and input-data versions
with the result.

## Support and license

Open a [GitHub issue](https://github.com/imalsky/VULCAN3/issues) and include
the command, configuration, package versions, and full error message.

VULCAN3 is released under GPLv3.
