# Method

VULCAN3 is a JAX port of the VULCAN 2 photochemical-kinetics solver
([Tsai et al. 2017](https://doi.org/10.3847/1538-4365/228/2/20),
[Tsai et al. 2021](https://doi.org/10.3847/1538-4357/ac29bc)). This page holds the equations and solver design that the
module docstrings point to. Measurements and reasons for each decision are in `notes.md`.

## State and chemistry

`y[nz, ni]` holds number densities (cm^-3), `M[nz]` the total third-body density (cm^-3) and `k[nr+1, nz]` the rate
constants per reaction and layer, 1-based. The rate of reaction `i` in layer `z` is

$$\mathrm{rate}_{i,z} = k_{i,z} \prod_{\mathrm{slot}} y_{r(i,\mathrm{slot}),z}^{\,s(i,\mathrm{slot})} \quad (\times M_z \text{ if three-body}).$$

The RHS comes from `make_chem_funs.build_chem_rhs`: SymPy-faithful per-reaction code with master's term order
(see "Generated RHS" in `architecture.md`). The analytical Jacobian (`chem.chem_jac_analytical`) uses the `y_r ** stoich`
form, because the leave-one-out derivative is naturally written that way and has no cancellation amplifier. It agrees
with master at machine precision. Test oracles are in `tests/_oracles.py`.

## Network files

`network.parse_network` reads the sections in file order: two-body, three-body with `k_inf` (Lindemann), three-body
without `k_inf`, special (hardcoded), condensation, radiative recombination, photo and ionisation. Radiative-recombination
networks are refused.

Forward reactions sit at odd parser indices with a reverse slot at `i+1`. Reverse rates come from NASA-9 Gibbs energies.
Thermal reverse rates fill only slots below `stop_rev_indx` and skip condensation, photo and ion rows: photo and ion reverse
slots stay zero, and a condensation reverse slot holds the evaporation rate set at run time.

Species order is first appearance in the file. A species that appears only in the config (an inert `const_mix` gas like
Ar) is not appended; `runtime_validation` refuses such a config.

## Initial abundances

`ini_mix` has five modes, in `ini_abun`:

| Mode | What it does |
|---|---|
| `EQ` | Gibbs minimisation (ExoGibbs) on the network's own NASA-9 thermochemistry |
| `const_mix` | a per-species mixing dict from the config |
| `vulcan_ini` | the composition of a previous `.vul` file |
| `table` | a per-layer mixing-ratio table |
| `const_lowT` | the five-molecule H2/H2O/CH4/He/NH3 system by Newton |

The EQ seed uses the same NASA-9 polynomials as the reverse rates, so the seed and the kinetics cannot disagree about
thermochemistry. Its tangent is zero, and a batched lane equals the solo seed bitwise. Cite
[ExoGibbs](https://github.com/HajimeKawahara/exogibbs) when a run uses it.

## Time step

`jax_step.jax_ros2_step` is the second-order two-stage Rosenbrock step of VULCAN
([Verwer et al. 1999](https://doi.org/10.1137/S1064827597326651)), with `gamma = 1 + 1/sqrt(2)`. The Jacobian models
transport as a diagonal-in-species tridiagonal without the `ysum` coupling, so it is a W-method, not an L-stable
Rosenbrock method.

## Block-tridiagonal solve

Each stage solves `[A_j; B_{j-1}, C_j] k = rhs` for `nz` layers of `ni` species. Forward elimination computes
`A'_j = A_j - C_j inv(A'_{j-1}) B_{j-1}` by one LU factorisation per block, at cost `O(nz ni^3)`.

The factorisation builds `inv(A'_{j-1})` by solving against the identity. With diagonal off-blocks the update is the
elementwise product `A_j - (c[:, None] * b[None, :]) * inv(A'_{j-1})`, which replaces a cubic matrix product with a
quadratic scaling but still reads all `ni^2` entries. The sweep stays cubic in `ni`. Both Ros2 stages share one
factorisation. The dense-off-block oracle is in `tests/_oracles.py`.

`solver_fast` differs from the plain `solver` pair in two ways:

1. `solve` is a `lax.custom_linear_solve`. Its tangent is `A dx = db - dA x` on the primal factors, so both stages and every
   tangent direction share one factorisation instead of differentiating through the pivoted LU. The primal is the same
   code on the same inputs and is bitwise equal. Under `fast` and `reference`, reverse mode transposes the primal sweep on
   the same factors; under `ffi` it is refused.
2. With `VULCAN3_SOLVER=ffi` the raw factor and solve are one C++ call each (`csrc/block_thomas_cpu.cc`), the CPU reference
   for the fused GPU kernel. `custom_linear_solve` makes these calls differentiable in forward mode.

On a CUDA device `ffi` runs `csrc/block_thomas_cuda.cu`: one factor launch and two solve launches per Ros2 step, with the CPU
kernel's factor format and its results to roundoff, not bitwise. It takes at most `CUDA_MAX_NI` species per layer; a wider
network is refused while the step is traced. The kernel's header gives the launch layouts. `python -m vulcan3.solver_fast
--cuda` builds it, on a host with nvcc, for sm_80 and later. Without the library a device call under `ffi` fails at dispatch.

## Condensation

`conden.update_conden_rates` recomputes each condensation and evaporation rate:

$$\mathrm{rate}_{r,z} = D_{g,r,z} \frac{m_r}{\rho_{p,r} r_{p,r}^2} (y_{z,\mathrm{sp}(r)} - n_{\mathrm{sat},r,z}).$$

`apply_h2o_relax_jax` and `apply_nh3_relax_jax` run an implicit-Euler cold-trap relaxation toward saturation. NH3 also limits
the condensation region to layers at or below `conden_top`.

`make_conden_spec` (host, once per config) extracts the temperature-independent metadata. `build_conden_profile` (pure JAX)
rebuilds every array that depends on temperature or structure, so a caller with a live T(P) can regenerate the `ProfileVars`
`c_*` fields and splice them into the runner carry. `_build_statics.build_conden_static` delegates to it, so host and on-graph
paths share one implementation.

## Reverse-mode sensitivities

The runner's `lax.while_loop` has no reverse mode. `steady_state_grad` solves the fixed-point adjoint

$$(I - \partial G/\partial y)^T z = \partial L/\partial y^*$$

of the solver map `G` (the hydrostatic-renormalised Ros2 step, `SOLVER_MAP`) in log-abundance coordinates. The conserved-element
null space is deflated by a QR projector, with LGMRES on the deflated system. One solve gives `dL/d(ln k_r)` for every rate
row; one more VJP gives `dL/dp` for an input pytree.

- Only `ros2_step (+ renorm) (+ photo recompute)` and optional `body_terms` are linearised. Run `audit_adjoint_scope` on the
  converged run first.
- On a photo-on column pass `make_photo_recompute_k(...)`, so the operator carries `dJ/dy`.
- `body_dt` is an adjoint-only probe step: scan `BODY_MAP_DT_CANDIDATES`.
- The gradient is a twin-ensemble mean; `info["ensemble_spread"]` is the magnitude error bar. With a large residual or spread,
  read it as a ranking.
- A reversible row's detailed-balance perturbation is the pair sum `g[fwd] + g[rev]`; `info["pair_antisym"]` near 1 is not an
  error signal.

The recipe is `examples/grad_reverse_example.py`.
