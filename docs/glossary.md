# Glossary

Short names follow VULCAN 2 (`op.py`, `build_atm.py`, `vulcan_cfg.py` at the pinned revision) and the published equations.
Master's `plot_py/` reads these names from a `.vul` file, and `pyproject.toml` lists the families for the naming lint rule.

| Name | Meaning | Unit |
|---|---|---|
| `Tco`, `T`, `Ti` | temperature at layer centres, a temperature, at interfaces | K |
| `pco`, `pico`, `P_b`, `P_t` | pressure at centres, at interfaces, at the bottom and top | dyn cm^-2 |
| `Kzz`, `const_Kzz`, `K_deep`, `K_max` | eddy diffusion coefficient | cm^2 s^-1 |
| `Dzz` | molecular diffusion coefficient | cm^2 s^-1 |
| `Dg` | gas diffusivity in the condensation rate | cm^2 s^-1 |
| `Hp`, `Hpi`, `Hi` | pressure scale height at centres, at interfaces, of a species | cm |
| `dz`, `dzi`, `zco` | layer thickness, interface spacing, height of interfaces | cm |
| `Rp`, `Mp` | planet radius, planet mass | cm, g |
| `M`, `n_0` | total number density, as third body and as column | cm^-3 |
| `nz`, `ni`, `nr` | number of layers; number of species; number of reaction slots, forward and reverse counted apart | |
| `y` | species number densities, shape `[nz, ni]` | cm^-3 |
| `k` | rate constants, shape `[nr+1, nz]`, 1-based (row 0 unused) | |
| `J`, `J_br`, `cross_J`, `Jion` | photolysis rate of a branch, its cross section, photoionisation | s^-1, cm^2 |
| `A`, `B`, `C`, `A_eddy`, `B_mol` | diagonal, upper and lower block-tridiagonal coefficients, by process | s^-1 |
| `G`, `K_eq` | Gibbs energy; equilibrium constant of a reversible row | |
| `G` in `steady_state_grad` | the solver map whose fixed point is the converged state | |
| `L`, `dL_dlnk`, `dL_dp` | a scalar loss of the state, its sensitivities | |
| `E`, `E_inf` | activation energy of a rate | K |
| `F`, `Fc` | Troe broadening factor and its centre | |
| `Q`, `R`, `S`, `V`, `Y` | local terms named as in the cited source or equation beside them | |
| `C_H`, `N_H`, `O_H`, `He_H` | element ratio to hydrogen | |
| `eps_L`, `ka_CIA`, `T_irr`, `T_int` | T-P parameters, as in `analytical_TP_H14` | |
| `Navo`, `kb`, `G_grav` | Avogadro's number, Boltzmann's constant, the gravitational constant | CGS |
| `PAD` | the species index `ni`, padding a short term list in the RHS and Jacobian tables | |
| `rtol` | relative tolerance of the step control: a trial is accepted when its error ratio `delta` is at most `rtol` as built, and the next step size scales with `(rtol/delta)^0.5` | |
| `longdy` | largest relative change in the state against an earlier saved state: the maximum over cells of `abs(y - y_old) / n_0`, divided by `ymix`, with cells below `atol` or `mtol_conv` and ignored species counted as 0 | |
| master | the pinned VULCAN 2 code that this package ports and is tested against | |
| lane | one profile (column) of a batched run: one slice of the batch axis of the carry in `run_batch` and `run_queue` | |
| carry | the state of the `lax.while_loop` that `OuterLoop` runs (`JaxIntegState`): `y`, step size, time, counters, termination reason | |
| certificate | the test that a run has reached steady state (`conv_normal`): the tight or loose `longdy` branch, the photo-flux gate, the geometry and element-budget terms, and not frozen at `dt_min` | |

Master-facing classes and methods keep their names: `Atm`, `Output`, `ReadRate`, `InitialAbun`, `Ros2JAX`, `naming_solver`,
`load_TPK`, `f_mu_dz`, and the containers `var`, `atm`, `para`.

## Carry fields

`ProfileVars` prefixes: `r_` rates and structure for the refresh, `c_` condensation, `p_` photolysis. `JaxIntegState` fields
keep the name of the runtime variable they carry.
