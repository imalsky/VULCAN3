"""Validate one Ros2 step against VULCAN's Ros2.solver.

Compares sol after a single 2nd-order Rosenbrock step.
"""

from __future__ import annotations

import numpy as np
from _helpers import relerr
from oracle import assert_below, enter_master, leave_master, run_oracle_subprocess, upstream_preloop


def main() -> None:
    # === VULCAN-master pipeline + Ros2 step ===
    enter_master()
    import build_atm as ba_v
    import op as op_v
    import store as st_v

    from vulcan3.config import default_config

    data_var, data_atm = upstream_preloop(
        default_config(),
        st_v.Variables(),
        st_v.AtmData(),
        ba_v.Atm(),
        op_v.ReadRate(),
        ba_v.InitialAbun(),
        op_v.Output(),
    )
    data_var.dt = 1e-10
    data_var.ymix = data_var.y / np.vstack(np.sum(data_var.y, axis=1))

    data_para = st_v.Parameters()
    solver_v = op_v.Ros2()
    solver_v.naming_solver(data_para)

    # Snapshot y BEFORE the step (since solver mutates var.y)
    y0 = np.asarray(data_var.y, dtype=np.float64).copy()
    k_dict = {i: np.asarray(v, dtype=np.float64).copy() for i, v in data_var.k.items()}

    # Reference one Ros2 step (mutates var.y)
    var_after, _ = solver_v.solver(data_var, data_atm, data_para)
    sol_ref = np.asarray(var_after.y, dtype=np.float64).copy()

    # === Switch to VULCAN-JAX ===
    leave_master()

    import jax.numpy as jnp

    from vulcan3 import chem_funs
    from vulcan3.jax_step import jax_ros2_step, make_atm_static

    nz, ni = y0.shape

    # Pack k into JAX array
    k_arr = np.zeros((chem_funs.nr + 1, nz), dtype=np.float64)
    for i, v in k_dict.items():
        k_arr[i] = v

    # Production one-step kernel, the same hot path the outer loop calls.
    atm_static = make_atm_static(data_atm, ni, nz)
    sol_jax, _ = jax_ros2_step(
        jnp.asarray(y0),
        jnp.asarray(k_arr),
        jnp.float64(data_var.dt),
        atm_static,
        chem_funs.NET_JAX,
    )

    # generous; the integrator self-corrects
    assert_below({"sol_jax vs sol_ref": (relerr(sol_jax, sol_ref, floor=1e-12), 1e-3)})


def test_main():
    run_oracle_subprocess(__file__, "vulcan2_ncho", "cfg_examples/vulcan_cfg_HD189.py")


if __name__ == "__main__":
    main()
