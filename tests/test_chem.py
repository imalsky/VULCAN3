"""Validate chem_rhs and chem_jac against VULCAN-master's chemdf and symjac.

We compare on a real (y, M, k) state captured from VULCAN-master's pipeline.
"""

from __future__ import annotations

import numpy as np
from _helpers import relerr
from oracle import assert_below, enter_master, leave_master, run_oracle_subprocess, upstream_preloop


# size-limit-ok: one oracle script, run in a subprocess on a copy of master
def main() -> None:
    # === 1. Run VULCAN-master pipeline to get reference y/M/k state and chemdf/symjac ===
    enter_master()
    import build_atm as ba_v
    import chem_funs as cf_v
    import op as op_v
    import store as st_v
    import vulcan_cfg as cfg_v

    data_var, data_atm = upstream_preloop(
        cfg_v, st_v.Variables(), st_v.AtmData(), ba_v.Atm(), op_v.ReadRate(), ba_v.InitialAbun()
    )

    M = np.asarray(data_atm.M, dtype=np.float64).copy()
    y = np.asarray(data_var.y, dtype=np.float64).copy()
    k_dict = {i: np.asarray(v, dtype=np.float64).copy() for i, v in data_var.k.items()}
    nz, ni = y.shape

    # Get reference chemdf and symjac on this state
    dydt_ref = np.asarray(cf_v.chemdf(y, M, k_dict)).copy()
    J_ref = np.asarray(cf_v.symjac(y, M, k_dict)).copy()

    # === 2. Switch over to VULCAN-JAX modules ===
    leave_master()

    import jax.numpy as jnp

    import vulcan3.chem as chem_mod
    import vulcan3.network as net_mod
    from vulcan3._paths import resolve_data_path

    net = net_mod.parse_network(resolve_data_path(cfg_v.network))
    net_jax = chem_mod.to_jax(net)

    # Use the same k_dict from VULCAN's pipeline (already validated against ours).
    # Pack into [nr+1, nz] array.
    k_full = np.zeros((net.nr + 1, nz), dtype=np.float64)
    for i, vec in k_dict.items():
        k_full[i] = vec

    # === 3. Compute JAX chem_rhs ===
    y_j = jnp.asarray(y)
    M_j = jnp.asarray(M)
    k_j = jnp.asarray(k_full)
    # Use the codegen path that the integrator actually runs. We build it
    # against `net` (parsed from cfg_v.network) so the test isn't sensitive
    # to JAX's default vulcan_cfg.network.
    import vulcan3.make_chem_funs as mcf

    chem_rhs_codegen = mcf.build_chem_rhs(net)
    dydt_jax = np.asarray(chem_rhs_codegen(y_j, M_j, k_j))

    # Skip cells below 1e-6 of the species's peak |dydt|: cancellation
    # residues near zero, where XLA's FMA fusion differs from NumPy but both
    # are within ULP of zero at the species's natural scale.
    per_species_peak = np.abs(dydt_ref).max(axis=0)
    cell_floor = 1e-6 * np.maximum(per_species_peak[None, :], 1e-30)
    significant = np.abs(dydt_ref) > cell_floor

    # === 4. Compute JAX chem_jac ===
    from _oracles import chem_jac

    Jblk_jax = np.asarray(chem_jac(y_j, M_j, k_j, net_jax))
    # symjac layout: per-layer (ni, ni) blocks live at J_ref[j*ni:(j+1)*ni, j*ni:(j+1)*ni]
    Jblk_ref = np.stack([J_ref[j * ni : (j + 1) * ni, j * ni : (j + 1) * ni] for j in range(nz)])

    # Bulk species: bound every cell, including those below the significance
    # floor, against the species' peak |dydt|.
    bulk_relerr = 0.0
    for sp in ("H2O", "CO2", "SO", "SO2", "H2", "CO", "S", "H2S"):
        if sp in net.species_idx:
            j = net.species_idx[sp]
            r = float(np.abs(dydt_jax[:, j] - dydt_ref[:, j]).max() / max(np.abs(dydt_ref[:, j]).max(), 1e-30))
            bulk_relerr = max(bulk_relerr, r)

    assert_below(
        {
            "chem_rhs (cells > 1e-6 of species peak)": (relerr(dydt_jax, dydt_ref, mask=significant), 1e-5),
            "chem_rhs bulk species": (bulk_relerr, 1e-5),
            "chem_jac": (relerr(Jblk_jax, Jblk_ref), 1e-6),
        }
    )


def test_main():
    run_oracle_subprocess(__file__, "vulcan2_ncho", "cfg_examples/vulcan_cfg_HD189.py")


if __name__ == "__main__":
    main()
