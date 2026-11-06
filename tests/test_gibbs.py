"""Validate forward rates and the NASA-9 reverse-rate path against VULCAN-master's read_rate and rev_rate.

Runs master's setup until `data_var.k` holds the forward rates, compares
`rates_jax.compute_forward_k` on the same T/M atmosphere, then lets master
add the reverse rates and compares `rates_jax.build_rate_array` reverse slot
for reverse slot. This is the only check that the Gibbs energies, the
(kB T/P0)^dn factor and the forward/reverse pairing are right; a
self-comparison against another copy of the same polynomial would prove
nothing.
"""

from __future__ import annotations

import numpy as np
from _helpers import relerr
from oracle import assert_below, enter_master, leave_master, run_oracle_subprocess, upstream_preloop


def main() -> None:
    enter_master()
    import jax.numpy as jnp

    from vulcan3.config import default_config

    vulcan_cfg = default_config()
    import store, op  # noqa: E401, I001 (master's modules, in the order it imports them)
    from vulcan3.atm_setup import Atm

    import vulcan3.network as net_mod
    from vulcan3 import rates_jax
    import vulcan3.gibbs as gibbs_mod

    # === 1. VULCAN setup: build atm + read forward rates, then reverse rates ===
    use_lowT = bool(getattr(vulcan_cfg, "use_lowT_limit_rates", False))
    rate = op.ReadRate()
    data_var, data_atm = upstream_preloop(vulcan_cfg, store.Variables(), store.AtmData(), Atm(), rate)
    k_forward = {i: np.array(v, dtype=np.float64) for i, v in data_var.k.items()}
    if use_lowT:
        data_var = rate.lim_lowT_rates(data_var, data_atm)
    data_var = rate.rev_rate(data_var, data_atm)
    leave_master()

    T = np.asarray(data_atm.Tco, dtype=np.float64)
    M = np.asarray(data_atm.M, dtype=np.float64)

    # === 2. VULCAN-JAX computation on the same atmosphere ===
    net = net_mod.parse_network(vulcan_cfg.network)
    k_fwd = np.asarray(rates_jax.compute_forward_k(net, T, M))
    coeffs, present = gibbs_mod.load_nasa9(net.species, "thermo")
    k_full = np.asarray(
        rates_jax.build_rate_array(
            net,
            jnp.asarray(T),
            jnp.asarray(M),
            coeffs,
            remove_list=vulcan_cfg.remove_list,
            use_lowT_caps=use_lowT,
        )
    )
    missing = [sp for sp, p in zip(net.species, present, strict=False) if not p]
    if missing:
        print(f"  Missing: {missing}")

    # === 3. Forward k against read_rate's. Only forward, non-photo/conden/
    # ion/radiative reactions are populated by compute_forward_k; read_rate
    # leaves the photo, condensation and radiative slots at zero, and the
    # port must too.
    fwd = [i for i in range(1, net.nr + 1, 2) if i in k_forward]
    live = [i for i in fwd if k_forward[i].max() > 0]
    dead = [i for i in fwd if not k_forward[i].max() > 0]
    print(f"forward k compared: {len(live)} nonzero, {len(dead)} zero rows")

    # === 4. Reverse k against rev_rate's ===
    rev = [i for i in range(2, net.stop_rev_indx, 2) if i in data_var.k]
    print(f"reverse k compared: {len(rev)}")

    # === 5. Beyond stop_rev_indx all reverses are zero ===
    bad_zero = sum(k_full[i].max() != 0.0 for i in range(net.stop_rev_indx + 1, net.nr + 1, 2))

    assert_below(
        {
            # Both sides can be very small or zero: relative error with a floor.
            "forward k (nonzero rows)": (
                relerr(k_fwd[live], np.stack([k_forward[i] for i in live]), floor=1e-300),
                1e-10,
            ),
            "forward k (zero rows) max |k|": (float(np.max(np.abs(k_fwd[dead]), initial=0.0)), 1e-300),
            "reverse k": (relerr(k_full[rev], np.stack([data_var.k[i] for i in rev]), floor=1e-300), 1e-8),
            "non-zero reverses beyond stop_rev_indx": (bad_zero, 1),
        }
    )


def test_main():
    run_oracle_subprocess(__file__, "vulcan2_ncho", "cfg_examples/vulcan_cfg_HD189.py")


if __name__ == "__main__":
    main()
