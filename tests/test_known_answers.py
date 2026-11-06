"""Known answers: first-order decay through the Ros2 step, and the order of accuracy in time and in space."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from vulcan3 import chem_funs, jax_step

_K_PHOTO = 1e-3  # s^-1, a first-order loss rate with a timescale of 1000 s
_T_END = 3.0e3  # s, three e-folds
_MIXING_AMPLITUDE = 1e-4
_STRETCH = 15.0  # thickest to thinnest layer: the ratio of the production grid


def _column(nz: int, ni: int, dzi, kzz: float) -> jax_step.AtmStatic:
    """Build a static column with eddy diffusion only: no molecular diffusion, advection or boundary flux."""
    return jax_step.AtmStatic(
        Kzz=jnp.full(nz - 1, kzz),
        Dzz=jnp.zeros((nz - 1, ni)),
        dzi=jnp.asarray(dzi),
        vz=jnp.zeros(nz - 1),
        Hpi=jnp.ones(nz - 1),
        Ti=jnp.ones(nz - 1),
        Tco=jnp.ones(nz),
        g=jnp.ones(nz),
        ms=jnp.ones(ni),
        alpha=jnp.zeros(ni),
        M=jnp.full(nz, 1.0e18),
        vm=jnp.zeros((nz - 1, ni)),
        vs=jnp.zeros((nz - 1, ni)),
        top_flux=jnp.zeros(ni),
        bot_flux=jnp.zeros(ni),
        bot_vdep=jnp.zeros(ni),
        gas_indx_mask=jnp.ones(ni, dtype=bool),
        diff_esc_mask=jnp.zeros(ni, dtype=bool),
        use_vm_mol=False,
        use_settling=False,
        use_topflux=False,
        use_botflux=False,
    )


def test_first_order_decay_converges_to_the_analytic_answer_at_second_order():
    """A -> products at rate k: y(t) = y0 exp(-k t), to the W-method's second order.

    The Jacobian is exact for this linear problem, so the Ros2 pair reaches its full order. Measured 7.9e-3 at 64 steps
    and an order of 1.83 from 32 to 64, 1.91 from 64 to 128 (x86-64, jax 0.6.2); the bars leave 5 %.
    """
    network, net = chem_funs.NETWORK, chem_funs.NET_JAX
    nz, ni = 4, network.ni
    row = int(np.nonzero(network.is_photo)[0][0])
    parent = int(network.reactant_idx[row, 0])
    k_arr = np.zeros((network.nr + 1, nz))
    k_arr[row] = _K_PHOTO
    y0 = jnp.full((nz, ni), 1.0e10)
    atm = _column(nz, ni, np.full(nz - 1, 1.0e5), 0.0)

    def step(y, dt):
        return jax_step.jax_ros2_step(y, jnp.asarray(k_arr), dt, atm, net, matrix_free=False)[0]

    @jax.jit
    def run(n, dt):
        return jax.lax.fori_loop(0, n, lambda _, y: step(y, dt), y0)

    exact = float(y0[0, parent]) * np.exp(-_K_PHOTO * _T_END)
    errors = {}
    for n in (32, 64, 128):
        y = np.asarray(run(jnp.int32(n), jnp.float64(_T_END / n)))
        errors[n] = np.max(np.abs(y[:, parent] - exact)) / exact
    assert errors[64] < 8.3e-3, errors
    assert np.log2(errors[32] / errors[64]) > 1.74, errors
    assert np.log2(errors[64] / errors[128]) > 1.81, errors


def test_eddy_transport_is_second_order_on_a_stretched_grid():
    """K d2y/dz2 of a smooth profile, on a grid stretched as the production ones are.

    With a uniform total density, the eddy kernel is the three-point second difference on the smooth grid. Measured
    order 1.99 from 40 to 80 layers and 2.00 from 80 to 160 (x86-64, jax 0.6.2), as notes "Do not retry" records for the
    shipped kernels; the bar leaves 5 %.
    """
    kzz, length, ni = 1.0e8, 1.0e8, 2
    errors = []
    for nz in (40, 80, 160):
        s = (np.arange(nz) + 0.5) / nz
        z = length * (_STRETCH**s - 1.0) / (_STRETCH - 1.0)
        atm = _column(nz, ni, np.diff(z), kzz)
        wave = 2.0 * np.pi / length
        mix = _MIXING_AMPLITUDE * np.sin(wave * z)
        y = np.stack([1.0 - mix, mix], axis=1) * np.asarray(atm.M)[:, None]
        grav = jax_step.compute_diff_grav(atm)
        blocks = jax_step._build_diff_coeffs_jax(jnp.asarray(y), atm, grav)
        diff = np.asarray(jax_step._apply_diffusion_jax(jnp.asarray(y), *blocks[:6], atm))
        exact = -kzz * wave**2 * y[:, 1]
        errors.append(np.max(np.abs(diff[1:-1, 1] - exact[1:-1])) / np.max(np.abs(exact)))
    orders = np.log2(np.asarray(errors[:-1]) / np.asarray(errors[1:]))
    assert orders[-1] > 1.9, (errors, orders)
