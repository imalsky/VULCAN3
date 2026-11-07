"""Vectorised JAX chemistry RHS and Jacobian.

`y[nz, ni]` holds number densities (cm^-3), `M[nz]` the third-body density (cm^-3) and `k[nr+1, nz]` the rate constants,
1-based. The RHS is `make_chem_funs.build_chem_rhs`; this module packs the network and builds the analytical Jacobian.

Public names: `NetworkArrays`, `to_jax`, `chem_jac_analytical`, `chem_jac_entries`. Equations: docs/method.md.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

if TYPE_CHECKING:
    from .network import Network


__all__ = ["NetworkArrays", "chem_jac_analytical", "chem_jac_entries", "to_jax"]


@dataclasses.dataclass(frozen=True, eq=False)
class NetworkArrays:
    """Network stoichiometry packed for JAX.

    A pytree with `ni`/`nr` as static meta fields so jit/vmap don't retrace
    per network and `num_segments` stays concrete. `jac_terms` / `jac_place`
    are the analytical Jacobian's static gather tables (`_jac_gather_tables`),
    `jac_pow` its power slots (`_jac_pow_tables`).
    """

    ni: int
    nr: int
    reactant_idx: jnp.ndarray
    product_idx: jnp.ndarray
    reactant_stoich: jnp.ndarray
    product_stoich: jnp.ndarray
    is_three_body: jnp.ndarray
    jac_terms: tuple
    jac_place: jnp.ndarray
    jac_pow: tuple


jtu.register_dataclass(
    NetworkArrays,
    data_fields=[
        "reactant_idx",
        "product_idx",
        "reactant_stoich",
        "product_stoich",
        "is_three_body",
        "jac_terms",
        "jac_place",
        "jac_pow",
    ],
    meta_fields=["ni", "nr"],
)


def _jac_gather_tables(net: Network) -> tuple[tuple, jnp.ndarray]:
    """Return static tables behind `chem_jac_entries`.

    Every nonzero J[i, j] is a sum over the (reaction r, reactant slot s)
    pairs with species j in slot s and species i among r's reactants or
    products, of `signed stoich_i * drate_dy[r, s]`. The pairs are listed
    per entry in (r, s) order and bucketed by power-of-two padded length, so
    the assembly is a gather, one fixed-order sum per entry
    (`_left_to_right_sum_axis1`) and a final gather into the dense block: no scatter-add, hence one summation
    order on every backend and in every program (a GPU scatter-add is atomic
    and reorders run to run) and no chunked transient.
    Returns `(terms, place)`: `terms` is a tuple of `(r, s, coef)` int32 /
    int32 / float64 tables of shape (entries, width) with zero-coefficient
    pads, and `place[i * ni + j]` indexes the concatenated per-entry sums,
    with the index one past the end meaning a structural zero.
    """
    r_idx = np.asarray(net.reactant_idx)
    r_st = np.asarray(net.reactant_stoich)
    out_idx = np.concatenate([r_idx, np.asarray(net.product_idx)], axis=1)
    out_st = np.concatenate([-r_st, np.asarray(net.product_stoich)], axis=1)
    ni = net.ni
    terms: dict[tuple[int, int], list[tuple[int, int, float]]] = {}
    for r in range(r_idx.shape[0]):
        for sj in range(r_idx.shape[1]):
            if r_st[r, sj] <= 0 or r_idx[r, sj] >= ni:
                continue
            for si in range(out_idx.shape[1]):
                if out_st[r, si] == 0 or out_idx[r, si] >= ni:
                    continue
                key = (int(out_idx[r, si]), int(r_idx[r, sj]))
                terms.setdefault(key, []).append((r, sj, float(out_st[r, si])))
    keys = sorted(terms, key=lambda ij: (len(terms[ij]), ij))
    buckets = []
    start = 0
    while start < len(keys):
        width = 1 << (len(terms[keys[start]]) - 1).bit_length()
        end = start
        while end < len(keys) and len(terms[keys[end]]) <= width:
            end += 1
        r_tab = np.zeros((end - start, width), dtype=np.int32)
        s_tab = np.zeros_like(r_tab)
        c_tab = np.zeros(r_tab.shape, dtype=np.float64)
        for a, key in enumerate(keys[start:end]):
            for b, (r, sj, c) in enumerate(terms[key]):
                r_tab[a, b], s_tab[a, b], c_tab[a, b] = r, sj, c
        buckets.append(tuple(jnp.asarray(t) for t in (r_tab, s_tab, c_tab)))
        start = end
    place = np.full(ni * ni, len(keys), dtype=np.int32)
    for p, (i, j) in enumerate(keys):
        place[i * ni + j] = p
    return tuple(buckets), jnp.asarray(place)


def _jac_pow_tables(net: Network) -> tuple:
    """Return the reactant slots with stoich >= 2, the only ones `chem_jac_entries` raises to a power.

    `(r, s, place)`, with `place[r, s]` the slot's row in the per-slot power results and one past the end for every
    other slot.
    """
    st = np.asarray(net.reactant_stoich)
    r, s = np.nonzero(st >= 2)
    place = np.full(st.shape, r.size, dtype=np.int32)
    place[r, s] = np.arange(r.size, dtype=np.int32)
    return tuple(jnp.asarray(t, dtype=jnp.int32) for t in (r, s, place))


def to_jax(net: Network) -> NetworkArrays:
    """Pack a Network's relevant arrays into jnp form for the chemistry RHS."""
    jac_terms, jac_place = _jac_gather_tables(net)
    return NetworkArrays(
        ni=int(net.ni),
        nr=int(net.nr),
        reactant_idx=jnp.asarray(net.reactant_idx, dtype=jnp.int64),
        product_idx=jnp.asarray(net.product_idx, dtype=jnp.int64),
        reactant_stoich=jnp.asarray(net.reactant_stoich, dtype=jnp.float64),
        product_stoich=jnp.asarray(net.product_stoich, dtype=jnp.float64),
        is_three_body=jnp.asarray(net.is_three_body, dtype=jnp.bool_),
        jac_terms=jac_terms,
        jac_place=jac_place,
        jac_pow=_jac_pow_tables(net),
    )


def _left_to_right_sum_axis1(x: jnp.ndarray) -> jnp.ndarray:
    """Sum over axis 1 (the term axis of a column table) in one fixed order.

    XLA may reorder a `reduce` per program; a fixed chain of adds keeps the batch and queue runners bit-identical.
    """
    acc = x[:, 0]
    for j in range(1, x.shape[1]):
        acc = acc + x[:, j]
    return acc


def chem_jac_entries(
    y: jnp.ndarray,  # [nz, ni]
    M: jnp.ndarray,  # [nz]
    k: jnp.ndarray,  # [nr+1, nz]
    net: NetworkArrays,
) -> jnp.ndarray:
    """Return the nonzero entries of the analytical Jacobian for a whole column.

    [n_entries + 1, nz] in `net.jac_place` row order, the last row zero (the structural-zero slot).
    `chem_jac_analytical` gathers them into the dense blocks.

    The layer axis trails every intermediate. Under the batched runner's
    lane vmap that keeps (reaction, slot) major and (lane, layer)
    contiguous, so each per-entry gather reads whole rows; a per-layer form
    (the same operations with the layer axis leading, vmapped over layers)
    has a (lane, layer, reaction, slot) layout that puts every gathered
    element in its own memory sector (~4.6 ms a step at 396 lanes on a
    GH200). Against that per-layer form on CPU: bitwise equal
    on HD189; on W39b 39 of 1.19M entries one ulp apart, all <= 7e-19 of
    their layer's peak.
    """
    nz = y.shape[0]
    yp = jnp.concatenate([y.T, jnp.ones((1, nz), dtype=y.dtype)], axis=0)  # [ni+1, nz]
    st = net.reactant_stoich[:, :, None]  # [nr+1, max_terms, 1]

    y_r = yp[net.reactant_idx]  # [nr+1, max_terms, nz]
    # Only the stoich >= 2 slots (`net.jac_pow`) take a pow, at the exponents
    # the full-table form used; an f64 pow costs ~8 ns a call on x86-64 and
    # XLA re-evaluates it in every fused consumer. A stoich-1 factor is y: on
    # XLA:CPU (x86-64 and arm64 measured) pow(y, 1) returns y for every
    # normal y, and a subnormal y meets only multiplies, which XLA:CPU
    # flushes to the same signed zero, so this is bitwise the full-table form
    # there; on a GPU that is not measured (todo.md).
    pr, ps, pplace = net.jac_pow
    y_p = y_r[pr, ps]  # [n_pow, nz]
    st_p = net.reactant_stoich[pr, ps][:, None]
    one = jnp.ones((1, nz), dtype=y.dtype)
    pow_st = jnp.concatenate([y_p**st_p, one], axis=0)[pplace]  # [nr+1, max_terms, nz]
    pow_st_m1 = jnp.concatenate([y_p ** (st_p - 1), one], axis=0)[pplace]
    factor_l = jnp.where(st >= 2, pow_st, jnp.where(st > 0, y_r, 1.0))

    # Leave-one-out product Π_{l != j} factor_l: the remaining factors
    # multiplied left to right in slot order (a fixed order, as for the sums).
    max_terms = net.reactant_idx.shape[1]
    leave_out_cols = []
    for j in range(max_terms):
        acc = None
        for slot in range(max_terms):
            if slot != j:
                acc = factor_l[:, slot] if acc is None else acc * factor_l[:, slot]
        leave_out_cols.append(jnp.ones_like(factor_l[:, 0]) if acc is None else acc)
    leave_out = jnp.stack(leave_out_cols, axis=1)

    # AD-safe power rule: `y_r ** 0` has a NaN jvp at y_r == 0 (clipped
    # cells), so stoich==1 gives a constant 1 and stoich>=2 an exponent >= 1.
    # Primal is bit-identical to y_r**(stoich-1); masks fold at trace time.
    pow_minus_one = jnp.where(
        st >= 2,
        pow_st_m1,
        jnp.where(st == 1, 1.0, 0.0),
    )
    drate_dy = st * pow_minus_one * leave_out * k[:, None, :]
    drate_dy = jnp.where(net.is_three_body[:, None, None], drate_dy * M, drate_dy)

    entry_sums = [_left_to_right_sum_axis1(coef[:, :, None] * drate_dy[r, s]) for r, s, coef in net.jac_terms]
    zero = jnp.zeros((1, nz), dtype=drate_dy.dtype)
    return jnp.concatenate([*entry_sums, zero], axis=0)


def chem_jac_analytical(
    y: jnp.ndarray,  # [nz, ni]
    M: jnp.ndarray,  # [nz]
    k: jnp.ndarray,  # [nr+1, nz]
    net: NetworkArrays,
) -> jnp.ndarray:
    """Stoichiometry-driven chemistry Jacobian for a whole column.

    Returns [nz, ni, ni]: J[z, i, j] = Σ_r sign_i * stoich_i * (∂rate[r, z]/∂y[z, j]), built from the network tables
    without `jacrev`'s ni reverse-mode passes (`chem_jac_entries` placed densely).
    """
    nz, ni = y.shape[0], net.ni
    dense = chem_jac_entries(y, M, k, net)[net.jac_place]  # [ni*ni, nz]
    return jnp.moveaxis(dense.reshape(ni, ni, nz), -1, 0)
