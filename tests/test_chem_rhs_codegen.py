"""Validate the SymPy-faithful chem_rhs codegen against `chem_rhs_numpy`
(master-faithful term order, NumPy float64): rtol=1e-5 on significant cells.
The threshold absorbs XLA FMA fusion vs NumPy `*`-chains; actual worst-cell
agreement is ~2e-13. The comparison with master's `chemdf` is test_chem.py.
The grouped RHS (`VULCAN3_CHEM_RHS=grouped`) must equal the codegen bitwise
(an expected failure on Apple Silicon, where the two agree to roundoff only).

The (y, M, k) state is captured fresh from `RunState.with_pre_loop_setup`.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent

from _helpers import atom_count_matrix

# Roundoff allowance for an atom residual, as a fraction of the summed
# magnitude of the terms it cancels: a few thousand ulps over hundreds of
# terms.
_ATOM_RESIDUAL_EPS = 1.0e-12
CODEGEN_RTOL = 1e-5  # codegen vs NumPy RHS: absorbs XLA FMA fusion
PEAK_FLOOR_FRAC = 1e-12  # per-species denominator floor, a fraction of the species' peak


def _toy_network_for_codegen(tmp_path: Path):
    """Return a tiny network that exercises codegen order-sensitive cases."""
    from vulcan3.network import Network

    ni = 3
    nr = 4
    pad = ni
    max_terms = 2
    reactant_idx = np.full((nr + 1, max_terms), pad, dtype=np.int64)
    product_idx = np.full((nr + 1, max_terms), pad, dtype=np.int64)
    reactant_stoich = np.zeros((nr + 1, max_terms), dtype=np.float64)
    product_stoich = np.zeros((nr + 1, max_terms), dtype=np.float64)

    # R1/R2: 2A + M -> B, with an asymmetric non-three-body reverse B -> 2A.
    reactant_idx[1, 0] = 0
    reactant_stoich[1, 0] = 2.0
    product_idx[1, 0] = 1
    product_stoich[1, 0] = 1.0
    reactant_idx[2, 0] = 1
    reactant_stoich[2, 0] = 1.0
    product_idx[2, 0] = 0
    product_stoich[2, 0] = 2.0

    # R3/R4: B -> A + C, with a three-body reverse A + C + M -> B.
    reactant_idx[3, 0] = 1
    reactant_stoich[3, 0] = 1.0
    product_idx[3, 0] = 0
    product_stoich[3, 0] = 1.0
    product_idx[3, 1] = 2
    product_stoich[3, 1] = 1.0
    reactant_idx[4, 0] = 0
    reactant_stoich[4, 0] = 1.0
    reactant_idx[4, 1] = 2
    reactant_stoich[4, 1] = 1.0
    product_idx[4, 0] = 1
    product_stoich[4, 0] = 1.0

    is_forward = np.array([False, True, False, True, False])
    is_three_body = np.array([False, True, False, False, True])
    zeros = np.zeros(nr + 1, dtype=np.float64)
    bools = np.zeros(nr + 1, dtype=bool)
    return Network(
        species=("A", "B", "C"),
        species_idx={"A": 0, "B": 1, "C": 2},
        ni=ni,
        nr=nr,
        reactant_idx=reactant_idx,
        product_idx=product_idx,
        reactant_stoich=reactant_stoich,
        product_stoich=product_stoich,
        a=zeros.copy(),
        n=zeros.copy(),
        E=zeros.copy(),
        a_inf=zeros.copy(),
        n_inf=zeros.copy(),
        E_inf=zeros.copy(),
        is_forward=is_forward,
        is_three_body=is_three_body,
        has_kinf=bools.copy(),
        is_special=bools.copy(),
        is_conden=bools.copy(),
        is_photo=bools.copy(),
        is_ion=bools.copy(),
        stop_rev_indx=nr + 1,
        conden_indx=nr + 1,
        photo_sp=(),
        pho_rate_index={},
        n_branch={},
        ion_sp=(),
        ion_rate_index={},
        ion_branch={},
        Rf={1: "2*A + M -> B", 3: "B -> A + C"},
        network_path=str(tmp_path / "toy_network.txt"),
    )


def test_codegen_source_preserves_master_emission_rules(tmp_path):
    """Generated RHS keeps VULCAN-master's order-sensitive chemdf shape."""
    import vulcan3.make_chem_funs as mcf

    net = _toy_network_for_codegen(tmp_path)
    src = mcf.emit_chem_rhs_source(net)

    assert "v_1 = k[1]*y[0]*y[0]*M - k[2]*y[1]" in src
    assert "v_3 = k[3]*y[1] - k[4]*y[0]*y[2]*M" in src
    assert "dydt_0 = -1.0*v_1 -1.0*v_1 +1.0*v_3" in src
    assert "dydt_1 = +1.0*v_1 -1.0*v_3" in src
    assert "dydt_2 = +1.0*v_3" in src


def test_codegen_cache_key_changes_with_the_generator(tmp_path, monkeypatch):
    """A codegen fix must re-key, or the stale cache file is exec'd instead."""
    import vulcan3.make_chem_funs as mcf

    net = _toy_network_for_codegen(tmp_path)
    key0 = mcf.chem_rhs_cache_key(net)

    original = mcf._emit_rate_term

    def _emit_rate_term(net, slot_i, max_terms, PAD):  # same behaviour, new source
        return original(net, slot_i, max_terms, PAD)

    monkeypatch.setattr(mcf, "_emit_rate_term", _emit_rate_term)
    assert mcf.chem_rhs_cache_key(net) != key0


def test_codegen_cache_key_changes_with_consumed_network_fields(tmp_path):
    """Cache key invalidates when any codegen-consumed network table changes."""
    import dataclasses
    import vulcan3.make_chem_funs as mcf

    net = _toy_network_for_codegen(tmp_path)
    key0 = mcf.chem_rhs_cache_key(net)

    st_changed = net.reactant_stoich.copy()
    st_changed[1, 0] = 1.0
    key_stoich = mcf.chem_rhs_cache_key(
        dataclasses.replace(net, reactant_stoich=st_changed)
    )

    m_changed = net.is_three_body.copy()
    m_changed[2] = True
    key_m = mcf.chem_rhs_cache_key(dataclasses.replace(net, is_three_body=m_changed))

    assert key_stoich != key0
    assert key_m != key0


def _hd209_repeated_final_layer_fixture() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, object, tuple[str, ...], np.ndarray
]:
    """Return one HD209 final-state layer repeated to the production column shape
    (`tests/data/state_hd209.npz`, the shipped HD209 run's final state)."""
    from vulcan3.config import default_config
    import vulcan3.network as net_mod

    saved = ROOT / "tests" / "data" / "state_hd209.npz"
    if not saved.exists():
        # conftest refuses a missing fixture unless VULCAN3_ALLOW_MISSING_FIXTURES=1
        pytest.skip(f"{saved.name} missing (VULCAN3_ALLOW_MISSING_FIXTURES=1); build it "
                    "with `python tests/gen_fixtures.py --only hd209`")
    state = np.load(saved)

    net = net_mod.parse_network(default_config().network)
    if list(state["species"]) != list(net.species):
        pytest.fail("Current default network does not match the saved HD209 state; rebuild "
                    "it with `python tests/gen_fixtures.py --only hd209 --force`")

    layer = 2
    n_layers = int(state["y"].shape[0])
    y = np.repeat(state["y"][layer : layer + 1], n_layers, axis=0)
    M = np.repeat(state["M"][layer : layer + 1], n_layers)
    k_arr = np.zeros((net.nr + 1, n_layers), dtype=np.float64)
    k_arr[1:] = state["k_arr"][1 : net.nr + 1, layer, None]
    atoms = ("H", "O", "C", "N")
    atom_counts = atom_count_matrix(net, atoms)
    return y, M, k_arr, net, atoms, atom_counts


def test_hd209_jit_rhs_projection_removes_atom_residual() -> None:
    """Projected HD209 RHS carries no atom residual above float64 roundoff. The
    raw jitted residual depends on the compiler (FMA fusion differs by
    platform), so only the unjitted and projected residuals are asserted."""
    import jax
    import jax.numpy as jnp
    import vulcan3.make_chem_funs as mcf
    import vulcan3.jax_step as jax_step
    from _oracles import chem_rhs_numpy

    y, M, k_arr, net, atoms, atom_counts = _hd209_repeated_final_layer_fixture()
    ns: dict = {}
    exec(compile(mcf.emit_chem_rhs_source(net), "<hd209_codegen>", "exec"), ns)
    raw_jit = jax.jit(ns["chem_rhs_codegen"])

    y_j = jnp.asarray(y)
    M_j = jnp.asarray(M)
    k_j = jnp.asarray(k_arr)
    out_jit = np.asarray(raw_jit(y_j, M_j, k_j).block_until_ready())
    with jax.disable_jit():
        out_nojit = np.asarray(raw_jit(y_j, M_j, k_j))
    out_numpy = chem_rhs_numpy(y, M, k_arr, net)

    np.testing.assert_array_equal(out_nojit, out_numpy)

    c_idx = atoms.index("C")
    nojit_residual = out_nojit @ atom_counts
    projected = np.asarray(jax_step._project_chem_rhs(jnp.asarray(out_jit)))
    projected_residual = projected @ atom_counts
    # Floor from the state itself: the magnitude the residual cancels.
    floor = _ATOM_RESIDUAL_EPS * (np.abs(out_jit) @ np.abs(atom_counts))
    # The projection zeroes the residual to roundoff whatever the raw one was,
    # so this also says it is no worse than the raw residual beyond roundoff.
    assert np.all(np.abs(projected_residual) <= floor), (
        projected_residual[0, c_idx], floor[0, c_idx])
    assert np.all(np.abs(nojit_residual) <= floor)

    reservoir_idx = [net.species_idx[sp] for sp in ("H2", "H2O", "CO", "N2")]
    non_reservoir_delta = np.delete(projected - out_jit, reservoir_idx, axis=1)
    assert float(np.max(np.abs(non_reservoir_delta))) == 0.0


def test_hd209_jacobian_projection_uses_same_reservoir_rows() -> None:
    """Projected chemistry Jacobian: atom residual at roundoff, only reservoir rows changed."""
    import jax
    import jax.numpy as jnp
    import vulcan3.chem as chem_mod
    import vulcan3.jax_step as jax_step

    y, M, k_arr, net, _, atom_counts = _hd209_repeated_final_layer_fixture()
    net_jax = chem_mod.to_jax(net)
    chem_jac = np.asarray(
        jax.jit(chem_mod.chem_jac_analytical)(
            jnp.asarray(y), jnp.asarray(M), jnp.asarray(k_arr), net_jax
        ).block_until_ready()
    )
    projected = np.asarray(jax_step._project_chem_jac(jnp.asarray(chem_jac)))

    after = np.einsum("ia,zij->zaj", atom_counts, projected)
    floor = _ATOM_RESIDUAL_EPS * np.einsum(
        "ia,zij->zaj", np.abs(atom_counts), np.abs(chem_jac)
    )
    assert np.all(np.abs(after) <= floor), float(np.max(np.abs(after) - floor))

    reservoir_idx = [net.species_idx[sp] for sp in ("H2", "H2O", "CO", "N2")]
    non_reservoir_delta = np.delete(projected - chem_jac, reservoir_idx, axis=1)
    assert float(np.max(np.abs(non_reservoir_delta))) == 0.0


def test_codegen_matches_numpy_oracle(hd189_state):
    """Codegen RHS matches chem_rhs_numpy; the per-species floor absorbs
    cancellation on trace species. The column is scaled by exp(U(-1,1)) off
    equilibrium: at the EQ seed the net RHS is a small difference of large
    terms, where a per-cell relative comparison is ill-posed."""
    import vulcan3.network as net_mod
    from vulcan3.config import default_config

    net = net_mod.parse_network(default_config().network)
    M = np.asarray(hd189_state.atm.M, dtype=np.float64)
    k_arr = np.asarray(hd189_state.var.k_arr, dtype=np.float64)
    y = np.asarray(hd189_state.var.y, dtype=np.float64)
    y = y * np.exp(np.random.default_rng(0).uniform(-1.0, 1.0, y.shape))

    import jax.numpy as jnp
    import vulcan3.make_chem_funs as mcf
    from _oracles import chem_rhs_numpy

    fn = mcf.build_chem_rhs(net)
    out_codegen = np.asarray(fn(jnp.asarray(y), jnp.asarray(M), jnp.asarray(k_arr)))
    out_numpy = chem_rhs_numpy(y, M, k_arr, net)

    per_species_max = np.maximum(np.abs(out_numpy).max(axis=0), 1e-30)
    denom = np.maximum(np.abs(out_numpy), PEAK_FLOOR_FRAC * per_species_max[None, :])
    relerr = np.abs(out_codegen - out_numpy) / denom
    max_rel = float(relerr.max())
    idx = np.unravel_index(int(relerr.argmax()), relerr.shape)

    assert max_rel < CODEGEN_RTOL, (
        f"codegen vs numpy oracle disagreement: max relerr={max_rel:.3e} "
        f"at layer {idx[0]} species {net.species[idx[1]]} (threshold {CODEGEN_RTOL:g})"
    )



@pytest.mark.xfail(
    sys.platform == "darwin" and platform.machine() == "arm64", strict=True,
    reason="XLA:CPU on Apple Silicon rounds the grouped and codegen operation orders "
    "differently: roundoff only (notes \"Efficiency program\", grouped RHS row)",
)
@pytest.mark.parametrize("n_lanes", [0, 8], ids=["solo", "vmap8"])
def test_grouped_rhs_is_bitwise_codegen(hd189_state, monkeypatch, n_lanes):
    """`VULCAN3_CHEM_RHS=grouped` evaluates the codegen's operation trees as
    vectorised gathers, so it must equal the codegen RHS bit for bit, solo and
    under a lane vmap of perturbed columns (the batched runner's shape)."""
    import jax
    import jax.numpy as jnp
    import vulcan3.make_chem_funs as mcf
    import vulcan3.network as net_mod
    from vulcan3.config import default_config

    monkeypatch.setattr(mcf, "CHEM_RHS_IMPL", "codegen")  # whatever the env selects
    net = net_mod.parse_network(default_config().network)
    y = np.asarray(hd189_state.var.y, dtype=np.float64)
    M = np.asarray(hd189_state.atm.M, dtype=np.float64)
    k = np.asarray(hd189_state.var.k_arr, dtype=np.float64)
    codegen, grouped = mcf.build_chem_rhs(net), mcf.build_chem_rhs_grouped(net)
    if n_lanes:
        rng = np.random.default_rng(0)
        y = y * np.exp(rng.uniform(-1.0, 1.0, (n_lanes, *y.shape)))
        M = np.broadcast_to(M, (n_lanes, *M.shape))
        k = np.broadcast_to(k, (n_lanes, *k.shape))
        codegen, grouped = jax.vmap(codegen), jax.vmap(grouped)
    args = tuple(jnp.asarray(a) for a in (y, M, k))
    assert np.array_equal(np.asarray(codegen(*args)), np.asarray(grouped(*args)))
