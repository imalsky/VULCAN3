"""Rank the reactions that make and destroy one species in a saved run (VULCAN 2's ``diagnose.py``).

Reaction structure comes from `network.parse_network`; rate constants, densities and the total density
come from the ``.vul`` pickle. See docs/api.md.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ._paths import resolve_data_path
from .config import default_config
from .network import Network, parse_network

__all__ = ["ReactionSet", "SpeciesRates", "reaction_rates"]


@dataclass(frozen=True)
class ReactionSet:
    """Reactions of one kind (producing or destroying), strongest first.

    Attributes
    ----------
    slot : np.ndarray
        Shape [m]. Reaction ids as in ``variable['k']``. Odd ids are forward, even ids the reverse of the id below.
    equation : tuple[str, ...]
        Length m. Reaction text; a reverse slot reads "<forward text> (reverse)".
    stoich : np.ndarray
        Shape [m]. Molecules of the species made (or lost) per reaction event.
    rate : np.ndarray
        Shape [m, nl]. Reaction rate, cm^-3 s^-1, at each requested layer.
    flux : np.ndarray
        Shape [m, nl]. ``stoich * rate``: the species' production (or loss) rate, cm^-3 s^-1.
    """

    slot: np.ndarray
    equation: tuple[str, ...]
    stoich: np.ndarray
    rate: np.ndarray
    flux: np.ndarray


@dataclass(frozen=True)
class SpeciesRates:
    """Production and loss reactions of one species.

    Attributes
    ----------
    species : str
        The species name.
    layers : np.ndarray
        Shape [nl]. Indices of the requested layers, bottom (0) to top.
    production : ReactionSet
        The reactions that make the species. A reaction with the species on both sides is also in `loss`.
    loss : ReactionSet
        The reactions that destroy the species.
    """

    species: str
    layers: np.ndarray
    production: ReactionSet
    loss: ReactionSet


def reaction_rates(
    vul: Path | str | dict[str, Any],
    species: str,
    layers: slice | None = None,
    top: int | None = 20,
    *,
    network: Path | str | None = None,
) -> SpeciesRates:
    """Rank the reactions that produce and destroy `species` in a saved run.

    The rate of slot j is ``k[j] * prod(y_r ** n_r)``, times the total density for a row that carries M. This is the
    term the chemistry RHS uses, so summing ``production.flux - loss.flux`` over all reactions gives the chemistry
    tendency of the species. Photolysis rows use the saved rate J (s^-1). Unlike VULCAN 2's ``diagnose.py``, the
    condensate S8_l_s is not skipped.

    Parameters
    ----------
    vul : Path, str or dict
        A ``.vul`` file (pickle; load only files you trust) or its loaded dict.
    species : str
        Species name, as in ``variable['species']``.
    layers : slice or None
        Layers to rank over, bottom (0) to top; None takes all.
    top : int or None
        Keep this many reactions of each kind, ranked by the maximum of ``flux`` over the layers; None keeps all.
    network : Path, str or None
        Network file the run used; None takes the default config's. Refused if its species or reaction text differ
        from the run's.

    Returns
    -------
    SpeciesRates
        Production and loss reactions, each with rates of shape [m, nl] in cm^-3 s^-1.

    Raises
    ------
    ValueError
        If `species` is not in the run, or the network does not match the run.

    Examples
    --------
    >>> r = reaction_rates("output/HD189.vul", "CH4", slice(25, -1), top=3)  # doctest: +SKIP
    >>> r.loss.equation[0], r.loss.flux.max(axis=1)[0]  # doctest: +SKIP
    """
    if isinstance(vul, dict):
        data = vul
    else:
        with Path(vul).open("rb") as f:
            data = pickle.load(f)
    var = data["variable"]
    net = parse_network(resolve_data_path(default_config().network) if network is None else network)
    if list(var["species"]) != list(net.species) or dict(var["Rf"]) != net.Rf:
        raise ValueError(
            f"the run's species or reaction text differ from {net.network_path}; "
            "pass network= the file the run used (its config's `network`)"
        )
    if species not in net.species_idx:
        raise ValueError(f"{species!r} is not a species of the run; pick one of {list(net.species)}")

    layer_index = np.arange(len(var["y"]))[slice(None) if layers is None else layers]
    y = np.asarray(var["y"], dtype=np.float64)[layer_index]
    n_0 = np.asarray(data["atm"]["n_0"], dtype=np.float64)[layer_index]
    sp = net.species_idx[species]
    made = _ranked(net, sp, net.product_idx, net.product_stoich, var["k"], y, n_0, layer_index, top)
    lost = _ranked(net, sp, net.reactant_idx, net.reactant_stoich, var["k"], y, n_0, layer_index, top)
    return SpeciesRates(species, layer_index, made, lost)


def _ranked(
    net: Network,
    sp: int,
    side_idx: np.ndarray,
    side_stoich: np.ndarray,
    k: dict[int, Any],
    y: np.ndarray,
    n_0: np.ndarray,
    layer_index: np.ndarray,
    top: int | None,
) -> ReactionSet:
    """Reactions with `sp` on one side of the table pair, ranked by the maximum flux over the layers."""
    slots = np.flatnonzero((side_idx == sp).any(axis=1))
    stoich = np.array([side_stoich[j][side_idx[j] == sp].sum() for j in slots])
    rate = np.array([_slot_rate(net, int(j), k, y, n_0, layer_index) for j in slots]).reshape(len(slots), -1)
    flux = stoich[:, None] * rate
    order = np.argsort(-flux.max(axis=1), kind="stable")[:top]
    slots, stoich, rate, flux = slots[order], stoich[order], rate[order], flux[order]
    equation = tuple(net.Rf[j] if net.is_forward[j] else f"{net.Rf[j - 1]} (reverse)" for j in slots)
    return ReactionSet(slots, equation, stoich, rate, flux)


def _slot_rate(
    net: Network, j: int, k: dict[int, Any], y: np.ndarray, n_0: np.ndarray, layers: np.ndarray
) -> np.ndarray:
    """Rate of slot `j` per layer, cm^-3 s^-1: ``k * prod(y ** stoich)``, times n_0 if the row carries M."""
    rate: np.ndarray = np.asarray(k[j], dtype=np.float64)[layers].copy()
    for idx, st in zip(net.reactant_idx[j], net.reactant_stoich[j], strict=True):
        if st > 0:
            rate *= y[:, idx] ** int(st)
    if net.is_three_body[j]:
        rate *= n_0
    return rate
