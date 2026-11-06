"""Parse the network once per process at import and expose its tables.

Exposes ni, nr, spec_list, the parsed network (NETWORK, NET_JAX) and the codegen chemistry RHS (chem_rhs_codegen).
"""

from __future__ import annotations

from . import chem as _chem, make_chem_funs as _make_chem_funs, network as _network
from ._paths import resolve_data_path
from .config import default_config

__all__ = ["NETWORK", "NET_JAX", "chem_rhs_codegen", "ni", "nr", "spec_list"]

_CFG = default_config()


NETWORK = _network.parse_network(str(resolve_data_path(_CFG.network)))
NET_JAX = _chem.to_jax(NETWORK)
_NETWORK, _NET_JAX = NETWORK, NET_JAX  # deprecated (docs/api.md)

# The production RHS (the integrator binds it), built or reloaded from cache.
# Memoised in `_make_chem_funs._BUILD_CACHE`.
chem_rhs_codegen = _make_chem_funs.build_chem_rhs(NETWORK)

ni: int = NETWORK.ni
nr: int = NETWORK.nr
spec_list: list[str] = list(NETWORK.species)
