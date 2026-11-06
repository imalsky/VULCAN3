"""Parse the network once per process at import and expose its tables.

Exposes ni, nr, spec_list, the parsed network (_NETWORK, _NET_JAX) and the codegen chemistry RHS (chem_rhs_codegen).
"""

from __future__ import annotations

from . import chem as _chem, make_chem_funs as _make_chem_funs, network as _network
from ._paths import resolve_data_path
from .config import default_config

__all__ = ["NETWORK", "NET_JAX", "chem_rhs_codegen", "ni", "nr", "spec_list"]

_CFG = default_config()


_NETWORK = _network.parse_network(str(resolve_data_path(_CFG.network)))
_NET_JAX = _chem.to_jax(_NETWORK)
# Public names of the two above; the underscored ones are deprecated and go after one release.
NETWORK = _NETWORK
NET_JAX = _NET_JAX

# The production RHS (the integrator binds it), built or reloaded from cache.
# Memoised in `_make_chem_funs._BUILD_CACHE`.
chem_rhs_codegen = _make_chem_funs.build_chem_rhs(_NETWORK)

ni: int = _NETWORK.ni
nr: int = _NETWORK.nr
spec_list: list[str] = list(_NETWORK.species)
