"""VULCAN3: JAX-accelerated photochemical kinetics for exoplanet atmospheres.

Float64 is enabled at import time (non-negotiable for rate-constant
arithmetic spanning ~50 orders of magnitude).
"""

from __future__ import annotations

from typing import Any

# Unguarded: a jax that cannot enable x64 must
# fail the import, not run float32 chemistry (rate constants span ~50 dex).
from jax import config as _config

from ._version import __version__

_config.update("jax_enable_x64", True)

from .config import Config, default_config, load_config, validate_overrides  # noqa: E402 (x64 is set first)
from .state import RunState  # noqa: E402 (x64 is set first)

__all__ = [
    "Config",
    "RunState",
    "__version__",
    "conv_normal",
    "default_config",
    "load_config",
    "make_config",
    "validate_overrides",
]


def make_config(**overrides: Any) -> Config:
    """Load the default config with user overrides (alias for ``load_config``).

    Example::

        cfg = vulcan3.make_config(nz=100, use_photo=False)
        rs = vulcan3.RunState.with_pre_loop_setup(cfg)
    """
    return load_config("default", **overrides)


def __getattr__(name: str) -> Any:
    # Lazy: outer_loop imports jax_step, which freezes $VULCAN3_SOLVER;
    # `import vulcan3` must leave that choice open.
    if name == "conv_normal":
        from .outer_loop import conv_normal

        return conv_normal
    raise AttributeError(f"module 'vulcan3' has no attribute {name!r}")
