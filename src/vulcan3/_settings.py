"""The one reader of the `VULCAN3_*` environment variables; docs/settings.md lists them.

Each function reads the environment when called, so `import vulcan3` leaves `$VULCAN3_SOLVER` open until `jax_step`
first loads, and `$VULCAN3_CHEM_RHS` open until `make_chem_funs` does.
"""

from __future__ import annotations

import os

SOLVERS = ("fast", "ffi", "reference")
CHEM_RHS_FORMS = ("codegen", "grouped")
RELAUNCH_GUARD = "VULCAN3_CLI_RELAUNCHED"  # set on the re-executed CLI, so a failed relaunch raises instead of looping


def solver() -> str:
    """Return the block solve: `fast` (default), `ffi` or `reference`."""
    value = os.environ.get("VULCAN3_SOLVER", "fast")
    if value not in SOLVERS:
        raise ValueError(f"VULCAN3_SOLVER={value!r}: expected 'fast' (default), 'ffi', or 'reference'")
    return value


def chem_rhs() -> str:
    """Return the chemistry RHS form: `codegen` (default) or `grouped`."""
    value = os.environ.get("VULCAN3_CHEM_RHS", "codegen")
    if value not in CHEM_RHS_FORMS:
        raise ValueError(f"VULCAN3_CHEM_RHS={value!r}: expected 'codegen' (default) or 'grouped'")
    return value


def frozen_override(env_name: str) -> str | None:
    """Return the value of an import-frozen knob's variable, or None when unset."""
    return os.environ.get(env_name)


def relaunched() -> bool:
    """Say whether this process is the CLI's re-exec."""
    return bool(os.environ.get(RELAUNCH_GUARD))


def refuse_old_prefix() -> None:
    """Refuse a `VULCAN_JAX_*` variable: the name it asks for is ignored, so the job would run the default."""
    old = sorted(k for k in os.environ if k.startswith("VULCAN_JAX_"))
    if old:
        raise ImportError(
            f"{old} were renamed: use the VULCAN3_ prefix (e.g. VULCAN3_SOLVER) in the environment or job script."
        )
