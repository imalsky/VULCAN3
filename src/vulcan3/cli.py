"""VULCAN3 driver: build RunState, integrate, pickle .vul. Public name: `cli_main`."""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import jax

from . import _settings, legacy_io, op_jax, outer_loop
from .config import _FROZEN_ENV, dump_config, load_config
from .state import RunState, _frozen_mismatch

if TYPE_CHECKING:
    from .config import Config

__all__ = ["cli_main"]

_RELAUNCH_GUARD = _settings.RELAUNCH_GUARD


def _frozen_knob_mismatch(cfg: Config) -> dict[str, str]:
    """Import-frozen knobs `cfg` disagrees with (`state._frozen_mismatch`), as ``{env var: cfg's value}``.

    `load_config` folds ``$VULCAN3_*`` over the YAML before returning, so a
    knob the caller overrode explicitly matches the frozen value and never
    appears here.
    """
    return {
        _FROZEN_ENV[knob]: (",".join(str(a) for a in cfg.atom_list) if knob == "atom_list" else str(getattr(cfg, knob)))
        for knob in _frozen_mismatch(cfg)
    }


def _relaunch_for_frozen_knobs(cfg: Config, config_ref: str) -> None:
    """Re-exec with ``$VULCAN3_*`` set when the config needs a different network.

    ``network`` / ``atom_list`` / ``com_file`` are read once at the first
    ``import vulcan3`` -- the parsed network, the composition tables, and the
    codegen RHS are all built from them -- so a config naming a different
    network cannot be honored in this process. Re-exec once with the variables
    set instead of making the caller prefix them onto every command, so
    ``vulcan3 --config W39b`` works on its own.

    Returns normally when no relaunch is needed; otherwise does not return.
    """
    mismatch = _frozen_knob_mismatch(cfg)
    if not mismatch:
        return
    assignments = " ".join(f"{k}={v}" for k, v in sorted(mismatch.items()))
    if _settings.relaunched():
        raise RuntimeError(
            f"config {config_ref!r} still disagrees with the import-frozen "
            f"network settings after a relaunch with {assignments}. This should "
            "not happen; run with the environment variables set manually and "
            "report the config."
        )
    env = dict(os.environ)
    env.update(mismatch)
    env[_RELAUNCH_GUARD] = "1"
    # One OpenMP thread, read when the new process loads NumPy: the hot path is one XLA program and BLAS threads only
    # oversubscribe.
    env["OMP_NUM_THREADS"] = "1"
    print(
        f"Config {config_ref!r} needs import-frozen settings that differ from the "
        f"defaults; relaunching with {assignments}"
    )
    sys.stdout.flush()
    os.execve(
        sys.executable,
        [
            sys.executable,
            "-m",
            "vulcan3.cli",
            "--config",
            str(config_ref),
        ],
        env,
    )


def cli_main(argv: list[str] | None = None) -> None:
    """Entry point for the ``vulcan3`` console script.

    ``--config`` selects a config by name (``default``, ``W39b``, ...; resolved
    CWD-first then packaged) or an explicit YAML path. The resolved config is
    written next to the output as ``<out_name>.config.yaml`` so the run can be
    reproduced with ``vulcan3 --config <that file>``.

    A config selecting a non-default network re-execs once with the matching
    ``$VULCAN3_*`` variables set (see `_relaunch_for_frozen_knobs`).
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    # Persistent XLA compile cache: the codegen RHS and the whole-loop while_loop take seconds to compile cold and are
    # reused verbatim across runs. JAX's default thresholds apply, so compiles faster than 1 s are not cached.
    jax.config.update(
        "jax_compilation_cache_dir",
        os.environ.get("JAX_COMPILATION_CACHE_DIR", str(Path("~/.cache/vulcan3").expanduser())),
    )
    parser = argparse.ArgumentParser(prog="vulcan3")
    parser.add_argument(
        "--config",
        "-c",
        default="default",
        help="config name (default, HD189, W39b, ...) or path to a YAML file.",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    # Before any validation: a config naming a non-default network can only be
    # honored by a process that imported vulcan3 with it already selected.
    _relaunch_for_frozen_knobs(cfg, args.config)
    # Validates cfg before any setup work.
    runstate = RunState.with_pre_loop_setup(cfg)

    dname = str(Path.cwd())
    output = legacy_io.Output(cfg=cfg)
    output.save_cfg(dname)

    # Save the fully-resolved config as YAML so the run is reproducible.
    resolved_path = Path(cfg.output_dir).absolute() / f"{cfg.out_name}.config.yaml"
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, resolved_path)
    print(f"Resolved config written to {resolved_path}")

    solver = op_jax.Ros2JAX()
    integ = outer_loop.OuterLoop(solver, output, cfg=cfg)

    print(f"VULCAN3 starting integration at t=0, dt={float(runstate.step.dt):.2e}")
    try:
        # Chunked so Ctrl+C lands between chunks (`OuterLoop.__call__`).
        runstate = integ(runstate, chunk=int(cfg.print_prog_num))
    except KeyboardInterrupt:
        if integ.interrupted_runstate is not None:
            runstate = integ.interrupted_runstate
            print(
                f"\nInterrupted. `runstate` is the last finished chunk "
                f"(step {int(runstate.params.count)}, t={float(runstate.step.t):.3e} s)."
            )
        else:
            print("\nInterrupted before the first chunk finished; `runstate` is the initial state.")
        print("`output.save_out(runstate, dname)` writes it as a .vul; Ctrl-D quits.")
        _drop_into_repl({"cfg": cfg, "runstate": runstate, "integ": integ, "output": output, "dname": dname})
        # An interrupted run is not a success, also for a job wrapper whose
        # stdin is not a terminal (the shell then exits on EOF at once).
        sys.exit(130)

    print(f"VULCAN3 done. Saving output to {cfg.output_dir}{cfg.out_name}")
    output.save_out(runstate, dname)
    if cfg.use_plot_end:
        output.plot_end(runstate)
    if cfg.use_plot_evo:
        output.plot_evo(runstate)

    print(f"Total wall time: {time.time() - runstate.metadata.start_time:.1f}s")


def _drop_into_repl(namespace: dict[str, object]) -> None:
    """IPython if installed, else the standard console with line editing."""
    try:
        from IPython import embed
    except ImportError:
        import code

        with contextlib.suppress(ImportError):
            import readline  # noqa: F401 (arrow keys and history in `code`)
        code.interact(local=namespace)
    else:
        embed(user_ns=namespace, colors="neutral")


if __name__ == "__main__":
    cli_main()
