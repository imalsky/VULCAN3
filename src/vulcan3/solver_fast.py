"""Default block-Thomas path (`VULCAN3_SOLVER=fast|ffi`, import-frozen).

Same call shape as `solver.py`: `factor(diag, sup_d, sub_d)` once, then `solve(factors, rhs)` per Ros2 stage. `solve` is
a `lax.custom_linear_solve`, so both stages and every tangent direction share one factorisation. `ffi` runs the C++
kernel on the CPU and the CUDA kernel on a GPU; `python -m vulcan3.solver_fast [--cuda]` builds them. `reference` is
handled in `jax_step`. Design: docs/method.md. Variables: docs/settings.md.
"""

from __future__ import annotations

import ctypes
import functools
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp

from . import _settings, solver as _ref

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = ["BACKEND", "CUDA_MAX_NI", "Factors", "build", "factor", "solve"]

BACKEND = _settings.solver()
if BACKEND == "reference":
    raise ValueError("VULCAN3_SOLVER='reference' does not use this module: jax_step takes the plain solver.py pair")


class Factors(NamedTuple):
    """Primal LU factors and permutations, plus the bands they factor.

    The factors have no tangent (computed on stop_gradient inputs); the bands carry the tangents.
    """

    diag_lu: jnp.ndarray
    diag_perm: jnp.ndarray
    diag: jnp.ndarray
    sup_d: jnp.ndarray
    sub_d: jnp.ndarray


def _matvec(diag: jnp.ndarray, sup_d: jnp.ndarray, sub_d: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    out = jnp.einsum("zij,zj->zi", diag, x)
    out = out.at[:-1].add(sup_d * x[1:])
    return out.at[1:].add(sub_d * x[:-1])


def _raw_factor(diag: jnp.ndarray, sup_d: jnp.ndarray, sub_d: jnp.ndarray) -> Sequence[jnp.ndarray]:
    if BACKEND == "ffi":
        return _ffi_factor(diag, sup_d, sub_d)
    f = _ref.factor_block_thomas_diag_offdiag(diag, sup_d, sub_d)
    return f.diag_lu, f.diag_perm


def _raw_solve(
    lu: jnp.ndarray, perm: jnp.ndarray, sup_d: jnp.ndarray, sub_d: jnp.ndarray, rhs: jnp.ndarray
) -> jnp.ndarray:
    if BACKEND == "ffi":
        return _ffi_solve(lu, perm, sup_d, sub_d, rhs)
    return _ref.solve_block_thomas_diag_offdiag(_ref.BlockThomasDiagFactors(lu, perm, sup_d, sub_d), rhs)


def factor(diag: jnp.ndarray, sup_d: jnp.ndarray, sub_d: jnp.ndarray) -> Factors:
    """Factor the block-tridiagonal stage matrix on the selected backend.

    Parameters
    ----------
    diag : jnp.ndarray [nz, ni, ni]
        Diagonal blocks.
    sup_d : jnp.ndarray [nz - 1, ni]
        Diagonals of the super-diagonal blocks: row j couples to ``sup_d[j] * x[j + 1]``.
    sub_d : jnp.ndarray [nz - 1, ni]
        Diagonals of the sub-diagonal blocks: row j + 1 couples to ``sub_d[j] * x[j]``.

    Returns
    -------
    Factors
        The primal LU factors (no tangent) and the bands, which carry the tangents.
    """
    sg = jax.lax.stop_gradient
    lu, perm = _raw_factor(sg(diag), sg(sup_d), sg(sub_d))
    return Factors(lu, perm, diag, sup_d, sub_d)


def solve(
    factors: Factors, rhs: jnp.ndarray, matvec: Callable[[jnp.ndarray], jnp.ndarray] | None = None
) -> jnp.ndarray:
    """Solve the factored block-tridiagonal system for `rhs`, as a `lax.custom_linear_solve`.

    Under `ffi` there is no transpose rule, so reverse mode is refused.

    Parameters
    ----------
    factors : Factors
        From `factor`.
    rhs : jnp.ndarray [nz, ni]
        Right-hand side.
    matvec : callable or None, optional
        The operator as a function of x ([nz, ni] -> [nz, ni]). The primal never runs it; the AD rules do (the
        tangent's `dA x`, the transpose's cotangent). None takes the dense bands in `factors`, whose tangent is the
        whole dense `dA` per direction; `jax_step._ros2_stages` passes a matrix-free one.

    Returns
    -------
    jnp.ndarray [nz, ni]
        The solution.
    """
    lu, perm, diag, sup_d, sub_d = factors
    sg = jax.lax.stop_gradient
    sup0, sub0 = sg(sup_d), sg(sub_d)

    if matvec is None:
        matvec = functools.partial(_matvec, diag, sup_d, sub_d)

    def solve_(_matvec: object, b: jnp.ndarray) -> jnp.ndarray:
        return _raw_solve(lu, perm, sup0, sub0, b)

    def transpose_solve(_vecmat: object, c: jnp.ndarray) -> jnp.ndarray:
        # Transpose of the primal sweep on the same factors (no second
        # factorisation). The FFI call has no transpose rule, so
        # this always runs the JAX scan, on whichever backend's factors.
        def sweep(b: jnp.ndarray) -> jnp.ndarray:
            return _ref.solve_block_thomas_diag_offdiag(_ref.BlockThomasDiagFactors(lu, perm, sup0, sub0), b)

        _, vjp = jax.vjp(sweep, c)
        out: jnp.ndarray = vjp(c)[0]
        return out

    # Reverse mode through `ffi` is refused by leaving the transpose out: `custom_linear_solve` traces it eagerly, so a
    # raise inside it would fire on every run.
    sol: jnp.ndarray = jax.lax.custom_linear_solve(
        matvec, rhs, solve_, transpose_solve=None if BACKEND == "ffi" else transpose_solve
    )
    return sol


# --- C++ CPU kernel and its CUDA twin ---------------------------------------

_CSRC = Path(__file__).with_name("csrc")
_LIB = _CSRC / ("libblock_thomas_cpu" + (".dylib" if platform.system() == "Darwin" else ".so"))
_CUDA_LIB = _CSRC / "libblock_thomas_cuda.so"
_HASH_TAG = b"vulcan-bt-source-sha256:"  # VulcanBtSourceHash, block_thomas_common.h
# The CUDA factor's register tile (block_thomas_cuda.cu kGjMaxNi); its shared
# memory at that size (77.6 kB) fits every sm_80+ GPU.
CUDA_MAX_NI = 96


def _source_hash(src: Path) -> str:
    """sha256 of a kernel source and the header it includes."""
    h = hashlib.sha256(src.read_bytes())
    h.update((_CSRC / "block_thomas_common.h").read_bytes())
    return h.hexdigest()


def _is_current(lib: Path, stamp: str) -> bool:
    """Return True if `lib` embeds `stamp`, alone or as the start of a longer stamp.

    The file is read, not loaded: dlopen would keep a stale library cached under its path.
    """
    return lib.exists() and _HASH_TAG + stamp.encode() in lib.read_bytes()


def _nvcc_arch() -> str:
    """Return $NVCC_ARCH, else sm_XY of JAX's first CUDA device.

    Falls back on the host's first GPU per nvidia-smi (a build shell may hide the GPU from JAX, e.g. with
    JAX_PLATFORMS=cpu).
    """
    if arch := os.environ.get("NVCC_ARCH"):
        return arch
    try:
        cc: str = jax.devices("cuda")[0].compute_capability
    except RuntimeError:
        try:
            cc = subprocess.run(
                ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.split()[0]
        except (OSError, subprocess.CalledProcessError, IndexError) as e:
            raise RuntimeError("no CUDA device to build the kernel for; set NVCC_ARCH (sm_90 on a GH200)") from e
    return "sm_" + cc.replace(".", "")


def build(force: bool = False, cuda: bool = False) -> Path:
    """Compile the CPU kernel with the system C++ compiler ($CXX or c++).

    With `cuda=True`, compile the GPU twin with nvcc ($NVCC, arch from `_nvcc_arch`). Returns the library at once if it
    embeds the stamp of the present build, unless `force`: the hash of the sources and, for the GPU twin, after it a
    hash of the arch and `nvcc --version`, so a build for another GPU or toolchain is redone.

    Parameters
    ----------
    force : bool, optional
        Rebuild even when the library is current.
    cuda : bool, optional
        Build the CUDA kernel instead of the CPU one.

    Returns
    -------
    Path
        The shared library under ``csrc/``.
    """
    src = _CSRC / ("block_thomas_cuda.cu" if cuda else "block_thomas_cpu.cc")
    lib = _CUDA_LIB if cuda else _LIB
    stamp = _source_hash(src)
    if cuda:
        nvcc, arch = os.environ.get("NVCC", "nvcc"), _nvcc_arch()
        version = subprocess.run([nvcc, "--version"], check=True, capture_output=True, text=True).stdout
        stamp += "_" + hashlib.sha256(f"{arch}\n{version}".encode()).hexdigest()
    if not force and _is_current(lib, stamp):
        return lib
    # -isystem: the XLA headers' own warnings are not reported. No -Werror: the CPU build runs on the first `ffi` call,
    # where a newer compiler's warning must not fail.
    flags = ["-O2", "-std=c++17", "-shared", f"-DVULCAN_BT_SOURCE_HASH={stamp}", "-isystem", jax.ffi.include_dir()]
    if cuda:
        cmd = [nvcc, *flags, f"-arch={arch}", "-lineinfo", "-Xcompiler", "-fPIC,-Wall,-Wextra"]
    else:
        cmd = [os.environ.get("CXX", "c++"), *flags, "-fPIC", "-Wall", "-Wextra"]
        if platform.system() == "Darwin":
            cmd += ["-undefined", "dynamic_lookup"]
    # Compile in a private directory beside the library and rename it into
    # place, so a concurrent first build never loads a half-written file.
    tmp_dir = tempfile.mkdtemp(dir=lib.parent)
    try:
        tmp = Path(tmp_dir) / lib.name
        subprocess.run([*cmd, str(src), "-o", tmp], check=True)
        tmp.replace(lib)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return lib


_REGISTERED = False


def _register() -> None:
    global _REGISTERED  # noqa: PLW0603 (registration happens once per process)
    if _REGISTERED:
        return
    # The CUDA twin is built only by `--cuda`, never here, so a stale one is
    # refused rather than loaded (`build` rebuilds the CPU kernel). Only its
    # sources are checked here, the start of its stamp: a GPU job's JAX
    # process need not have nvcc. `build` checks the arch and the nvcc version.
    cu_src = _CSRC / "block_thomas_cuda.cu"
    if _CUDA_LIB.exists() and not _is_current(_CUDA_LIB, _source_hash(cu_src)):
        raise RuntimeError(
            f"{_CUDA_LIB} was not built from the present {cu_src.name} and "
            "block_thomas_common.h; rebuild it on the GPU host with `python -m "
            "vulcan3.solver_fast --cuda`, or delete it to run the CPU kernel only."
        )
    lib = ctypes.CDLL(str(build()))
    jax.ffi.register_ffi_target("vulcan_bt_factor", jax.ffi.pycapsule(lib.VulcanBtFactor), platform="cpu")
    jax.ffi.register_ffi_target("vulcan_bt_solve", jax.ffi.pycapsule(lib.VulcanBtSolve), platform="cpu")
    if _CUDA_LIB.exists():
        gpu = ctypes.CDLL(str(_CUDA_LIB))
        jax.ffi.register_ffi_target("vulcan_bt_factor", jax.ffi.pycapsule(gpu.VulcanBtFactorCuda), platform="CUDA")
        jax.ffi.register_ffi_target("vulcan_bt_solve", jax.ffi.pycapsule(gpu.VulcanBtSolveCuda), platform="CUDA")
    _REGISTERED = True


def _on_cuda() -> bool:
    """Return True if the step runs on a CUDA GPU (JAX's default backend)."""
    return jax.default_backend() in ("gpu", "cuda")


def _ffi_factor(diag: jnp.ndarray, sup_d: jnp.ndarray, sub_d: jnp.ndarray) -> Sequence[jnp.ndarray]:
    ni = diag.shape[-1]
    if ni > CUDA_MAX_NI and _on_cuda():
        raise ValueError(
            f"the CUDA ffi solver handles at most {CUDA_MAX_NI} species per layer "
            f"(this network has ni={ni}); run with VULCAN3_SOLVER=fast"
        )
    _register()
    out = (
        jax.ShapeDtypeStruct(diag.shape, diag.dtype),
        jax.ShapeDtypeStruct(diag.shape[:-1], jnp.int32),
    )
    return jax.ffi.ffi_call("vulcan_bt_factor", out, vmap_method="broadcast_all")(diag, sup_d, sub_d)


def _ffi_solve(
    lu: jnp.ndarray, perm: jnp.ndarray, sup_d: jnp.ndarray, sub_d: jnp.ndarray, rhs: jnp.ndarray
) -> jnp.ndarray:
    # expand_dims, not broadcast_all: at a vmap level where only `rhs` is
    # batched (the tangent directions share one factorisation) the factors get
    # a size-1 axis instead of one copy per direction, and the handler
    # broadcasts their leading dimensions against the rhs's.
    _register()
    return jax.ffi.ffi_call(
        "vulcan_bt_solve",
        jax.ShapeDtypeStruct(rhs.shape, rhs.dtype),
        vmap_method="expand_dims",
    )(lu, perm, sup_d, sub_d, rhs)


if __name__ == "__main__":
    print(build(force="--force" in sys.argv, cuda="--cuda" in sys.argv))
