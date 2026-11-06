#!/bin/bash
# Run clang-tidy on the CPU and CUDA kernels. The .cu file holds host code, so it parses only with the toolkit of the
# nvcc on PATH.
set -uo pipefail
status=0
nvcc_path=$(command -v nvcc) || { echo "tidy_kernels: nvcc is not on PATH; add a CUDA toolkit's bin folder" >&2; exit 1; }
toolkit=$(dirname "$(dirname "$nvcc_path")")
[ -d "$toolkit/targets/x86_64-linux" ] && toolkit=$toolkit/targets/x86_64-linux
jax_include=$(python3 -c 'import jax.ffi; print(jax.ffi.include_dir())')
common=(-std=c++17 -DVULCAN_BT_SOURCE_HASH=tidy -isystem "$jax_include")
# clang-tidy's wheel ships no C++ standard library: take the system compiler's include folders.
while read -r dir; do common+=(-isystem "$dir"); done < <(c++ -E -xc++ - -v </dev/null 2>&1 | sed -n '/search starts here:/,/End of search/p' | sed -n 's/^ \(\/.*\)$/\1/p')
clang-tidy --quiet --warnings-as-errors='*' src/vulcan3/csrc/block_thomas_cpu.cc -- -xc++ "${common[@]}" || status=1
clang-tidy --quiet --warnings-as-errors='*' src/vulcan3/csrc/block_thomas_cuda.cu -- -xcuda --cuda-host-only \
  "--cuda-path=$toolkit" --cuda-gpu-arch=sm_90 --no-cuda-version-check -isystem "$toolkit/include" "${common[@]}" || status=1
exit "$status"
