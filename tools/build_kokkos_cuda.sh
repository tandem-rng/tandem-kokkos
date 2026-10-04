#!/usr/bin/env bash
# Build Kokkos with the Serial, OpenMP and CUDA backends into build/kokkos-cuda/install, for
# the pixi `cuda` environment. KOKKOS_ARCH picks the GPU, Ampere 8.0 (A100) by default.
set -euo pipefail

tag=${KOKKOS_TAG:-5.2.2}
arch=${KOKKOS_ARCH:-AMPERE80}
root=$(cd "$(dirname "$0")/.." && pwd)/build/kokkos-cuda

if [ ! -d "$root/src" ]; then
    git clone --depth 1 --branch "$tag" https://github.com/kokkos/kokkos "$root/src"
fi
export NVCC_WRAPPER_DEFAULT_COMPILER=${CXX:-g++}
cmake -S "$root/src" -B "$root/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$root/install" \
    -DCMAKE_CXX_COMPILER="$root/src/bin/nvcc_wrapper" \
    -DCMAKE_CXX_STANDARD=20 \
    -DCUDAToolkit_ROOT="$CONDA_PREFIX" \
    -DKokkos_ENABLE_SERIAL=ON \
    -DKokkos_ENABLE_OPENMP=ON \
    -DKokkos_ENABLE_CUDA=ON \
    -DKokkos_ARCH_"$arch"=ON \
    -DKokkos_ENABLE_TESTS=OFF
cmake --build "$root/build" --parallel 16
cmake --install "$root/build"
