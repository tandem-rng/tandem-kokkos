#!/usr/bin/env bash
set -euo pipefail

cmake -S . -B build -G Ninja ${CMAKE_ARGS} \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
  -DTANDEM_TESTS=ON -DTANDEM_BENCH=OFF
cmake --build build
# The OpenMP backend of a cross-built package cannot run on the build host.
if [[ "${CONDA_BUILD_CROSS_COMPILATION:-0}" != "1" ]]; then
  OMP_PROC_BIND=false ctest --test-dir build --output-on-failure
fi
cmake --install build
