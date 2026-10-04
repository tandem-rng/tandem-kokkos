# tandem-kokkos documentation

- [API](api.md): `tandem::fill` and the other fills, `tandem::Rng` draws and parallel use.
- [Design](design.md): the kernels and the portable core.
- [Tests](tests.md): what the suite checks.
- [Speed](speed.md): M4 Pro and A100 figures against `Kokkos::fill_random`.

## Build

Header-only. Clone with the submodule:

```sh
git clone --recurse-submodules https://github.com/tandem-rng/tandem-kokkos
git submodule update --init           # in a clone made without it
```

With CMake, as a subdirectory or through `FetchContent` next to an existing Kokkos:

```cmake
add_subdirectory(tandem-kokkos)        # finds Kokkos unless the Kokkos::kokkos target exists
target_link_libraries(app PRIVATE tandem::kokkos)
```

or installed:

```sh
cmake -S . -B build -DKokkos_ROOT=<kokkos prefix> && cmake --install build --prefix <prefix>
```

```cmake
find_package(tandem-kokkos REQUIRED)
target_link_libraries(app PRIVATE tandem::kokkos)
```

Without CMake, add `include/` and `external/tandem-cuda/include/` to the include path of a
Kokkos build. The headers need C++20, as Kokkos 5 does. `external/tandem-cuda` is a
submodule pinned at c5c5725.

`pixi.toml` provides Kokkos 5.2.1, the newest conda-forge build, with the Serial and OpenMP
backends (`pixi run test`), built with clang. The latest release is 5.2.2. `pixi run -e gcc test-gcc` builds
the same with GCC 14 on Linux. The conda-forge package has no CUDA backend, so on a GPU host the `cuda`
environment builds Kokkos 5.2.2 with Serial, OpenMP and CUDA from source into `build/`, with CUDA 12.8,
clang 19 as the nvcc host compiler (`-ccbin`) and GCC 14 for its libstdc++. CUDA 13 needs a newer
driver than the 570 on the GPU host, so the toolkit stays at 12.8:

```sh
pixi run -e cuda kokkos               # KOKKOS_ARCH=AMPERE80 by default
pixi run -e cuda test-cuda
```

## Packaging

The `packaging/` directory holds a Spack recipe (`spack/package.py`, with `openmp` and `cuda` variants)
and a conda-forge style recipe (`conda/recipe.yaml`, Serial and OpenMP Kokkos). Neither is submitted to
Spack or conda-forge yet, and both build from the `main` branch.

## AI assistance

This port was written with the help of large language models under human
direction. The design and the specification are human work, as is much of the
Julia implementation. The code is tested bit for bit against every vector of
the specification and against long stream dumps from the Julia implementation,
and every value must match. The output does not depend on who or what wrote the
code.
