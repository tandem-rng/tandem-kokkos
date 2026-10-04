<p align="center"><img src="assets/lockup.png" width="560" alt="tandem rng .kokkos"></p>

# tandem-kokkos

[![CI](https://github.com/tandem-rng/tandem-kokkos/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/tandem-rng/tandem-kokkos/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-tandem--rng.github.io-7fb3ee.svg)](https://tandem-rng.github.io/tandem-kokkos/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)

[Kokkos](https://kokkos.org) implementation of [Tandem8x32](https://github.com/tandem-rng/spec),
a noncryptographic random number generator. It writes the specification's stream bit for bit
on every Kokkos backend, fast on CPUs and GPUs. Two headers, C++20, no compiled library.

Needs C++20 and Kokkos 5. The submodule `external/tandem-cuda` is pinned at c5c5725.

```sh
git clone --recurse-submodules https://github.com/tandem-rng/tandem-kokkos
```

```cmake
add_subdirectory(tandem-kokkos)        # or find_package(tandem-kokkos) after cmake --install
target_link_libraries(app PRIVATE tandem::kokkos)
```

```cpp
#include <tandem/kokkos.hpp>

tandem::Rng rng(42);                       // 128-bit seed as two halves, default K = 32
Kokkos::View<double *> x("x", n);
tandem::fill(x, rng);                      // the spec's Float64 fill, on x's execution space
tandem::Rng w = rng.split(7);              // child stream from the key alone
tandem::fill_exponential(x, w);            // standard exponentials, -log(1 - u)
```

See [API](docs/api.md) for every fill and `Rng` draw, and [design](docs/design.md),
[tests](docs/tests.md) and [speed](docs/speed.md) for the rest.

Portions of the code were generated with the assistance of LLMs.

[Documentation](https://tandem-rng.github.io/tandem-kokkos/) · [Apache 2.0 license](LICENSE)
