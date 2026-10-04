<p align="center"><img src="assets/lockup.png" width="560" alt="tandem rng .kokkos"></p>

# tandem-kokkos

[Kokkos](https://kokkos.org) implementation of [Tandem8x32](https://github.com/tandem-rng/spec),
a noncryptographic random number generator. It writes the specification's stream bit for bit
on every Kokkos backend, fast on CPUs and GPUs. Two headers, C++20, no compiled library.

## Install

```sh
git clone --recurse-submodules https://github.com/tandem-rng/tandem-kokkos
```

```cmake
add_subdirectory(tandem-kokkos)        # or find_package(tandem-kokkos) after cmake --install
target_link_libraries(app PRIVATE tandem::kokkos)
```

Without CMake, add `include/` and `external/tandem-cuda/include/` to the include path.
Needs C++20 and Kokkos 5. `external/tandem-cuda` is a submodule pinned at 5806e51.
`pixi run test` builds Kokkos 5.2.1 with Serial and OpenMP. `pixi run -e cuda test-cuda` builds
Kokkos 5.2.2 with CUDA 12.8 and runs on a GPU host. Recipes for Spack and conda-forge sit in
`packaging/`.

Full notes on the API, kernels, tests and speed: [docs/notes.md](docs/notes.md).

## Use

```cpp
#include <tandem/kokkos.hpp>

tandem::Rng rng(42);                       // 128-bit seed as two halves, default K = 32
Kokkos::View<double *> x("x", n);
tandem::fill(x, rng);                      // the spec's Float64 fill, on x's execution space
double u = rng.drand();                    // continues the stream after the fill
tandem::Rng w = rng.split(7);              // child stream from the key alone
```

```cpp
Kokkos::View<uint32_t *> die("die", n);
tandem::fill_below(die, rng, 6u);          // uniform on [0, 6)
Kokkos::View<float *> g("g", n);
tandem::fill_normal(g, rng);               // standard normals

Kokkos::View<float *> y("y", m);
Kokkos::parallel_for(m, KOKKOS_LAMBDA(int i) {
    tandem::Rng r = rng.split(i);          // one generator per work item
    y(i) = r.frand() + r.frand();
});
```

## What it provides

- `tandem::fill(exec, view, rng)` and `tandem::fill(view, rng)`: contiguous Views of any rank
  and layout, of `bool`, 8 to 64-bit integers, `float`, `double`, `Kokkos::complex` and `half_t`.
  The first form does not fence.
- `tandem::fill_f16_bits(view, rng)`: binary16 bit patterns in a `uint16_t` View.
- `tandem::fill_below(view, rng, range)`: bounded `uint32_t` and `uint64_t` by Lemire's method.
- `tandem::fill_normal(view, rng)`: `float` and `double` normals by Box-Muller.
- `tandem::Rng`: draws in kernels, `bit`, `urand`, `frand`, `drand`, `normal`, `normal2`.
- `Rng::at_urand(i)` and the other `at_` forms: random access without advancing.
- `Rng::split(i)`, `sub(purpose)`, `fork(children, n)`: child generators.
- `Rng::key()`, `position()`, `set_position(p)`: transport form.
- Parallel use: ranks that start at the position of their first element reproduce a serial run.
- No `Kokkos::fill_random` pool: pools hand out states by scheduling and are not reproducible.
- `tandem/core.hpp`: the portable core from tandem-cuda, reusable by other backends.

Bounded and normal draws are not in the specification. Normals agree across backends to a few
ulps. Host backends match tandem-c bit for bit when built with `-ffp-contract=off`, and with
`-mfma` on x86. A non-contiguous View throws `std::invalid_argument`. Where Kokkos has no half
type, as in the conda-forge build for macOS, `half_t` is `float`.

## Tests

`pixi run test` runs `tests/test_tandem.cpp` on every enabled backend.

- Every vector of the specification, from `tests/vectors.hpp`, made by `tools/gen_vectors.py`.
- Stream dumps from tandem-c in `tests/data` for every type, plus random keys, positions and cuts.
- Bounded and normal fills against the fixtures of tandem-cuda and tandem-c.
- Every backend writes the bytes Serial writes.

## Speed

`tools/bench.cpp` (`-DTANDEM_BENCH=ON`): half-second warm-up, minimum of 15 runs, GiB/s written.
Apple M4 Pro, clang 21, Kokkos 5.2.1, OpenMP with 10 threads (`OMP_PROC_BIND=spread`).

| | elements | `uint32_t` | `uint64_t` | `float` | `double` |
|---|---|---|---|---|---|
| Serial, `tandem::fill` | 2^26 | 17.1 | 16.9 | 14.1 | 14.0 |
| Serial, `tandem::fill` | 2^28 | 16.6 | 16.4 | 13.8 | 13.5 |
| Serial, chunk kernel | 2^28 | 7.8 | 7.4 | 7.2 | 7.1 |
| Serial, `Kokkos::fill_random`, `Random_XorShift64_Pool` | 2^28 | 2.4 | 4.4 | 2.6 | 5.1 |
| OpenMP, `tandem::fill` | 2^26 | 110 | 109 | 115 | 106 |
| OpenMP, `tandem::fill` | 2^28 | 107 | 109 | 111 | 109 |
| OpenMP, chunk kernel | 2^28 | 71 | 68 | 67 | 65 |
| OpenMP, `Kokkos::fill_random`, `Random_XorShift64_Pool` | 2^28 | 21 | 37 | 23 | 44 |


NVIDIA A100 40 GB (PCIe), CUDA 12.8, Kokkos 5.2.2 built for `AMPERE80`, GPU idle. Eight
back-to-back fills and one fence.

| | elements | `uint32_t` | `uint64_t` | `float` | `double` |
|---|---|---|---|---|---|
| `tandem::fill`, tile kernel | 2^26 | 1278 | 1313 | 1285 | 1312 |
| `tandem::fill`, tile kernel | 2^28 | 1329 | 1342 | 1339 | 1345 |
| chunk kernel | 2^26 | 1278 | 1298 | 1268 | 1286 |
| chunk kernel | 2^28 | 1309 | 1315 | 1306 | 1312 |
| `Kokkos::fill_random`, `Random_XorShift64_Pool` | 2^26 | 91 | 127 | 92 | 97 |
| `Kokkos::fill_random`, `Random_XorShift64_Pool` | 2^28 | 95 | 130 | 96 | 104 |
| tandem-cuda tile kernel, `cudaEvent` timing, earlier session | 2^28 | 1385 | 1394 | 1379 | 1393 |

The other fills on the same GPU:

| | elements | GiB/s |
|---|---|---|
| `fill`, `uint8_t` | 2^26 | 1093 |
| `fill`, `uint8_t` | 2^28 | 1245 |
| `fill`, `Kokkos::complex<double>` | 2^26 | 1330 |
| `fill`, `Kokkos::complex<double>` | 2^28 | 1349 |
| `fill_below`, `uint32_t` | 2^26 | 1275 |
| `fill_below`, `uint32_t` | 2^28 | 1342 |
| `fill_below`, `uint64_t` | 2^26 | 1309 |
| `fill_below`, `uint64_t` | 2^28 | 1328 |
| `fill_normal`, `float` | 2^26 | 1091 |
| `fill_normal`, `float` | 2^28 | 1100 |
| `fill_normal`, `double` | 2^26 | 674 |
| `fill_normal`, `double` | 2^28 | 686 |

`fill_below` uses range 1000.

## AI assistance

This port was written with the help of large language models under human
direction. The design and the specification are human work, as is much of the
Julia implementation. The code is tested bit for bit against every vector of
the specification and against long stream dumps from the Julia implementation,
and every value must match. The output does not depend on who or what wrote the
code.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
