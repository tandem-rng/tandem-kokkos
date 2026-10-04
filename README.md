<p align="center"><img src="assets/lockup.png" width="560" alt="tandem rng .kokkos"></p>

# tandem-kokkos

[Kokkos](https://kokkos.org) implementation of [Tandem8x32](https://github.com/tandem-rng/spec),
a noncryptographic pseudorandom number generator built to be fast on CPUs and GPUs alike. Two
headers, C++20, no compiled library. It produces the stream the specification defines, bit for
bit, on every Kokkos backend.

- `tandem::fill(exec, view, rng)` and `tandem::fill(view, rng)`: fill a contiguous
  `Kokkos::View` of any rank and layout with the draws that start at the generator's position,
  as the specification's fill defines, and move the position past them. The value type is
  `bool`, an integer of 8 to 64 bits, `float`, `double`, `Kokkos::complex<float>`,
  `Kokkos::complex<double>` or `Kokkos::Experimental::half_t`. The fill runs in memory order.
  The host generator advances exactly as a CPU fill would, so CPU and GPU draws interleave on one
  stream. The first form does not fence, the second fences.
- `tandem::fill_f16_bits(view, rng)`: the binary16 bit patterns of the Float16 draws in a
  `uint16_t` View. A `uint16_t` View in `fill` gets raw 16-bit draws.
- `tandem::fill_below(view, rng, range)`: uniform integers on `[0, range)` in a `uint32_t` or
  `uint64_t` View, by Lemire's method as `Rng::urand(range)`. Element `i` takes draw `i` of the
  u32 (u64) fill and consumes exactly that draw, so the fill advances the position by 32 n
  (64 n) bits whatever the draws are. A rejected draw retries on a fallback stream, `split(g)`
  of `sub(PURPOSE_BELOW32)` (or `64`), keyed by the global draw index `g`, the aligned start
  position over the width plus `i`. A fill without rejections equals the sequential
  `urand(range)` calls, a fill cut at any element equals the whole fill, and the rare rejection
  costs no coordination.
- `tandem::fill_normal(view, rng)`: standard normals in a `float` or `double` View by
  Box-Muller, the flattened sequence of `Rng::normalf2` or `Rng::normal2` calls. Pair `j`, the
  elements `2j` and `2j + 1` with the cos half first, comes from the draws `2j` and `2j + 1` of
  the Float32 (Float64) fill, one work item per pair. An odd count drops the last sin half and
  still consumes both draws, 64 (128) bits per pair. An empty fill leaves the position alone.
  Both fills have `exec` forms like `fill`.
- `tandem::Rng`: a value type for draws inside kernels. It holds the transport form (128-bit
  key, 64-bit bit position, chunk length `K`) and one cached chunk state, about 80 bytes. Each
  work item takes its own generator with `rng.split(i)` or by position. There is no state pool.
- `tandem/core.hpp`: the specification's building blocks, `Rng` and an eight-lane row in
  portable C++ without Kokkos or CUDA types. It comes from
  [tandem-cuda](https://github.com/tandem-rng/tandem-cuda), a git submodule in
  `external/tandem-cuda`.

## Use

```cpp
#include <tandem/kokkos.hpp>

tandem::Rng rng(42);                       // 128-bit seed as two halves, default K = 32
Kokkos::View<double *> x("x", n);
tandem::fill(x, rng);                      // the spec's Float64 fill, on x's execution space
double u = rng.drand();                    // continues the stream after the fill

Kokkos::View<Kokkos::complex<float> **, Kokkos::LayoutLeft> z("z", 64, 64);
tandem::fill(z, rng);                      // real then imaginary part per value, in memory order

Kokkos::View<uint32_t *> die("die", n);
tandem::fill_below(die, rng, 6u);          // uniform on [0, 6)
Kokkos::View<float *> g("g", n);
tandem::fill_normal(g, rng);               // standard normals

Kokkos::View<float *> y("y", m);
Kokkos::parallel_for(m, KOKKOS_LAMBDA(int i) {
    tandem::Rng r = rng.split(i);          // one generator per work item, from the key alone
    y(i) = r.frand() + r.frand();
});
```

`Rng` draws:

| | |
|---|---|
| `bit()`, `urand()`, `urand64()`, `frand()`, `drand()` | the specification's Bool, UInt32, UInt64, Float32 and Float64 draws |
| `at_urand(i)`, `at_urand64(i)`, `at_frand(i)`, `at_drand(i)` | element `i` of the fill that would start here, without advancing |
| `urand(range)`, `urand64(range)`, `rand(start, end)`, `rand64(start, end)`, `frand(range)`, `drand(start, end)`, ... | bounded draws, uniform by Lemire's multiply and reject |
| `normal()`, `normalf()`, `normal(mean, sd)` | the cos half of a Box-Muller step from two Float64 or Float32 draws |
| `normal2()`, `normalf2()` | both halves of the step as a pair `z0`, `z1` |
| `split(i)`, `sub(purpose)`, `fork(children, n)` | child generators as the specification defines them |
| `key()`, `position()`, `set_position(p)`, `chunk_length()` | transport form |

Signed integers hold the two's complement of the unsigned draw of the same width. A complex
value takes two draws, the real and then the imaginary component. `half_t` needs a Kokkos with a
half type, which the conda-forge build for macOS lacks: there `half_t` is `float` and `fill`
writes Float32 draws. A View that is not contiguous, such as a column of a `LayoutRight` matrix,
throws `std::invalid_argument`.

Bounded and normal draws are not part of the specification, so other ports may produce other
values for them. The bounded and normal fills follow the contract in `core.hpp` and the same
fills in tandem-cuda. Device `log` and `cos` differ from the host's in the last bits, so a
normal fill agrees across backends to a few ulps, not bit for bit. Host backends write the
values of the scalar `normal2()` and `normalf2()` calls exactly, and the same bits as tandem-c and tandem-cuda's host code on every compiler and CPU, because `core.hpp` fuses its multiply-adds explicitly. Build with `-ffp-contract=off`, and with `-mfma` on x86 so the fused operations stay inline. The method names and the `MAX_*` constants follow the Kokkos generators, so
`Kokkos::rand<tandem::Rng, T>::draw(rng, ...)` works.

### No `Kokkos::fill_random` pool

`Kokkos::fill_random(view, pool, range)` calls `pool.get_state()` without an index, and the
Kokkos pools hand out states by thread id or atomic locks. Which state draws which element then
depends on scheduling, so no pool adaptor can make that fill reproducible. `tandem::fill`
replaces it, and in your own kernels `rng.split(i)` keyed by the work item index gives every
item its own stream.

Parallel use: element `i` of a fill is draw `i`, so ranks, threads or devices that start at the
position of their first element, or draw from `split(task)`, reproduce a serial run for any
decomposition, as
[Appendix B](https://github.com/tandem-rng/spec/blob/main/SPEC.md#appendix-b-parallel-decomposition-non-normative)
of the specification shows.

## Kernels

On host execution spaces a work item steps the eight chunks of a group together, a whole
1024-bit row per step. With GCC 12+ or clang the row lives in 128-bit vectors (NEON or
SSE/AVX) and leaves registers by a 4x4 transpose. Define `TANDEM_NO_SIMD` for the scalar loop.
The host normal fills turn the pairs of each row into normals with `normal_block_f64` or `normal_block_f32` from `core.hpp`, a Box-Muller without libm that the compiler vectorizes, so a host fill equals the scalar `normal2()` calls bit for bit on the same machine. On device spaces one thread steps one chunk. For `K >= 8` a team of 256 threads stages eight
steps of 32 groups in 32 KiB of scratch memory and writes them as 512 contiguous bytes per
warp. For smaller `K` each thread stores its blocks directly. Both pick a 16-byte store when the
output's blocks are 16-byte aligned.

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
Kokkos build. The headers need C++20, as Kokkos 5 does.

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

## Install

The `packaging/` directory holds a Spack recipe (`spack/package.py`, with `openmp` and `cuda` variants)
and a conda-forge style recipe (`conda/recipe.yaml`, Serial and OpenMP Kokkos). Neither is submitted to
Spack or conda-forge yet, and both build from the `main` branch.

## Tests

`tests/test_tandem.cpp` runs every check on each enabled backend: Serial, OpenMP and CUDA. It
checks every vector of the specification, compares fills of every type with every kernel and
in-kernel scalar draws against reference stream dumps in `tests/data` (from tandem-c), compares
fills against in-kernel draws at random keys, chunk lengths, positions, lengths and output
alignments, checks that fills split at arbitrary points with host draws between them continue
one stream, checks mixed-width draws, random access, derived keys and fork positions, and checks
the bounded and normal draws. Bounded fills are checked against the contract in `core.hpp`
written out on host generators, against the sequential `urand(range)` calls, against a fill cut in two, and against
fixtures from tandem-cuda that include rejected draws. Normal fills are checked against the
scalar `normal2()` calls from random positions and counts, odd and even, on host spaces bit for bit, against the shared hash of 1e6-pair fills at five starts (the one tandem-c's `dump_normals` and tandem-cuda's `host_core.cpp` produce), and against fixtures from tandem-cuda
(`cross_fill_normal.h` from tandem-cuda and `cross_normal.h` from tandem-c: 16 ulps plus 1e-6 for float on CUDA, 8 ulps on host spaces, 1e-12 relative for double). It also checks 8- and 16-bit, signed, Float16, `half_t` and complex
fills against the stream dumps and the u32 stream, and Views of rank 0 to 3 in both layouts. Every non-Serial backend must then write the bytes Serial writes.
`tests/vectors.hpp` is generated from the spec repository's `vectors.json` by
`tools/gen_vectors.py`, and CI fails when it is out of date. CI runs the tests with Serial and OpenMP on Linux and macOS with clang, on Linux with GCC 14 as a
compatibility check, and once with clang and `TANDEM_NO_SIMD`. GitHub runners have no GPU, so the CUDA
tests run on a GPU host.

## Speed

`tools/bench.cpp` (`-DTANDEM_BENCH=ON`): a half-second warm-up per row, then the minimum of 15
runs. GiB/s written.

Apple M4 Pro, clang 21, Kokkos 5.2.1 from conda-forge, one fenced fill per run. OpenMP with 10
threads (`OMP_PROC_BIND=spread`).

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

At 2^24 elements the Serial fill runs within 5% of tandem-c's single-thread fill on the same
machine. The chunk
kernel, one scalar chunk per work item, shows what the vector row buys.

NVIDIA A100 40 GB (PCIe), CUDA 12.8, Kokkos 5.2.2 built for `AMPERE80`, the GPU idle before each
run. Time per fill over eight back-to-back fills and one fence:

| | elements | `uint32_t` | `uint64_t` | `float` | `double` |
|---|---|---|---|---|---|
| `tandem::fill`, tile kernel | 2^26 | 1278 | 1313 | 1285 | 1312 |
| `tandem::fill`, tile kernel | 2^28 | 1329 | 1342 | 1339 | 1345 |
| chunk kernel | 2^26 | 1278 | 1298 | 1268 | 1286 |
| chunk kernel | 2^28 | 1309 | 1315 | 1306 | 1312 |
| `Kokkos::fill_random`, `Random_XorShift64_Pool` | 2^26 | 91 | 127 | 92 | 97 |
| `Kokkos::fill_random`, `Random_XorShift64_Pool` | 2^28 | 95 | 130 | 96 | 104 |
| tandem-cuda tile kernel, `cudaEvent` timing, earlier session | 2^28 | 1385 | 1394 | 1379 | 1393 |

The other fills on the same GPU, GiB/s written, eight back-to-back fills per run, `tandem::fill_below`
with range 1000:

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

The bounded fill adds a multiply and a compare per element and stays within 2% of the plain
fill of the same width. On CUDA the float normal takes the angle through `__sincosf` (shifted by half a turn, within 16 ulps + 1e-6 of the precise step), and a pair or two pairs of one block leave as one 8 or 16-byte store when the output is aligned. The CUDA normal fills run below tandem-cuda's own kernels on the same GPU (1100 and 686 GiB/s against 1290 and 765 at 2^28), a known gap with no further tuning planned. A normal pair costs a logarithm, a square root, a sine and a cosine and two
draws. The `uint8_t` fill writes one byte per draw and the 2^26 case runs 15% below 2^28.

Timing every fill alone with its own fence lowers the 2^26 figures by up to 3.2% and the 2^28
figures by under 1%. The tile kernel runs 3-4% below the same kernel written in CUDA. The chunk
kernel runs within 1% of tandem-cuda's direct kernel (1309 to 1324 GiB/s).

## Portable core

`tandem/core.hpp` lives in tandem-cuda, which builds its CUDA kernels on it, and depends on
nothing but the C++ standard library. `TANDEM_FN` expands
to `KOKKOS_INLINE_FUNCTION` when Kokkos is included first, to `__host__ __device__ inline` under
nvcc or hipcc, and to `inline` otherwise. It holds `T`, `F`, `F_keyed`, `block`, the stream
position arithmetic, the float mappings including `to_f16_bits`, `Rng` on the `Draws<D>` base
that device generators share, `GenState`, the bounded-fill helpers `below_u32` and `below_u64`
with `PURPOSE_BELOW32` and `PURPOSE_BELOW64`, `box_muller2` and `box_muller2_f32` with the host blocks `normal_block_f64` and `normal_block_f32`, and `Row`, the
eight lanes of a group. HIP and
SYCL ports can reuse it and add only their fill kernels.

## AI assistance

This port was written with the help of large language models under human
direction. The design and the specification are human work, as is much of the
Julia implementation. The code is tested bit for bit against every vector of
the specification and against long stream dumps from the Julia implementation,
and every value must match. The output does not depend on who or what wrote the
code.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
