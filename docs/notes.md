# tandem-kokkos notes

Material moved out of the README.

## Install

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

## What it provides

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
- `tandem::fill_exponential(view, rng)`: standard exponentials `-log(1 - u)` in a `float` or
  `double` View, the sequence of `Rng::exponentialf` or `Rng::exponential` calls. Element `i`
  comes from draw `i` of the Float32 (Float64) fill, so the fill consumes `n` draws. An empty fill
  leaves the position alone.
- `tandem::Rng`: a value type for draws inside kernels. It holds the transport form (128-bit
  key, 64-bit bit position, chunk length `K`) and one cached chunk state, about 80 bytes. Each
  work item takes its own generator with `rng.split(i)` or by position. There is no state pool.
- `tandem/core.hpp`: the specification's building blocks, `Rng` and an eight-lane row in
  portable C++ without Kokkos or CUDA types. It comes from
  [tandem-cuda](https://github.com/tandem-rng/tandem-cuda), a git submodule in
  `external/tandem-cuda`.

`Rng` draws:

| | |
|---|---|
| `bit()`, `urand()`, `urand64()`, `frand()`, `drand()` | the specification's Bool, UInt32, UInt64, Float32 and Float64 draws |
| `at_urand(i)`, `at_urand64(i)`, `at_frand(i)`, `at_drand(i)` | element `i` of the fill that would start here, without advancing |
| `urand(range)`, `urand64(range)`, `rand(start, end)`, `rand64(start, end)`, `frand(range)`, `drand(start, end)`, ... | bounded draws, uniform by Lemire's multiply and reject |
| `normal()`, `normalf()`, `normal(mean, sd)` | the cos half of a Box-Muller step from two Float64 or Float32 draws |
| `normal2()`, `normalf2()` | both halves of the step as a pair `z0`, `z1` |
| `exponential()`, `exponentialf()` | `-log(1 - u)` of one Float64 or Float32 draw |
| `split(i)`, `sub(purpose)`, `fork(children, n)` | child generators as the specification defines them |
| `key()`, `position()`, `set_position(p)`, `chunk_length()` | transport form |

Signed integers hold the two's complement of the unsigned draw of the same width. A complex
value takes two draws, the real and then the imaginary component. `half_t` needs a Kokkos with a
half type, which the conda-forge build for macOS lacks: there `half_t` is `float` and `fill`
writes Float32 draws. A View that is not contiguous, such as a column of a `LayoutRight` matrix,
throws `std::invalid_argument`.

Bounded, normal and exponential draws follow the specification's non-normative Appendix A and
the same fills in tandem-cuda. `core.hpp` computes `log`, `sin` and `cos` with polynomials whose
multiply-adds are explicit fused ones, so every backend writes tandem-c's bits for bounded
integers, double normals and exponentials, and host backends do for float normals. CUDA float
normals take the fast `__sincosf` and agree to 16 ulps + 1e-6. Build with `-ffp-contract=off`, and with `-mfma` on x86 so the fused operations stay inline. The method names and the `MAX_*` constants follow the Kokkos generators, so
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
(`cross_fill_normal.h` from tandem-cuda and `cross_normal.h` from tandem-c: bit for bit, except
float on CUDA at 16 ulps plus 1e-6). Exponential fills are checked against the scalar
`exponential()` calls with every kernel, against `cross_fill_exponential.h` from tandem-cuda and
the hash of tandem-c's `tests/test_exponential_bits.c` on every backend, against a fill cut in
two, and for the raw moments 1, 2, 6, 24 and a Kolmogorov-Smirnov test of Exp(1) on 1e7 draws. It also checks 8- and 16-bit, signed, Float16, `half_t` and complex
fills against the stream dumps and the u32 stream, and Views of rank 0 to 3 in both layouts. Every non-Serial backend must then write the bytes Serial writes.

## Speed

At 2^24 elements the Serial fill runs within 5% of tandem-c's single-thread fill on the same
machine. The chunk
kernel, one scalar chunk per work item, shows what the vector row buys.

The bounded fill adds a multiply and a compare per element and stays within 2% of the plain
fill of the same width. On CUDA the float normal takes the angle through `__sincosf` (shifted by half a turn, within 16 ulps + 1e-6 of the precise step), and a pair or two pairs of one block leave as one 8 or 16-byte store when the output is aligned. The CUDA normal fills run below tandem-cuda's own kernels on the same GPU (1099 and 815 GiB/s against 1270 and 833 at 2^28), a known gap with no further tuning planned. The exponential fills run the chunk kernel on devices, because the log makes them compute bound and the tile kernel's write phase then costs more than it gains: 1023 and 908 GiB/s at 2^28, against 1022 and 948 for tandem-cuda. A normal pair costs a logarithm, a square root, a sine and a cosine and two
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
with `PURPOSE_BELOW32` and `PURPOSE_BELOW64`, `box_muller2` and `box_muller2_f32` with the host blocks `normal_block_f64` and `normal_block_f32`, `exponential_f64` and `exponential_f32`, and `Row`, the
eight lanes of a group. HIP and
SYCL ports can reuse it and add only their fill kernels.
