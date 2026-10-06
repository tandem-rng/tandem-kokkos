# Tests

```sh
pixi run test                         # Serial and OpenMP
pixi run -e gcc test-gcc              # the same with GCC 14 on Linux
pixi run -e cuda test-cuda            # Serial, OpenMP and CUDA on a GPU host
```

## Suite

`tests/test_tandem.cpp` runs every check on each enabled backend: Serial, OpenMP and CUDA. It
checks every vector of the specification and every case of its conformance files (see below),
hashes fills of every type with every kernel and in-kernel scalar draws against the stream
SHA-256 sums of `hashes.json`, compares fills against in-kernel draws at random keys, chunk
lengths, positions, lengths and output alignments, checks that fills split at arbitrary points
with host draws between them continue one stream, and checks mixed-width draws, random access,
derived keys and fork positions. Bounded fills are checked against the contract in `core.hpp`
written out on host generators and against the sequential `urand(range)` calls. Double normal
fills must equal the scalar `normal()` calls bit for bit from random positions, counts and chunk
lengths with every kernel and through `fill_normal` itself. Float normal fills are checked
against the `normalf2()` calls: bit for bit on host backends, to 16 ulps plus 1e-6 on CUDA.
Exponential fills must equal the scalar `exponential()` calls bit for bit with every kernel.
Choice fills of every kernel must equal the `choice()` calls at random keys, chunk lengths,
positions, lengths and output alignments. 1e7 draws of each normal and exponential fill have
the moments and a Kolmogorov-Smirnov statistic of N(0, 1) and Exp(1). It also checks 8- and
16-bit, signed, Float16, `half_t` and complex fills against the u32 stream, and Views of rank 0
to 3 in both layouts. Every non-Serial backend must then write the bytes Serial writes.

## Conformance files

`tests/conformance/*.json` are byte-identical copies of tandem-spec f420545
`conformance/*.json`, read by `tests/conformance.hpp`. Each item of the spec's
`conformance/CHECKLIST.md` has its test, run on every backend:

| checklist section | test |
|---|---|
| Fallback by global draw index | `check_fill_cases` on `fill_below.json` and `normal.json`, `check_shift` on the `_AT[4]`, `_AT[6]` and `CROSS_NORMAL[1]` pairs |
| Width from range | `fill_below` names the width by its element type, checked by the `fill_below.json` cases; `check_range0_and_empty` |
| n = 0 | the seven `n = 0` cases through the public fills, onto a sentinel; `check_range0_and_empty` for uniform fills |
| Odd n | the `CROSS_NORMAL32` cases, values and end |
| Pair rule for Float32 Box-Muller | `CROSS_NORMALF`, `check_shift` on `CROSS_NORMAL32[1]` and `[2]`, `check_scalars` for `normalf()` |
| Weighted choice | `test_choice`: every table of `choice.json` (whole where pinned, else `capacity`), every case, `check_shift` on `CROSS_CHOICE[1]`, `check_scalars` for `choice()`, the `n = 0` case, rejected weights; `m = 1` is the `choice single` case |
| Cut fill | `check_fill_cases` cuts every case of the four fill files at 1, 7, 20, 21 and n − 1, Float32 normals at the even ones; `check_scalars` |
| Block and 2^63 position boundaries | `test_streams`, `test_dumps`, `check_complex_straddle`, `check_random_access`, `check_position_bounds` |

The fill cases run on every path: the public fill and `detail::fill_kind` with each kernel of
the backend, group and chunk on host backends, tile and chunk on CUDA. Every fill writes into a
View one element longer whose last element must keep its sentinel. Values are bit for bit,
except CUDA Float32 normals, which take the case's tolerance. A Float32 normal fill cut at an
odd element drops that piece's last sin half, so it cuts only between pairs. The Float32 normal
dump hash holds for C's polynomials, which host fills take and CUDA's `__sincosf` does not, so
that dump runs on host backends. `hashes.json` has no fill here for its UInt128 and Char
streams. `Kokkos::rand<tandem::Rng, uint64_t>::draw(rng, range)` calls `urand64(range)`, which
names the width.

`tests/vectors.hpp` is generated from the spec repository's `vectors.json` by
`tools/gen_vectors.py`.

## CI

- Ubuntu and macOS with clang, Ubuntu with GCC, and Ubuntu with clang and `-DTANDEM_NO_SIMD`.
  Each builds with `-Wall -Wextra -Werror` and the benchmark, then runs `ctest` with four OpenMP
  threads.
- One job checks that the vector header is current and that the conformance copies equal
  tandem-spec f420545, and one that the tandem-cuda pin is on its main branch.
