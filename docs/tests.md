# Tests

```sh
pixi run test                         # Serial and OpenMP
pixi run -e gcc test-gcc              # the same with GCC 14 on Linux
pixi run -e cuda test-cuda            # Serial, OpenMP and CUDA on a GPU host
```

## Suite

`pixi run test` runs `tests/test_tandem.cpp` on every enabled backend.

- Every vector of the specification, from `tests/vectors.hpp`, made by `tools/gen_vectors.py`.
- Stream dumps from tandem-c in `tests/data` for every type, plus random keys, positions and cuts.
- Bounded, normal and exponential fills against the fixtures of tandem-cuda and tandem-c.
- Exponentials: the moments and a Kolmogorov-Smirnov test of Exp(1) on 1e7 draws.
- Every backend writes the bytes Serial writes.

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

## Fixtures

`tests/vectors.hpp` is made by `tools/gen_vectors.py` from the specification's vectors. The
stream dumps in `tests/data` come from tandem-c. `cross_fill_normal.h` and
`cross_fill_exponential.h` come from tandem-cuda, and `cross_normal.h` and the hashes of
`dump_normals` and `tests/test_exponential_bits.c` from tandem-c.

## CI

- Ubuntu and macOS with clang, Ubuntu with GCC, and Ubuntu with clang and `-DTANDEM_NO_SIMD`.
  Each builds with `-Wall -Wextra -Werror` and the benchmark, then runs `ctest` with four OpenMP
  threads.
- One job checks that the vector header is current, and one that the tandem-cuda pin is on
  its main branch.
