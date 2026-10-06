# API

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
tandem::fill_exponential(g, rng);          // standard exponentials

// Weighted choice: build the alias table on the host, copy it to the device's memory.
const double w[] = {1, 2, 3, 4};
Kokkos::View<uint64_t *> cut("cut", 4);
Kokkos::View<uint32_t *> alias("alias", 4);
auto hcut = Kokkos::create_mirror_view(cut);
auto halias = Kokkos::create_mirror_view(alias);
tandem::ChoiceTable t;
tandem::choice_build(t, w, 4, hcut.data(), halias.data());
Kokkos::deep_copy(cut, hcut);
Kokkos::deep_copy(alias, halias);
t.cut = cut.data();
t.alias = alias.data();
Kokkos::View<uint32_t *> idx("idx", n);
tandem::fill_choice(idx, rng, t);          // index i with probability w[i] / 10

Kokkos::View<float *> y("y", m);
Kokkos::parallel_for(m, KOKKOS_LAMBDA(int i) {
    tandem::Rng r = rng.split(i);          // one generator per work item
    y(i) = r.frand() + r.frand();
});
```

## Reference

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
- `tandem::fill_normal(view, rng)`: standard normals in a `float` or `double` View. Doubles use
  the 1024-layer ziggurat of Appendix A, the sequence of `Rng::normal` calls: element `i` comes
  from draw `i` of the u64 fill, and a draw outside the inner rectangles, 0.43 % of them,
  continues on a fallback stream keyed by its global draw index, so a fill cut at any element
  equals the whole fill. An empty double fill aligns the position to 64. Floats use Box-Muller,
  the flattened sequence of `Rng::normalf2` calls: pair `j`, the elements `2j` and `2j + 1`
  with the cos half first, comes from the draws `2j` and `2j + 1` of the Float32 fill. An odd
  count drops the last sin half and still consumes both draws, and an empty float fill leaves
  the position alone. Both fills have `exec` forms like `fill`.
- `tandem::fill_exponential(view, rng)`: standard exponentials `-log(1 - u)` in a `float` or
  `double` View, the sequence of `Rng::exponentialf` or `Rng::exponential` calls. Element `i`
  comes from draw `i` of the Float32 (Float64) fill, so the fill consumes `n` draws. An empty fill
  leaves the position alone.
- `tandem::choice_build(table, weights, m, cut, alias)` and
  `tandem::fill_choice(view, rng, table)`: weighted choice of Appendix C. `choice_build` builds
  the alias table of `m` Float64 weights on the host in exact integers into the caller's `cut`
  and `alias` arrays of `m` entries, and returns false for no weights, a negative, infinite or
  NaN weight, or weights that are all zero. `fill_choice` writes indices in `[0, m)` to a
  `uint32_t` View: element `i` maps draw `i` of the UInt64 fill through the table, so it
  consumes 64 bits, never retries, and a fill cut at any element equals the whole fill. The
  table's arrays must be readable by the execution space, so a device fill takes copies in
  device memory. An empty fill aligns the position to 64. It has an `exec` form like `fill`.
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
| `normal()`, `normal(mean, sd)` | a ziggurat normal from one UInt64 draw |
| `normal2()` | two ziggurat normals from two UInt64 draws, as a pair `z0`, `z1` |
| `normalf()`, `normalf2()` | the cos half, or both halves, of a Box-Muller step from two Float32 draws |
| `exponential()`, `exponentialf()` | `-log(1 - u)` of one Float64 or Float32 draw |
| `choice(table)` | a weighted choice index from one UInt64 draw |
| `split(i)`, `sub(purpose)`, `fork(children, n)` | child generators as the specification defines them |
| `key()`, `position()`, `set_position(p)`, `chunk_length()` | transport form. `set_position` returns false and changes nothing for `p >= 2^63`, and `from_key` with such a start keeps position 0 |

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

## Parallel use

Element `i` of a fill is draw `i`, so ranks, threads or devices that start at the
position of their first element, or draw from `split(task)`, reproduce a serial run for any
decomposition, as
[Appendix B](https://github.com/tandem-rng/spec/blob/main/SPEC.md#appendix-b-parallel-decomposition-non-normative)
of the specification shows.
