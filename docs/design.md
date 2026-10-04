# Design

## Kernels

On host execution spaces a work item steps the eight chunks of a group together, a whole
1024-bit row per step. With GCC 12+ or clang the row lives in 128-bit vectors (NEON or
SSE/AVX) and leaves registers by a 4x4 transpose. Define `TANDEM_NO_SIMD` for the scalar loop.
The host normal fills turn the pairs of each row into normals with `normal_block_f64` or `normal_block_f32` from `core.hpp`, a Box-Muller without libm that the compiler vectorizes, so a host fill equals the scalar `normal2()` calls bit for bit on the same machine. On device spaces one thread steps one chunk. For `K >= 8` a team of 256 threads stages eight
steps of 32 groups in 32 KiB of scratch memory and writes them as 512 contiguous bytes per
warp. For smaller `K` each thread stores its blocks directly. Both pick a 16-byte store when the
output's blocks are 16-byte aligned.

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
