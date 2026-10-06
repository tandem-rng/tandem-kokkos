# Speed

`tools/bench.cpp` (`-DTANDEM_BENCH=ON`): half-second warm-up, minimum of 15 runs, GiB/s written.

## CPU

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

At 2^24 elements the Serial fill runs within 5% of tandem-c's single-thread fill on the same
machine. The chunk
kernel, one scalar chunk per work item, shows what the vector row buys.

## GPU

NVIDIA A100 40 GB (PCIe), CUDA 12.8, Kokkos 5.2.2 built for `AMPERE80`, `build/cuda/bench cuda
15 8`: eight back-to-back fills and one fence, minimum of 15 runs after a half-second warm-up. The
GPU had no other process, and a second run agreed within 2 %. The cuRAND rows and column are
Philox4x32-10 of cuRAND 10.3.9 in the same run, by the same method, on the execution space's
stream. cuRAND has no 64- or 8-bit integer output for Philox, no bounded integers and no
exponentials, so those cells give the nearest call, marked "nearest": `curandGenerate` into the
same bytes, or the uniform the exponential reads.

| | elements | `uint32_t` | `uint64_t` | `float` | `double` |
|---|---|---|---|---|---|
| `tandem::fill`, tile kernel | 2^26 | 1279 | 1314 | 1282 | 1312 |
| `tandem::fill`, tile kernel | 2^28 | 1332 | 1342 | 1337 | 1344 |
| chunk kernel | 2^26 | 1273 | 1295 | 1270 | 1287 |
| chunk kernel | 2^28 | 1309 | 1315 | 1305 | 1310 |
| `Kokkos::fill_random`, `Random_XorShift64_Pool` | 2^26 | 92 | 127 | 92 | 97 |
| `Kokkos::fill_random`, `Random_XorShift64_Pool` | 2^28 | 95 | 131 | 96 | 105 |
| cuRAND `curandGenerate`, `curandGenerateUniform`, `curandGenerateUniformDouble` | 2^26 | 1256 | 1257, nearest | 1233 | 782 |
| cuRAND `curandGenerate`, `curandGenerateUniform`, `curandGenerateUniformDouble` | 2^28 | 1255 | 1250, nearest | 1238 | 788 |
| tandem-cuda tile kernel, from its [speed page](https://github.com/tandem-rng/tandem-cuda/blob/main/docs/speed.md) | 2^28 | 1377 | 1388 | 1379 | 1388 |

The other fills on the same GPU:

| | elements | GiB/s | cuRAND Philox4x32-10 | cuRAND call |
|---|---|---|---|---|
| `fill`, `uint8_t` | 2^26 | 1087 | 1124 | `curandGenerate`, nearest |
| `fill`, `uint8_t` | 2^28 | 1246 | 1250 | `curandGenerate`, nearest |
| `fill`, `Kokkos::complex<double>` | 2^26 | 1331 | 789 | `curandGenerateUniformDouble` |
| `fill`, `Kokkos::complex<double>` | 2^28 | 1346 | 790 | `curandGenerateUniformDouble` |
| `fill_below`, `uint32_t` | 2^26 | 1271 | 1256 | `curandGenerate`, nearest |
| `fill_below`, `uint32_t` | 2^28 | 1340 | 1255 | `curandGenerate`, nearest |
| `fill_below`, `uint64_t` | 2^26 | 1318 | 1257 | `curandGenerate`, nearest |
| `fill_below`, `uint64_t` | 2^28 | 1362 | 1250 | `curandGenerate`, nearest |
| `fill_normal`, `float` | 2^26 | 1091 | 862 | `curandGenerateNormal` |
| `fill_normal`, `float` | 2^28 | 1114 | 873 | `curandGenerateNormal` |
| `fill_normal`, `double` | 2^26 | 976 | 577 | `curandGenerateNormalDouble` |
| `fill_normal`, `double` | 2^28 | 1017 | 570 | `curandGenerateNormalDouble` |
| `fill_exponential`, `float` | 2^26 | 1119 | 1233 | `curandGenerateUniform`, nearest |
| `fill_exponential`, `float` | 2^28 | 1121 | 1238 | `curandGenerateUniform`, nearest |
| `fill_exponential`, `double` | 2^26 | 925 | 782 | `curandGenerateUniformDouble`, nearest |
| `fill_exponential`, `double` | 2^28 | 942 | 788 | `curandGenerateUniformDouble`, nearest |

`fill_below` uses range 1000.

The bounded fill adds a multiply and a compare per element and stays within 2% of the plain
fill of the same width. Its rejection threshold is computed once per fill: with a division per
element the fills ran at 938 and 620 GiB/s at 2^28. On CUDA the float normal takes the angle through `__sincosf` (shifted by half a turn, within 16 ulps + 1e-6 of the precise step), and a pair or two pairs of one block leave as one 8 or 16-byte store when the output is aligned. The double normal fill is the ziggurat: a team stores the fast path and continues the misses, 0.43 % of draws, from a scratch queue every eight steps. With the slow path inline in the stepping loop it ran at 511 GiB/s. The CUDA normal fills run below tandem-cuda's own kernels on the same GPU (1114 and 1017 GiB/s against 1172 and 1056 at 2^28). The exponential fills run the chunk kernel on devices, because the log makes them compute bound and the tile kernel's write phase then costs more than it gains: 1121 and 942 GiB/s at 2^28, against 1150 and 931 for tandem-cuda. The normal and exponential fills run below the uniforms because the card holds 250 W: their arithmetic lowers its clock until it bounds them, see tandem-cuda's [design](https://github.com/tandem-rng/tandem-cuda/blob/main/docs/design.md). A float normal pair costs a logarithm, a square root, a sine and a cosine and two
draws. The `uint8_t` fill writes one byte per draw and the 2^26 case runs 13% below 2^28.

Timing every fill alone with its own fence lowers the 2^26 figures by up to 3.2% and the 2^28
figures by under 1%. The tile kernel runs 3-4% below the same kernel written in CUDA. The chunk
kernel runs within 1% of tandem-cuda's direct kernel (1306 to 1316 GiB/s).
