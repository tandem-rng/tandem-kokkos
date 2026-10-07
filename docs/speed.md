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
| `tandem::fill`, tile kernel | 2^26 | 1284 | 1314 | 1286 | 1314 |
| `tandem::fill`, tile kernel | 2^28 | 1328 | 1341 | 1338 | 1345 |
| chunk kernel | 2^26 | 1274 | 1294 | 1274 | 1288 |
| chunk kernel | 2^28 | 1309 | 1311 | 1306 | 1306 |
| `Kokkos::fill_random`, `Random_XorShift64_Pool` | 2^26 | 92 | 127 | 92 | 97 |
| `Kokkos::fill_random`, `Random_XorShift64_Pool` | 2^28 | 95 | 131 | 95 | 105 |
| cuRAND `curandGenerate`, `curandGenerateUniform`, `curandGenerateUniformDouble` | 2^26 | 1261 | 1266, nearest | 1235 | 782 |
| cuRAND `curandGenerate`, `curandGenerateUniform`, `curandGenerateUniformDouble` | 2^28 | 1261 | 1248, nearest | 1240 | 788 |
| tandem-cuda tile kernel, from its [speed page](https://github.com/tandem-rng/tandem-cuda/blob/main/docs/speed.md) | 2^28 | 1377 | 1388 | 1377 | 1389 |

The other fills on the same GPU:

| | elements | GiB/s | cuRAND Philox4x32-10 | cuRAND call |
|---|---|---|---|---|
| `fill`, `uint8_t` | 2^26 | 1091 | 1121 | `curandGenerate`, nearest |
| `fill`, `uint8_t` | 2^28 | 1246 | 1251 | `curandGenerate`, nearest |
| `fill`, `Kokkos::complex<double>` | 2^26 | 1328 | 788 | `curandGenerateUniformDouble` |
| `fill`, `Kokkos::complex<double>` | 2^28 | 1351 | 789 | `curandGenerateUniformDouble` |
| `fill_below`, `uint32_t` | 2^26 | 1272 | 1261 | `curandGenerate`, nearest |
| `fill_below`, `uint32_t` | 2^28 | 1340 | 1261 | `curandGenerate`, nearest |
| `fill_below`, `uint64_t` | 2^26 | 1317 | 1266 | `curandGenerate`, nearest |
| `fill_below`, `uint64_t` | 2^28 | 1361 | 1248 | `curandGenerate`, nearest |
| `fill_normal`, `float` | 2^26 | 1172 | 844 | `curandGenerateNormal` |
| `fill_normal`, `float` | 2^28 | 1203 | 873 | `curandGenerateNormal` |
| `fill_normal`, `double` | 2^26 | 987 | 569 | `curandGenerateNormalDouble` |
| `fill_normal`, `double` | 2^28 | 1006 | 569 | `curandGenerateNormalDouble` |
| `fill_exponential`, `float` | 2^26 | 1134 | 1235 | `curandGenerateUniform`, nearest |
| `fill_exponential`, `float` | 2^28 | 1152 | 1240 | `curandGenerateUniform`, nearest |
| `fill_exponential`, `double` | 2^26 | 938 | 782 | `curandGenerateUniformDouble`, nearest |
| `fill_exponential`, `double` | 2^28 | 928 | 788 | `curandGenerateUniformDouble`, nearest |

`fill_below` uses range 1000. The `float` exponential rows predate tandem-cuda e98daee, whose
two-float logarithm adds 10 f32 operations per draw, and have not been measured since.

The bounded fill adds a multiply and a compare per element and stays within 2% of the plain
fill of the same width. Its rejection threshold is computed once per fill: with a division per
element the fills ran at 938 and 620 GiB/s at 2^28. On CUDA the float normal takes the angle through `__sincosf` (shifted by half a turn, within 16 ulps + 1e-6 of the precise step) and its radius through tandem-cuda's square root without the range check, which took the fill from 1114 to 1203 GiB/s at 2^28, and a pair or two pairs of one block leave as one 8 or 16-byte store when the output is aligned. The double normal fill is the ziggurat: a team stores the fast path and continues the misses, 0.43 % of draws, from a scratch queue every eight steps. With the slow path inline in the stepping loop it ran at 511 GiB/s. The CUDA double normal fill runs below tandem-cuda's own kernels on the same GPU (1006 GiB/s against 1058 at 2^28). The exponential fills run the chunk kernel on devices, because the log makes them compute bound and the tile kernel's write phase then costs more than it gains: 1152 and 928 GiB/s at 2^28, against 1149 and 941 for tandem-cuda. The normal and exponential fills run below the uniforms because the card holds 250 W: their arithmetic lowers its clock until it bounds them, see tandem-cuda's [design](https://github.com/tandem-rng/tandem-cuda/blob/main/docs/design.md). A float normal pair costs a logarithm, a square root, a sine and a cosine and two
draws. The `uint8_t` fill writes one byte per draw and the 2^26 case runs 12% below 2^28.

cuRAND leads the `uint8_t` and `float` exponential rows, where its nearest call does less work:
32-bit words where the fill stores single bytes, and uniforms without the logarithm.

Timing every fill alone with its own fence lowers the 2^26 figures by up to 3.2% and the 2^28
figures by under 1%. The tile kernel runs 3-4% below the same kernel written in CUDA. The chunk
kernel runs within 1% of tandem-cuda's direct kernel (1307 to 1317 GiB/s).
