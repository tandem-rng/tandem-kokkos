# Speed

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
| `fill_normal`, `float` | 2^26 | 1096 |
| `fill_normal`, `float` | 2^28 | 1099 |
| `fill_normal`, `double` | 2^26 | 819 |
| `fill_normal`, `double` | 2^28 | 815 |
| `fill_exponential`, `float` | 2^26 | 1020 |
| `fill_exponential`, `float` | 2^28 | 1023 |
| `fill_exponential`, `double` | 2^26 | 892 |
| `fill_exponential`, `double` | 2^28 | 908 |

`fill_below` uses range 1000.

At 2^24 elements the Serial fill runs within 5% of tandem-c's single-thread fill on the same
machine. The chunk
kernel, one scalar chunk per work item, shows what the vector row buys.

The bounded fill adds a multiply and a compare per element and stays within 2% of the plain
fill of the same width. On CUDA the float normal takes the angle through `__sincosf` (shifted by half a turn, within 16 ulps + 1e-6 of the precise step), and a pair or two pairs of one block leave as one 8 or 16-byte store when the output is aligned. The CUDA normal fills run below tandem-cuda's own kernels on the same GPU (1099 and 815 GiB/s against 1270 and 833 at 2^28), a known gap with no further tuning planned. The exponential fills run the chunk kernel on devices, because the log makes them compute bound and the tile kernel's write phase then costs more than it gains: 1023 and 908 GiB/s at 2^28, against 1022 and 948 for tandem-cuda. A normal pair costs a logarithm, a square root, a sine and a cosine and two
draws. The `uint8_t` fill writes one byte per draw and the 2^26 case runs 15% below 2^28.

Timing every fill alone with its own fence lowers the 2^26 figures by up to 3.2% and the 2^28
figures by under 1%. The tile kernel runs 3-4% below the same kernel written in CUDA. The chunk
kernel runs within 1% of tandem-cuda's direct kernel (1309 to 1324 GiB/s).
