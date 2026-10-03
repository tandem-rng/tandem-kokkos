// Fill throughput: tandem::fill with its default kernel and with the chunk kernel, and
// Kokkos::fill_random with Random_XorShift64_Pool. Each row: a half-second warm-up, then the
// minimum over `runs` runs of the time per fill of `batch` back-to-back fills and one fence.
// A batch of 1 includes the launch and fence latency in every fill.
//
// Usage: bench serial|openmp|cuda [runs] [batch] [log2 n ...]
//        defaults: 15 runs, batch 1, 2^26 and 2^28
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <Kokkos_Random.hpp>
#include <tandem/kokkos.hpp>

using tandem::detail::Kernel;

static int runs = 15, batch = 1;

template <class Exec, class F> static double best_seconds(const Exec &exec, F &&f) {
    Kokkos::Timer warm;
    while (warm.seconds() < 0.5) {
        f();
        exec.fence();
    }
    double best = 1e30;
    for (int i = 0; i < runs; i++) {
        Kokkos::Timer t;
        for (int b = 0; b < batch; b++)
            f();
        exec.fence();
        double s = t.seconds() / batch;
        best = s < best ? s : best;
    }
    return best;
}

static void row(const char *what, const char *type, int lg, size_t size, double s) {
    std::printf("%-32s %-4s 2^%d %8.1f GiB/s\n", what, type, lg,
                (double)((size_t)1 << lg) * size / s / (1u << 30));
}

template <class Exec, class E> static void bench_type(const char *type, int lg) {
    Exec exec;
    Kokkos::View<E *, typename Exec::memory_space> v("v", (size_t)1 << lg);
    tandem::Rng rng(42);
    row("tandem::fill", type, lg, sizeof(E),
        best_seconds(exec, [&] { tandem::fill(exec, v, rng); }));
    row("tandem::fill, chunk kernel", type, lg, sizeof(E),
        best_seconds(exec, [&] { tandem::detail::fill(exec, v, rng, Kernel::Chunk); }));
    Kokkos::Random_XorShift64_Pool<Exec> pool(42);
    E range = std::is_floating_point_v<E> ? E(1) : std::numeric_limits<E>::max();
    row("Kokkos::fill_random, XorShift64", type, lg, sizeof(E),
        best_seconds(exec, [&] { Kokkos::fill_random(exec, v, pool, range); }));
}

template <class Exec> static void bench(const std::vector<int> &logs) {
    std::printf("%s, concurrency %d, minimum of %d runs, %d fills per run\n", Exec::name(),
                Exec().concurrency(), runs, batch);
    for (int lg : logs) {
        bench_type<Exec, uint32_t>("u32", lg);
        bench_type<Exec, uint64_t>("u64", lg);
        bench_type<Exec, float>("f32", lg);
        bench_type<Exec, double>("f64", lg);
    }
}

int main(int argc, char **argv) {
    Kokkos::ScopeGuard guard(argc, argv);
    const char *space = argc > 1 ? argv[1] : "serial";
    if (argc > 2)
        runs = std::atoi(argv[2]);
    if (argc > 3)
        batch = std::atoi(argv[3]);
    std::vector<int> logs;
    for (int i = 4; i < argc; i++)
        logs.push_back(std::atoi(argv[i]));
    if (logs.empty())
        logs = {26, 28};
    if (!std::strcmp(space, "serial"))
        bench<Kokkos::Serial>(logs);
#ifdef KOKKOS_ENABLE_OPENMP
    else if (!std::strcmp(space, "openmp"))
        bench<Kokkos::OpenMP>(logs);
#endif
#ifdef KOKKOS_ENABLE_CUDA
    else if (!std::strcmp(space, "cuda"))
        bench<Kokkos::Cuda>(logs);
#endif
    else {
        std::printf("unknown or disabled execution space: %s\n", space);
        return 1;
    }
}
