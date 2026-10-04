#include <cstdio>

#include <tandem/kokkos.hpp>

int main(int argc, char **argv) {
    Kokkos::initialize(argc, argv);
    int bad = 0;
    {
        Kokkos::View<double *> x("x", 1000);
        tandem::Rng rng(42);
        tandem::fill(x, rng);
        auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x);
        for (int i = 0; i < 1000; i++) bad += !(h(i) >= 0.0 && h(i) < 1.0);
    }
    Kokkos::finalize();
    std::puts(bad ? "FAIL" : "ok");
    return bad;
}
