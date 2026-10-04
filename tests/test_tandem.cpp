// Spec vectors, reference stream dumps, fills against in-kernel draws, split fills, derived
// keys, bounded draws, and byte identity across backends, on every enabled execution space.
#include <cmath>
#include <cstdio>
#include <cstring>

#include <random>
#include <string>
#include <vector>

#include <Kokkos_Random.hpp>
#include <tandem/kokkos.hpp>

#include "../external/tandem-cuda/tests/cross_fill_below.h"
#include "../external/tandem-cuda/tests/cross_fill_normal.h"
#include "cross_normal.h"
#include "vectors.hpp"

using tandem::Key;
using tandem::Rng;
using tandem::detail::Kernel;

// A kernel captures the generator by value, so its size is part of the contract.
static_assert(sizeof(Rng) == 80, "Rng must stay 80 bytes");

static long checks, failures;

#define CHECK(cond)                                                                                \
    do {                                                                                           \
        checks++;                                                                                  \
        if (!(cond)) {                                                                             \
            failures++;                                                                            \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                            \
        }                                                                                          \
    } while (0)

static Key key_of(const uint32_t w[4]) { return Key{{w[0], w[1], w[2], w[3]}}; }
static bool words_equal(const uint32_t a[4], const uint32_t b[4]) {
    return std::memcmp(a, b, 16) == 0;
}

template <class E> constexpr unsigned bits_of = tandem::detail::elem<E>::bits;

// Host copies and dumps hold bools as bytes, because std::vector<bool> has no data().
template <class E> using host_t = std::conditional_t<std::is_same_v<E, bool>, uint8_t, E>;

template <class E> KOKKOS_INLINE_FUNCTION E draw(Rng &r) {
    if constexpr (std::is_same_v<E, bool>)
        return r.bit();
    else if constexpr (std::is_same_v<E, uint32_t>)
        return r.urand();
    else if constexpr (std::is_same_v<E, uint64_t>)
        return r.urand64();
    else if constexpr (std::is_same_v<E, float>)
        return r.frand();
    else
        return r.drand();
}

template <class Exec> static std::vector<Kernel> kernels() {
    if (Kokkos::SpaceAccessibility<Kokkos::HostSpace, typename Exec::memory_space>::accessible)
        return {Kernel::Group, Kernel::Chunk};
    return {Kernel::Tile, Kernel::Chunk};
}

static const char *name(Kernel k) {
    return k == Kernel::Group ? "group" : k == Kernel::Tile ? "tile" : "chunk";
}

// Host copy of a fill of n elements from (key, pos, K) on Exec, `shift` elements into the
// allocation so that the output's alignment varies.
template <class Exec, class E>
static std::vector<host_t<E>> device_fill(const Key &key, uint64_t pos, uint32_t K, size_t n,
                                          Kernel kernel, size_t shift = 0,
                                          uint64_t *end = nullptr) {
    Kokkos::View<E *, typename Exec::memory_space> buf("fill", n + 4);
    auto out = Kokkos::subview(buf, Kokkos::pair<size_t, size_t>(shift, shift + n));
    Rng r = Rng::from_key(key, pos, K);
    tandem::detail::fill(Exec(), out, r, kernel);
    Exec().fence();
    if (end)
        *end = r.position();
    auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
    return std::vector<host_t<E>>(host.data(), host.data() + n);
}

// The same for Float16 bit patterns.
template <class Exec>
static std::vector<uint16_t> device_f16_bits(const Key &key, uint64_t pos, uint32_t K, size_t n,
                                             Kernel kernel) {
    Kokkos::View<uint16_t *, typename Exec::memory_space> out("f16", n);
    Rng r = Rng::from_key(key, pos, K);
    tandem::detail::fill_kind<Exec, tandem::detail::f16_bits>(Exec(), out.data(), n, r, kernel);
    Exec().fence();
    auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
    return std::vector<uint16_t>(host.data(), host.data() + n);
}

// Host copy of n scalar draws on Exec, 256 consecutive draws per work item, each work item
// with its own generator placed at its first element.
template <class Exec, class E>
static std::vector<host_t<E>> device_draws(const Key &key, uint64_t pos, uint32_t K, size_t n) {
    Kokkos::View<E *, typename Exec::memory_space> v("draws", n);
    constexpr int64_t per = 256;
    uint64_t p0 = tandem::align_pos(pos, bits_of<E>);
    Kokkos::parallel_for(
        Kokkos::RangePolicy<Exec>(0, (int64_t)((n + per - 1) / per)), KOKKOS_LAMBDA(int64_t i) {
            Rng r = Rng::from_key(key, p0 + (uint64_t)i * per * bits_of<E>, K);
            for (int64_t k = i * per; k < (i + 1) * per && k < (int64_t)v.extent(0); k++)
                v(k) = draw<E>(r);
        });
    auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
    return std::vector<host_t<E>>(host.data(), host.data() + n);
}

template <class A, class B> static size_t first_diff(const A &a, const B &b) {
    size_t i = 0;
    while (i < a.size() && i < b.size() && a[i] == (typename A::value_type)b[i])
        i++;
    return a.size() == b.size() && i == a.size() ? SIZE_MAX : i;
}

// w bits (1 to 64) at the w-aligned stream position q, read from u32 words that start at the
// 32-aligned position base.
static uint64_t bits_at(const std::vector<uint32_t> &words, uint64_t base, uint64_t q, unsigned w) {
    size_t i = (size_t)((q - base) >> 5);
    if (w == 64)
        return words[i] | ((uint64_t)words[i + 1] << 32);
    return (words[i] >> (q & 31u)) & (0xffffffffu >> (32u - w));
}

// ---- Vectors ------------------------------------------------------------------------------

template <class Exec> static void test_vectors() {
    for (const auto &v : VEC_T) {
        uint32_t o[4], h[4];
        std::memcpy(o, v.o, 16);
        std::memcpy(h, v.h, 16);
        tandem::T(o, h);
        CHECK(words_equal(o, v.o_out) && words_equal(h, v.h_out));
    }
    for (const auto &v : VEC_F) {
        uint32_t o[4], h[4];
        tandem::F_keyed(VEC_KEY, v.counter, tandem::DOMAIN_STREAM, tandem::AUX_STREAM, o, h);
        CHECK(words_equal(o, v.o) && words_equal(h, v.h));
    }
    const Key key = key_of(VEC_KEY);
    for (Kernel kernel : kernels<Exec>()) {
        auto u32 = device_fill<Exec, uint32_t>(key, 0, VEC_K, 64, kernel);
        for (const auto &s : VEC_STREAM) {
            CHECK(std::memcmp(&u32[s.first_word], s.words, 16) == 0);
            uint64_t p = s.first_word * 32;
            uint32_t b[4];
            tandem::block(VEC_KEY, tandem::chunk_of(p, VEC_K), tandem::step_of(p, VEC_K), b);
            CHECK(words_equal(b, s.words));
        }
        auto f64 = device_fill<Exec, double>(key, 0, VEC_K, 32, kernel);
        for (const auto &v : VEC_F64)
            CHECK(f64[v.index] == v.value);
        auto f32 = device_fill<Exec, float>(key, 0, VEC_K, 32, kernel);
        for (const auto &v : VEC_F32)
            CHECK(f32[v.index] == v.value);
        auto bits = device_fill<Exec, bool>(key, 0, VEC_K, 129, kernel);
        for (const auto &v : VEC_BOOL)
            CHECK(bits[v.index] == (v.value != 0));
    }

    // Derived keys, computed in a kernel.
    Kokkos::View<uint32_t *[4], typename Exec::memory_space> k("keys", 6);
    Kokkos::parallel_for(
        Kokkos::RangePolicy<Exec>(0, 1), KOKKOS_LAMBDA(int) {
            Rng r = Rng::from_key(key, 0, VEC_K);
            Rng kids[2];
            Rng f = r;
            f.fork(kids, 2);
            const Key ks[6] = {r.split(0).key(), r.split(1).key(), r.sub(7).key(),
                               kids[0].key(),    kids[1].key(),    Rng(VEC_SEED, 0, VEC_K).key()};
            for (int i = 0; i < 6; i++)
                for (int w = 0; w < 4; w++)
                    k(i, w) = ks[i].w[w];
        });
    auto hk = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), k);
    uint32_t got[6][4];
    for (int i = 0; i < 6; i++)
        for (int w = 0; w < 4; w++)
            got[i][w] = hk(i, w);
    CHECK(words_equal(got[0], VEC_SPLIT0));
    CHECK(words_equal(got[1], VEC_SPLIT1));
    CHECK(words_equal(got[2], VEC_PURPOSE7));
    CHECK(words_equal(got[3], VEC_FORK0));
    CHECK(words_equal(got[5], VEC_SEED_KEY));
    // Fork child 1 is the hidden half of the same F as child 0.
    uint32_t o[4], h[4];
    tandem::F_keyed(VEC_KEY, 0, tandem::DOMAIN_FORK, 0, o, h);
    CHECK(words_equal(got[4], h));

    // A fork advances the parent past the current block, from a block boundary to the next.
    Rng r = Rng::from_key(key, 128, VEC_K), kid;
    r.fork(&kid, 1);
    CHECK(r.position() == 256);
    r.fork(&kid, 0);
    CHECK(r.position() == 384);
    // split and sub keep the parent position and give children position 0 at the parent's K.
    Rng s = Rng::from_key(key, 77, 8).split(3);
    CHECK(s.position() == 0 && s.chunk_length() == 8);

    const Key seed_key = Rng(VEC_SEED, 0, VEC_K).key();
    auto sf64 = device_fill<Exec, double>(seed_key, 0, VEC_K, 32, kernels<Exec>()[0]);
    for (const auto &v : VEC_SEED_F64)
        CHECK(sf64[v.index] == v.value);
    auto su32 = device_fill<Exec, uint32_t>(seed_key, 0, VEC_K, 32, kernels<Exec>()[0]);
    for (const auto &v : VEC_SEED_U32)
        CHECK(su32[v.index] == v.value);
}

// ---- Dumps --------------------------------------------------------------------------------

template <class T> static std::vector<T> slurp(const std::string &dir, const char *file) {
    std::string path = dir + "/" + file;
    FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::printf("FAIL cannot open %s\n", path.c_str());
        failures++;
        return {};
    }
    std::fseek(f, 0, SEEK_END);
    size_t len = (size_t)std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<T> v(len / sizeof(T));
    if (std::fread(v.data(), 1, len, f) != len)
        failures++;
    std::fclose(f);
    return v;
}

template <class Exec, class E>
static void check_dump_fill(const std::string &dir, const char *file, const Key &key,
                            uint32_t K) {
    std::vector<host_t<E>> want = slurp<host_t<E>>(dir, file);
    if (want.empty())
        return;
    for (Kernel kernel : kernels<Exec>()) {
        size_t i = first_diff(want, device_fill<Exec, E>(key, 0, K, want.size(), kernel));
        CHECK(i == SIZE_MAX);
        if (i != SIZE_MAX)
            std::printf("  %s: %s fill differs at %zu\n", file, name(kernel), i);
    }
}

template <class Exec, class E>
static void check_dump(const std::string &dir, const char *file, const Key &key, uint32_t K) {
    check_dump_fill<Exec, E>(dir, file, key, K);
    std::vector<host_t<E>> want = slurp<host_t<E>>(dir, file);
    if (want.empty())
        return;
    size_t i = first_diff(want, device_draws<Exec, E>(key, 0, K, want.size()));
    CHECK(i == SIZE_MAX);
    if (i != SIZE_MAX)
        std::printf("  %s: draws differ at %zu\n", file, i);
}

template <class Exec> static void test_dumps(const std::string &dir) {
    const uint32_t k1234[4] = {1, 2, 3, 4};
    const Key k = key_of(k1234), s42 = Rng(42).key();
    check_dump<Exec, uint32_t>(dir, "k1234_K32_u32.bin", k, 32);
    check_dump<Exec, uint64_t>(dir, "k1234_K32_u64.bin", k, 32);
    check_dump<Exec, uint32_t>(dir, "k1234_K8_u32.bin", k, 8);
    check_dump<Exec, double>(dir, "seed42_K32_f64.bin", s42, 32);
    check_dump<Exec, float>(dir, "seed42_K32_f32.bin", s42, 32);
    check_dump<Exec, bool>(dir, "seed42_K32_bool.bin", s42, 32);
    check_dump_fill<Exec, uint8_t>(dir, "seed42_K32_u8.bin", s42, 32);
    check_dump_fill<Exec, Kokkos::complex<float>>(dir, "seed42_K32_c32.bin", s42, 32);
    check_dump_fill<Exec, Kokkos::complex<double>>(dir, "seed42_K32_c64.bin", s42, 32);

    std::vector<uint16_t> f16 = slurp<uint16_t>(dir, "seed42_K32_f16bits.bin");
    for (Kernel kernel : kernels<Exec>())
        CHECK(first_diff(f16, device_f16_bits<Exec>(s42, 0, 32, f16.size(), kernel)) ==
              SIZE_MAX);
}

// ---- Fills against draws at random keys, chunk lengths, positions, lengths, alignments ----

struct Trial {
    Key key;
    uint32_t K;
    uint64_t pos;
    size_t n, shift;
};

static std::vector<Trial> trials(uint64_t seed, int count) {
    std::mt19937_64 gen(seed);
    std::vector<Trial> out;
    for (int t = 0; t < count; t++) {
        Trial x;
        for (auto &w : x.key.w)
            w = (uint32_t)gen();
        x.K = 1u << (gen() % 9);
        x.pos = gen() % (1u << 20);
        x.n = (size_t)(gen() % (t < count * 3 / 4 ? 5000 : 200000));
        x.shift = gen() % 4;
        out.push_back(x);
    }
    // Position 0 and shift 0 put every block of the output on a 16-byte address.
    out.push_back(Trial{out.back().key, 32, 0, 5000, 0});
    return out;
}

template <class Exec, class E> static void check_against_draws(const char *label) {
    for (const Trial &t : trials(2026, 24)) {
        auto want = device_draws<Exec, E>(t.key, t.pos, t.K, t.n);
        for (Kernel kernel : kernels<Exec>()) {
            uint64_t end;
            auto got = device_fill<Exec, E>(t.key, t.pos, t.K, t.n, kernel, t.shift, &end);
            size_t i = first_diff(want, got);
            CHECK(i == SIZE_MAX);
            CHECK(end == tandem::align_pos(t.pos, bits_of<E>) + t.n * bits_of<E>);
            if (i != SIZE_MAX)
                std::printf("  %s %s fill vs draws (K=%u pos=%llu n=%zu shift=%zu) at %zu\n", label,
                            name(kernel), t.K, (unsigned long long)t.pos, t.n, t.shift, i);
        }
    }
}

// A fill split in pieces, with host draws between them, continues one stream: the pieces and
// the draws equal the u32 words of one whole fill.
template <class Exec> static void test_split_fills() {
    const uint32_t kw[4] = {0xdeadbeef, 7, 99, 0x12345678};
    const Key key = key_of(kw);
    for (uint32_t K : {1u, 8u, 32u}) {
        const uint64_t start = 37; // inside row 0
        const uint64_t base = tandem::align_pos(start, 32);
        auto whole = device_fill<Exec, uint32_t>(key, start, K, 40000, Kernel::Chunk);
        Rng r = Rng::from_key(key, start, K);
        Kokkos::View<uint32_t *, typename Exec::memory_space> a("a", 1001);
        Kokkos::View<uint64_t *, typename Exec::memory_space> b("b", 5003);
        Kokkos::View<double *, typename Exec::memory_space> c("c", 3333);
        std::vector<std::pair<uint64_t, unsigned>> at; // stream position and width of each value
        std::vector<uint64_t> got;
        bool ok = true;

        auto take = [&](auto view) {
            using E = typename decltype(view)::non_const_value_type;
            uint64_t p = tandem::align_pos(r.position(), bits_of<E>);
            tandem::fill(Exec(), view, r);
            auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view);
            for (size_t i = 0; i < view.extent(0); i++) {
                at.push_back({p + i * bits_of<E>, bits_of<E> == 32 ? 32u : 64u});
                if constexpr (std::is_same_v<E, double>) {
                    uint64_t raw = bits_at(whole, base, p + i * 64, 64);
                    ok = ok && h(i) == tandem::to_f64(raw);
                    got.push_back(raw);
                } else {
                    got.push_back((uint64_t)h(i));
                }
            }
        };
        take(a);
        for (int i = 0; i < 3; i++) {
            at.push_back({tandem::align_pos(r.position(), 32), 32});
            got.push_back(r.urand());
        }
        take(b);
        at.push_back({tandem::align_pos(r.position(), 64), 64});
        got.push_back(r.urand64());
        take(c);
        for (size_t i = 0; i < got.size(); i++)
            ok = ok && got[i] == bits_at(whole, base, at[i].first, at[i].second);
        CHECK(ok);
        CHECK(at.back().first + 64 <= base + 32 * whole.size());
    }
}

// Mixed widths through one in-kernel generator read the stream at the aligned positions.
template <class Exec> static void test_mixed_draws() {
    const uint32_t kw[4] = {9, 8, 7, 6};
    const Key key = key_of(kw);
    constexpr int rounds = 2000;
    Kokkos::View<uint64_t *, typename Exec::memory_space> v("mixed", 5 * rounds + 1);
    Kokkos::parallel_for(
        Kokkos::RangePolicy<Exec>(0, 1), KOKKOS_LAMBDA(int) {
            Rng r = Rng::from_key(key, 5, 32);
            for (int i = 0; i < rounds; i++) {
                v(5 * i + 0) = r.bit();
                v(5 * i + 1) = r.urand();
                double d = r.drand();
                std::memcpy(&v(5 * i + 2), &d, 8);
                v(5 * i + 3) = r.urand64();
                float f = r.frand();
                uint32_t fb;
                std::memcpy(&fb, &f, 4);
                v(5 * i + 4) = fb;
            }
            v(5 * rounds) = r.position();
        });
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
    auto words = device_fill<Exec, uint32_t>(key, 0, 32, 5 * rounds * 6, kernels<Exec>()[0]);
    uint64_t p = 5;
    bool ok = true;
    auto next = [&](unsigned w) {
        p = tandem::align_pos(p, w);
        uint64_t x = bits_at(words, 0, p, w);
        p += w;
        return x;
    };
    for (int i = 0; i < rounds; i++) {
        ok = ok && h(5 * i + 0) == next(1);
        ok = ok && h(5 * i + 1) == next(32);
        double d = tandem::to_f64(next(64));
        ok = ok && std::memcmp(&h(5 * i + 2), &d, 8) == 0;
        ok = ok && h(5 * i + 3) == next(64);
        float f = tandem::to_f32((uint32_t)next(32));
        uint32_t fb;
        std::memcpy(&fb, &f, 4);
        ok = ok && h(5 * i + 4) == fb;
    }
    CHECK(ok);
    CHECK(h(5 * rounds) == p);

    // Random access reads the fill that would start at the position.
    Rng r = Rng::from_key(key, 5, 32);
    CHECK(r.at_urand(0) == words[1] && r.at_urand(1000) == words[1001]);
    CHECK(r.at_urand64(100) == bits_at(words, 0, 64 + 6400, 64));
    CHECK(r.at_drand(3) == tandem::to_f64(bits_at(words, 0, 64 + 192, 64)));
    CHECK(r.at_frand(7) == tandem::to_f32(words[8]));
}

// ---- Bounded and normal draws -------------------------------------------------------------

template <class Exec> static void test_bounded() {
    const Key key = Rng(7).key();
    constexpr int n = 60000;
    // Counts of urand(6), and violations of: bounds, the power-of-two identity
    // urand(2^k) = urand() >> (32 - k), its 64-bit analogue, and the signed ranges.
    Kokkos::View<int64_t[8], typename Exec::memory_space> c("counts");
    Kokkos::View<double[2], typename Exec::memory_space> m("moments");
    Kokkos::parallel_for(
        Kokkos::RangePolicy<Exec>(0, 1), KOKKOS_LAMBDA(int) {
            Rng r = Rng::from_key(key, 0, 32);
            Rng a = r, b = r;
            int64_t bad = 0;
            for (int i = 0; i < n; i++) {
                c(r.urand(6u))++;
                uint32_t big = r.urand(0x80000001u);
                uint64_t big64 = r.urand64(0x8000000000000001ull);
                int32_t s = r.rand(-5, 5);
                int64_t s64 = r.rand64(-3, 1000);
                bad += big > 0x80000000u || big64 > 0x8000000000000000ull || s < -5 || s >= 5 ||
                       s64 < -3 || s64 >= 1000;
                bad += a.urand(1u << 5) != b.urand() >> 27;
                bad += a.urand64(1ull << 40) != b.urand64() >> 24;
            }
            c(6) = bad;
            double s1 = 0, s2 = 0;
            for (int i = 0; i < n; i++) {
                double z = r.normal();
                s1 += z;
                s2 += z * z;
            }
            m(0) = s1 / n;
            m(1) = s2 / n;
        });
    auto hc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), c);
    auto hm = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), m);
    double chi2 = 0;
    for (int k = 0; k < 6; k++)
        chi2 += std::pow(hc(k) - n / 6.0, 2) / (n / 6.0);
    CHECK(chi2 < 20.5); // 5 degrees of freedom, p = 0.001
    CHECK(hc(6) == 0);
    // Mean and second moment of a standard normal, within five standard errors.
    CHECK(std::abs(hm(0)) < 5 / std::sqrt((double)n));
    CHECK(std::abs(hm(1) - 1) < 5 * std::sqrt(2.0 / n));

    // Kokkos::rand<Generator, T> drives Rng through the method names of the Kokkos generators.
    using KR = Kokkos::rand<Rng, double>;
    using KF = Kokkos::rand<Rng, float>;
    using KU = Kokkos::rand<Rng, uint32_t>;
    using KI = Kokkos::rand<Rng, int64_t>;
    using KU64 = Kokkos::rand<Rng, uint64_t>;
    Rng a = Rng::from_key(key, 0, 32), b = a;
    CHECK(KR::draw(a) == b.drand());
    CHECK(KF::draw(a, 2.0f) == b.frand(2.0f));
    CHECK(KU::draw(a, 10u) == b.urand(10u));
    CHECK(KI::draw(a, -4, 9) == b.rand64(-4, 9));
    CHECK(KU64::draw(a) == b.urand64());
}

// ---- Bounded and normal fills --------------------------------------------------------------

// The contract in core.hpp, written out on host generators: element i takes the draw at its own
// index, and a rejected draw retries on the draws of split(g) of sub(PURPOSE_BELOW) of the
// fill's generator at position 0, g being the global draw index: the aligned start position over
// the width plus i.
static uint32_t ref_below32(const Key &key, uint32_t K, uint32_t u, uint32_t range, uint64_t e) {
    uint64_t m = (uint64_t)u * range;
    if ((uint32_t)m < range) {
        uint32_t t = (0u - range) % range;
        if ((uint32_t)m < t) {
            Rng f = Rng::from_key(key, 0, K).sub(tandem::PURPOSE_BELOW32).split(e);
            do
                m = (uint64_t)f.urand() * range;
            while ((uint32_t)m < t);
        }
    }
    return (uint32_t)(m >> 32);
}

static uint64_t ref_below64(const Key &key, uint32_t K, uint64_t x, uint64_t range, uint64_t e) {
    unsigned __int128 m = (unsigned __int128)x * range;
    if ((uint64_t)m < range) {
        uint64_t t = (0u - range) % range;
        if ((uint64_t)m < t) {
            Rng f = Rng::from_key(key, 0, K).sub(tandem::PURPOSE_BELOW64).split(e);
            do
                m = (unsigned __int128)f.urand64() * range;
            while ((uint64_t)m < t);
        }
    }
    return (uint64_t)(m >> 64);
}

template <class E> static E ref_below(const Key &key, uint32_t K, E draw, E range, uint64_t e) {
    if constexpr (sizeof(E) == 4)
        return ref_below32(key, K, draw, range, e);
    else
        return ref_below64(key, K, draw, range, e);
}

// Host copy of a bounded fill of n elements on Exec with an explicit kernel.
template <class Exec, class E>
static std::vector<E> device_below(const Key &key, uint64_t pos, uint32_t K, size_t n, E range,
                                   Kernel kernel, size_t shift, uint64_t *end) {
    using Kind = std::conditional_t<sizeof(E) == 4, tandem::detail::below32,
                                    tandem::detail::below64>;
    Kokkos::View<E *, typename Exec::memory_space> buf("below", n + 4);
    auto out = Kokkos::subview(buf, Kokkos::pair<size_t, size_t>(shift, shift + n));
    Rng r = Rng::from_key(key, pos, K);
    tandem::detail::fill_kind<Exec, Kind>(Exec(), out.data(), n, r, kernel, range);
    Exec().fence();
    *end = r.position();
    auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
    return std::vector<E>(host.data(), host.data() + n);
}

// Ranges 0 and 1, small ranges, 2^31 + 1 and the maximum, which reject often.
template <class Exec, class E> static void check_below(const char *label) {
    const E ranges[] = {0, 1, 3, 1000, (E)1 << (8 * sizeof(E) - 1) | 1u, (E) ~(E)0};
    for (const Trial &t : trials(41, 12)) {
        for (E range : ranges) {
            uint64_t p0 = tandem::align_pos(t.pos, 8 * sizeof(E));
            auto draws = device_fill<Exec, E>(t.key, t.pos, t.K, t.n, kernels<Exec>()[0]);
            std::vector<E> want(t.n);
            for (size_t i = 0; i < t.n; i++)
                want[i] = ref_below<E>(t.key, t.K, draws[i], range, p0 / (8 * sizeof(E)) + i);
            for (Kernel kernel : kernels<Exec>()) {
                uint64_t end;
                auto got = device_below<Exec, E>(t.key, t.pos, t.K, t.n, range, kernel, t.shift,
                                                 &end);
                CHECK(first_diff(want, got) == SIZE_MAX);
                CHECK(end == p0 + t.n * 8 * sizeof(E));
                if (first_diff(want, got) != SIZE_MAX)
                    std::printf("  %s %s below(%llu) (K=%u pos=%llu n=%zu shift=%zu) at %zu\n",
                                label, name(kernel), (unsigned long long)range, t.K,
                                (unsigned long long)t.pos, t.n, t.shift, first_diff(want, got));
            }
        }
    }
}

// A fill cut at an arbitrary element boundary equals the whole fill, at a start with
// rejections: the range 2^31 + 1 rejects about half of the u32 draws.
template <class Exec, class E> static void check_below_cut() {
    const Key key = Rng(13).key();
    constexpr size_t n = 3001;
    const E range = (E)1 << (8 * sizeof(E) - 1) | 1u;
    Kokkos::View<E *, typename Exec::memory_space> whole("whole", n), a("a", 1234), b("b", n - 1234);
    Rng r = Rng::from_key(key, 77, 32), w = r;
    tandem::fill_below(Exec(), whole, w, range);
    tandem::fill_below(Exec(), a, r, range);
    tandem::fill_below(Exec(), b, r, range);
    Exec().fence();
    auto hw = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), whole);
    auto ha = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), a);
    auto hb = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b);
    bool same = r.position() == w.position();
    for (size_t i = 0; i < n; i++)
        same = same && hw(i) == (i < 1234 ? ha(i) : hb(i - 1234));
    CHECK(same);
}

// Without a rejection a bounded fill equals the sequential urand(range) calls, and fixtures from
// tandem-cuda pin the fallback stream: 41 of the 2^31 + 1 elements and 34 of the 2^63 + 1
// elements reject.
template <class Exec> static void test_below() {
    check_below<Exec, uint32_t>("u32");
    check_below<Exec, uint64_t>("u64");
    check_below_cut<Exec, uint32_t>();
    check_below_cut<Exec, uint64_t>();

    const Key key = Rng(11).key();
    uint64_t end0;
    auto got = device_below<Exec, uint32_t>(key, 5, 32, 5000, 1000u, kernels<Exec>()[0], 0,
                                            &end0);
    Rng r = Rng::from_key(key, 5, 32);
    bool same = true;
    for (uint32_t v : got)
        same = same && v == r.urand(1000u);
    CHECK(same);

    CHECK(words_equal(CROSS_FILL_KEY, Rng(42).key().w));
    const Key k42 = Rng(42).key();
    for (Kernel kernel : kernels<Exec>()) {
        uint64_t end;
        for (const auto &f : CROSS_BELOW32) {
            auto v = device_below<Exec, uint32_t>(k42, 0, 32, 64, f.range, kernel, 0, &end);
            CHECK(std::memcmp(v.data(), f.out, sizeof f.out) == 0 && end == 64 * 32);
        }
        for (const auto &f : CROSS_BELOW64) {
            auto v = device_below<Exec, uint64_t>(k42, 0, 32, 64, f.range, kernel, 0, &end);
            CHECK(std::memcmp(v.data(), f.out, sizeof f.out) == 0 && end == 64 * 64);
        }
    }
}

// Normals agree across devices to a few ulps. The host's polynomial differs from the device's
// sincospi near a zero of cos or sin by about 1e-15 in absolute terms, which the absolute floor
// covers.
template <class E> static bool near_normal(E got, E want, float ulps = 16.0f) {
    if constexpr (std::is_same_v<E, double>)
        return std::abs(got - want) <= 1e-12 * std::abs(want) + 1e-14;
    else
        return std::abs(got - want) <= ulps * 0x1p-23f * std::abs(want) + 1e-6f;
}

// A host fill runs core.hpp's explicit-fma Box-Muller, which equals the scalar normal2() calls
// and the other host fills bit for bit. A device differs in the last bits.
template <class Exec, class E> static bool same_normal(E got, E want) {
    if constexpr (tandem::detail::is_host<Exec>)
        return std::memcmp(&got, &want, sizeof got) == 0;
    else
        return near_normal(got, want);
}

template <class Exec, class E>
static std::vector<E> device_normal(const Key &key, uint64_t pos, uint32_t K, size_t n,
                                    size_t shift, uint64_t *end) {
    Kokkos::View<E *, typename Exec::memory_space> buf("normal", n + 4);
    auto out = Kokkos::subview(buf, Kokkos::pair<size_t, size_t>(shift, shift + n));
    Rng r = Rng::from_key(key, pos, K);
    tandem::fill_normal(Exec(), out, r);
    Exec().fence();
    *end = r.position();
    auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
    return std::vector<E>(host.data(), host.data() + n);
}

// The flattened normal2 calls: an odd count drops the last sin half and still consumes both
// draws.
template <class E> static std::vector<E> sequential_normals(Rng &r, size_t n) {
    std::vector<E> z(n);
    for (size_t i = 0; i < n; i += 2) {
        auto pair = [&] {
            if constexpr (std::is_same_v<E, double>)
                return r.normal2();
            else
                return r.normalf2();
        }();
        z[i] = pair.z0;
        if (i + 1 < n)
            z[i + 1] = pair.z1;
    }
    return z;
}

template <class Exec, class E> static void check_normal(const char *label) {
    for (const Trial &t : trials(42, 24)) {
        Rng r = Rng::from_key(t.key, t.pos, t.K);
        std::vector<E> want = sequential_normals<E>(r, t.n);
        uint64_t end;
        auto got = device_normal<Exec, E>(t.key, t.pos, t.K, t.n, t.shift, &end);
        bool ok = true;
        for (size_t i = 0; i < t.n; i++)
            ok = ok && same_normal<Exec>(got[i], want[i]);
        CHECK(ok);
        CHECK(end == r.position());
        if (!ok)
            std::printf("  %s normal (K=%u pos=%llu n=%zu shift=%zu)\n", label, t.K,
                        (unsigned long long)t.pos, t.n, t.shift);
    }
}

// The fixtures predate the explicit-fma core, so a host fill matches them to a tolerance only.
// Fixtures from tandem-cuda at positions that put the first pair at an even and an odd draw,
// with an odd count, and empty fills, which leave the position alone.
template <class Exec> static void test_normal() {
    check_normal<Exec, double>("f64");
    check_normal<Exec, float>("f32");

    CHECK(words_equal(CROSS_FILL_KEY, Rng(42).key().w));
    const Key k42 = Rng(42).key();
    for (const auto &f : CROSS_NORMAL64) {
        uint64_t end;
        auto v = device_normal<Exec, double>(k42, f.pos, 32, f.n, 0, &end);
        bool ok = true;
        for (unsigned i = 0; i < f.n; i++)
            ok = ok && near_normal(v[i], f.out[i]);
        CHECK(ok);
        CHECK(end == tandem::align_pos(f.pos, 64) + 128 * ((f.n + 1) / 2));
    }
    for (const auto &f : CROSS_NORMAL32) {
        uint64_t end;
        auto v = device_normal<Exec, float>(k42, f.pos, 32, f.n, 0, &end);
        bool ok = true;
        for (unsigned i = 0; i < f.n; i++)
            ok = ok && near_normal(v[i], f.out[i]);
        CHECK(ok);
        CHECK(end == tandem::align_pos(f.pos, 32) + 64 * ((f.n + 1) / 2));
    }

    // tandem-c's pair fixtures, after one Bool draw. Host spaces agree with its libm to 8 ulps.
    constexpr size_t m = 2 * CROSS_NORMAL_COUNT;
    uint64_t end;
    auto d = device_normal<Exec, double>(k42, 1, 32, m, 0, &end);
    bool ok = end == CROSS_NORMAL_END_POS;
    for (size_t i = 0; i < m; i++)
        ok = ok && near_normal(d[i], CROSS_NORMAL[i]);
    CHECK(ok);
    auto f = device_normal<Exec, float>(k42, 1, 32, m, 0, &end);
    ok = end == CROSS_NORMALF_END_POS;
    for (size_t i = 0; i < m; i++)
        ok = ok && near_normal(f[i], CROSS_NORMALF[i], tandem::detail::is_host<Exec> ? 8.0f : 16.0f);
    CHECK(ok);

    device_normal<Exec, double>(k42, 1, 32, 0, 0, &end);
    CHECK(end == 1);
    device_normal<Exec, float>(k42, 1, 32, 0, 0, &end);
    CHECK(end == 1);
    Kokkos::View<uint32_t *, typename Exec::memory_space> none("none", 0);
    Rng r = Rng::from_key(k42, 1, 32);
    tandem::fill_below(Exec(), none, r, 7u);
    CHECK(r.position() == 1);
}

// The hash tandem-c's tools/dump_normals and tandem-cuda's host_core.cpp print for the same fills:
// one bit pattern on every compiler and port.
template <class Exec> static void test_normal_bits() {
    if constexpr (tandem::detail::is_host<Exec>) {
        constexpr size_t n = 2 * 1000000 - 1;
        uint64_t h = 0xcbf29ce484222325ull;
        auto fnv = [&h](const void *p, size_t bytes) {
            for (size_t i = 0; i < bytes; i++)
                h = (h ^ static_cast<const unsigned char *>(p)[i]) * 0x100000001b3ull;
        };
        Kokkos::View<double *, typename Exec::memory_space> d("d", n);
        Kokkos::View<float *, typename Exec::memory_space> f("f", n);
        for (uint64_t start : {0ull, 1ull, 77ull, 12345ull, 1ull << 30}) {
            Rng r(2026, 7, 0);
            r.set_position(start);
            tandem::fill_normal(Exec(), d, r);
            tandem::fill_normal(Exec(), f, r);
            Exec().fence();
            fnv(d.data(), n * sizeof(double));
            fnv(f.data(), n * sizeof(float));
        }
        CHECK(h == 0x9414e1315e2653beull);
    }
}

// ---- Value types ---------------------------------------------------------------------------

// Fills of narrow and signed integers take the bits of the u32 stream at their own alignment.
template <class Exec, class E> static void check_narrow(const char *label) {
    for (const Trial &t : trials(31, 12)) {
        constexpr unsigned w = bits_of<E>;
        uint64_t p0 = tandem::align_pos(t.pos, w);
        auto words = device_fill<Exec, uint32_t>(t.key, 0, t.K, (p0 + t.n * w) / 32 + 2,
                                                 Kernel::Chunk);
        std::vector<E> want(t.n);
        for (size_t i = 0; i < t.n; i++)
            want[i] = (E)bits_at(words, 0, p0 + i * w, w);
        for (Kernel kernel : kernels<Exec>()) {
            uint64_t end;
            auto got = device_fill<Exec, E>(t.key, t.pos, t.K, t.n, kernel, t.shift, &end);
            CHECK(first_diff(want, got) == SIZE_MAX);
            CHECK(end == p0 + t.n * w);
            if (first_diff(want, got) != SIZE_MAX)
                std::printf("  %s %s fill (K=%u pos=%llu n=%zu shift=%zu)\n", label, name(kernel),
                            t.K, (unsigned long long)t.pos, t.n, t.shift);
        }
    }
}

// A signed fill holds the two's complement of the unsigned fill of the same width.
template <class Exec, class S, class U> static void check_signed() {
    for (const Trial &t : trials(32, 6)) {
        auto u = device_fill<Exec, U>(t.key, t.pos, t.K, t.n, kernels<Exec>()[0], t.shift);
        auto s = device_fill<Exec, S>(t.key, t.pos, t.K, t.n, kernels<Exec>()[0], t.shift);
        CHECK(std::memcmp(u.data(), s.data(), t.n * sizeof(S)) == 0);
    }
}

// The binary16 bits decode to (raw >> 5) * 2^-11, checked by decoding the fields by hand, and
// a half_t View holds the same bits where Kokkos has a half type, which it lacks on some hosts.
template <class Exec> static void test_f16() {
    for (const Trial &t : trials(33, 8)) {
        auto raw = device_fill<Exec, uint16_t>(t.key, t.pos, t.K, t.n, kernels<Exec>()[0]);
        for (Kernel kernel : kernels<Exec>()) {
            auto bits = device_f16_bits<Exec>(t.key, t.pos, t.K, t.n, kernel);
            bool ok = true;
            for (size_t i = 0; i < t.n; i++) {
                unsigned e = bits[i] >> 10 & 31u, m = bits[i] & 1023u;
                double v = bits[i] == 0 ? 0.0 : std::ldexp(1.0 + m / 1024.0, (int)e - 15);
                ok = ok && !(bits[i] >> 15) && e != 31 && v == (raw[i] >> 5) * 0x1p-11;
            }
            CHECK(ok);
        }
#if !KOKKOS_HALF_T_IS_FLOAT
        Kokkos::View<Kokkos::Experimental::half_t *, typename Exec::memory_space> h("h", t.n);
        Rng r = Rng::from_key(t.key, t.pos, t.K);
        tandem::fill(Exec(), h, r);
        Exec().fence();
        CHECK(r.position() == tandem::align_pos(t.pos, 16) + t.n * 16);
        auto hh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), h);
        auto bits = device_f16_bits<Exec>(t.key, t.pos, t.K, t.n, kernels<Exec>()[0]);
        bool ok = true;
        for (size_t i = 0; i < t.n; i++) {
            uint16_t held;
            std::memcpy(&held, &hh(i), 2);
            ok = ok && (float)hh(i) == (float)(raw[i] >> 5) * 0x1p-11f && held == bits[i];
        }
        CHECK(ok);
#endif
    }
}

// A complex value takes two draws, the real and then the imaginary component.
template <class Exec, class T> static void check_complex() {
    for (const Trial &t : trials(34, 8)) {
        auto parts = device_fill<Exec, T>(t.key, t.pos, t.K, 2 * t.n, kernels<Exec>()[0]);
        for (Kernel kernel : kernels<Exec>()) {
            uint64_t end;
            auto c = device_fill<Exec, Kokkos::complex<T>>(t.key, t.pos, t.K, t.n, kernel,
                                                           t.shift, &end);
            CHECK(std::memcmp(c.data(), parts.data(), t.n * sizeof(Kokkos::complex<T>)) == 0);
            CHECK(end == tandem::align_pos(t.pos, 8 * sizeof(T)) + 16 * sizeof(T) * t.n);
        }
    }
}

// Views of any rank and layout fill in memory order, and a strided View is refused.
template <class Exec> static void test_ranks() {
    const Key key = Rng(5).key();
    constexpr size_t a = 7, b = 11, c = 13, n = a * b * c;
    auto want = device_fill<Exec, double>(key, 3, 32, n, kernels<Exec>()[0]);
    auto host_copy = [](auto v) {
        auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
        return std::vector<double>(h.data(), h.data() + h.span());
    };
    Kokkos::View<double ***, Kokkos::LayoutRight, typename Exec::memory_space> r3("r3", a, b, c);
    Kokkos::View<double ***, Kokkos::LayoutLeft, typename Exec::memory_space> l3("l3", a, b, c);
    Kokkos::View<double, Kokkos::LayoutRight, typename Exec::memory_space> r0("r0");
    Rng g = Rng::from_key(key, 3, 32), h = g;
    tandem::fill(Exec(), r3, g);
    tandem::fill(Exec(), l3, h);
    Exec().fence();
    CHECK(first_diff(want, host_copy(r3)) == SIZE_MAX);
    CHECK(first_diff(want, host_copy(l3)) == SIZE_MAX);
    CHECK(g.position() == h.position() && g.position() == 64 + 64 * n);
    tandem::fill(Exec(), r0, g);
    Exec().fence();
    CHECK(g.position() == 64 + 64 * (n + 1));

    Kokkos::View<double **, Kokkos::LayoutRight, typename Exec::memory_space> m("m", 8, 8);
    auto column = Kokkos::subview(m, Kokkos::ALL, 2);
    bool threw = false;
    try {
        tandem::fill(Exec(), column, g);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    CHECK(threw);
}

// ---- Backends -----------------------------------------------------------------------------

template <class Exec> static void run(const char *space, const std::string &dir) {
    long c0 = checks, f0 = failures;
    test_vectors<Exec>();
    test_dumps<Exec>(dir);
    check_against_draws<Exec, uint32_t>("u32");
    check_against_draws<Exec, uint64_t>("u64");
    check_against_draws<Exec, float>("f32");
    check_against_draws<Exec, double>("f64");
    check_against_draws<Exec, bool>("bool");
    test_split_fills<Exec>();
    test_mixed_draws<Exec>();
    test_bounded<Exec>();
    check_narrow<Exec, uint8_t>("u8");
    check_narrow<Exec, uint16_t>("u16");
    check_signed<Exec, int8_t, uint8_t>();
    check_signed<Exec, int16_t, uint16_t>();
    check_signed<Exec, int32_t, uint32_t>();
    check_signed<Exec, int64_t, uint64_t>();
    test_f16<Exec>();
    check_complex<Exec, float>();
    check_complex<Exec, double>();
    test_ranks<Exec>();
    test_below<Exec>();
    test_normal<Exec>();
    test_normal_bits<Exec>();
    std::printf("%s: %ld checks, %ld failures\n", space, checks - c0, failures - f0);
}

// Every backend writes the bytes Serial writes, with every kernel it runs.
template <class Exec, class E> static void same_as_serial(const char *space) {
    for (const Trial &t : trials(77, 12)) {
        auto want = device_fill<Kokkos::Serial, E>(t.key, t.pos, t.K, t.n, Kernel::Group);
        for (Kernel kernel : kernels<Exec>()) {
            auto got = device_fill<Exec, E>(t.key, t.pos, t.K, t.n, kernel, t.shift);
            CHECK(std::memcmp(want.data(), got.data(), t.n * sizeof(E)) == 0);
            if (std::memcmp(want.data(), got.data(), t.n * sizeof(E)) != 0)
                std::printf("  %s %s differs from Serial\n", space, name(kernel));
        }
    }
}

// Bounded fills are integers and equal Serial's bytes. Normals equal them on host spaces and
// agree to a few ulps on devices.
template <class Exec> static void same_as_serial_below_normal(const char *space) {
    for (const Trial &t : trials(78, 12)) {
        uint64_t e1, e2;
        auto w32 = device_below<Kokkos::Serial, uint32_t>(t.key, t.pos, t.K, t.n, 6000001u,
                                                          Kernel::Group, 0, &e1);
        auto g32 = device_below<Exec, uint32_t>(t.key, t.pos, t.K, t.n, 6000001u,
                                                kernels<Exec>()[0], t.shift, &e2);
        CHECK(w32 == g32 && e1 == e2);
        auto wn = device_normal<Kokkos::Serial, double>(t.key, t.pos, t.K, t.n, 0, &e1);
        auto gn = device_normal<Exec, double>(t.key, t.pos, t.K, t.n, t.shift, &e2);
        bool ok = e1 == e2;
        for (size_t i = 0; i < t.n; i++)
            ok = ok && same_normal<Exec>(gn[i], wn[i]);
        CHECK(ok);
        if (!ok)
            std::printf("  %s normal differs from Serial\n", space);
    }
}

template <class Exec> static void compare_with_serial(const char *space) {
    long c0 = checks;
    same_as_serial<Exec, uint32_t>(space);
    same_as_serial<Exec, uint64_t>(space);
    same_as_serial<Exec, float>(space);
    same_as_serial<Exec, double>(space);
    same_as_serial<Exec, bool>(space);
    same_as_serial<Exec, uint8_t>(space);
    same_as_serial<Exec, uint16_t>(space);
    same_as_serial<Exec, int64_t>(space);
    same_as_serial<Exec, Kokkos::complex<float>>(space);
#if !KOKKOS_HALF_T_IS_FLOAT
    same_as_serial<Exec, Kokkos::Experimental::half_t>(space);
#endif
    same_as_serial_below_normal<Exec>(space);
    std::printf("%s vs Serial: %ld checks\n", space, checks - c0);
}

int main(int argc, char **argv) {
    Kokkos::ScopeGuard guard(argc, argv);
    std::string dir = argc > 1 ? argv[argc - 1] : "tests/data";
    run<Kokkos::Serial>("Serial", dir);
#ifdef KOKKOS_ENABLE_OPENMP
    run<Kokkos::OpenMP>("OpenMP", dir);
    compare_with_serial<Kokkos::OpenMP>("OpenMP");
#endif
#ifdef KOKKOS_ENABLE_CUDA
    run<Kokkos::Cuda>("Cuda", dir);
    compare_with_serial<Kokkos::Cuda>("Cuda");
#endif
    if (failures) {
        std::printf("%ld of %ld checks failed\n", failures, checks);
        return 1;
    }
    std::printf("ok: %ld checks\n", checks);
    return 0;
}
