// Spec vectors, the spec's conformance files, fills against in-kernel draws, split fills,
// derived keys, bounded draws, and byte identity across backends, on every enabled execution
// space.
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>

#include <limits>
#include <random>
#include <string>
#include <vector>

#include <Kokkos_Random.hpp>
#include <tandem/kokkos.hpp>

#include "conformance.hpp"
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

// ---- Stream hashes of hashes.json ---------------------------------------------------------

static Key json_key(const Json &words) {
    Key k;
    for (int w = 0; w < 4; w++)
        k.w[w] = (uint32_t)words.items[w].hex();
    return k;
}

template <class T> static std::string sha256_of(const std::vector<T> &v) {
    return sha256_hex(v.data(), v.size() * sizeof(T));
}

// The fills of every kernel, and the in-kernel scalar draws of the types that have one, hash to
// the stream's SHA-256. Float16 as bit patterns, which test_f16 relates to half_t Views.
template <class Exec, class E> static void check_stream(const Json &s) {
    const Key key = json_key(s["key"]);
    const uint32_t K = (uint32_t)s["K"].u64();
    const uint64_t start = s["start"].u64();
    const size_t n = s["n"].u64();
    const std::string &want = s["sha256"].text;
    for (Kernel kernel : kernels<Exec>()) {
        bool ok;
        if constexpr (std::is_same_v<E, tandem::detail::f16_bits>)
            ok = sha256_of(device_f16_bits<Exec>(key, start, K, n, kernel)) == want;
        else
            ok = sha256_of(device_fill<Exec, E>(key, start, K, n, kernel)) == want;
        CHECK(ok);
        if (!ok)
            std::printf("  %s: %s fill hash differs\n", s["file"].text.c_str(), name(kernel));
    }
    if constexpr (std::is_same_v<E, bool> || std::is_same_v<E, uint32_t> ||
                  std::is_same_v<E, uint64_t> || std::is_same_v<E, float> ||
                  std::is_same_v<E, double>)
        CHECK(sha256_of(device_draws<Exec, E>(key, start, K, n)) == want);
}

// UInt128 and Char have no fill here.
template <class Exec> static void test_streams(const Json &hashes) {
    for (const Json &s : hashes["streams"].items) {
        const std::string &type = s["type"].text;
        if (type == "UInt32")
            check_stream<Exec, uint32_t>(s);
        else if (type == "UInt64")
            check_stream<Exec, uint64_t>(s);
        else if (type == "UInt8")
            check_stream<Exec, uint8_t>(s);
        else if (type == "Bool")
            check_stream<Exec, bool>(s);
        else if (type == "Float32")
            check_stream<Exec, float>(s);
        else if (type == "Float64")
            check_stream<Exec, double>(s);
        else if (type == "Float16")
            check_stream<Exec, tandem::detail::f16_bits>(s);
        else if (type == "ComplexF32")
            check_stream<Exec, Kokkos::complex<float>>(s);
        else if (type == "ComplexF64")
            check_stream<Exec, Kokkos::complex<double>>(s);
    }
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

// Without a rejection a bounded fill equals the sequential urand(range) calls.
template <class Exec> static void test_below() {
    check_below<Exec, uint32_t>("u32");
    check_below<Exec, uint64_t>("u64");

    const Key key = Rng(11).key();
    uint64_t end0;
    auto got = device_below<Exec, uint32_t>(key, 5, 32, 5000, 1000u, kernels<Exec>()[0], 0,
                                            &end0);
    Rng r = Rng::from_key(key, 5, 32);
    bool same = true;
    for (uint32_t v : got)
        same = same && v == r.urand(1000u);
    CHECK(same);
}

// Host normals and every double normal run core.hpp's explicit-fma polynomials and equal
// tandem-c bit for bit. CUDA float normals take the fast __sincosf, within 16 ulps + 1e-6.
template <class Exec, class E> static bool same_normal(E got, E want) {
    if (tandem::detail::is_host<Exec> || std::is_same_v<E, double>)
        return std::memcmp(&got, &want, sizeof got) == 0;
    return std::abs(got - want) <= 16 * 0x1p-23f * std::abs(want) + 1e-6f;
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

// Host copy of a double normal fill with an explicit kernel.
template <class Exec>
static std::vector<double> device_normal64(const Key &key, uint64_t pos, uint32_t K, size_t n,
                                           Kernel kernel, size_t shift, uint64_t *end) {
    Kokkos::View<double *, typename Exec::memory_space> buf("normal64", n + 4);
    auto out = Kokkos::subview(buf, Kokkos::pair<size_t, size_t>(shift, shift + n));
    Rng r = Rng::from_key(key, pos, K);
    tandem::detail::fill_kind<Exec, tandem::detail::norm64>(Exec(), out.data(), n, r, kernel);
    Exec().fence();
    *end = r.position();
    auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
    return std::vector<double>(host.data(), host.data() + n);
}

// A double fill equals the Rng::normal calls bit for bit with every kernel, and takes one u64
// draw per element from the start aligned to 64.
template <class Exec> static void check_normal64() {
    for (const Trial &t : trials(42, 24)) {
        Rng r = Rng::from_key(t.key, t.pos, t.K);
        std::vector<double> want(t.n);
        for (double &x : want)
            x = r.normal();
        for (Kernel kernel : kernels<Exec>()) {
            uint64_t end;
            auto got = device_normal64<Exec>(t.key, t.pos, t.K, t.n, kernel, t.shift, &end);
            bool ok = std::memcmp(want.data(), got.data(), t.n * sizeof(double)) == 0;
            CHECK(ok);
            CHECK(end == tandem::align_pos(t.pos, 64) + 64 * t.n);
            if (!ok)
                std::printf("  f64 normal %s (K=%u pos=%llu n=%zu shift=%zu)\n", name(kernel),
                            t.K, (unsigned long long)t.pos, t.n, t.shift);
        }
        // fill_normal itself, which on devices queues the misses in a team kernel.
        uint64_t end;
        auto got = device_normal<Exec, double>(t.key, t.pos, t.K, t.n, t.shift, &end);
        CHECK(std::memcmp(want.data(), got.data(), t.n * sizeof(double)) == 0);
        CHECK(end == tandem::align_pos(t.pos, 64) + 64 * t.n);
    }
}

// A float fill equals the flattened normalf2 calls: an odd count drops the last sin half and
// still consumes both draws.
template <class Exec> static void check_normal32() {
    for (const Trial &t : trials(42, 24)) {
        Rng r = Rng::from_key(t.key, t.pos, t.K);
        std::vector<float> want(t.n);
        for (size_t i = 0; i < t.n; i += 2) {
            auto pair = r.normalf2();
            want[i] = pair.z0;
            if (i + 1 < t.n)
                want[i + 1] = pair.z1;
        }
        uint64_t end;
        auto got = device_normal<Exec, float>(t.key, t.pos, t.K, t.n, t.shift, &end);
        bool ok = true;
        for (size_t i = 0; i < t.n; i++)
            ok = ok && same_normal<Exec>(got[i], want[i]);
        CHECK(ok);
        CHECK(end == r.position());
        if (!ok)
            std::printf("  f32 normal (K=%u pos=%llu n=%zu shift=%zu)\n", t.K,
                        (unsigned long long)t.pos, t.n, t.shift);
    }
}

template <class Exec> static void test_normal() {
    check_normal64<Exec>();
    check_normal32<Exec>();
}

// N(0, 1) on 1e7 draws: raw moments 0, 1, 0, 3 within five standard errors, and the
// Kolmogorov-Smirnov distance below its p = 0.001 critical value 1.95 / sqrt(n).
template <class E> static void test_normal_law() {
    constexpr size_t n = 10000000;
    Kokkos::View<E *, Kokkos::HostSpace> v("v", n);
    Rng r(2026, 10, 5);
    tandem::fill_normal(Kokkos::DefaultHostExecutionSpace(), v, r);
    Kokkos::fence();
    std::vector<double> x(v.data(), v.data() + n);
    double m[4] = {0, 0, 0, 0};
    for (double z : x) {
        double p = z;
        for (int k = 0; k < 4; k++, p *= z)
            m[k] += p;
    }
    const double want[4] = {0, 1, 0, 3}, var[4] = {1, 2, 15, 96};
    for (int k = 0; k < 4; k++)
        CHECK(std::abs(m[k] / n - want[k]) < 5 * std::sqrt(var[k] / n));
    std::sort(x.begin(), x.end());
    double ks = 0;
    for (size_t i = 0; i < n; i++) {
        double F = 0.5 * std::erfc(-x[i] / std::sqrt(2.0));
        ks = std::max(ks, std::max(F - (double)i / n, (double)(i + 1) / n - F));
    }
    CHECK(ks < 1.95 / std::sqrt((double)n));
}

// ---- Exponential fills ----------------------------------------------------------------------

template <class Exec, class E>
static std::vector<E> device_exponential(const Key &key, uint64_t pos, uint32_t K, size_t n,
                                         Kernel kernel, size_t shift, uint64_t *end) {
    using Kind =
        std::conditional_t<std::is_same_v<E, float>, tandem::detail::exp32, tandem::detail::exp64>;
    Kokkos::View<E *, typename Exec::memory_space> buf("exponential", n + 4);
    auto out = Kokkos::subview(buf, Kokkos::pair<size_t, size_t>(shift, shift + n));
    Rng r = Rng::from_key(key, pos, K);
    tandem::detail::fill_kind<Exec, Kind>(Exec(), out.data(), n, r, kernel);
    Exec().fence();
    *end = r.position();
    auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
    return std::vector<E>(host.data(), host.data() + n);
}

// A fill equals the scalar exponential() calls on the host, bit for bit, with every kernel.
template <class Exec, class E> static void check_exponential(const char *label) {
    for (const Trial &t : trials(43, 12)) {
        Rng r = Rng::from_key(t.key, t.pos, t.K);
        std::vector<E> want(t.n);
        for (E &x : want)
            if constexpr (std::is_same_v<E, float>)
                x = r.exponentialf();
            else
                x = r.exponential();
        for (Kernel kernel : kernels<Exec>()) {
            uint64_t end;
            auto got = device_exponential<Exec, E>(t.key, t.pos, t.K, t.n, kernel, t.shift, &end);
            bool ok = std::memcmp(want.data(), got.data(), t.n * sizeof(E)) == 0;
            CHECK(ok);
            CHECK(t.n == 0 || end == r.position());
            if (!ok)
                std::printf("  %s exponential %s (K=%u pos=%llu n=%zu shift=%zu)\n", label,
                            name(kernel), t.K, (unsigned long long)t.pos, t.n, t.shift);
        }
    }
}

template <class Exec> static void test_exponential() {
    check_exponential<Exec, double>("f64");
    check_exponential<Exec, float>("f32");
}

// Exp(1) on 1e7 draws: raw moments 1, 2, 6, 24 within five standard errors, and the
// Kolmogorov-Smirnov distance below its p = 0.001 critical value 1.95 / sqrt(n).
template <class E> static void test_exponential_law() {
    constexpr size_t n = 10000000;
    Kokkos::View<E *, Kokkos::HostSpace> v("v", n);
    Rng r(2026, 10, 4);
    tandem::fill_exponential(Kokkos::DefaultHostExecutionSpace(), v, r);
    Kokkos::fence();
    std::vector<double> x(v.data(), v.data() + n);
    double m[4] = {0, 0, 0, 0};
    for (double e : x) {
        double p = e;
        for (int k = 0; k < 4; k++, p *= e)
            m[k] += p;
    }
    const double want[4] = {1, 2, 6, 24}, var[4] = {1, 20, 684, 39744};
    for (int k = 0; k < 4; k++)
        CHECK(std::abs(m[k] / n - want[k]) < 5 * std::sqrt(var[k] / n));
    std::sort(x.begin(), x.end());
    double ks = 0;
    for (size_t i = 0; i < n; i++) {
        double F = -std::expm1(-x[i]);
        ks = std::max(ks, std::max(F - (double)i / n, (double)(i + 1) / n - F));
    }
    CHECK(ks < 1.95 / std::sqrt((double)n));
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

// ---- Conformance cases of the specification ------------------------------------------------

// One case of below.json, fill_below.json, normal.json, exponential.json or choice.json.
struct Case {
    std::string id, kind;
    Key key;
    uint32_t K;
    unsigned w; // draw width
    uint64_t start, n, end, range = 0;
    std::vector<uint64_t> values;
    double ulps = 0, abs = 0; // the tolerance of Float32 normals and exponentials
    unsigned rejected = 0;
    std::vector<double> weights;
    uint64_t capacity = 0;
    std::vector<uint64_t> cut, alias; // where the case pins the whole table
};

static std::vector<Case> read_cases(const std::string &dir, const char *file) {
    const Json cases = read_json(dir + "/" + file);
    std::vector<Case> out;
    for (const Json &j : cases["cases"].items) {
        Case c;
        c.id = j["id"].text;
        c.kind = j["kind"].text;
        c.key = json_key(j["key"]);
        c.K = (uint32_t)j["K"].u64();
        c.w = c.kind.ends_with("32") ? 32 : 64;
        c.start = j["start"].u64();
        c.n = j["n"].u64();
        if (const Json *r = j.find("range"))
            c.range = r->hex();
        for (const Json &v : j["values"].items)
            c.values.push_back(v.hex());
        if (const Json *t = j.find("tol")) {
            c.ulps = (*t)["ulps"].number();
            c.abs = (*t)["abs"].number();
        }
        if (const Json *r = j.find("rejected"))
            c.rejected = (unsigned)r->u64();
        if (const Json *ws = j.find("weights"))
            for (const Json &x : ws->items)
                c.weights.push_back(std::bit_cast<double>(x.hex()));
        if (const Json *cap = j.find("capacity"))
            c.capacity = cap->hex();
        if (const Json *cut = j.find("cut")) {
            for (const Json &x : cut->items)
                c.cut.push_back(x.hex());
            for (const Json &x : j["alias"].items)
                c.alias.push_back(x.hex());
        }
        // Where the source pins no end: one draw per element, a pair per two Float32 normals.
        const uint64_t p0 = tandem::align_pos(c.start, c.w);
        if (const Json *e = j.find("end"))
            c.end = e->u64();
        else
            c.end = c.kind == "fill_normal_f32" ? p0 + 64 * ((c.n + 1) / 2) : p0 + c.w * c.n;
        out.push_back(c);
    }
    return out;
}

// The name is a pointer: GCC's dangling-reference warning fires on a std::string temporary.
static const Case &case_named(const std::vector<Case> &cases, const char *name) {
    for (const Case &c : cases)
        if (c.id.ends_with(std::string(" ") + name))
            return c;
    throw std::runtime_error(std::string("no conformance case ") + name);
}

// Bit for bit, except CUDA Float32 normals, which take the fast __sincosf and agree to the
// case's tolerance |y - x| <= ulps 2^-23 |x| + abs.
template <class Exec> static bool matches(const Case &c, const std::vector<uint64_t> &got) {
    if (got.size() != c.values.size())
        return false;
    const bool exact = tandem::detail::is_host<Exec> || c.kind != "fill_normal_f32";
    for (size_t i = 0; i < got.size(); i++) {
        if (exact) {
            if (got[i] != c.values[i])
                return false;
            continue;
        }
        double y = std::bit_cast<float>((uint32_t)got[i]);
        double x = std::bit_cast<float>((uint32_t)c.values[i]);
        if (!(std::abs(y - x) <= c.ulps * 0x1p-23 * std::abs(x) + c.abs))
            return false;
    }
    return true;
}

// The fill paths of a case: Kernel::Auto is the public fill, the others detail::fill_kind with
// that kernel. Float32 normals have no kernel choice.
template <class Exec> static std::vector<Kernel> case_paths(const Case &c) {
    std::vector<Kernel> paths = {Kernel::Auto};
    if (c.kind != "fill_normal_f32")
        for (Kernel k : kernels<Exec>())
            paths.push_back(k);
    return paths;
}

template <class E> KOKKOS_INLINE_FUNCTION uint64_t bit_pattern(E x) {
    using U = std::conditional_t<sizeof(E) == 8, uint64_t, uint32_t>;
    U u;
    std::memcpy(&u, &x, sizeof u);
    return u;
}

// The first n elements of a View of n + 1 elements of E, written by `fill`, as bit patterns.
// The last element must keep its sentinel, so a fill writes nothing past its n elements.
template <class Exec, class E, class F> static std::vector<uint64_t> case_values(uint64_t n, F fill) {
    Kokkos::View<E *, typename Exec::memory_space> buf("case", n + 1);
    using U = std::conditional_t<sizeof(E) == 8, uint64_t, uint32_t>;
    const E sentinel = std::bit_cast<E>(~(U)0);
    Kokkos::deep_copy(buf, sentinel);
    fill(Kokkos::subview(buf, Kokkos::pair<size_t, size_t>(0, n)));
    Exec().fence();
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), buf);
    CHECK(bit_pattern(h(n)) == bit_pattern(sentinel));
    std::vector<uint64_t> v(n);
    for (size_t i = 0; i < n; i++)
        v[i] = bit_pattern(h(i));
    return v;
}

// The alias table of a case's weights, built on the host, with its arrays copied to Exec's
// memory space.
template <class Exec> struct DeviceTable {
    std::vector<uint64_t> cut;
    std::vector<uint32_t> alias;
    tandem::ChoiceTable host{}, dev{};
    bool built;
    Kokkos::View<uint64_t *, typename Exec::memory_space> dcut;
    Kokkos::View<uint32_t *, typename Exec::memory_space> dalias;
    explicit DeviceTable(const std::vector<double> &weights)
        : cut(weights.size()), alias(weights.size()),
          built(tandem::choice_build(host, weights.data(), weights.size(), cut.data(),
                                     alias.data())),
          dcut("cut", cut.size()), dalias("alias", alias.size()) {
        Kokkos::deep_copy(dcut, Kokkos::View<uint64_t *, Kokkos::HostSpace>(cut.data(), cut.size()));
        Kokkos::deep_copy(dalias,
                          Kokkos::View<uint32_t *, Kokkos::HostSpace>(alias.data(), alias.size()));
        dev = tandem::ChoiceTable{host.capacity, dcut.data(), dalias.data(), host.m};
    }
};

// n elements of the case's fill from r on Exec by one path, as bit patterns. An empty fill takes
// the public fill, which owns the position rule of an empty fill.
template <class Exec>
static std::vector<uint64_t> fill_case(const Case &c, Rng &r, uint64_t n, Kernel kernel) {
    namespace td = tandem::detail;
    const Exec exec;
    const bool pub = kernel == Kernel::Auto || n == 0;
    const std::string &k = c.kind;
    if (k == "fill_choice") {
        const DeviceTable<Exec> t(c.weights);
        return case_values<Exec, uint32_t>(n, [&](auto out) {
            if (pub)
                tandem::fill_choice(exec, out, r, t.dev);
            else
                td::fill_kind<Exec, td::choice_idx>(exec, out.data(), n, r, kernel, 0, t.dev);
        });
    }
    auto below = [&](auto out) {
        using E = typename decltype(out)::non_const_value_type;
        using Kind = std::conditional_t<sizeof(E) == 4, td::below32, td::below64>;
        if (pub)
            tandem::fill_below(exec, out, r, (E)c.range);
        else
            td::fill_kind<Exec, Kind>(exec, out.data(), n, r, kernel, c.range);
    };
    auto normal = [&](auto out) {
        if (pub)
            tandem::fill_normal(exec, out, r);
        else
            td::fill_kind<Exec, td::norm64>(exec, out.data(), n, r, kernel);
    };
    auto exponential = [&](auto out) {
        using E = typename decltype(out)::non_const_value_type;
        using Kind = std::conditional_t<std::is_same_v<E, float>, td::exp32, td::exp64>;
        if (pub)
            tandem::fill_exponential(exec, out, r);
        else
            td::fill_kind<Exec, Kind>(exec, out.data(), n, r, kernel);
    };
    if (k == "fill_below_u32")
        return case_values<Exec, uint32_t>(n, below);
    if (k == "fill_below_u64")
        return case_values<Exec, uint64_t>(n, below);
    if (k == "fill_normal_f64")
        return case_values<Exec, double>(n, normal);
    if (k == "fill_normal_f32")
        return case_values<Exec, float>(n, [&](auto out) { tandem::fill_normal(exec, out, r); });
    if (k == "fill_exponential_f64")
        return case_values<Exec, double>(n, exponential);
    if (k == "fill_exponential_f32")
        return case_values<Exec, float>(n, exponential);
    throw std::runtime_error("no fill for kind " + k);
}

// Every case on every path: whole, and cut at elements 1, 7, 20, 21 and n - 1 into pieces
// filled in order on one generator, values and end position. A Float32 normal fill cuts only
// between pairs, since an odd piece drops its last sin half. The cases hold the fallbacks by
// global draw index, the empty fills, odd Float32 normal counts and the pair rule.
template <class Exec> static void check_fill_cases(const std::vector<Case> &cases) {
    for (const Case &c : cases)
        for (Kernel path : case_paths<Exec>(c)) {
            Rng r = Rng::from_key(c.key, c.start, c.K);
            bool ok = matches<Exec>(c, fill_case<Exec>(c, r, c.n, path)) && r.position() == c.end;
            const uint64_t cuts[] = {1, 7, 20, 21, c.n - 1};
            for (uint64_t cut : cuts) {
                if (c.n == 0 || cut >= c.n || (c.kind == "fill_normal_f32" && cut % 2))
                    continue;
                Rng g = Rng::from_key(c.key, c.start, c.K);
                auto a = fill_case<Exec>(c, g, cut, path);
                auto b = fill_case<Exec>(c, g, c.n - cut, path);
                a.insert(a.end(), b.begin(), b.end());
                ok = ok && matches<Exec>(c, a) && g.position() == c.end;
            }
            CHECK(ok);
            if (!ok)
                std::printf("  %s (%s) differs\n", c.id.c_str(),
                            path == Kernel::Auto ? "public" : name(path));
        }
}

// Element i of case a equals element i + shift of case b, both filled here.
template <class Exec>
static void check_shift(const std::vector<Case> &cases, const char *a, const char *b,
                        size_t shift) {
    const Case &ca = case_named(cases, a), &cb = case_named(cases, b);
    Rng ra = Rng::from_key(ca.key, ca.start, ca.K), rb = Rng::from_key(cb.key, cb.start, cb.K);
    auto va = fill_case<Exec>(ca, ra, ca.n, Kernel::Auto);
    auto vb = fill_case<Exec>(cb, rb, cb.n, Kernel::Auto);
    bool ok = !va.empty();
    for (size_t i = 0; i < va.size() && i + shift < vb.size(); i++)
        ok = ok && va[i] == vb[i + shift];
    CHECK(ok);
}

enum class Scalar { below_u32, below_u64, normal_f64, normal_f32, exp_f64, exp_f32, choice };

KOKKOS_INLINE_FUNCTION uint64_t scalar_draw(Rng &r, Scalar s, uint64_t range,
                                            const tandem::ChoiceTable &t) {
    switch (s) {
    case Scalar::choice:
        return r.choice(t);
    case Scalar::below_u32:
        return r.urand((uint32_t)range);
    case Scalar::below_u64:
        return r.urand64(range);
    case Scalar::normal_f64:
        return bit_pattern(r.normal());
    case Scalar::normal_f32:
        return bit_pattern(r.normalf());
    case Scalar::exp_f64:
        return bit_pattern(r.exponential());
    default:
        return bit_pattern(r.exponentialf());
    }
}

// The first n values of a case as n scalar draws on the host and in a kernel on Exec, and the
// position after them.
template <class Exec> static void check_scalars(const Case &c, Scalar s, uint64_t n, uint64_t end) {
    Case head = c;
    head.values.resize(n);
    const DeviceTable<Exec> table(c.weights);
    const tandem::ChoiceTable dev_table = table.dev;
    Rng r = Rng::from_key(c.key, c.start, c.K);
    std::vector<uint64_t> host(n);
    for (uint64_t &v : host)
        v = scalar_draw(r, s, c.range, table.host);
    Kokkos::View<uint64_t *, typename Exec::memory_space> d("scalars", n + 1);
    const Key key = c.key;
    const uint64_t start = c.start, range = c.range;
    const uint32_t K = c.K;
    Kokkos::parallel_for(
        Kokkos::RangePolicy<Exec>(0, 1), KOKKOS_LAMBDA(int) {
            Rng g = Rng::from_key(key, start, K);
            for (uint64_t i = 0; i < n; i++)
                d(i) = scalar_draw(g, s, range, dev_table);
            d(n) = g.position();
        });
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d);
    std::vector<uint64_t> dev(h.data(), h.data() + n);
    bool ok = matches<Kokkos::Serial>(head, host) && r.position() == end &&
              matches<Exec>(head, dev) && h(n) == end;
    CHECK(ok);
    if (!ok)
        std::printf("  %s scalar draws differ\n", c.id.c_str());
}

// Range 0 gives 0 and consumes one draw of the width that the interface names, and an empty
// uniform fill aligns the position to its width.
template <class Exec> static void check_range0_and_empty() {
    Kokkos::View<uint32_t *, typename Exec::memory_space> u32("u32", 1), none32("none32", 0);
    Kokkos::View<uint64_t *, typename Exec::memory_space> u64("u64", 1), none64("none64", 0);
    Kokkos::View<bool *, typename Exec::memory_space> none1("none1", 0);
    Kokkos::deep_copy(u32, ~0u);
    Kokkos::deep_copy(u64, ~0ull);
    Rng r = Rng::from_key(Rng(42).key(), 33, 32);
    tandem::fill_below(u32, r, 0u);
    CHECK(Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), u32)(0) == 0 &&
          r.position() == 96);
    tandem::fill_below(u64, r, (uint64_t)0);
    CHECK(Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), u64)(0) == 0 &&
          r.position() == 192);
    CHECK(r.urand(0u) == 0 && r.position() == 224);
    CHECK(r.urand64((uint64_t)0) == 0 && r.position() == 320);

    for (uint64_t pos : {33ull, 65ull}) {
        Rng g = Rng::from_key(Rng(42).key(), pos, 32);
        tandem::fill(none32, g);
        CHECK(g.position() == tandem::align_pos(pos, 32));
        g.set_position(pos);
        tandem::fill(none64, g);
        CHECK(g.position() == tandem::align_pos(pos, 64));
        g.set_position(pos);
        tandem::fill(none1, g);
        CHECK(g.position() == pos);
    }
}

template <class Exec> static void test_cases(const std::string &dir) {
    const auto below = read_cases(dir, "below.json");
    const auto fill_below = read_cases(dir, "fill_below.json");
    const auto normal = read_cases(dir, "normal.json");
    const auto exponential = read_cases(dir, "exponential.json");
    check_fill_cases<Exec>(fill_below);
    check_fill_cases<Exec>(normal);
    check_fill_cases<Exec>(exponential);
    unsigned rejected = 0;
    for (const Case &c : fill_below)
        rejected += c.rejected;
    CHECK(rejected > 0);

    // Scalar bounded draws retry in sequence. Scalar normals and exponentials equal the fills.
    for (const Case &c : below)
        check_scalars<Exec>(c, c.w == 32 ? Scalar::below_u32 : Scalar::below_u64, c.n, c.end);
    for (const Case &c : normal)
        if (c.kind == "fill_normal_f64" && c.n)
            check_scalars<Exec>(c, Scalar::normal_f64, c.n, c.end);
    for (const Case &c : exponential)
        if (c.n)
            check_scalars<Exec>(c, c.w == 64 ? Scalar::exp_f64 : Scalar::exp_f32, c.n, c.end);
    // A scalar Float32 normal is the cos half of a pair and consumes two draws.
    const Case &nf = case_named(normal, "CROSS_NORMALF");
    check_scalars<Exec>(nf, Scalar::normal_f32, 1, tandem::align_pos(nf.start, 32) + 64);

    // A later start shifts the elements: the fallback follows the global draw index, and a
    // Float32 normal start one pair later shifts the output by one pair.
    check_shift<Exec>(fill_below, "CROSS_BELOW32_AT[4]", "CROSS_BELOW32[4]", 1);
    check_shift<Exec>(fill_below, "CROSS_BELOW64_AT[6]", "CROSS_BELOW64[6]", 1);
    check_shift<Exec>(normal, "CROSS_NORMAL[1]", "CROSS_NORMAL[0]", 1);
    check_shift<Exec>(normal, "CROSS_NORMAL32[2]", "CROSS_NORMAL32[0]", 2);
    check_shift<Exec>(normal, "CROSS_NORMAL32[1]", "CROSS_NORMALF", 0);
    check_range0_and_empty<Exec>();
}

// ---- Weighted choice ------------------------------------------------------------------------

// Every kernel equals the host's choice() calls at random keys, chunk lengths, positions,
// lengths and output alignments.
template <class Exec> static void check_choice_draws(const DeviceTable<Exec> &t) {
    for (const Trial &tr : trials(57, 12)) {
        Rng r = Rng::from_key(tr.key, tr.pos, tr.K);
        std::vector<uint32_t> want(tr.n);
        for (uint32_t &x : want)
            x = r.choice(t.host);
        for (Kernel kernel : kernels<Exec>()) {
            Kokkos::View<uint32_t *, typename Exec::memory_space> buf("choice", tr.n + 4);
            Rng g = Rng::from_key(tr.key, tr.pos, tr.K);
            tandem::detail::fill_kind<Exec, tandem::detail::choice_idx>(
                Exec(), buf.data() + tr.shift, tr.n, g, kernel, 0, t.dev);
            Exec().fence();
            auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), buf);
            std::vector<uint32_t> got(h.data() + tr.shift, h.data() + tr.shift + tr.n);
            CHECK(first_diff(want, got) == SIZE_MAX && g.position() == r.position());
        }
    }
}

// Every case's table, whole where the case pins it, fill, cuts and scalar draws, the shift of a
// later start, fills at any alignment, and the weights that build no table. The case "choice
// single" has m = 1.
template <class Exec> static void test_choice(const std::string &dir) {
    const auto cases = read_cases(dir, "choice.json");
    size_t tables = 0;
    for (const Case &c : cases) {
        const DeviceTable<Exec> t(c.weights);
        CHECK(t.built && t.host.capacity == c.capacity);
        if (c.cut.empty())
            continue;
        CHECK(t.cut == c.cut &&
              std::equal(t.alias.begin(), t.alias.end(), c.alias.begin(), c.alias.end()));
        tables++;
    }
    CHECK(tables > 0);
    check_fill_cases<Exec>(cases);
    for (const Case &c : cases)
        if (c.n)
            check_scalars<Exec>(c, Scalar::choice, c.n, c.end);
    check_shift<Exec>(cases, "CROSS_CHOICE[1]", "CROSS_CHOICE[0]", 1);
    check_choice_draws<Exec>(DeviceTable<Exec>(case_named(cases, "choice mixed").weights));

    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const std::vector<std::vector<double>> no_table = {{1, -1}, {1, inf}, {nan, 1}, {0, -0.0}, {}};
    for (const auto &w : no_table)
        CHECK(!DeviceTable<Exec>(w).built);
}

// ---- Dumps of hashes.json and position boundaries ------------------------------------------

static uint64_t fnv1a(uint64_t h, const void *p, size_t n) {
    const unsigned char *b = static_cast<const unsigned char *>(p);
    for (size_t i = 0; i < n; i++)
        h = (h ^ b[i]) * 0x100000001b3ull;
    return h;
}

template <class Exec, class E, class F> static uint64_t dump_fill(uint64_t h, size_t n, F fill) {
    Kokkos::View<E *, typename Exec::memory_space> v(Kokkos::view_alloc("dump", Kokkos::WithoutInitializing), n);
    fill(v);
    Exec().fence();
    auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
    return fnv1a(h, host.data(), n * sizeof(E));
}

// The fills of each dump in order on one generator per start. The Float32 normal hash holds for
// C's polynomials, which host fills take and CUDA's fast __sincosf does not, so that dump runs
// on host spaces only.
template <class Exec> static void test_dumps(const Json &hashes) {
    const Exec exec;
    for (const Json &dump : hashes["dumps"].items) {
        const Json &draws = dump["draws"];
        bool f32_normal = false;
        for (const Json &d : draws.items)
            f32_normal = f32_normal || d["kind"].text == "fill_normal_f32";
        if (f32_normal && !tandem::detail::is_host<Exec>)
            continue;
        const Key key = json_key(dump["key"]);
        uint64_t h = 0xcbf29ce484222325ull, end = 0;
        for (const Json &start : dump["starts"].items) {
            Rng r = Rng::from_key(key, start.u64(), (uint32_t)dump["K"].u64());
            for (const Json &d : draws.items) {
                const std::string &kind = d["kind"].text;
                const size_t n = d["n"].u64();
                if (kind == "fill_normal_f64")
                    h = dump_fill<Exec, double>(h, n, [&](auto v) { tandem::fill_normal(exec, v, r); });
                else if (kind == "fill_normal_f32")
                    h = dump_fill<Exec, float>(h, n, [&](auto v) { tandem::fill_normal(exec, v, r); });
                else if (kind == "fill_exponential_f64")
                    h = dump_fill<Exec, double>(h, n, [&](auto v) { tandem::fill_exponential(exec, v, r); });
                else if (kind == "fill_exponential_f32")
                    h = dump_fill<Exec, float>(h, n, [&](auto v) { tandem::fill_exponential(exec, v, r); });
                else
                    throw std::runtime_error("no fill for dump kind " + kind);
            }
            end = r.position();
        }
        const Json *want_end = dump.find("end");
        bool ok = h == dump["fnv1a"].hex() && (!want_end || end == want_end->u64());
        CHECK(ok);
        if (!ok)
            std::printf("  %s: hash %016llx\n", dump["id"].text.c_str(), (unsigned long long)h);
    }
}

// A complex value whose real part ends a block takes its imaginary part from the next block.
template <class Exec> static void check_complex_straddle() {
    const Key key = Rng(42).key();
    const Rng at0 = Rng::from_key(key, 0, 32);
    for (Kernel kernel : kernels<Exec>()) {
        uint64_t end;
        auto d = device_fill<Exec, Kokkos::complex<double>>(key, 64, 32, 1, kernel, 0, &end);
        CHECK(end == 192 && d[0].real() == at0.at_drand(1) && d[0].imag() == at0.at_drand(2));
        auto f = device_fill<Exec, Kokkos::complex<float>>(key, 96, 32, 1, kernel, 0, &end);
        CHECK(end == 160 && f[0].real() == at0.at_frand(3) && f[0].imag() == at0.at_frand(4));
    }
}

// Random access on the host and in a kernel equals the fill from the same position, from starts
// on both sides of block, row and chunk boundaries. A chunk at K = 8 is 8192 bits.
template <class Exec> static void check_random_access() {
    const Key key = Rng(42).key();
    constexpr size_t n = 64;
    for (uint64_t p : {0ull, 100ull, 127ull, 128ull, 1000ull, 1023ull, 1024ull, 6000ull, 8191ull,
                       8192ull, 8193ull, 16389ull}) {
        auto u64 = device_fill<Exec, uint64_t>(key, p, 8, n, Kernel::Chunk);
        auto u32 = device_fill<Exec, uint32_t>(key, p, 8, n, Kernel::Chunk);
        Kokkos::View<uint64_t *, typename Exec::memory_space> v("at", 2 * n);
        Kokkos::parallel_for(
            Kokkos::RangePolicy<Exec>(0, (int64_t)n), KOKKOS_LAMBDA(int64_t i) {
                const Rng r = Rng::from_key(key, p, 8);
                v(i) = r.at_urand64((uint64_t)i);
                v(n + i) = r.at_urand((uint64_t)i);
            });
        auto dev = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
        const Rng r = Rng::from_key(key, p, 8);
        bool ok = true;
        for (size_t i = 0; i < n; i++)
            ok = ok && r.at_urand64(i) == u64[i] && dev(i) == u64[i] && r.at_urand(i) == u32[i] &&
                 dev(n + i) == u32[i];
        CHECK(ok);
    }
}

// A generator accepts start 2^63 - 1 and rejects 2^63 and 2^64 - 1 without changing state, on
// the host and in a kernel: set_position returns false, and from_key keeps position 0. A UInt64
// draw at 2^63 - 1 aligns to 2^63, on the host, in a kernel and in a fill, and a fill that ends
// past 2^63 moves the generator to its end. A fill whose end reaches 2^64 throws before it writes or moves the position: its View has 2^57 elements of
// 64 bits from 2^63, viewed over one element, which the check reads back.
template <class Exec> static void check_position_bounds() {
    const Key key = Rng(42).key();
    const uint64_t top = 1ull << 63;
    Rng r = Rng::from_key(key, top - 1, 32);
    CHECK(r.position() == top - 1);
    const uint64_t x = r.urand64();
    CHECK(r.position() == top + 64 && x == Rng::from_key(key, top - 1, 32).at_urand64(0));
    Kokkos::View<uint64_t[8], typename Exec::memory_space> k("bounds");
    Kokkos::parallel_for(
        Kokkos::RangePolicy<Exec>(0, 1), KOKKOS_LAMBDA(int) {
            Rng g = Rng::from_key(key, top - 1, 32);
            k(0) = g.position();
            k(1) = g.urand64();
            k(2) = g.position();
            Rng s = Rng::from_key(key, 5, 32);
            k(3) = s.set_position(top) || s.set_position(~0ull);
            k(4) = s.position();
            k(5) = Rng::from_key(key, top, 32).position();
            k(6) = Rng::from_key(key, ~0ull, 32).position();
            k(7) = s.set_position(top - 1) && s.position() == top - 1;
        });
    auto hk = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), k);
    CHECK(hk(0) == top - 1 && hk(1) == x && hk(2) == top + 64);
    CHECK(hk(3) == 0 && hk(4) == 5 && hk(5) == 0 && hk(6) == 0 && hk(7) == 1);
    for (uint64_t bad : {top, ~(uint64_t)0}) {
        Rng s = Rng::from_key(key, 5, 32);
        CHECK(!s.set_position(bad) && s.position() == 5);
        CHECK(Rng::from_key(key, bad, 32).position() == 0);
    }
    for (Kernel kernel : kernels<Exec>()) {
        uint64_t end;
        auto v = device_fill<Exec, uint64_t>(key, top - 1, 32, 1, kernel, 0, &end);
        CHECK(v[0] == x && end == top + 64);
    }
    // A fill that ends past 2^63 moves the generator there, so the next fill continues the
    // stream instead of repeating it.
    Kokkos::View<uint64_t *, typename Exec::memory_space> a("a", 3), b("b", 2);
    Rng f = Rng::from_key(key, top - 1, 32);
    tandem::fill(a, f);
    CHECK(f.position() == top + 192);
    tandem::fill(b, f);
    auto ha = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), a);
    auto hb = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b);
    CHECK(f.position() == top + 320 && ha(0) == x && hb(0) == r.at_urand64(2) &&
          hb(1) == r.at_urand64(3));

    Kokkos::View<uint64_t *, typename Exec::memory_space> one("one", 1);
    Kokkos::deep_copy(one, ~0ull);
    Kokkos::View<uint64_t *, typename Exec::memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        huge(one.data(), (size_t)1 << 57);
    Rng g = Rng::from_key(key, top - 1, 32);
    bool threw = false;
    try {
        tandem::fill(Exec(), huge, g);
    } catch (const std::overflow_error &) {
        threw = true;
    }
    Exec().fence();
    CHECK(threw && g.position() == top - 1 &&
          Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), one)(0) == ~0ull);
}

// ---- Backends -----------------------------------------------------------------------------

template <class Exec>
static void run(const char *space, const std::string &dir, const Json &hashes) {
    long c0 = checks, f0 = failures;
    test_vectors<Exec>();
    test_streams<Exec>(hashes);
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
    test_exponential<Exec>();
    test_cases<Exec>(dir);
    test_choice<Exec>(dir);
    test_dumps<Exec>(hashes);
    check_complex_straddle<Exec>();
    check_random_access<Exec>();
    check_position_bounds<Exec>();
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

// Bounded fills and double normals equal Serial's bytes.
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
    const std::string dir = argc > 1 ? argv[argc - 1] : "tests/conformance";
    const Json hashes = read_json(dir + "/hashes.json");
    run<Kokkos::Serial>("Serial", dir, hashes);
    test_exponential_law<double>();
    test_exponential_law<float>();
    test_normal_law<double>();
    test_normal_law<float>();
#ifdef KOKKOS_ENABLE_OPENMP
    run<Kokkos::OpenMP>("OpenMP", dir, hashes);
    compare_with_serial<Kokkos::OpenMP>("OpenMP");
#endif
#ifdef KOKKOS_ENABLE_CUDA
    run<Kokkos::Cuda>("Cuda", dir, hashes);
    compare_with_serial<Kokkos::Cuda>("Cuda");
#endif
    if (failures) {
        std::printf("%ld of %ld checks failed\n", failures, checks);
        return 1;
    }
    std::printf("ok: %ld checks\n", checks);
    return 0;
}
