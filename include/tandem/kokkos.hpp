/* Tandem8x32 for Kokkos: fills of Views on any execution space, and the scalar
 * generator tandem::Rng from tandem/core.hpp for draws inside kernels.
 *
 * Implements https://github.com/tandem-rng/spec and produces the stream it defines, bit for
 * bit. Copyright 2026 Jessica Cox. Apache License 2.0, see LICENSE.
 */
#pragma once

#include <Kokkos_Core.hpp>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <tandem/core.hpp>

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "tandem/kokkos.hpp assumes a little-endian host"
#endif

namespace tandem {

namespace detail {

struct f16_bits {}; /* binary16 bit patterns of the Float16 draws, stored as uint16_t */
struct below32 {};  /* Lemire bounded draws over the u32 fill, see PURPOSE_BELOW32 */
struct below64 {};
struct exp32 {}; /* exponentials over the f32 fill, see exponential_f32 */
struct exp64 {};

/* The fill's geometry: stream bits [p0, p1), rows r0 .. r1 inclusive, groups g0 .. g1, and
 * what the bounded kinds need: the fill's key and chunk length, and the range. */
struct Span {
    uint64_t p0, p1, r0, r1, g0, g1;
    uint32_t K;
    Key key;
    uint64_t range;
};

/* How an output element is made from a block: element k of the block takes bits
 * [k bits, (k + 1) bits). `e` is its index in the fill, which the bounded kinds need for their
 * fallback stream. */
template <class Kind> struct elem;

template <> struct elem<bool> {
    using out_t = bool;
    static constexpr unsigned bits = 1;
    KOKKOS_INLINE_FUNCTION static bool make(const uint32_t w[4], unsigned k, uint64_t,
                                            const Span &) {
        return (w[k >> 5] >> (k & 31u)) & 1u;
    }
};
template <> struct elem<uint8_t> {
    using out_t = uint8_t;
    static constexpr unsigned bits = 8;
    KOKKOS_INLINE_FUNCTION static uint8_t make(const uint32_t w[4], unsigned k, uint64_t,
                                               const Span &) {
        return (uint8_t)(w[k >> 2] >> ((k & 3u) * 8u));
    }
};
template <> struct elem<uint16_t> {
    using out_t = uint16_t;
    static constexpr unsigned bits = 16;
    KOKKOS_INLINE_FUNCTION static uint16_t make(const uint32_t w[4], unsigned k, uint64_t,
                                                const Span &) {
        return (uint16_t)(w[k >> 1] >> ((k & 1u) * 16u));
    }
};
template <> struct elem<f16_bits> {
    using out_t = uint16_t;
    static constexpr unsigned bits = 16;
    KOKKOS_INLINE_FUNCTION static uint16_t make(const uint32_t w[4], unsigned k, uint64_t e,
                                                const Span &s) {
        return to_f16_bits(elem<uint16_t>::make(w, k, e, s));
    }
};
template <> struct elem<uint32_t> {
    using out_t = uint32_t;
    static constexpr unsigned bits = 32;
    KOKKOS_INLINE_FUNCTION static uint32_t make(const uint32_t w[4], unsigned k, uint64_t,
                                                const Span &) {
        return w[k];
    }
};
template <> struct elem<float> {
    using out_t = float;
    static constexpr unsigned bits = 32;
    KOKKOS_INLINE_FUNCTION static float make(const uint32_t w[4], unsigned k, uint64_t,
                                             const Span &) {
        return to_f32(w[k]);
    }
};
template <> struct elem<uint64_t> {
    using out_t = uint64_t;
    static constexpr unsigned bits = 64;
    KOKKOS_INLINE_FUNCTION static uint64_t make(const uint32_t w[4], unsigned k, uint64_t,
                                                const Span &) {
        return w[2 * k] | ((uint64_t)w[2 * k + 1] << 32);
    }
};
template <> struct elem<double> {
    using out_t = double;
    static constexpr unsigned bits = 64;
    KOKKOS_INLINE_FUNCTION static double make(const uint32_t w[4], unsigned k, uint64_t,
                                              const Span &) {
        return to_f64(w[2 * k] | ((uint64_t)w[2 * k + 1] << 32));
    }
};
template <> struct elem<exp32> {
    using out_t = float;
    static constexpr unsigned bits = 32;
    KOKKOS_INLINE_FUNCTION static float make(const uint32_t w[4], unsigned k, uint64_t,
                                             const Span &) {
        return exponential_f32(to_f32(w[k]));
    }
};
template <> struct elem<exp64> {
    using out_t = double;
    static constexpr unsigned bits = 64;
    KOKKOS_INLINE_FUNCTION static double make(const uint32_t w[4], unsigned k, uint64_t,
                                              const Span &) {
        return exponential_f64(to_f64(w[2 * k] | ((uint64_t)w[2 * k + 1] << 32)));
    }
};
template <> struct elem<below32> {
    using out_t = uint32_t;
    static constexpr unsigned bits = 32;
    KOKKOS_INLINE_FUNCTION static uint32_t make(const uint32_t w[4], unsigned k, uint64_t e,
                                                const Span &s) {
        return below_u32(w[k], (uint32_t)s.range, s.key.w, s.K, (s.p0 >> 5) + e);
    }
};
template <> struct elem<below64> {
    using out_t = uint64_t;
    static constexpr unsigned bits = 64;
    KOKKOS_INLINE_FUNCTION static uint64_t make(const uint32_t w[4], unsigned k, uint64_t e,
                                                const Span &s) {
        return below_u64(w[2 * k] | ((uint64_t)w[2 * k + 1] << 32), s.range, s.key.w, s.K,
                        (s.p0 >> 6) + e);
    }
};

/* Store the elements of the block at stream bit P that fall inside the fill's bits [p0, p1).
 * With ALIGNED the output's blocks sit at 16-byte addresses, and a block fully inside is one
 * 16-byte store. */
template <class Kind, bool ALIGNED>
KOKKOS_FORCEINLINE_FUNCTION void store_block(typename elem<Kind>::out_t *out, const Span &s,
                                             uint64_t P, const uint32_t w[4]) {
    using O = typename elem<Kind>::out_t;
    constexpr unsigned bits = elem<Kind>::bits, per_block = 128 / bits;
    if constexpr (ALIGNED && sizeof(O) * per_block == 16) {
        if (P >= s.p0 && P + 128u <= s.p1) {
            struct alignas(16) Vec {
                O v[per_block];
            } x;
            uint64_t e0 = (P - s.p0) / bits;
            for (unsigned k = 0; k < per_block; k++)
                x.v[k] = elem<Kind>::make(w, k, e0 + k, s);
            *reinterpret_cast<Vec *>(out + e0) = x;
            return;
        }
    }
    for (unsigned k = 0; k < per_block; k++) {
        uint64_t q = P + k * bits;
        if (q >= s.p0 && q < s.p1) {
            uint64_t e = (q - s.p0) / bits;
            out[e] = elem<Kind>::make(w, k, e, s);
        }
    }
}

struct alignas(16) Block {
    uint32_t w[4];
};

/* Auto is Group on host spaces, else Tile. Tile needs K >= 8 and falls back to Chunk. */
enum class Kernel {
    Auto,
    Group, /* host: one work item per group of eight chunks, a whole row per step */
    Chunk, /* one work item per chunk, one block per step */
    Tile,  /* device: one team per 32 groups, eight steps staged in scratch memory */
};

/* A whole row of a host fill. The stream is little-endian bytes, so a row of unsigned
 * integers is the row of u32 words. Row::store writes through memcpy, so the cast aliases
 * nothing. */
template <class Kind>
KOKKOS_INLINE_FUNCTION void store_row(typename elem<Kind>::out_t *dst, const Row &R,
                                      uint64_t e0, const Span &s) {
    if constexpr (std::is_same_v<Kind, uint8_t> || std::is_same_v<Kind, uint16_t> ||
                  std::is_same_v<Kind, uint32_t> || std::is_same_v<Kind, uint64_t>) {
        R.store(reinterpret_cast<uint32_t *>(dst));
    } else if constexpr (std::is_same_v<Kind, float> || std::is_same_v<Kind, double>) {
        R.store(dst);
    } else if constexpr (std::is_same_v<Kind, exp32> || std::is_same_v<Kind, exp64>) {
        /* Uniforms first, then the log in place, a loop the compiler vectorizes. */
        R.store(dst);
        for (unsigned k = 0; k < 1024 / elem<Kind>::bits; k++) {
            if constexpr (std::is_same_v<Kind, exp32>)
                dst[k] = exponential_f32(dst[k]);
            else
                dst[k] = exponential_f64(dst[k]);
        }
    } else {
        constexpr unsigned per_row = 1024 / elem<Kind>::bits, per_block = 128 / elem<Kind>::bits;
        uint32_t w[32];
        R.store(w);
        if constexpr (std::is_same_v<Kind, below32>) {
            /* The common row has no draw whose low product word is below the range, so none can
             * reject, and a plain loop over the 32 products vectorizes. */
            uint32_t hi[32], range = (uint32_t)s.range, low = 0;
#ifdef TANDEM_ROW_SIMD
            typedef uint32_t u32x4 __attribute__((vector_size(16)));
            typedef uint64_t u64x4 __attribute__((vector_size(32)));
            u32x4 under = {0, 0, 0, 0};
            for (unsigned k = 0; k < 32; k += 4) {
                u32x4 x;
                std::memcpy(&x, w + k, 16);
                u64x4 m = __builtin_convertvector(x, u64x4) * (uint64_t)range;
                u32x4 h = __builtin_convertvector(m >> 32, u32x4);
                under |= (u32x4)(__builtin_convertvector(m, u32x4) < range);
                std::memcpy(hi + k, &h, 16);
            }
            low = under[0] | under[1] | under[2] | under[3];
#else
            for (unsigned k = 0; k < 32; k++) {
                uint64_t m = (uint64_t)w[k] * range;
                hi[k] = (uint32_t)(m >> 32);
                low |= (uint32_t)m < range;
            }
#endif
            if (!low) {
                std::memcpy(dst, hi, sizeof hi);
                return;
            }
        }
        for (unsigned k = 0; k < per_row; k++)
            dst[k] = elem<Kind>::make(w + 4 * (k / per_block), k % per_block, e0 + k, s);
    }
}

/* Host spaces only: R holds a row in vector registers. */
template <class Kind, class Exec>
void fill_group(const Exec &exec, const Span s, typename elem<Kind>::out_t *out) {
    Kokkos::parallel_for(
        "tandem::fill (group)",
        Kokkos::RangePolicy<Exec, Kokkos::IndexType<int64_t>>(exec, (int64_t)s.g0,
                                                              (int64_t)s.g1 + 1),
        KOKKOS_LAMBDA(int64_t g) {
            Row R;
            R.seed(s.key.w, (uint64_t)g);
            uint64_t row = (uint64_t)g * s.K;
            uint32_t j0 = row < s.r0 ? (uint32_t)(s.r0 - row) : 0u;
            uint32_t j1 = (uint32_t)(s.r1 - row < s.K - 1u ? s.r1 - row : s.K - 1u);
            for (uint32_t j = 0; j <= j1; j++) {
                R.step();
                if (j < j0)
                    continue;
                uint64_t P = (row + j) * 1024u;
                if (P >= s.p0 && P + 1024u <= s.p1) {
                    uint64_t e0 = (P - s.p0) / elem<Kind>::bits;
                    store_row<Kind>(out + e0, R, e0, s);
                    continue;
                }
                for (unsigned l = 0; l < 8; l++) {
                    uint32_t w[4];
                    R.lane(l, w);
                    store_block<Kind, false>(out, s, P + l * 128u, w);
                }
            }
        });
}

template <class Kind, bool ALIGNED, class Exec>
void fill_chunk(const Exec &exec, const Span s, typename elem<Kind>::out_t *out) {
    Kokkos::parallel_for(
        "tandem::fill (chunk)",
        Kokkos::RangePolicy<Exec, Kokkos::IndexType<int64_t>>(exec, 0,
                                                              (int64_t)(8u * (s.g1 - s.g0 + 1u))),
        KOKKOS_LAMBDA(int64_t t) {
            uint64_t c = 8u * s.g0 + (uint64_t)t, row = (c >> 3) * s.K;
            uint32_t o[4], h[4];
            F_keyed(s.key.w, c, DOMAIN_STREAM, AUX_STREAM, o, h);
            uint32_t j0 = row < s.r0 ? (uint32_t)(s.r0 - row) : 0u;
            uint32_t j1 = (uint32_t)(s.r1 - row < s.K - 1u ? s.r1 - row : s.K - 1u);
            for (uint32_t j = 0; j <= j1; j++) {
                T(o, h);
                if (j >= j0)
                    store_block<Kind, ALIGNED>(out, s, (row + j) * 1024u + (c & 7u) * 128u, o);
            }
        });
}

constexpr unsigned TILE_THREADS = 256, TILE_GROUPS = TILE_THREADS / 8, TILE_STEPS = 8;

/* One thread per chunk, 32 groups per team. Every TILE_STEPS steps the team holds, for each of
 * its groups, TILE_STEPS consecutive rows, 1024 contiguous bytes of the stream. The write phase
 * hands consecutive 16-byte slots to consecutive threads, so a warp writes 512 contiguous
 * bytes. Needs K >= TILE_STEPS. */
template <class Kind, bool ALIGNED, class Exec>
void fill_tile(const Exec &exec, const Span s, typename elem<Kind>::out_t *out) {
    using Policy = Kokkos::TeamPolicy<Exec, Kokkos::LaunchBounds<TILE_THREADS>>;
    constexpr unsigned SLOTS = TILE_GROUPS * TILE_STEPS * 8;
    uint64_t groups = s.g1 - s.g0 + 1u;
    Policy policy(exec, (int)((groups + TILE_GROUPS - 1) / TILE_GROUPS), (int)TILE_THREADS);
    Kokkos::parallel_for(
        "tandem::fill (tile)",
        policy.set_scratch_size(0, Kokkos::PerTeam(SLOTS * sizeof(Block) + 16)),
        KOKKOS_LAMBDA(const typename Policy::member_type &team) {
            Block *tile = static_cast<Block *>(
                team.team_shmem().get_shmem_aligned(SLOTS * sizeof(Block), 16));
            uint64_t gb = s.g0 + (uint64_t)team.league_rank() * TILE_GROUPS;
            unsigned rank = (unsigned)team.team_rank(), gi = rank >> 3, lane = rank & 7u;
            bool mine = gb + gi <= s.g1;
            uint32_t o[4], h[4];
            F_keyed(s.key.w, 8u * (gb + gi) + lane, DOMAIN_STREAM, AUX_STREAM, o, h);
            uint64_t first_bit = gb * s.K * 1024u;
            for (uint32_t jb = 0; jb < s.K; jb += TILE_STEPS) {
                if (first_bit + jb * 1024u >= s.p1)
                    break;
                if (mine) {
                    for (unsigned j = 0; j < TILE_STEPS; j++) {
                        T(o, h);
                        tile[(gi * TILE_STEPS + j) * 8 + lane] = Block{{o[0], o[1], o[2], o[3]}};
                    }
                }
                team.team_barrier();
                for (unsigned slot = rank; slot < SLOTS; slot += TILE_THREADS) {
                    unsigned sg = slot / (TILE_STEPS * 8), within = slot % (TILE_STEPS * 8);
                    uint64_t P = ((gb + sg) * s.K + jb) * 1024u + within * 128u;
                    if (P >= s.p1)
                        continue;
                    store_block<Kind, ALIGNED>(out, s, P, tile[slot].w);
                }
                team.team_barrier();
            }
        });
}

template <class Exec>
constexpr bool is_host =
    Kokkos::SpaceAccessibility<Kokkos::HostSpace, typename Exec::memory_space>::accessible;

/* Host spaces compile the group and chunk kernels, others the tile and chunk kernels. */
template <class Kind, bool ALIGNED, class Exec>
void fill_with(const Exec &exec, const Span &s, typename elem<Kind>::out_t *out, Kernel kernel) {
    if constexpr (is_host<Exec>) {
        if (kernel == Kernel::Chunk)
            fill_chunk<Kind, ALIGNED>(exec, s, out);
        else
            fill_group<Kind>(exec, s, out);
    } else {
        if (kernel == Kernel::Chunk || s.K < TILE_STEPS)
            fill_chunk<Kind, ALIGNED>(exec, s, out);
        else
            fill_tile<Kind, ALIGNED>(exec, s, out);
    }
}

/* Set the span of a fill that takes `n` elements of `bits` bits from the generator's position
 * aligned to `align` bits, and move the position past them. Returns false for an empty fill. */
inline bool plan_span(Rng &rng, uint64_t n, unsigned align, unsigned bits, Span &s) {
    uint64_t p0 = align_pos(rng.position(), align);
    if (p0 < rng.position() || n > (~(uint64_t)0 - p0) / bits)
        throw std::overflow_error("tandem::fill: the fill runs past stream position 2^64");
    s.K = rng.chunk_length();
    s.key = rng.key();
    s.p0 = p0;
    s.p1 = p0 + n * bits;
    s.range = 0;
    rng.set_position(s.p1);
    return n != 0;
}

/* The rows and groups that hold the blocks [ba, bb]. */
inline void set_rows(Span &s, uint64_t ba, uint64_t bb) {
    s.r0 = ba >> 3;
    s.r1 = bb >> 3;
    s.g0 = s.r0 / s.K;
    s.g1 = s.r1 / s.K;
}

template <class Exec, class Kind>
void fill_kind(const Exec &exec, typename elem<Kind>::out_t *out, uint64_t n, Rng &rng,
               Kernel kernel, uint64_t range = 0) {
    constexpr unsigned bits = elem<Kind>::bits;
    Span s;
    if (!plan_span(rng, n, bits, bits, s))
        return;
    s.range = range;
    set_rows(s, s.p0 >> 7, (s.p1 - 1) >> 7);
    /* Blocks land on 16-byte addresses when the output's first byte and the fill's first
     * stream byte agree modulo 16. */
    if (sizeof(typename elem<Kind>::out_t) * 8 == bits && bits >= 8 &&
        ((reinterpret_cast<uintptr_t>(out) - s.p0 / 8) & 15u) == 0)
        fill_with<Kind, true>(exec, s, out, kernel);
    else
        fill_with<Kind, false>(exec, s, out, kernel);
}

/* The float Box-Muller step. On CUDA the angle goes through the fast __sincosf, which is accurate
 * only on [-pi, pi], so the angle is shifted by half a turn and both halves change sign. With
 * the precise logf and sqrtf this keeps the result within 16 ulps + 1e-6 of box_muller2_f32 and
 * makes the fill memory bound. Define TANDEM_PRECISE_F32_NORMAL for the precise step. */
KOKKOS_FORCEINLINE_FUNCTION Pair2<float> normal_step_f32(float a, float b) {
#if defined(__CUDA_ARCH__) && !defined(TANDEM_PRECISE_F32_NORMAL)
    float r = sqrtf(-2.0f * logf(1.0f - a)), s, c;
    __sincosf(6.2831853071795864769f * (b - 0.5f), &s, &c);
    return Pair2<float>{-r * c, -r * s};
#else
    return box_muller2_f32(a, b);
#endif
}

/* The normal pair of the uniforms in the words v: two Float64 draws (four words) or two Float32
 * draws (two words). */
template <class O> KOKKOS_FORCEINLINE_FUNCTION Pair2<O> normal_pair(const uint32_t *v) {
    if constexpr (std::is_same_v<O, double>)
        return box_muller2(to_f64(v[0] | ((uint64_t)v[1] << 32)),
                           to_f64(v[2] | ((uint64_t)v[3] << 32)));
    else
        return normal_step_f32(to_f32(v[0]), to_f32(v[1]));
}

/* Elements 2i and 2i + 1 of an n-element fill. `wide` says the pair sits at a 2 sizeof(O)
 * aligned address. */
template <class O>
KOKKOS_FORCEINLINE_FUNCTION void store_pair(O *dst, bool wide, uint64_t n, uint64_t i,
                                            Pair2<O> z) {
    if (2 * i + 1 >= n) {
        dst[2 * i] = z.z0;
    } else if (wide) {
        struct alignas(2 * sizeof(O)) Two {
            O v[2];
        } two = {{z.z0, z.z1}};
        *reinterpret_cast<Two *>(dst + 2 * i) = two;
    } else {
        dst[2 * i] = z.z0;
        dst[2 * i + 1] = z.z1;
    }
}

/* Normal fill by Box-Muller pairs: pair j, the elements 2j and 2j + 1, comes from the L = 2
 * (float) or 4 (double) 32-bit stream slots that start at slot S + j L, S being the first slot
 * of the fill, with the cos half first. A work item owns the blocks that hold the last slot of
 * a pair. s.range is the element count, which an odd count leaves one past the last pair. A
 * pair goes out as one 16-byte (double) or 8-byte (float) store when the output allows it, and
 * two float pairs of one block as one 16-byte store when the fill starts on a block. With S % L != 0 an element can start in the
 * previous block, so the item also steps the chunk of that block, which for lane 0 is lane 7
 * one step back, or at step 0 the previous group's last chunk. */
template <class O, bool STRADDLE, class Exec>
void fill_normal_chunk(const Exec &exec, const Span s, uint64_t ba, uint64_t bb, O *out) {
    constexpr unsigned L = std::is_same_v<O, double> ? 4u : 2u;
    const uint64_t S = s.p0 >> 5, pairs = (s.p1 - s.p0) / (32u * L), n = s.range;
    const bool wide = (reinterpret_cast<uintptr_t>(out) & (2 * sizeof(O) - 1)) == 0;
    const bool quad = !STRADDLE && L == 2 && (S & 3u) == 0 &&
                      (reinterpret_cast<uintptr_t>(out) & 15u) == 0;
    Kokkos::parallel_for(
        "tandem::fill_normal",
        Kokkos::RangePolicy<Exec, Kokkos::IndexType<int64_t>>(exec, 0,
                                                              (int64_t)(8u * (s.g1 - s.g0 + 1u))),
        KOKKOS_LAMBDA(int64_t t) {
            O *const dst = out;
            uint64_t c = 8u * s.g0 + (uint64_t)t, g = c >> 3, lane = c & 7u, row = g * s.K;
            uint32_t o[4], h[4], po[4] = {0, 0, 0, 0}, ph[4] = {0, 0, 0, 0};
            F_keyed(s.key.w, c, DOMAIN_STREAM, AUX_STREAM, o, h);
            if (STRADDLE)
                F_keyed(s.key.w, lane ? c - 1u : 8u * g + 7u, DOMAIN_STREAM, AUX_STREAM, po, ph);
            uint32_t j0 = row < s.r0 ? (uint32_t)(s.r0 - row) : 0u;
            uint32_t j1 = (uint32_t)(s.r1 - row < s.K - 1u ? s.r1 - row : s.K - 1u);
            for (uint32_t j = 0; j <= j1; j++) {
                T(o, h);
                if (STRADDLE && (lane || j))
                    T(po, ph);
                uint64_t b = (row + j) * 8u + lane;
                if (j < j0 || b < ba || b > bb)
                    continue;
                                if (quad && 4u * (b - ba) + 4u <= n) {
                    Pair2<float> z0 = normal_step_f32(to_f32(o[0]), to_f32(o[1]));
                    Pair2<float> z1 = normal_step_f32(to_f32(o[2]), to_f32(o[3]));
                    struct alignas(16) Quad {
                        float v[4];
                    } q4 = {{z0.z0, z0.z1, z1.z0, z1.z1}};
                    *reinterpret_cast<Quad *>(dst + 4u * (b - ba)) = q4;
                    continue;
                }
                if (!STRADDLE) {
                    /* S % L == 0: the pairs of the block end at its halves (float) or its end
                     * (double), and lie inside it. */
                    for (unsigned m = L == 2 ? 1u : 3u; m < 4; m += L) {
                        uint64_t last = 4u * b + m + 1u; /* one past the pair's last slot */
                        if (last < S + L)
                            continue;
                        uint64_t i = (last - S) / L - 1u;
                        if (i < pairs)
                            store_pair<O>(dst, wide, n, i, normal_pair<O>(o + (m + 1u - L)));
                    }
                    continue;
                }
                /* Pairs can start in the previous block: x holds it, then this one. */
                uint32_t x[8] = {0, 0, 0, 0, o[0], o[1], o[2], o[3]};
                uint32_t q[4] = {0, 0, 0, 0};
                const uint32_t *prev = po;
                if (lane == 0 && j == 0) {
                    if (g > 0)
                        block(s.key.w, 8u * (g - 1u) + 7u, s.K - 1u, q);
                    prev = q;
                }
                for (unsigned k = 0; k < 4; k++)
                    x[k] = prev[k];
                for (unsigned m = 0; m < 4; m++) {
                    uint64_t last = 4u * b + m + 1u;
                    if (last < S + L || (last - S) % L)
                        continue;
                    uint64_t i = (last - S) / L - 1u;
                    if (i < pairs)
                        store_pair<O>(dst, wide, n, i, normal_pair<O>(x + (m + 5u - L)));
                }
            }
        });
}

/* GCC 14 with -mfma reports w as maybe uninitialized, although it is zeroed at its definition. */
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
/* Normal fill on host spaces: a work item steps the eight chunks of a group together and turns
 * the pairs that end in each row into normals with one normal_block call, which the compiler
 * vectorizes. The stream slots of a row are its 32 words, and the pairs are the L-word windows
 * at slots S + j L. A pair that starts in the previous row takes its first words from the
 * last three words of that row, `tail`, which for the first row of a group is the last block
 * of the previous group. */
template <class O, class Exec> void fill_normal_group(const Exec &exec, const Span s, O *out) {
    constexpr unsigned L = std::is_same_v<O, double> ? 4u : 2u;
    const uint64_t S = s.p0 >> 5, pairs = (s.p1 - s.p0) / (32u * L), n = s.range;
    Kokkos::parallel_for(
        "tandem::fill_normal (group)",
        Kokkos::RangePolicy<Exec, Kokkos::IndexType<int64_t>>(exec, (int64_t)s.g0,
                                                              (int64_t)s.g1 + 1),
        [=](int64_t g) { /* host only: normal_block_* are host functions */
            Row R;
            R.seed(s.key.w, (uint64_t)g);
            uint64_t row = (uint64_t)g * s.K;
            uint32_t w[35] = {0, 0, 0}; /* the three words before the row, then the row */
            if (g > 0 && S % L) {
                uint32_t q[4];
                block(s.key.w, 8u * ((uint64_t)g - 1u) + 7u, s.K - 1u, q);
                w[0] = q[1];
                w[1] = q[2];
                w[2] = q[3];
            }
            uint32_t j1 = (uint32_t)(s.r1 - row < s.K - 1u ? s.r1 - row : s.K - 1u);
            for (uint32_t j = 0; j <= j1; j++) {
                R.step();
                R.store(w + 3);
                uint64_t r = row + j, base = 32u * r;
                uint64_t lo = base + 1u > S + L ? base + 1u : S + L;
                uint64_t rem = (lo - S) % L;
                uint64_t t0 = rem ? lo + (L - rem) : lo;
                if (t0 <= base + 32u && r >= s.r0) {
                    O u[32], z[32];
                    unsigned m = 0;
                    uint64_t first = (t0 - S) / L - 1u;
                    for (uint64_t t = t0; t <= base + 32u && first + m < pairs; t += L, m++) {
                        const uint32_t *v = w + (t - L - base + 3u);
                        if constexpr (L == 4) {
                            u[2 * m] = to_f64(v[0] | ((uint64_t)v[1] << 32));
                            u[2 * m + 1] = to_f64(v[2] | ((uint64_t)v[3] << 32));
                        } else {
                            u[2 * m] = to_f32(v[0]);
                            u[2 * m + 1] = to_f32(v[1]);
                        }
                    }
                    if constexpr (L == 4)
                        normal_block_f64(u, z, m);
                    else
                        normal_block_f32(u, z, m);
                    for (unsigned q = 0; q < 2 * m; q++)
                        if (2 * first + q < n)
                            out[2 * first + q] = z[q];
                }
                w[0] = w[32];
                w[1] = w[33];
                w[2] = w[34];
            }
        });
}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

template <class Exec, class O>
void fill_normal_ptr(const Exec &exec, O *out, uint64_t n, Rng &rng) {
    constexpr unsigned L = std::is_same_v<O, double> ? 4u : 2u;
    if (n == 0) /* no draws, so no alignment either */
        return;
    uint64_t pairs = (n + 1u) / 2u;
    Span s;
    plan_span(rng, pairs, 32u * L / 2u, 32u * L, s);
    s.range = n;
    uint64_t S = s.p0 >> 5, ba = (S + L - 1u) >> 2, bb = (S + pairs * L - 1u) >> 2;
    set_rows(s, ba, bb);
    if constexpr (is_host<Exec>) {
        fill_normal_group<O>(exec, s, out);
    } else {
        if (S % L)
            fill_normal_chunk<O, true>(exec, s, ba, bb, out);
        else
            fill_normal_chunk<O, false>(exec, s, ba, bb, out);
    }
}

/* What a View's value type draws: the kind of fill, and how many of its elements one value
 * takes. Signed integers read the unsigned draw of the same width in two's complement, a
 * complex value the real then the imaginary component. */
template <unsigned N> struct uint_of;
template <> struct uint_of<1> { using type = uint8_t; };
template <> struct uint_of<2> { using type = uint16_t; };
template <> struct uint_of<4> { using type = uint32_t; };
template <> struct uint_of<8> { using type = uint64_t; };

template <class E, class = void> struct plan;
template <> struct plan<bool> {
    using kind = bool;
    static constexpr unsigned parts = 1;
};
template <class E>
struct plan<E, std::enable_if_t<std::is_integral_v<E> && !std::is_same_v<E, bool>>> {
    using kind = typename uint_of<sizeof(E)>::type;
    static constexpr unsigned parts = 1;
};
template <> struct plan<float> {
    using kind = float;
    static constexpr unsigned parts = 1;
};
template <> struct plan<double> {
    using kind = double;
    static constexpr unsigned parts = 1;
};
template <class T> struct plan<Kokkos::complex<T>, std::enable_if_t<std::is_floating_point_v<T>>> {
    static_assert(sizeof(Kokkos::complex<T>) == 2 * sizeof(T), "unexpected complex layout");
    using kind = T;
    static constexpr unsigned parts = 2;
};
/* half_t is float when Kokkos has no half type. */
template <class E>
struct plan<E, std::enable_if_t<std::is_same_v<E, Kokkos::Experimental::half_t> &&
                                !std::is_same_v<E, float>>> {
    static_assert(sizeof(E) == 2, "unexpected half_t layout");
    using kind = f16_bits;
    static constexpr unsigned parts = 1;
};

template <class Exec, class View> void check_view(const View &view, const char *who) {
    static_assert(Kokkos::SpaceAccessibility<Exec, typename View::memory_space>::accessible,
                  "tandem: the execution space cannot access the View's memory");
    if (!view.span_is_contiguous())
        throw std::invalid_argument(std::string(who) + ": the View must be contiguous");
}

/* Fill any View whose value type has a plan, in memory order, with an explicit kernel. */
template <class Exec, class View>
void fill(const Exec &exec, const View &view, Rng &rng, Kernel kernel) {
    using E = typename View::non_const_value_type;
    using P = plan<E>;
    check_view<Exec>(view, "tandem::fill");
    fill_kind<Exec, typename P::kind>(
        exec, reinterpret_cast<typename elem<typename P::kind>::out_t *>(view.data()),
        (uint64_t)view.size() * P::parts, rng, kernel);
}

} // namespace detail

/* Fill a View of any rank and layout with the draws that start at the generator's position,
 * the values the specification's fill defines, and move the position past them. The View must
 * be contiguous, and the fill runs in memory order: the last index first for LayoutRight, the
 * first for LayoutLeft. The value type is bool, an integer of 8 to 64 bits, float, double,
 * Kokkos::complex of float or double, or Kokkos::Experimental::half_t. Signed integers hold the
 * two's complement of the unsigned draw. A complex value takes two draws, the real and then the
 * imaginary component, as the specification's complex fill does. A uint16_t View gets raw
 * draws: use fill_f16_bits for Float16 bit patterns.
 *
 * The kernel runs on `exec` and is not fenced, but the generator advances at once, so the
 * next fill or draw continues the stream. */
template <class Exec, class View> void fill(const Exec &exec, const View &view, Rng &rng) {
    detail::fill(exec, view, rng, detail::Kernel::Auto);
}

/* The same on the View's execution space, fenced. */
template <class View> void fill(const View &view, Rng &rng) {
    typename View::execution_space exec;
    detail::fill(exec, view, rng, detail::Kernel::Auto);
    exec.fence("tandem::fill: fence after the fill");
}

/* Binary16 bit patterns of the specification's Float16 draws, (raw >> 5) * 2^-11, into a
 * uint16_t View of any rank. */
template <class Exec, class View>
void fill_f16_bits(const Exec &exec, const View &view, Rng &rng) {
    static_assert(std::is_same_v<typename View::non_const_value_type, uint16_t>,
                  "tandem::fill_f16_bits: the value type must be uint16_t");
    detail::check_view<Exec>(view, "tandem::fill_f16_bits");
    detail::fill_kind<Exec, detail::f16_bits>(exec, view.data(), view.size(), rng,
                                              detail::Kernel::Auto);
}
template <class View> void fill_f16_bits(const View &view, Rng &rng) {
    typename View::execution_space exec;
    fill_f16_bits(exec, view, rng);
    exec.fence("tandem::fill_f16_bits: fence after the fill");
}

/* Uniform integers on [0, range) in a uint32_t or uint64_t View, by Lemire's method as
 * Rng::urand(range). Element i takes draw i of the u32 (u64) fill and consumes exactly that one
 * draw, so the fill advances the position by 32 n (64 n) bits whatever the draws are. A rejected
 * draw retries on a fallback stream, see PURPOSE_BELOW32 in core.hpp. Not part of the
 * specification. */
template <class Exec, class View>
void fill_below(const Exec &exec, const View &view, Rng &rng,
                typename View::non_const_value_type range) {
    using E = typename View::non_const_value_type;
    static_assert(std::is_unsigned_v<E> && !std::is_same_v<E, bool> &&
                      (sizeof(E) == 4 || sizeof(E) == 8),
                  "tandem::fill_below: the value type must be uint32_t or uint64_t");
    detail::check_view<Exec>(view, "tandem::fill_below");
    if (view.size() == 0) /* no draws, so no alignment either */
        return;
    using Kind = std::conditional_t<sizeof(E) == 4, detail::below32, detail::below64>;
    detail::fill_kind<Exec, Kind>(exec, reinterpret_cast<typename detail::elem<Kind>::out_t *>(
                                            view.data()),
                                  view.size(), rng, detail::Kernel::Auto, range);
}
template <class View>
void fill_below(const View &view, Rng &rng, typename View::non_const_value_type range) {
    typename View::execution_space exec;
    fill_below(exec, view, rng, range);
    exec.fence("tandem::fill_below: fence after the fill");
}

/* Standard normals in a float or double View by Box-Muller, the flattened sequence of
 * Rng::normalf2 or Rng::normal2 calls: pair j, the elements 2j and 2j + 1 with the cos half
 * first, is made from the draws 2j and 2j + 1 of the f32 or f64 fill. An odd count drops the
 * last sin half and still consumes both draws, so the fill takes 64 (float) or 128 (double)
 * bits per pair. An empty fill leaves the position alone. Specification Appendix A. Double
 * normals and host float normals equal tandem-c bit for bit. CUDA float normals take the fast
 * __sincosf and agree to 16 ulps + 1e-6. */
template <class Exec, class View> void fill_normal(const Exec &exec, const View &view, Rng &rng) {
    using E = typename View::non_const_value_type;
    static_assert(std::is_same_v<E, float> || std::is_same_v<E, double>,
                  "tandem::fill_normal: the value type must be float or double");
    detail::check_view<Exec>(view, "tandem::fill_normal");
    detail::fill_normal_ptr(exec, view.data(), view.size(), rng);
}
template <class View> void fill_normal(const View &view, Rng &rng) {
    typename View::execution_space exec;
    fill_normal(exec, view, rng);
    exec.fence("tandem::fill_normal: fence after the fill");
}

/* Standard exponentials -log(1 - u) in a float or double View, the sequence of
 * Rng::exponentialf or Rng::exponential calls: element i from draw i of the f32 or f64 fill.
 * Specification Appendix A, bit for bit equal to tandem-c on every backend. An empty fill leaves
 * the position alone. */
template <class Exec, class View>
void fill_exponential(const Exec &exec, const View &view, Rng &rng) {
    using E = typename View::non_const_value_type;
    static_assert(std::is_same_v<E, float> || std::is_same_v<E, double>,
                  "tandem::fill_exponential: the value type must be float or double");
    detail::check_view<Exec>(view, "tandem::fill_exponential");
    if (view.size() == 0) /* no draws, so no alignment either */
        return;
    using Kind = std::conditional_t<std::is_same_v<E, float>, detail::exp32, detail::exp64>;
    /* The log makes a device fill compute bound, and there the tile kernel's separate write
     * phase costs more than its coalescing gains. */
    detail::fill_kind<Exec, Kind>(exec, view.data(), view.size(), rng,
                                  detail::is_host<Exec> ? detail::Kernel::Auto
                                                        : detail::Kernel::Chunk);
}
template <class View> void fill_exponential(const View &view, Rng &rng) {
    typename View::execution_space exec;
    fill_exponential(exec, view, rng);
    exec.fence("tandem::fill_exponential: fence after the fill");
}

} // namespace tandem
