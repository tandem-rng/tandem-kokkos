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

/* The fill's geometry: stream bits [p0, p1), rows r0 .. r1 inclusive, groups g0 .. g1, and
 * the fill's key and chunk length. */
struct Span {
    uint64_t p0, p1, r0, r1, g0, g1;
    uint32_t K;
    Key key;
};

/* How an output element is made from a block: element k of the block takes bits
 * [k bits, (k + 1) bits). `e` is its index in the fill. */
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
/* Store the elements of the block at stream bit P that fall inside the fill's bits [p0, p1).
 * With ALIGNED the output's blocks sit at 16-byte addresses, and a block fully inside is one
 * 16-byte store. */
template <class Kind, bool ALIGNED>
KOKKOS_FORCEINLINE_FUNCTION void store_block(typename elem<Kind>::out_t *out, const Span &s,
                                             uint64_t P, const uint32_t w[4]) {
    using O = typename elem<Kind>::out_t;
    constexpr unsigned bits = elem<Kind>::bits, per_block = 128 / bits;
    if constexpr (ALIGNED && bits >= 32) {
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
    } else {
        constexpr unsigned per_row = 1024 / elem<Kind>::bits, per_block = 128 / elem<Kind>::bits;
        uint32_t w[32];
        R.store(w);
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
               Kernel kernel) {
    constexpr unsigned bits = elem<Kind>::bits;
    Span s;
    if (!plan_span(rng, n, bits, bits, s))
        return;
    set_rows(s, s.p0 >> 7, (s.p1 - 1) >> 7);
    /* Blocks land on 16-byte addresses when the output's first byte and the fill's first
     * stream byte agree modulo 16. */
    if (bits >= 32 && ((reinterpret_cast<uintptr_t>(out) - s.p0 / 8) & 15u) == 0)
        fill_with<Kind, true>(exec, s, out, kernel);
    else
        fill_with<Kind, false>(exec, s, out, kernel);
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

} // namespace tandem
