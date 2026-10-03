/* Tandem8x32 for Kokkos: fills of rank-1 Views on any execution space, and the scalar
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
#include <type_traits>

#include <tandem/core.hpp>

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "tandem/kokkos.hpp assumes a little-endian host"
#endif

namespace tandem {

namespace detail {

/* How an output element is made from the words of a block: element k of the block takes bits
 * [k bits, (k + 1) bits). */
template <class E> struct elem;

template <> struct elem<bool> {
    static constexpr unsigned bits = 1;
    KOKKOS_INLINE_FUNCTION static bool make(const uint32_t w[4], unsigned k) {
        return (w[k >> 5] >> (k & 31u)) & 1u;
    }
};
template <> struct elem<uint32_t> {
    static constexpr unsigned bits = 32;
    KOKKOS_INLINE_FUNCTION static uint32_t make(const uint32_t w[4], unsigned k) { return w[k]; }
};
template <> struct elem<float> {
    static constexpr unsigned bits = 32;
    KOKKOS_INLINE_FUNCTION static float make(const uint32_t w[4], unsigned k) {
        return to_f32(w[k]);
    }
};
template <> struct elem<uint64_t> {
    static constexpr unsigned bits = 64;
    KOKKOS_INLINE_FUNCTION static uint64_t make(const uint32_t w[4], unsigned k) {
        return w[2 * k] | ((uint64_t)w[2 * k + 1] << 32);
    }
};
template <> struct elem<double> {
    static constexpr unsigned bits = 64;
    KOKKOS_INLINE_FUNCTION static double make(const uint32_t w[4], unsigned k) {
        return to_f64(w[2 * k] | ((uint64_t)w[2 * k + 1] << 32));
    }
};

/* Store the elements of the block at stream bit P that fall inside the fill's bits [p0, p1).
 * With ALIGNED the output's blocks sit at 16-byte addresses, and a block fully inside is one
 * 16-byte store. */
template <class E, bool ALIGNED>
KOKKOS_FORCEINLINE_FUNCTION void store_block(E *out, uint64_t p0, uint64_t p1, uint64_t P,
                                             const uint32_t w[4]) {
    constexpr unsigned bits = elem<E>::bits, per_block = 128 / bits;
    if constexpr (ALIGNED && bits >= 32) {
        if (P >= p0 && P + 128u <= p1) {
            struct alignas(16) Vec {
                E v[per_block];
            } x;
            for (unsigned k = 0; k < per_block; k++)
                x.v[k] = elem<E>::make(w, k);
            *reinterpret_cast<Vec *>(out + (P - p0) / bits) = x;
            return;
        }
    }
    for (unsigned k = 0; k < per_block; k++) {
        uint64_t q = P + k * bits;
        if (q >= p0 && q < p1)
            out[(q - p0) / bits] = elem<E>::make(w, k);
    }
}

struct alignas(16) Block {
    uint32_t w[4];
};

/* The fill's geometry: stream bits [p0, p1), rows r0 .. r1 inclusive, groups g0 .. g1. */
struct Span {
    uint64_t p0, p1, r0, r1, g0, g1;
    uint32_t K;
};

/* Auto is Group on host spaces, else Tile. Tile needs K >= 8 and falls back to Chunk. */
enum class Kernel {
    Auto,
    Group, /* host: one work item per group of eight chunks, a whole row per step */
    Chunk, /* one work item per chunk, one block per step */
    Tile,  /* device: one team per 32 groups, eight steps staged in scratch memory */
};

/* A whole row of a host fill. The stream is little-endian bytes, so a row of u64 is the
 * row of u32 words. Row::store writes through memcpy, so the cast aliases nothing. */
template <class E> KOKKOS_INLINE_FUNCTION void store_row(E *dst, const Row &R) {
    if constexpr (std::is_same_v<E, uint64_t>) {
        R.store(reinterpret_cast<uint32_t *>(dst));
    } else if constexpr (std::is_same_v<E, bool>) {
        uint32_t w[32];
        R.store(w);
        for (unsigned k = 0; k < 1024; k++)
            dst[k] = elem<E>::make(w, k);
    } else {
        R.store(dst);
    }
}

/* Host spaces only: R holds a row in vector registers. */
template <class E, class Exec>
void fill_group(const Exec &exec, const Key key, const Span s, E *out) {
    Kokkos::parallel_for(
        "tandem::fill (group)",
        Kokkos::RangePolicy<Exec, Kokkos::IndexType<int64_t>>(exec, (int64_t)s.g0,
                                                              (int64_t)s.g1 + 1),
        KOKKOS_LAMBDA(int64_t g) {
            Row R;
            R.seed(key.w, (uint64_t)g);
            uint64_t row = (uint64_t)g * s.K;
            uint32_t j0 = row < s.r0 ? (uint32_t)(s.r0 - row) : 0u;
            uint32_t j1 = (uint32_t)(s.r1 - row < s.K - 1u ? s.r1 - row : s.K - 1u);
            for (uint32_t j = 0; j <= j1; j++) {
                R.step();
                if (j < j0)
                    continue;
                uint64_t P = (row + j) * 1024u;
                if (P >= s.p0 && P + 1024u <= s.p1) {
                    store_row(out + (P - s.p0) / elem<E>::bits, R);
                    continue;
                }
                for (unsigned l = 0; l < 8; l++) {
                    uint32_t w[4];
                    R.lane(l, w);
                    store_block<E, false>(out, s.p0, s.p1, P + l * 128u, w);
                }
            }
        });
}

template <class E, bool ALIGNED, class Exec>
void fill_chunk(const Exec &exec, const Key key, const Span s, E *out) {
    Kokkos::parallel_for(
        "tandem::fill (chunk)",
        Kokkos::RangePolicy<Exec, Kokkos::IndexType<int64_t>>(exec, 0,
                                                              (int64_t)(8u * (s.g1 - s.g0 + 1u))),
        KOKKOS_LAMBDA(int64_t t) {
            uint64_t c = 8u * s.g0 + (uint64_t)t, row = (c >> 3) * s.K;
            uint32_t o[4], h[4];
            F_keyed(key.w, c, DOMAIN_STREAM, AUX_STREAM, o, h);
            uint32_t j0 = row < s.r0 ? (uint32_t)(s.r0 - row) : 0u;
            uint32_t j1 = (uint32_t)(s.r1 - row < s.K - 1u ? s.r1 - row : s.K - 1u);
            for (uint32_t j = 0; j <= j1; j++) {
                T(o, h);
                if (j >= j0)
                    store_block<E, ALIGNED>(out, s.p0, s.p1, (row + j) * 1024u + (c & 7u) * 128u,
                                            o);
            }
        });
}

constexpr unsigned TILE_THREADS = 256, TILE_GROUPS = TILE_THREADS / 8, TILE_STEPS = 8;

/* One thread per chunk, 32 groups per team. Every TILE_STEPS steps the team holds, for each of
 * its groups, TILE_STEPS consecutive rows, 1024 contiguous bytes of the stream. The write phase
 * hands consecutive 16-byte slots to consecutive threads, so a warp writes 512 contiguous
 * bytes. Needs K >= TILE_STEPS. */
template <class E, bool ALIGNED, class Exec>
void fill_tile(const Exec &exec, const Key key, const Span s, E *out) {
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
            F_keyed(key.w, 8u * (gb + gi) + lane, DOMAIN_STREAM, AUX_STREAM, o, h);
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
                    store_block<E, ALIGNED>(out, s.p0, s.p1, P, tile[slot].w);
                }
                team.team_barrier();
            }
        });
}

template <class Exec>
constexpr bool is_host =
    Kokkos::SpaceAccessibility<Kokkos::HostSpace, typename Exec::memory_space>::accessible;

/* Host spaces compile the group and chunk kernels, others the tile and chunk kernels. */
template <class E, bool ALIGNED, class Exec>
void fill_with(const Exec &exec, const Key &key, const Span &s, E *out, Kernel kernel) {
    if constexpr (is_host<Exec>) {
        if (kernel == Kernel::Chunk)
            fill_chunk<E, ALIGNED>(exec, key, s, out);
        else
            fill_group<E>(exec, key, s, out);
    } else {
        if (kernel == Kernel::Chunk || s.K < TILE_STEPS)
            fill_chunk<E, ALIGNED>(exec, key, s, out);
        else
            fill_tile<E, ALIGNED>(exec, key, s, out);
    }
}

template <class Exec, class View>
void fill(const Exec &exec, const View &view, Rng &rng, Kernel kernel) {
    using E = typename View::non_const_value_type;
    static_assert(View::rank == 1, "tandem::fill: the View must have rank 1");
    static_assert(std::is_same_v<E, bool> || std::is_same_v<E, uint32_t> ||
                      std::is_same_v<E, uint64_t> || std::is_same_v<E, float> ||
                      std::is_same_v<E, double>,
                  "tandem::fill: the value type must be bool, uint32_t, uint64_t, float or double");
    static_assert(Kokkos::SpaceAccessibility<Exec, typename View::memory_space>::accessible,
                  "tandem::fill: the execution space cannot access the View's memory");
    constexpr unsigned bits = elem<E>::bits;
    if (!view.span_is_contiguous())
        throw std::invalid_argument("tandem::fill: the View must be contiguous");
    uint64_t n = view.extent(0), p0 = align_pos(rng.position(), bits);
    if (p0 < rng.position() || n > (~(uint64_t)0 - p0) / bits)
        throw std::overflow_error("tandem::fill: the fill runs past stream position 2^64");
    uint64_t p1 = p0 + n * bits;
    rng.set_position(p1);
    if (n == 0)
        return;

    Span s;
    s.K = rng.chunk_length();
    s.p0 = p0;
    s.p1 = p1;
    s.r0 = p0 >> 10;
    s.r1 = (p1 - 1) >> 10;
    s.g0 = s.r0 / s.K;
    s.g1 = s.r1 / s.K;
    E *out = view.data();
    /* Blocks land on 16-byte addresses when the output's first byte and the fill's first
     * stream byte agree modulo 16. */
    if (bits >= 32 && ((reinterpret_cast<uintptr_t>(out) - p0 / 8) & 15u) == 0)
        fill_with<E, true>(exec, rng.key(), s, out, kernel);
    else
        fill_with<E, false>(exec, rng.key(), s, out, kernel);
}

} // namespace detail

/* Fill a rank-1 View of bool, uint32_t, uint64_t, float or double with the draws that start at
 * the generator's position, the values the specification's fill defines, and move the position
 * past them. The kernel runs on `exec` and is not fenced, but the generator advances at once,
 * so the next fill or draw continues the stream. */
template <class Exec, class View> void fill(const Exec &exec, const View &view, Rng &rng) {
    detail::fill(exec, view, rng, detail::Kernel::Auto);
}

/* The same on the View's execution space, fenced. */
template <class View> void fill(const View &view, Rng &rng) {
    typename View::execution_space exec;
    detail::fill(exec, view, rng, detail::Kernel::Auto);
    exec.fence("tandem::fill: fence after the fill");
}

} // namespace tandem
