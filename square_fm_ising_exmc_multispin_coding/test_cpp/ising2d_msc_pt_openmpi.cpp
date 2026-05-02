#include <mpi.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// ============================================================
// 2D square-lattice Ising model
// Multi-spin coding (64 replicas packed in uint64_t)
// + Replica exchange MC (OpenMPI parallel over temperature replicas)
// + Incremental energy/magnetization update
// + Auto temperature ladder optimization
// + Binder cumulant / susceptibility / specific heat
//
// Compile:
//   mpic++ -O3 -std=c++17 ising2d_msc_pt_openmpi.cpp -o ising2d_msc_pt_openmpi
//
// Example:
//   mpirun -np 8 ./ising2d_msc_pt_openmpi --L 32 --Tmin 1.4 --Tmax 3.4 \
//       --therm 1500 --meas 1500 --skip 8 --target-c 1.0 --output outdir
//
// Distribution rule requested by user:
//   local_capacity = ceil(nrep / ncore)
//   actual local replicas depend on the last rank.
// ============================================================

constexpr double J = 1.0;
constexpr int WORD_BITS = 64;

struct Params {
    int L = 24;
    double Tmin = 1.4;
    double Tmax = 3.4;
    int nrep = -1;                 // if > 0, use uniform beta ladder; else auto optimize
    double target_c = 1.15;        // auto ladder difficulty target
    int coarse_points = 16;
    int pilot_therm = 60;
    int pilot_meas = 60;
    int pilot_skip = 2;
    int therm = 250;
    int meas = 250;
    int skip = 5;
    std::uint64_t seed = 20260430ULL;
    int resync_every = 0;          // 0 = no periodic exact resync
    bool use_popcount_resync = false;
    std::string output = "output_ising2d_msc_pt_openmpi";
};

struct Replica {
    int gid = -1;
    double beta = 0.0;
    std::vector<std::uint64_t> spins;
    std::array<std::int64_t, WORD_BITS> E{};
    std::array<std::int64_t, WORD_BITS> M{};
};

struct PilotData {
    std::vector<double> beta_coarse;
    std::vector<double> T_coarse;
    std::vector<double> sigmaE;
    std::vector<double> evar_density;
    std::vector<double> emean_density;
    std::vector<double> S;
    std::vector<double> T_ladder;
    std::vector<double> beta_ladder;
};

struct Observables {
    std::vector<double> e_mean, e_err;
    std::vector<double> mabs_mean, mabs_err;
    std::vector<double> cv, chi, chi_conn;
    std::vector<double> m2_mean, sqrt_m2, m4_mean, binder;
    std::vector<double> exch_ratio;
};

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

static inline std::uint64_t bernoulli_mask(double p, std::mt19937_64 &rng) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::uint64_t m = 0ULL;
    for (int b = 0; b < WORD_BITS; ++b) {
        if (dist(rng) < p) m |= (1ULL << b);
    }
    return m;
}

static inline int bit_count_u64(std::uint64_t x) {
#if defined(__GNUG__) || defined(__clang__)
    return __builtin_popcountll(x);
#else
    int c = 0; while (x) { x &= (x - 1); ++c; } return c;
#endif
}

static inline std::uint64_t mask_eq0(std::uint64_t a, std::uint64_t b, std::uint64_t c, std::uint64_t d) {
    return ~(a | b | c | d);
}

static inline std::uint64_t mask_eq1(std::uint64_t a, std::uint64_t b, std::uint64_t c, std::uint64_t d) {
    return ((a & ~b & ~c & ~d) |
            (~a & b & ~c & ~d) |
            (~a & ~b & c & ~d) |
            (~a & ~b & ~c & d));
}

static inline std::uint64_t mask_eq3(std::uint64_t a, std::uint64_t b, std::uint64_t c, std::uint64_t d) {
    return ((a & b & c & ~d) |
            (a & b & ~c & d) |
            (a & ~b & c & d) |
            (~a & b & c & d));
}

static inline std::uint64_t mask_eq4(std::uint64_t a, std::uint64_t b, std::uint64_t c, std::uint64_t d) {
    return a & b & c & d;
}

static inline std::uint64_t masked_swap_word(std::uint64_t a, std::uint64_t b, std::uint64_t mask) {
    return a ^ ((a ^ b) & mask);
}

static inline void add_int_delta_from_mask(std::array<std::int64_t, WORD_BITS> &arr, std::uint64_t mask, int delta) {
    for (int b = 0; b < WORD_BITS; ++b) {
        if ((mask >> b) & 1ULL) arr[b] += delta;
    }
}

static inline void total_energy_per_replica(const std::vector<std::uint64_t> &spins, int L,
                                            std::array<std::int64_t, WORD_BITS> &E) {
    E.fill(0);
    for (int i = 0; i < L; ++i) {
        int ip = (i + 1) % L;
        for (int j = 0; j < L; ++j) {
            int jp = (j + 1) % L;
            std::uint64_t s = spins[i * L + j];
            std::uint64_t sr = spins[i * L + jp];
            std::uint64_t sd = spins[ip * L + j];
            for (int b = 0; b < WORD_BITS; ++b) {
                int si = ((s >> b) & 1ULL) ? 1 : -1;
                int sj = ((sr >> b) & 1ULL) ? 1 : -1;
                int sk = ((sd >> b) & 1ULL) ? 1 : -1;
                E[b] += -si * sj - si * sk;
            }
        }
    }
}

static inline void total_magnetization_per_replica_naive(const std::vector<std::uint64_t> &spins, int L,
                                                         std::array<std::int64_t, WORD_BITS> &M) {
    M.fill(0);
    for (int idx = 0; idx < L * L; ++idx) {
        std::uint64_t s = spins[idx];
        for (int b = 0; b < WORD_BITS; ++b) {
            M[b] += ((s >> b) & 1ULL) ? 1 : -1;
        }
    }
}

static inline void transpose64x64_block(const std::array<std::uint64_t, WORD_BITS> &inp,
                                        std::array<std::uint64_t, WORD_BITS> &out) {
    out.fill(0ULL);
    for (int r = 0; r < WORD_BITS; ++r) {
        std::uint64_t x = inp[r];
        for (int c = 0; c < WORD_BITS; ++c) {
            if ((x >> c) & 1ULL) out[c] |= (1ULL << r);
        }
    }
}

static inline void total_magnetization_per_replica_popcount(const std::vector<std::uint64_t> &spins, int L,
                                                            std::array<std::int64_t, WORD_BITS> &M) {
    const int nsite = L * L;
    std::array<std::uint64_t, WORD_BITS> block{};
    std::array<std::uint64_t, WORD_BITS> trans{};
    std::array<std::int64_t, WORD_BITS> counts{};
    counts.fill(0);
    int idx = 0;
    for (int p = 0; p < nsite; ++p) {
        block[idx % WORD_BITS] = spins[p];
        ++idx;
        if (idx % WORD_BITS == 0) {
            transpose64x64_block(block, trans);
            for (int b = 0; b < WORD_BITS; ++b) counts[b] += bit_count_u64(trans[b]);
            block.fill(0ULL);
        }
    }
    int rem = idx % WORD_BITS;
    if (rem != 0) {
        for (int k = rem; k < WORD_BITS; ++k) block[k] = 0ULL;
        transpose64x64_block(block, trans);
        for (int b = 0; b < WORD_BITS; ++b) counts[b] += bit_count_u64(trans[b]);
    }
    for (int b = 0; b < WORD_BITS; ++b) M[b] = 2 * counts[b] - nsite;
}

static inline void init_replica_random(Replica &rep, int L, std::mt19937_64 &rng) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    rep.spins.assign(L * L, 0ULL);
    rep.M.fill(0);
    for (int idx = 0; idx < L * L; ++idx) {
        std::uint64_t w = 0ULL;
        for (int b = 0; b < WORD_BITS; ++b) {
            if (dist(rng) < 0.5) {
                w |= (1ULL << b);
                rep.M[b] += 1;
            } else {
                rep.M[b] -= 1;
            }
        }
        rep.spins[idx] = w;
    }
    total_energy_per_replica(rep.spins, L, rep.E);
}

static inline void sweep_checkerboard_incremental(Replica &rep, int L, std::mt19937_64 &rng) {
    double beta = rep.beta;
    double p4 = std::exp(-4.0 * beta);
    double p8 = std::exp(-8.0 * beta);
    for (int parity = 0; parity < 2; ++parity) {
        for (int i = 0; i < L; ++i) {
            int j0 = (i + parity) & 1;
            for (int j = j0; j < L; j += 2) {
                int idx = i * L + j;
                std::uint64_t s  = rep.spins[idx];
                std::uint64_t up = rep.spins[((i - 1 + L) % L) * L + j];
                std::uint64_t dn = rep.spins[((i + 1) % L) * L + j];
                std::uint64_t lf = rep.spins[i * L + ((j - 1 + L) % L)];
                std::uint64_t rt = rep.spins[i * L + ((j + 1) % L)];

                std::uint64_t a = s ^ up;
                std::uint64_t b = s ^ dn;
                std::uint64_t c = s ^ lf;
                std::uint64_t d = s ^ rt;

                std::uint64_t eq0 = mask_eq0(a, b, c, d);
                std::uint64_t eq1 = mask_eq1(a, b, c, d);
                std::uint64_t eq3 = mask_eq3(a, b, c, d);
                std::uint64_t eq4 = mask_eq4(a, b, c, d);
                std::uint64_t eq2 = ~(eq0 | eq1 | eq3 | eq4);

                std::uint64_t acc0 = eq0 & bernoulli_mask(p8, rng);
                std::uint64_t acc1 = eq1 & bernoulli_mask(p4, rng);
                std::uint64_t acc2 = eq2;
                std::uint64_t acc3 = eq3;
                std::uint64_t acc4 = eq4;
                std::uint64_t flip = acc0 | acc1 | acc2 | acc3 | acc4;

                add_int_delta_from_mask(rep.E, acc0,  8);
                add_int_delta_from_mask(rep.E, acc1,  4);
                add_int_delta_from_mask(rep.E, acc3, -4);
                add_int_delta_from_mask(rep.E, acc4, -8);

                std::uint64_t up_to_down = flip & s;
                std::uint64_t down_to_up = flip & (~s);
                add_int_delta_from_mask(rep.M, up_to_down, -2);
                add_int_delta_from_mask(rep.M, down_to_up,  2);

                rep.spins[idx] = s ^ flip;
            }
        }
    }
}

static inline int exchange_local(Replica &a, Replica &b, int L, std::mt19937_64 &rng) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::uint64_t acc_mask = 0ULL;
    for (int bit = 0; bit < WORD_BITS; ++bit) {
        double delta = (a.beta - b.beta) * static_cast<double>(a.E[bit] - b.E[bit]);
        bool accept = (delta >= 0.0) || (dist(rng) < std::exp(delta));
        if (accept) acc_mask |= (1ULL << bit);
    }
    int nacc = bit_count_u64(acc_mask);
    if (nacc == 0) return 0;
    const int nword = L * L;
    for (int idx = 0; idx < nword; ++idx) {
        std::uint64_t wa = a.spins[idx];
        std::uint64_t wb = b.spins[idx];
        a.spins[idx] = masked_swap_word(wa, wb, acc_mask);
        b.spins[idx] = masked_swap_word(wb, wa, acc_mask);
    }
    for (int bit = 0; bit < WORD_BITS; ++bit) {
        if ((acc_mask >> bit) & 1ULL) {
            std::swap(a.E[bit], b.E[bit]);
            std::swap(a.M[bit], b.M[bit]);
        }
    }
    return nacc;
}

static inline int exchange_boundary_with_right(Replica &local_last, int right_rank, int L,
                                               std::mt19937_64 &rng, MPI_Comm comm) {
    // exchange energies with right rank (which owns the next temperature replica)
    std::array<std::int64_t, WORD_BITS> E_right{};
    MPI_Sendrecv(local_last.E.data(), WORD_BITS, MPI_LONG_LONG, right_rank, 100,
                 E_right.data(),        WORD_BITS, MPI_LONG_LONG, right_rank, 100,
                 comm, MPI_STATUS_IGNORE);

    // lower-rank side computes acceptance mask and sends it to the right rank.
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::uint64_t acc_mask = 0ULL;
    for (int bit = 0; bit < WORD_BITS; ++bit) {
        double delta = (local_last.beta - 0.0) * 0.0; // placeholder overwritten below to silence warnings?
        (void)delta;
    }
    // Need beta of right replica as well.
    double beta_right = 0.0;
    MPI_Sendrecv(&local_last.beta, 1, MPI_DOUBLE, right_rank, 101,
                 &beta_right,       1, MPI_DOUBLE, right_rank, 101,
                 comm, MPI_STATUS_IGNORE);
    for (int bit = 0; bit < WORD_BITS; ++bit) {
        double delta = (local_last.beta - beta_right) * static_cast<double>(local_last.E[bit] - E_right[bit]);
        bool accept = (delta >= 0.0) || (dist(rng) < std::exp(delta));
        if (accept) acc_mask |= (1ULL << bit);
    }
    MPI_Send(&acc_mask, 1, MPI_UNSIGNED_LONG_LONG, right_rank, 102, comm);
    int nacc = bit_count_u64(acc_mask);
    if (nacc == 0) return 0;

    // exchange full configuration and magnetization data
    std::vector<std::uint64_t> spins_right(L * L, 0ULL);
    std::array<std::int64_t, WORD_BITS> M_right{};
    MPI_Sendrecv(local_last.spins.data(), L * L, MPI_UNSIGNED_LONG_LONG, right_rank, 103,
                 spins_right.data(),      L * L, MPI_UNSIGNED_LONG_LONG, right_rank, 103,
                 comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(local_last.M.data(), WORD_BITS, MPI_LONG_LONG, right_rank, 104,
                 M_right.data(),      WORD_BITS, MPI_LONG_LONG, right_rank, 104,
                 comm, MPI_STATUS_IGNORE);

    for (int idx = 0; idx < L * L; ++idx) {
        std::uint64_t wl = local_last.spins[idx];
        std::uint64_t wr = spins_right[idx];
        local_last.spins[idx] = masked_swap_word(wl, wr, acc_mask);
    }
    for (int bit = 0; bit < WORD_BITS; ++bit) {
        if ((acc_mask >> bit) & 1ULL) {
            local_last.E[bit] = E_right[bit];
            local_last.M[bit] = M_right[bit];
        }
    }
    return nacc;
}

static inline int exchange_boundary_with_left(Replica &local_first, int left_rank, int L, MPI_Comm comm) {
    // Right side of a boundary pair: send data, receive mask, apply masked swap.
    std::array<std::int64_t, WORD_BITS> E_left_dummy{};
    MPI_Sendrecv(local_first.E.data(), WORD_BITS, MPI_LONG_LONG, left_rank, 100,
                 E_left_dummy.data(),   WORD_BITS, MPI_LONG_LONG, left_rank, 100,
                 comm, MPI_STATUS_IGNORE);
    double beta_left = 0.0;
    MPI_Sendrecv(&local_first.beta, 1, MPI_DOUBLE, left_rank, 101,
                 &beta_left,         1, MPI_DOUBLE, left_rank, 101,
                 comm, MPI_STATUS_IGNORE);
    std::uint64_t acc_mask = 0ULL;
    MPI_Recv(&acc_mask, 1, MPI_UNSIGNED_LONG_LONG, left_rank, 102, comm, MPI_STATUS_IGNORE);
    int nacc = bit_count_u64(acc_mask);
    if (nacc == 0) return 0;

    std::vector<std::uint64_t> spins_left(L * L, 0ULL);
    std::array<std::int64_t, WORD_BITS> M_left{};
    MPI_Sendrecv(local_first.spins.data(), L * L, MPI_UNSIGNED_LONG_LONG, left_rank, 103,
                 spins_left.data(),        L * L, MPI_UNSIGNED_LONG_LONG, left_rank, 103,
                 comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(local_first.M.data(), WORD_BITS, MPI_LONG_LONG, left_rank, 104,
                 M_left.data(),        WORD_BITS, MPI_LONG_LONG, left_rank, 104,
                 comm, MPI_STATUS_IGNORE);

    for (int idx = 0; idx < L * L; ++idx) {
        std::uint64_t wr = local_first.spins[idx];
        std::uint64_t wl = spins_left[idx];
        local_first.spins[idx] = masked_swap_word(wr, wl, acc_mask);
    }
    for (int bit = 0; bit < WORD_BITS; ++bit) {
        if ((acc_mask >> bit) & 1ULL) {
            local_first.E[bit] = E_left_dummy[bit];
            local_first.M[bit] = M_left[bit];
        }
    }
    return nacc;
}

static inline double tc_exact() {
    return 2.0 * J / std::log(1.0 + std::sqrt(2.0));
}

static inline double exact_spontaneous_magnetization(double T) {
    const double Tc = tc_exact();
    if (T >= Tc) return 0.0;
    double beta = 1.0 / T;
    double x = std::sinh(2.0 * beta * J);
    return std::pow(1.0 - std::pow(x, -4.0), 0.125);
}

// Complete elliptic integral K(m) by AGM.
static inline double elliptic_K_from_m(double m) {
    if (m < 0.0) m = 0.0;
    if (m >= 1.0) m = std::nextafter(1.0, 0.0);
    double a = 1.0;
    double b = std::sqrt(1.0 - m);
    for (int it = 0; it < 50; ++it) {
        double an = 0.5 * (a + b);
        double bn = std::sqrt(a * b);
        if (std::abs(an - bn) < 1e-15 * an) {
            a = an;
            break;
        }
        a = an;
        b = bn;
    }
    return M_PI / (2.0 * a);
}

static inline double exact_energy_density(double T) {
    double beta = 1.0 / T;
    double x = 2.0 * beta * J;
    double sh = std::sinh(x);
    double ch = std::cosh(x);
    double th = std::tanh(x);
    double k = 2.0 * sh / (ch * ch);
    double K = elliptic_K_from_m(k * k);
    return -J / th * (1.0 + (2.0 / M_PI) * (2.0 * th * th - 1.0) * K);
}

static inline void exact_ladder_from_nrep(double Tmin, double Tmax, int nrep,
                                          std::vector<double> &Ts, std::vector<double> &betas) {
    Ts.resize(nrep);
    betas.resize(nrep);
    double beta_min = 1.0 / Tmax;
    double beta_max = 1.0 / Tmin;
    for (int i = 0; i < nrep; ++i) {
        double t = (nrep == 1) ? 0.0 : static_cast<double>(i) / (nrep - 1);
        betas[i] = beta_min + (beta_max - beta_min) * t;
        Ts[i] = 1.0 / betas[i];
    }
}

static inline void run_single_temp_short(int L, double beta, int n_therm, int n_meas, int n_skip,
                                         std::uint64_t seed, double &emean, double &var_e, double &sigma_E) {
    std::mt19937_64 rng(seed);
    Replica rep;
    rep.beta = beta;
    init_replica_random(rep, L, rng);
    for (int i = 0; i < n_therm; ++i) sweep_checkerboard_incremental(rep, L, rng);
    const int N = L * L;
    double e1 = 0.0, e2 = 0.0;
    long long ns = 0;
    for (int it = 0; it < n_meas; ++it) {
        for (int s = 0; s < n_skip; ++s) sweep_checkerboard_incremental(rep, L, rng);
        for (int b = 0; b < WORD_BITS; ++b) {
            double e = static_cast<double>(rep.E[b]) / N;
            e1 += e;
            e2 += e * e;
            ++ns;
        }
    }
    emean = e1 / ns;
    var_e = std::max(0.0, e2 / ns - emean * emean);
    sigma_E = std::sqrt(var_e) * N;
}

static inline void optimize_temperature_ladder(const Params &p, std::vector<double> &Ts,
                                               std::vector<double> &betas, PilotData &pilot) {
    const int ncoarse = p.coarse_points;
    double beta_min = 1.0 / p.Tmax;
    double beta_max = 1.0 / p.Tmin;
    pilot.beta_coarse.resize(ncoarse);
    pilot.T_coarse.resize(ncoarse);
    pilot.sigmaE.resize(ncoarse);
    pilot.evar_density.resize(ncoarse);
    pilot.emean_density.resize(ncoarse);
    pilot.S.assign(ncoarse, 0.0);

    for (int i = 0; i < ncoarse; ++i) {
        double t = (ncoarse == 1) ? 0.0 : static_cast<double>(i) / (ncoarse - 1);
        pilot.beta_coarse[i] = beta_min + (beta_max - beta_min) * t;
        pilot.T_coarse[i] = 1.0 / pilot.beta_coarse[i];
        double em, ve, sig;
        run_single_temp_short(std::min(p.L, 24), pilot.beta_coarse[i], p.pilot_therm, p.pilot_meas,
                              p.pilot_skip, p.seed + 101ULL * i + 7ULL, em, ve, sig);
        pilot.emean_density[i] = em;
        pilot.evar_density[i] = ve;
        pilot.sigmaE[i] = std::max(sig, 1e-12);
    }
    for (int i = 1; i < ncoarse; ++i) {
        double db = pilot.beta_coarse[i] - pilot.beta_coarse[i - 1];
        pilot.S[i] = pilot.S[i - 1] + 0.5 * (pilot.sigmaE[i] + pilot.sigmaE[i - 1]) * db;
    }
    double Stot = pilot.S.back();
    int nrep = std::max(6, static_cast<int>(std::ceil(Stot / p.target_c)) + 1);
    pilot.beta_ladder.resize(nrep);
    pilot.T_ladder.resize(nrep);
    for (int k = 0; k < nrep; ++k) {
        double st = (nrep == 1) ? 0.0 : Stot * static_cast<double>(k) / (nrep - 1);
        auto it = std::lower_bound(pilot.S.begin(), pilot.S.end(), st);
        int hi = static_cast<int>(it - pilot.S.begin());
        if (hi <= 0) {
            pilot.beta_ladder[k] = pilot.beta_coarse.front();
        } else if (hi >= ncoarse) {
            pilot.beta_ladder[k] = pilot.beta_coarse.back();
        } else {
            int lo = hi - 1;
            double frac = (st - pilot.S[lo]) / (pilot.S[hi] - pilot.S[lo]);
            pilot.beta_ladder[k] = pilot.beta_coarse[lo] + frac * (pilot.beta_coarse[hi] - pilot.beta_coarse[lo]);
        }
        pilot.T_ladder[k] = 1.0 / pilot.beta_ladder[k];
    }
    Ts = pilot.T_ladder;
    betas = pilot.beta_ladder;
}

static inline void parse_args(int argc, char **argv, Params &p) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const std::string &opt) {
            if (i + 1 >= argc) throw std::runtime_error("Missing value for " + opt);
            return std::string(argv[++i]);
        };
        if (a == "--L") p.L = std::stoi(need(a));
        else if (a == "--Tmin") p.Tmin = std::stod(need(a));
        else if (a == "--Tmax") p.Tmax = std::stod(need(a));
        else if (a == "--nrep") p.nrep = std::stoi(need(a));
        else if (a == "--target-c") p.target_c = std::stod(need(a));
        else if (a == "--pilot-therm") p.pilot_therm = std::stoi(need(a));
        else if (a == "--pilot-meas") p.pilot_meas = std::stoi(need(a));
        else if (a == "--pilot-skip") p.pilot_skip = std::stoi(need(a));
        else if (a == "--therm") p.therm = std::stoi(need(a));
        else if (a == "--meas") p.meas = std::stoi(need(a));
        else if (a == "--skip") p.skip = std::stoi(need(a));
        else if (a == "--seed") p.seed = static_cast<std::uint64_t>(std::stoull(need(a)));
        else if (a == "--resync-every") p.resync_every = std::stoi(need(a));
        else if (a == "--use-popcount-resync") p.use_popcount_resync = (std::stoi(need(a)) != 0);
        else if (a == "--coarse-points") p.coarse_points = std::stoi(need(a));
        else if (a == "--output") p.output = need(a);
        else if (a == "--help") {
            if (true) {
                std::cout << "Options: --L --Tmin --Tmax --nrep --target-c --pilot-therm --pilot-meas --pilot-skip --therm --meas --skip --seed --resync-every --use-popcount-resync --coarse-points --output\n";
            }
            MPI_Abort(MPI_COMM_WORLD, 0);
        }
    }
}

static inline void validate_incremental_state(int rank, std::ostringstream &os) {
    if (rank != 0) return;
    std::mt19937_64 rng(1234);
    Replica rep;
    rep.beta = 0.4;
    init_replica_random(rep, 8, rng);
    for (int i = 0; i < 12; ++i) sweep_checkerboard_incremental(rep, 8, rng);
    std::array<std::int64_t, WORD_BITS> E2{}, M2{}, M3{};
    total_energy_per_replica(rep.spins, 8, E2);
    total_magnetization_per_replica_naive(rep.spins, 8, M2);
    total_magnetization_per_replica_popcount(rep.spins, 8, M3);
    long long maxe = 0, maxm2 = 0, maxm3 = 0;
    for (int b = 0; b < WORD_BITS; ++b) {
        maxe = std::max(maxe, std::llabs(rep.E[b] - E2[b]));
        maxm2 = std::max(maxm2, std::llabs(rep.M[b] - M2[b]));
        maxm3 = std::max(maxm3, std::llabs(rep.M[b] - M3[b]));
    }
    os << "validation.max_energy_mismatch=" << maxe << "\n";
    os << "validation.max_mag_naive_mismatch=" << maxm2 << "\n";
    os << "validation.max_mag_popcount_mismatch=" << maxm3 << "\n";
}

static inline void ensure_output_dir(const std::string &path) {
    std::string cmd = "mkdir -p '" + path + "'";
    int ret = std::system(cmd.c_str());
    (void)ret;
}

static inline void write_pilot_csv(const std::string &dir, const PilotData &pilot) {
    std::ofstream f(dir + "/pilot_ladder.csv");
    f << "T,beta,sigmaE,evar_density,emean_density,S\n";
    for (std::size_t i = 0; i < pilot.beta_coarse.size(); ++i) {
        f << std::setprecision(17)
          << pilot.T_coarse[i] << ',' << pilot.beta_coarse[i] << ',' << pilot.sigmaE[i] << ','
          << pilot.evar_density[i] << ',' << pilot.emean_density[i] << ',' << pilot.S[i] << '\n';
    }
}

static inline void write_observables_csv(const std::string &dir, const std::vector<double> &Ts,
                                         const std::vector<double> &betas, const Observables &obs) {
    std::ofstream f(dir + "/observables.csv");
    f << "T,beta,e_mc,e_err,mabs,mabs_err,sqrt_m2,cv,chi,chi_conn,m2,m4,binder,e_exact,m_exact,left_exchange_ratio,right_exchange_ratio\n";
    for (std::size_t i = 0; i < Ts.size(); ++i) {
        double left = (i == 0) ? std::numeric_limits<double>::quiet_NaN() : obs.exch_ratio[i - 1];
        double right = (i + 1 >= Ts.size()) ? std::numeric_limits<double>::quiet_NaN() : obs.exch_ratio[i];
        f << std::setprecision(17)
          << Ts[i] << ',' << betas[i] << ','
          << obs.e_mean[i] << ',' << obs.e_err[i] << ','
          << obs.mabs_mean[i] << ',' << obs.mabs_err[i] << ','
          << obs.sqrt_m2[i] << ',' << obs.cv[i] << ',' << obs.chi[i] << ',' << obs.chi_conn[i] << ','
          << obs.m2_mean[i] << ',' << obs.m4_mean[i] << ',' << obs.binder[i] << ','
          << exact_energy_density(Ts[i]) << ',' << exact_spontaneous_magnetization(Ts[i]) << ','
          << left << ',' << right << '\n';
    }
}

static inline void write_summary_json(const std::string &dir, const Params &p, int world_size,
                                      const std::vector<double> &Ts, const Observables &obs) {
    std::ofstream f(dir + "/summary.json");
    f << "{\n";
    f << "  \"L\": " << p.L << ",\n";
    f << "  \"Tmin\": " << p.Tmin << ",\n";
    f << "  \"Tmax\": " << p.Tmax << ",\n";
    f << "  \"nrep\": " << Ts.size() << ",\n";
    f << "  \"ncore\": " << world_size << ",\n";
    f << "  \"local_capacity\": " << ceil_div(static_cast<int>(Ts.size()), world_size) << ",\n";
    f << "  \"Tc_exact\": " << std::setprecision(17) << tc_exact() << ",\n";
    f << "  \"exchange_ratio\": [";
    for (std::size_t i = 0; i < obs.exch_ratio.size(); ++i) {
        if (i) f << ", ";
        f << obs.exch_ratio[i];
    }
    f << "],\n";
    f << "  \"Ts\": [";
    for (std::size_t i = 0; i < Ts.size(); ++i) {
        if (i) f << ", ";
        f << Ts[i];
    }
    f << "]\n";
    f << "}\n";
}

static inline void write_plot_script(const std::string &dir) {
    std::ofstream f(dir + "/plot_observables.py");
    f << R"PY(#!/usr/bin/env python3
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

base = Path(__file__).resolve().parent
rows = []
with open(base / 'observables.csv') as fp:
    r = csv.DictReader(fp)
    for row in r:
        rows.append(row)

def arr(key):
    out = []
    for row in rows:
        v = row[key]
        if v == 'nan' or v == 'NaN' or v == '':
            out.append(np.nan)
        else:
            out.append(float(v))
    return np.array(out, dtype=float)

T = arr('T')
e_mc = arr('e_mc'); e_err = arr('e_err'); e_ex = arr('e_exact')
mabs = arr('mabs'); mabs_err = arr('mabs_err'); sqrtm2 = arr('sqrt_m2'); mex = arr('m_exact')
cv = arr('cv'); chi = arr('chi'); chi_conn = arr('chi_conn'); binder = arr('binder')
left = arr('left_exchange_ratio'); right = arr('right_exchange_ratio')
Tc = 2.0 / np.log(1.0 + np.sqrt(2.0))
mids = 0.5 * (T[:-1] + T[1:])
exch = right[:-1]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
ax = axes[0,0]
ax.errorbar(T, e_mc, yerr=e_err, fmt='o', capsize=3, label='MC')
ax.plot(T, e_ex, '-', lw=2, label='Exact')
ax.axvline(Tc, color='k', ls='--', alpha=0.6, label=r'$T_c$')
ax.set_xlabel('T'); ax.set_ylabel(r'$e=\langle E\rangle/N$'); ax.set_title('Energy density'); ax.legend()

ax = axes[0,1]
ax.errorbar(T, mabs, yerr=mabs_err, fmt='o', capsize=3, label=r'$\langle |m| \rangle$')
ax.plot(T, sqrtm2, 's-', lw=1.5, label=r'$\sqrt{\langle m^2\rangle}$')
ax.plot(T, mex, '-', lw=2, label='Exact spontaneous magnetization')
ax.axvline(Tc, color='k', ls='--', alpha=0.6)
ax.set_xlabel('T'); ax.set_ylabel('magnetization'); ax.set_title('Magnetization comparison'); ax.legend()

ax = axes[0,2]
ax.plot(T, cv, 'o-', label=r'$C_v$')
ax.axvline(Tc, color='k', ls='--', alpha=0.6)
ax.set_xlabel('T'); ax.set_ylabel(r'$C_v$'); ax.set_title('Specific heat'); ax.legend()

ax = axes[1,0]
ax.plot(T, chi, 'o-', label=r'$\chi=\beta N\langle m^2\rangle$')
ax.plot(T, chi_conn, 's--', label=r'$\chi_{conn}$')
ax.axvline(Tc, color='k', ls='--', alpha=0.6)
ax.set_xlabel('T'); ax.set_ylabel(r'$\chi$'); ax.set_title('Susceptibility'); ax.legend()

ax = axes[1,1]
ax.plot(T, binder, 'o-', label='Binder cumulant')
ax.axvline(Tc, color='k', ls='--', alpha=0.6)
ax.set_xlabel('T'); ax.set_ylabel(r'$U_4$'); ax.set_title('Binder cumulant'); ax.legend()

ax = axes[1,2]
ax.plot(mids, exch, 'o-')
ax.axvline(Tc, color='k', ls='--', alpha=0.6)
ax.set_xlabel('midpoint temperature'); ax.set_ylabel('exchange ratio'); ax.set_title('Replica exchange acceptance')

fig.tight_layout()
fig.savefig(base / 'observables.png', dpi=180, bbox_inches='tight')
print(base / 'observables.png')
)PY";
}

static inline Observables run_simulation(const Params &p, const std::vector<double> &betas,
                                         int rank, int world_size, MPI_Comm comm) {
    const int nrep = static_cast<int>(betas.size());
    const int local_cap = ceil_div(nrep, world_size);
    const int start = rank * local_cap;
    const int actual_n = std::max(0, std::min(local_cap, nrep - start));
    std::vector<Replica> reps(actual_n);
    std::mt19937_64 rng(p.seed + 1000003ULL * rank + 17ULL);
    for (int i = 0; i < actual_n; ++i) {
        reps[i].gid = start + i;
        reps[i].beta = betas[start + i];
        init_replica_random(reps[i], p.L, rng);
    }

    std::vector<double> e1(nrep, 0.0), e2(nrep, 0.0), mabs1(nrep, 0.0), mabs2(nrep, 0.0), m21(nrep, 0.0), m41(nrep, 0.0);
    std::vector<long long> ns(nrep, 0LL), exch_try(nrep > 0 ? nrep - 1 : 0, 0LL), exch_acc(nrep > 0 ? nrep - 1 : 0, 0LL);
    const int N = p.L * p.L;

    auto maybe_resync = [&](int counter) {
        if (!p.use_popcount_resync || p.resync_every <= 0) return;
        if (counter % p.resync_every != 0) return;
        for (auto &rep : reps) {
            total_magnetization_per_replica_popcount(rep.spins, p.L, rep.M);
            total_energy_per_replica(rep.spins, p.L, rep.E);
        }
    };

    for (int it = 0; it < p.therm; ++it) {
        for (auto &rep : reps) sweep_checkerboard_incremental(rep, p.L, rng);
        int parity = it & 1;
        // local neighboring pairs
        for (int li = 0; li + 1 < actual_n; ++li) {
            int g = start + li;
            if ((g & 1) == parity) {
                exch_try[g] += WORD_BITS;
                exch_acc[g] += exchange_local(reps[li], reps[li + 1], p.L, rng);
            }
        }
        // boundary pair with right rank (handled by left rank)
        if (actual_n > 0) {
            int g_last = start + actual_n - 1;
            if (g_last < nrep - 1 && (g_last & 1) == parity) {
                exch_try[g_last] += WORD_BITS;
                if (rank < world_size - 1 && (rank + 1) * local_cap < nrep) {
                    exch_acc[g_last] += exchange_boundary_with_right(reps.back(), rank + 1, p.L, rng, comm);
                }
            }
        }
        // boundary pair with left rank (passive side)
        if (actual_n > 0) {
            int g_left = start - 1;
            if (start > 0 && (g_left & 1) == parity) {
                if (rank > 0) {
                    exchange_boundary_with_left(reps.front(), rank - 1, p.L, comm);
                }
            }
        }
        maybe_resync(it + 1);
        MPI_Barrier(comm);
    }

    for (int it = 0; it < p.meas; ++it) {
        for (int s = 0; s < p.skip; ++s) {
            for (auto &rep : reps) sweep_checkerboard_incremental(rep, p.L, rng);
            int parity = (it + s) & 1;
            for (int li = 0; li + 1 < actual_n; ++li) {
                int g = start + li;
                if ((g & 1) == parity) {
                    exch_try[g] += WORD_BITS;
                    exch_acc[g] += exchange_local(reps[li], reps[li + 1], p.L, rng);
                }
            }
            if (actual_n > 0) {
                int g_last = start + actual_n - 1;
                if (g_last < nrep - 1 && (g_last & 1) == parity) {
                    exch_try[g_last] += WORD_BITS;
                    if (rank < world_size - 1 && (rank + 1) * local_cap < nrep) {
                        exch_acc[g_last] += exchange_boundary_with_right(reps.back(), rank + 1, p.L, rng, comm);
                    }
                }
            }
            if (actual_n > 0) {
                int g_left = start - 1;
                if (start > 0 && (g_left & 1) == parity) {
                    if (rank > 0) exchange_boundary_with_left(reps.front(), rank - 1, p.L, comm);
                }
            }
            MPI_Barrier(comm);
        }
        maybe_resync(it + 1);
        for (int li = 0; li < actual_n; ++li) {
            int g = start + li;
            for (int bit = 0; bit < WORD_BITS; ++bit) {
                double e = static_cast<double>(reps[li].E[bit]) / N;
                double m = static_cast<double>(reps[li].M[bit]) / N;
                double ma = std::abs(m);
                e1[g] += e; e2[g] += e * e;
                mabs1[g] += ma; mabs2[g] += ma * ma;
                m21[g] += m * m; m41[g] += m * m * m * m;
                ns[g] += 1;
            }
        }
    }

    // Reduce to root.
    Observables obs;
    obs.e_mean.resize(nrep); obs.e_err.resize(nrep); obs.mabs_mean.resize(nrep); obs.mabs_err.resize(nrep);
    obs.cv.resize(nrep); obs.chi.resize(nrep); obs.chi_conn.resize(nrep); obs.m2_mean.resize(nrep);
    obs.sqrt_m2.resize(nrep); obs.m4_mean.resize(nrep); obs.binder.resize(nrep); obs.exch_ratio.resize(std::max(0, nrep - 1));

    std::vector<double> ge1(nrep), ge2(nrep), gmabs1(nrep), gmabs2(nrep), gm21(nrep), gm41(nrep);
    std::vector<long long> gns(nrep), gtry(std::max(0, nrep - 1)), gacc(std::max(0, nrep - 1));

    MPI_Reduce(e1.data(), ge1.data(), nrep, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(e2.data(), ge2.data(), nrep, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(mabs1.data(), gmabs1.data(), nrep, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(mabs2.data(), gmabs2.data(), nrep, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(m21.data(), gm21.data(), nrep, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(m41.data(), gm41.data(), nrep, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(ns.data(), gns.data(), nrep, MPI_LONG_LONG, MPI_SUM, 0, comm);
    if (nrep > 1) {
        MPI_Reduce(exch_try.data(), gtry.data(), nrep - 1, MPI_LONG_LONG, MPI_SUM, 0, comm);
        MPI_Reduce(exch_acc.data(), gacc.data(), nrep - 1, MPI_LONG_LONG, MPI_SUM, 0, comm);
    }

    if (rank == 0) {
        for (int i = 0; i < nrep; ++i) {
            double inv_ns = 1.0 / static_cast<double>(gns[i]);
            obs.e_mean[i] = ge1[i] * inv_ns;
            obs.mabs_mean[i] = gmabs1[i] * inv_ns;
            obs.m2_mean[i] = gm21[i] * inv_ns;
            obs.m4_mean[i] = gm41[i] * inv_ns;
            obs.sqrt_m2[i] = std::sqrt(std::max(0.0, obs.m2_mean[i]));
            double ve = std::max(0.0, ge2[i] * inv_ns - obs.e_mean[i] * obs.e_mean[i]);
            double vabs = std::max(0.0, gmabs2[i] * inv_ns - obs.mabs_mean[i] * obs.mabs_mean[i]);
            obs.e_err[i] = std::sqrt(ve * inv_ns);
            obs.mabs_err[i] = std::sqrt(vabs * inv_ns);
            double beta = betas[i];
            obs.cv[i] = beta * beta * N * ve;
            obs.chi[i] = beta * N * obs.m2_mean[i];
            obs.chi_conn[i] = beta * N * (obs.m2_mean[i] - obs.mabs_mean[i] * obs.mabs_mean[i]);
            obs.binder[i] = (obs.m2_mean[i] > 1e-30) ? 1.0 - obs.m4_mean[i] / (3.0 * obs.m2_mean[i] * obs.m2_mean[i]) : 0.0;
        }
        for (int i = 0; i < nrep - 1; ++i) {
            obs.exch_ratio[i] = (gtry[i] > 0) ? static_cast<double>(gacc[i]) / static_cast<double>(gtry[i]) : 0.0;
        }
    }
    return obs;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    Params p;
    try {
        parse_args(argc, argv, p);
    } catch (const std::exception &e) {
        if (rank == 0) std::cerr << "Argument error: " << e.what() << std::endl;
        MPI_Finalize();
        return 1;
    }

    std::vector<double> Ts, betas;
    PilotData pilot;
    std::ostringstream validation_stream;
    validate_incremental_state(rank, validation_stream);

    if (rank == 0) {
        if (p.nrep > 0) {
            exact_ladder_from_nrep(p.Tmin, p.Tmax, p.nrep, Ts, betas);
        } else {
            optimize_temperature_ladder(p, Ts, betas, pilot);
        }
    }

    int nrep = 0;
    if (rank == 0) nrep = static_cast<int>(betas.size());
    MPI_Bcast(&nrep, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) { Ts.resize(nrep); betas.resize(nrep); }
    if (nrep > 0) {
        MPI_Bcast(betas.data(), nrep, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank != 0) for (int i = 0; i < nrep; ++i) Ts[i] = 1.0 / betas[i];
    }

    Observables obs = run_simulation(p, betas, rank, world_size, MPI_COMM_WORLD);

    if (rank == 0) {
        ensure_output_dir(p.output);
        if (p.nrep <= 0) write_pilot_csv(p.output, pilot);
        write_observables_csv(p.output, Ts, betas, obs);
        write_summary_json(p.output, p, world_size, Ts, obs);
        write_plot_script(p.output);
        std::ofstream ftxt(p.output + "/summary.txt");
        ftxt << "2D square-lattice Ising (MSC + OpenMPI PT)\n";
        ftxt << "L = " << p.L << "\n";
        ftxt << "nrep = " << nrep << "\n";
        ftxt << "ncore = " << world_size << "\n";
        ftxt << "local_capacity = " << ceil_div(nrep, world_size) << "\n";
        ftxt << "Tc exact = " << std::setprecision(17) << tc_exact() << "\n";
        ftxt << validation_stream.str();
        ftxt << "Temperature ladder:\n";
        for (double T : Ts) ftxt << "  " << std::setprecision(17) << T << "\n";
        ftxt << "Exchange ratios:\n";
        for (int i = 0; i < nrep - 1; ++i) {
            ftxt << "  " << Ts[i] << " <-> " << Ts[i + 1] << ": " << obs.exch_ratio[i] << "\n";
        }
        std::cout << "Output directory: " << p.output << "\n";
        std::cout << "observables.csv written. To create PNG, run:\n";
        std::cout << "  python3 " << p.output << "/plot_observables.py\n";
    }

    MPI_Finalize();
    return 0;
}
