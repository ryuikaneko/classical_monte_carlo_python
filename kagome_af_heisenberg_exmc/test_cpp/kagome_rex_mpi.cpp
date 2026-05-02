
#include <mpi.h>
#include <algorithm>
#include <array>
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

struct Vec3 { double x, y, z; };
static inline Vec3 operator/(const Vec3& a, double c) { return {a.x/c, a.y/c, a.z/c}; }
static inline double dot(const Vec3& a, const Vec3& b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
static inline double norm2(const Vec3& a) { return dot(a,a); }
static inline double norm(const Vec3& a) { return std::sqrt(norm2(a)); }

struct Bond { int i, j; };
struct KagomeLattice {
    int L = 0;
    int N = 0;
    std::vector<Bond> bonds;
    std::vector<std::array<int,4>> neighbors;
};

static inline int modp(int a, int m) {
    int r = a % m;
    return (r < 0) ? r + m : r;
}
static inline int site_index(int x, int y, int s, int L) {
    return 3 * (x + L * y) + s;
}

KagomeLattice build_kagome(int L) {
    if (L <= 0) throw std::runtime_error("L must be positive");
    KagomeLattice lat;
    lat.L = L;
    lat.N = 3 * L * L;
    lat.neighbors.assign(lat.N, {{-1,-1,-1,-1}});

    // Correct unique nearest-neighbor bond set for kagome lattice with basis
    // A=(0,0), B=(1/2,0), C=(0,1/2) in fractional (a1,a2) coordinates.
    // For each unit cell (x,y), add these 6 bonds exactly once:
    //   A(x,y)-B(x,y)
    //   A(x,y)-B(x-1,y)
    //   A(x,y)-C(x,y)
    //   A(x,y)-C(x,y-1)
    //   B(x,y)-C(x,y)
    //   B(x,y)-C(x+1,y-1)
    // Then total bonds = 6*L^2 = N*z/2 with z=4.
    lat.bonds.reserve(6 * L * L);
    for (int x = 0; x < L; ++x) {
        for (int y = 0; y < L; ++y) {
            int A = site_index(x, y, 0, L);
            int B = site_index(x, y, 1, L);
            int C = site_index(x, y, 2, L);
            int B_left = site_index(modp(x - 1, L), y, 1, L);
            int C_down = site_index(x, modp(y - 1, L), 2, L);
            int C_diag = site_index(modp(x + 1, L), modp(y - 1, L), 2, L);
            lat.bonds.push_back({A, B});
            lat.bonds.push_back({A, B_left});
            lat.bonds.push_back({A, C});
            lat.bonds.push_back({A, C_down});
            lat.bonds.push_back({B, C});
            lat.bonds.push_back({B, C_diag});
        }
    }

    std::vector<int> cnt(lat.N, 0);
    for (const auto& b : lat.bonds) {
        int c1 = cnt[b.i]++;
        int c2 = cnt[b.j]++;
        if (c1 >= 4 || c2 >= 4) {
            throw std::runtime_error("Neighbor overflow while building kagome lattice");
        }
        lat.neighbors[b.i][c1] = b.j;
        lat.neighbors[b.j][c2] = b.i;
    }
    for (int i = 0; i < lat.N; ++i) {
        if (cnt[i] != 4) {
            std::ostringstream oss;
            oss << "Site " << i << " has coordination " << cnt[i] << " (expected 4)";
            throw std::runtime_error(oss.str());
        }
    }
    return lat;
}

struct Params {
    int L = 6;
    double J = 1.0;
    int nrep = 28;
    double Tmin = 1e-3;
    double Tmax = 0.5;
    int therm = 20000;
    int meas = 100000;
    int sample_every = 10;
    int samples_per_block = 100;
    int seed = 12345;
    bool do_or = true;
    bool do_rex = true;
    bool debug_lattice = false;
    std::string out_prefix = "kagome_rex_mpi";
};

struct Distribution {
    int world_size = 1, world_rank = 0, nrep = 0;
    int local_cap = 0, local_n = 0;
    int g_begin = 0, g_end = 0;
};
Distribution make_distribution(int nrep, int world_size, int world_rank) {
    Distribution d;
    d.world_size = world_size;
    d.world_rank = world_rank;
    d.nrep = nrep;
    d.local_cap = (nrep + world_size - 1) / world_size;
    d.g_begin = world_rank * d.local_cap;
    d.g_end = std::min(nrep, d.g_begin + d.local_cap);
    d.local_n = std::max(0, d.g_end - d.g_begin);
    return d;
}

std::vector<double> geomspace(double Tmin, double Tmax, int nrep) {
    std::vector<double> T(nrep);
    if (nrep == 1) { T[0] = Tmin; return T; }
    double log_min = std::log(Tmin), log_max = std::log(Tmax);
    for (int i = 0; i < nrep; ++i) {
        double a = static_cast<double>(i) / static_cast<double>(nrep - 1);
        T[i] = std::exp((1.0 - a) * log_min + a * log_max);
    }
    return T;
}

struct Replica {
    std::vector<Vec3> spins;
    double energy = 0.0;
    double T = 0.0;
};

Vec3 marsaglia_unit_vector(std::mt19937_64& rng) {
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    while (true) {
        double u = 2.0 * unif(rng) - 1.0;
        double v = 2.0 * unif(rng) - 1.0;
        double s = u*u + v*v;
        if (s > 0.0 && s < 1.0) {
            double fac = std::sqrt(1.0 - s);
            return {2.0*u*fac, 2.0*v*fac, 1.0 - 2.0*s};
        }
    }
}

double total_energy_replica(const std::vector<Vec3>& spins, const std::vector<Bond>& bonds, double J) {
    double E = 0.0;
    for (const auto& b : bonds) E += J * dot(spins[b.i], spins[b.j]);
    return E;
}
Vec3 local_field(const std::vector<Vec3>& spins, const std::vector<std::array<int,4>>& neighbors, int i, double J) {
    Vec3 h{0.0,0.0,0.0};
    for (int k = 0; k < 4; ++k) {
        int j = neighbors[i][k];
        h.x += J * spins[j].x;
        h.y += J * spins[j].y;
        h.z += J * spins[j].z;
    }
    return h;
}

long long metropolis_sweep(std::vector<Replica>& reps, const KagomeLattice& lat, double J, std::mt19937_64& rng) {
    if (reps.empty()) return 0;
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    std::uniform_int_distribution<int> pick_site(0, lat.N - 1);
    long long acc = 0;
    for (auto& rep : reps) {
        double beta = 1.0 / rep.T;
        for (int t = 0; t < lat.N; ++t) {
            int i = pick_site(rng);
            Vec3 old = rep.spins[i];
            Vec3 h = local_field(rep.spins, lat.neighbors, i, J);
            double old_dot = dot(old, h);
            Vec3 neu = marsaglia_unit_vector(rng);
            double new_dot = dot(neu, h);
            double dE = new_dot - old_dot;
            if (dE <= 0.0 || unif(rng) < std::exp(-beta * dE)) {
                rep.spins[i] = neu;
                rep.energy += dE;
                ++acc;
            }
        }
    }
    return acc;
}

long long overrelaxation_sweep(std::vector<Replica>& reps, const KagomeLattice& lat, double J, std::mt19937_64& rng) {
    if (reps.empty()) return 0;
    std::uniform_int_distribution<int> pick_site(0, lat.N - 1);
    long long changed = 0;
    for (auto& rep : reps) {
        for (int t = 0; t < lat.N; ++t) {
            int i = pick_site(rng);
            Vec3 h = local_field(rep.spins, lat.neighbors, i, J);
            double h2 = norm2(h);
            if (h2 < 1e-30) continue;
            Vec3 s = rep.spins[i];
            double sh = dot(s, h);
            double fac = 2.0 * sh / h2;
            Vec3 n{fac*h.x - s.x, fac*h.y - s.y, fac*h.z - s.z};
            rep.spins[i] = n / norm(n);
            ++changed;
        }
    }
    return changed;
}

long long replica_exchange_internal(std::vector<Replica>& reps, int parity, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    long long acc = 0;
    for (int r = parity; r + 1 < (int)reps.size(); r += 2) {
        double beta_r = 1.0 / reps[r].T;
        double beta_s = 1.0 / reps[r+1].T;
        double d = (beta_r - beta_s) * (reps[r+1].energy - reps[r].energy);
        if (d >= 0.0 || unif(rng) < std::exp(d)) {
            std::swap(reps[r], reps[r+1]);
            ++acc;
        }
    }
    return acc;
}

void pack_spins(const std::vector<Vec3>& spins, std::vector<double>& buf) {
    buf.resize(3 * spins.size());
    for (size_t i = 0; i < spins.size(); ++i) {
        buf[3*i+0] = spins[i].x;
        buf[3*i+1] = spins[i].y;
        buf[3*i+2] = spins[i].z;
    }
}
void unpack_spins(const std::vector<double>& buf, std::vector<Vec3>& spins) {
    size_t N = buf.size() / 3;
    spins.resize(N);
    for (size_t i = 0; i < N; ++i) {
        spins[i].x = buf[3*i+0];
        spins[i].y = buf[3*i+1];
        spins[i].z = buf[3*i+2];
    }
}

long long replica_exchange_boundary(std::vector<Replica>& reps, const Distribution& dist, int parity,
                                    std::mt19937_64& rng, MPI_Comm comm) {
    const int rank = dist.world_rank, size = dist.world_size;
    if (size <= 1) return 0;
    long long acc = 0;
    const int local_n = dist.local_n, g_begin = dist.g_begin, g_end = dist.g_end;

    if (local_n > 0 && rank + 1 < size && g_end < dist.nrep) {
        int g = g_end - 1;
        if ((g % 2) == parity) {
            double partner_energy = 0.0, partner_T = 0.0;
            MPI_Sendrecv(&reps.back().energy, 1, MPI_DOUBLE, rank+1, 100,
                         &partner_energy, 1, MPI_DOUBLE, rank+1, 100, comm, MPI_STATUS_IGNORE);
            MPI_Sendrecv(&reps.back().T, 1, MPI_DOUBLE, rank+1, 101,
                         &partner_T, 1, MPI_DOUBLE, rank+1, 101, comm, MPI_STATUS_IGNORE);
            double beta_r = 1.0 / reps.back().T;
            double beta_s = 1.0 / partner_T;
            double d = (beta_r - beta_s) * (partner_energy - reps.back().energy);
            int accept = 0;
            std::uniform_real_distribution<double> unif(0.0, 1.0);
            if (d >= 0.0 || unif(rng) < std::exp(d)) accept = 1;
            MPI_Send(&accept, 1, MPI_INT, rank+1, 102, comm);
            if (accept) {
                std::vector<double> sendbuf, recvbuf;
                pack_spins(reps.back().spins, sendbuf);
                recvbuf.resize(sendbuf.size());
                MPI_Sendrecv(sendbuf.data(), (int)sendbuf.size(), MPI_DOUBLE, rank+1, 103,
                             recvbuf.data(), (int)recvbuf.size(), MPI_DOUBLE, rank+1, 103,
                             comm, MPI_STATUS_IGNORE);
                unpack_spins(recvbuf, reps.back().spins);
                reps.back().energy = partner_energy;
                reps.back().T = partner_T;
                ++acc;
            }
        }
    }
    if (local_n > 0 && rank - 1 >= 0 && g_begin > 0) {
        int g = g_begin - 1;
        if ((g % 2) == parity) {
            double partner_energy = 0.0, partner_T = 0.0;
            MPI_Sendrecv(&reps.front().energy, 1, MPI_DOUBLE, rank-1, 100,
                         &partner_energy, 1, MPI_DOUBLE, rank-1, 100, comm, MPI_STATUS_IGNORE);
            MPI_Sendrecv(&reps.front().T, 1, MPI_DOUBLE, rank-1, 101,
                         &partner_T, 1, MPI_DOUBLE, rank-1, 101, comm, MPI_STATUS_IGNORE);
            int accept = 0;
            MPI_Recv(&accept, 1, MPI_INT, rank-1, 102, comm, MPI_STATUS_IGNORE);
            if (accept) {
                std::vector<double> sendbuf, recvbuf;
                pack_spins(reps.front().spins, sendbuf);
                recvbuf.resize(sendbuf.size());
                MPI_Sendrecv(sendbuf.data(), (int)sendbuf.size(), MPI_DOUBLE, rank-1, 103,
                             recvbuf.data(), (int)recvbuf.size(), MPI_DOUBLE, rank-1, 103,
                             comm, MPI_STATUS_IGNORE);
                unpack_spins(recvbuf, reps.front().spins);
                reps.front().energy = partner_energy;
                reps.front().T = partner_T;
            }
        }
    }
    return acc;
}

struct BlockMeasureAgg {
    int nrep = 0;
    int nblock = 0;
    std::vector<double> sumE_blk;
    std::vector<double> sumE2_blk;
    std::vector<long long> cnt_blk;

    BlockMeasureAgg() = default;
    BlockMeasureAgg(int nrep_, int nblock_)
        : nrep(nrep_), nblock(nblock_),
          sumE_blk(nrep_ * nblock_, 0.0),
          sumE2_blk(nrep_ * nblock_, 0.0),
          cnt_blk(nrep_ * nblock_, 0) {}

    int temp_index(double T, const std::vector<double>& Tgrid) const {
        for (int i = 0; i < nrep; ++i) if (T == Tgrid[i]) return i;
        double best = std::abs(T - Tgrid[0]);
        int idx = 0;
        for (int i = 1; i < nrep; ++i) {
            double d = std::abs(T - Tgrid[i]);
            if (d < best) { best = d; idx = i; }
        }
        return idx;
    }

    void add(double T, double E, const std::vector<double>& Tgrid, int block_id) {
        int tid = temp_index(T, Tgrid);
        int idx = tid * nblock + block_id;
        sumE_blk[idx] += E;
        sumE2_blk[idx] += E * E;
        cnt_blk[idx] += 1;
    }
};

Params parse_args(int argc, char** argv, int rank) {
    Params p;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need_val = [&](const std::string& key) {
            if (i + 1 >= argc) throw std::runtime_error("Missing value after " + key);
            return std::string(argv[++i]);
        };
        if (a == "--L") p.L = std::stoi(need_val(a));
        else if (a == "--J") p.J = std::stod(need_val(a));
        else if (a == "--nrep") p.nrep = std::stoi(need_val(a));
        else if (a == "--Tmin") p.Tmin = std::stod(need_val(a));
        else if (a == "--Tmax") p.Tmax = std::stod(need_val(a));
        else if (a == "--therm") p.therm = std::stoi(need_val(a));
        else if (a == "--meas") p.meas = std::stoi(need_val(a));
        else if (a == "--sample_every") p.sample_every = std::stoi(need_val(a));
        else if (a == "--samples-per-block") p.samples_per_block = std::stoi(need_val(a));
        else if (a == "--seed") p.seed = std::stoi(need_val(a));
        else if (a == "--out") p.out_prefix = need_val(a);
        else if (a == "--no-or") p.do_or = false;
        else if (a == "--no-rex") p.do_rex = false;
        else if (a == "--debug-lattice") p.debug_lattice = true;
        else if (a == "--help" || a == "-h") {
            if (rank == 0) {
                std::cout << "Usage: mpirun -np <cores> ./kagome_rex_mpi_v3 [options]\n"
                          << "  --L <int>                  linear size in unit cells (default 6)\n"
                          << "  --J <double>               coupling (default 1.0)\n"
                          << "  --nrep <int>               number of replicas (default 28)\n"
                          << "  --Tmin <double>            min temperature (default 1e-3)\n"
                          << "  --Tmax <double>            max temperature (default 0.5)\n"
                          << "  --therm <int>              thermalization sweeps (default 20000)\n"
                          << "  --meas <int>               measurement sweeps (default 100000)\n"
                          << "  --sample_every <int>       sampling interval (default 10)\n"
                          << "  --samples-per-block <int>  number of sampled points per block for Cv error bar (default 100)\n"
                          << "  --seed <int>               RNG seed (default 12345)\n"
                          << "  --out <prefix>             output prefix (default kagome_rex_mpi)\n"
                          << "  --no-or                    disable overrelaxation\n"
                          << "  --no-rex                   disable replica exchange\n"
                          << "  --debug-lattice            print first few neighbor lists on rank 0\n";
            }
            MPI_Finalize();
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + a);
        }
    }
    if (p.sample_every <= 0) throw std::runtime_error("sample_every must be positive");
    if (p.samples_per_block <= 0) throw std::runtime_error("samples-per-block must be positive");
    return p;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    try {
        Params prm = parse_args(argc, argv, rank);
        KagomeLattice lat = build_kagome(prm.L);
        std::vector<double> Tgrid = geomspace(prm.Tmin, prm.Tmax, prm.nrep);
        Distribution dist = make_distribution(prm.nrep, size, rank);
        std::mt19937_64 rng(static_cast<uint64_t>(prm.seed) + 0x9e3779b97f4a7c15ULL * (rank + 1));

        const int total_sample_events = prm.meas / prm.sample_every;
        const int nblock = std::max(1, (total_sample_events + prm.samples_per_block - 1) / prm.samples_per_block);

        if (rank == 0) {
            std::cerr << "MPI ranks (=requested cores): " << size << "\n";
            std::cerr << "nrep = " << prm.nrep << ", ceil(nrep/cores) = " << dist.local_cap << " replicas per rank (capacity)\n";
            std::cerr << "L = " << prm.L << ", N = " << lat.N << ", Nbonds = " << lat.bonds.size() << "\n";
            std::cerr << "measurement sample events = " << total_sample_events
                      << ", samples_per_block = " << prm.samples_per_block
                      << ", nblock = " << nblock << "\n";
            if (prm.debug_lattice) {
                for (int i = 0; i < std::min(lat.N, 6); ++i) {
                    std::cerr << "site " << i << " neighbors:";
                    for (int k = 0; k < 4; ++k) std::cerr << ' ' << lat.neighbors[i][k];
                    std::cerr << "\n";
                }
            }
        }

        std::vector<Replica> reps(dist.local_n);
        for (int lr = 0; lr < dist.local_n; ++lr) {
            int g = dist.g_begin + lr;
            reps[lr].spins.resize(lat.N);
            for (int i = 0; i < lat.N; ++i) reps[lr].spins[i] = marsaglia_unit_vector(rng);
            reps[lr].T = Tgrid[g];
            reps[lr].energy = total_energy_replica(reps[lr].spins, lat.bonds, prm.J);
        }

        long long metro_acc_local = 0, or_acc_local = 0, rex_acc_local = 0;
        BlockMeasureAgg agg(prm.nrep, nblock);

        auto one_sweep = [&](int sweep) {
            metro_acc_local += metropolis_sweep(reps, lat, prm.J, rng);
            if (prm.do_or) or_acc_local += overrelaxation_sweep(reps, lat, prm.J, rng);
            if (prm.do_rex) {
                int parity = sweep & 1;
                rex_acc_local += replica_exchange_internal(reps, parity, rng);
                rex_acc_local += replica_exchange_boundary(reps, dist, parity, rng, MPI_COMM_WORLD);
            }
        };

        for (int s = 0; s < prm.therm; ++s) one_sweep(s);

        int sample_id = 0;
        for (int s = 0; s < prm.meas; ++s) {
            one_sweep(prm.therm + s);
            if ((s + 1) % prm.sample_every == 0) {
                int block_id = std::min(sample_id / prm.samples_per_block, nblock - 1);
                for (const auto& rep : reps) agg.add(rep.T, rep.energy, Tgrid, block_id);
                ++sample_id;
            }
        }

        std::vector<double> g_sumE_blk(prm.nrep * nblock, 0.0), g_sumE2_blk(prm.nrep * nblock, 0.0);
        std::vector<long long> g_cnt_blk(prm.nrep * nblock, 0);
        MPI_Reduce(agg.sumE_blk.data(), g_sumE_blk.data(), prm.nrep * nblock, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(agg.sumE2_blk.data(), g_sumE2_blk.data(), prm.nrep * nblock, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(agg.cnt_blk.data(), g_cnt_blk.data(), prm.nrep * nblock, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

        long long metro_acc = 0, or_acc = 0, rex_acc = 0;
        MPI_Reduce(&metro_acc_local, &metro_acc, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&or_acc_local, &or_acc, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&rex_acc_local, &rex_acc, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            std::ofstream ofs(prm.out_prefix + ".csv");
            ofs << std::setprecision(17);
            ofs << "T,E_per_spin,C_per_spin,C_per_spin_err,count,used_blocks\n";

            for (int tid = 0; tid < prm.nrep; ++tid) {
                const double T = Tgrid[tid];
                double total_sumE = 0.0, total_sumE2 = 0.0;
                long long total_cnt = 0;
                int used_blocks = 0;
                std::vector<double> blk_sumE(nblock, 0.0), blk_sumE2(nblock, 0.0);
                std::vector<long long> blk_cnt(nblock, 0);

                for (int b = 0; b < nblock; ++b) {
                    int idx = tid * nblock + b;
                    blk_sumE[b] = g_sumE_blk[idx];
                    blk_sumE2[b] = g_sumE2_blk[idx];
                    blk_cnt[b] = g_cnt_blk[idx];
                    total_sumE += blk_sumE[b];
                    total_sumE2 += blk_sumE2[b];
                    total_cnt += blk_cnt[b];
                    if (blk_cnt[b] > 0) ++used_blocks;
                }

                double meanE_per_spin = std::numeric_limits<double>::quiet_NaN();
                double Cv = std::numeric_limits<double>::quiet_NaN();
                double Cv_err = std::numeric_limits<double>::quiet_NaN();

                if (total_cnt > 0) {
                    double meanE = total_sumE / static_cast<double>(total_cnt);
                    double meanE2 = total_sumE2 / static_cast<double>(total_cnt);
                    meanE_per_spin = meanE / lat.N;
                    Cv = (meanE2 - meanE * meanE) / (lat.N * T * T);
                }

                // Delete-one-block jackknife for Cv. Works even if blocks have unequal counts.
                if (used_blocks >= 2) {
                    std::vector<double> jk_vals;
                    jk_vals.reserve(used_blocks);
                    for (int b = 0; b < nblock; ++b) {
                        if (blk_cnt[b] == 0) continue;
                        long long cnt_loo = total_cnt - blk_cnt[b];
                        if (cnt_loo <= 0) continue;
                        double sumE_loo = total_sumE - blk_sumE[b];
                        double sumE2_loo = total_sumE2 - blk_sumE2[b];
                        double meanE_loo = sumE_loo / static_cast<double>(cnt_loo);
                        double meanE2_loo = sumE2_loo / static_cast<double>(cnt_loo);
                        double Cv_loo = (meanE2_loo - meanE_loo * meanE_loo) / (lat.N * T * T);
                        jk_vals.push_back(Cv_loo);
                    }
                    int B = static_cast<int>(jk_vals.size());
                    if (B >= 2) {
                        double jk_mean = std::accumulate(jk_vals.begin(), jk_vals.end(), 0.0) / static_cast<double>(B);
                        double s = 0.0;
                        for (double x : jk_vals) {
                            double d = x - jk_mean;
                            s += d * d;
                        }
                        Cv_err = std::sqrt((static_cast<double>(B - 1) / static_cast<double>(B)) * s);
                    }
                }

                ofs << T << ',' << meanE_per_spin << ',' << Cv << ',' << Cv_err << ',' << total_cnt << ',' << used_blocks << "\n";
            }
            ofs.close();

            std::cout << "Wrote " << prm.out_prefix << ".csv\n";
            std::cout << "Metropolis accepted moves (global total): " << metro_acc << "\n";
            std::cout << "Overrelaxation updates attempted/changed (global total): " << or_acc << "\n";
            std::cout << "Replica-exchange accepted swaps (sum over proposing ranks): " << rex_acc << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Rank error: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return 0;
}
