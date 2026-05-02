
import numpy as np
from numba import njit

# ============================================================
# Classical kagome Heisenberg model (LxL unit cells, 3 sites/cell)
# Replica Exchange Monte Carlo + Metropolis + Overrelaxation
# Random spin initialization via Marsaglia's 2-variable sphere picking
# JIT compilation with numba
# ============================================================

# -----------------------------
# Lattice construction (Python)
# -----------------------------
def kagome_bonds(L: int):
    """
    Build nearest-neighbor bond list for kagome lattice with periodic BC.
    Unit-cell convention:
      Bravais vectors: a1=(1,0), a2=(1/2, sqrt(3)/2)
      basis in fractional coordinates of (a1,a2):
        A=(0,0), B=(1/2,0), C=(0,1/2)
    Returns:
      bonds : (Nb,2) int32 unique nearest-neighbor pairs
      neighbors : (N,4) int32 neighbor table
    """
    basis = np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]], dtype=np.float64)
    a1 = np.array([1.0, 0.0], dtype=np.float64)
    a2 = np.array([0.5, np.sqrt(3.0)/2.0], dtype=np.float64)

    N = 3 * L * L
    frac = np.zeros((N, 2), dtype=np.float64)
    for x in range(L):
        for y in range(L):
            cell = x + L * y
            for s in range(3):
                i = 3 * cell + s
                frac[i] = np.array([x, y], dtype=np.float64) + basis[s]

    def frac_to_cart(v):
        return v[0] * a1 + v[1] * a2

    coords = np.array([frac_to_cart(v) for v in frac])
    d2_target = 0.25  # nearest-neighbor distance^2 = (1/2)^2
    tol = 1e-9
    bonds = []

    for i in range(N):
        for j in range(i + 1, N):
            best_d2 = 1e100
            # search 9 image shifts; sufficient here to find minimum image
            for m in (-1, 0, 1):
                for n in (-1, 0, 1):
                    dv_frac = frac[j] - frac[i] + np.array([m * L, n * L], dtype=np.float64)
                    dv = frac_to_cart(dv_frac)
                    d2 = dv[0] * dv[0] + dv[1] * dv[1]
                    if d2 < best_d2:
                        best_d2 = d2
            if abs(best_d2 - d2_target) < tol:
                bonds.append((i, j))

    bonds = np.array(bonds, dtype=np.int32)
    if len(bonds) != 2 * N:
        raise RuntimeError(f"Unexpected number of bonds: {len(bonds)} != {2*N}")

    deg = np.zeros(N, dtype=np.int32)
    for i, j in bonds:
        deg[i] += 1
        deg[j] += 1
    if not np.all(deg == 4):
        raise RuntimeError(f"Degree check failed. Degrees found: {np.unique(deg)}")

    neighbors = -np.ones((N, 4), dtype=np.int32)
    ctr = np.zeros(N, dtype=np.int32)
    for i, j in bonds:
        neighbors[i, ctr[i]] = j; ctr[i] += 1
        neighbors[j, ctr[j]] = i; ctr[j] += 1
    return bonds, neighbors


# -----------------------------
# Random unit vectors: Marsaglia sphere point picking (numba)
# -----------------------------
@njit
def marsaglia_unit_vector():
    while True:
        u = 2.0 * np.random.random() - 1.0
        v = 2.0 * np.random.random() - 1.0
        s = u * u + v * v
        if s > 0.0 and s < 1.0:
            fac = np.sqrt(1.0 - s)
            x = 2.0 * u * fac
            y = 2.0 * v * fac
            z = 1.0 - 2.0 * s
            return x, y, z


@njit
def random_spins(nrep, N):
    spins = np.empty((nrep, N, 3), dtype=np.float64)
    for r in range(nrep):
        for i in range(N):
            x, y, z = marsaglia_unit_vector()
            spins[r, i, 0] = x
            spins[r, i, 1] = y
            spins[r, i, 2] = z
    return spins


# -----------------------------
# Energy and local field
# -----------------------------
@njit
def total_energy_replica(spins_r, bonds, J):
    E = 0.0
    for b in range(bonds.shape[0]):
        i = bonds[b, 0]
        j = bonds[b, 1]
        E += J * (spins_r[i, 0] * spins_r[j, 0] +
                  spins_r[i, 1] * spins_r[j, 1] +
                  spins_r[i, 2] * spins_r[j, 2])
    return E


@njit
def total_energies(spins, bonds, J):
    nrep = spins.shape[0]
    E = np.empty(nrep, dtype=np.float64)
    for r in range(nrep):
        E[r] = total_energy_replica(spins[r], bonds, J)
    return E


@njit
def local_field(spins_r, neighbors, i, J):
    hx = 0.0
    hy = 0.0
    hz = 0.0
    for k in range(neighbors.shape[1]):
        j = neighbors[i, k]
        hx += J * spins_r[j, 0]
        hy += J * spins_r[j, 1]
        hz += J * spins_r[j, 2]
    return hx, hy, hz


# -----------------------------
# Local updates
# -----------------------------
@njit
def metropolis_sweep(spins, energies, bonds, neighbors, Tlist, J):
    nrep, N, _ = spins.shape
    acc = 0
    for r in range(nrep):
        beta = 1.0 / Tlist[r]
        for _ in range(N):
            i = np.random.randint(0, N)
            oldx = spins[r, i, 0]
            oldy = spins[r, i, 1]
            oldz = spins[r, i, 2]
            hx, hy, hz = local_field(spins[r], neighbors, i, J)
            old_dot = oldx * hx + oldy * hy + oldz * hz
            nx, ny, nz = marsaglia_unit_vector()
            new_dot = nx * hx + ny * hy + nz * hz
            dE = new_dot - old_dot
            if dE <= 0.0 or np.random.random() < np.exp(-beta * dE):
                spins[r, i, 0] = nx
                spins[r, i, 1] = ny
                spins[r, i, 2] = nz
                energies[r] += dE
                acc += 1
    return acc


@njit
def overrelaxation_sweep(spins, energies, neighbors, J):
    # Microcanonical update: reflect S_i about local field H_i.
    # Performed with the same frequency as Metropolis sweeps.
    nrep, N, _ = spins.shape
    changed = 0
    for r in range(nrep):
        for _ in range(N):
            i = np.random.randint(0, N)
            hx, hy, hz = local_field(spins[r], neighbors, i, J)
            h2 = hx * hx + hy * hy + hz * hz
            if h2 < 1e-30:
                continue
            sx = spins[r, i, 0]
            sy = spins[r, i, 1]
            sz = spins[r, i, 2]
            sh = sx * hx + sy * hy + sz * hz
            fac = 2.0 * sh / h2
            nx = fac * hx - sx
            ny = fac * hy - sy
            nz = fac * hz - sz
            # normalize for numerical stability
            nrm = np.sqrt(nx * nx + ny * ny + nz * nz)
            spins[r, i, 0] = nx / nrm
            spins[r, i, 1] = ny / nrm
            spins[r, i, 2] = nz / nrm
            changed += 1
    # energies remain unchanged up to roundoff
    return changed


# -----------------------------
# Replica exchange
# -----------------------------
@njit
def replica_exchange_step(spins, energies, Tlist, parity):
    nrep = spins.shape[0]
    acc = 0
    start = parity
    for r in range(start, nrep - 1, 2):
        beta_r = 1.0 / Tlist[r]
        beta_s = 1.0 / Tlist[r + 1]
        d = (beta_r - beta_s) * (energies[r + 1] - energies[r])
        if d >= 0.0 or np.random.random() < np.exp(d):
            # swap temperatures only (equivalent to swapping configurations)
            tmpT = Tlist[r]
            Tlist[r] = Tlist[r + 1]
            Tlist[r + 1] = tmpT
            tmpE = energies[r]
            energies[r] = energies[r + 1]
            energies[r + 1] = tmpE
            tmp = spins[r].copy()
            spins[r] = spins[r + 1]
            spins[r + 1] = tmp
            acc += 1
    return acc


# -----------------------------
# Production driver
# -----------------------------
@njit
def run_simulation(spins, bonds, neighbors, T_init, J,
                   n_therm, n_meas, sample_every,
                   do_or=True, do_rex=True):
    nrep = spins.shape[0]
    N = spins.shape[1]
    Tlist = T_init.copy()
    energies = total_energies(spins, bonds, J)

    n_samples = n_meas // sample_every
    E_samples = np.empty((n_samples, nrep), dtype=np.float64)
    T_samples = np.empty((n_samples, nrep), dtype=np.float64)
    acc_metro = 0
    acc_swap = 0
    n_metro_attempt = 0
    n_swap_attempt = 0

    # thermalization
    parity = 0
    for sweep in range(n_therm):
        acc_metro += metropolis_sweep(spins, energies, bonds, neighbors, Tlist, J)
        n_metro_attempt += nrep * N
        if do_or:
            overrelaxation_sweep(spins, energies, neighbors, J)
        if do_rex:
            acc_swap += replica_exchange_step(spins, energies, Tlist, parity)
            n_swap_attempt += (nrep - parity - 1 + 1) // 2
            parity = 1 - parity

    # measurement
    idx = 0
    for sweep in range(n_meas):
        acc_metro += metropolis_sweep(spins, energies, bonds, neighbors, Tlist, J)
        n_metro_attempt += nrep * N
        if do_or:
            overrelaxation_sweep(spins, energies, neighbors, J)
        if do_rex:
            acc_swap += replica_exchange_step(spins, energies, Tlist, parity)
            n_swap_attempt += (nrep - parity - 1 + 1) // 2
            parity = 1 - parity
        if (sweep + 1) % sample_every == 0:
            E_samples[idx, :] = energies
            T_samples[idx, :] = Tlist
            idx += 1

    metro_rate = acc_metro / max(1, n_metro_attempt)
    swap_rate = acc_swap / max(1, n_swap_attempt)
    return E_samples, T_samples, metro_rate, swap_rate


# -----------------------------
# Binning / observable analysis
# -----------------------------
def analyze_by_temperature(E_samples, T_samples, T_grid, N, window_rel=0.015):
    """
    Collect energies belonging to each target temperature by matching sampled T labels.
    Since replica exchange permutes temperatures among replicas, we sort by temperature identity.
    For each target T, gather all energies with |T_sample/T - 1| <= window_rel.
    Returns dict with per-spin energy, heat capacity per spin, counts, and stderr from simple binning.
    """
    out = []
    flat_E = E_samples.reshape(-1)
    flat_T = T_samples.reshape(-1)
    for T in T_grid:
        mask = np.abs(flat_T / T - 1.0) <= window_rel
        Es = flat_E[mask]
        if Es.size == 0:
            out.append((T, np.nan, np.nan, 0, np.nan, np.nan))
            continue
        e_mean = Es.mean() / N
        c_mean = (Es.var(ddof=1) / (N * T * T)) if Es.size > 1 else np.nan
        # crude stderr via 16 bins
        nb = min(16, max(1, Es.size // 20))
        if nb >= 2:
            m = (Es.size // nb) * nb
            Eb = Es[:m].reshape(nb, -1)
            e_bins = Eb.mean(axis=1) / N
            c_bins = Eb.var(axis=1, ddof=1) / (N * T * T)
            e_err = e_bins.std(ddof=1) / np.sqrt(nb)
            c_err = c_bins.std(ddof=1) / np.sqrt(nb)
        else:
            e_err = np.nan
            c_err = np.nan
        out.append((T, e_mean, c_mean, Es.size, e_err, c_err))
    return np.array(out, dtype=np.float64)


def integrated_autocorr_time(x, max_lag=None):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    if n < 8:
        return np.nan
    if max_lag is None:
        max_lag = min(n // 2, 5000)
    c0 = np.dot(x, x) / n
    if c0 <= 0:
        return np.nan
    tau = 0.5
    prev = 1.0
    for lag in range(1, max_lag):
        c = np.dot(x[:-lag], x[lag:]) / (n - lag)
        rho = c / c0
        if rho <= 0.0 and prev <= 0.0:
            break
        tau += rho
        prev = rho
    return tau


def estimate_required_sweeps(E_samples, T_samples, target_T, N, window_rel=0.01):
    flat_E = E_samples.reshape(-1)
    flat_T = T_samples.reshape(-1)
    mask = np.abs(flat_T / target_T - 1.0) <= window_rel
    Es = flat_E[mask]
    if Es.size < 100:
        return dict(T=target_T, n=Es.size, tau=np.nan, suggested_meas_sweeps=np.nan,
                    C_per_spin=np.nan)
    tau = integrated_autocorr_time(Es)
    Cn = Es.var(ddof=1) / (N * target_T * target_T)
    # Heuristic: ~100 effectively independent measurements of E^2 for stable C estimate.
    suggested_samples = int(np.ceil(200.0 * tau)) if np.isfinite(tau) else np.nan
    return dict(T=target_T, n=int(Es.size), tau=float(tau),
                suggested_meas_sweeps=suggested_samples,
                C_per_spin=float(Cn))


def default_temperature_grid(Tmin=1e-3, Tmax=0.5, nrep=28):
    return np.geomspace(Tmin, Tmax, nrep).astype(np.float64)


def main():
    import argparse, time
    parser = argparse.ArgumentParser(description='Classical kagome Heisenberg model REMC')
    parser.add_argument('--L', type=int, default=6)
    parser.add_argument('--J', type=float, default=1.0)
    parser.add_argument('--nrep', type=int, default=28)
    parser.add_argument('--Tmin', type=float, default=1e-3)
    parser.add_argument('--Tmax', type=float, default=0.5)
#    parser.add_argument('--therm', type=int, default=20000)
#    parser.add_argument('--meas', type=int, default=100000)
    parser.add_argument('--therm', type=int, default=100000)
    parser.add_argument('--meas', type=int, default=5000000)
    parser.add_argument('--sample_every', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1234)
#    parser.add_argument('--out', type=str, default='kagome_L6_rex_results.npz')
    parser.add_argument('--out', type=str, default='kagome')
    args = parser.parse_args()

    np.random.seed(args.seed)
    bonds, neighbors = kagome_bonds(args.L)
    N = 3 * args.L * args.L
    T0 = default_temperature_grid(args.Tmin, args.Tmax, args.nrep)
    spins = random_spins(args.nrep, N)

    t0 = time.time()
    E_samples, T_samples, metro_rate, swap_rate = run_simulation(
        spins, bonds, neighbors, T0, args.J,
        args.therm, args.meas, args.sample_every,
        do_or=True, do_rex=True)
    dt = time.time() - t0

    stats = analyze_by_temperature(E_samples, T_samples, T0, N)
    np.savez(args.out+'_L{}'.format(args.L)+'_rex_results.npz',
             E_samples=E_samples,
             T_samples=T_samples,
             T_grid=T0,
             stats=stats,
             bonds=bonds,
             neighbors=neighbors,
             metro_rate=metro_rate,
             swap_rate=swap_rate,
             runtime_sec=dt)
    np.savetxt(args.out+'_L{}'.format(args.L)+'_rex_stats.txt',stats,
             header="T, E/N, C/N, nsamples, Eerr, Cerr")

    print(f'N={N}, bonds={len(bonds)}, runtime={dt:.3f} sec')
    print(f'Metropolis acceptance = {metro_rate:.4f}')
    print(f'Replica-exchange acceptance = {swap_rate:.4f}')
    print('\n# T, E/N, C/N, nsamples, Eerr, Cerr')
    for row in stats:
        print(f'{row[0]:.8g} {row[1]: .8f} {row[2]: .8f} {int(row[3])} {row[4]:.3e} {row[5]:.3e}')

    for TT in (5e-3, 1e-3):
        est = estimate_required_sweeps(E_samples, T_samples, TT, N)
        print('\nEstimate at T/J =', TT)
        for k, v in est.items():
            print(f'  {k}: {v}')

if __name__ == '__main__':
    main()
