import numpy as np
import argparse
from numba import jit
import sys

@jit(nopython=True)
def initial_lattice(L):
    return 2 * (np.random.rand(L, L) > 0.5) - 1

@jit(nopython=True)
def periodic(i, L):
    return i % L

@jit(nopython=True)
def delta_energy(spins, L, i, j):
    S = spins[i, j]
    nb = (
        spins[periodic(i + 1, L), j] +
        spins[periodic(i - 1, L), j] +
        spins[i, periodic(j + 1, L)] +
        spins[i, periodic(j - 1, L)]
    )
    return 2 * S * nb

@jit(nopython=True)
def total_energy(spins, L):
    E = 0
    for i in range(L):
        for j in range(L):
            S = spins[i, j]
            nb = spins[periodic(i + 1, L), j] + spins[i, periodic(j + 1, L)]
            E -= S * nb
    return E

@jit(nopython=True)
def metropolis(spins, L, T):
    for _ in range(L * L):
        i = int(np.random.rand() * L)
        j = int(np.random.rand() * L)
        dE = delta_energy(spins, L, i, j)
        if dE <= 0 or np.random.rand() < np.exp(-dE / T):
            spins[i, j] *= -1

@jit(nopython=True)
def calc_magnetization(spins):
    return np.sum(spins)

@jit(nopython=True)
def replica_exchange(replicas, energies, betas):
    num_replicas = len(replicas)
    for i in range(num_replicas - 1):
        beta_i = betas[i]
        beta_j = betas[i + 1]
        E_i = energies[i]
        E_j = energies[i + 1]
        d = (beta_j - beta_i) * (E_i - E_j)
        if d <= 0 or np.random.rand() < np.exp(-d):
            tmp = np.copy(replicas[i])
            replicas[i][:] = replicas[i + 1]
            replicas[i + 1][:] = tmp
            energies[i], energies[i + 1] = E_j, E_i

@jit(nopython=True)
def calc_ave_err(ms, step_expect):
    ave = np.mean(ms)
    err = np.std(ms) / np.sqrt(step_expect)
    return ave, err

@jit(nopython=True)
def monte_carlo(L, step_thermal, step_expect, exchange_interval, temps):
    num_replicas = len(temps)
    betas = 1.0 / temps

    # initialization
    replicas = [initial_lattice(L) for _ in range(num_replicas)]
    energies = np.array([total_energy(replica, L) for replica in replicas])

    # thermalization
    for _ in range(step_thermal):
        for i in range(num_replicas):
            metropolis(replicas[i], L, temps[i])
            energies[i] = total_energy(replicas[i], L)
        if _ % exchange_interval == 0:
            replica_exchange(replicas, energies, betas)

    # collect data
    m1s = np.zeros((num_replicas, step_expect))
    m2s = np.zeros((num_replicas, step_expect))
    m4s = np.zeros((num_replicas, step_expect))
    e1s = np.zeros((num_replicas, step_expect))
    e2s = np.zeros((num_replicas, step_expect))
    for step in range(step_expect):
        for i in range(num_replicas):
            metropolis(replicas[i], L, temps[i])
            energies[i] = total_energy(replicas[i], L)
            e1s[i, step] = energies[i]
            e2s[i, step] = energies[i]**2
            m1s[i, step] = calc_magnetization(replicas[i])
            m2s[i, step] = m1s[i, step]**2
            m4s[i, step] = m1s[i, step]**4
        if step % exchange_interval == 0:
            replica_exchange(replicas, energies, betas)
        if step % 1000 == 0:
            print(f"Step {step}/{step_expect} done.")

    # save data
    results = []
    for i in range(num_replicas):
        am1a, am1e = calc_ave_err(np.abs(m1s[i])/L**2, step_expect)
        m1a, m1e = calc_ave_err(m1s[i]/L**2, step_expect)
        m2a, m2e = calc_ave_err(m2s[i]/L**4, step_expect)
        m4a, m4e = calc_ave_err(m4s[i]/L**8, step_expect)
        e1a, e1e = calc_ave_err(e1s[i]/L**2, step_expect)
        e2a, e2e = calc_ave_err(e2s[i]/L**4, step_expect)
        susc = (m2a - am1a**2) / temps[i] * L**2
        spec = (e2a - e1a**2) / temps[i]**2 * L**2
        if m2a==0:
            m2a = 1e-16
        bind = 1.0 - m4a / m2a**2 / 3.0
        results.append([temps[i],
            am1a, am1e, m1a, m1e, m2a, m2e, m4a, m4e, e1a, e1e, e2a, e2e,
            susc, spec, bind, 
            ])
    return results

@jit(nopython=True)
def set_seed(seed):
    np.random.seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, default=16, help="System size (default=16)")
    parser.add_argument("--step_thermal", type=int, default=5000, help="Thermalization steps (default=5000)")
    parser.add_argument("--step_expect", type=int, default=20000, help="Sampling steps (default=20000)")
    parser.add_argument("--exchange_interval", type=int, default=1, help="Replica exchange interval (default=1)")
#    parser.add_argument("--seed", type=int, default=12345, help="Random seed")
    args = parser.parse_args()

    L = args.L
    step_thermal = args.step_thermal
    step_expect = args.step_expect
    exchange_interval = args.exchange_interval
#    seed = args.seed
    temps = np.arange(0.5, 5.01, 0.1)
    seeds = np.arange(12345, 12345+10)

    tmp = []
    for seed in seeds:
        print(f"Seed {seed}:")
        set_seed(seed)
        results = monte_carlo(L, step_thermal, step_expect, exchange_interval, temps)
        np.savetxt("dat_2d_ising_L{}".format(L)+"_seed{}".format(seed), np.array(results), fmt="%+.10f",
            header="T, am1a, am1e, m1a, m1e, m2a, m2e, m4a, m4e, e1a, e1e, e2a, e2e, susc, spec, bind",)
        tmp.append(results)
        print()
    tmp = np.array(tmp)
    tmp_ave = np.average(tmp,axis=0)
    tmp_err = np.std(tmp,axis=0) / np.sqrt(tmp.shape[0])
    dat_ave_err = np.array([
        tmp_ave[:,0],tmp_ave[:,1],tmp_err[:,1],tmp_ave[:,3],tmp_err[:,3],tmp_ave[:,5],tmp_err[:,5],tmp_ave[:,7],tmp_err[:,7],
        tmp_ave[:,9],tmp_err[:,9],tmp_ave[:,11],tmp_err[:,11],
        tmp_ave[:,13],tmp_err[:,13],tmp_ave[:,14],tmp_err[:,14],tmp_ave[:,15],tmp_err[:,15],
        ]).T
    np.savetxt("dat_2d_ising_L{}".format(L)+"_ave_err", dat_ave_err, fmt="%+.10f",
        header="T, am1a, am1e, m1a, m1e, m2a, m2e, m4a, m4e, e1a, e1e, e2a, e2e, susc, susce, spec, spece, bind, binde",)

if __name__ == "__main__":
    main()
