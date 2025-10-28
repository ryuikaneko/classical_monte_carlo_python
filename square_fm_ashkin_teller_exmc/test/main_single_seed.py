import numpy as np
import argparse
from numba import jit

@jit(nopython=True)
def initial_lattice(L):
    sgm = 2 * (np.random.rand(L, L) > 0.5) - 1
    tau = 2 * (np.random.rand(L, L) > 0.5) - 1
    return sgm, tau

@jit(nopython=True)
def periodic(i, L):
    return i % L

@jit(nopython=True)
def local_energy(sgm, tau, L, i, j, J, K):
    s = sgm[i, j]
    t = tau[i, j]
    e = 0.0
    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
        ni = periodic(i + dx, L)
        nj = periodic(j + dy, L)
        sn = sgm[ni, nj]
        tn = tau[ni, nj]
        e -= J * (s * sn + t * tn) + K * s * sn * t * tn
    return e

@jit(nopython=True)
def total_energy(sgm, tau, L, J, K):
    e = 0.0
    for i in range(L):
        for j in range(L):
            e += local_energy(sgm, tau, L, i, j, J, K)
    return 0.5 * e

@jit(nopython=True)
def metropolis(sgm, tau, L, T, J, K):
    for _ in range(L * L):
        i = int(np.random.rand() * L)
        j = int(np.random.rand() * L)
        # flip sgm
        dE_sgm = 2 * sgm[i, j] * sum([
            J * sgm[periodic(i + dx, L), periodic(j + dy, L)] +
            K * sgm[periodic(i + dx, L), periodic(j + dy, L)] * tau[i, j] * tau[periodic(i + dx, L), periodic(j + dy, L)]
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]
        ])
        if dE_sgm <= 0 or np.random.rand() < np.exp(-dE_sgm / T):
            sgm[i, j] *= -1
        # flip tau
        dE_tau = 2 * tau[i, j] * sum([
            J * tau[periodic(i + dx, L), periodic(j + dy, L)] +
            K * tau[periodic(i + dx, L), periodic(j + dy, L)] * sgm[i, j] * sgm[periodic(i + dx, L), periodic(j + dy, L)]
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]
        ])
        if dE_tau <= 0 or np.random.rand() < np.exp(-dE_tau / T):
            tau[i, j] *= -1

@jit(nopython=True)
def calc_magnetization(sgm, tau):
    # average m_sgm and m_tau
    m_sgm = np.sum(sgm)
    m_tau = np.sum(tau)
    return 0.5 * (m_sgm + m_tau)
#    # average |m_sgm| and |m_tau|
#    m_sgm = np.abs(np.sum(sgm))
#    m_tau = np.abs(np.sum(tau))
#    return 0.5 * (m_sgm + m_tau)

@jit(nopython=True)
def replica_exchange(sgms, taus, energies, betas):
    num_replicas = len(sgms)
    for i in range(num_replicas - 1):
        beta_i = betas[i]
        beta_j = betas[i + 1]
        E_i = energies[i]
        E_j = energies[i + 1]
        d = (beta_j - beta_i) * (E_i - E_j)
        if d <= 0 or np.random.rand() < np.exp(-d):
            # swap sgm
            tmp_sgm = np.copy(sgms[i])
            sgms[i][:] = sgms[i + 1]
            sgms[i + 1][:] = tmp_sgm
            # swap tau
            tmp_tau = np.copy(taus[i])
            taus[i][:] = taus[i + 1]
            taus[i + 1][:] = tmp_tau
            # swap energy
            energies[i], energies[i + 1] = E_j, E_i

@jit(nopython=True)
def calc_ave_err(ms, step_expect):
    ave = np.mean(ms)
    err = np.std(ms) / np.sqrt(step_expect)
    return ave, err

@jit(nopython=True)
def monte_carlo(L, step_thermal, step_expect, exchange_interval, temps, J, K):
    num_replicas = len(temps)
    betas = 1.0 / temps

    # initialization
    sgms = []
    taus = []
    for _ in range(num_replicas):
        s, t = initial_lattice(L)
        sgms.append(s)
        taus.append(t)
    energies = np.zeros(num_replicas)
    for i in range(num_replicas):
        energies[i] = total_energy(sgms[i], taus[i], L, J, K)

    # thermalization
    for step in range(step_thermal):
        for i in range(num_replicas):
            metropolis(sgms[i], taus[i], L, temps[i], J, K)
            energies[i] = total_energy(sgms[i], taus[i], L, J, K)
        if step % exchange_interval == 0:
            replica_exchange(sgms, taus, energies, betas)

    # collect data
    m1s = np.zeros((num_replicas, step_expect))
    m2s = np.zeros((num_replicas, step_expect))
    m4s = np.zeros((num_replicas, step_expect))
    e1s = np.zeros((num_replicas, step_expect))
    e2s = np.zeros((num_replicas, step_expect))
    for step in range(step_expect):
        for i in range(num_replicas):
            metropolis(sgms[i], taus[i], L, temps[i], J, K)
            energies[i] = total_energy(sgms[i], taus[i], L, J, K)
            e1s[i, step] = energies[i]
            e2s[i, step] = energies[i]**2
            m1s[i, step] = calc_magnetization(sgms[i], taus[i])
            m2s[i, step] = m1s[i, step]**2
            m4s[i, step] = m1s[i, step]**4
        if step % exchange_interval == 0:
            replica_exchange(sgms, taus, energies, betas)
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
    parser.add_argument("--L", type=int, default=8, help="System size (default=8)")
    parser.add_argument("--step_thermal", type=int, default=20000, help="Thermalization steps (default=20000)")
    parser.add_argument("--step_expect", type=int, default=40000, help="Sampling steps (default=40000)")
    parser.add_argument("--exchange_interval", type=int, default=1, help="Replica exchange interval (default=1)")
    parser.add_argument("--J", type=float, default=1.0, help="Spin exchange interaction J (default=1.0)")
    parser.add_argument("--K", type=float, default=4.0, help="Spin exchange interaction K (default=4.0)")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed")
    args = parser.parse_args()

    L = args.L
    step_thermal = args.step_thermal
    step_expect = args.step_expect
    exchange_interval = args.exchange_interval
    J = args.J
    K = args.K
    seed = args.seed
    temps = np.arange(2.0, 12.01, 0.2)

    set_seed(seed)
    results = monte_carlo(L, step_thermal, step_expect, exchange_interval, temps, J, K)
    np.savetxt("dat_2d_ashkin_teller_L{}".format(L)+"_seed{}".format(seed), np.array(results), fmt="%+.10f",
        header="T, am1a, am1e, m1a, m1e, m2a, m2e, m4a, m4e, e1a, e1e, e2a, e2e, susc, spec, bind",)

if __name__ == "__main__":
    main()
