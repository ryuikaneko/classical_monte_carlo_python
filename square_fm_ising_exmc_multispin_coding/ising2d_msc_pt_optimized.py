#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from scipy.special import ellipk

J = 1.0
NREP = 64

# ------------------------------------------------------------
# bit / random helpers
# ------------------------------------------------------------
@njit(cache=True)
def set_seed(seed):
    np.random.seed(seed)

@njit(cache=True)
def bernoulli_mask(p):
    m = np.uint64(0)
    one = np.uint64(1)
    for b in range(64):
        if np.random.random() < p:
            m |= (one << np.uint64(b))
    return m

@njit(cache=True)
def bit_count_u64(x):
    c = 0
    while x != np.uint64(0):
        x &= (x - np.uint64(1))
        c += 1
    return c

@njit(cache=True)
def add_int_delta_from_mask(arr, mask, delta):
    one = np.uint64(1)
    for b in range(64):
        if ((mask >> np.uint64(b)) & one) != 0:
            arr[b] += delta

@njit(cache=True)
def masked_swap_words(a, b, mask):
    diff = (a ^ b) & mask
    return a ^ diff, b ^ diff

@njit(cache=True)
def mask_eq0(a, b, c, d):
    return ~(a | b | c | d)

@njit(cache=True)
def mask_eq1(a, b, c, d):
    return ((a & ~b & ~c & ~d) |
            (~a & b & ~c & ~d) |
            (~a & ~b & c & ~d) |
            (~a & ~b & ~c & d))

@njit(cache=True)
def mask_eq3(a, b, c, d):
    return ((a & b & c & ~d) |
            (a & b & ~c & d) |
            (a & ~b & c & d) |
            (~a & b & c & d))

@njit(cache=True)
def mask_eq4(a, b, c, d):
    return a & b & c & d


# ------------------------------------------------------------
# init / exact observables
# ------------------------------------------------------------
@njit(cache=True)
def init_spins_random_with_energy_mag(L):
    spins = np.zeros((L, L), dtype=np.uint64)
    mags = np.zeros(64, dtype=np.int64)
    one = np.uint64(1)
    for i in range(L):
        for j in range(L):
            w = np.uint64(0)
            for b in range(64):
                if np.random.random() < 0.5:
                    w |= (one << np.uint64(b))
                    mags[b] += 1
                else:
                    mags[b] -= 1
            spins[i, j] = w
    energies = total_energy_per_replica(spins)
    return spins, energies, mags

@njit(cache=True)
def total_energy_per_replica(spins):
    L = spins.shape[0]
    E = np.zeros(64, dtype=np.int64)
    one = np.uint64(1)
    for i in range(L):
        ip = (i + 1) % L
        for j in range(L):
            jp = (j + 1) % L
            s = spins[i, j]
            sr = spins[i, jp]
            sd = spins[ip, j]
            for b in range(64):
                si = 1 if ((s >> np.uint64(b)) & one) else -1
                sj = 1 if ((sr >> np.uint64(b)) & one) else -1
                sk = 1 if ((sd >> np.uint64(b)) & one) else -1
                E[b] += -si * sj - si * sk
    return E

@njit(cache=True)
def total_magnetization_per_replica_naive(spins):
    L = spins.shape[0]
    M = np.zeros(64, dtype=np.int64)
    one = np.uint64(1)
    for i in range(L):
        for j in range(L):
            s = spins[i, j]
            for b in range(64):
                if ((s >> np.uint64(b)) & one) != 0:
                    M[b] += 1
                else:
                    M[b] -= 1
    return M

# ------------------------------------------------------------
# popcount-based magnetization helper via 64x64 transpose
# used for validation / optional resync, while production path
# uses incremental magnetization updates for maximum speed.
# ------------------------------------------------------------
@njit(cache=True)
def transpose64x64_block(inp):
    out = np.zeros(64, dtype=np.uint64)
    one = np.uint64(1)
    for r in range(64):
        x = inp[r]
        for c in range(64):
            if ((x >> np.uint64(c)) & one) != 0:
                out[c] |= (one << np.uint64(r))
    return out

@njit(cache=True)
def total_magnetization_per_replica_popcount(spins):
    L = spins.shape[0]
    nsite = L * L
    counts = np.zeros(64, dtype=np.int64)
    block = np.zeros(64, dtype=np.uint64)
    idx = 0
    for i in range(L):
        for j in range(L):
            block[idx % 64] = spins[i, j]
            idx += 1
            if idx % 64 == 0:
                t = transpose64x64_block(block)
                for b in range(64):
                    counts[b] += bit_count_u64(t[b])
                block[:] = np.uint64(0)
    rem = idx % 64
    if rem != 0:
        for k in range(rem, 64):
            block[k] = np.uint64(0)
        t = transpose64x64_block(block)
        for b in range(64):
            counts[b] += bit_count_u64(t[b])
    M = np.zeros(64, dtype=np.int64)
    for b in range(64):
        M[b] = 2 * counts[b] - nsite
    return M


# ------------------------------------------------------------
# local sweep with incremental E and M updates
# ------------------------------------------------------------
@njit(cache=True)
def sweep_checkerboard_incremental(spins, energies, mags, beta):
    L = spins.shape[0]
    p4 = np.exp(-4.0 * beta)
    p8 = np.exp(-8.0 * beta)
    for parity in range(2):
        for i in range(L):
            j0 = (i + parity) & 1
            for j in range(j0, L, 2):
                s = spins[i, j]
                up = spins[(i - 1) % L, j]
                dn = spins[(i + 1) % L, j]
                lf = spins[i, (j - 1) % L]
                rt = spins[i, (j + 1) % L]

                a = s ^ up
                b = s ^ dn
                c = s ^ lf
                d = s ^ rt

                eq0 = mask_eq0(a, b, c, d)
                eq1 = mask_eq1(a, b, c, d)
                eq3 = mask_eq3(a, b, c, d)
                eq4 = mask_eq4(a, b, c, d)
                eq2 = ~(eq0 | eq1 | eq3 | eq4)

                acc0 = eq0 & bernoulli_mask(p8)
                acc1 = eq1 & bernoulli_mask(p4)
                acc2 = eq2
                acc3 = eq3
                acc4 = eq4
                flip = acc0 | acc1 | acc2 | acc3 | acc4

                # incremental energy updates
                add_int_delta_from_mask(energies, acc0,  8)
                add_int_delta_from_mask(energies, acc1,  4)
                add_int_delta_from_mask(energies, acc3, -4)
                add_int_delta_from_mask(energies, acc4, -8)

                # incremental magnetization updates
                up_to_down = flip & s
                down_to_up = flip & (~s)
                add_int_delta_from_mask(mags, up_to_down, -2)
                add_int_delta_from_mask(mags, down_to_up,  2)

                spins[i, j] = s ^ flip


# ------------------------------------------------------------
# replica exchange (correct sign)
# ------------------------------------------------------------
@njit(cache=True)
def exchange_neighbor_temperatures(configs, energies, mags, betas, i):
    beta_i = betas[i]
    beta_j = betas[i + 1]
    Ei = energies[i]
    Ej = energies[i + 1]
    acc_mask = np.uint64(0)
    one = np.uint64(1)
    for b in range(64):
        delta = (beta_i - beta_j) * (Ei[b] - Ej[b])  # correct sign
        accept = False
        if delta >= 0.0:
            accept = True
        else:
            if np.random.random() < np.exp(delta):
                accept = True
        if accept:
            acc_mask |= (one << np.uint64(b))
    nacc = bit_count_u64(acc_mask)
    if nacc == 0:
        return 0
    L = configs.shape[1]
    for x in range(L):
        for y in range(L):
            wi, wj = masked_swap_words(configs[i, x, y], configs[i + 1, x, y], acc_mask)
            configs[i, x, y] = wi
            configs[i + 1, x, y] = wj
    # swap energies and magnetizations consistently
    for b in range(64):
        if ((acc_mask >> np.uint64(b)) & one) != 0:
            tmp = energies[i, b]
            energies[i, b] = energies[i + 1, b]
            energies[i + 1, b] = tmp
            tmpm = mags[i, b]
            mags[i, b] = mags[i + 1, b]
            mags[i + 1, b] = tmpm
    return nacc


# ------------------------------------------------------------
# single-temperature short run for pilot sigma_E estimate
# ------------------------------------------------------------
@njit(cache=True)
def run_single_temp_short(L, beta, n_therm, n_meas, n_skip, seed):
    set_seed(seed)
    spins, energies, mags = init_spins_random_with_energy_mag(L)
    for _ in range(n_therm):
        sweep_checkerboard_incremental(spins, energies, mags, beta)
    N = L * L
    e1 = 0.0
    e2 = 0.0
    ns = 0
    for _ in range(n_meas):
        for __ in range(n_skip):
            sweep_checkerboard_incremental(spins, energies, mags, beta)
        for b in range(64):
            e = energies[b] / N
            e1 += e
            e2 += e * e
            ns += 1
    em = e1 / ns
    ve = e2 / ns - em * em
    if ve < 0.0:
        ve = 0.0
    sigma_E = np.sqrt(ve) * N
    return em, ve, sigma_E


# ------------------------------------------------------------
# full PT run with Binder, chi, Cv
# ------------------------------------------------------------
@njit(cache=True)
def run_parallel_tempering_incremental(L, betas, n_therm, n_meas, n_skip, seed,
                                       resync_every, use_popcount_resync):
    set_seed(seed)
    nT = len(betas)
    configs = np.zeros((nT, L, L), dtype=np.uint64)
    energies = np.zeros((nT, 64), dtype=np.int64)
    mags = np.zeros((nT, 64), dtype=np.int64)

    for t in range(nT):
        sp, en, mg = init_spins_random_with_energy_mag(L)
        configs[t] = sp
        energies[t] = en
        mags[t] = mg

    # thermalization
    for it in range(n_therm):
        for t in range(nT):
            sweep_checkerboard_incremental(configs[t], energies[t], mags[t], betas[t])
        parity = it & 1
        for t in range(parity, nT - 1, 2):
            exchange_neighbor_temperatures(configs, energies, mags, betas, t)
        # optional popcount resync / check
        if use_popcount_resync == 1 and resync_every > 0 and ((it + 1) % resync_every == 0):
            for t in range(nT):
                mags[t] = total_magnetization_per_replica_popcount(configs[t])
                energies[t] = total_energy_per_replica(configs[t])

    N = L * L
    e1 = np.zeros(nT, dtype=np.float64)
    e2 = np.zeros(nT, dtype=np.float64)
    mabs1 = np.zeros(nT, dtype=np.float64)
    mabs2 = np.zeros(nT, dtype=np.float64)
    m21 = np.zeros(nT, dtype=np.float64)
    m41 = np.zeros(nT, dtype=np.float64)
    ns = np.zeros(nT, dtype=np.int64)
    exch_acc_bits = np.zeros(nT - 1, dtype=np.int64)
    exch_try_bits = np.zeros(nT - 1, dtype=np.int64)

    for it in range(n_meas):
        for step in range(n_skip):
            for t in range(nT):
                sweep_checkerboard_incremental(configs[t], energies[t], mags[t], betas[t])
            parity = (it + step) & 1
            for t in range(parity, nT - 1, 2):
                exch_try_bits[t] += 64
                exch_acc_bits[t] += exchange_neighbor_temperatures(configs, energies, mags, betas, t)

        if use_popcount_resync == 1 and resync_every > 0 and ((it + 1) % resync_every == 0):
            for t in range(nT):
                mags[t] = total_magnetization_per_replica_popcount(configs[t])
                energies[t] = total_energy_per_replica(configs[t])

        for t in range(nT):
            for b in range(64):
                e = energies[t, b] / N
                m = mags[t, b] / N
                ma = abs(m)
                e1[t] += e
                e2[t] += e * e
                mabs1[t] += ma
                mabs2[t] += ma * ma
                m21[t] += m * m
                m41[t] += (m * m) * (m * m)
                ns[t] += 1

    e_mean = np.zeros(nT, dtype=np.float64)
    e_err = np.zeros(nT, dtype=np.float64)
    mabs_mean = np.zeros(nT, dtype=np.float64)
    mabs_err = np.zeros(nT, dtype=np.float64)
    cv = np.zeros(nT, dtype=np.float64)
    chi = np.zeros(nT, dtype=np.float64)
    chi_conn = np.zeros(nT, dtype=np.float64)
    m2_mean = np.zeros(nT, dtype=np.float64)
    sqrt_m2 = np.zeros(nT, dtype=np.float64)
    m4_mean = np.zeros(nT, dtype=np.float64)
    binder = np.zeros(nT, dtype=np.float64)
    exch_ratio = np.zeros(nT - 1, dtype=np.float64)

    for t in range(nT):
        e_mean[t] = e1[t] / ns[t]
        mabs_mean[t] = mabs1[t] / ns[t]
        m2_mean[t] = m21[t] / ns[t]
        m4_mean[t] = m41[t] / ns[t]
        sqrt_m2[t] = np.sqrt(m2_mean[t])
        ve = e2[t] / ns[t] - e_mean[t] * e_mean[t]
        vabs = mabs2[t] / ns[t] - mabs_mean[t] * mabs_mean[t]
        if ve < 0.0:
            ve = 0.0
        if vabs < 0.0:
            vabs = 0.0
        e_err[t] = np.sqrt(ve / ns[t])
        mabs_err[t] = np.sqrt(vabs / ns[t])
        beta = betas[t]
        cv[t] = beta * beta * N * ve
        chi[t] = beta * N * m2_mean[t]
        chi_conn[t] = beta * N * (m2_mean[t] - mabs_mean[t] * mabs_mean[t])
        if m2_mean[t] > 1e-30:
            binder[t] = 1.0 - m4_mean[t] / (3.0 * m2_mean[t] * m2_mean[t])
        else:
            binder[t] = 0.0

    for t in range(nT - 1):
        if exch_try_bits[t] > 0:
            exch_ratio[t] = exch_acc_bits[t] / exch_try_bits[t]

    return e_mean, e_err, mabs_mean, mabs_err, cv, chi, chi_conn, m2_mean, sqrt_m2, m4_mean, binder, exch_ratio


# ------------------------------------------------------------
# exact infinite-volume formulas
# ------------------------------------------------------------
def Tc_exact(J=1.0):
    return 2.0 * J / np.log(1.0 + np.sqrt(2.0))


def exact_energy_density(T, J=1.0):
    beta = 1.0 / T
    x = 2.0 * beta * J
    sh = np.sinh(x)
    ch = np.cosh(x)
    th = np.tanh(x)
    k = 2.0 * sh / (ch * ch)
    K = ellipk(k * k)
    return -J / th * (1.0 + (2.0 / np.pi) * (2.0 * th * th - 1.0) * K)


def exact_spontaneous_magnetization(T, J=1.0):
    Tc = Tc_exact(J)
    if np.isscalar(T):
        if T >= Tc:
            return 0.0
        beta = 1.0 / T
        x = np.sinh(2.0 * beta * J)
        return (1.0 - x ** (-4.0)) ** 0.125
    T = np.asarray(T)
    out = np.zeros_like(T, dtype=np.float64)
    mask = T < Tc
    beta = 1.0 / T[mask]
    x = np.sinh(2.0 * beta * J)
    out[mask] = (1.0 - x ** (-4.0)) ** 0.125
    return out


# ------------------------------------------------------------
# temperature ladder auto optimization from pilot sigma_E(beta)
# ------------------------------------------------------------
def optimize_temperature_ladder(T_min, T_max, L, target_c=1.15,
                                n_coarse=16,
                                n_therm_pilot=60,
                                n_meas_pilot=60,
                                n_skip_pilot=2,
                                seed=12345):
    """
    Pilot estimate of sigma_E(beta) and adaptive beta ladder.

    We estimate sigma_E(beta) = sqrt(var(E)).
    For approximately constant exchange difficulty, choose equal spacing in
        s(beta) = integral sigma_E(beta) d beta
    with target step target_c.

    This is a practical Gaussian-overlap heuristic, not an exact optimum.
    """
    beta_min = 1.0 / T_max
    beta_max = 1.0 / T_min
    beta_coarse = np.linspace(beta_min, beta_max, n_coarse)
    sigmaE = np.zeros_like(beta_coarse)
    evar = np.zeros_like(beta_coarse)
    emean = np.zeros_like(beta_coarse)

    # warm-up compile on the first pilot point
    _ = run_single_temp_short(4, beta_coarse[0], 2, 2, 1, seed)

    for i, beta in enumerate(beta_coarse):
        em, ve, sig = run_single_temp_short(
            L=L,
            beta=beta,
            n_therm=n_therm_pilot,
            n_meas=n_meas_pilot,
            n_skip=n_skip_pilot,
            seed=seed + 101 * i,
        )
        emean[i] = em
        evar[i] = ve
        sigmaE[i] = max(sig, 1e-12)

    # cumulative integral of sigmaE over beta using trapezoidal rule
    S = np.zeros_like(beta_coarse)
    for i in range(1, len(beta_coarse)):
        db = beta_coarse[i] - beta_coarse[i - 1]
        S[i] = S[i - 1] + 0.5 * (sigmaE[i] + sigmaE[i - 1]) * db
    S_total = S[-1]

    nT = max(6, int(np.ceil(S_total / target_c)) + 1)
    S_target = np.linspace(0.0, S_total, nT)
    beta_ladder = np.interp(S_target, S, beta_coarse)
    T_ladder = 1.0 / beta_ladder

    pilot = {
        'beta_coarse': beta_coarse,
        'T_coarse': 1.0 / beta_coarse,
        'sigmaE': sigmaE,
        'evar_density': evar,
        'emean_density': emean,
        'S': S,
        'target_c': target_c,
        'nT': int(nT),
        'T_ladder': T_ladder,
        'beta_ladder': beta_ladder,
    }
    return T_ladder, beta_ladder, pilot


# ------------------------------------------------------------
# diagnostics / file outputs
# ------------------------------------------------------------
def validate_incremental_state(L=8, beta=0.4, nsweeps=20, seed=1234):
    set_seed(seed)
    sp, E, M = init_spins_random_with_energy_mag(L)
    for _ in range(nsweeps):
        sweep_checkerboard_incremental(sp, E, M, beta)
    E_exact = total_energy_per_replica(sp)
    M_exact_naive = total_magnetization_per_replica_naive(sp)
    M_exact_pop = total_magnetization_per_replica_popcount(sp)
    return {
        'max_energy_mismatch': int(np.max(np.abs(E - E_exact))),
        'max_mag_naive_mismatch': int(np.max(np.abs(M - M_exact_naive))),
        'max_mag_popcount_mismatch': int(np.max(np.abs(M - M_exact_pop))),
    }


def save_pilot_csv(path, pilot):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['T', 'beta', 'sigmaE', 'evar_density', 'emean_density', 'S'])
        for T, beta, sigmaE, ve, em, S in zip(
                pilot['T_coarse'], pilot['beta_coarse'], pilot['sigmaE'],
                pilot['evar_density'], pilot['emean_density'], pilot['S']):
            w.writerow([T, beta, sigmaE, ve, em, S])


def save_observables_csv(path, Ts, betas, e_mc, de_mc, mabs_mc, dmabs_mc,
                         cv_mc, chi_mc, chi_conn_mc, m2_mc, sqrt_m2_mc, m4_mc,
                         binder_mc, exch_ratio, e_ex, m_ex):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'T', 'beta', 'e_mc', 'e_err', 'e_exact', 'e_diff',
            'mabs_mc', 'mabs_err', 'sqrt_m2_mc', 'm_exact', 'sqrtm2_minus_mexact',
            'cv', 'chi', 'chi_conn', 'm2', 'm4', 'binder',
            'left_exchange_ratio', 'right_exchange_ratio'
        ])
        nT = len(Ts)
        for i in range(nT):
            left = exch_ratio[i - 1] if i - 1 >= 0 else np.nan
            right = exch_ratio[i] if i < nT - 1 else np.nan
            w.writerow([
                Ts[i], betas[i], e_mc[i], de_mc[i], e_ex[i], e_mc[i] - e_ex[i],
                mabs_mc[i], dmabs_mc[i], sqrt_m2_mc[i], m_ex[i], sqrt_m2_mc[i] - m_ex[i],
                cv_mc[i], chi_mc[i], chi_conn_mc[i], m2_mc[i], m4_mc[i], binder_mc[i],
                left, right
            ])


def make_figure(fig_path, Ts, Tc, e_mc, de_mc, e_ex, mabs_mc, dmabs_mc,
                sqrt_m2_mc, m_ex, cv_mc, chi_mc, chi_conn_mc, binder_mc, exch_ratio):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    ax = axes[0, 0]
    ax.errorbar(Ts, e_mc, yerr=de_mc, fmt='o', capsize=3, label='MC')
    ax.plot(Ts, e_ex, '-', lw=2, label='Exact')
    ax.axvline(Tc, color='k', ls='--', alpha=0.6, label=r'$T_c$')
    ax.set_xlabel('T')
    ax.set_ylabel(r'$e=\langle E\rangle/N$')
    ax.set_title('Energy density')
    ax.legend()

    ax = axes[0, 1]
    ax.errorbar(Ts, mabs_mc, yerr=dmabs_mc, fmt='o', capsize=3, label=r'$\langle |m| \rangle$')
    ax.plot(Ts, sqrt_m2_mc, 's-', lw=1.5, label=r'$\sqrt{\langle m^2\rangle}$')
    ax.plot(Ts, m_ex, '-', lw=2, label='Exact spontaneous magnetization')
    ax.axvline(Tc, color='k', ls='--', alpha=0.6)
    ax.set_xlabel('T')
    ax.set_ylabel('magnetization')
    ax.set_title('Magnetization comparison')
    ax.legend()

    ax = axes[0, 2]
    ax.plot(Ts, cv_mc, 'o-', label=r'$C_v$')
    ax.axvline(Tc, color='k', ls='--', alpha=0.6)
    ax.set_xlabel('T')
    ax.set_ylabel(r'$C_v$')
    ax.set_title('Specific heat per site')
    ax.legend()

    ax = axes[1, 0]
    ax.plot(Ts, chi_mc, 'o-', label=r'$\chi=\beta N\langle m^2\rangle$')
    ax.plot(Ts, chi_conn_mc, 's--', label=r'$\chi_{conn}$')
    ax.axvline(Tc, color='k', ls='--', alpha=0.6)
    ax.set_xlabel('T')
    ax.set_ylabel(r'$\chi$')
    ax.set_title('Susceptibility')
    ax.legend()

    ax = axes[1, 1]
    ax.plot(Ts, binder_mc, 'o-', label='Binder cumulant')
    ax.axvline(Tc, color='k', ls='--', alpha=0.6)
    ax.set_xlabel('T')
    ax.set_ylabel(r'$U_4=1-\langle m^4\rangle/(3\langle m^2\rangle^2)$')
    ax.set_title('Binder cumulant')
    ax.legend()

    ax = axes[1, 2]
    mids = 0.5 * (Ts[:-1] + Ts[1:])
    ax.plot(mids, exch_ratio, 'o-')
    ax.axvline(Tc, color='k', ls='--', alpha=0.6)
    ax.set_xlabel('midpoint temperature')
    ax.set_ylabel('exchange ratio')
    ax.set_title('Replica exchange acceptance')

    fig.tight_layout()
    fig.savefig(fig_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def run_and_save(output_dir='output_ising2d_msc_pt_optimized',
                 L=24,
                 T_min=1.4,
                 T_max=3.4,
                 target_c=1.15,
                 n_therm_pilot=60,
                 n_meas_pilot=60,
                 n_skip_pilot=2,
                 n_therm=250,
                 n_meas=250,
                 n_skip=5,
                 seed=20260430,
                 resync_every=0,
                 use_popcount_resync=False):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    validation = validate_incremental_state(L=8, beta=0.4, nsweeps=12, seed=seed % 100000)

    Ts, betas, pilot = optimize_temperature_ladder(
        T_min=T_min,
        T_max=T_max,
        L=min(L, 24),
        target_c=target_c,
        n_coarse=16,
        n_therm_pilot=n_therm_pilot,
        n_meas_pilot=n_meas_pilot,
        n_skip_pilot=n_skip_pilot,
        seed=seed + 7,
    )

    # warm-up compile
    _ = run_parallel_tempering_incremental(
        L=4,
        betas=np.array([1.0 / 3.0, 1.0 / 2.3, 1.0 / 1.7], dtype=np.float64),
        n_therm=2,
        n_meas=2,
        n_skip=1,
        seed=1234,
        resync_every=0,
        use_popcount_resync=0,
    )

    (e_mc, de_mc,
     mabs_mc, dmabs_mc,
     cv_mc, chi_mc, chi_conn_mc,
     m2_mc, sqrt_m2_mc,
     m4_mc, binder_mc,
     exch_ratio) = run_parallel_tempering_incremental(
        L=L,
        betas=betas,
        n_therm=n_therm,
        n_meas=n_meas,
        n_skip=n_skip,
        seed=seed,
        resync_every=resync_every,
        use_popcount_resync=1 if use_popcount_resync else 0,
    )

    e_ex = np.array([exact_energy_density(T) for T in Ts])
    m_ex = exact_spontaneous_magnetization(Ts)
    Tc = Tc_exact()

    save_pilot_csv(out / 'pilot_ladder.csv', pilot)
    save_observables_csv(out / 'observables.csv', Ts, betas, e_mc, de_mc, mabs_mc, dmabs_mc,
                         cv_mc, chi_mc, chi_conn_mc, m2_mc, sqrt_m2_mc, m4_mc,
                         binder_mc, exch_ratio, e_ex, m_ex)
    make_figure(out / 'observables.png', Ts, Tc, e_mc, de_mc, e_ex, mabs_mc, dmabs_mc,
                sqrt_m2_mc, m_ex, cv_mc, chi_mc, chi_conn_mc, binder_mc, exch_ratio)

    summary = {
        'L': int(L),
        'T_min': float(T_min),
        'T_max': float(T_max),
        'target_c': float(target_c),
        'n_therm_pilot': int(n_therm_pilot),
        'n_meas_pilot': int(n_meas_pilot),
        'n_skip_pilot': int(n_skip_pilot),
        'n_therm': int(n_therm),
        'n_meas': int(n_meas),
        'n_skip': int(n_skip),
        'seed': int(seed),
        'Tc_exact': float(Tc),
        'nT': int(len(Ts)),
        'Ts': [float(x) for x in Ts],
        'exchange_ratio': [float(x) for x in exch_ratio],
        'validation': validation,
        'files': {
            'pilot_ladder_csv': str((out / 'pilot_ladder.csv').resolve()),
            'observables_csv': str((out / 'observables.csv').resolve()),
            'figure_png': str((out / 'observables.png').resolve()),
        }
    }
    with open(out / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    with open(out / 'summary.txt', 'w') as f:
        f.write('2D square-lattice Ising (multi-spin + PT optimized)\n')
        f.write(f'L = {L}\n')
        f.write(f'nT = {len(Ts)}\n')
        f.write(f'T range = [{T_min}, {T_max}]\n')
        f.write(f'Tc exact = {Tc}\n')
        f.write(f'Validation: {validation}\n')
        f.write('Temperature ladder:\n')
        for t in Ts:
            f.write(f'  {t:.8f}\n')
        f.write('Exchange ratios:\n')
        for i, x in enumerate(exch_ratio):
            f.write(f'  {Ts[i]:.8f} <-> {Ts[i+1]:.8f}: {x:.6f}\n')

    return summary


if __name__ == '__main__':
    summary = run_and_save()
    print(json.dumps(summary, indent=2))
