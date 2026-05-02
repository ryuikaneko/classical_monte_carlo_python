#!/usr/bin/env python3
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
