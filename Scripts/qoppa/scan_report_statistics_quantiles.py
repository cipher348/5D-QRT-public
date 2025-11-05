#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib, pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
import scipy.optimize as opt

# --------------------------------------------------------------
# 1. CSV‑Dateien einlesen (automatisch)
# --------------------------------------------------------------
files = sorted(pathlib.Path('.').glob('scan_report_*.csv'))
dfs   = {}
Ns    = []
for f in files:
    n = int(f.stem.split('_')[-1])            # Zahl hinter dem Unterstrich
    dfs[n] = pd.read_csv(f)
    Ns.append(n)

Ns = np.array(sorted(Ns), dtype=float)

# --------------------------------------------------------------
# 2. Hilfsfunktionen
# --------------------------------------------------------------
def first_cross(df, thr=10.0):
    crossed = df[df['max_abs_delta_r'] > thr]
    return crossed.iloc[0][['omega','max_abs_delta_r']] if not crossed.empty else None

def robust_stats(df):
    q95 = np.quantile(df['max_abs_delta_r'], 0.95)
    q99 = np.quantile(df['max_abs_delta_r'], 0.99)
    mae = np.mean(np.abs(df['max_abs_delta_r']))
    med = np.median(df['max_abs_delta_r'])
    mad = np.median(np.abs(df['max_abs_delta_r'] - med))
    return q95, q99, mae, med, mad

# --------------------------------------------------------------
# 3. Basis‑Statistiken ausgeben
# --------------------------------------------------------------
print('\n=== Basis‑Statistiken ===')
for n in sorted(dfs):
    df = dfs[n]
    first = first_cross(df)
    q95, q99, mae, med, mad = robust_stats(df)
    print(f'\n{n:,} Punkte')
    print('Erste Überschreitung >10 :', first.to_dict())
    print('95 %‑Quantil               : {:.2f}'.format(q95))
    print('99 %‑Quantil               : {:.2f}'.format(q99))
    print('MAE (Durchschnitt)        : {:.2f}'.format(mae))
    print('Median                    : {:.2f}'.format(med))
    print('MAD                       : {:.2f}'.format(mad))

# --------------------------------------------------------------
# 4. 95‑%‑Quantil vs. N (log‑linear Fit)
# --------------------------------------------------------------
Q95 = np.array([robust_stats(dfs[n])[0] for n in Ns])
def linlog(x, a, b):
    return a + b*np.log(x)

popt, _ = opt.curve_fit(linlog, Ns, Q95, p0=[200., 5.])
a_fit, b_fit = popt
print('\n=== Fit‑Ergebnis (log‑linear) ===')
print(f'Q95(N) ≈ a + b·log(N)  →  a = {a_fit:.2f},  b = {b_fit:.2f}')

# Plot des Fits
plt.figure(figsize=(6,4))
plt.scatter(Ns, Q95, label='Messwerte', color='C0')
x_fit = np.linspace(Ns.min(), Ns.max(), 200)
plt.plot(x_fit, linlog(x_fit, *popt), '--', color='C1', label='Log‑linear‑Fit')
plt.xscale('log')
plt.xlabel('Anzahl Ω‑Samples (log‑Skala)')
plt.ylabel(r'95‑%‑Quantil von $|\Delta r|_{\max}$')
plt.title('Skalierung des 95‑%‑Quantils')
plt.legend()
plt.grid(True, which='both', ls=':')
plt.tight_layout()
plt.show()

# --------------------------------------------------------------
# 5. Histogramm für das feinste Netz (höchste N)
# --------------------------------------------------------------
max_n = Ns.max()
df_max = dfs[int(max_n)]
plt.figure(figsize=(8,4))
sns.histplot(df_max['max_abs_delta_r'],
             bins=200, log_scale=(True, False),
             kde=False, color='steelblue')
plt.xlabel(r'$|\Delta r|_{\max}$')
plt.title(rf'Histogramm von $|\Delta r|_{{\max}}$ ({int(max_n):,} Punkte)')
plt.grid(True, which='both', ls=':')
plt.tight_layout()
plt.show()
