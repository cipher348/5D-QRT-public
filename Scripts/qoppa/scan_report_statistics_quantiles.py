#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyse von scan_report‑CSV‑Dateien.
- Gibt Basis‑Statistiken (Quantile, MAE, Median, MAD) aus.
- Falls ≥2 Dateien vorhanden sind: log‑linearer Fit des 95 %-Quantils.
- Plottet den Fit und ein Histogramm des größten Datensatzes.
"""

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as opt

# ----------------------------------------------------------------------
# 1. Argument‑Parser & Dateiauswahl
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Analyse von scan_report CSV‑Dateien"
)
parser.add_argument(
    "path",
    nargs="*",
    default=["scan_report.csv"],
    help="Eine oder mehrere CSV‑Dateien (Standard: scan_report.csv)",
)
args = parser.parse_args()

# Nur Pfade, die tatsächlich CSV‑Endungen besitzen, übernehmen
files = sorted(
    [pathlib.Path(p) for p in args.path if pathlib.Path(p).suffix.lower() == ".csv"]
)

if not files:
    sys.exit("❌ Keine CSV‑Dateien gefunden. Bitte Pfad(e) prüfen.")
# ----------------------------------------------------------------------
# 2. Daten einlesen & Index‑Zuweisung (N)
# ----------------------------------------------------------------------
dfs = {}          # Schlüssel = N (z. B. 1, 2, 3 …)
Ns  = []          # Liste aller N‑Werte

for f in files:
    # Versuche, eine Nummer aus dem Dateinamen zu extrahieren:
    #   scan_report_42.csv → 42
    # Falls das nicht klappt, verwende einfach einen fortlaufenden Index.
    try:
        n = int(f.stem.split("_")[-1])
    except ValueError:
        n = len(Ns) + 1
    dfs[n] = pd.read_csv(f)
    Ns.append(n)

Ns = np.array(sorted(Ns), dtype=float)

# ----------------------------------------------------------------------
# 3. Hilfsfunktionen
# ----------------------------------------------------------------------
def first_cross(df, thr: float = 10.0):
    """Erste Zeile, in der max_abs_delta_r > thr."""
    crossed = df[df["max_abs_delta_r"] > thr]
    return (
        crossed.iloc[0][["omega", "max_abs_delta_r"]]
        if not crossed.empty
        else None
    )


def robust_stats(df):
    """95‑%‑Quantil, 99‑%‑Quantil, MAE, Median, MAD."""
    q95 = np.quantile(df["max_abs_delta_r"], 0.95)
    q99 = np.quantile(df["max_abs_delta_r"], 0.99)
    mae = np.mean(np.abs(df["max_abs_delta_r"]))
    med = np.median(df["max_abs_delta_r"])
    mad = np.median(np.abs(df["max_abs_delta_r"] - med))
    return q95, q99, mae, med, mad


# ----------------------------------------------------------------------
# 4. Basis‑Statistiken ausgeben
# ----------------------------------------------------------------------
print("\n=== Basis‑Statistiken ===")
for n in sorted(dfs):
    df = dfs[n]
    first = first_cross(df)
    q95, q99, mae, med, mad = robust_stats(df)

    print(f"\n{n:,} Punkte")
    if first is not None:
        print("Erste Überschreitung >10 :", first.to_dict())
    else:
        print("Erste Überschreitung >10 :", "Keine")
    print(f"95 %‑Quantil               : {q95:.2f}")
    print(f"99 %‑Quantil               : {q99:.2f}")
    print(f"MAE (Durchschnitt)         : {mae:.2f}")
    print(f"Median                     : {med:.2f}")
    print(f"MAD                        : {mad:.2f}")

# ----------------------------------------------------------------------
# 5. Log‑linearer Fit (nur wenn ≥2 Punkte vorhanden)
# ----------------------------------------------------------------------
Q95 = np.array([robust_stats(dfs[n])[0] for n in Ns])

def linlog(x, a, b):
    """a + b·log(x)"""
    return a + b * np.log(x)


if len(Ns) >= 2:
    # curve_fit benötigt mindestens so viele Datenpunkte wie Parameter
    popt, pcov = opt.curve_fit(linlog, Ns, Q95, p0=[200.0, 5.0])
    a_fit, b_fit = popt
    perr = np.sqrt(np.diag(pcov))  # Standardfehler

    print("\n=== Fit‑Ergebnis (log‑linear) ===")
    print(f"Q95(N) ≈ a + b·log(N)  →  a = {a_fit:.2f} ± {perr[0]:.2f}, "
          f"b = {b_fit:.2f} ± {perr[1]:.2f}")

    # Plot des Fits
    plt.figure(figsize=(6, 4))
    plt.scatter(Ns, Q95, label="Messwerte", color="C0")
    x_fit = np.linspace(Ns.min(), Ns.max(), 200)
    plt.plot(x_fit, linlog(x_fit, *popt), "--", color="C1", label="Log‑linear‑Fit")
    plt.xscale("log")
    plt.xlabel("Anzahl Ω‑Samples (log‑Skala)")
    plt.ylabel(r"95 %‑Quantil von $|\Delta r|_{\max}$")
    plt.title("Skalierung des 95 %‑Quantils")
    plt.legend()
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.show()
else:
    print("\n⚠️  Nur ein Datensatz vorhanden → kein Fit möglich.")

# ----------------------------------------------------------------------
# 6. Histogramm des feinsten Netzes (größtes N)
# ----------------------------------------------------------------------
max_n = Ns.max()
df_max = dfs[int(max_n)]

plt.figure(figsize=(8, 4))
sns.histplot(
    df_max["max_abs_delta_r"],
    bins=200,
    log_scale=(True, False),
    kde=False,
    color="steelblue",
)
plt.xlabel(r"$|\Delta r|_{\max}$")
plt.title(rf"Histogramm von $|\Delta r|_{{\max}}$ ({int(max_n):,} Punkte)")
plt.grid(True, which="both", ls=":")
plt.tight_layout()
plt.show()
