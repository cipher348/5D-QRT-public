#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyse_large_peak_v2.py

- Lädt alle scan_report_*.csv (200 – 5000 Punkte).
- Passt das Analyse‑Fenster dynamisch an die Datenmenge an.
- Filtert extreme Ausreißer (> 10 000) heraus.
- Bestimmt ω_peak, Δr_peak, FWHM (falls möglich) und die Fläche.
- Gibt eine saubere Text‑ und CSV‑Tabelle aus.
"""

import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# ----------------------------------------------------------------------
# 1. Hilfsfunktionen
# ----------------------------------------------------------------------
def dynamic_window(df, centre=3.301165, min_points=5):
    """
    Wählt ein Fenster um `centre`, das mindestens `min_points`
    Daten enthält. Das Fenster wird schrittweise vergrößert,
    bis die Bedingung erfüllt ist (max. 1e‑3 Breite als Obergrenze).
    """
    half_width = 2e-5                     # Start‑Breite
    while True:
        mask = (df["omega"] > centre - half_width) & (df["omega"] < centre + half_width)
        sub = df[mask]
        if len(sub) >= min_points or half_width > 1e-3:
            return sub
        half_width *= 2                    # Fenster verdoppeln, falls zu wenig Punkte


def filter_outliers(df, threshold=10_000):
    """Entfernt Zeilen, deren Δr den Schwellenwert überschreitet."""
    return df[df["max_abs_delta_r"] < threshold]


def spline_peak(omega, delta, fine_factor=30_000):
    """Kubischer Spline → exakte Peak‑Position."""
    cs = CubicSpline(omega, delta)
    omega_fine = np.linspace(omega.min(), omega.max(), fine_factor)
    delta_fine = cs(omega_fine)
    idx = np.argmax(delta_fine)
    return omega_fine[idx], delta_fine[idx], cs, omega_fine, delta_fine


def fwhm_from_spline(cs, omega_max, delta_max, rel_tol=1e-3):
    """
    Halb‑Maximum‑Schnitte (links/rechts) aus dem Spline.
    Wenn kein Schnitt gefunden wird, gibt (nan, nan, nan) zurück.
    """
    half = delta_max / 2.0

    # links – wir suchen das erste Element ≥ half (mit Toleranz)
    left_grid = np.linspace(cs.x[0], omega_max, 10_000)
    left_vals = cs(left_grid)
    left_candidates = np.where(left_vals >= half * (1 - rel_tol))[0]
    if left_candidates.size == 0:
        return np.nan, np.nan, np.nan
    omega_left = left_grid[left_candidates[0]]

    # rechts – erstes Element ≤ half
    right_grid = np.linspace(omega_max, cs.x[-1], 10_000)
    right_vals = cs(right_grid)
    right_candidates = np.where(right_vals <= half * (1 + rel_tol))[0]
    if right_candidates.size == 0:
        return np.nan, np.nan, np.nan
    omega_right = right_grid[right_candidates[0]]

    width = omega_right - omega_left
    if width <= 0:
        width = np.nan
    return omega_left, omega_right, width


def area_under_peak(cs, omega_left, omega_right, n=10_000):
    """Trapez‑Integration zwischen den Halb‑Maximum‑Grenzen."""
    if np.isnan(omega_left) or np.isnan(omega_right):
        return np.nan
    xs = np.linspace(omega_left, omega_right, n)
    ys = cs(xs)
    return np.trapezoid(ys, xs)


def pretty_float(x, sig=5):
    """Rundet auf `sig` signifikante Stellen, robust gegen NaN/Inf."""
    if pd.isna(x) or not np.isfinite(x):
        return "—"
    if x == 0:
        return "0"
    from math import log10, floor
    digits = max(sig - int(floor(log10(abs(x)))) - 1, 0)
    fmt = f"{{:.{digits}f}}"
    return fmt.format(x)


# ----------------------------------------------------------------------
# 2. Verarbeitung aller CSV‑Dateien
# ----------------------------------------------------------------------
data_dir = pathlib.Path(".")
csv_files = sorted(data_dir.glob("scan_report.csv"))

summary = []   # sammelt Zeilen für die End‑Tabelle

for csv_path in csv_files:
    # -------------------- 2.1 Daten einlesen --------------------
    df = pd.read_csv(csv_path)

    # -------------------- 2.2 Dynamisches Fenster um den Peak --------------------
    peak_df = dynamic_window(df, centre=3.301165, min_points=5)

    # -------------------- 2.3 Ausreißer‑Filter (Δr > 10 000) --------------------
    peak_df = filter_outliers(peak_df, threshold=10_000)

    if peak_df.empty:
        print(f"[WARN] Keine gültigen Daten im Peak‑Fenster für {csv_path.name}")
        continue

    # -------------------- 2.4 Spline‑Fit & Maximum --------------------
    omega_max, delta_max, cs, omega_fine, delta_fine = spline_peak(
        peak_df["omega"].values,
        peak_df["max_abs_delta_r"].values,
        fine_factor=40_000,
    )

    # -------------------- 2.5 Peak‑Parameter --------------------
    try:
        omega_left, omega_right, fwhm = fwhm_from_spline(cs, omega_max, delta_max)
        area = area_under_peak(cs, omega_left, omega_right)
    except Exception as exc:
        omega_left = omega_right = fwhm = np.nan
        area = np.nan
        print(f"[INFO] FWHM‑Berechnung fehlgeschlagen für {csv_path.name}: {exc}")

    # -------------------- 2.6 Ergebnis speichern --------------------
    summary.append(
        {
            "file": csv_path.name,
            "N_points": len(df),
            "omega_peak": omega_max,
            "delta_peak": delta_max,
            "FWHM": fwhm,
            "area": area,
        }
    )

    # -------------------- 2.7 Plot erzeugen --------------------
    plt.figure(figsize=(7, 4))
    plt.plot(df["omega"], df["max_abs_delta_r"], "o", ms=3,
             color="steelblue", alpha=0.6, label="raw data")
    plt.plot(omega_fine, delta_fine, "-", color="darkorange",
             lw=1.5, label="cubic spline")

    # Peak‑Markierungen
    plt.axvline(omega_max, color="red", ls="--", lw=1,
                label=r"$\omega_{\rm peak}$")
    plt.scatter([omega_max], [delta_max], color="red", zorder=5)

    if not np.isnan(fwhm):
        plt.axvline(omega_left,  color="green", ls=":", lw=1,
                    label=r"$\omega_{\rm HM,left}$")
        plt.axvline(omega_right, color="green", ls=":", lw=1,
                    label=r"$\omega_{\rm HM,right}$")
        plt.hlines(delta_max/2, omega_left, omega_right,
                   color="purple", ls="-.", lw=1,
                   label=r"$\Delta r/2$")

    plt.title(f"Large Δr‑peak – {csv_path.name}")
    plt.xlabel(r"$\omega$")
    plt.ylabel(r"$|\Delta r|_{\max}$")

    # Deduplication der Legende (verhindert Dopplungen)
    handles, labels = plt.gca().get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    plt.legend(list(uniq.values()), list(uniq.keys()),
               loc="upper left", fontsize="small")
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.savefig(csv_path.with_suffix(".png"), dpi=200)
    plt.close()

# ----------------------------------------------------------------------
# 3. Gesamttabelle ausgeben & speichern
# ----------------------------------------------------------------------
summary_df = pd.DataFrame(summary).sort_values("N_points")

print("\n=== Summary of the large Δr‑peak ===")
header = "{:>20} {:>9} {:>12} {:>12} {:>10} {:>12}".format(
    "File", "N", "ω_peak", "Δr_peak", "FWHM", "Area")
print(header)
print("-" * len(header))
for _, row in summary_df.iterrows():
    print("{:>20} {:>9} {:>12} {:>12} {:>10} {:>12}".format(
        row["file"],
        row["N_points"],
        pretty_float(row["omega_peak"], 6),
        pretty_float(row["delta_peak"], 0),          # keine Dezimalstellen für Δr
        pretty_float(row["FWHM"], 5) if not pd.isna(row["FWHM"]) else "—",
        pretty_float(row["area"], 5)   if not pd.isna(row["area"]) else "—",
    ))

# CSV‑Export (für LaTeX‑Tabellen, Excel, …)
summary_df.to_csv("large_peak_summary_v2.csv", index=False)

# ----------------------------------------------------------------------
# 4. (Optional) LaTeX‑Tabelle – einfach auskommentieren
# ----------------------------------------------------------------------
# latex_tbl = summary_df.copy()
# latex_tbl["omega_peak"] = latex_tbl["omega_peak"].apply(lambda x: f"{x:.6f}")
# latex_tbl["delta_peak"] = latex_tbl["delta_peak"].apply(lambda x: f"{x:.0f}")
# latex_tbl["FWHM"]       = latex_tbl["FWHM"].apply(lambda x: f"{x:.5f}" if not pd.isna(x) else "")
# latex_tbl["area"]       = latex_tbl["area"].apply(lambda x: f"{x:.5f}" if not pd.isna(x) else "")
#
# print("\nLaTeX‑Tabelle (copy‑paste in dein Manuskript):")
# print(latex_tbl.to_latex(index=False,
#                          column_format="lrrrrr",
#                          caption="Characteristics of the large $|\\Delta r|$ peak for the different scan resolutions.",
#                          label="tab:large-peak",
#                          escape=False))
