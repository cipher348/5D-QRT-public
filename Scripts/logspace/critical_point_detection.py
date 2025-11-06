#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
log_point_analysis.py

Ziel:
    - Untersuche den Punkt x_q = ln(theta) = 3.302 (q) in einer
      log‑periodisch modulierten Funktion
          r(θ) = exp(a·θ) * [1 + D·cos(ω·lnθ + φ)] .
    - Bestimme den exakten Δ_phase‑Wert, die erste und zweite Ableitung
      an dieser Stelle und visualisiere das Ergebnis.
    - Entscheide, ob x_q ein natürlicher „Reaktions‑/Critical‑Point“
      (Extrem‑ oder Wendepunkt) ist.

Verwendung:
    - Das Skript erzeugt synthetische Testdaten (wie im ursprünglichen
      Beispiel).  Wenn du eigene Messwerte hast, ersetze einfach den
      Daten‑Erzeugungs‑Block durch das Einlesen deiner Dateien.
"""

import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# 1. Parameter (können beliebig geändert werden)
# -------------------------------------------------------------------------
a_true     = 1.0 / np.pi          # ≈ 0.318309886
D_true     = 0.40
omega_true = 3.302                # log‑Frequenz (unsere q‑Markierung)
phi_true   = 0.20

# Gitter in lnθ (x) – wir wählen ein feines Gitter, damit die Interpolation
# exakt ist, aber das Prinzip funktioniert auch für beliebige Messdaten.
N      = 5000
x_grid = np.linspace(-2, 4, N)          # x = lnθ  von -2 bis 4
theta  = np.exp(x_grid)                # zurück zu θ

# -------------------------------------------------------------------------
# 2. Modellfunktion (synthetische Daten)
# -------------------------------------------------------------------------
r = np.exp(a_true * theta) * (1.0 +
            D_true * np.cos(omega_true * np.log(theta) + phi_true))

# Optional: kleines Rauschen hinzufügen (realistische Messungen)
rng = np.random.default_rng(123)
r *= np.exp(0.001 * rng.standard_normal(N))   # ≈0.1 % multiplicatives Rauschen

# -------------------------------------------------------------------------
# 3. Power‑Law‑Basis (Fit von ln r gegen ln θ)
# -------------------------------------------------------------------------
x = np.log(theta)                     # x = lnθ (identisch zu x_grid)
lnr = np.log(r)

# Linearer Fit in Log‑Log → Power‑Law
a_fit, b_fit = np.polyfit(x, lnr, 1)   # lnr ≈ a_fit·x + b_fit
r_base = np.exp(a_fit * x + b_fit)    # glatte Basisfunktion

# -------------------------------------------------------------------------
# 4. Log‑Phase Δ_phase(x) = log(r / r_base)
# -------------------------------------------------------------------------
delta_phase = np.log(r / r_base)       # reines log‑periodisches Signal

# -------------------------------------------------------------------------
# 5. Analyse exakt bei x_q = 3.302
# -------------------------------------------------------------------------
x_q = 3.302
theta_q = np.exp(x_q)                 # ≈ 27.18

# ---- 5.1. Interpolierter Δ_phase-Wert ----
# Wir benutzen lineare Interpolation zwischen den beiden nächstliegenden Gitterpunkten.
if not (x.min() <= x_q <= x.max()):
    raise ValueError("x_q liegt außerhalb des Datenbereichs.")

idx_left  = np.searchsorted(x, x_q) - 1   # Index des linken Punktes
idx_right = idx_left + 1

x_l,   x_r   = x[idx_left],   x[idx_right]
dp_l,  dp_r  = delta_phase[idx_left], delta_phase[idx_right]

t = (x_q - x_l) / (x_r - x_l)            # Gewicht für lineare Interpolation
delta_q = dp_l + t * (dp_r - dp_l)        # interpolierter Δ_phase bei x_q

# ---- 5.2. Numerische Ableitungen (zentrale Differenzen) ----
# Wir benötigen ein feines lokales Gitter um x_q, also interpolieren wir
# die Funktion auf ein feineres Raster rund um x_q.
dx_fine = 1e-5
x_fine = np.arange(x_q - 5*dx_fine, x_q + 5*dx_fine + dx_fine, dx_fine)

# Interpolation von delta_phase auf das feine Raster (lineare Interpolation)
delta_fine = np.interp(x_fine, x, delta_phase)

# Erste Ableitung (zentrale Differenz)
d1 = (delta_fine[6] - delta_fine[4]) / (2*dx_fine)   # (x_q+dx) – (x_q-dx) / 2dx
# Zweite Ableitung (zentrale Differenz)
d2 = (delta_fine[6] - 2*delta_fine[5] + delta_fine[4]) / (dx_fine**2)

# ---- 5.3. Relative Änderung des Originalsignals r ----
# Wir können die relative Änderung von r an derselben Stelle prüfen:
r_q = np.interp(x_q, x, r)
r_plus  = np.interp(x_q + dx_fine, x, r)
r_minus = np.interp(x_q - dx_fine, x, r)

rel_derivative = (r_plus - r_minus) / (2*dx_fine * r_q)   # (dr/dx)/r

# -------------------------------------------------------------------------
# 6. Ausgabe der Kennzahlen
# -------------------------------------------------------------------------
print("\n=== Analyse bei x_q = ln(theta) = 3.302 ===")
print(f"Theta_q               = {theta_q:.6f}")
print(f"Δ_phase (interpoliert) = {delta_q:.6f}")
print(f"Erste Ableitung dΔ/dx = {d1:.6e}")
print(f"Zweite Ableitung d²Δ/dx² = {d2:.6e}")
print(f"Relative Änderung von r ( (dr/dx)/r ) = {rel_derivative:.6e}")

# Hinweis zur Interpretation:
# • Ist d1 ≈ 0 → lokales Extrem (Maximum/Minimum) von Δ_phase.
# • Ist d2 ≈ 0 → Wendepunkt (Änderung der Krümmung).
# • Ein signifikanter Wert von d1 oder d2 im Vergleich zu den
#   typischen Schwankungen (≈ D·ω) deutet darauf hin, dass x_q
#   ein „kritischer“ Punkt ist.

# -------------------------------------------------------------------------
# 7. Plot – Δ_phase mit hervorgehobenem Punkt
# -------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(x, delta_phase, 'k.', markersize=2, label=r'$\Delta_{\mathrm{phase}}$')
plt.axvline(x_q, color='b', ls='--', label=r'$q = 3.302$')
plt.plot(x_q, delta_q, 'mo', markersize=10, label=r'Punkt bei $q$')
plt.xlabel(r'$x = \ln\theta$')
plt.ylabel(r'$\Delta_{\mathrm{phase}}$')
plt.title(r'Log‑phase und Analyse des Punktes $q = 3.302$')
plt.legend()
plt.grid(True, ls='--', alpha=0.5)
plt.tight_layout()
plt.show()
