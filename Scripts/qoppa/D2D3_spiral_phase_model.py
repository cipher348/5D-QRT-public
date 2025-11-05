#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Lumo – Simulation des Übergangs von einem frequenzbasierten Spiral‑Zeitfeld
f_D2(n) = sin(ω·ln n) zu einer raumhaltigen Phase r(t) ≈ π·φ·e^{‑γt}.

Das Skript enthält:
  • sichere Definition von f_D2 und seiner logarithmischen Ableitung
  • kontinuierliche Integration (Trapez‑Regel) für r(t)
  • diskrete Variante r_n (optional)
  • ideale Referenz r_ideal(t) = π·φ·e^{‑γt}
  • Plot‑Erzeugung für f_D2, r und Δr
  • optionale FFT‑Analyse, um Peaks bei 1/π und 1/φ zu prüfen
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from math import pi, sqrt

# ----------------------------------------------------------------------
# Globale Konstanten
# ----------------------------------------------------------------------
phi = (1 + sqrt(5)) / 2          # Goldener Schnitt

# ----------------------------------------------------------------------
# 1️⃣  Spiral‑Zeitfeld  f_D2(t)  (sicheres Log‑Handling)
# ----------------------------------------------------------------------
def f_D2(t: np.ndarray, omega: float) -> np.ndarray:
    """
    f_D2(t) = sin( ω * ln(t) ),   t > 0
    Ein kleiner epsilon verhindert log(0).
    """
    eps = 1e-12
    t_safe = np.maximum(t, eps)
    return np.sin(omega * np.log(t_safe))


# ----------------------------------------------------------------------
# 2️⃣  Ableitung von ln|f_D2(t)|
# ----------------------------------------------------------------------
def dlog_f_D2_dt(t: np.ndarray, omega: float) -> np.ndarray:
    """
    Analytische Ableitung von ln|f_D2(t)|.
    Formel:
        d/dt ln|sin(ω·ln t)|
        = (ω / t) * cot(ω·ln t)

    Numerisch wird an Stellen, wo sin(ω·ln t) ≈ 0,
    der Beitrag auf 0 gesetzt (Singularität wird durch
    das Dämpfungsgewicht e^{-γt} eliminiert).
    """
    eps = 1e-12
    t = np.maximum(t, eps)

    arg = omega * np.log(t)          # ω·ln t
    sin_arg = np.sin(arg)

    # Maske für sichere Division (|sin| > Schwelle)
    safe = np.abs(sin_arg) > 1e-12

    deriv = np.zeros_like(t)
    deriv[safe] = (omega / t[safe]) * (np.cos(arg[safe]) / sin_arg[safe])
    # unsafe positions bleiben 0
    return deriv


# ----------------------------------------------------------------------
# 3️⃣  Kontinuierliche Integration für r(t)
# ----------------------------------------------------------------------
def r_continuous(t: np.ndarray, omega: float, gamma: float) -> np.ndarray:
    """
    r(t) = ∫_0^t d/dt' ln|f_D2(t')| · e^{-γ t'} dt'
    Numerisch per kumulativem Trapez‑Integral (gleichmäßige dt).
    """
    dt = t[1] - t[0]                     # gleichmäßiger Schritt
    integrand = dlog_f_D2_dt(t, omega) * np.exp(-gamma * t)
    r = np.cumsum(integrand) * dt
    return r


# ----------------------------------------------------------------------
# 4️⃣  Diskrete Variante (optional)
# ----------------------------------------------------------------------
def r_discrete(n_max: int, omega: float, gamma: float) -> np.ndarray:
    """
    Diskrete Summe:
        r_n = Σ_{k=1}^n (ln|f(k)| - ln|f(k‑1)|)·e^{-γ k}
    """
    ks = np.arange(1, n_max + 1)
    f_k   = f_D2(ks,     omega)
    f_k_1 = f_D2(ks - 1, omega)          # f(0) = sin(ω·ln 1) = 0
    delta_log = np.log(np.abs(f_k)   + 1e-15) - \
                np.log(np.abs(f_k_1) + 1e-15)
    weights = np.exp(-gamma * ks)
    r = np.cumsum(delta_log * weights)
    return r


# ----------------------------------------------------------------------
# 5️⃣  Ideale Referenz r_ideal(t) = π·φ·e^{-γ t}
# ----------------------------------------------------------------------
def r_ideal(t: np.ndarray, gamma: float) -> np.ndarray:
    return pi * phi * np.exp(-gamma * t)


# ----------------------------------------------------------------------
# 6️⃣  FFT‑Analyse (optional)
# ----------------------------------------------------------------------
def fft_peaks(signal: np.ndarray, dt: float, n_peaks: int = 5):
    """
    Liefert die n_peaks stärksten Frequenzen (Hz) und ihre Amplituden.
    """
    N = len(signal)
    yf = np.abs(fft(signal - np.mean(signal))) / N
    xf = fftfreq(N, d=dt)[:N // 2]
    mags = yf[:N // 2]

    idx = np.argsort(mags)[-n_peaks:][::-1]
    return xf[idx], mags[idx]


# ----------------------------------------------------------------------
# 7️⃣  Haupt‑Simulation & Plot‑Erzeugung
# ----------------------------------------------------------------------
def run_simulation(omega: float, gamma: float,
                   t_max: float = 200.0, dt: float = 0.01):
    """
    Führt die komplette Pipeline aus:
        * f_D2(t)
        * r(t) (kontinuierlich)
        * Δr(t) = r(t) - π·φ·e^{-γt}
    und erzeugt drei Plots plus eine kurze FFT‑Ausgabe.
    """
    # Zeitgrid (beginnend bei dt, damit ln(t) definiert ist)
    t = np.arange(dt, t_max + dt, dt)

    # 1️⃣ f_D2(t)
    f = f_D2(t, omega)

    # 2️⃣ r(t) – kontinuierlich
    r = r_continuous(t, omega, gamma)

    # 3️⃣ Ideale Referenz
    r_ref = r_ideal(t, gamma)

    # 4️⃣ Δr(t)
    delta_r = r - r_ref

    # --------------------------------------------------------------
    # Plot 1 – Spiral‑Zeitfeld
    # --------------------------------------------------------------
    plt.figure(figsize=(10, 3))
    plt.plot(t, f, label=r"$f_{D2}(t)=\sin(\omega\ln t)$")
    plt.title(rf"Spiral‑Zeitfeld (ω={omega:.5f})")
    plt.xlabel("Zeit $t$")
    plt.ylabel(r"$f_{D2}(t)$")
    plt.grid(True, ls=":")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------------
    # Plot 2 – Raumphase r(t) + Ideal
    # --------------------------------------------------------------
    plt.figure(figsize=(10, 3))
    plt.plot(t, r, label=r"$r(t)$ (numerisch)")
    plt.plot(t, r_ref, '--', label=r"$\pi\varphi\,e^{-\gamma t}$ (Ideal)")
    plt.title(rf"Raum‑Phase (ω={omega:.5f}, γ={gamma:.5e})")
    plt.xlabel("Zeit $t$")
    plt.ylabel(r"$r(t)$")
    plt.grid(True, ls=":")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------------
    # Plot 3 – Abweichung Δr(t)
    # --------------------------------------------------------------
    plt.figure(figsize=(10, 3))
    plt.plot(t, delta_r, label=r"$\Delta r(t)=r(t)-\pi\varphi e^{-\gamma t}$")
    plt.axhline(0, color="gray", lw=0.8, ls="--")
    plt.title(rf"Abweichung vom Ideal (ω={omega:.5f})")
    plt.xlabel("Zeit $t$")
    plt.ylabel(r"$\Delta r(t)$")
    plt.grid(True, ls=":")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------------
    # FFT‑Check (optional)
    # --------------------------------------------------------------
    freqs, amps = fft_peaks(r, dt, n_peaks=6)
    print("\nFFT‑Peaks (Hz) – Amplituden (normiert):")
    for f, a in zip(freqs, amps):
        print(f"  {f: .5f}  →  {a:.4f}")

    # Erwartete Peaks (theoretisch) bei 1/π ≈ 0.31831 Hz und 1/φ ≈ 0.61803 Hz.
    # Da die Zeiteinheit abstrakt ist, können die Werte leicht skaliert sein.
    # Anpassungen von dt ändern die physikalische Frequenzinterpretation.


# ----------------------------------------------------------------------
# 8️⃣  Parameter‑Sweep (Beispiel)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    """
    Hier kannst du bequem verschiedene ω‑Werte und die zugehörige
    Kopplungsfunktion γ(ω) testen.  Das Skript führt für jedes Paar
    einen kompletten Durchlauf (Plots + FFT‑Ausgabe) aus.
    """

    # ---- 1️⃣  Auswahl der ω‑Werte (nach deiner Vorgabe) ----
    omega_vals = np.linspace(3.300, 3.304, 5)   # z. B. 5 äquidistante Werte

    # ---- 2️⃣  Kopplungsfunktion γ(ω) wählen ----
    # Variante A: γ = 1/ω²
    gamma_fun_A = lambda w: 1.0 / (w ** 2)

    # Variante B: γ = ln(ω) / π
    gamma_fun_B = lambda w: np.log(w) / np.pi

    # ---- 3️⃣  Welches Modell verwenden? ----
    # Setze `use_variant = "A"` oder `"B"`
    use_variant = "A"

    gamma_fun = gamma_fun_A if use_variant == "A" else gamma_fun_B

    # ---- 4️⃣  Schleife über alle ω‑Werte ----
    for omega in omega_vals:
        gamma = gamma_fun(omega)

        print("\n" + "=" * 70)
        print(f"Simulation für ω = {omega:.6f}  →  γ = {gamma:.6e}  (Variante {use_variant})")
        print("=" * 70)

        # Jeder Durchlauf erzeugt die drei Plots und gibt die FFT‑Peaks aus.
        run_simulation(
            omega=omega,
            gamma=gamma,
            t_max=200.0,   # Gesamtdauer (kann angepasst werden)
            dt=0.01        # Zeitschritt (kleiner → höher Auflösung)
        )

        # Optional: kurze Pause zwischen den Durchläufen, falls du das Skript
        # interaktiv nutzt und nicht sofort das nächste Plot‑Fenster öffnen willst.
        # import time
        # time.sleep(1)
