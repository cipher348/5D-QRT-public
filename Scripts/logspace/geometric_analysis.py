#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
logperiodic_analysis_v2.py

Freier Analyse‑Workflow für mögliche log‑periodische Modulationen
in einer Messreihe r(θ) = exp(a·θ)·[1 + D·cos(ω·lnθ + φ)].

Neu:
* Zwei separate Figuren (Δ‑Phase‑Plot + FFT‑Plot).
* Im Δ‑Phase‑Plot eine blaue vertikale Linie bei q = 3.302 (rad/lnθ)
  – die Linie erscheint in der Legende als “q”.
* Keine “Soll‑/Reference‑Werte” mehr im Code oder in den Ausgaben.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.optimize import curve_fit

# -------------------------------------------------------------------------
# Hilfsfunktionen
# -------------------------------------------------------------------------
def fit_powerlaw(theta, r):
    """
    Fit ln(r) = a·ln(theta) + b  (Power‑Law).
    Rückgabe: a, b, r_base (glatte Basisfunktion).
    """
    x = np.log(theta)
    y = np.log(r)
    a, b = np.polyfit(x, y, 1)          # lineare Regression in Log‑Log
    r_base = np.exp(a * x + b)          # zurück in lineare Werte
    return a, b, r_base


def fft_peak(x, signal):
    """
    Berechnet das Amplitudenspektrum und gibt
    (omega_peak, amplitude, freq_array, amp_array) zurück.
    """
    dx = x[1] - x[0]                    # konstantes Δx (x = lnθ)
    freq = rfftfreq(len(signal), d=dx)   # Einheit: rad / lnθ
    amp = np.abs(rfft(signal))

    # DC‑Komponente entfernen, weil wir das periodische Signal suchen
    amp[0] = 0.0
    idx = np.argmax(amp)                # größtes verbleibendes Maximum
    return freq[idx], amp[idx], freq, amp


def bootstrap_omega(x, signal, window, n_boot=500):
    """
    Resampelt das detrendete Signal (mit Ersatz), wendet das gleiche Fenster
    an und bestimmt ω für jedes Bootstrap‑Sample.
    Rückgabe: Array mit allen ω‑Schätzungen.
    """
    N = len(signal)
    omegas = []

    for _ in range(n_boot):
        idx = np.random.randint(0, N, N)   # Resampling mit Ersatz
        s = signal[idx]

        # Fenster + Normierung (wie im Original)
        s = s - np.mean(s)
        s = s * window
        s = s / np.std(s)

        ω, _, _, _ = fft_peak(x, s)
        omegas.append(ω)

    return np.array(omegas)


def logperiodic_model(x, D, phi, c0, c1, omega):
    """
    Δ_phase(x) ≈ D·cos(omega·x + phi) + c0 + c1·x
    (c0, c1) erlauben einen kleinen linearen Rest‑Trend.
    """
    return D * np.cos(omega * x + phi) + c0 + c1 * x


# -------------------------------------------------------------------------
# Haupt‑Analyse‑Routine
# -------------------------------------------------------------------------
def analyse_logperiodic(theta, r,
                       n_bootstrap=800,
                       window_func=np.hanning,
                       plot=True):
    """
    Komplett‑Workflow:
    1. Power‑Law‑Basis fitten
    2. Log‑Phase extrahieren
    3. Detrending + Fenster
    4. FFT → ω̂
    5. Bootstrap → σ(ω)
    6. Nicht‑linearer Fit (D, φ, c0, c1) unter Verwendung von ω̂
    7. (optional) Plot (zwei separate Figuren)
    Rückgabe: dict mit allen geschätzten Parametern + Unsicherheiten.
    """
    # -------------------------------------------------
    # 1. Basis‑Fit (Power‑Law)
    # -------------------------------------------------
    a, b, r_base = fit_powerlaw(theta, r)

    # -------------------------------------------------
    # 2. Log‑Phase (reines log‑periodisches Signal)
    # -------------------------------------------------
    delta_phase = np.log(r / r_base)

    # -------------------------------------------------
    # 3. Detrending + Fenster
    # -------------------------------------------------
    delta_dt = delta_phase - np.mean(delta_phase)   # DC entfernen
    window   = window_func(len(delta_dt))           # z. B. Hann
    delta_win = delta_dt * window
    delta_win /= np.std(delta_win)                 # Normierung

    # -------------------------------------------------
    # 4. FFT → dominante Frequenz ω̂
    # -------------------------------------------------
    x = np.log(theta)                              # x = lnθ (FFT‑Achse)
    omega_hat, amp_max, freq_arr, amp_arr = fft_peak(x, delta_win)

    # -------------------------------------------------
    # 5. Bootstrap‑Unsicherheit für ω̂
    # -------------------------------------------------
    boot_omegas = bootstrap_omega(x, delta_dt, window,
                                  n_boot=n_bootstrap)
    omega_std   = boot_omegas.std(ddof=1)

    # -------------------------------------------------
    # 6. Nicht‑linearer Fit (D, φ, c0, c1) – ω̂ fix
    # -------------------------------------------------
    def model_fixed_omega(x, D, phi, c0, c1):
        return logperiodic_model(x, D, phi, c0, c1, omega_hat)

    p0 = [0.3, 0.0, 0.0, 0.0]                     # Startwerte
    bounds = ([-0.95, -np.pi, -np.inf, -np.inf],
              [ 0.95,  np.pi,  np.inf,  np.inf])

    popt, pcov = curve_fit(model_fixed_omega,
                           x, delta_phase,
                           p0=p0,
                           bounds=bounds,
                           maxfev=20000)

    D_hat, phi_hat, c0_hat, c1_hat = popt
    perr = np.sqrt(np.diag(pcov))

    # -------------------------------------------------
    # 7. Plot (zwei separate Figuren)
    # -------------------------------------------------
    if plot:
        # ---------- Figur 1 – Δ_phase + Fit ----------
        fig1 = plt.figure(figsize=(7, 5))
        ax1 = fig1.add_subplot(111)

        ax1.plot(x, delta_phase, 'k.', markersize=3,
                 label='Δ_phase (Daten)')
        ax1.plot(x,
                 model_fixed_omega(x, D_hat, phi_hat, c0_hat, c1_hat),
                 'r-', lw=2, label='Fit (D, φ, c0, c1)')

        # Vertikale Linie bei q = 3.302 (blau) + Legende‑Eintrag „q“
        q_val = 3.302
        ax1.axvline(q_val, color='b', ls='--', label='ϙ')

        ax1.set_xlabel(r"$x = \ln\theta$")
        ax1.set_ylabel(r"$\Delta_{\mathrm{phase}}$")
        ax1.set_title("Log‑Phase mit nicht‑linearem Fit")
        ax1.legend()
        ax1.grid(True, ls='--', alpha=0.5)

        # ---------- Figur 2 – FFT‑Spektrum ----------
        fig2 = plt.figure(figsize=(7, 5))
        ax2 = fig2.add_subplot(111)

        ax2.semilogy(freq_arr, amp_arr, 'b')
        ax2.axvline(omega_hat, color='r', ls='--',
                    label=r"$\omega={:.3f}$".format(omega_hat))
        ax2.set_xlabel(r"Frequenz $f$ (rad / ln θ)")
        ax2.set_ylabel("Amplitude")
        ax2.set_title("FFT‑Spektrum von Δ_phase")
        ax2.set_xlim(0, max(8, 1.2 * omega_hat))
        ax2.legend()
        ax2.grid(True, ls='--', alpha=0.5)

        plt.show()

    # -------------------------------------------------
    # Ergebnis‑Dictionary
    # -------------------------------------------------
    result = {
        "a": a, "b": b,                     # Power‑Law‑Parameter
        "omega_hat": omega_hat,
        "omega_std": omega_std,
        "D_hat": D_hat, "D_err": perr[0],
        "phi_hat": phi_hat, "phi_err": perr[1],
        "c0_hat": c0_hat, "c0_err": perr[2],
        "c1_hat": c1_hat, "c1_err": perr[3],
        "bootstrap_omegas": boot_omegas,
        "r_base": r_base,
        "delta_phase": delta_phase,
    }
    return result


# -------------------------------------------------------------------------
# Beispiel‑Aufruf (synthetische Testdaten)
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # -------------------------------------------------
    # 1. Synthetische Daten erzeugen (wie im Original‑Beispiel)
    # -------------------------------------------------
    a_true     = 1.0 / np.pi          # ≈ 0.318309886
    D_true     = 0.40
    omega_true = 3.302
    phi_true   = 0.20

    N      = 2500
    theta  = np.logspace(-2, 2, N)                # θ von 0.01 bis 100
    r_true = np.exp(a_true * theta) * (1.0 +
               D_true * np.cos(omega_true * np.log(theta) + phi_true))

    # Leichtes multiplicatives Rauschen (≈0.2 %)
    r_meas = r_true * np.exp(0.002 * np.random.standard_normal(N))

    # -------------------------------------------------
    # 2. Analyse starten
    # -------------------------------------------------
    res = analyse_logperiodic(theta, r_meas,
                              n_bootstrap=600,
                              plot=True)

    # -------------------------------------------------
    # 3. Ergebnis‑Printout (nur numerische Schätzungen)
    # -------------------------------------------------
    print("\n=== Analyse‑Ergebnis ===")
    print(f"a (Power‑Law)          = {res['a']:.8f}")
    print(f"ω̂ (FFT)                = {res['omega_hat']:.6f} ± {res['omega_std']:.6f}")
    print(f"D̂ (Amplitude)          = {res['D_hat']:.6f} ± {res['D_err']:.6f}")
    print(f"φ̂ (Phase)              = {res['phi_hat']:.6f} ± {res['phi_err']:.6f}")
    print(f"c0 (Offset)             = {res['c0_hat']:.3e} ± {res['c0_err']:.3e}")
    print(f"c1 (Linear‑Trend)       = {res['c1_hat']:.3e} ± {res['c1_err']:.3e}")
