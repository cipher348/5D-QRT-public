#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Log‑periodische Invarianzprüfung von π
================================================
Wir vergleichen die modulierte Radiusfunktion

    r(θ) = exp(a·θ) · [ 1 + D·cos( ω·lnθ ) ]

mit einer glatten Basis ohne Modulation

    r_base(θ) = exp(a·θ)

und untersuchen das logarithmische Phasen‑Signal

    Δ_phase(θ) = log[ r(θ) / r_base(θ) ]
               = log[ 1 + D·cos( ω·lnθ ) ].

Für D≪1 gilt Δ_phase ≈ D·cos( ω·lnθ ), d.h. ein klarer
Peak bei der Frequenz ω im FFT‑Spektrum von Δ_phase
gegenüber lnθ.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

# ------------------------------------------------------------
# 1. Parameter
# ------------------------------------------------------------
a     = 1.0/np.pi                # ≈ 0.318309886  (normierte Wachstumsrate)
omega = 3.302                     # Resonanzparameter
D     = np.exp(-0.91822)         # Dämpfungsfaktor ≈ 0.399

# ------------------------------------------------------------
# 2. θ‑Stichprobe (log‑gleichmäßig, weil wir später FFT in lnθ machen)
# ------------------------------------------------------------
# Wir wählen 2000 Punkte zwischen θ_min und θ_max.
theta_min = 1e-2
theta_max = 1e2
N         = 2000
theta = np.logspace(np.log10(theta_min), np.log10(theta_max), N)

# ------------------------------------------------------------
# 3. Funktionen
# ------------------------------------------------------------
def r_mod(theta):
    """Modulierte Radiusfunktion."""
    return np.exp(a*theta) * (1.0 + D*np.cos(omega*np.log(theta)))

def r_base(theta):
    """Glatte Basis ohne Modulation."""
    return np.exp(a*theta)

def delta_phase(theta):
    """log‑phase‑Signal Δ_phase(θ) = log[r(θ)/r_base(θ)]."""
    # Direkt aus der Definition: log[1 + D·cos(ω·lnθ)]
    return np.log(1.0 + D*np.cos(omega*np.log(theta)))

# ------------------------------------------------------------
# 4. Berechnungen
# ------------------------------------------------------------
r_mod_vals   = r_mod(theta)
r_base_vals  = r_base(theta)
U_vals       = 2*np.pi * r_mod_vals               # Umfang
delta_vals   = delta_phase(theta)                 # Δ_phase(θ)

# ------------------------------------------------------------
# 5. Plot: r_mod vs. r_base (log‑log) und Δ_phase
# ------------------------------------------------------------
plt.figure(figsize=(12, 5))

# 5a – Log‑log‑Plot von r_mod und r_base
plt.subplot(1, 2, 1)
plt.loglog(theta, r_mod_vals,  label='r_mod (mit Modulation)', lw=2)
plt.loglog(theta, r_base_vals, '--', label='r_base (glatt)', lw=2)
plt.xlabel(r'$\theta$')
plt.ylabel(r'$r(\theta)$')
plt.title('Radiusfunktionen (Log‑Log)')
plt.legend()
plt.grid(True, which='both', ls='--', alpha=0.6)

# 5b – Δ_phase vs. lnθ
plt.subplot(1, 2, 2)
ln_theta = np.log(theta)
plt.plot(ln_theta, delta_vals, 'k', lw=1.5)
plt.xlabel(r'$\ln\theta$')
plt.ylabel(r'$\Delta_{\mathrm{phase}}(\theta)$')
plt.title(r'$\Delta_{\mathrm{phase}}(\theta)=\log[1+D\cos(\omega\ln\theta)]$')
plt.grid(True, which='both', ls='--', alpha=0.6)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 6. FFT‑Analyse von Δ_phase(lnθ)
# ------------------------------------------------------------
# Wir verwenden die reelle FFT (rfft) – das Signal ist reell.
dt = ln_theta[1] - ln_theta[0]          # konstantes Abstand in lnθ
freqs = rfftfreq(N, d=dt)               # Frequenzen in (rad/lnθ)

fft_vals = rfft(delta_vals)
amp_spectrum = np.abs(fft_vals) / N     # normierte Amplitude

# Plot des Amplitudenspektrums
plt.figure(figsize=(8, 4))
plt.semilogy(freqs, amp_spectrum, 'b')
plt.axvline(omega, color='r', linestyle='--',
            label=r'$\omega_{\text{theo}} = %.3f$' % omega)
plt.xlabel(r'Frequenz $f$ (rad / ln θ)')
plt.ylabel('Amplitude (normiert)')
plt.title(r'FFT‑Spektrum von $\Delta_{\mathrm{phase}}(\theta)$')
plt.legend()
plt.grid(True, which='both', ls='--', alpha=0.6)
plt.xlim(0, 8)          # genug Platz, um den Peak bei ω zu sehen
plt.show()

# ------------------------------------------------------------
# 7. Numerische Prüfung der Approximation Δ ≈ D·cos(ω·lnθ)
# ------------------------------------------------------------
approx = D * np.cos(omega * np.log(theta))
error  = delta_vals - approx

print(f"Maximaler absolute Fehler zwischen exaktem Δ und linearer Approx.: {np.max(np.abs(error)):.3e}")
print(f"Mittlerer quadratischer Fehler (MSE): {np.mean(error**2):.3e}")

# Optional: Plot des Fehlers
plt.figure(figsize=(6, 4))
plt.plot(ln_theta, error, 'm')
plt.xlabel(r'$\ln\theta$')
plt.ylabel('Fehler Δ - D·cos(ω·lnθ)')
plt.title('Abweichung der linearen Approximation')
plt.grid(True, which='both', ls='--', alpha=0.6)
plt.show()
