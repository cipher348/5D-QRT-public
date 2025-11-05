#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# --------------------------------------------------------------
# 1. Grundparameter
# --------------------------------------------------------------
a      = 1.0                     # Voxel‑Kantenlänge (Einheit)
Nx, Ny = 151, 151               # Gittergröße (2‑D‑Slice)
K      = 3.0                     # Kopplungs­konstante (stark)
gamma  = 0.91822                # Dämpfungs‑Parameter
qn     = 320.0 / 99.0           # interne Skalen‑Frequenz
beta   = 2.0                     # Fraktal‑Exponent (Power‑Law)
steps  = 400                     # Zeitschritte
safety = 0.8                     # Sicherheitsfaktor für CFL

# --------------------------------------------------------------
# 2. CFL‑konformer Zeitschritt (27‑Nachbarn‑Stencil)
# --------------------------------------------------------------
# Für ein 27‑Nachbarn‑Laplacian gilt ungefähr:
#   Δt_max ≈ a² / (6·K)
Δt = safety * a**2 / (6.0 * K)

# --------------------------------------------------------------
# 3. Fraktaler Driving‑Term (einmal erzeugen, dann zeitlich modulieren)
# --------------------------------------------------------------
def fractal_drive(Nx, Ny, a, beta, seed=42):
    """Power‑law‑Feld ~ k^{-beta}. Rückgabe: normiertes 2‑D‑Array."""
    rng = np.random.default_rng(seed)

    # k‑Gitter (FFT‑Konvention)
    kx = np.fft.fftfreq(Nx, d=a) * 2*np.pi
    ky = np.fft.fftfreq(Ny, d=a) * 2*np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    Kmag = np.sqrt(KX**2 + KY**2)

    eps = 1e-12
    amp = (Kmag + eps) ** (-beta/2.0)   # sqrt, weil Power‑Law für |F|²
    phase = rng.uniform(0, 2*np.pi, size=amp.shape)
    spec = amp * np.exp(1j * phase)

    field = np.fft.ifftn(spec).real
    field /= np.sqrt(np.mean(field**2))   # RMS‑Normierung
    return field

drive_static = fractal_drive(Nx, Ny, a, beta)

def driving_term(t):
    """Zeitlich moduliertes Driving‑Feld (hier cos‑Modulation)."""
    Ω = 5.0                     # niedrige Modulations‑Frequenz
    return np.cos(Ω * t) * drive_static

# --------------------------------------------------------------
# 4. Hilfsfunktionen
# --------------------------------------------------------------
def laplacian_27(u, a):
    """
    27‑Nachbarn‑Laplacian (periodische Randbedingungen) via FFT.
    Das 3×3×3‑Kernel wird auf die Feldgröße aufgepolstert.
    """
    # 3×3×3‑Kernel (alle Nachbarn = 1, Zentrum = -26)
    kernel = np.ones((3, 3, 3))
    kernel[1, 1, 1] = -26

    # Pad den Kernel auf die Größe von u (2‑D → erweitere um eine singuläre Z‑Achse)
    # Wir benutzen eine 3‑D‑FFT, also fügen wir eine Dummy‑Achse (size=1) hinzu.
    kernel_fft = np.fft.fftn(kernel, s=(Nx, Ny, 1))

    # 2‑D‑Feld ebenfalls in 3‑D‑Form (letzte Achse = 1)
    u_fft = np.fft.fftn(u[:, :, np.newaxis], s=(Nx, Ny, 1))

    conv = np.fft.ifftn(u_fft * kernel_fft).real
    # Ergebnis zurück auf 2‑D‑Form bringen
    return conv[:, :, 0] / a**2

def D_factor(t):
    """Log‑periodischer Dämpfungsterm."""
    return np.exp(-gamma) * np.log(1.0 + np.abs(np.cos(qn * t)))

def rhs(u, t):
    """Rechte Seite von (1) inkl. Driving‑Term."""
    return K * laplacian_27(u, a) - D_factor(t) * u + driving_term(t)

def rk4_step(u, t, dt):
    """Explizites RK4‑Update."""
    k1 = rhs(u, t)
    k2 = rhs(u + 0.5*dt*k1, t + 0.5*dt)
    k3 = rhs(u + 0.5*dt*k2, t + 0.5*dt)
    k4 = rhs(u + dt*k3,     t + dt)
    return u + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

# --------------------------------------------------------------
# 5. Front‑Messung (fester absoluter Schwellenwert)
# --------------------------------------------------------------
mid = (Nx//2, Ny//2)          # Zentrum des Impulses
abs_thresh = 0.05             # absoluter Schwellenwert (unabhängig von Dämpfung)

def front_distance(u):
    """Äußerste Manhattan‑Distanz, bei der |u| > abs_thresh."""
    mask = np.abs(u) > abs_thresh
    if not np.any(mask):
        return None
    y, x = np.where(mask)
    return np.max(np.abs(x-mid[0]) + np.abs(y-mid[1]))

# --------------------------------------------------------------
# 6. Initialisierung (kurzer Impuls in der Mitte)
# --------------------------------------------------------------
u = np.zeros((Nx, Ny))
u[mid] = 1.0                     # Dirac‑ähnlicher Startimpuls

# --------------------------------------------------------------
# 7. Zeitschleife
# --------------------------------------------------------------
times   = []
fronts  = []

t = 0.0
for n in range(steps):
    u = rk4_step(u, t, Δt)
    t += Δt

    dist = front_distance(u)
    # Falls das Feld unter dem Schwellenwert liegt, halte die letzte Distanz
    if dist is None:
        dist = fronts[-1] if fronts else 0.0
    fronts.append(dist)
    times.append(t)

# --------------------------------------------------------------
# 8. Glätten & Geschwindigkeit bestimmen
# --------------------------------------------------------------
fronts_smooth = savgol_filter(fronts, window_length=11, polyorder=3)
v_empirical   = np.gradient(fronts_smooth, Δt)   # Voxel pro Δt

# --------------------------------------------------------------
# 9. Plot: Distanz & Geschwindigkeit
# --------------------------------------------------------------
plt.figure(figsize=(10, 4))

# ---- Distanz‑Kurve -------------------------------------------------
plt.subplot(1, 2, 1)
plt.plot(times, fronts_smooth, label='Front‑Distanz')
plt.xlabel('Zeit $t$')
plt.ylabel('Distanz (Voxel)')
plt.title('Front‑Ausbreitung (fraktaler Drive)')
plt.grid(alpha=0.3)

# ---- Geschwindigkeit ------------------------------------------------
plt.subplot(1, 2, 2)
plt.plot(times, v_empirical, color='orange', label='empirische $v$')
v_cfl = a / Δt
plt.axhline(v_cfl, color='red', ls='--', lw=1,
            label=f'CFL‑Grenze $v_{{max}}={v_cfl:.2f}$')
plt.xlabel('Zeit $t$')
plt.ylabel('Geschwindigkeit (Voxel/Δt)')
plt.title('Geschwindigkeit über die Zeit')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()
