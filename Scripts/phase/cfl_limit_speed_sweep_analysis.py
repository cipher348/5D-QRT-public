#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simulation: Versuch, die Front‑Geschwindigkeit über das CFL‑Maximum zu treiben.
Wir variieren die Kopplungs­konstante K (oder Δt) und messen die
Front‑Distanz. Das Ergebnis demonstriert, dass die maximale
Ausbreitungsgeschwindigkeit ausschließlich durch a/Δt bestimmt wird
und nicht von K (also nicht von der Frequenz) abhängt.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# --------------------------------------------------------------
# 1. Grundparameter (gemeinsam für alle Läufe)
# --------------------------------------------------------------
a      = 1.0                     # Voxel‑Kantenlänge (Einheit)
Nx, Ny = 151, 151               # 2‑D‑Gitter (ausreichend groß)
gamma  = 0.91822                # Dämpfungs‑Parameter (log‑periodisch)
qn     = 320.0 / 99.0           # interne Skalen‑Frequenz
beta   = 2.0                     # Fraktaler Exponent (Power‑Law)
steps  = 250                    # Zeitschritte pro Lauf
abs_thresh = 0.05               # fester Schwellenwert für Front‑Messung

# --------------------------------------------------------------
# 2. Hilfsfunktionen (identisch zum vorherigen Skript)
# --------------------------------------------------------------
def fractal_drive(Nx, Ny, a, beta, seed=42):
    rng = np.random.default_rng(seed)
    kx = np.fft.fftfreq(Nx, d=a) * 2*np.pi
    ky = np.fft.fftfreq(Ny, d=a) * 2*np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    Kmag = np.sqrt(KX**2 + KY**2)
    eps = 1e-12
    amp = (Kmag + eps) ** (-beta/2.0)
    phase = rng.uniform(0, 2*np.pi, size=amp.shape)
    spec = amp * np.exp(1j * phase)
    field = np.fft.ifftn(spec).real
    field /= np.sqrt(np.mean(field**2))
    return field

drive_static = fractal_drive(Nx, Ny, a, beta)

def driving_term(t):
    Ω = 5.0
    return np.cos(Ω * t) * drive_static

def D_factor(t):
    """log‑periodischer Dämpfungsterm."""
    return np.exp(-gamma) * np.log(1.0 + np.abs(np.cos(qn * t)))

def laplacian_27(u, a):
    """27‑Nachbarn‑Laplacian via FFT (2‑D‑Gitter, dummy Z‑Achse)."""
    kernel = np.ones((3, 3, 3))
    kernel[1, 1, 1] = -26
    kernel_fft = np.fft.fftn(kernel, s=(Nx, Ny, 1))
    u_fft = np.fft.fftn(u[:, :, np.newaxis], s=(Nx, Ny, 1))
    conv = np.fft.ifftn(u_fft * kernel_fft).real
    return conv[:, :, 0] / a**2

def rhs(u, t, K):
    """Rechte Seite der PDE (Kopplung + Dämpfung + Driving)."""
    return K * laplacian_27(u, a) - D_factor(t) * u + driving_term(t)

def rk4_step(u, t, dt, K):
    """Explizites RK4‑Update (K ist ein Parameter)."""
    k1 = rhs(u, t,               K)
    k2 = rhs(u + 0.5*dt*k1, t+0.5*dt, K)
    k3 = rhs(u + 0.5*dt*k2, t+0.5*dt, K)
    k4 = rhs(u + dt*k3,     t+dt,   K)
    return u + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

mid = (Nx//2, Ny//2)

def front_distance(u):
    """Äußerste Manhattan‑Distanz, bei der |u| > abs_thresh."""
    mask = np.abs(u) > abs_thresh
    if not np.any(mask):
        return None
    y, x = np.where(mask)
    return np.max(np.abs(x-mid[0]) + np.abs(y-mid[1]))

# --------------------------------------------------------------
# 3. Sweep‑Funktion – Variation von K (CFL‑konform)
# --------------------------------------------------------------
def sweep_K(K_values, safety=1.0):
    """
    Für jede Kopplungs­konstante K wird ein kurzer Lauf gemacht.
    safety = 1.0 → Δt = a²/(6*K) (exakte CFL‑Grenze).
    """
    v_empirical = []   # gemessene Geschwindigkeit
    v_cfl       = []   # theoretisches Maximum a/Δt

    for K in K_values:
        # CFL‑konformer Zeitschritt (Safety‑Faktor wird hier fest auf 1.0 gesetzt)
        dt = safety * a**2 / (6.0 * K)   # Δt = a²/(6K)  (CFL‑Grenze)

        # Theoretisches Maximum (unabhängig von K!)
        v_cfl.append(a / dt)

        # Initialisierung
        u = np.zeros((Nx, Ny))
        u[mid] = 1.0                     # Dirac‑Impuls
        t = 0.0
        front_hist = []

        # Zeitschleife
        for _ in range(steps):
            u = rk4_step(u, t, dt, K)
            t += dt
            d = front_distance(u)
            if d is None:
                d = front_hist[-1] if front_hist else 0.0
            front_hist.append(d)

        # Glätten und Geschwindigkeit bestimmen (letzte Ableitung)
        front_smooth = savgol_filter(front_hist, window_length=9, polyorder=3)
        v_emp = np.gradient(front_smooth, dt)[-1]
        v_empirical.append(v_emp)

    return v_empirical, v_cfl

# --------------------------------------------------------------
# 4. Sweep‑Durchläufe
# --------------------------------------------------------------

# 4.1  Variation von K (CFL‑konform, safety=1.0)
K_vals = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 8.0])   # verschiedene Kopplungen
v_emp_K, v_cfl_K = sweep_K(K_vals, safety=1.0)

# 4.2  Variation von Δt (indirekt über safety < 1)
# Wir halten K fest (z. B. K=3) und ändern den Safety‑Faktor.
def sweep_safety(safety_vals, K_fixed=3.0):
    v_emp = []
    v_cfl = []
    for sf in safety_vals:
        dt = sf * a**2 / (6.0 * K_fixed)   # Δt = sf·Δt_max
        v_cfl.append(a / dt)

        # Initialisierung
        u = np.zeros((Nx, Ny))
        u[mid] = 1.0
        t = 0.0
        front_hist = []

        for _ in range(steps):
            u = rk4_step(u, t, dt, K_fixed)
            t += dt
            d = front_distance(u)
            if d is None:
                d = front_hist[-1] if front_hist else 0.0
            front_hist.append(d)

        front_smooth = savgol_filter(front_hist, window_length=9, polyorder=3)
        v_emp.append(np.gradient(front_smooth, dt)[-1])
    return v_emp, v_cfl

safety_vals = np.linspace(0.4, 1.0, 7)   # von 0.4·Δt_max bis 1.0·Δt_max
v_emp_s, v_cfl_s = sweep_safety(safety_vals, K_fixed=3.0)

# --------------------------------------------------------------
# 5. Plot
# --------------------------------------------------------------
plt.figure(figsize=(10, 4))

# ---- 5.1  K‑Sweep (CFL‑konform) ----
plt.subplot(1, 2, 1)
plt.plot(K_vals, v_emp_K, 'o-', label='empirische $v$')
plt.plot(K_vals, v_cfl_K, '--', label=r'$v_{\mathrm{cfl}}=a/\Delta t$')
plt.xlabel('Kopplungs­konstante $K$')
plt.ylabel('Geschwindigkeit (Voxel/Δt)')
plt.title('Sweep über $K$ (Δt = a²/(6K) → CFL‑Grenze)')
plt.legend()
plt.grid(alpha=0.3)

# ---- 5.2  Safety‑Sweep (Δt‑Variation) ----
plt.subplot(1, 2, 2)
plt.plot(safety_vals, v_emp_s, 's-', label='empirische $v$')
plt.plot(safety_vals, v_cfl_s, '--', label=r'$v_{\mathrm{cfl}}=a/\Delta t$')
plt.xlabel('Safety‑Faktor (Δt / Δtₘₐₓ)')
plt.ylabel('Geschwindigkeit (Voxel/Δt)')
plt.title('Sweep über Δt (K fest, $K=3$)')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()
