# --------------------------------------------------------------
#  Spiralen mit Dämpfung und Pi‑Einfluss (ohne klassischen φ)
# --------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# -------------------- 1. Exakte Konstanten --------------------
pi   = sp.pi
phi  = (1 + sp.sqrt(5)) / 2                 # nur für R‑Berechnung nötig
q    = (3 + sp.sqrt(13)) / 2                # qoppa maximal
qn   = sp.Rational(320, 99)                 # qoppa neutral
Q    = sp.nsimplify('8.123252248375445')    # Q‑Oppräsenz (bleibt Float)
D    = sp.exp(-sp.nsimplify('0.91822'))     # Dämpfung

# -------------------- 2. Faktoren ---------------------------
c   = 4 + sp.sqrt(5) + sp.sqrt(13)          # gemeinsamer Zähler

Rpi = pi * phi * D                          # Raumphasenfaktor inkl. Pi
R   = phi * D                                # Raumphasenfaktor ohne Pi

Kpi = c / Rpi                               # Kopplungsfaktor inkl. Pi
K   = c / R                                 # Kopplungsfaktor ohne Pi
Kq  = (qn + sp.sqrt(Q)) / D                # Kopplungsfaktor mit qoppa

Spi = Kpi * pi                              # Spiralzeit‑Faktor inkl. Pi
S   = K                                      # Spiralzeit‑Faktor ohne Pi
Sq  = Kq                                     # Spiralzeit‑Faktor mit qoppa

# -------------------- 3. Numerische Werte -------------------
# Wir wandeln alles in normale Floats (15 Signifikante Stellen reichen)
S_val   = float(sp.N(S, 15))        # = K
Spi_val = float(sp.N(Spi, 15))      # = Kpi·π

# -------------------- 4. Winkel‑Inkremente -----------------
# Δθ = 2π / (Spiralzeit‑Faktor)
delta_theta_S   = 2 * np.pi / S_val      # ohne Pi‑Einfluss
delta_theta_Spi = 2 * np.pi / Spi_val    # mit Pi‑Einfluss

print(f"S (ohne π)   = {S_val:.12f}")
print(f"Sπ (mit π)   = {Spi_val:.12f}")
print(f"Δθ_S   = {np.rad2deg(delta_theta_S):.6f}°")
print(f"Δθ_Sπ  = {np.rad2deg(delta_theta_Spi):.6f}°")

# -------------------- 5. Spiralen erzeugen ----------------
def make_spiral(delta_theta, a=0.5, b=0.04, N=800):
    """
    Logarithmische Spirale:
        r(θ) = a * exp(b * θ)
    delta_theta : Winkel‑Inkrement pro Schritt (Radiant)
    a,b,N      : Standard‑Parameter (kann angepasst werden)
    """
    angles = np.cumsum(np.full(N, delta_theta))   # kumulative Winkel
    radii  = a * np.exp(b * angles)               # exponentiell wachsender Radius
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return x, y

# Spirale A – nur K (ohne π)
x_A, y_A = make_spiral(delta_theta_S)

# Spirale B – Kπ·π (mit π)
x_B, y_B = make_spiral(delta_theta_Spi)

# -------------------- 6. Plot ----------------------------
plt.figure(figsize=(8, 8))
plt.plot(x_A, y_A, label=r"$S = K$ (ohne π)",  color="#1f77b4", linewidth=2)
plt.plot(x_B, y_B, label=r"$S_{\pi}=K_{\pi}\,\pi$ (mit π)",
         color="#ff7f0e", linewidth=2, linestyle="--")

# Optional: Verbindungslinien zwischen korrespondierenden Punkten
idx = np.arange(0, len(x_A), 50)   # alle 50 Punkte verbinden
for i in idx:
    plt.plot([x_A[i], x_B[i]], [y_A[i], y_B[i]],
             color="gray", alpha=0.4, linewidth=0.8)

plt.title("Logarithmische Spiralen – Dämpfung & π‑Einfluss")
plt.axis("equal")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True, ls=":", alpha=0.5)
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()
