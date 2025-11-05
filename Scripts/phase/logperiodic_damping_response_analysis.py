import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------
# 1. Parameter
# --------------------------------------------------------------
gamma_vals = np.array([0.91818, 0.91822, 0.91827])   # die korrekten Dämpfungs‑Werte
qn        = 320.0 / 99.0                           # 3.232323…
ln2       = np.log(2.0)

# Zeitvektor – mehrere Perioden (z. B. 0 … 10·π/qn)
t_max = 10 * np.pi / qn
t = np.linspace(0, t_max, 6000)   # feine Auflösung

# --------------------------------------------------------------
# 2. Funktionsdefinition
# --------------------------------------------------------------
def F(gamma, t):
    """Dämpfungstriple‑Funktion mit den korrigierten γ‑Werten."""
    return np.exp(-gamma) * np.log(1.0 + np.abs(np.cos(qn * t)))

# --------------------------------------------------------------
# 3. Plot: Zeit‑Domain (drei γ‑Kurven)
# --------------------------------------------------------------
plt.figure(figsize=(10, 6))
for g in gamma_vals:
    plt.plot(t, F(g, t), label=f'γ = {g:.5f}')

# Titel – raw‑String, nur Matplotlib‑kompatible Befehle
plt.title(
    r'$F(\gamma,q_n,t)=e^{-\gamma}\,\ln\!\left(1+|\cos(q_n\,t)|\right)$'
)
plt.xlabel('Zeit $t$ (arb. Einheiten)')
plt.ylabel('Amplitude $F$')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()          # funktioniert jetzt, weil der Titel parsbar ist
plt.show()

# --------------------------------------------------------------
# 4. Log‑log‑Darstellung (zeigt die Log‑Periodizität)
# --------------------------------------------------------------
plt.figure(figsize=(10, 6))
for g in gamma_vals:
    plt.loglog(t[1:], F(g, t[1:]), label=f'γ = {g:.5f}')   # t>0 nötig für log‑log
plt.title(
    r'Log‑log‑Plot: $F(\gamma,q_n,t)$ (log‑periodisches Muster)'
)
plt.xlabel('log$_{10}$(t)')
plt.ylabel('log$_{10}$(F)')
plt.legend()
plt.grid(which='both', alpha=0.3)
plt.tight_layout()
plt.show()

# --------------------------------------------------------------
# 5. Fourier‑Spektrum (zeigt harmonische Peaks)
# --------------------------------------------------------------
from numpy.fft import rfft, rfftfreq

Fs = 1.0 / (t[1] - t[0])          # Abtastrate
freq = rfftfreq(len(t), d=1/Fs)   # Frequenzachse

plt.figure(figsize=(10, 5))
for g in gamma_vals:
    spectrum = np.abs(rfft(F(g, t)))
    plt.semilogy(freq, spectrum, label=f'γ = {g:.5f}')
plt.xlim(0, 5*qn)                # bis zu 5‑fache Grundfrequenz
plt.title(
    r'Fourier‑Spektrum von $F(\gamma,q_n,t)$'
)
plt.xlabel('Frequenz (Einheiten $q$)')
plt.ylabel('|FFT| (log‑scale)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
