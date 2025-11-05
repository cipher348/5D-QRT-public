import numpy as np
from scipy.integrate import quad
import pywt
import matplotlib.pyplot as plt
#%matplotlib inline

# --------------------------------------------------------------
# Parameter
# --------------------------------------------------------------
e_sq   = 4*np.pi/137.0
m      = 0.511          # MeV
gamma  = 0.91822
D      = np.exp(-gamma)   # ≈ 0.399
phi    = 3.302            # ϙ

# --------------------------------------------------------------
# Analytischer Grundterm (ohne Log‑Modulation)
# --------------------------------------------------------------
a      = 0.1               # Gitterabstand
Lambda = np.pi / a
bare   = (e_sq/(8*np.pi**2)) * 0.5 * (Lambda**2 -
                                      m**2*np.log(1+Lambda**2/m**2))
print(f"Δm_bare (analytisch) = {bare:.6e} MeV")

# --------------------------------------------------------------
# Direktes Integral des log‑periodischen Terms (B)
# --------------------------------------------------------------
def integrand_log(u):
    k = np.exp(u)
    return (k**3)/(k**2 + m**2) * np.sin(phi*u)

upper = np.log(Lambda)      # obere Grenze = ln(Λ)
result, err = quad(integrand_log, -30, upper, limit=200)
delta_m_fract = (e_sq/(8*np.pi**2)) * D * result
print(f"Δm_fract (direktes Integral) = {delta_m_fract:.6e} MeV")

# --------------------------------------------------------------
# Wavelet‑Analyse des reinen log‑periodischen Spectra
# --------------------------------------------------------------
# Wir erzeugen ein 1‑D‑Spektrum E(k) = 0.5*ω_k*[1+D*sin(phi*ln k)]
k_vals = np.logspace(-3, np.log10(Lambda), 2000)
omega  = k_vals                     # m≈0 → ω≈k
E_mod  = 0.5 * omega * (1.0 + D*np.sin(phi*np.log(k_vals)))

# CWT (Morlet) im Log‑k‑Raum
logk   = np.log(k_vals)
scales = np.arange(1, 300)
coeffs, _ = pywt.cwt(E_mod, scales, 'cmor1.5-1.0',
                     sampling_period=logk[1]-logk[0])

plt.figure(figsize=(8,4))
plt.imshow(np.abs(coeffs), extent=[logk.min(), logk.max(),
                                   scales.min(), scales.max()],
           cmap='viridis', aspect='auto', origin='lower')
plt.colorbar(label='|CWT|')
plt.xlabel(r'$\ln(k)$')
plt.ylabel('Scale')
plt.title('CWT – Log‑periodisches Muster (nur fraktaler Teil)')
plt.show()
