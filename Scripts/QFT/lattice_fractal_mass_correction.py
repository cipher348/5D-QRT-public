import numpy as np

# --------------------------------------------------------------
# 1. Lattice‑Parameter
# --------------------------------------------------------------
N   = 64                 # points per direction
a   = 0.1                # lattice spacing  (sets UV cutoff Λ = π/a)
L   = N * a
pi  = np.pi

# momentum components (periodic BC)
n = np.arange(-N//2, N//2)
kx, ky, kz, kt = np.meshgrid(n, n, n, n, indexing='ij')
kvec = (2*pi/(N*a)) * np.stack([kx, ky, kz, kt], axis=-1)   # shape (N,N,N,N,4)
kmag = np.linalg.norm(kvec, axis=-1)                       # |k|

# --------------------------------------------------------------
# 2. Physical parameters
# --------------------------------------------------------------
e_sq   = 0.0924          # α = e^2/(4π) ≈ 1/137 → e^2 ≈ 4π/137
e_sq   = 4*np.pi/137.0
m      = 0.511           # MeV (electron mass) – you can set m=0 for massless test
gamma  = 0.91822
D      = np.exp(-gamma)  # ≈ 0.399
phi    = 3.302           # ϙ

# --------------------------------------------------------------
# 3. Modified integrand (eq. 2)
# --------------------------------------------------------------
# Avoid k=0 because ln(k) diverges – set a tiny floor
kmag_safe = np.where(kmag==0, 1e-12, kmag)

mod_factor = 1.0 + D * np.sin(phi * np.log(kmag_safe))
integrand  = (kmag_safe**3) / (kmag_safe**2 + m**2) * mod_factor

# --------------------------------------------------------------
# 4. Numerical integration (Riemann sum)
# --------------------------------------------------------------
prefactor = e_sq / (8 * np.pi**2) * ( (2*np.pi/(N*a))**4 )   # d^4k → (2π/Na)^4
delta_m   = prefactor * integrand.sum()

print(f"Δm (lattice, N={N}) = {delta_m:.6e}  (MeV units if m in MeV)")

# --------------------------------------------------------------
# 5. Analytischer Grundterm (ohne log‑Modulation)
# --------------------------------------------------------------
Lambda = pi / a
bare = (e_sq/(8*np.pi**2)) * 0.5 * (Lambda**2 -
                                    m**2 * np.log(1 + Lambda**2/m**2))

print(f"Analytischer Grundterm Δm_bare = {bare:.6e}  (MeV)")
print(f"Fraktale Korrektur Δm_fract = Δm - Δm_bare = {(delta_m - bare):.6e}")
