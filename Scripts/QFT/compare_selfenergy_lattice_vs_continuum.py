#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
selfenergy_fixed.py

Fully consistent comparison between lattice and continuum for the
log‑periodic (fractal) self‑energy term.

Key fixes:
  * UV cutoff Λ = π/a applied as a spherical mask on the lattice.
  * Correct measure factor for a Cartesian 4‑D sum: (2π)^‑4.
  * Analytic bare term uses the same Λ.
  * Only the D·sin(φ·ln k) part is summed on the lattice.
  * Optional debug output shows max k‑values and mask fraction.
"""

import sys, math, argparse
import numpy as np
from scipy.integrate import quad

# ----------------------------------------------------------------------
# 1.  Argument parsing (so you can change parameters from the console)
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Consistent lattice ↔ continuum self‑energy")
parser.add_argument("-N", type=int, default=32,
                    help="Lattice points per direction (default: %(default)s)")
parser.add_argument("-a", type=float, default=0.10,
                    help="Lattice spacing a (UV cutoff Λ = π/a) (default: %(default)s)")
parser.add_argument("-m", type=float, default=0.511,
                    help="Particle mass m in MeV (default: %(default)s)")
parser.add_argument("--gamma", type=float, default=0.91822,
                    help="Damping exponent γ (default: %(default)s)")
parser.add_argument("--phi", type=float, default=3.302,
                    help="Spiral‑time constant ϙ (default: %(default)s)")
parser.add_argument("--alpha", type=float, default=1.0/137.0,
                    help="Fine‑structure constant α (default: 1/137)")
parser.add_argument("--debug", action="store_true",
                    help="Print additional diagnostic information")
args = parser.parse_args()

# ----------------------------------------------------------------------
# 2.  Derived constants
# ----------------------------------------------------------------------
D      = math.exp(-args.gamma)          # D = e^(‑γ) ≈ 0.399
e_sq   = 4.0 * math.pi * args.alpha    # e² = 4π α
Lambda = math.pi / args.a               # UV cutoff (identisch für Gitter & Kontinuum)

# ----------------------------------------------------------------------
# 3.  Analytic bare term (with the SAME UV cutoff)
# ----------------------------------------------------------------------
def analytic_bare(e_sq, m, a):
    """Quadratically divergent piece with Λ = π/a."""
    Lambda = math.pi / a
    term   = 0.5 * (Lambda**2 -
                    m**2 * math.log(1.0 + Lambda**2 / m**2))
    return (e_sq / (8.0 * math.pi**2)) * term   # matches radial integral

# ----------------------------------------------------------------------
# 4.  Lattice implementation (4‑D hyper‑cube → spherical mask)
# ----------------------------------------------------------------------
def lattice_fractal(N, a, m, D, phi, e_sq, debug=False):
    """
    Build a 4‑D momentum lattice, apply a spherical UV‑cutoff |k| ≤ Λ,
    remove the k=0 mode, and sum ONLY the D·sin(φ·ln k) contribution.
    Returns the lattice approximation of the *fractal* self‑energy.
    """
    # ----- momentum grid -------------------------------------------------
    n = np.arange(-N//2, N//2, dtype=np.float64)      # -N/2 … N/2-1
    kx, ky, kz, kt = np.meshgrid(n, n, n, n, indexing='ij')
    factor = (2.0 * math.pi) / (N * a)                # Δk in each direction

    # physical momenta
    kx = kx * factor
    ky = ky * factor
    kz = kz * factor
    kt = kt * factor

    # magnitude |k|
    kmag = np.sqrt(kx**2 + ky**2 + kz**2 + kt**2)

    # ----- spherical mask |k| ≤ Λ ----------------------------------------
    mask = kmag <= Lambda

    # ----- eliminate the exact zero mode (k = 0) -------------------------
    kmag_safe = np.where(kmag == 0.0, np.nan, kmag)   # NaN for k=0

    # ----- log‑periodic modulation (only the D·sin part) ---------------
    mod = D * np.sin(phi * np.log(kmag_safe))

    # ----- integrand ----------------------------------------------------
    integrand = (kmag_safe**3) / (kmag_safe**2 + m**2) * mod

    # replace NaN (from k=0) by 0 – the whole contribution should be zero.
    integrand = np.where(np.isnan(integrand), 0.0, integrand)

    # apply spherical mask
    integrand_masked = np.where(mask, integrand, 0.0)

    # ----- correct measure factor for a Cartesian 4‑D sum ---------------
    prefactor = (e_sq / (2.0 * math.pi)**4) * (factor**4)

    result = prefactor * integrand_masked.sum()

    if debug:
        k_comp_max = factor * (N // 2)               # max component in one direction
        k_radial_max = math.sqrt(4) * k_comp_max     # corner of hyper‑cube
        print(f"[DEBUG] UV cutoff Λ (target) = {Lambda:.6f}")
        print(f"[DEBUG] Max cartesian component = {k_comp_max:.6f}")
        print(f"[DEBUG] Radial max (corner)   = {k_radial_max:.6f}")
        print(f"[DEBUG] Fraction of lattice points kept by spherical mask = "
              f"{mask.sum() / mask.size:.6f}")

    return result

# ----------------------------------------------------------------------
# 5.  Continuum quadrature of the *pure* fractal term
# ----------------------------------------------------------------------
def quad_fractal(m, D, phi, e_sq, a):
    """
    Direct numerical integration of
        (k³)/(k²+m²) * D·sin(φ·ln k)
    from k=0 up to the same UV cutoff Λ = π/a.
    Returns (value, estimated_error).
    """
    Lambda = math.pi / a
    upper  = math.log(Lambda)          # ln Λ

    def integrand_log(u):
        """u = ln k  →  k = e^u."""
        k = math.exp(u)
        return (k**3) / (k**2 + m**2) * D * np.sin(phi * u)

    val, err = quad(integrand_log,
                    -30.0,               # effectively 0
                    upper,
                    epsabs=1e-12, epsrel=1e-12, limit=200)

    prefactor = (e_sq / (2.0 * math.pi)**4)   # same measure as lattice
    return prefactor * val, err

# ----------------------------------------------------------------------
# 6.  Main driver – compute, compare, and print diagnostics
# ----------------------------------------------------------------------
def main():
    # ---- 1. Analytic bare term (for completeness) --------------------
    delta_m_bare = analytic_bare(e_sq, args.m, args.a)
    print(f"Δm_bare (analytic, with UV cutoff Λ=π/a) = {delta_m_bare:.6e} MeV")

    # ---- 2. Lattice fractal contribution -----------------------------
    delta_m_lat = lattice_fractal(args.N, args.a, args.m,
                                 D, args.phi, e_sq,
                                 debug=args.debug)
    print(f"Δm_fract (lattice, spherical mask, D·sin only) = {delta_m_lat:.6e} MeV")

    # ---- 3. Continuum quadrature (same term) -------------------------
    delta_m_quad, err = quad_fractal(args.m, D, args.phi, e_sq, args.a)
    print(f"Δm_fract (continuum quad) = {delta_m_quad:.6e} MeV  (error ≈ {err:.2e})")

    # ---- 4. Consistency check ----------------------------------------
    diff = delta_m_lat - delta_m_quad
    print("\n=== CONSISTENCY CHECK ===")
    print(f"lattice – quad = {diff:.6e} MeV")
    if abs(diff) < 1e-5:
        print("✅  Agreement within numerical noise.")
    else:
        print("⚠️  Visible discrepancy – check resolution / mask.")

    # ---- 5. Quick numbers (UV‑cutoff, period, damping) ---------------
    print("\n--- QUICK NUMBERS ---")
    print(f"UV cutoff Λ = π/a = {Lambda:.6f} MeV")
    print(f"Log‑period length Δ(ln k) = 2π/ϙ = {2*math.pi/args.phi:.6f}")
    print(f"Damping amplitude D = e^(-γ) = {D:.6f}")

if __name__ == "__main__":
    # Prevent loading stale .pyc files (just in case)
    sys.dont_write_bytecode = True
    main()
