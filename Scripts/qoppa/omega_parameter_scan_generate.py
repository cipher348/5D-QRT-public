#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minimal Phase‑II scan (D₂ → D₃)

For each ω in a user defined interval we compute

    f_D2(t) = sin( ω·ln t )
    r(t)    = ∫₀ᵗ d/dt' ln|f_D2(t')| · e^{-γ t'} dt'
    Δr(t)   = r(t) – π·φ·e^{-γ t}

Three overlay plots are produced (f_D2, r + r_ideal, Δr) and a CSV
report with the maximal absolute error per ω is written.
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import viridis
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from math import pi, sqrt

# ----------------------------------------------------------------------
# 1. Constants
# ----------------------------------------------------------------------
PHI = (1 + sqrt(5)) / 2          # golden ratio


# ----------------------------------------------------------------------
# 2. Core model functions
# ----------------------------------------------------------------------
def f_D2(t: np.ndarray, omega: float) -> np.ndarray:
    """Spiral field: sin( ω·ln t ),   t > 0."""
    eps = 1e-12
    t_safe = np.maximum(t, eps)    # avoid log(0)
    return np.sin(omega * np.log(t_safe))


def dlog_f_D2_dt(t: np.ndarray, omega: float) -> np.ndarray:
    """
    Analytic derivative of ln|f_D2| :

        d/dt ln|sin(ω·ln t)| = (ω/t)·cot(ω·ln t)

    Where sin(...)≈0 we set the contribution to 0 to avoid singularities.
    """
    eps = 1e-12
    t = np.maximum(t, eps)
    arg = omega * np.log(t)                # ω·ln t
    sin_arg = np.sin(arg)

    safe = np.abs(sin_arg) > 1e-12        # safe division
    deriv = np.zeros_like(t)

    deriv[safe] = (omega / t[safe]) * (np.cos(arg[safe]) / sin_arg[safe])
    return deriv


def r_continuous(t: np.ndarray, omega: float, gamma: float) -> np.ndarray:
    """Continuous integral r(t) = ∫ dlog_f_D2_dt · e^{-γt} dt."""
    dt = t[1] - t[0]                       # uniform step (t starts at dt)
    integrand = dlog_f_D2_dt(t, omega) * np.exp(-gamma * t)
    return np.cumsum(integrand) * dt       # simple rectangle rule


def r_ideal(t: np.ndarray, gamma: float) -> np.ndarray:
    """Ideal reference: π·φ·exp(-γ·t)."""
    return pi * PHI * np.exp(-gamma * t)


# ----------------------------------------------------------------------
# 3. Scan routine (core of the script)
# ----------------------------------------------------------------------
def scan_omega(
    omega_min: float = 3.300,
    omega_max: float = 3.302,
    n_omega: int = 200,
    t_max: float = 200.0,
    dt: float = 0.01,
    gamma_func=lambda w: 1.0 / (w ** 2)   # user‑editable damping law
):
    """
    Perform the scan, create three overlay plots and write a CSV report.
    Returns the DataFrame‑like list of rows for possible further use.
    """
    # ----- time grid (starts at dt so that ln(t) is defined) -----
    t = np.arange(dt, t_max + dt, dt)

    # ----- colour map for the ω‑dimension -----
    cmap = viridis
    norm = Normalize(vmin=omega_min, vmax=omega_max)
    sm = ScalarMappable(norm=norm, cmap=cmap)

    # ----- prepare figures -----
    fig_f, ax_f = plt.subplots(figsize=(9, 2.5))
    fig_r, ax_r = plt.subplots(figsize=(9, 2.5))
    fig_d, ax_d = plt.subplots(figsize=(9, 2.5))

    # ----- CSV header -----
    csv_rows = [["omega", "gamma", "max_abs_delta_r"]]

    # ----- main ω‑loop -----
    for omega in np.linspace(omega_min, omega_max, n_omega):
        gamma = gamma_func(omega)

        # compute signals
        r = r_continuous(t, omega, gamma)
        r_ref = r_ideal(t, gamma)
        delta_r = r - r_ref

        # colour for this ω
        col = sm.to_rgba(omega)

        # ---- plot f_D2 (only once – the last ω gives a representative curve) ----
        if omega == np.linspace(omega_min, omega_max, n_omega)[-1]:
            ax_f.plot(t, f_D2(t, omega), color=col, label=f"ω={omega:.5f}")

        # ---- plot r(t) and its ideal counterpart ----
        ax_r.plot(t, r,      color=col, alpha=0.6)
        ax_r.plot(t, r_ref, '--',  color=col, alpha=0.4)

        # ---- plot Δr(t) ----
        ax_d.plot(t, delta_r, color=col, alpha=0.6)

        # ---- CSV entry: maximal absolute deviation ----
        max_err = np.max(np.abs(delta_r))
        csv_rows.append([omega, gamma, max_err])

    # ------------------------------------------------------------------
    # 4. Plot cosmetics & saving
    # ------------------------------------------------------------------
    # f_D2
    ax_f.set_title("Spiral field f_D2(t) (last ω shown)")
    ax_f.set_xlabel("t")
    ax_f.set_ylabel("f_D2(t)")
    ax_f.grid(True, ls=":")
    ax_f.legend(loc="upper right")
    fig_f.tight_layout()

    # r(t) + ideal
    ax_r.set_title(f"r(t) vs ideal (γ = γ(ω))  ω∈[{omega_min:.3f},{omega_max:.3f}]")
    ax_r.set_xlabel("t")
    ax_r.set_ylabel("r(t)")
    ax_r.grid(True, ls=":")
    cbar_r = fig_r.colorbar(sm, ax=ax_r, orientation="vertical")
    cbar_r.set_label("ω")
    fig_r.tight_layout()

    # Δr(t)
    ax_d.set_title(r"Deviation $\Delta r(t)=r(t)-\pi\varphi e^{-\gamma t}$")
    ax_d.set_xlabel("t")
    ax_d.set_ylabel(r"$\Delta r(t)$")
    ax_d.grid(True, ls=":")
    cbar_d = fig_d.colorbar(sm, ax=ax_d, orientation="vertical")
    cbar_d.set_label("ω")
    fig_d.tight_layout()

    # Show all three figures
    plt.show()

    # ------------------------------------------------------------------
    # 5. Write CSV report
    # ------------------------------------------------------------------
    csv_path = "scan_report.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)

    print(f"\nCSV report written to '{csv_path}' (columns: omega, gamma, max_abs_delta_r)")

    return csv_rows   # optional return for downstream use


# ----------------------------------------------------------------------
# 6. Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Feel free to change any of the arguments here
    scan_omega(
        omega_min=3.300,
        omega_max=3.302,
        n_omega=200,          # number of ω samples
        t_max=200.0,
        dt=0.01,
        gamma_func=lambda w: 1.0 / (w ** 2)   # you can replace with np.log(w)/np.pi, etc.
    )
