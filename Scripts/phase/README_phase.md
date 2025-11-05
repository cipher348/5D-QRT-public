# Spiral Phase & Lattice Wave Simulation: Analysis and Limits

This folder contains a concise, validated workflow exploring the analytic background, numerical simulation, and rigorous speed/stability limits of signal propagation on discrete lattices governed by spiral/phase resonance and log-periodic damping.

---

## Included Scripts & Their Purpose

### 1. `spiral_phase_factor_analysis.py`
> Symbolic and numerical computation of spiral/phase resonance factors (K, S, etc.), and visualization of corresponding logarithmic spirals—clarifies the geometric and algebraic structures governing wave phenomena in the discrete models.

### 2. `logperiodic_damping_response_analysis.py`
> Analyzes a damped, log-periodic ("fractal resonance") response function central to the propagation models. Provides time- and frequency-domain plots for various damping parameters, highlighting spectral structure and periodicity.

### 3. `lattice_wavefront_propagation_simulation.py`
> Simulates front propagation on a 2D lattice using a 27-neighbor Laplacian, robust RK4 integration, log-periodic damping, and a fractal spatial driving field. Tracks and visualizes empirical signal speed, directly relating numerics to theoretical limits.

### 4. `cfl_limit_speed_sweep_analysis.py`
> Systematically varies both the coupling (K) and the time-step (safety factor) in the wavefront simulation. Demonstrates, empirically and unambiguously, that the propagation speed is limited strictly by the CFL numeric stability bound, independent of physical coupling/frequency.

---

## Scientific Narrative

1. **Theoretical Foundation:**  
   Begin with phase and resonance constants and their geometric context (`spiral_phase_factor_analysis.py`).

2. **Dynamic Response Structure:**  
   Understand the analytic form and spectra of the damping/response function (`logperiodic_damping_response_analysis.py`).

3. **Realistic Simulation:**  
   Simulate, visualize, and measure front propagation with all relevant geometry and damping (`lattice_wavefront_propagation_simulation.py`).

4. **Empirical Limits:**  
   Validate and sweep the actual numerical speed limit, explaining theoretical and practical constraints (`cfl_limit_speed_sweep_analysis.py`).

---

## Requirements

- **Python ≥ 3.8**
- Libraries: `numpy`, `matplotlib`, `scipy`, `sympy`
  ```bash
  pip install numpy matplotlib scipy sympy
  ```

---

## Citation

If using this code in published work, please cite or link to [cipher348/5D-QRT](https://github.com/cipher348/5D-QRT).

**Authors & Maintainers:** [cipher348](https://github.com/cipher348)

### Scripts created using Lumo AI (Proton)
