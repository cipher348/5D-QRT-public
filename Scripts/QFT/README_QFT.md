# 5D-QRT // QFT

## Overview

This repository provides several scripts for analyzing log-periodic (`fractal`) corrections in quantum field theory (QFT), comparing lattice and continuum self-energy calculations, and demonstrating both analytic and numerical approaches. The scripts are well-commented and suitable for computational physics research, educational purposes, and reproducible science.

---

## Scripts Overview

### 1. `compare_selfenergy_lattice_vs_continuum.py`

**Description:**  
Performs a robust, side-by-side comparison of self-energy mass corrections arising in QFT, using both lattice-based and continuum-based numerical methods. Applies consistent UV cutoffs, parses arguments from the command line, and includes detailed diagnostic checks for numerical agreement.

**Features:**  
- Command-line parameterization for reproducible studies.
- Consistent lattice and continuum calculations.
- Diagnostic/debug output and consistency checks.

---

### 2. `analytical_and_wavelet_fractal_mass_correction.py`

**Description:**  
Combines analytical, direct integration, and wavelet-based analysis of fractal (log-periodic) corrections to self-energy. Provides visual output for understanding these corrections and bridges theoretical formulas and data visualization.

**Features:**  
- Analytic and numerical evaluation of fractal corrections.
- Visualization using matplotlib and wavelet transforms.
- Fast, instructive demonstration for new users.

---

### 3. `lattice_fractal_mass_correction.py`

**Description:**  
Focuses on the numerical approximation of self-energy corrections on a 4D lattice. Clearly explains grid setup, momentum discretization, and Riemann-sum integration for log-periodic (fractal) effects.

**Features:**  
- Straightforward 4D lattice setup.
- Direct computation of mass corrections.
- Useful for understanding numerical lattice field theory.

---

## Dependencies

All scripts require:
- `numpy`
- `scipy`
- `matplotlib`
- `pywt` (PyWavelets)

Install via pip:
```bash
pip install numpy scipy matplotlib pywt
```

---

## Getting Started

All scripts can be run independently. For parameterized scripts, use the `--help` option to see command-line arguments, e.g.:

```bash
python compare_selfenergy_lattice_vs_continuum.py --help
```

Visualizations (matplotlib plots) will display interactively; if running on a server, ensure X-forwarding or adapt for saving plots.

---

## Citation and License

Creative Commons Zero v1.0 Universal

---

## Contact

For questions or collaboration, open an issue or contact Kai5.

### Scripts created using Lumo AI (Proton)
