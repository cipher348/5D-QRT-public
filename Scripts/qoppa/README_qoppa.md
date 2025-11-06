# D2→D3 Spiral-Phase Transition Analysis Suite

This folder collects a **scientifically robust pipeline** for simulating, scanning, and analyzing the "D2 → D3" transition using spiral time/phase-field models. All filenames are self-explaining and the scripts are intended to be used sequentially or independently for the study of critical resonances and transient features in the system.

---

## Scripts Included

### 1. `D2D3_spiral_phase_model.py`
> Generates and visualizes the core spiral-phase dynamical model, including analytical, numerical, and discrete (continued fraction) solutions. Serves as a reference foundation to understand the D2 → D3 transition.

### 2. `omega_parameter_scan_generate.py`
> Scans a user-defined frequency (ω) range, simulates the full model for each ω, and outputs data and plots summarizing the deviation from the ideal response. Generates CSV files for downstream feature extraction or statistical analysis.

### 3. `extract_peak_features_from_scan.py`
> Processes the output of the parameter scan, robustly identifies the main |Δr| resonance peak, extracts its location, amplitude, width (FWHM), and area, and exports results in human- and machine-readable formats. Handles outliers and ensures subgrid accuracy.

### 4. `scan_report_statistics_quantiles.py`
> Computes robust statistics (quantiles, median, MAD, MAE) on the scan outputs for quality assessment, convergence checks, and statistical reporting. Provides plots and tabulated summaries.

---

## Recommended Pipeline

1. **Model grounding:**  
   Run `D2D3_spiral_phase_model.py` to visualize and explore the fundamental spiral-phase phenomenon and validate expected behaviors.

2. **Parameter scan:**  
   Use `omega_parameter_scan_generate.py` to systematically probe the ω parameter space. This creates scan report CSV(s) and visualizations.

3. **Peak feature extraction:**  
   Apply `extract_peak_features_from_scan.py` to the scan report(s) to obtain precise peak/resonance metrics for your system.

4. **Statistical postprocessing:**  
   Run `scan_report_statistics_quantiles.py` for global scan statistics, histograms, and quantile analysis.

---

## Requirements

- Python 3.8+
- Standard scientific packages: `numpy`, `matplotlib`, `pandas`, `scipy`
- Optionally: `seaborn` for enhanced plots

Install dependencies with:
```bash
pip install numpy matplotlib pandas scipy seaborn
```

---

## Citation

If using this code in published work, please cite or link to [cipher348/5D-QRT](https://github.com/cipher348/5D-QRT).

**Authors & Maintainers:** [cipher348](https://github.com/cipher348) (Kai5)

### Scripts created using Lumo AI (Proton)
