# Final Report

## 1. Abstract
A single-file PF-1 cardiovascular simulation was implemented for healthy dynamics and an atrial arrhythmia overlay. The deliverable includes one-page comparative visualization, pressure-flow (P-Q) analysis, and summary metrics using standardized variable names.

## 2. Disease Background (Atrial Arrhythmia)
Atrial arrhythmia in this work is represented as irregular atrial timing and reduced atrial activation amplitude, while preserving the baseline hemodynamic equations.

## 3. Methods (Baseline + Atrial Arrhythmia Overlay)
- Baseline: PF-1 equations from the PDF implemented in `cardio_onefile.py`.
- Overlay: atrial activation timing jitter and amplitude scaling only.
- Solver: `solve_ivp` with fixed tolerances for reproducibility.

## 4. Results (4-Panel Figure + P-Q + Metrics)
- Figure: `report/figures/comparison_onepage.pdf`.
- P-Q mapping:
  - Mitral: \(\Delta P =\) `left_atrium_pressure_cgs` \(-\) `left_ventricle_pressure_cgs`, \(Q=\) `mitral_flow_ml_per_s`
  - Tricuspid: \(\Delta P =\) `right_atrium_pressure_cgs` \(-\) `right_ventricle_pressure_cgs`, \(Q=\) `tricuspid_flow_ml_per_s`
- Figure signals:
  - Activity panel: `sig_activation_sine`, `sig_activation_square`
  - Stiffness panel: `left_ventricle_stiffness_cgs`
  - Pressure/volume panel: `left_ventricle_pressure_mmhg`, `arterial_compliance_pressure_mmhg`, `left_ventricle_volume_ml`
  - PV panel: `left_ventricle_pressure_mmhg` vs `left_ventricle_volume_ml`
- Quantitative outputs: `out/summary_metrics.csv`.
  - `met_mean_left_ventricle_pressure_mmhg_healthy`
  - `met_mean_left_ventricle_pressure_mmhg_af`
  - `met_mean_arterial_compliance_pressure_mmhg_healthy`
  - `met_mean_arterial_compliance_pressure_mmhg_af`
  - `met_stroke_volume_proxy_ml_healthy`
  - `met_stroke_volume_proxy_ml_af`
  - `met_beat_to_beat_variability_index_healthy`
  - `met_beat_to_beat_variability_index_af`

## 5. Limitations and Assumptions
The atrial arrhythmia behavior is an assumptions-based overlay and is not a re-derived arrhythmia physiology model. Assumptions are listed in `docs/assumptions.md`.

## 6. Conclusion
The required healthy-versus-atrial-arrhythmia comparison pipeline is reproducible from one command and exports figure, CSV time series, metrics, and metadata with standardized naming.

## 7. Appendix
See `report/appendix_traceability.md` and `docs/pdf_traceability.md`.
