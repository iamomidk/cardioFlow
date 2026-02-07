# Python API Reference

Main module:
- `cardio_onefile.py`

## Data Classes
- `PF1Params`
  - Baseline PF-1 parameter container.
- `ODESettings`
  - Solver method/tolerances/max-step settings.
- `AtrialArrhythmiaConfig`
  - Arrhythmia physiology overlay parameters.

## Core Simulation Functions
- `simulate_healthy(params, tf, dt_out, solver_cfg=None)`
  - Runs baseline model.
- `simulate_af(params, tf, dt_out, seed=1, overlay_cfg=None, solver_cfg=None)`
  - Runs arrhythmia overlay.
- `simulate_compare(params, duration_s, dt_output_s, random_seed, solver_cfg=None, overlay_cfg=None)`
  - Runs healthy + AF and returns aligned bundle.

## Post-processing Functions
- `extract_pq_mitral(outputs)`
  - Returns mitral `(delta_pressure_mmhg, flow_ml_per_s)`.
- `extract_pq_tricuspid(outputs)`
  - Returns tricuspid `(delta_pressure_mmhg, flow_ml_per_s)`.
- `extract_pq_signals(outputs)`
  - Returns dictionary of both P-Q pairs.
- `compute_summary_metrics(healthy_outputs, af_outputs)`
  - Returns standardized `met_*` summary metrics dictionary.

## Plot/Export Functions
- `plot_onepage_comparison(healthy_outputs, af_outputs, t_start_s=0.0, t_end_s=3.0)`
- `make_healthy_vs_af_figure(...)` (compatibility alias)
- `export_outputs(...)`
- `save_timeseries_csv(...)`
- `export_onepage(...)`

## Output Dictionary Keys
Canonical keys include:
- `time_s`
- `sig_activation_square`, `sig_activation_sine`
- `left_ventricle_stiffness_cgs`
- `left_ventricle_pressure_mmhg`
- `arterial_compliance_pressure_mmhg`
- `left_ventricle_volume_ml`
- `mitral_flow_ml_per_s`, `tricuspid_flow_ml_per_s`
- pressure support for P-Q: `left_atrium_pressure_cgs`, `left_ventricle_pressure_cgs`, `right_atrium_pressure_cgs`, `right_ventricle_pressure_cgs`

Backward-compatible legacy keys (`PLVM`, `PCAM`, `QLV`, etc.) remain for test compatibility.
