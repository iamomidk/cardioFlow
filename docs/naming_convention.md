# Naming Convention Policy

## Case Rules
- `snake_case`: variables and functions
- `UPPER_SNAKE_CASE`: constants
- `PascalCase`: dataclasses/types only

## Prefix Rules
- `sig_`: generated signals
- `state_`: ODE states
- `der_`: derivatives
- `param_`: parameters
- `out_`: exported/postprocessed
- `met_`: metrics

## Required Unit Suffixes
- `_s`, `_ms`, `_hz`, `_mmhg`, `_cgs`, `_ml`, `_ml_per_s`, `_l_per_min`, `_ratio`, `_index`, `_flag`

## Cardiac Name Map
- `PLVM` -> `left_ventricle_pressure_mmhg`
- `PCAM` -> `arterial_compliance_pressure_mmhg`
- `QLV` -> `left_ventricle_volume_ml`
- `SLV` -> `left_ventricle_stiffness_cgs`
- `SSW` -> `sig_activation_sine`
- `STW` -> `sig_activation_square`
- `FLA` -> `mitral_flow_ml_per_s`
- `FRA` -> `tricuspid_flow_ml_per_s`
- `PLA` -> `left_atrium_pressure_cgs` (internal); converted output: `left_atrium_pressure_mmhg`
- `PRA` -> `right_atrium_pressure_cgs` (internal); converted output: `right_atrium_pressure_mmhg`
- `PLV` -> `left_ventricle_pressure_cgs` (internal)

## Function Naming Style
- `build_activation_signals(...)`
- `compute_baseline_derivatives(...)`
- `apply_atrial_arrhythmia_overlay(...)`
- `run_simulation(...)`
- `extract_pq_signals(...)`
- `compute_summary_metrics(...)`
- `plot_onepage_comparison(...)`
- `export_outputs(...)`
