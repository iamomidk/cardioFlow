# Outputs and Artifacts

The compare run writes all artifacts under `out/`.

## Figure Artifacts
- `comparison_onepage.pdf` (required)
- `comparison_onepage.png` (when `--export-format` includes PNG)

Figure layout (fixed 2x2):
1. Activity generator waveforms
2. Ventricular stiffness
3. Pressures and volume (dual axis)
4. Ventricular PV loop

## Time-Series CSVs
- `timeseries_healthy.csv`
- `timeseries_af.csv`

CSV columns are self-descriptive with unit suffixes.
Examples:
- `time_s`
- `left_ventricle_pressure_mmhg_healthy`
- `left_ventricle_volume_ml_af`
- `mitral_flow_ml_per_s_healthy`

## Metrics CSV
- `summary_metrics.csv`

Contains:
- `met_mean_left_ventricle_pressure_mmhg_healthy`
- `met_mean_left_ventricle_pressure_mmhg_af`
- `met_mean_arterial_compliance_pressure_mmhg_healthy`
- `met_mean_arterial_compliance_pressure_mmhg_af`
- `met_stroke_volume_proxy_ml_healthy`
- `met_stroke_volume_proxy_ml_af`
- `met_beat_to_beat_variability_index_healthy`
- `met_beat_to_beat_variability_index_af`

## Reproducibility Metadata
- `run_metadata.txt`

Contains run settings needed for rerun:
- simulation horizon and output step
- random seed
- solver method and tolerances
- assumption IDs applied
