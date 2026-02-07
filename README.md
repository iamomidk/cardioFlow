# CardioFlow

Single-file PF-1 cardiovascular simulation focused on:
- healthy baseline,
- physiology-derived atrial arrhythmia overlay,
- pressure-flow (P-Q) comparison,
- one-page 4-panel figure export.

## Quick Start
```bash
cd /Users/omid/ayda/cardioFlow
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python cardio_onefile.py compare --duration-s 3 --dt-output-s 0.001 --random-seed 1 --output-dir out --export-format both
```

## Main Commands
Compare (default):
```bash
python cardio_onefile.py compare --duration-s 3 --dt-output-s 0.001 --random-seed 1 --output-dir out --export-format both
```

Healthy only:
```bash
python cardio_onefile.py healthy --duration-s 3 --dt-output-s 0.001 --output-dir out_healthy
```

Atrial arrhythmia only:
```bash
python cardio_onefile.py atrial-arrhythmia --duration-s 3 --dt-output-s 0.001 --random-seed 1 --output-dir out_af
```

## Required Artifact Set (`out/`)
- `comparison_onepage.pdf`
- `comparison_onepage.png` (if requested)
- `timeseries_healthy.csv`
- `timeseries_af.csv`
- `summary_metrics.csv`
- `run_metadata.txt`

## Validation
```bash
python -m pytest
python tools/check_naming.py
```

## Documentation
- Full docs index: `docs/index.md`
- Setup: `docs/setup.md`
- CLI reference: `docs/cli_reference.md`
- Model overview: `docs/model_overview.md`
- API reference: `docs/api_reference.md`
- Outputs: `docs/outputs.md`
- Development: `docs/development.md`
- Troubleshooting: `docs/troubleshooting.md`

## Academic/Scope Documents
- Scope lock: `docs/project_scope.md`
- PDF traceability: `docs/pdf_traceability.md`
- Assumptions registry: `docs/assumptions.md`
- Naming policy: `docs/naming_convention.md`
- Variable glossary: `docs/variable_glossary.csv`

## Report Package
- `report/final_report.md`
- `report/final_report.pdf`
- `report/figures/comparison_onepage.pdf`
- `report/appendix_traceability.md`
