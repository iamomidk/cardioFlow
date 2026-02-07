# CardioFlow

Single-file PF-1 implementation for:
- healthy baseline,
- atrial arrhythmia overlay,
- P-Q comparison,
- one-page 4-panel figure.

## Scope and Rules
- Scope lock: `docs/project_scope.md`
- PDF traceability: `docs/pdf_traceability.md`
- Assumptions registry: `docs/assumptions.md`
- Naming policy: `docs/naming_convention.md`
- Legacy glossary: `docs/variable_glossary.csv`

## Run
```bash
python cardio_onefile.py compare --duration-s 3 --dt-output-s 0.001 --random-seed 1 --output-dir out --export-format both
```

Modes:
- `healthy`
- `atrial-arrhythmia`
- `compare` (default)

## Required Artifacts
Running compare creates under `out/`:
- `comparison_onepage.pdf`
- `comparison_onepage.png` (if requested)
- `timeseries_healthy.csv`
- `timeseries_af.csv`
- `summary_metrics.csv`
- `run_metadata.txt`

## Tests
```bash
pytest
```

## Naming Checker
```bash
python tools/check_naming.py
```
