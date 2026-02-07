# Development and Testing

## Local Workflow
1. Activate virtual environment:
```bash
cd /Users/omid/ayda/cardioFlow
source .venv/bin/activate
```
2. Run tests:
```bash
python -m pytest
```
3. Run naming checks:
```bash
python tools/check_naming.py
```
4. Run compare export:
```bash
python cardio_onefile.py compare --duration-s 3 --dt-output-s 0.001 --random-seed 1 --output-dir out --export-format both
```

## Test Suite
Current tests cover:
- unit conversions and switching primitives
- healthy + arrhythmia simulation smoke and stability checks
- reproducibility for fixed random seeds
- P-Q signal extraction
- one-page figure generation

Test files are in `tests/`.

## Naming and Consistency Rules
- Follow `docs/naming_convention.md`
- Keep legacy-to-standard mapping in `docs/variable_glossary.csv`
- Ensure exported columns include unit suffixes

## Documentation Maintenance
When changing model behavior:
- Update `docs/pdf_traceability.md` for any equation mapping changes.
- Update `docs/assumptions.md` for any non-PDF logic changes.
- Update `report/appendix_traceability.md` for submission consistency.

## Release Checklist
- Compare command completes
- `out/` contains required artifact set
- `pytest` passes
- Naming check passes
- `report/final_report.pdf` and `report/figures/comparison_onepage.pdf` are current
