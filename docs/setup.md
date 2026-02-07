# Setup and Installation

## Requirements
- macOS/Linux shell
- Python 3.10+
- `venv` support

## Recommended Installation (virtual environment)
```bash
cd /Users/omid/ayda/cardioFlow
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Why this is required on Homebrew Python
If you see `externally-managed-environment` (PEP 668), do not install globally with system pip. Use the local `.venv` as above.

## Verify environment
```bash
python --version
python -c "import numpy, scipy, matplotlib; print('ok')"
```

## Quick smoke run
```bash
python cardio_onefile.py compare --duration-s 3 --dt-output-s 0.001 --random-seed 1 --output-dir out --export-format both
```

Expected output directory:
- `out/comparison_onepage.pdf`
- `out/comparison_onepage.png`
- `out/timeseries_healthy.csv`
- `out/timeseries_af.csv`
- `out/summary_metrics.csv`
- `out/run_metadata.txt`
