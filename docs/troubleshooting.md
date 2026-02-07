# Troubleshooting

## 1. `externally-managed-environment` during pip install
Cause: Homebrew/system Python enforces PEP 668.

Fix:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## 2. `ModuleNotFoundError: No module named matplotlib` (or numpy/scipy)
Cause: Dependencies installed outside active interpreter.

Fix:
```bash
source .venv/bin/activate
python -m pip install -e .
```

## 3. PyCharm uses wrong interpreter
Fix:
- Open interpreter settings
- Select `/Users/omid/ayda/cardioFlow/.venv/bin/python`

## 4. Figure generation crashes with GUI backend
This project forces `matplotlib` backend to `Agg` in code for headless-safe plotting.

## 5. Reproducibility mismatch
Check:
- same `--random-seed`
- same `--duration-s` and `--dt-output-s`
- same package versions (`requirements.lock`)

## 6. Missing expected artifacts in `out/`
Run compare mode explicitly:
```bash
python cardio_onefile.py compare --duration-s 3 --dt-output-s 0.001 --random-seed 1 --output-dir out --export-format both
```
