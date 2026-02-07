# CLI Reference

Entrypoint:
```bash
python cardio_onefile.py [mode] [options]
```

Modes:
- `healthy`
- `atrial-arrhythmia`
- `compare` (default)

Backward-compatible alias:
- `af` (same as `atrial-arrhythmia`)

## Primary Options
- `--duration-s FLOAT` simulation horizon in seconds
- `--dt-output-s FLOAT` output sampling step in seconds
- `--random-seed INT` AF random seed
- `--output-dir PATH` destination directory for outputs
- `--export-format {pdf,png,both}` figure export format

## Backward-Compatible Options
- `--tf` -> `--duration-s`
- `--dt-out` -> `--dt-output-s`
- `--seed` -> `--random-seed`
- `--outdir` -> `--output-dir`
- `--format` -> `--export-format`

## Examples
Compare run:
```bash
python cardio_onefile.py compare --duration-s 3 --dt-output-s 0.001 --random-seed 1 --output-dir out --export-format both
```

Healthy-only run:
```bash
python cardio_onefile.py healthy --duration-s 5 --dt-output-s 0.002 --output-dir out_healthy
```

Atrial-arrhythmia-only run:
```bash
python cardio_onefile.py atrial-arrhythmia --duration-s 5 --dt-output-s 0.002 --random-seed 7 --output-dir out_af
```
