#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

FORBIDDEN_SHORT_NAMES = {"tmp", "val", "data2", "foo", "bar", "baz"}
ALLOWED_SHORT_LOOP = {"i", "j", "k"}
UNIT_SUFFIXES = (
    "_s",
    "_ms",
    "_hz",
    "_mmhg",
    "_cgs",
    "_ml",
    "_ml_per_s",
    "_l_per_min",
    "_ratio",
    "_index",
    "_flag",
)


def check_code_names(code_text: str) -> list[str]:
    violations: list[str] = []
    assign_pattern = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=", re.MULTILINE)
    for name in assign_pattern.findall(code_text):
        if name in FORBIDDEN_SHORT_NAMES:
            violations.append(f"forbidden short name: {name}")
        if len(name) == 1 and name not in ALLOWED_SHORT_LOOP and name not in {"_"}:
            violations.append(f"single-letter non-loop name: {name}")
        if name.isupper() and not re.match(r"^[A-Z][A-Z0-9_]*$", name):
            violations.append(f"invalid uppercase identifier: {name}")
    return violations


def check_export_headers(csv_path: Path) -> list[str]:
    if not csv_path.exists():
        return [f"missing csv for header check: {csv_path}"]
    with csv_path.open(newline="") as handle:
        reader = csv.reader(handle)
        headers = next(reader, [])
    violations: list[str] = []
    for header in headers:
        if header == "metric_name" or header == "metric_value" or header == "time_s":
            continue
        normalized = header
        if normalized.endswith("_healthy"):
            normalized = normalized[: -len("_healthy")]
        elif normalized.endswith("_af"):
            normalized = normalized[: -len("_af")]
        if not normalized.endswith(UNIT_SUFFIXES):
            violations.append(f"missing unit suffix in export column: {header}")
    return violations


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    code_path = repo_root / "cardio_onefile.py"
    code_text = code_path.read_text(encoding="utf-8")
    violations = check_code_names(code_text)

    out_dir = repo_root / "out"
    for csv_name in ("timeseries_healthy.csv", "timeseries_af.csv"):
        violations.extend(check_export_headers(out_dir / csv_name))

    if violations:
        for violation in violations:
            print(f"[naming-check] {violation}")
        return 1
    print("[naming-check] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
