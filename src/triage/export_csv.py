"""Flattens the JSONL dataset into a CSV view for spreadsheet inspection.

JSONL stays the canonical format; CSV is a lossy human-friendly projection
(nested fields become underscore-joined columns, lists become '; '-joined cells).
"""

import csv
import json
import sys
from pathlib import Path


def flatten(obj, prefix: str = "") -> dict[str, str]:
    flat: dict[str, str] = {}
    for key, value in obj.items():
        column = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, prefix=f"{column}_"))
        elif isinstance(value, list):
            flat[column] = "; ".join(str(v) for v in value)
        else:
            flat[column] = "" if value is None else str(value)
    return flat


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/sample_batch.jsonl")
    dst = src.with_suffix(".csv")
    rows = [flatten(json.loads(line)) for line in src.read_text().splitlines()]
    columns: list[str] = []
    for row in rows:
        for column in row:
            if column not in columns:
                columns.append(column)
    with dst.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows x {len(columns)} columns -> {dst}")


if __name__ == "__main__":
    main()
