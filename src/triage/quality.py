"""Quality gates for the synthetic dataset.

Errors (hard failures): schema violations, unfilled template slots, duplicate ids.
Warnings (review flags): ungrounded causal factors, acuity/vitals mismatches,
excessive narrative duplication.
"""

import json
import sys
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from triage.schema import PatientIntake

STOPWORDS = {
    "with", "from", "while", "during", "after", "before", "this", "that", "them",
    "then", "than", "were", "been", "have", "into", "onto", "over", "under",
    "their", "there", "when", "where", "which", "taken", "took", "without",
}


def _flatten_values(obj) -> list[str]:
    if isinstance(obj, dict):
        return [v for val in obj.values() for v in _flatten_values(val)]
    if isinstance(obj, list):
        return [v for item in obj for v in _flatten_values(item)]
    if isinstance(obj, str):
        return [obj]
    return []


def _grounded(factor: str, blob_prefixes: set[str]) -> bool:
    """A factor is grounded if any content word appears (by 4-char prefix) in the visible record."""
    words = [w.strip(".,;:()").lower() for w in factor.split()]
    content = [w for w in words if len(w) >= 4 and w not in STOPWORDS]
    return any(w[:4] in blob_prefixes for w in content)


def check_dataset(path: Path) -> int:
    errors: list[str] = []
    warnings: list[str] = []
    records: list[PatientIntake] = []

    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        try:
            records.append(PatientIntake.model_validate_json(line))
        except ValidationError as e:
            errors.append(f"line {line_no}: schema violation: {e.errors()[0]['loc']} {e.errors()[0]['msg']}")

    ids = Counter(r.patient_id for r in records)
    for pid, n in ids.items():
        if n > 1:
            errors.append(f"duplicate patient_id {pid} x{n}")

    ungrounded = Counter()
    for r in records:
        text = r.chief_complaint + " " + r.history_of_present_illness
        if "{" in text or "}" in text:
            errors.append(f"{r.patient_id}: unfilled template slot in narrative")

        blob_words = " ".join(_flatten_values(r.agent_view())).lower().split()
        blob_prefixes = {w.strip(".,;:()")[:4] for w in blob_words if len(w.strip(".,;:()")) >= 4}
        for factor in r.ground_truth.contributing_factors:
            if not _grounded(factor, blob_prefixes):
                ungrounded[factor] += 1

        v = r.vitals
        vitals_normal = (
            55 <= v.heart_rate <= 100
            and 100 <= v.systolic_bp <= 145
            and v.respiratory_rate <= 20
            and v.spo2 >= 95
            and v.temperature_c < 38.0
        )
        # ESI-2 can legitimately have normal vitals (risk-based acuity, e.g. early-pregnancy
        # bleeding or suicidal ideation); only ESI-1 with normal vitals is suspect.
        if r.ground_truth.esi_level == 1 and vitals_normal:
            warnings.append(f"{r.patient_id}: ESI-1 but vitals look normal")

    for factor, n in ungrounded.items():
        warnings.append(f"ungrounded factor ({n}x): '{factor}' not visible in any record field")

    narrative_pairs = Counter((r.chief_complaint, r.history_of_present_illness) for r in records)
    exact_dupes = sum(n - 1 for n in narrative_pairs.values() if n > 1)
    dupe_rate = exact_dupes / len(records) if records else 0
    if dupe_rate > 0.10:
        warnings.append(f"narrative duplication {dupe_rate:.0%} exceeds 10% budget ({exact_dupes} dupes)")

    dept = Counter(r.ground_truth.department.value for r in records)
    esi = Counter(r.ground_truth.esi_level.value for r in records)
    sexes = Counter(r.sex.value for r in records)
    ages = Counter(
        "0-12" if r.age < 13 else "13-17" if r.age < 18 else "18-39" if r.age < 40 else "40-64" if r.age < 65 else "65+"
        for r in records
    )

    print(f"Records checked : {len(records)}")
    print(f"Departments     : {dict(dept.most_common())}")
    print(f"ESI levels      : {dict(sorted(esi.items()))}")
    print(f"Sex             : {dict(sexes)}")
    print(f"Age bands       : {dict(sorted(ages.items()))}")
    print(f"Exact narrative dupes: {exact_dupes} ({dupe_rate:.1%})")
    print(f"\nERRORS   ({len(errors)}):")
    for e in errors[:20]:
        print(f"  [E] {e}")
    print(f"WARNINGS ({len(warnings)}):")
    for w in warnings[:20]:
        print(f"  [W] {w}")
    print(f"\n{'FAIL' if errors else 'PASS'}")
    return 1 if errors else 0


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/sample_batch.jsonl")
    raise SystemExit(check_dataset(target))
