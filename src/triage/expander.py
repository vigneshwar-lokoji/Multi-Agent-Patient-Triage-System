"""Expands scenario blueprints into full, schema-valid PatientIntake records.

The blueprint fixes the causal story; this module varies demographics, vitals
(within acuity-consistent ranges), phrasing, and background details.
"""

import json
import random
from collections import Counter
from pathlib import Path

from triage.blueprint import Blueprint, load_blueprints
from triage.genplan import sample_arrival_mode
from triage.schema import (
    ESILevel,
    GroundTruth,
    PatientIntake,
    Sex,
    SexualHistory,
    SymptomDetail,
    Vitals,
)

INT_VITALS = {"heart_rate", "systolic_bp", "diastolic_bp", "respiratory_rate", "spo2", "pain_score"}


def _sample_vitals(rng: random.Random, ranges: dict[str, tuple[float, float]]) -> Vitals:
    values: dict[str, float | int] = {}
    for name, (lo, hi) in ranges.items():
        if name in INT_VITALS:
            values[name] = rng.randint(int(lo), int(hi))
        else:
            values[name] = round(rng.uniform(lo, hi), 1)
    return Vitals(**values)


def _sample_body(rng: random.Random, age: int, sex: Sex) -> tuple[float, float]:
    if age < 2:
        h = rng.uniform(68, 90)
    elif age < 13:
        h = 85 + (age - 2) * rng.uniform(5.5, 7.0)
    else:
        h = rng.gauss(176, 7) if sex == Sex.MALE else rng.gauss(162, 6)
    h = max(60.0, min(200.0, h))
    if age < 13:
        bmi = rng.uniform(14, 19)
    else:
        bmi = rng.choices(
            [rng.uniform(17, 18.4), rng.uniform(18.5, 24.9), rng.uniform(25, 29.9), rng.uniform(30, 38)],
            weights=[0.08, 0.42, 0.32, 0.18],
        )[0]
    w = bmi * (h / 100) ** 2
    return round(h, 1), round(w, 1)


def _sample_sexual_history(rng: random.Random, age: int) -> SexualHistory | None:
    if age < 16:
        return None
    r = rng.random()
    if r < 0.15:
        return None
    if r < 0.30:
        return SexualHistory()
    active = rng.random() < (0.75 if 18 <= age <= 60 else 0.30)
    if not active:
        return SexualHistory(sexually_active=False)
    since_years = max(1, age - rng.randint(17, 24))
    last = rng.choice(["2 days ago", "last week", "about 2 weeks ago", "about a month ago", "a few months ago"])
    return SexualHistory(sexually_active=True, active_since=f"about {since_years} years", last_activity=last)


def expand_blueprint(bp: Blueprint, count: int, rng: random.Random, id_start: int) -> list[PatientIntake]:
    records = []
    for i in range(count):
        esi = ESILevel(rng.choice(bp.esi_levels))
        age = rng.randint(*bp.age_range)
        sex = bp.sex or rng.choice([Sex.MALE, Sex.FEMALE])
        height, weight = _sample_body(rng, age, sex)
        eligible = [b for b in bp.backgrounds if b.age_range is None or b.age_range[0] <= age <= b.age_range[1]]
        if not eligible:
            raise ValueError(f"{bp.blueprint_id}: no background bundle eligible for age {age}")
        background = rng.choice(eligible)
        lifestyle_bundle = rng.choice(bp.lifestyles)
        self_treatment = rng.choice(bp.self_treatments)

        factors: list[str] = []
        for f in (
            bp.contributing_factors
            + lifestyle_bundle.contributing_factors
            + self_treatment.contributing_factors
            + background.contributing_factors
        ):
            if f not in factors:
                factors.append(f)

        slot_values = {name: rng.choice(options) for name, options in bp.slots.items()}
        slot_values["duration"] = rng.choice(bp.duration_options)
        slot_values["work_pattern"] = lifestyle_bundle.work_pattern or ""
        slot_values["last_medication"] = self_treatment.last_medication_taken or "no medicines"
        slot_values["pharmacy_meds"] = (
            ", ".join(self_treatment.pharmacy_meds_taken) if self_treatment.pharmacy_meds_taken else "no pharmacy medicines"
        )
        slot_values["home_remedies"] = (
            ", ".join(self_treatment.home_remedies_tried) if self_treatment.home_remedies_tried else "nothing"
        )

        record = PatientIntake(
            patient_id=f"PT-{id_start + i:05d}",
            age=age,
            sex=sex,
            height_cm=height,
            weight_kg=weight,
            arrival_mode=sample_arrival_mode(rng, esi),
            chief_complaint=rng.choice(bp.chief_complaint_templates).format(**slot_values),
            history_of_present_illness=rng.choice(bp.hpi_templates).format(**slot_values),
            symptom_detail=SymptomDetail(
                duration=slot_values["duration"],
                exact_problem=bp.exact_problem,
                pain_location=bp.pain_location,
                pain_pattern=rng.choice(bp.pain_patterns) if bp.pain_patterns else None,
                fever_pattern=rng.choice(bp.fever_pattern_options),
            ),
            vitals=_sample_vitals(rng, bp.vitals_ranges),
            lifestyle={
                "diet_routine": lifestyle_bundle.diet_routine,
                "recent_diet_change": lifestyle_bundle.recent_diet_change,
                "sleep_routine": lifestyle_bundle.sleep_routine,
                "physical_activity": lifestyle_bundle.physical_activity,
                "last_meal": rng.choice(lifestyle_bundle.last_meal_options),
            },
            self_treatment=self_treatment.model_dump(exclude={"contributing_factors"}),
            exposure_history={
                "blood_transfusion": rng.choice([None] * 9 + ["received one unit after surgery, years ago"]),
                "recent_accidents": None,
            },
            sexual_history=_sample_sexual_history(rng, age),
            family_history={"parents_conditions": background.family_history},
            medical_history=background.medical_history,
            medications=background.medications,
            allergies=background.allergies,
            past_surgeries=background.past_surgeries,
            ongoing_treatments=background.ongoing_treatments,
            recent_infections=background.recent_infections,
            prior_episodes_of_complaint=rng.choice(bp.prior_episodes_options),
            last_treated_at=rng.choice(bp.last_treated_options),
            ground_truth=GroundTruth(
                esi_level=esi,
                department=bp.department,
                probable_condition=bp.probable_condition,
                contributing_factors=factors,
                red_flags=bp.red_flags,
                rationale=bp.rationale,
            ),
        )
        records.append(record)
    return records


def build_dataset(total: int, seed: int, out_path: Path) -> list[PatientIntake]:
    rng = random.Random(seed)
    blueprints = load_blueprints()
    base, remainder = divmod(total, len(blueprints))
    records: list[PatientIntake] = []
    for i, bp in enumerate(blueprints):
        count = base + (1 if i < remainder else 0)
        records.extend(expand_blueprint(bp, count, rng, id_start=len(records) + 1))
    with out_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r.model_dump(mode="json")) + "\n")
    return records


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("data/sample_batch.jsonl"))
    args = parser.parse_args()

    records = build_dataset(total=args.total, seed=args.seed, out_path=args.out)
    blueprints = load_blueprints()
    print(f"Generated {len(records)} records from {len(blueprints)} blueprints -> {args.out}\n")

    dept = Counter(r.ground_truth.department for r in records)
    esi = Counter(r.ground_truth.esi_level for r in records)
    conditions = len({r.ground_truth.probable_condition for r in records})
    unique_cc = len({r.chief_complaint for r in records})
    print("Departments:", {d.value: c for d, c in dept.most_common()})
    print("ESI:", {f"ESI-{k.value}": esi[k] for k in sorted(esi)})
    print(f"Distinct conditions: {conditions} | Unique chief complaints: {unique_cc}/{len(records)}")


if __name__ == "__main__":
    main()
