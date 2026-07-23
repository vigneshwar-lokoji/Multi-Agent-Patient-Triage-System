"""Generation plan: decides WHAT cases exist before any LLM writes a word.

The RNG (not the LLM) controls the dataset's statistics — archetype mix,
age/sex spread, acuity distribution. This prevents mode collapse.
"""

import json
import random
from pathlib import Path

from pydantic import BaseModel

from triage.schema import ArrivalMode, Department, ESILevel, Sex


class Archetype(BaseModel):
    name: str
    department: Department
    esi_levels: list[ESILevel]
    age_range: tuple[int, int]
    weight: float
    sex: Sex | None = None


ARCHETYPES = [
    # name, department, allowed ESI, age range, weight (rough ED frequency)
    Archetype(name="cardiac_arrest", department=Department.RESUSCITATION, esi_levels=[ESILevel.RESUSCITATION], age_range=(45, 90), weight=0.5),
    Archetype(name="acute_coronary_syndrome", department=Department.RESUSCITATION, esi_levels=[ESILevel.RESUSCITATION, ESILevel.EMERGENT], age_range=(40, 88), weight=1.5),
    Archetype(name="anaphylaxis", department=Department.RESUSCITATION, esi_levels=[ESILevel.RESUSCITATION, ESILevel.EMERGENT], age_range=(5, 70), weight=0.7),
    Archetype(name="septic_shock", department=Department.RESUSCITATION, esi_levels=[ESILevel.RESUSCITATION, ESILevel.EMERGENT], age_range=(50, 95), weight=0.8),
    Archetype(name="stable_chest_pain", department=Department.CARDIOLOGY, esi_levels=[ESILevel.EMERGENT, ESILevel.URGENT], age_range=(35, 80), weight=3.0),
    Archetype(name="palpitations_afib", department=Department.CARDIOLOGY, esi_levels=[ESILevel.EMERGENT, ESILevel.URGENT], age_range=(50, 90), weight=1.5),
    Archetype(name="syncope", department=Department.CARDIOLOGY, esi_levels=[ESILevel.EMERGENT, ESILevel.URGENT], age_range=(18, 85), weight=1.2),
    Archetype(name="heart_failure_exacerbation", department=Department.CARDIOLOGY, esi_levels=[ESILevel.EMERGENT, ESILevel.URGENT], age_range=(55, 92), weight=1.0),
    Archetype(name="stroke_symptoms", department=Department.NEUROLOGY, esi_levels=[ESILevel.RESUSCITATION, ESILevel.EMERGENT], age_range=(45, 92), weight=1.5),
    Archetype(name="migraine", department=Department.NEUROLOGY, esi_levels=[ESILevel.URGENT, ESILevel.LESS_URGENT], age_range=(16, 60), weight=2.0),
    Archetype(name="first_seizure", department=Department.NEUROLOGY, esi_levels=[ESILevel.EMERGENT, ESILevel.URGENT], age_range=(16, 70), weight=0.8),
    Archetype(name="vertigo", department=Department.NEUROLOGY, esi_levels=[ESILevel.URGENT, ESILevel.LESS_URGENT], age_range=(30, 80), weight=1.0),
    Archetype(name="ankle_injury", department=Department.ORTHOPEDICS, esi_levels=[ESILevel.LESS_URGENT, ESILevel.NON_URGENT], age_range=(12, 60), weight=2.5),
    Archetype(name="wrist_fracture", department=Department.ORTHOPEDICS, esi_levels=[ESILevel.URGENT, ESILevel.LESS_URGENT], age_range=(8, 85), weight=1.8),
    Archetype(name="acute_back_pain", department=Department.ORTHOPEDICS, esi_levels=[ESILevel.URGENT, ESILevel.LESS_URGENT, ESILevel.NON_URGENT], age_range=(25, 70), weight=2.2),
    Archetype(name="pediatric_fever", department=Department.PEDIATRICS, esi_levels=[ESILevel.URGENT, ESILevel.LESS_URGENT], age_range=(0, 12), weight=2.5),
    Archetype(name="pediatric_asthma", department=Department.PEDIATRICS, esi_levels=[ESILevel.EMERGENT, ESILevel.URGENT], age_range=(2, 14), weight=1.2),
    Archetype(name="pediatric_ear_infection", department=Department.PEDIATRICS, esi_levels=[ESILevel.LESS_URGENT, ESILevel.NON_URGENT], age_range=(0, 10), weight=1.5),
    Archetype(name="uti", department=Department.GENERAL_MEDICINE, esi_levels=[ESILevel.LESS_URGENT, ESILevel.NON_URGENT], age_range=(18, 90), weight=2.0),
    Archetype(name="influenza_like_illness", department=Department.GENERAL_MEDICINE, esi_levels=[ESILevel.URGENT, ESILevel.LESS_URGENT, ESILevel.NON_URGENT], age_range=(16, 85), weight=2.8),
    Archetype(name="hyperglycemia", department=Department.GENERAL_MEDICINE, esi_levels=[ESILevel.EMERGENT, ESILevel.URGENT], age_range=(25, 80), weight=1.0),
    Archetype(name="med_refill_hypertension", department=Department.GENERAL_MEDICINE, esi_levels=[ESILevel.NON_URGENT], age_range=(40, 85), weight=1.0),
    Archetype(name="appendicitis", department=Department.SURGERY, esi_levels=[ESILevel.EMERGENT, ESILevel.URGENT], age_range=(8, 50), weight=1.2),
    Archetype(name="deep_laceration", department=Department.SURGERY, esi_levels=[ESILevel.URGENT, ESILevel.LESS_URGENT], age_range=(14, 70), weight=1.5),
    Archetype(name="acute_abdomen", department=Department.SURGERY, esi_levels=[ESILevel.EMERGENT, ESILevel.URGENT], age_range=(30, 85), weight=1.3),
    Archetype(name="first_trimester_bleeding", department=Department.OBGYN, esi_levels=[ESILevel.EMERGENT, ESILevel.URGENT], age_range=(18, 44), weight=1.0, sex=Sex.FEMALE),
    Archetype(name="active_labor", department=Department.OBGYN, esi_levels=[ESILevel.EMERGENT], age_range=(18, 44), weight=0.8, sex=Sex.FEMALE),
    Archetype(name="pelvic_pain", department=Department.OBGYN, esi_levels=[ESILevel.URGENT, ESILevel.LESS_URGENT], age_range=(16, 50), weight=1.0, sex=Sex.FEMALE),
    Archetype(name="suicidal_ideation", department=Department.PSYCHIATRY, esi_levels=[ESILevel.EMERGENT, ESILevel.URGENT], age_range=(14, 65), weight=1.0),
    Archetype(name="panic_attack", department=Department.PSYCHIATRY, esi_levels=[ESILevel.URGENT, ESILevel.LESS_URGENT], age_range=(16, 55), weight=1.2),
    Archetype(name="acute_psychosis", department=Department.PSYCHIATRY, esi_levels=[ESILevel.EMERGENT, ESILevel.URGENT], age_range=(18, 55), weight=0.7),
    Archetype(name="moderate_allergic_reaction", department=Department.EMERGENCY, esi_levels=[ESILevel.URGENT, ESILevel.LESS_URGENT], age_range=(5, 70), weight=1.2),
    Archetype(name="opioid_overdose", department=Department.EMERGENCY, esi_levels=[ESILevel.RESUSCITATION, ESILevel.EMERGENT], age_range=(18, 60), weight=0.8),
    Archetype(name="dehydration", department=Department.EMERGENCY, esi_levels=[ESILevel.URGENT, ESILevel.LESS_URGENT], age_range=(16, 90), weight=1.3),
    Archetype(name="minor_head_injury", department=Department.EMERGENCY, esi_levels=[ESILevel.URGENT, ESILevel.LESS_URGENT], age_range=(10, 80), weight=1.5),
]


class CaseSpec(BaseModel):
    """One slot in the dataset the generator must fill."""

    patient_id: str
    archetype: str
    department: Department
    esi_level: ESILevel
    age: int
    sex: Sex
    arrival_mode: ArrivalMode
    twist: str


TWISTS = [
    "presentation is textbook-typical",
    "presentation is textbook-typical",
    "presentation is textbook-typical",
    "patient downplays symptoms, chief complaint sounds milder than reality",
    "patient is anxious and exaggerates a minor condition",
    "atypical presentation for this condition",
    "patient has poor English, chief complaint is fragmented",
    "relevant detail buried mid-narrative, complaint mentions something else",
    "elderly patient with vague, nonspecific complaint",
    "patient self-diagnosed incorrectly and leads with the wrong theory",
]


def sample_arrival_mode(rng: random.Random, esi: ESILevel) -> ArrivalMode:
    if esi == ESILevel.RESUSCITATION:
        return rng.choices([ArrivalMode.AMBULANCE, ArrivalMode.WALK_IN], [0.9, 0.1])[0]
    if esi == ESILevel.EMERGENT:
        return rng.choices(
            [ArrivalMode.AMBULANCE, ArrivalMode.WALK_IN, ArrivalMode.WHEELCHAIR, ArrivalMode.POLICE],
            [0.45, 0.4, 0.1, 0.05],
        )[0]
    return rng.choices(
        [ArrivalMode.WALK_IN, ArrivalMode.WHEELCHAIR, ArrivalMode.AMBULANCE],
        [0.8, 0.12, 0.08],
    )[0]


def build_plan(n: int, seed: int = 42) -> list[CaseSpec]:
    rng = random.Random(seed)
    weights = [a.weight for a in ARCHETYPES]
    specs = []
    for i in range(1, n + 1):
        arch = rng.choices(ARCHETYPES, weights)[0]
        esi = rng.choice(arch.esi_levels)
        specs.append(
            CaseSpec(
                patient_id=f"PT-{i:05d}",
                archetype=arch.name,
                department=arch.department,
                esi_level=esi,
                age=rng.randint(*arch.age_range),
                sex=arch.sex or rng.choice([Sex.MALE, Sex.FEMALE]),
                arrival_mode=sample_arrival_mode(rng, esi),
                twist=rng.choice(TWISTS),
            )
        )
    return specs


def main(n: int = 120, seed: int = 42) -> None:
    specs = build_plan(n, seed)
    out = Path("data/generation_plan.json")
    out.write_text(json.dumps([s.model_dump(mode="json") for s in specs], indent=2))

    from collections import Counter

    dept = Counter(s.department for s in specs)
    esi = Counter(s.esi_level for s in specs)
    print(f"Wrote {len(specs)} case specs -> {out}")
    print("\nDepartment mix:")
    for d, c in dept.most_common():
        print(f"  {d.value:<18} {c:>3}  {'#' * c}")
    print("\nESI mix:")
    for lvl in sorted(esi):
        print(f"  ESI-{lvl.value}  {esi[lvl]:>3}  {'#' * esi[lvl]}")


if __name__ == "__main__":
    main()
