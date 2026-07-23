"""Scenario blueprints: one hand-authored causal story, expandable into many records.

The blueprint fixes the CAUSAL CHAIN (condition + contributing factors) and the
clinically-consistent ranges; the expander varies everything else.
"""

import json
from pathlib import Path

from pydantic import BaseModel, Field

from triage.schema import Department, ESILevel, PainPattern, Sex


class BackgroundBundle(BaseModel):
    """A coherent patient background — bundled so correlated facts stay together
    (a diabetic has metformin in medications, not just a history entry)."""

    medical_history: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    past_surgeries: list[str] = Field(default_factory=list)
    ongoing_treatments: list[str] = Field(default_factory=list)
    recent_infections: list[str] = Field(default_factory=list)
    family_history: list[str] = Field(default_factory=list)
    age_range: tuple[int, int] | None = None
    contributing_factors: list[str] = Field(default_factory=list)


class LifestyleBundle(BaseModel):
    diet_routine: str
    recent_diet_change: str | None = None
    sleep_routine: str
    physical_activity: str
    last_meal_options: list[str]
    work_pattern: str | None = None
    contributing_factors: list[str] = Field(default_factory=list)


class SelfTreatmentBundle(BaseModel):
    home_remedies_tried: list[str] = Field(default_factory=list)
    pharmacy_meds_taken: list[str] = Field(default_factory=list)
    last_medication_taken: str | None = None
    contributing_factors: list[str] = Field(default_factory=list)


class Blueprint(BaseModel):
    blueprint_id: str
    condition_family: str
    department: Department
    esi_levels: list[ESILevel]
    age_range: tuple[int, int]
    sex: Sex | None = None

    probable_condition: str
    contributing_factors: list[str] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)
    rationale: str

    duration_options: list[str]
    exact_problem: str
    pain_location: str | None = None
    pain_patterns: list[PainPattern] = Field(default_factory=list)
    fever_pattern_options: list[str | None] = Field(default=[None])

    chief_complaint_templates: list[str]
    hpi_templates: list[str]
    slots: dict[str, list[str]] = Field(default_factory=dict)

    vitals_ranges: dict[str, tuple[float, float]]

    backgrounds: list[BackgroundBundle]
    lifestyles: list[LifestyleBundle]
    self_treatments: list[SelfTreatmentBundle]
    prior_episodes_options: list[int] = Field(default=[0])
    last_treated_options: list[str | None] = Field(default=[None])


def load_blueprints(directory: Path = Path("data/blueprints")) -> list[Blueprint]:
    blueprints = []
    for path in sorted(directory.glob("*.json")):
        for raw in json.loads(path.read_text()):
            blueprints.append(Blueprint.model_validate(raw))
    ids = [b.blueprint_id for b in blueprints]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate blueprint_id found")
    return blueprints
