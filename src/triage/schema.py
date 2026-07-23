"""Patient intake schema — the contract between the generator, validators, and agents."""

import json
from enum import IntEnum, StrEnum

from pydantic import BaseModel, Field, computed_field, model_validator


class Sex(StrEnum):
    MALE = "male"
    FEMALE = "female"


class ArrivalMode(StrEnum):
    WALK_IN = "walk_in"
    AMBULANCE = "ambulance"
    WHEELCHAIR = "wheelchair"
    POLICE = "police"


class Department(StrEnum):
    RESUSCITATION = "resuscitation"
    EMERGENCY = "emergency"
    CARDIOLOGY = "cardiology"
    NEUROLOGY = "neurology"
    ORTHOPEDICS = "orthopedics"
    PEDIATRICS = "pediatrics"
    GENERAL_MEDICINE = "general_medicine"
    SURGERY = "surgery"
    OBGYN = "obgyn"
    PSYCHIATRY = "psychiatry"


class ESILevel(IntEnum):
    """Emergency Severity Index: 1 = immediate life threat, 5 = non-urgent."""

    RESUSCITATION = 1
    EMERGENT = 2
    URGENT = 3
    LESS_URGENT = 4
    NON_URGENT = 5


class PainPattern(StrEnum):
    CONSTANT = "constant"
    INTERMITTENT = "intermittent"
    WORSENING = "worsening"
    IMPROVING = "improving"


class Vitals(BaseModel):
    heart_rate: int = Field(ge=20, le=250, description="Beats per minute")
    systolic_bp: int = Field(ge=50, le=260, description="Systolic blood pressure, mmHg")
    diastolic_bp: int = Field(ge=20, le=160, description="Diastolic blood pressure, mmHg")
    respiratory_rate: int = Field(ge=4, le=60, description="Breaths per minute")
    temperature_c: float = Field(ge=32.0, le=43.0, description="Body temperature, Celsius")
    spo2: int = Field(ge=50, le=100, description="Oxygen saturation, percent")
    pain_score: int = Field(ge=0, le=10, description="Self-reported pain, 0-10")


class SymptomDetail(BaseModel):
    duration: str = Field(description="How long the issue has been present, e.g. '45 minutes', '2 weeks'")
    exact_problem: str = Field(description="The physical problem in concrete terms, e.g. 'burning pain in upper abdomen'")
    pain_location: str | None = Field(default=None, description="Precise body location if pain is present, e.g. 'right lower quadrant'")
    pain_pattern: PainPattern | None = Field(default=None, description="Constant vs intermittent vs worsening; None if no pain")
    fever_pattern: str | None = Field(default=None, description="How temperature behaves, e.g. 'spikes every evening'; None if no fever")


class Lifestyle(BaseModel):
    diet_routine: str = Field(description="Typical eating pattern, e.g. 'skips breakfast, heavy spicy dinner'")
    recent_diet_change: str | None = Field(default=None, description="Any change in food habits recently; None if unchanged")
    sleep_routine: str = Field(description="Typical sleep, e.g. '4-5h/night for 2 weeks, night shifts'")
    physical_activity: str = Field(description="Regular activity level, e.g. 'sedentary desk job' or 'gym 4x/week'")
    last_meal: str = Field(description="What and when the patient last ate, e.g. 'street food chaat, 10pm yesterday'")


class SelfTreatment(BaseModel):
    home_remedies_tried: list[str] = Field(default_factory=list, description="e.g. 'ginger tea', 'hot water bag on stomach'")
    pharmacy_meds_taken: list[str] = Field(default_factory=list, description="Non-prescribed meds suggested by pharmacy/friends for this issue")
    last_medication_taken: str | None = Field(default=None, description="Most recent medicine of any kind with timing, e.g. 'ibuprofen 400mg, 6h ago'")


class ExposureHistory(BaseModel):
    blood_transfusion: str | None = Field(default=None, description="Any transfusion received/donated and when; None if never")
    recent_accidents: str | None = Field(default=None, description="Accidents/trauma and how long ago, e.g. 'motorbike fall, 10 days ago'")


class SexualHistory(BaseModel):
    """Only collected for age >= 13; patients may decline (fields stay None)."""

    sexually_active: bool | None = None
    active_since: str | None = Field(default=None, description="Roughly how long sexually active, e.g. 'about 10 years'")
    last_activity: str | None = Field(default=None, description="Last sexual activity, e.g. '3 days ago'")


class FamilyHistory(BaseModel):
    parents_conditions: list[str] = Field(
        default_factory=list,
        description="Parent conditions with attribution, e.g. 'father: type 2 diabetes', 'mother: peptic ulcer'",
    )


class GroundTruth(BaseModel):
    """Labels the generator assigns but agents must never see. Used only for evaluation."""

    esi_level: ESILevel
    department: Department
    probable_condition: str = Field(description="The actual underlying condition, e.g. 'NSAID-induced gastritis'")
    contributing_factors: list[str] = Field(
        default_factory=list,
        description="Causal chain that produced this condition, e.g. ['unknown NSAID on empty stomach', 'heavy lifting', 'sleep deprivation']",
    )
    red_flags: list[str] = Field(
        default_factory=list,
        description="Clinical danger signs present in this case, e.g. 'crushing chest pain radiating to arm'",
    )
    rationale: str = Field(description="One-sentence clinical justification for the labels")


class PatientIntake(BaseModel):
    patient_id: str = Field(pattern=r"^PT-\d{5}$")
    age: int = Field(ge=0, le=110)
    sex: Sex
    height_cm: float = Field(ge=45, le=220)
    weight_kg: float = Field(ge=2.5, le=300)
    arrival_mode: ArrivalMode
    chief_complaint: str = Field(
        min_length=10,
        description="The patient's own words at the desk — informal, vague, sometimes misleading",
    )
    history_of_present_illness: str = Field(
        min_length=30,
        description="Nurse's free-text note: onset, duration, severity, associated symptoms",
    )
    symptom_detail: SymptomDetail
    vitals: Vitals
    lifestyle: Lifestyle
    self_treatment: SelfTreatment
    exposure_history: ExposureHistory
    sexual_history: SexualHistory | None = Field(
        default=None, description="None for pediatric patients (< 13) or when not collected"
    )
    family_history: FamilyHistory
    medical_history: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    past_surgeries: list[str] = Field(default_factory=list)
    ongoing_treatments: list[str] = Field(
        default_factory=list,
        description="Active treatment courses, e.g. chemotherapy, dialysis, antibiotic course, physiotherapy",
    )
    recent_infections: list[str] = Field(
        default_factory=list,
        description="Infectious illnesses in the past 6 months, e.g. covid, dengue, influenza",
    )
    prior_episodes_of_complaint: int = Field(
        ge=0, description="Times this same complaint occurred before; 0 = first episode"
    )
    last_treated_at: str | None = Field(
        default=None,
        description="Where the patient last received any care, e.g. 'urgent care, 3 days ago'",
    )
    ground_truth: GroundTruth

    @computed_field
    @property
    def bmi(self) -> float:
        return round(self.weight_kg / (self.height_cm / 100) ** 2, 1)

    @model_validator(mode="after")
    def no_sexual_history_for_children(self) -> "PatientIntake":
        if self.age < 13 and self.sexual_history is not None:
            raise ValueError("sexual_history must be None for patients under 13")
        return self

    def agent_view(self) -> dict:
        """The record as agents see it: ground truth stripped out."""
        return self.model_dump(mode="json", exclude={"ground_truth"})


if __name__ == "__main__":
    print(json.dumps(PatientIntake.model_json_schema(), indent=2))
