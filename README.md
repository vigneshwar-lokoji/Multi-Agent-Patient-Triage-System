# Multi-Agent Patient Triage System

A learning project that builds an emergency-department triage pipeline end to end: a frontier LLM authors a fully synthetic clinical intake dataset, and small local models (via Ollama) act as cooperating agents that read each patient record, check triage guidelines, and route the patient — then get scored against hidden ground truth.

> **Disclaimer:** every record in this repository is synthetic, generated from hand-authored scenario blueprints. There is no real patient data (no PHI) anywhere in this project, and nothing here is medical advice. This is an educational systems/ML project.

## Why this project

Most agent demos call one big cloud model in a loop. This project explores the opposite setup, sometimes called **data-level knowledge distillation**:

- A **frontier model** (the "master chef") does the expensive, one-time work: designing the schema, encoding medical-style causal reasoning into scenario blueprints, and generating a large labeled dataset.
- **Small local models** (the "line cooks", running on a 6 GB consumer GPU through Ollama) do the runtime work: extraction, guideline checking, and routing — cheap, private, and repeatable.
- Because the dataset carries **hidden ground-truth labels**, the agent pipeline can be evaluated objectively instead of by vibes.

The interesting engineering problem is in the middle: how do you generate 2,000 synthetic patients that are *internally coherent* — where the hidden diagnosis is actually inferable from the visible lifestyle, history, and self-treatment details — instead of random field soup?

## Project blueprint

```
Stage 1 — Synthetic dataset (this repo, complete)
  Pydantic schema  ->  scenario blueprints  ->  expander  ->  quality gates  ->  JSONL/CSV

Stage 2 — Local models (next)
  Ollama + quantized small models, structured JSON output, prompt design per agent role

Stage 3 — Multi-agent pipeline + evaluation
  Extractor agent      : reads the raw intake record, pulls out clinical signals
  Guideline agent (RAG): retrieves triage guideline snippets relevant to the case
  Router agent         : assigns ESI acuity level + destination department
  Evaluator            : compares agent output vs hidden ground truth (accuracy per label)
```

### Stage 1 design (what's in this repo)

**Schema first.** `src/triage/schema.py` defines `PatientIntake` — demographics, vitals, symptom detail, lifestyle, self-treatment, exposure/sexual/family history, prior episodes — plus a nested `GroundTruth` (ESI level 1–5, department, probable condition, contributing factors, red flags, rationale). Agents only ever see `agent_view()`, which strips the ground truth. Safety rules live in the schema itself (e.g. `sexual_history` must be `None` for children; BMI is computed, never stored).

**Blueprints × expander.** Hand-writing 2,000 coherent records doesn't scale, and naive LLM sampling mode-collapses. Instead:

- Each **blueprint** (`data/blueprints/*.json`) fixes one causal story — e.g. *gastritis from unsupervised NSAID use*, *tension headache from screen strain*, *holiday-heart atrial fibrillation* — including which lifestyle/background/self-treatment bundles are plausible for it, phrasing templates, and acuity-consistent vitals ranges.
- The **expander** (`src/triage/expander.py`) multiplies each blueprint into many patients with a seeded RNG: varied age/sex/body metrics, sampled bundles, filled templates. Contributing factors travel *inside* the sampled bundles, so a record's hidden causes can never contradict its visible story — coherence by construction.

**Quality gates as code.** `src/triage/quality.py` re-validates every line against the schema and enforces: no duplicate IDs, no unfilled template slots, a narrative-duplication budget, and a **grounding check** — every hidden contributing factor must be lexically traceable in the agent-visible fields, otherwise the dataset would ask agents to infer causes that were never observable.

## Dataset

`data/sample_batch.jsonl` — 2,000 records, 24 distinct conditions across 10 departments (resuscitation, emergency, cardiology, neurology, orthopedics, pediatrics, general medicine, surgery, OB-GYN, psychiatry), ESI levels 1–5 with a realistic ED pyramid. `data/sample_batch.csv` is a flattened, lossy view for spreadsheet inspection; JSONL is canonical.

The blueprint library is still growing toward a 100–150 condition target; the dataset is regenerated deterministically after each batch.

## Repo layout

```
src/triage/
  schema.py      # PatientIntake contract + hidden GroundTruth + agent_view()
  blueprint.py   # blueprint format + loader
  expander.py    # blueprints -> records (seeded, coherent)
  quality.py     # validation, grounding, distribution report
  genplan.py     # condition archetypes + arrival-mode sampling
  export_csv.py  # JSONL -> flat CSV projection
data/
  blueprints/    # hand-authored causal scenarios (JSON)
  sample_batch.jsonl / .csv   # generated dataset
```

## Reproduce

```bash
uv sync
uv run python -m triage.expander --total 2000 --seed 7 --out data/sample_batch.jsonl
uv run python -m triage.quality data/sample_batch.jsonl
uv run python -m triage.export_csv data/sample_batch.jsonl
```

Same seed, same blueprints, same 2,000 records.
