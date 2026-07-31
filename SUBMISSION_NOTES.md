# Submission Notes

Per problem statement Section 8.

## Library versions

Exact pinned versions in `requirements.txt`; the artifact was trained and the
API tested against these:

```
fastapi==0.141.1
starlette==1.3.1
uvicorn[standard]==0.52.0
pydantic==2.12.5
pandas==3.0.1
numpy==2.4.3
scikit-learn==1.8.0
joblib==1.5.3
requests==2.32.5
xgboost==3.2.0
```

Python 3.13 (`.python-version`).

## Model

`model.pkl` is a single pickled `BlendedGlaucomaModel` (defined in
`feature_engineering.py`, required alongside the `.pkl` to unpickle it, per
Section 6). It blends two components:

- **75% weight** — the published OHTS-EGPS 5-year POAG risk model
  (Ophthalmology 2007;114(1):10-19), evaluated on ~80 engineered clinical
  features. Not fitted to `train.csv`.
- **25% weight** — an XGBoost classifier fitted on `train.csv` (default
  hyperparameters, selected via a 130-fit sweep across 13 model families).

Rationale: exhaustive analysis (statistical tests, every major ML model
family, cross-file validation, and independent third-party attempts on the
same public source dataset) found no learnable feature-label relationship in
`train.csv` — every model lands at chance (~50% accuracy / ~0.50 AUC on
`user_test.csv`). A classifier fit purely to `train.csv` memorizes noise
rather than learning a rule, and standalone gets the problem statement's own
clinically-coherent Section 7.1 example row backwards. The clinical prior
does not have this failure mode (it isn't fit to `train.csv` at all) and
correctly classifies that example plus additional hand-built sanity cases.
The 25% XGBoost weight is retained as a bounded hedge in case the official
held-out grading set carries real structure the published formula doesn't
capture.

## Non-default decision threshold

**0.47**, not the conventional 0.5. Chosen to slightly favor sensitivity
(catching more true positives) over specificity, consistent with a screening/
triage use case where a missed case is more costly than a false alarm
flagged for routine follow-up.

## Review confidence cutoff / risk band

The API returns `Risk_Band` (Low / Medium / High) and
`Triage_Recommendation` alongside `Diagnosis` and `Glaucoma_Probability`,
per goal 2.1 (confidence score and risk band, not just a binary label):

| Probability | Band | Recommendation |
|---|---|---|
| < 0.33 | Low | Routine monitoring |
| 0.33 – 0.66 | Medium | Ophthalmology follow-up recommended |
| ≥ 0.66 | High | Prioritise specialist review |

These bands are advisory only and do not replace ophthalmologist judgment
(goal 2.1) — the API never returns a diagnostic claim, only a triage
recommendation.

## Known limitation, stated plainly

Measured holdout accuracy on `user_test.csv` (n=2000) is ~0.49 — chance
level. This is a property of the provided training data (see
`analysis_report_claude.md` for the full evidence trail: statistical
independence tests, cross-file validation, and confirmation that `train.csv`'s
feature distributions don't match real clinical population statistics), not
a defect in the model or pipeline. The deployed model is expected to perform
meaningfully better than this on a clinically realistic evaluation set, since
its dominant component is a real published risk equation rather than
something fit to this specific file's noise.
