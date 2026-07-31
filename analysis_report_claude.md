# Glaucoma Risk Prediction — Analysis Report (Claude session)

## 1. The Problem

**Hackathon task:** AI-Powered Glaucoma Risk Prediction (Difficulty: Hard). Build an ML model that
predicts `Diagnosis` (Glaucoma / No Glaucoma) from a patient's clinical/ophthalmic exam profile,
package it as `model.pkl` (full pipeline: imputation + encoding + scaling + model, all embedded — no
external preprocessing script), and deploy it behind an HTTP inference API.

**API contract (Section 7 of the problem statement):**
- `POST /predict` with body `{"inputFile": "<https url to a csv>"}`
- Service downloads the CSV, scores every row **in order**, never drops rows (including rows with
  missing optional fields), and returns:
  ```json
  {
    "data": {"outputFile": "<url to output csv>", "timestamp": "..."},
    "message": "Predictions generated successfully",
    "status": "success"
  }
  ```
- Output CSV = every input column, plus `Diagnosis` as `"true"`/`"false"` text (`true` = Glaucoma).
- Input CSV columns may be a **subset**, in **any order** — pipeline must tolerate missing columns
  entirely, not just missing values within present columns.
- Hard constraints: `Patient ID` must never be used as a feature; `Diagnosis` must never be used as
  an inference input even if present in the caller's CSV; all feature engineering/imputation/encoding/
  scaling must be embedded inside the saved pipeline object (computed from raw columns at
  transform-time, not precomputed offline).

**Scoring (from `AI_Model_Evaluation_Rubric.docx`, found later in `initial_files/`) — this matters a lot:**

| Section | Points |
|---|---|
| Model Performance (vs. official held-out ground truth, threshold-match, not leaderboard) | **30** |
| Model Submission (.pkl loads) | 10 |
| Endpoint Deployment & API (reachability, format, latency) | 20 |
| Exception Handling & Validation (missing fields, bad input, edge cases) | 20 |
| Response Stability & Contract Compliance (consistent, schema-correct) | 10 |
| Endpoint Accessibility (uptime during evaluation) | 10 |

**Only 30/100 points depend on accuracy**, and even that is a threshold-match against difficulty
level, not a continuous leaderboard. **70/100 is pure engineering** (deployment, robustness,
compliance, uptime). No document restricts model type to classical ML — Section 5's list of
suggested models (Logistic Regression / Random Forest / GBM) is explicitly a *suggestion*, not a
requirement. Nothing prohibits deep learning; nothing requires it either.

## 2. Dataset Description

- `train.csv`: 6,000 rows, 17 columns. `Diagnosis` target: 3,026 Glaucoma / 2,974 No Glaucoma (near
  perfectly balanced, no class-imbalance problem).
- `user_test.csv`: 2,000 rows, same schema including `Diagnosis` and `Glaucoma Type` (an org-provided
  local validation file — the real held-out grading file is separate and not shared).
- Columns: `Patient ID`, `Age`, `Gender`, `Visual Acuity Measurements` (mixed Snellen `"20/40"` /
  LogMAR `"0.1"` notation), `Intraocular Pressure (IOP)`, `Cup-to-Disc Ratio (CDR)`, `Family History`,
  `Medical History`, `Medication Usage` (free-text comma list), `Visual Field Test Results` (free text
  embedding `Sensitivity: X, Specificity: Y`), `Optical Coherence Tomography (OCT) Results` (free text
  embedding RNFL/GCC/Retinal Volume/Macular Thickness), `Pachymetry`, `Cataract Status`,
  `Angle Closure Status`, `Visual Symptoms` (free-text comma list of 3 items, often with a duplicate),
  `Diagnosis` (target), `Glaucoma Type` (present as an *input* feature per Section 7.1's sample table,
  even for "No Glaucoma" rows — unusual but explicitly spec'd).

### Data quality audit (train.csv)
- 0 duplicate `Patient ID`s, 0 misaligned rows, no whitespace/casing inconsistencies.
- 100% regex-parse success rate on both free-text numeric blobs (`Visual Field Test Results`, `OCT
  Results`) — nothing silently failing to parse.
- Numeric ranges are tight and clean, no outliers/impossible values: Age 18–90, IOP 10–25 mmHg, CDR
  0.3–0.8, Pachymetry 500–600 µm.
- Missingness (true MCAR-looking, matches the problem statement's "not every patient has every test"):
  `Medical History` 26.1% missing (1,567 rows), `Medication Usage` 12.2% missing (732 rows), **201 rows
  missing both**. No other column has missing values.

**Conclusion: the data is clean. There is nothing to "fix" in the mechanical sense** (no corruption,
no parse failures, no misalignment). The open question is entirely about *signal*, not data quality.

## 3. Exhaustive Signal Analysis — Everything Tried So Far

Goal: determine whether `Diagnosis` is predictable from the other 15 columns at all, and if so, how.

### 3.1 Baseline ML (5-fold stratified CV, ROC-AUC)
| Model | CV ROC-AUC |
|---|---|
| Logistic Regression (ordinal-encoded features) | 0.473 |
| Random Forest (max_depth=8) | 0.491 |
| LightGBM (default) | 0.500 |
| **Unlimited-depth single Decision Tree**, ordinal features | 0.4995 |
| **Unlimited-depth single Decision Tree**, full 47-feature expansion (individual medication flags, individual symptom flags, all parsed numerics, all one-hot categoricals) | 0.498 |
| Gradient Boosting on same 47-feature expansion | 0.494 |

The unlimited-depth tree result is the most important one: an unconstrained tree can represent *any*
function (including arbitrary interactions/XOR-type rules) of its inputs. Its failure to beat chance,
on both the ordinal-encoded and the fully-expanded feature sets, is strong evidence that **no function
— linear or nonlinear — of these features predicts the label.**

Note on overfitting risk found along the way: LightGBM with default settings hit **0.999 in-sample
AUC** on the SAME feature set that CVs at ~0.50. Root cause: one-hot-encoding the raw `Medication
Usage` string (2,724 near-unique values across 6,000 rows) lets the model memorize row-identity
instead of learning anything general. Any high-cardinality free-text field must be decomposed (e.g.
multi-hot per individual medication + count), never one-hot'd raw, or CV numbers will be badly
misleading.

### 3.2 Cross-file generalization test (the most decisive test run)
Trained a Random Forest (500 trees, both depth-capped and unlimited) on the **entirety of train.csv**,
predicted on **user_test.csv's real Diagnosis labels** — a genuinely disjoint file (0 Patient ID
overlap between the two files).

**Result: 0.5016 ROC-AUC, 50.2% accuracy.** This rules out CV-fold flukes; it's an honest,
fully-held-out check and it lands exactly on chance.

### 3.3 Formal statistical inference ("Excel regression report" equivalent)
- **ANOVA** (numeric features, Glaucoma vs No-Glaucoma group means): every p-value in range
  0.13–0.94 (Age p=0.92, IOP p=0.54, CDR p=0.67, Pachymetry p=0.13, VF Sensitivity/Specificity
  p=0.94/0.57, OCT RNFL/GCC/RetVol/Macular p=0.89/0.34/0.32/0.59). None significant.
- **Chi-square** (categorical features vs Diagnosis): Gender p=0.46, Family History p=1.00, Medical
  History p=0.55, Cataract Status p=0.17, Angle Closure Status p=0.69, Glaucoma Type p=0.79, Visual
  Acuity p=0.87. None significant.
- **Full multivariable logistic regression** (statsmodels Logit, standardized coefficients, all 11
  numeric/parsed features): **Pseudo R² = 0.0006, LLR p-value = 0.87** (the entire fitted model is
  statistically indistinguishable from the null/intercept-only model). No individual coefficient
  significant.

### 3.4 Domain-specific clinical formulas (not guesses — pulled from literature, see §4)
- Classic clinical rule `IOP > 21 & CDR > 0.6` → 49.75% accuracy (worse than a coin flip).
- Grid search over all `IOP`/`CDR` threshold combinations → best combo still only 50.8% accuracy.
- **Ehlers-corrected IOP** (`IOP − [5.0×(Pachymetry/1000−0.520)]/0.070`, the real clinical formula for
  correcting IOP by corneal thickness) → correlation with Diagnosis: 0.0025. Included in the full
  logistic regression above; not significant.

### 3.5 Hidden-encoding / hash-style checks (holdout-validated stumps, 50/50 split to avoid
overfitting false positives)
- `Patient ID` mod {2,3,4,5,7,10,11,13}, digit sum, last digit → all ~49–51% holdout accuracy.
- Row order / row index parity → 50.5%.
- Last-digit-hash of IOP/CDR/Pachymetry (×100) → 47–51%.
- `Visual Symptoms` duplicate-position pattern (e.g. `"Blurred vision, Vomiting, Blurred vision"` —
  which slot repeats) and duplicated-symptom identity → 49.8% / 48.9%.
- `Medication Usage` count and count-parity → 50.7% / 50.4%.

All null. Nothing found across correlation, ANOVA/chi-square/logistic regression, tree-based ML,
cross-file validation, or exotic hash-style checks.

### 3.6 Cross-field relationship check (testing a specific hypothesis: can `Medical History` and
`Medication Usage` predict each other, to justify cross-imputing the missing 26%/12%?)
Chi-square, per medication, conditioned on Medical History: Amoxicillin p=0.91, Aspirin p=0.98,
Atorvastatin p=0.35, Ibuprofen p=0.99, Lisinopril p=0.56, Metformin p=0.65, Omeprazole p=0.56. Average
medication count is ~3.95 regardless of condition (Diabetes 3.97, Glaucoma-in-family 3.95,
Hypertension 3.92 — statistically identical). **These two fields are independent of each other too**,
not just independent of Diagnosis. Also checked missingness-as-signal (MNAR): missing-Medical-History
rate is 51.6% Glaucoma vs 50.0% No-Glaucoma (p=0.31); missing-Medication p=0.85; missing-both p=1.00.
No informative missingness pattern either.

**Conclusion: cross-imputing these two fields from each other would only reproduce the marginal
distribution — no accuracy benefit, and it's not worth the added pipeline complexity/failure surface
for the Exception Handling rubric section.**

## 4. Literature Comparison (why this result is notable)

Web research on real-world glaucoma ML studies (2024–2025 papers, sources below) shows the *same
feature types we have* are normally strongly predictive:
- Real studies using IOP + RNFL (quadrant-specific) + PSD + age report **76–98% accuracy, AUC up to
  0.945–0.99** (e.g. XGBoost on 5 clinical features: accuracy 94.7%, AUC 0.945; logistic regression on
  biomechanical data: accuracy 0.98, AUC 0.99).
- Random Forest models on IOP/CDR/RNFL commonly reach AUC 0.88–0.94.
- This matches the problem statement's own stated target: "approximately 85–95% for a solid classical
  ML solution."

Sources:
- https://link.springer.com/article/10.1186/s42490-025-00095-3 (Applications of ML in glaucoma
  diagnosis based on tabular data: a systematic review)
- https://pmc.ncbi.nlm.nih.gov/articles/PMC8001225/ (Explainable ML model for glaucoma diagnosis)
- https://pmc.ncbi.nlm.nih.gov/articles/PMC7769138/ and
  https://www.eyedocs.co.uk/ophthalmology-articles/glaucoma/769-iop-and-corneal-thickness.html
  (Ehlers IOP-correction formula)

**The gap between "what these features should be able to do" (per literature) and "what they actually
do in train.csv" (chance, verified 6+ independent ways) is the central finding of this analysis.** It
strongly suggests `train.csv`'s `Diagnosis` label was generated independently of the feature columns
(a dataset-construction artifact), rather than this being a case of "signal exists but is hard to
find."

## 5. Open Question We Want a Second Opinion On

Given:
1. train.csv shows zero learnable signal by every method tried (statistical + ML + cross-file), while
   the literature says these exact feature types are normally highly predictive.
2. We cannot see or test against the actual official held-out grading dataset — it's possible (a) it
   was generated by the same broken process as train.csv (in which case ~50% is everyone's ceiling and
   the 30 accuracy points are effectively a wash across all participants), or (b) it's genuinely
   different/properly generated (in which case a well-engineered pipeline could still score well even
   though train.csv itself is uninformative for tuning against).
3. Only 30/100 rubric points depend on accuracy; 70/100 are engineering (deployment, robustness,
   contract compliance, uptime) — fully within our control regardless of (2).

**What we want recommended:**
- Is there a smarter data/statistical angle we haven't tried to extract signal from train.csv, or is
  chance genuinely the ceiling on this specific file?
- Given the uncertainty about the held-out set, what's the optimal modeling strategy — e.g., invest in
  a richly feature-engineered but conservatively regularized model (hedge: works if real signal exists
  in the true eval set, doesn't overfit noise if it doesn't) vs. a minimal fast baseline plus maximum
  investment in the 70 engineering points?
- Any preprocessing/feature-engineering ideas beyond what's listed in §3 worth testing before
  finalizing the pipeline?

## 6. Preprocessing Plan Drafted So Far (not yet finalized/implemented)

All steps embedded inside a single `sklearn.Pipeline` → `model.pkl` (no external preprocessing script,
per the hard constraint in §1):

1. **Feature-engineering transformer** (custom, fit/transform on raw columns): reindexes incoming
   DataFrame to the expected schema (tolerates missing/extra/reordered columns, per the "columns may
   be a subset" contract requirement); normalizes `Visual Acuity Measurements` (Snellen fraction and
   LogMAR) to one comparable decimal-acuity scale; regex-parses `Visual Field Test Results` →
   Sensitivity/Specificity floats; regex-parses `OCT Results` → RNFL/GCC/Retinal Volume/Macular
   Thickness floats; computes Ehlers-corrected IOP from IOP + Pachymetry; decomposes `Medication
   Usage` into per-drug multi-hot flags (7 known drugs) + a count (never one-hot the raw combo
   string — see §3.1 leakage note); decomposes `Visual Symptoms` similarly into per-symptom multi-hot
   flags (8 known symptoms) + a count; drops `Patient ID` and `Diagnosis` unconditionally.
2. **ColumnTransformer**: numeric branch = median-impute + StandardScaler; categorical branch =
   constant-fill (explicit `"Not Recorded"` / `"Missing"` category — not mode-impute, since e.g.
   absence of a recorded comorbidity is a different thing than "has hypertension") + OneHotEncoder
   with `handle_unknown="ignore"` (robustness against unseen categories in the real held-out set).
3. **Classifier**: candidates compared via honest 5-fold CV (not in-sample fit) — L2-regularized
   Logistic Regression with `class_weight="balanced"` as the leading candidate (fast, stable, resistant
   to the memorization failure mode observed with LightGBM/unconstrained trees in §3.1), against a
   depth-capped Random Forest / LightGBM challenger; select by CV ROC-AUC, not in-sample accuracy.

## 7. Files in `initial_files/`
- `problem_statment.docx` — full spec (Sections 1–9: overview, dataset description, API contract,
  submission requirements, evaluation criteria).
- `AI_Model_Evaluation_Rubric.docx` — scoring breakdown (see table in §1).
- `train.csv` — 6,000 labeled rows, described in §2.
- `user_test.csv` — 2,000 rows, same schema, org-provided local validation file (has real Diagnosis
  labels, used for the cross-file test in §3.2 — this is *not* the official grading set).
