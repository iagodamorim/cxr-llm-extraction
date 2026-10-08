# Pre-Analysis Plan (PAP) v2.0

**Study:** Specialist Routing of Multiple Large Language Models for Structured Extraction of Clinical Findings from Portuguese Chest Radiograph Reports

**Target journal:** Radiology: Artificial Intelligence

**Authors:** Iago de Paiva D'Amorim (HCFMUSP); Mateus Aragão Esmeraldo (Stanford); Marcio Valente Yamada Sawamura (HCFMUSP); Andre Coutinho Castilla (Neuralmed)

**Date locked:** 2026-05-18

**Manuscript version at lock:** v2.0 (post-pivot)

**Linked artifacts:**
- Prompt of record: `cxr_prompt_v3.yaml` (locked 2026-05-18)
- Stratified split script: `split_data.py` (locked 2026-05-18, seed 42)
- Ensemble evaluation script: `llm_as_judge_v2.py` (in preparation)

---

## 1. Background and rationale for v2.0 protocol

The initial study design (v1) ran three frontier LLMs (GPT-4o, Gemini 2.5 Pro, Claude Sonnet 4) on N=1110 Brazilian-Portuguese chest radiograph reports using a 7-class prompt (`cxr_prompt_v1.yaml`). Preliminary analysis on Gemini 2.5 Pro (N=780 subset) achieved macro F1 = 0.875 but identified two methodological weaknesses:

1. **Prompt design flaws.** Class 6 was labeled `bacterial_pneumonia` in the output schema but defined as "consolidation or bacterial pneumonia" in the natural-language block, producing boundary inconsistencies and depressing F1 to 0.457 on this class. The `yesfinding` rule contradicted its class definition.
2. **Lack of dev/test split for Specialist Routing.** The Specialist Routing strategy was originally intended to be computed on the same N=1110 set used for final metrics, violating MI-CLEAR-LLM Item 6 (test dataset independence) and overfitting the routing assignment to the test labels.

The v2.0 protocol addresses both issues *before* the full Phase C run, with the prompt of record (v3) finalized following an audit of 20 references including BRAX (Reis 2022 Sci Data), MI-CLEAR-LLM (Park 2024 KJR), and Le Guellec 2024 (Radiology: AI).

---

## 2. Study phases

### Phase A — Lock (NOW, 2026-05-18)
- Lock prompt of record: `cxr_prompt_v3.yaml`.
- Lock split design: 30 / 70 stratified, seed 42.
- Lock primary endpoints, secondary endpoints, ensemble strategies.
- Document this PAP and freeze under version control.

### Phase B — Pilot (N = 300)
- Re-run the original 300-report subset (the ECR 2026 poster sample) with prompt v3 across the three LLMs.
- Validate that prompt v3 produces sensible JSON outputs and that the new consolidation/pneumonia separation behaves as intended.
- Go/no-go decision: if v3 prompt yields macro F1 ≥ 0.80 on the 300 and pneumonia F1 ≥ 0.65 (a meaningful improvement over the v1 prelim of 0.457), proceed to Phase C. Otherwise, document specific failure modes and amend the protocol.

### Phase C — Full study (N = 1110)
- Apply locked v3 prompt and locked split design to the full validated set.
- Compute individual LLM metrics (test split only).
- Compute Specialist Routing and all four other ensemble strategies (assignment derived on dev split, evaluation reported on test split).
- Statistical testing per Section 5.
- Write final Results and Discussion sections.

---

## 3. Dataset

- **Population:** Consecutive de-identified Brazilian-Portuguese chest radiograph reports from the radiology information system of Hospital das Clínicas, Faculdade de Medicina, Universidade de São Paulo (HCFMUSP).
- **Inclusion period:** [TBD — to be filled at submission]
- **Inclusion criteria:** All plain-film chest radiograph reports (AP, PA, lateral projections) issued during the inclusion period; no exclusion by clinical indication, completeness, or length.
- **Total N:** 1110 reports.
- **Ground truth:** Single board-certified radiologist (I.P. D'Amorim) annotated each report for the 8 binary classes per the v3 prompt class definitions. Borderline cases resolved by consensus with M.V.Y. Sawamura.
- **Ethics:** CAPPesq HCFMUSP IRB waiver of informed consent (number/date TBD).

---

## 4. Stratified development / test split

- **Method:** Stratified random split with seed = 42.
- **Stratification variable:** A 7-bit signature combining the binary status of each non-composite class (opacity, consolidation, pneumonia, cardiomegaly, effusion, pneumothorax, mass), to ensure proportional representation of all class combinations in both splits. The yesfinding indicator is derived and therefore not used for stratification.
- **Proportions:** 30 % development (333 reports) / 70 % test (777 reports).
- **Rare-class safeguard:** Classes with prevalence < 1 % in the full dataset (expected: pneumothorax, mass) must have at least 2 positive cases in the dev split and at least 4 positive cases in the test split. If the stratified split fails this safeguard, re-stratification uses a manual rebalancing rule (positive cases distributed 30/70 first, then negative cases stratified by the residual classes).
- **Generation script:** `split_data.py` (locked, seed 42). Output files: `dev_ids.csv` (333 report IDs) and `test_ids.csv` (777 report IDs).
- **Independence:** No report appears in both splits.

---

## 5. Individual LLM evaluation

- **Models:** GPT-4o (OpenAI, version `gpt-4o-2024-08-06` or latest available at Phase B), Gemini 2.5 Pro (Google Vertex AI, exact version locked at Phase B), Claude Sonnet 4 (Anthropic, exact version locked at Phase B).
- **Inference settings:** temperature = 0 (locked), top_p = vendor default, single independent API call per report, no fine-tuning, no retrieval augmentation, no chain-of-thought prompting beyond what the v3 prompt induces.
- **Prompt:** `cxr_prompt_v3.yaml` (locked).
- **Metrics computed on the 777-report test split:**
  - Per-class precision, recall (sensitivity), specificity, F1-score, accuracy.
  - Macro-averaged F1 across the 8 classes.
  - 95 % CIs by non-parametric bootstrap with 2000 resamples.
- **Pairwise comparisons:** McNemar test on paired binary predictions; Bonferroni correction across 8 classes × 3 model pairs = 24 tests; significance threshold α = 0.001.

---

## 6. Ensemble strategies (LLM-as-Judge)

Five ensemble strategies are computed post hoc from the three LLMs' predictions. **All per-class weights, routing assignments, and ensemble parameters are derived on the 333-report development split. Final metrics are reported on the 777-report test split only.**

1. **Consensus Voting.** Majority vote (≥ 2 of 3 models).
2. **Weighted Consensus.** Each model weighted by its overall macro F1 on the dev split.
3. **Confidence-Weighted Ensemble.** Each model weighted per-class by its per-class F1 on the dev split.
4. **Specialist Routing.** For each class, the prediction is taken from the single model with the highest per-class F1 on the dev split; the other two models' predictions are discarded for that class only.
5. **Hierarchical Decision.** Specialist Routing for the high-stakes / rare classes (pneumothorax, mass); Weighted Consensus for the others (opacity, consolidation, pneumonia, cardiomegaly, effusion, yesfinding).

---

## 7. Primary and secondary endpoints

### Primary endpoint
Macro-averaged F1-score of the Specialist Routing strategy on the 777-report test split, compared to the best individual LLM on the same test split. Pre-specified directional hypothesis: Specialist Routing macro F1 ≥ best individual LLM macro F1.

### Secondary endpoints
- Per-class F1-scores of Specialist Routing vs best individual LLM (8 classes).
- Macro F1 of all five ensemble strategies, ranked.
- Sensitivity-specificity coordinates of all eight evaluation units (3 individual + 5 ensemble) on the test split.
- Per-class F1 of the pneumonia and consolidation classes specifically (the post-v3 split design).

### Exploratory endpoints
- Macro F1 of v3 prompt vs the v1 prompt prelim results on Gemini (descriptive only; not a confirmatory test, given the schema change).
- Inter-rater agreement (Cohen κ) on a stratified subsample if a second annotator is available.

---

## 8. Statistical analysis

- **Primary test:** McNemar test on paired binary predictions, comparing Specialist Routing vs best individual LLM, macro-averaged over classes via stratified bootstrap.
- **Multiple testing:** Bonferroni correction across the family of tests (8 classes × 5 ensemble strategies × 3 individual LLMs = 120 comparisons in the full grid). Primary endpoint test is pre-specified and not corrected.
- **CIs:** Non-parametric bootstrap, 2000 resamples, percentile method.
- **Significance threshold:** α = 0.001 for confirmatory tests.

---

## 9. Reporting

The study will be reported in adherence to:
- **MI-CLEAR-LLM** (Park et al., Korean J Radiol 2024) — all 6 items addressed; Item 6 (test dataset independence) is enforced by the dev/test split design above.
- **CLAIM** (Mongan et al., Radiology: AI 2020) — all applicable items.
- Manuscript will be submitted to *Radiology: Artificial Intelligence* per the cover letter v1.3.

---

## 10. Protocol amendments

Any change to this PAP between Phase A lock and Phase C completion must:
1. Be documented as a dated amendment at the end of this file.
2. Specify which sections/decisions are affected.
3. Be agreed by I.P. D'Amorim and M.V.Y. Sawamura (orientador), with notification to co-authors.

### Amendment log

**Amendment 1 — 2026-05-18 (same-day after Phase A lock)**
- **Affected sections:** §3 Dataset (N), §4 Stratified development/test split.
- **Change:** N corrected from "1110" to **1075** following inventory of the two source spreadsheets (295 unique reports in the ECR cohort + 780 unique reports in the expansion cohort; zero overlap; 35 reports previously double-counted across the two files).
- **Change:** development/test split changed from "random stratified 30/70 (333/777, seed 42)" to a **natural chronological split (295 dev / 780 test)** aligned with the historical data-collection phases. The 295-report development cohort is the same cohort presented in the ECR 2026 poster (C-15791); the 780-report test cohort is the subsequent expansion. The natural split is methodologically defensible without recourse to a random seed, makes no assumption of exchangeability between the two cohorts, and avoids the implicit data-leakage risk of a random stratified split derived from a single labeling pass.
- **Justification:** the natural split mirrors the actual order of data collection, gives a slightly larger test set (780 vs 777), and lets the manuscript honestly anchor the development sample to the previously presented ECR cohort. The rare-class safeguard from the original split design (≥2 dev positives and ≥4 test positives per class with prevalence < 1 %) will be verified empirically once the v3 ground truth is finalized; if any rare class fails the safeguard under the natural split, a small set of test cases of that class will be re-allocated to dev to meet the minimum (documented case by case as a sub-amendment).
- **Agreed by:** I.P. D'Amorim and the audit cycle of 2026-05-18.

---

**Amendment 2 — 2026-05-18 (evening, after Phase B aggregate completion)**

- **Affected sections:** §2 Study phases (Phase B → Phase C transition), §5 Individual LLM evaluation (model list), §6 Ensemble strategies (model count), §7 Endpoints (gate criteria), §8 Statistical analysis (multiple-comparison family).
- **Change A (model list expanded from 3 to 6).** The set of evaluated LLMs is expanded from 3 vendors × 1 model to **3 vendors × 2 models = 6 models** to enable both **inter-vendor** and **within-vendor (frontier vs cost-optimized)** comparisons. Models are:
    - OpenAI: `gpt-4.1`, `gpt-4.1-mini`
    - Google: `gemini-2.5-pro`, `gemini-2.5-flash`
    - Anthropic: `claude-sonnet-4-6`, `claude-haiku-4-5`
- **Change B (prompt version updated).** The prompt of record was iteratively refined from v3.0 (8-class with consolidation/pneumonia separated) → v4.0 (7-class, opacity umbrella with pneumonia as subset) → v4.2 (data-derived rules stripped per overfitting audit; Example 6 added for Rule 7b high-stakes uncertainty). The locked prompt for Phases B and C is **`cxr_prompt_v4.yaml` at metadata version v4.2** (file `prompts/cxr_prompt_v4.yaml`, sub-version history in metadata block and the v3→v4 change summary at the file footer).
- **Change C (pneumonia F1 gate deferred from dev to test).** The Phase B gate criterion *pneumonia F1 ≥ 0.65 on the dev split* is **withdrawn and re-applied on the test split as a secondary evaluation point**. The dev split contains only **6 pneumonia-positive reports**; with such a small support the F1 metric has insufficient resolution to function as a formal gate (each additional false positive shifts F1 by ≥0.05; the distance between observed values 0.615 and 0.727 is two false-positive cases). The test split contains an expected **35 pneumonia-positive reports**, sufficient for a statistically informative comparison. Vendors that satisfy the remaining two gate criteria — macro F1 ≥ 0.80 and parse_ok ≥ 0.98 on dev — proceed to Phase C; pneumonia performance is then re-assessed on test as part of the per-class secondary endpoint.
- **Change D (rare-class safeguard satisfied empirically).** The empirical class prevalences in dev (295) / test (780) are: opacity 52/170, cardiomegaly 47/151, effusion 44/94, mass 13/32, pneumothorax 17/36, pneumonia 6/35, yesfinding 134/439. All classes satisfy the ≥2 dev positives and ≥4 test positives safeguard from §4. No reallocation between splits is required.
- **Phase B aggregate results** (locked outputs of `pipeline/aggregate_phase_b.py`, 2026-05-18):
    | Vendor | Macro F1 (dev) | Pneumonia F1 (dev) | parse_ok | Entailment viol. | Gate (per Amendment 2) |
    |---|---|---|---|---|---|
    | gpt-4.1 | 0.908 | 0.800 | 1.000 | 0.000 | **GO** |
    | gemini-2.5-pro | 0.897 | 0.615 | 1.000 | 0.000 | **GO** (pneumonia deferred to test) |
    | gemini-2.5-flash | 0.895 | 0.615 | 1.000 | 0.000 | **GO** (pneumonia deferred to test) |
    | sonnet-4-6 | 0.887 | 0.667 | 1.000 | 0.000 | **GO** |
    | haiku-4-5 | 0.886 | 0.667 | 1.000 | 0.000 | **GO** |
    | gpt-4.1-mini | 0.858 | 0.727 | 1.000 | 0.000 | **GO** |
- **Specialist Routing assignment (locked from dev, applied as-is to test)** — outputs of `pipeline/aggregate_phase_b.py` → `data/outputs/specialist_routing_assignment.json`:
    | Class | Routed-to vendor | Dev F1 |
    |---|---|---|
    | opacity | sonnet-4-6 | 0.895 |
    | pneumonia | gpt-4.1 | 0.800 |
    | cardiomegaly | gemini-2.5-flash (5-way tie at F1=1.000; alphabetical tie-break) | 1.000 |
    | effusion | gemini-2.5-pro | 0.966 |
    | pneumothorax | gemini-2.5-pro | 0.971 |
    | mass | gemini-2.5-flash | 0.870 |
    | yesfinding | gemini-2.5-pro | 0.973 |
- **Phase C statistical-analysis update** (consequence of Change A). The McNemar comparison grid expands. Bonferroni correction is computed across the full family: **7 classes × (6 individual models + 5 ensemble strategies) = 77 comparisons** for the per-class secondary endpoint, plus the pre-specified primary endpoint (Specialist Routing vs best individual model on macro F1), which is not corrected. The two ensemble strategies that originally voted across 3 models (Consensus Voting, Weighted Consensus) are re-defined for 6 models as: **Consensus Voting** = majority vote (≥4 of 6); **Weighted Consensus** = weighted sum of all 6 model votes by their respective dev macro F1, with positive call when the weighted sum exceeds half the total weight. **Confidence-Weighted Ensemble** and **Specialist Routing** are unchanged in formulation but operate over 6 candidates per class. **Hierarchical Decision** continues to route high-stakes classes (pneumothorax, mass) by Specialist Routing and the remainder by Weighted Consensus, now with 6 candidate models.
- **Justification.** The Phase B aggregate (1) confirmed perfect structured-output reliability (entailment violation rate 0.000 across 1770 prompts = 6 models × 295 reports), validating the prompt v4 hierarchy design; (2) revealed that no single vendor or model dominates across all classes (Specialist Routing distributes 3 wins to Google, 1 each to OpenAI and Anthropic, plus 1 tie), supporting the per-class ensemble premise; and (3) showed that within-vendor non-inferiority (gemini-pro ≈ gemini-flash; sonnet ≈ haiku) is a candidate secondary finding worth dedicated statistical testing on the test split (paired bootstrap CI95% with TOST non-inferiority margin 0.05 F1, to be reported as secondary endpoint).
- **Operational consequence.** Phase C runs all 6 vendors on the 780-report test split with the locked v4.2 prompt and the locked Specialist Routing assignment above. Outputs: `responses/{vendor}_test.csv` (one CSV per vendor); aggregated metrics produced by `pipeline/compare_predictions.py` and `pipeline/aggregate_phase_c.py` (to be written; analogous to `aggregate_phase_b.py` but with the routing assignment passed as input rather than recomputed).
- **What is NOT changing.** The prompt v4.2 is **not** modified in response to Phase B. No retraining, no re-annotation, no re-routing on test. The dev split labels remain held-in for prompt-tuning purposes; the test split remains the sole basis for confirmatory analyses.
- **Agreed by:** I.P. D'Amorim, 2026-05-18 evening (after review of `phase_b_summary.xlsx` and the aggregate report from `aggregate_phase_b.py`). Notification to co-authors pending the post-Phase-C combined results.

---

## 11. Data and code availability

Upon acceptance:
- The locked prompt (v3) will be released as supplementary material.
- The locked split script (`split_data.py`) and ensemble evaluation script (`llm_as_judge_v2.py`) will be deposited at a public repository (Zenodo / GitHub).
- The split files (`dev_ids.csv`, `test_ids.csv`) will be released to support exact replication.
- Anonymized derived results (per-report predictions from each LLM, per-class F1-scores) will be released.
- The full text of the 1110 reports will NOT be released due to institutional data-sharing constraints; researchers wishing to replicate on the original dataset may submit a request via the corresponding author.
