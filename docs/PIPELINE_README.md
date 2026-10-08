# Pipeline — Stanford LLM CXR study (PAP v2.1)

Scripts to run the full data-processing and inference pipeline for the
v3 study. Designed to be run by **you** with your own API keys (so we
don't bother your boss).

All scripts are stand-alone, parameterized, idempotent (`--resume`
flag for inference), and produce one canonical artifact each.

---

## Setup (once)

```bash
pip install pandas openpyxl pyyaml tqdm openai google-genai anthropic scikit-learn
```

Set your API keys (one per shell session, or add to your shell rc):

```bash
export OPENAI_API_KEY="sk-..."        # for GPT-4o
export GOOGLE_API_KEY="AIza..."       # for Gemini 2.5 Pro
export ANTHROPIC_API_KEY="sk-ant-..." # for Claude Sonnet 4.5
```

---

## Pipeline order

### Step 1 — Build the master input CSV (1075 reports, dev/test labeled)

Consolidates the 295 reports of the ECR cohort + the 780 reports of the
expansion cohort into one CSV with the natural dev/test split.

```bash
cd /path/to/Stanford
python pipeline/build_master_input.py \
  --dev-xlsx  "493747c4-...gemini_extraction_validation - validado.xlsx" \
  --test-xlsx "68dfe258-...validacao_exportada OK.xlsx" \
  --output    reports_master.csv
```

Output: `reports_master.csv` with columns `accession_number, report, split,
source_file`.

---

### Step 2 — Reconstruct v1 ground truth from the spreadsheets

The existing spreadsheets contain the Gemini extraction (v1 prompt) plus
your FP/FN annotations. This script applies `GT = (extracted - FP) + FN`
to recover the true v1 7-class ground truth for all 1075 reports.

```bash
python pipeline/reconstruct_ground_truth_v1.py \
  --dev-xlsx  "493747c4-...gemini_extraction_validation - validado.xlsx" \
  --test-xlsx "68dfe258-...validacao_exportada OK.xlsx" \
  --output    ground_truth_v1.csv
```

Output: `ground_truth_v1.csv` — one row per report, columns for the 7 v1
classes + `abnormalities_found` + `annotation_status`.

---

### Step 3 — Generate the consolidation/pneumonia review spreadsheet

Pre-annotates the v3-specific split (consolidation vs pneumonia) via
Portuguese keyword matching in the report text. You only review the
rows where at least one keyword matched (~150-250 reports estimated).

```bash
python pipeline/re_annotate_v3_helper.py \
  --gt-csv      ground_truth_v1.csv \
  --reports-csv reports_master.csv \
  --output      consolidation_pneumonia_review.xlsx
```

Output: `consolidation_pneumonia_review.xlsx` with two sheets:
- `REVIEW_NEEDED`: open in Excel, the `USER_consolidation` and
  `USER_pneumonia` columns are pre-filled with suggestions (highlighted
  amber); edit them where wrong, save the file.
- `AUTO_BOTH_ZERO`: ~825 reports auto-assigned cons=pneu=0; no review
  needed but you can spot-check.

---

### Step 4 — Merge the reviewed split into the final v3 ground truth

After you've filled in the `USER_*` columns in Step 3:

```bash
python pipeline/merge_v3_groundtruth.py \
  --v1-gt-csv    ground_truth_v1.csv \
  --review-xlsx  consolidation_pneumonia_review.xlsx \
  --output       ground_truth_v3.csv
```

Output: `ground_truth_v3.csv` — final 8-class GT for 1075 reports.

---

### Step 5 — Run the 3 LLMs with prompt v3 on all 1075 reports

One run per vendor. Each saves to its own CSV with incremental writes
(use `--resume` if the run drops mid-way).

```bash
python pipeline/api_inference.py --vendor gpt4o  --input reports_master.csv
python pipeline/api_inference.py --vendor gemini --input reports_master.csv
python pipeline/api_inference.py --vendor claude --input reports_master.csv
```

Outputs:
- `gpt4o_responses.csv`
- `gemini_responses.csv`
- `claude_responses.csv`

Each has one row per (report, vendor) with raw response, parsed flag,
latency, and the 8 binary class predictions.

**Cost estimate (rough):** ~$50-100 per vendor × 3 vendors = ~$150-300.
Time: ~30 min per vendor if parallel; ~1.5h per vendor if rate-limited
sequential.

To smoke-test first (run on just 10 dev reports):
```bash
python pipeline/api_inference.py --vendor gemini --input reports_master.csv \
       --only-split dev --limit 10 --output gemini_test10.csv
```

---

### Step 6 — Compute individual + ensemble metrics (built later)

The next script (`compute_metrics.py`, to be written after Step 5
completes) will take the 3 vendor response CSVs and the v3 ground truth
to compute:

- Per-class precision, recall, specificity, F1 for each LLM (test split)
- Confidence-weighted, consensus, weighted, hierarchical, **Specialist
  Routing** ensembles (assignments computed on **dev split only**, then
  applied to test)
- McNemar tests + Bonferroni correction
- All numbers needed to populate the manuscript Results section

---

## Troubleshooting

- **`openpyxl` complains about read_only mode and edited xlsx:** open the
  file in Excel, save (no changes needed), retry. Excel sometimes leaves
  files in a state that read-only mode can't open.
- **API errors / 429 rate limits:** the script retries with exponential
  backoff (up to 5 attempts). For sustained rate limits, lower
  concurrency or wait and use `--resume`.
- **JSON parse errors:** check `parse_ok=False` rows in the vendor CSV;
  the `raw_response` column has the model's actual output. Most parse
  failures are recoverable by re-running just that row.

---

## File layout in `/Stanford/`

```
Stanford/
├── cxr_prompt_v3.yaml                # locked prompt (input to Step 5)
├── PAP_v2.0.md                       # pre-analysis plan
├── reports_master.csv                # output of Step 1 (input to all next)
├── ground_truth_v1.csv               # output of Step 2
├── consolidation_pneumonia_review.xlsx  # output of Step 3, input to Step 4
├── ground_truth_v3.csv               # output of Step 4 (final GT)
├── gpt4o_responses.csv               # output of Step 5
├── gemini_responses.csv              # output of Step 5
├── claude_responses.csv              # output of Step 5
└── pipeline/
    ├── README.md                     # this file
    ├── build_master_input.py
    ├── reconstruct_ground_truth_v1.py
    ├── re_annotate_v3_helper.py
    ├── api_inference.py
    └── merge_v3_groundtruth.py
```
