# Annotation-Free Structured Extraction of Chest Radiograph Findings with Commercial LLMs

Code, prompt and aggregated results for the study:

> D'Amorim IP, et al. **Annotation-Free Structured Extraction of Clinical Findings from Chest Radiograph Reports with Commercially Available Large Language Models: A Multi-Vendor, Cost-Aware Evaluation.** *Clinical Imaging*, under review (submitted August 2026).

Institute of Radiology (InRad), Hospital das Clínicas, University of São Paulo Medical School (HCFMUSP), Brazil.

## What the study does

Six commercially available LLMs from three vendors extract seven structured findings from Brazilian Portuguese chest radiograph reports, with **no task-specific training, fine-tuning or annotation**. A single locked prompt is used for every model, at temperature 0.

| | |
|---|---|
| Reports | 1,075 (295 development, 780 test) |
| Findings | opacity, pneumonia, cardiomegaly, effusion, pneumothorax, mass, any finding (`yesfinding`) |
| Models | Gemini 2.5 Flash, Gemini 2.5 Pro, GPT-4.1, GPT-4.1 mini, Claude Haiku 4.5, Claude Sonnet 4.6 |
| Reference standard | Labels assigned from report text by a radiologist, with adjudication of ambiguous cases by a more experienced radiologist |
| Analyses | per-finding F1, sensitivity, specificity, PPV, NPV with bootstrap CIs; McNemar tests with FDR correction; non-inferiority (TOST); multi-model ensembles; cost per 1,000 reports |

## Repository layout

```
prompts/
  cxr_prompt_v4.yaml        prompt of record (v4.2), released verbatim
  archive/                  earlier prompt version (v3), for transparency
pipeline/
  api_inference.py          runs one model over the report set (OpenAI, Google, Anthropic APIs)
  build_*.py, merge_*.py    reference-standard construction and review tooling
  compute_ensembles.py      multi-model ensemble strategies
  bootstrap_ci.py, sens_spec_ci.py, bca_sensitivity.py
  mcnemar_grid.py, fdr_sensitivity.py, paired_delta_analysis.py, tost_hd_vs_sonnet.py
  routing_stability.py      stability of per-finding model selection
  true_token_cost.py        per-vendor token count and cost
data/outputs/               aggregated results only (see below)
docs/                       pre-analysis plan and pipeline notes
run_*.sh                    convenience wrappers
```

## Data availability

The radiology reports are clinical records of HCFMUSP patients and **are not included in this repository**. The study was approved by the institutional ethics committee (CAPPesq). Only aggregated, non-identifiable outputs are released: per-model metrics, confidence intervals, statistical tests, ensemble summaries and cost tables. Per-report inputs, model responses and reference labels are excluded by `.gitignore`.

To run the pipeline on your own data, provide a CSV with one report per row (`accession_number`, `split`, `report`) in place of `data/outputs/reports_master.csv`.

## Running

```bash
bash setup.sh                      # creates .venv, installs requirements, copies .env.example to .env
$EDITOR .env                       # add OPENAI_API_KEY, GOOGLE_API_KEY, ANTHROPIC_API_KEY
bash run_dev_all_models.sh         # six models on the development split
bash run_test_all_models.sh        # six models on the test split
```

Single model:

```bash
python pipeline/api_inference.py --vendor gemini --model-id gemini-2.5-flash \
    --input data/outputs/reports_master.csv --only-split test
```

## Notes

- **Costs.** `true_token_cost.csv` reports costs from each vendor's own tokenizer and the assembled prompt, and supersedes the earlier single-tokenizer estimate (`pipeline/cost_estimation.py`).
- **Overfitting checks.** Files marked `overfit` or `DO_NOT_USE` derive model selection on the test split. They are kept as a sensitivity analysis and are not used for any reported result.
- **Reporting standard.** The prompt is released verbatim following MI-CLEAR-LLM.
- The study began under an earlier working title ("Specialist Routing of Multiple LLMs..."); some file headers keep it for provenance.
- The manuscript is authoritative for all reported numbers.

## Citation

Citation details will be added on publication.
