#!/usr/bin/env bash
# run_pipeline.sh — convenience wrapper for the pipeline steps.
# Usage:
#   bash run_pipeline.sh steps1to4     # build master + reconstruct GT + review + (manual!) + merge
#   bash run_pipeline.sh step1         # just the consolidation step
#   bash run_pipeline.sh step2         # just GT reconstruction
#   bash run_pipeline.sh step3         # generate the review xlsx
#   bash run_pipeline.sh step4         # merge after you've filled the review
#   bash run_pipeline.sh smoketest     # Gemini on 10 dev reports (sanity check)
#   bash run_pipeline.sh fullinference # all 3 LLMs on all 1075 (takes ~1.5h, costs ~$200)
#   bash run_pipeline.sh metrics       # compute_metrics.py (not yet written)
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "ERROR: .venv not found. Run 'bash setup.sh' first."
    exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate
set -a; [ -f .env ] && source .env; set +a

DEV_XLSX="data/uploads/dev_295_validated.xlsx"
TEST_XLSX="data/uploads/test_780_validated.xlsx"

step1() {
    python pipeline/build_master_input.py \
        --dev-xlsx  "$DEV_XLSX" \
        --test-xlsx "$TEST_XLSX" \
        --output    data/outputs/reports_master.csv
}
step2() {
    python pipeline/reconstruct_ground_truth_v1.py \
        --dev-xlsx  "$DEV_XLSX" \
        --test-xlsx "$TEST_XLSX" \
        --output    data/outputs/ground_truth_v1.csv
}
step3() {
    python pipeline/re_annotate_v3_helper.py \
        --gt-csv      data/outputs/ground_truth_v1.csv \
        --reports-csv data/outputs/reports_master.csv \
        --output      data/outputs/consolidation_pneumonia_review.xlsx
    echo ""
    echo "==============================================================="
    echo "MANUAL STEP: open data/outputs/consolidation_pneumonia_review.xlsx"
    echo "in Excel/Numbers, review the REVIEW_NEEDED sheet, edit the"
    echo "USER_consolidation and USER_pneumonia columns where needed,"
    echo "save, then run: bash run_pipeline.sh step4"
    echo "==============================================================="
}
step4() {
    if [ ! -f data/outputs/consolidation_pneumonia_review.xlsx ]; then
        echo "ERROR: review xlsx not found. Run step3 first."
        exit 1
    fi
    python pipeline/merge_v3_groundtruth.py \
        --v1-gt-csv    data/outputs/ground_truth_v1.csv \
        --review-xlsx  data/outputs/consolidation_pneumonia_review.xlsx \
        --output       data/outputs/ground_truth_v3.csv
}
smoketest() {
    [ -z "${GOOGLE_API_KEY:-}" ] && { echo "GOOGLE_API_KEY not set; edit .env"; exit 1; }
    python pipeline/api_inference.py \
        --vendor gemini --input data/outputs/reports_master.csv \
        --only-split dev --limit 10 \
        --output responses/gemini_smoketest10.csv
}
fullinference() {
    [ -z "${OPENAI_API_KEY:-}" ]    && { echo "OPENAI_API_KEY not set; edit .env";    exit 1; }
    [ -z "${GOOGLE_API_KEY:-}" ]    && { echo "GOOGLE_API_KEY not set; edit .env";    exit 1; }
    [ -z "${ANTHROPIC_API_KEY:-}" ] && { echo "ANTHROPIC_API_KEY not set; edit .env"; exit 1; }
    python pipeline/api_inference.py --vendor gpt4o  --input data/outputs/reports_master.csv --output responses/gpt4o_responses.csv  --resume
    python pipeline/api_inference.py --vendor gemini --input data/outputs/reports_master.csv --output responses/gemini_responses.csv --resume
    python pipeline/api_inference.py --vendor claude --input data/outputs/reports_master.csv --output responses/claude_responses.csv --resume
}

case "${1:-}" in
    step1)         step1 ;;
    step2)         step2 ;;
    step3)         step3 ;;
    step4)         step4 ;;
    steps1to4)     step1; step2; step3 ;;
    smoketest)     smoketest ;;
    fullinference) fullinference ;;
    *)
        echo "Usage: $0 {step1|step2|step3|step4|steps1to4|smoketest|fullinference}"
        exit 1
        ;;
esac
