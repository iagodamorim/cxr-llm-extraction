"""
build_master_input.py
=====================

Consolidates the 1075 reports from the two source spreadsheets into a
single master CSV with the natural dev/test split labels.

INPUT (default paths, override with flags):
  - --dev-xlsx  : "493747c4-...gemini_extraction_validation - validado.xlsx"
                  (295 unique reports, ECR poster cohort, marked as DEV)
  - --test-xlsx : "68dfe258-...validacao_exportada OK.xlsx"
                  (780 reports, the expansion cohort, marked as TEST)

OUTPUT:
  reports_master.csv with columns:
    accession_number  -- unique report ID (format MV_<digits>)
    report            -- raw Brazilian Portuguese report text
    split             -- "dev" or "test"
    source_file       -- short tag for traceability

Author: I.P. D'Amorim et al. PAP v2.1 (2026-05-18 amendment, natural split).
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import openpyxl


def read_reports_from_xlsx(path: Path, sheet_name: str, split_label: str, source_tag: str) -> pd.DataFrame:
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    if sheet_name not in wb.sheetnames:
        sys.exit(f"ERROR: sheet '{sheet_name}' not in {path}")
    sh = wb[sheet_name]

    rows = []
    seen = set()
    for row in sh.iter_rows(min_row=2, values_only=True):
        if row[0] is None:
            continue
        acc = str(row[0]).strip()
        if acc in seen:
            continue
        seen.add(acc)
        report = row[1]
        if report is None:
            continue
        rows.append({
            "accession_number": acc,
            "report": str(report).strip(),
            "split": split_label,
            "source_file": source_tag,
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dev-xlsx", required=True,
                    help="path to the 295-report validated xlsx (ECR cohort)")
    ap.add_argument("--test-xlsx", required=True,
                    help="path to the 780-report xlsx (expansion cohort)")
    ap.add_argument("--dev-sheet", default="sampled_300_reports_with_ground")
    ap.add_argument("--test-sheet", default="Validacao")
    ap.add_argument("--output", default="reports_master.csv")
    args = ap.parse_args()

    print(f"Reading dev cohort from: {args.dev_xlsx}")
    dev_df = read_reports_from_xlsx(Path(args.dev_xlsx), args.dev_sheet, "dev", "ecr_cohort_295")
    print(f"  -> {len(dev_df)} unique reports (dev)")

    print(f"Reading test cohort from: {args.test_xlsx}")
    test_df = read_reports_from_xlsx(Path(args.test_xlsx), args.test_sheet, "test", "expansion_cohort_780")
    print(f"  -> {len(test_df)} unique reports (test)")

    # Check no overlap
    overlap = set(dev_df["accession_number"]) & set(test_df["accession_number"])
    if overlap:
        print(f"WARNING: {len(overlap)} accession numbers appear in BOTH cohorts:")
        for a in list(overlap)[:10]:
            print(f"  {a}")
        sys.exit("Fix the overlap before proceeding.")

    master = pd.concat([dev_df, test_df], ignore_index=True)
    master.to_csv(args.output, index=False)

    print(f"\nWrote: {args.output}")
    print(f"  total reports: {len(master)}")
    print(f"  dev:  {len(dev_df)}")
    print(f"  test: {len(test_df)}")


if __name__ == "__main__":
    main()
