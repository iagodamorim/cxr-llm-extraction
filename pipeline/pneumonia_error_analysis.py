"""
pneumonia_error_analysis.py
============================

Qualitative + quantitative error analysis for the pneumonia class on the
test split. Designed to support Figure 5 of the paper (replacing the
preliminary fig5_bacterial_pneumonia from v1.4 manuscript).

For each pneumonia-positive case in the test set:
  - Record the report excerpt
  - Record predictions from each of the 6 vendors
  - Categorize: all-correct, partial-miss, all-miss

For pneumonia-NEGATIVE cases where any model predicted positive (FP):
  - Record the report excerpt
  - Record which vendors raised the false alarm

Inputs:
  --responses-dir   responses/*_test.csv
  --reports-csv     reports_master.csv (text)
  --gt              ground_truth_7class.csv
  --output-xlsx     pneumonia_errors.xlsx

Outputs (xlsx with 4 sheets):
  - all_correct        — positives where all 6 vendors got it right
  - partial_miss       — positives where ≥1 vendor missed but ≥1 caught
  - all_miss           — positives where ALL 6 vendors missed (hardest cases)
  - false_positives    — negatives where ≥1 vendor falsely raised pneumonia

Plus a console summary with per-vendor recall + counts.

Author: IPDA et al. PAP v2.1 Phase C. Substitutes Fig 5 of manuscript v1.4.
"""

import argparse
from glob import glob
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill, Border, Side
from openpyxl.utils import get_column_letter

V4_CLASSES = ["opacity", "pneumonia", "cardiomegaly", "effusion",
              "pneumothorax", "mass", "yesfinding"]

NAVY = "1F3864"; LIGHT = "F2F2F2"; GOLD = "FFE699"
GREEN = "C6EFCE"; AMBER = "FFEB9C"; RED = "FFC7CE"


def load_test_predictions(responses_dir: Path) -> dict[str, pd.DataFrame]:
    files = sorted(glob(str(responses_dir / "*_test.csv")))
    out = {}
    for f in files:
        name = Path(f).name
        if "smoke" in name.lower(): continue
        vendor = name.replace("_test.csv", "")
        df = pd.read_csv(f)
        df = df[df["split"] == "test"].set_index("accession_number")
        df["pneumonia"] = df["pneumonia"].fillna(0).astype(int)
        out[vendor] = df["pneumonia"]
    return out


def write_xlsx(positives_summary: pd.DataFrame, fps_summary: pd.DataFrame,
                vendors: list[str], out_path: Path) -> None:
    wb = Workbook()
    thin = Side(style="thin", color="BFBFBF")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    def write_sheet(ws, df, vendors_in_hdr=False):
        cols = list(df.columns)
        for ci, c in enumerate(cols, 1):
            cell = ws.cell(row=1, column=ci, value=c)
            cell.font = Font(name="Arial", size=10, bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", start_color=NAVY)
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell.border = border
        ws.row_dimensions[1].height = 28
        for ri, row in enumerate(df.itertuples(index=False), 2):
            for ci, val in enumerate(row, 1):
                cell = ws.cell(row=ri, column=ci, value=val if pd.notna(val) else "")
                cell.font = Font(name="Arial", size=9)
                cell.alignment = Alignment(vertical="top", wrap_text=True)
                cell.border = border
                # Color predictions: 1 = green-ish, 0 = neutral
                col_name = cols[ci-1]
                if col_name in vendors:
                    if val == 1:
                        cell.fill = PatternFill("solid", start_color=GREEN)
                    elif val == 0:
                        cell.fill = PatternFill("solid", start_color=RED)
            ws.row_dimensions[ri].height = 80
        # Width: accession=14, report=80, vendors=12, others=12
        widths = []
        for c in cols:
            if c == "report":
                widths.append(80)
            elif c == "accession_number":
                widths.append(14)
            else:
                widths.append(12)
        for ci, w in enumerate(widths, 1):
            ws.column_dimensions[get_column_letter(ci)].width = w
        ws.freeze_panes = "C2"

    # Categorize positives
    all_correct = positives_summary[positives_summary["n_correct"] == len(vendors)]
    all_miss    = positives_summary[positives_summary["n_correct"] == 0]
    partial     = positives_summary[(positives_summary["n_correct"] > 0) &
                                     (positives_summary["n_correct"] < len(vendors))]

    ws = wb.active
    ws.title = "all_correct"
    write_sheet(ws, all_correct)

    ws2 = wb.create_sheet("partial_miss")
    write_sheet(ws2, partial)

    ws3 = wb.create_sheet("all_miss (HARDEST)")
    write_sheet(ws3, all_miss)

    ws4 = wb.create_sheet("false_positives")
    write_sheet(ws4, fps_summary)

    wb.save(out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--responses-dir", default="responses")
    ap.add_argument("--reports-csv", default="data/outputs/reports_master.csv")
    ap.add_argument("--gt", default="data/outputs/ground_truth_7class.csv")
    ap.add_argument("--output-xlsx", default="data/outputs/pneumonia_errors.xlsx")
    args = ap.parse_args()

    # Load GT (test only)
    gt = pd.read_csv(args.gt)
    gt = gt[gt["split"] == "test"].set_index("accession_number")
    gt["pneumonia"] = gt["pneumonia"].fillna(0).astype(int)

    # Load reports
    reports = pd.read_csv(args.reports_csv).set_index("accession_number")

    # Load predictions
    preds = load_test_predictions(Path(args.responses_dir))
    vendors = sorted(preds.keys())
    print(f"Vendors: {vendors}")

    # Build summary table: accession × vendors + report
    pos_accs = gt[gt["pneumonia"] == 1].index.tolist()
    neg_accs = gt[gt["pneumonia"] == 0].index.tolist()
    print(f"Pneumonia positives in test: {len(pos_accs)}")
    print(f"Pneumonia negatives in test: {len(neg_accs)}")

    # ----- Positives summary -----
    pos_rows = []
    for acc in pos_accs:
        row = {"accession_number": acc}
        n_correct = 0
        for v in vendors:
            if acc in preds[v].index:
                p = int(preds[v].loc[acc])
                row[v] = p
                if p == 1: n_correct += 1
            else:
                row[v] = None
        row["n_correct"] = n_correct
        row["n_total"] = len(vendors)
        row["report"] = str(reports.loc[acc, "report"]) if acc in reports.index else ""
        pos_rows.append(row)
    positives_summary = pd.DataFrame(pos_rows)
    # Sort: hardest first (all_miss → partial → all_correct)
    positives_summary = positives_summary.sort_values("n_correct").reset_index(drop=True)
    # Reorder columns: accession, n_correct, n_total, vendor cols..., report
    col_order = ["accession_number", "n_correct", "n_total"] + vendors + ["report"]
    positives_summary = positives_summary[col_order]

    # ----- False positives summary -----
    fp_rows = []
    for acc in neg_accs:
        fp_vendors = []
        per_vendor = {}
        for v in vendors:
            if acc in preds[v].index:
                p = int(preds[v].loc[acc])
                per_vendor[v] = p
                if p == 1: fp_vendors.append(v)
            else:
                per_vendor[v] = None
        if not fp_vendors:
            continue  # not a FP case
        row = {"accession_number": acc, "n_fp": len(fp_vendors)}
        row.update(per_vendor)
        row["fp_vendors"] = ", ".join(fp_vendors)
        row["report"] = str(reports.loc[acc, "report"]) if acc in reports.index else ""
        fp_rows.append(row)
    fps_summary = pd.DataFrame(fp_rows).sort_values("n_fp", ascending=False).reset_index(drop=True)
    col_order_fp = ["accession_number", "n_fp", "fp_vendors"] + vendors + ["report"]
    if not fps_summary.empty:
        fps_summary = fps_summary[col_order_fp]

    # ----- Console summary -----
    print(f"\n=== Pneumonia error analysis summary ===")
    n_all_correct = int((positives_summary["n_correct"] == len(vendors)).sum())
    n_all_miss    = int((positives_summary["n_correct"] == 0).sum())
    n_partial     = len(positives_summary) - n_all_correct - n_all_miss
    print(f"Positives (N={len(positives_summary)}):")
    print(f"  all correct (all 6 caught):    {n_all_correct} ({100*n_all_correct/max(len(positives_summary),1):.1f}%)")
    print(f"  partial (some caught, some not): {n_partial} ({100*n_partial/max(len(positives_summary),1):.1f}%)")
    print(f"  all missed (hardest cases):     {n_all_miss} ({100*n_all_miss/max(len(positives_summary),1):.1f}%)")
    print(f"\nFalse positives: {len(fps_summary)} cases with at least 1 vendor raising pneumonia")

    print(f"\nPer-vendor recall on pneumonia (N={len(positives_summary)}):")
    for v in vendors:
        tp = int(positives_summary[v].sum())
        n = int(positives_summary[v].notna().sum())
        print(f"  {v:<22s}  {tp:>3d}/{n}  ({100*tp/max(n,1):.1f}%)")

    # ----- Write xlsx -----
    Path(args.output_xlsx).parent.mkdir(parents=True, exist_ok=True)
    write_xlsx(positives_summary, fps_summary, vendors, Path(args.output_xlsx))
    print(f"\nWrote: {args.output_xlsx}")


if __name__ == "__main__":
    main()
