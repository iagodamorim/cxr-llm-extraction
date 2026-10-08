"""
compare_predictions.py
=======================

Compare LLM predictions (one CSV per vendor from api_inference.py) against
the human-validated ground truth (ground_truth_7class.csv).

Both sides use the v4 7-class schema:
  opacity, pneumonia, cardiomegaly, effusion, pneumothorax, mass, yesfinding

Per class (and per split) computes: precision, recall, F1, support,
TP, FP, FN, TN.

Also reports the per-vendor entailment_violation rate (pneumonia=1 with
opacity=0), tracked transparently per prompt v4 hierarchical rule.

Outputs:
  metrics_{vendor}.csv         per-class metrics (long form)
  metrics_{vendor}.xlsx        same, formatted, with per-split sheet + summary
"""

import argparse
from pathlib import Path
import csv
from collections import defaultdict

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

GT_CLASSES = ["opacity", "pneumonia", "cardiomegaly", "effusion",
              "pneumothorax", "mass", "yesfinding"]


def read_gt(path):
    """Return {accession: {classe: 0/1, ..., 'split': 'dev'|'test'}}."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            acc = r["accession_number"]
            d = {"split": r["split"]}
            for c in GT_CLASSES:
                d[c] = int(r[c])
            out[acc] = d
    return out


def read_preds(path):
    """Return {accession: {classe: 0/1/None, 'split': str, 'parse_ok': bool, 'entailment_violation': bool}}."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            acc = r["accession_number"]
            parse_ok = (r.get("parse_ok") or "").strip().lower() in ("true", "1")
            violation = (r.get("entailment_violation") or "").strip().lower() in ("true", "1")
            def _v(k):
                v = r.get(k)
                if v is None or v == "":
                    return None
                try: return int(v)
                except ValueError: return None
            d = {
                "split": r["split"],
                "parse_ok": parse_ok,
                "entailment_violation": violation,
            }
            for c in GT_CLASSES:
                d[c] = _v(c)
            out[acc] = d
    return out


def confusion(gt_list, pred_list):
    """Both lists are 0/1; pred may have None for missing → counted as 0."""
    tp = fp = fn = tn = 0
    for g, p in zip(gt_list, pred_list):
        p = 0 if p is None else p
        if g == 1 and p == 1: tp += 1
        elif g == 0 and p == 1: fp += 1
        elif g == 1 and p == 0: fn += 1
        else: tn += 1
    return tp, fp, fn, tn


def metrics_from(tp, fp, fn, tn):
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def compute(gt, preds):
    """Return list of dicts: split, classe, tp, fp, fn, tn, precision, recall, f1, support, n."""
    rows = []
    splits = sorted({d["split"] for d in gt.values()} | {"all"})
    for split in splits:
        accs = [a for a in gt if (split == "all" or gt[a]["split"] == split)]
        accs_with_pred = [a for a in accs if a in preds]
        for c in GT_CLASSES:
            g = [gt[a][c] for a in accs_with_pred]
            p = [preds[a][c] for a in accs_with_pred]
            tp, fp, fn, tn = confusion(g, p)
            prec, rec, f1 = metrics_from(tp, fp, fn, tn)
            rows.append({
                "split": split,
                "classe": c,
                "n_evaluated": len(accs_with_pred),
                "support_positive": sum(g),
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision": round(prec, 4),
                "recall":    round(rec, 4),
                "f1":        round(f1, 4),
            })
    return rows


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows: w.writerow(r)


def write_xlsx(path, rows, vendor, n_gt, n_pred, n_overlap, n_parse_ok, n_violations):
    wb = Workbook()
    ws = wb.active
    ws.title = "metrics"
    NAVY, GREEN, AMBER, GREY = "1F3864", "C6EFCE", "FFEB9C", "F2F2F2"

    cols = list(rows[0].keys())
    for ci, c in enumerate(cols, 1):
        cell = ws.cell(row=1, column=ci, value=c)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", start_color=NAVY)
        cell.alignment = Alignment(horizontal="center", wrap_text=True)
    ws.row_dimensions[1].height = 28

    for ri, r in enumerate(rows, 2):
        for ci, c in enumerate(cols, 1):
            cell = ws.cell(row=ri, column=ci, value=r[c])
            cell.alignment = Alignment(horizontal="center", vertical="center")
            if c == "f1":
                v = r[c]
                if v >= 0.8:   cell.fill = PatternFill("solid", start_color=GREEN)
                elif v >= 0.6: cell.fill = PatternFill("solid", start_color=AMBER)
        if ri % 2 == 0:
            for ci in range(1, len(cols) + 1):
                cc = ws.cell(row=ri, column=ci)
                if not cc.fill.start_color.rgb or cc.fill.start_color.rgb == "00000000":
                    cc.fill = PatternFill("solid", start_color=GREY)
    widths = {"split": 8, "classe": 22, "n_evaluated": 12, "support_positive": 14,
              "tp": 6, "fp": 6, "fn": 6, "tn": 6,
              "precision": 12, "recall": 10, "f1": 10}
    for ci, c in enumerate(cols, 1):
        ws.column_dimensions[get_column_letter(ci)].width = widths.get(c, 12)
    ws.freeze_panes = "C2"

    # Summary sheet
    ws2 = wb.create_sheet("summary")
    viol_pct = (100 * n_violations / n_pred) if n_pred else 0
    info = [
        ("vendor", vendor),
        ("rows na GT", n_gt),
        ("rows nas preds", n_pred),
        ("overlap (avaliadas)", n_overlap),
        ("parse_ok nas preds", n_parse_ok),
        ("entailment violations", f"{n_violations} ({viol_pct:.2f}%)"),
        ("", ""),
        ("classes (7)", ", ".join(GT_CLASSES)),
        ("hierarchy", "pneumonia=1 entails opacity=1 (per prompt v4)"),
        ("nota", "raw F1 — violations tracked, NOT silently fixed"),
    ]
    for ri, (k, v) in enumerate(info, 1):
        ws2.cell(row=ri, column=1, value=k).font = Font(bold=True)
        ws2.cell(row=ri, column=2, value=v)
    ws2.column_dimensions["A"].width = 24
    ws2.column_dimensions["B"].width = 70

    wb.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt",    default="data/outputs/ground_truth_7class.csv")
    ap.add_argument("--preds", required=True, help="{vendor}_responses.csv")
    ap.add_argument("--vendor", default=None,
                    help="auto-derived from --preds filename if omitted")
    ap.add_argument("--out-dir", default="data/outputs")
    args = ap.parse_args()

    vendor = args.vendor or Path(args.preds).stem.split("_")[0]
    gt = read_gt(args.gt)
    preds = read_preds(args.preds)
    overlap = set(gt) & set(preds)
    n_parse_ok = sum(1 for d in preds.values() if d["parse_ok"])
    n_violations = sum(1 for d in preds.values() if d["entailment_violation"])

    rows = compute(gt, preds)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    csv_path  = out_dir / f"metrics_{vendor}.csv"
    xlsx_path = out_dir / f"metrics_{vendor}.xlsx"
    write_csv(csv_path, rows)
    write_xlsx(xlsx_path, rows, vendor, len(gt), len(preds), len(overlap), n_parse_ok, n_violations)

    print(f"GT rows         : {len(gt)}")
    print(f"Preds rows      : {len(preds)}  (parse_ok: {n_parse_ok})")
    print(f"Overlap         : {len(overlap)}")
    print(f"Entailment violations: {n_violations} ({100*n_violations/max(len(preds),1):.2f}%)")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {xlsx_path}")

    # quick stdout summary on "all" split
    print(f"\nF1 por classe (split=all):")
    for r in rows:
        if r["split"] != "all": continue
        print(f"  {r['classe']:>22s}  F1={r['f1']:.3f}  P={r['precision']:.3f}  R={r['recall']:.3f}  "
              f"(TP={r['tp']:3d} FP={r['fp']:3d} FN={r['fn']:3d}  support+={r['support_positive']})")


if __name__ == "__main__":
    main()
