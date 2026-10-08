"""
compute_ensembles.py
=====================

Computes the 5 ensemble strategies on the test split (PAP v2.1 §6, with
Amendment 2 redefinitions for 6 vendors). This is the script that produces
the PRIMARY ENDPOINT of the study: Specialist Routing ensemble F1 vs best
individual model on the test split.

Strategies:
  1. Consensus Voting       — majority (>=4 of 6)
  2. Weighted Consensus     — weight = vendor macro F1 on DEV
  3. Confidence-Weighted    — weight = vendor per-class F1 on DEV
  4. Specialist Routing     — per-class, prediction from assigned vendor (from JSON)
  5. Hierarchical Decision  — SR for {mass, pneumothorax}, WC for others

PAP COMPLIANCE:
  - Specialist Routing assignment MUST come from dev (PAP §6 #4 + Amendment 2)
    --> read from specialist_routing_assignment.json
  - Weights for WC/CWE MUST come from dev (PAP §6 #2, #3)
    --> read from metrics_*_dev.csv (passed via --dev-metrics-dir)
  - All metrics REPORTED on test (PAP §5)
    --> read response CSVs from responses/*_test.csv

If dev metrics are missing/empty for any vendor, the script aborts with
a clear error. NEVER falls back to test metrics for weights — that would
be data leakage.

INPUTS:
  --responses-dir       directory with {vendor}_test.csv files
  --gt                  ground_truth_7class.csv (v2)
  --routing-json        specialist_routing_assignment.json
  --dev-metrics-dir     directory with metrics_*_dev.csv (for WC/CWE weights)
  --output-dir          where to write outputs (default: data/outputs)

OUTPUTS:
  ensemble_predictions_test.csv   long-form: report × strategy × class × pred
  ensemble_metrics_test.csv       per-strategy per-class metrics (same schema as metrics_*)
  ensemble_summary_test.xlsx      pretty F1 matrix + comparison vs best individual

Usage:
  python pipeline/compute_ensembles.py
  python pipeline/compute_ensembles.py \\
      --responses-dir responses \\
      --gt data/outputs/ground_truth_7class.csv \\
      --routing-json data/outputs/specialist_routing_assignment.json \\
      --dev-metrics-dir data/outputs

Author: IPDA et al. PAP v2.1 + Amendment 2 (Phase C, May 2026).
"""

import argparse
import json
import re
import sys
from glob import glob
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

V4_CLASSES = ["opacity", "pneumonia", "cardiomegaly", "effusion",
              "pneumothorax", "mass", "yesfinding"]
HIGH_STAKES = {"mass", "pneumothorax"}

STRATEGIES = [
    "consensus_voting",
    "weighted_consensus",
    "confidence_weighted",
    "specialist_routing",
    "hierarchical_decision",
]


# ----- Loaders --------------------------------------------------------------

def load_responses(responses_dir: Path, suffix: str = "_test.csv") -> dict[str, pd.DataFrame]:
    files = sorted(glob(str(responses_dir / f"*{suffix}")))
    out = {}
    for f in files:
        name = Path(f).name
        if "smoke" in name.lower(): continue
        vendor = name[:-len(suffix)]
        df = pd.read_csv(f)
        # Keep only essential columns + class predictions
        keep_cols = ["accession_number", "split", "parse_ok", "entailment_violation"] + V4_CLASSES
        keep_cols = [c for c in keep_cols if c in df.columns]
        df = df[keep_cols].copy()
        # Filter to test split only (safety)
        if "split" in df.columns:
            df = df[df["split"] == "test"].copy()
        out[vendor] = df
    return out


def load_dev_metrics(metrics_dir: Path) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Return (per_class_f1[vendor][class], macro_f1[vendor]) from dev metrics."""
    files = sorted(glob(str(metrics_dir / "metrics_*_dev.csv")))
    per_class = {}
    macro = {}
    for f in files:
        name = Path(f).name
        if "smoke" in name.lower(): continue
        m = re.match(r"metrics_(.+)_dev\.csv$", name)
        if not m: continue
        vendor = m.group(1)
        df = pd.read_csv(f)
        df = df[df["split"] == "dev"]
        if df.empty:
            print(f"WARN: empty dev metrics for {vendor} in {f}", file=sys.stderr)
            continue
        per_class[vendor] = {row["classe"]: float(row["f1"]) for _, row in df.iterrows()}
        macro[vendor] = float(df["f1"].mean())
    return per_class, macro


def load_routing(json_path: Path) -> dict[str, str]:
    """Load the locked Specialist Routing assignment: {class: vendor}."""
    with open(json_path, encoding="utf-8") as f:
        d = json.load(f)
    return d["assignment"]


def load_gt(gt_path: Path) -> pd.DataFrame:
    df = pd.read_csv(gt_path)
    df = df[df["split"] == "test"].copy()
    keep = ["accession_number"] + V4_CLASSES
    return df[keep]


# ----- Ensemble computations ------------------------------------------------

def predict_consensus_voting(row_preds: dict[str, dict[str, int]],
                              vendors: list[str], cls: str) -> int:
    """Majority vote: >= ceil(N/2 + 1)."""
    n = len(vendors)
    threshold = (n // 2) + 1   # 6 vendors -> 4
    votes = sum(int(row_preds[v].get(cls, 0) == 1) for v in vendors)
    return 1 if votes >= threshold else 0


def predict_weighted_consensus(row_preds: dict[str, dict[str, int]],
                                vendors: list[str], cls: str,
                                macro_weights: dict[str, float]) -> int:
    """Weighted by vendor's overall macro F1 on dev."""
    total = sum(macro_weights[v] for v in vendors)
    weighted = sum(macro_weights[v] * int(row_preds[v].get(cls, 0) == 1) for v in vendors)
    return 1 if weighted > 0.5 * total else 0


def predict_confidence_weighted(row_preds: dict[str, dict[str, int]],
                                 vendors: list[str], cls: str,
                                 per_class_weights: dict[str, dict[str, float]]) -> int:
    """Weighted by vendor's per-class F1 on dev."""
    weights = {v: per_class_weights[v].get(cls, 0.5) for v in vendors}
    total = sum(weights.values())
    if total == 0: return 0
    weighted = sum(weights[v] * int(row_preds[v].get(cls, 0) == 1) for v in vendors)
    return 1 if weighted > 0.5 * total else 0


def predict_specialist_routing(row_preds: dict[str, dict[str, int]],
                                cls: str, assignment: dict[str, str]) -> int:
    """Take prediction from the locked vendor for this class."""
    vendor = assignment[cls]
    return int(row_preds[vendor].get(cls, 0) == 1)


def predict_hierarchical(row_preds: dict[str, dict[str, int]],
                          vendors: list[str], cls: str,
                          assignment: dict[str, str],
                          macro_weights: dict[str, float]) -> int:
    """SR for high-stakes classes, WC for others."""
    if cls in HIGH_STAKES:
        return predict_specialist_routing(row_preds, cls, assignment)
    else:
        return predict_weighted_consensus(row_preds, vendors, cls, macro_weights)


# ----- Per-row prediction matrix --------------------------------------------

def build_per_report_predictions(responses: dict[str, pd.DataFrame],
                                  vendors: list[str]) -> dict[str, dict[str, dict[str, int]]]:
    """Return {accession: {vendor: {class: 0/1}}}."""
    out = {}
    for v in vendors:
        df = responses[v]
        for _, row in df.iterrows():
            acc = str(row["accession_number"])
            if acc not in out: out[acc] = {}
            out[acc][v] = {cls: int(row[cls]) if pd.notna(row.get(cls)) else 0 for cls in V4_CLASSES}
    return out


def compute_ensemble_predictions(per_report: dict, vendors: list[str],
                                  assignment: dict[str, str],
                                  per_class_weights: dict, macro_weights: dict
                                  ) -> pd.DataFrame:
    """Long-form: accession × strategy × class × prediction."""
    rows = []
    for acc, row_preds in per_report.items():
        # Skip if any vendor missing this accession
        if not all(v in row_preds for v in vendors):
            continue
        for cls in V4_CLASSES:
            preds = {
                "consensus_voting":     predict_consensus_voting(row_preds, vendors, cls),
                "weighted_consensus":   predict_weighted_consensus(row_preds, vendors, cls, macro_weights),
                "confidence_weighted":  predict_confidence_weighted(row_preds, vendors, cls, per_class_weights),
                "specialist_routing":   predict_specialist_routing(row_preds, cls, assignment),
                "hierarchical_decision": predict_hierarchical(row_preds, vendors, cls, assignment, macro_weights),
            }
            for strat, p in preds.items():
                rows.append({
                    "accession_number": acc,
                    "strategy": strat,
                    "classe": cls,
                    "prediction": int(p),
                })
    return pd.DataFrame(rows)


# ----- Metrics --------------------------------------------------------------

def compute_per_strategy_metrics(ens_preds: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    """For each (strategy × class), compute tp/fp/fn/tn + precision/recall/specificity/f1/acc."""
    # Pivot ensemble preds: rows=accession, cols=(strategy, class) -> prediction
    ens_wide = ens_preds.pivot_table(index="accession_number",
                                       columns=["strategy", "classe"],
                                       values="prediction", aggfunc="first")
    gt_indexed = gt.set_index("accession_number")
    common = ens_wide.index.intersection(gt_indexed.index)
    ens_wide = ens_wide.loc[common]
    gt_indexed = gt_indexed.loc[common]

    rows = []
    for strat in STRATEGIES:
        for cls in V4_CLASSES:
            try:
                pred = ens_wide[(strat, cls)].astype(int).values
                truth = gt_indexed[cls].astype(int).values
            except KeyError:
                continue
            tp = int(((pred == 1) & (truth == 1)).sum())
            fp = int(((pred == 1) & (truth == 0)).sum())
            fn = int(((pred == 0) & (truth == 1)).sum())
            tn = int(((pred == 0) & (truth == 0)).sum())
            n = tp + fp + fn + tn
            support = tp + fn
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = (tp + tn) / n if n > 0 else 0.0
            rows.append({
                "strategy": strat, "classe": cls,
                "n_evaluated": n, "support_positive": support,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "specificity": round(specificity, 4),
                "f1": round(f1, 4),
                "accuracy": round(accuracy, 4),
            })
    return pd.DataFrame(rows)


def add_macro_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    """Append a macro-averaged F1 row per strategy."""
    macros = []
    for strat in STRATEGIES:
        s = metrics[metrics["strategy"] == strat]
        if s.empty: continue
        macros.append({
            "strategy": strat, "classe": "MACRO_AVG",
            "n_evaluated": int(s["n_evaluated"].max()),
            "support_positive": int(s["support_positive"].sum()),
            "tp": int(s["tp"].sum()),
            "fp": int(s["fp"].sum()),
            "fn": int(s["fn"].sum()),
            "tn": int(s["tn"].sum()),
            "precision": round(float(s["precision"].mean()), 4),
            "recall": round(float(s["recall"].mean()), 4),
            "specificity": round(float(s["specificity"].mean()), 4),
            "f1": round(float(s["f1"].mean()), 4),
            "accuracy": round(float(s["accuracy"].mean()), 4),
        })
    return pd.concat([metrics, pd.DataFrame(macros)], ignore_index=True)


# ----- Comparison vs individual models --------------------------------------

def load_individual_test_macro(metrics_dir: Path) -> dict[str, float]:
    """Load each vendor's macro F1 on test for comparison."""
    files = sorted(glob(str(metrics_dir / "metrics_*_test.csv")))
    out = {}
    for f in files:
        name = Path(f).name
        if "smoke" in name.lower(): continue
        m = re.match(r"metrics_(.+)_test\.csv$", name)
        if not m: continue
        vendor = m.group(1)
        df = pd.read_csv(f)
        df = df[df["split"] == "test"]
        if df.empty: continue
        out[vendor] = float(df["f1"].mean())
    return out


# ----- Output writers -------------------------------------------------------

def write_xlsx(metrics: pd.DataFrame, individual_macro: dict[str, float],
                assignment: dict[str, str], out_path: Path) -> None:
    wb = Workbook()
    NAVY = "1F3864"; LIGHT = "F2F2F2"; GOLD = "FFE699"; GREEN = "C6EFCE"; RED = "FFC7CE"
    thin = Side(style="thin", color="BFBFBF")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    # --- Sheet 1: F1 matrix (strategy × class + macro) ---
    ws = wb.active
    ws.title = "Ensemble F1 matrix"
    f1_mat = metrics.pivot(index="strategy", columns="classe", values="f1")
    # Reorder classes
    cols = [c for c in V4_CLASSES if c in f1_mat.columns] + (["MACRO_AVG"] if "MACRO_AVG" in f1_mat.columns else [])
    f1_mat = f1_mat[cols].reindex(index=STRATEGIES)

    hdr = ["strategy"] + cols
    for ci, h in enumerate(hdr, 1):
        c = ws.cell(row=1, column=ci, value=h)
        c.font = Font(name="Arial", size=10, bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", start_color=NAVY)
        c.alignment = Alignment(horizontal="center", vertical="center")
        c.border = border
    ws.row_dimensions[1].height = 22

    # Find best F1 per class column
    best_per_class = {col: f1_mat[col].idxmax() for col in cols if col in f1_mat.columns}

    for ri, strat in enumerate(STRATEGIES, 2):
        if strat not in f1_mat.index: continue
        ws.cell(row=ri, column=1, value=strat).font = Font(name="Arial", size=10, bold=True)
        ws.cell(row=ri, column=1).border = border
        for ci, col in enumerate(cols, 2):
            v = f1_mat.loc[strat, col]
            cell = ws.cell(row=ri, column=ci, value=round(float(v), 4) if pd.notna(v) else "")
            cell.font = Font(name="Arial", size=10)
            cell.alignment = Alignment(horizontal="center")
            cell.border = border
            if best_per_class.get(col) == strat:
                cell.fill = PatternFill("solid", start_color=GOLD)
                cell.font = Font(name="Arial", size=10, bold=True)
        if ri % 2 == 0:
            for ci in range(1, len(hdr) + 1):
                c = ws.cell(row=ri, column=ci)
                if not c.fill.start_color.rgb or c.fill.start_color.rgb in ("00000000",):
                    c.fill = PatternFill("solid", start_color=LIGHT)

    for ci, w in enumerate([22] + [12]*len(cols), 1):
        ws.column_dimensions[get_column_letter(ci)].width = w
    ws.freeze_panes = "B2"

    # --- Sheet 2: Specialist Routing applied + per-class winner comparison ---
    ws2 = wb.create_sheet("SR vs Individuals")
    hdr2 = ["classe", "SR_vendor (locked)", "SR_F1 on TEST",
            "best_individual_vendor (test)", "best_individual_F1 (test)",
            "delta (SR - best_ind)"]
    for ci, h in enumerate(hdr2, 1):
        c = ws2.cell(row=1, column=ci, value=h)
        c.font = Font(name="Arial", size=10, bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", start_color=NAVY)
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = border
    ws2.row_dimensions[1].height = 36

    sr_metrics = metrics[metrics["strategy"] == "specialist_routing"].set_index("classe")
    # Per-class winner on test from individual metrics dir
    # NOTE: we don't have per-class individual F1 here without re-reading metrics_*_test.csv
    # For brevity: leave that comparison cell blank if not loadable here.

    for ri, cls in enumerate(V4_CLASSES, 2):
        sr_vendor = assignment.get(cls, "?")
        sr_f1 = sr_metrics.loc[cls, "f1"] if cls in sr_metrics.index else None
        ws2.cell(row=ri, column=1, value=cls)
        ws2.cell(row=ri, column=2, value=sr_vendor).fill = PatternFill("solid", start_color=GOLD)
        ws2.cell(row=ri, column=3, value=round(float(sr_f1), 4) if sr_f1 is not None else "")
        # Cells 4,5,6 left empty here; full comparison done by mcnemar_grid.py
        for ci in range(1, len(hdr2) + 1):
            cell = ws2.cell(row=ri, column=ci)
            cell.font = Font(name="Arial", size=10)
            cell.alignment = Alignment(horizontal="center")
            cell.border = border

    for ci, w in enumerate([16, 22, 14, 26, 18, 18], 1):
        ws2.column_dimensions[get_column_letter(ci)].width = w

    # --- Sheet 3: Macro F1 ranking (ensembles vs individuals) ---
    ws3 = wb.create_sheet("Macro F1 ranking")
    macro_rows = metrics[metrics["classe"] == "MACRO_AVG"][["strategy", "f1"]].copy()
    macro_rows.columns = ["unit", "macro_f1"]
    macro_rows["unit_type"] = "ensemble"
    indiv_rows = pd.DataFrame([{"unit": v, "macro_f1": round(f, 4), "unit_type": "individual"}
                                for v, f in individual_macro.items()])
    all_rows = pd.concat([macro_rows, indiv_rows], ignore_index=True)
    all_rows = all_rows.sort_values("macro_f1", ascending=False).reset_index(drop=True)

    hdr3 = ["rank", "unit", "unit_type", "macro_F1"]
    for ci, h in enumerate(hdr3, 1):
        c = ws3.cell(row=1, column=ci, value=h)
        c.font = Font(name="Arial", size=10, bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", start_color=NAVY)
        c.alignment = Alignment(horizontal="center", vertical="center")
        c.border = border
    for ri, row in enumerate(all_rows.itertuples(index=False), 2):
        ws3.cell(row=ri, column=1, value=ri - 1)
        ws3.cell(row=ri, column=2, value=row.unit)
        ws3.cell(row=ri, column=3, value=row.unit_type)
        cell = ws3.cell(row=ri, column=4, value=round(float(row.macro_f1), 4))
        if row.unit_type == "ensemble":
            cell.fill = PatternFill("solid", start_color=GREEN)
            cell.font = Font(name="Arial", size=10, bold=True)
        for ci in range(1, len(hdr3) + 1):
            ws3.cell(row=ri, column=ci).border = border
            ws3.cell(row=ri, column=ci).alignment = Alignment(horizontal="center")
    for ci, w in enumerate([8, 22, 14, 12], 1):
        ws3.column_dimensions[get_column_letter(ci)].width = w

    wb.save(out_path)


# ----- Main -----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--responses-dir", default="responses")
    ap.add_argument("--gt", default="data/outputs/ground_truth_7class.csv")
    ap.add_argument("--routing-json", default="data/outputs/specialist_routing_assignment.json")
    ap.add_argument("--dev-metrics-dir", default="data/outputs")
    ap.add_argument("--output-dir", default="data/outputs")
    args = ap.parse_args()

    responses_dir = Path(args.responses_dir)
    dev_metrics_dir = Path(args.dev_metrics_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading responses from {responses_dir}/*_test.csv ...")
    responses = load_responses(responses_dir)
    if not responses:
        sys.exit(f"ERROR: no test responses in {responses_dir}")
    vendors = sorted(responses.keys())
    print(f"  vendors found: {vendors}")

    print(f"Loading dev metrics from {dev_metrics_dir}/metrics_*_dev.csv ...")
    per_class_w, macro_w = load_dev_metrics(dev_metrics_dir)
    missing_w = [v for v in vendors if v not in macro_w]
    if missing_w:
        sys.exit(f"ERROR: dev metrics missing for vendors {missing_w}. "
                 f"WC and CWE ensembles require dev metrics per PAP. "
                 f"Run compare_predictions.py on dev split first.")
    print(f"  dev macro F1: {macro_w}")

    print(f"Loading routing assignment from {args.routing_json} ...")
    assignment = load_routing(Path(args.routing_json))
    print(f"  Specialist Routing assignment:")
    for cls, v in assignment.items():
        print(f"    {cls:<15s} -> {v}")

    print(f"Loading test ground truth from {args.gt} ...")
    gt = load_gt(Path(args.gt))
    print(f"  test reports: {len(gt)}")

    print(f"Building per-report prediction matrix ...")
    per_report = build_per_report_predictions(responses, vendors)
    print(f"  reports with all vendors: "
          f"{sum(1 for r in per_report.values() if len(r) == len(vendors))}")

    print(f"Computing ensemble predictions for {len(STRATEGIES)} strategies ...")
    ens_preds = compute_ensemble_predictions(per_report, vendors, assignment,
                                              per_class_w, macro_w)
    ens_preds_path = out_dir / "ensemble_predictions_test.csv"
    ens_preds.to_csv(ens_preds_path, index=False)
    print(f"  wrote: {ens_preds_path} ({len(ens_preds)} rows)")

    print(f"Computing per-strategy metrics ...")
    metrics = compute_per_strategy_metrics(ens_preds, gt)
    metrics = add_macro_rows(metrics)
    metrics_path = out_dir / "ensemble_metrics_test.csv"
    metrics.to_csv(metrics_path, index=False)
    print(f"  wrote: {metrics_path}")

    # Compare against individual macro F1 on test
    individual_macro = load_individual_test_macro(dev_metrics_dir)
    print(f"\n=== Macro F1 ranking (ensembles + individuals on test) ===")
    all_macros = list(individual_macro.items())
    macro_avg_metrics = metrics[metrics["classe"] == "MACRO_AVG"]
    for _, r in macro_avg_metrics.iterrows():
        all_macros.append((f"[ENS] {r['strategy']}", float(r["f1"])))
    all_macros.sort(key=lambda x: -x[1])
    for i, (name, f1) in enumerate(all_macros, 1):
        marker = "  *" if name.startswith("[ENS]") else "   "
        print(f"  {i:>2d}. {name:<32s} {marker}  macro_F1 = {f1:.4f}")

    # XLSX summary
    xlsx_path = out_dir / "ensemble_summary_test.xlsx"
    write_xlsx(metrics, individual_macro, assignment, xlsx_path)
    print(f"\n  wrote: {xlsx_path}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
