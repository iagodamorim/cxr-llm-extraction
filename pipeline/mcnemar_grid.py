"""
mcnemar_grid.py
================

Pairwise McNemar tests between all (individual model + ensemble) pairs,
per class, on the test split. Also runs TOST non-inferiority tests for
the within-vendor pairs flagged in PAP §7 (Sonnet vs Haiku, Pro vs Flash,
GPT-4.1 vs Mini), and for the primary endpoint (Specialist Routing vs
best individual model).

PAP §8 (statistical analysis):
  - McNemar test on paired binary predictions
  - Bonferroni correction across the comparison family
  - α = 0.001 for confirmatory tests
  - TOST non-inferiority margin = 0.05 (F1 difference)

Inputs:
  --responses-dir         responses/*_test.csv
  --ensemble-preds        ensemble_predictions_test.csv
  --gt                    ground_truth_7class.csv
  --output-grid           mcnemar_grid_test.csv
  --output-tost           non_inferiority_test.csv

Output (mcnemar_grid_test.csv):
  unit_a, unit_b, classe, n, b, c, mcnemar_stat, p_value,
  p_bonferroni, significant

Output (non_inferiority_test.csv):
  pair, classe, f1_a, f1_b, diff, ci_lo, ci_hi, margin,
  non_inferior (boolean)

Author: IPDA et al. PAP v2.1 Phase C.
"""

import argparse
import itertools
import sys
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

V4_CLASSES = ["opacity", "pneumonia", "cardiomegaly", "effusion",
              "pneumothorax", "mass", "yesfinding"]

# Within-vendor non-inferiority pairs (PAP §7 secondary endpoint)
WITHIN_VENDOR_PAIRS = [
    ("sonnet-4-6", "haiku-4-5"),
    ("gemini-2-5-pro", "gemini-2-5-flash"),
    ("gpt-4-1", "gpt-4-1-mini"),
]

NI_MARGIN = 0.05  # F1 difference; non-inferior if (f1_a - f1_b) > -margin


def f1_score(y_true, y_pred):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    if tp + fp == 0 or tp + fn == 0:
        return 1.0 if (tp == 0 and fp == 0 and fn == 0) else 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def mcnemar_pair(pred_a: np.ndarray, pred_b: np.ndarray, y: np.ndarray):
    """Return (b, c, stat, p_value).

    b = count where A correct, B wrong
    c = count where B correct, A wrong
    Uses exact binomial when b+c < 25, chi-sq with continuity otherwise.
    """
    a_correct = (pred_a == y).astype(int)
    b_correct = (pred_b == y).astype(int)
    b = int(((a_correct == 1) & (b_correct == 0)).sum())
    c = int(((a_correct == 0) & (b_correct == 1)).sum())
    n_disc = b + c
    if n_disc == 0:
        return b, c, 0.0, 1.0
    if n_disc < 25:
        # Exact binomial (two-sided)
        k = min(b, c)
        p = 2 * stats.binom.cdf(k, n_disc, 0.5)
        p = min(p, 1.0)
        stat = abs(b - c)
        return b, c, float(stat), float(p)
    else:
        # Chi-square with continuity correction
        stat = (abs(b - c) - 1) ** 2 / n_disc
        p = 1 - stats.chi2.cdf(stat, df=1)
        return b, c, float(stat), float(p)


def bootstrap_diff_ci(pred_a, pred_b, y, n_boot=2000, seed=42):
    """Bootstrap CI95% for (F1_A - F1_B)."""
    n = len(y)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        f_a = f1_score(y[idx], pred_a[idx])
        f_b = f1_score(y[idx], pred_b[idx])
        diffs[i] = f_a - f_b
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)


# ----- Loaders --------------------------------------------------------------

def load_predictions(responses_dir: Path, ensemble_path: Path, gt_path: Path):
    """Returns:
        predictions_wide: {unit_name: {class: np.array of predictions}}
        gt: {class: np.array}
        accessions: sorted list of accession_numbers (alignment guarantee)
    """
    gt_df = pd.read_csv(gt_path)
    gt_df = gt_df[gt_df["split"] == "test"].set_index("accession_number")
    for c in V4_CLASSES:
        gt_df[c] = gt_df[c].fillna(0).astype(int)

    # Individuals
    units = {}
    files = sorted(glob(str(responses_dir / "*_test.csv")))
    for f in files:
        name = Path(f).name
        if "smoke" in name.lower(): continue
        vendor = name.replace("_test.csv", "")
        df = pd.read_csv(f)
        df = df[df["split"] == "test"].set_index("accession_number")
        for c in V4_CLASSES:
            if c in df.columns:
                df[c] = df[c].fillna(0).astype(int)
        units[vendor] = df

    # Align all on common accessions (gt + all individuals)
    common = set(gt_df.index)
    for u, df in units.items():
        common &= set(df.index)
    accs = sorted(common)
    gt_aligned = gt_df.loc[accs]

    pred_wide = {}
    for u, df in units.items():
        pred_wide[u] = {c: df.loc[accs, c].values.astype(int) for c in V4_CLASSES if c in df.columns}

    # Ensembles
    if ensemble_path.exists():
        ens = pd.read_csv(ensemble_path)
        ens_wide_df = ens.pivot_table(index="accession_number",
                                        columns=["strategy", "classe"],
                                        values="prediction", aggfunc="first")
        # Restrict to common accessions
        common_ens = set(ens_wide_df.index) & set(accs)
        accs_ens = sorted(common_ens)
        for strat in sorted(set(ens["strategy"])):
            name = f"[ENS] {strat}"
            pred_wide[name] = {}
            for c in V4_CLASSES:
                key = (strat, c)
                if key in ens_wide_df.columns:
                    series = ens_wide_df.loc[accs_ens, key].fillna(0).astype(int).values
                    pred_wide[name][c] = series

    return pred_wide, {c: gt_aligned[c].values.astype(int) for c in V4_CLASSES}, accs


# ----- Main -----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--responses-dir", default="responses")
    ap.add_argument("--ensemble-preds", default="data/outputs/ensemble_predictions_test.csv")
    ap.add_argument("--gt", default="data/outputs/ground_truth_7class.csv")
    ap.add_argument("--output-grid", default="data/outputs/mcnemar_grid_test.csv")
    ap.add_argument("--output-tost", default="data/outputs/non_inferiority_test.csv")
    ap.add_argument("--n-boot-tost", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.001)
    args = ap.parse_args()

    pred_wide, gt, accs = load_predictions(
        Path(args.responses_dir),
        Path(args.ensemble_preds),
        Path(args.gt),
    )
    units = list(pred_wide.keys())
    print(f"Units to compare ({len(units)}): {units}")
    print(f"Test reports aligned: {len(accs)}")

    # ----- McNemar grid -----
    print(f"\nRunning McNemar tests for all {len(list(itertools.combinations(units, 2)))} pairs × {len(V4_CLASSES)} classes...")
    rows = []
    for a, b in itertools.combinations(units, 2):
        for cls in V4_CLASSES:
            if cls not in pred_wide[a] or cls not in pred_wide[b]:
                continue
            pa = pred_wide[a][cls]
            pb = pred_wide[b][cls]
            # Ensure same length (align by accession; if ens has fewer, skip)
            n_min = min(len(pa), len(pb), len(gt[cls]))
            if n_min < 10:
                continue
            pa = pa[:n_min]; pb = pb[:n_min]; y = gt[cls][:n_min]
            b_disc, c_disc, stat, p = mcnemar_pair(pa, pb, y)
            rows.append({
                "unit_a": a, "unit_b": b, "classe": cls,
                "n": int(n_min),
                "b_a_correct_b_wrong": b_disc,
                "c_b_correct_a_wrong": c_disc,
                "mcnemar_stat": round(stat, 4),
                "p_value": float(p),
            })

    grid = pd.DataFrame(rows)
    # Bonferroni correction across the entire family
    n_tests = len(grid)
    grid["p_bonferroni"] = (grid["p_value"] * n_tests).clip(upper=1.0)
    grid["significant"] = grid["p_bonferroni"] < args.alpha
    Path(args.output_grid).parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(args.output_grid, index=False)
    print(f"  wrote: {args.output_grid}  ({n_tests} tests; α_corrected = {args.alpha}/{n_tests:.0f} = {args.alpha/n_tests:.2e})")
    n_sig = int(grid["significant"].sum())
    print(f"  significant after Bonferroni: {n_sig} / {n_tests} ({100*n_sig/n_tests:.1f}%)")

    # ----- TOST non-inferiority for within-vendor + primary endpoint -----
    print(f"\nRunning TOST non-inferiority for within-vendor pairs (margin = {NI_MARGIN}) ...")
    tost_rows = []

    # Within-vendor
    for a, b in WITHIN_VENDOR_PAIRS:
        if a not in pred_wide or b not in pred_wide:
            print(f"  SKIP: {a} or {b} not in units")
            continue
        for cls in V4_CLASSES:
            if cls not in pred_wide[a] or cls not in pred_wide[b]: continue
            pa = pred_wide[a][cls]; pb = pred_wide[b][cls]; y = gt[cls]
            n_min = min(len(pa), len(pb), len(y))
            pa = pa[:n_min]; pb = pb[:n_min]; y = y[:n_min]
            f_a = f1_score(y, pa)
            f_b = f1_score(y, pb)
            diff = f_a - f_b
            ci_lo, ci_hi = bootstrap_diff_ci(pa, pb, y, n_boot=args.n_boot_tost)
            # Non-inferior if (f_a - f_b) > -margin AND CI_lo > -margin
            non_inf = (ci_lo > -NI_MARGIN)
            tost_rows.append({
                "pair_type": "within_vendor",
                "unit_a": a, "unit_b": b, "classe": cls,
                "f1_a": round(f_a, 4), "f1_b": round(f_b, 4),
                "diff_a_minus_b": round(diff, 4),
                "ci95_lo": round(ci_lo, 4), "ci95_hi": round(ci_hi, 4),
                "margin": NI_MARGIN,
                "a_non_inferior_to_b": bool(non_inf),
            })

    # Primary endpoint: Specialist Routing vs best individual (macro F1 on test)
    if "[ENS] specialist_routing" in pred_wide:
        # Find best individual by computing macro F1
        macros = {}
        for u in [x for x in units if not x.startswith("[ENS]")]:
            f1s = [f1_score(gt[c], pred_wide[u][c]) for c in V4_CLASSES if c in pred_wide[u]]
            macros[u] = float(np.mean(f1s))
        best_ind = max(macros, key=macros.get)
        print(f"\nPrimary endpoint: [ENS] specialist_routing vs {best_ind} (best individual on test)")
        sr = "[ENS] specialist_routing"
        for cls in V4_CLASSES:
            if cls not in pred_wide[sr] or cls not in pred_wide[best_ind]: continue
            pa = pred_wide[sr][cls]; pb = pred_wide[best_ind][cls]; y = gt[cls]
            n_min = min(len(pa), len(pb), len(y))
            pa = pa[:n_min]; pb = pb[:n_min]; y = y[:n_min]
            f_a = f1_score(y, pa); f_b = f1_score(y, pb)
            diff = f_a - f_b
            ci_lo, ci_hi = bootstrap_diff_ci(pa, pb, y, n_boot=args.n_boot_tost)
            tost_rows.append({
                "pair_type": "primary_endpoint",
                "unit_a": sr, "unit_b": best_ind, "classe": cls,
                "f1_a": round(f_a, 4), "f1_b": round(f_b, 4),
                "diff_a_minus_b": round(diff, 4),
                "ci95_lo": round(ci_lo, 4), "ci95_hi": round(ci_hi, 4),
                "margin": NI_MARGIN,
                "a_non_inferior_to_b": bool(ci_lo > -NI_MARGIN),
            })

    tost = pd.DataFrame(tost_rows)
    tost.to_csv(args.output_tost, index=False)
    print(f"\n  wrote: {args.output_tost}  ({len(tost)} comparisons)")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
