"""
tost_hd_vs_sonnet.py
=====================

Formal TOST equivalence + one-sided superiority test for Hierarchical
Decision vs Claude Sonnet 4.6 on the held-out test split.

Three hypotheses tested:
  (i)  Superiority of HD over Sonnet 4.6 (one-sided, H1: delta > 0)
       -- bootstrap paired-delta p-value
  (ii) Non-inferiority of HD vs Sonnet (TOST, margin 0.05 F1, H1: delta > -0.05)
  (iii) Equivalence within margin 0.05 (TOST, H1: |delta| < 0.05)

Method: paired nonparametric bootstrap of macro F1 difference (10000 resamples),
percentile-based one-sided and two-sided tests. Same paired index applied to
both predictors so each resample produces a true pairwise delta.

Author: IPDA et al. PAP v2.1 Phase C+ (P1 hardening).
"""

import argparse
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd


V4_CLASSES = ["opacity", "pneumonia", "cardiomegaly", "effusion",
              "pneumothorax", "mass", "yesfinding"]


def f1_binary(y_true, y_pred):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    if tp + fp == 0 or tp + fn == 0:
        if tp == 0 and fp == 0 and fn == 0:
            return 1.0
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def macro_f1(y_true_mat, y_pred_mat):
    return float(np.mean([f1_binary(y_true_mat[:, c], y_pred_mat[:, c])
                          for c in range(y_true_mat.shape[1])]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--responses-dir", default="responses")
    ap.add_argument("--ensemble-preds", default="data/outputs/ensemble_predictions_test.csv")
    ap.add_argument("--gt", default="data/outputs/ground_truth_7class.csv")
    ap.add_argument("--reference", default="sonnet-4-6",
                    help="The reference individual model")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--margin", type=float, default=0.05,
                    help="TOST equivalence margin (F1)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="data/outputs/tost_hd_vs_sonnet.csv")
    args = ap.parse_args()

    # Load GT (test)
    gt = pd.read_csv(args.gt)
    gt = gt[gt["split"] == "test"].set_index("accession_number")
    for c in V4_CLASSES:
        gt[c] = gt[c].fillna(0).astype(int)
    print(f"GT test N = {len(gt)}")

    # Load reference (Sonnet by default)
    ref_file = Path(args.responses_dir) / f"{args.reference}_test.csv"
    ref = pd.read_csv(ref_file)
    ref = ref[ref["split"] == "test"].set_index("accession_number")
    for c in V4_CLASSES:
        ref[c] = ref[c].fillna(0).astype(int)
    print(f"Reference model: {args.reference}, N = {len(ref)}")

    # Load Hierarchical Decision predictions from ensemble file
    ens = pd.read_csv(args.ensemble_preds)
    hd = ens[ens["strategy"] == "hierarchical_decision"]
    hd_wide = hd.pivot_table(index="accession_number", columns="classe",
                              values="prediction", aggfunc="first")
    hd_wide = hd_wide[V4_CLASSES]
    print(f"Hierarchical Decision N = {len(hd_wide)}")

    # Align
    common = sorted(gt.index.intersection(ref.index).intersection(hd_wide.index))
    print(f"Aligned N = {len(common)}")

    gt_mat = gt.loc[common, V4_CLASSES].values.astype(int)
    ref_mat = ref.loc[common, V4_CLASSES].values.astype(int)
    hd_mat = hd_wide.loc[common].fillna(0).astype(int).values

    # Point estimate
    f1_hd = macro_f1(gt_mat, hd_mat)
    f1_ref = macro_f1(gt_mat, ref_mat)
    delta_point = f1_hd - f1_ref
    print(f"\n  HD macro F1   = {f1_hd:.4f}")
    print(f"  Ref macro F1  = {f1_ref:.4f}  ({args.reference})")
    print(f"  Delta point   = {delta_point:+.4f}")

    # Paired bootstrap of macro-F1 delta (same resample index for both)
    rng = np.random.default_rng(args.seed)
    n = len(common)
    boots = np.empty(args.n_boot)
    for i in range(args.n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = (macro_f1(gt_mat[idx], hd_mat[idx]) -
                    macro_f1(gt_mat[idx], ref_mat[idx]))

    # Two-sided 95% CI
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
    # One-sided 95% lower bound (for superiority H1: delta > 0)
    ci_lower_one_sided = np.percentile(boots, 5)

    # Test (i) Superiority: H0: delta <= 0; H1: delta > 0
    # p_one_sided = P(bootstrap delta <= 0 | observed)
    p_sup = float((boots <= 0).mean())

    # Test (ii) Non-inferiority: H0: delta <= -margin; H1: delta > -margin
    # p_ni = P(bootstrap delta <= -margin | observed)
    p_ni = float((boots <= -args.margin).mean())

    # Test (iii) Equivalence (TOST): H0_lower: delta <= -margin; H0_upper: delta >= +margin
    # H1: |delta| < margin (i.e., -margin < delta < +margin)
    # Reject lower bound if delta > -margin (one-sided), AND
    # Reject upper bound if delta < +margin (one-sided)
    # p_eq = max(p_lower_one_sided, p_upper_one_sided)
    p_lower_one = float((boots <= -args.margin).mean())
    p_upper_one = float((boots >= +args.margin).mean())
    p_eq = max(p_lower_one, p_upper_one)
    eq_supported = (ci_lo > -args.margin) and (ci_hi < +args.margin)

    print(f"\n  Bootstrap delta 95% CI:    [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"  Bootstrap delta one-sided 95% lower bound: {ci_lower_one_sided:+.4f}")
    print(f"\n  (i)   Superiority (H1: delta > 0):       p = {p_sup:.4f}")
    print(f"        => {'SUPPORTED' if p_sup < 0.05 else 'NOT supported'} at alpha=0.05")
    print(f"  (ii)  Non-inferiority (margin {args.margin}, H1: delta > -{args.margin}):  p = {p_ni:.4f}")
    print(f"        => {'SUPPORTED' if p_ni < 0.05 else 'NOT supported'} at alpha=0.05")
    print(f"  (iii) Equivalence (TOST, margin ±{args.margin}, H1: |delta| < {args.margin}):  p = {p_eq:.4f}")
    print(f"        => {'SUPPORTED' if eq_supported else 'NOT supported'} (CI within ±{args.margin})")

    rows = [
        {"test": "superiority", "h1": f"delta > 0",
         "delta_point": round(delta_point, 4),
         "ci_lo_two_sided": round(float(ci_lo), 4),
         "ci_hi_two_sided": round(float(ci_hi), 4),
         "p_value": round(p_sup, 5),
         "alpha": 0.05,
         "decision": "supported" if p_sup < 0.05 else "not supported",
         "n_boot": args.n_boot, "n": len(common)},
        {"test": "non_inferiority", "h1": f"delta > -{args.margin}",
         "delta_point": round(delta_point, 4),
         "ci_lo_two_sided": round(float(ci_lo), 4),
         "ci_hi_two_sided": round(float(ci_hi), 4),
         "p_value": round(p_ni, 5),
         "alpha": 0.05,
         "decision": "supported" if p_ni < 0.05 else "not supported",
         "n_boot": args.n_boot, "n": len(common)},
        {"test": "equivalence_TOST", "h1": f"|delta| < {args.margin}",
         "delta_point": round(delta_point, 4),
         "ci_lo_two_sided": round(float(ci_lo), 4),
         "ci_hi_two_sided": round(float(ci_hi), 4),
         "p_value": round(p_eq, 5),
         "alpha": 0.05,
         "decision": "supported" if eq_supported else "not supported",
         "n_boot": args.n_boot, "n": len(common)},
    ]
    out = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"\nWrote: {args.output}")


if __name__ == "__main__":
    main()
