"""
routing_stability.py
=====================

Tier 2C peer-review armor: bootstrap stability of the Specialist Routing
assignment.

For each of B bootstrap resamples of the development set:
  1. Compute per-class F1 for each of the 6 models on the resampled dev.
  2. Assign the routing winner per class (highest F1, alphabetical tie-break).
  3. Record which model won each class.

Output: per-class selection probability matrix (model x class) showing the
% of resamples in which each model was selected as the per-class specialist.

A class where one model is selected in ~95%+ of resamples is a stable routing
assignment; a class where three or four models trade the win is fragile, and
the paper can defend the hybrid (Hierarchical) design on the basis of that
fragility.

Inputs:
  --responses-dir   responses/*_dev.csv
  --gt              data/outputs/ground_truth_7class.csv  (uses split=='dev')
  --n-bootstrap     default 2000
  --output          data/outputs/routing_stability_dev.csv  (long form)

Author: IPDA et al. PAP v2.1 Phase C+ (statistical strengthening).
"""

import argparse
from glob import glob
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

V4_CLASSES = ["opacity", "pneumonia", "cardiomegaly", "effusion",
              "pneumothorax", "mass", "yesfinding"]


def f1_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--responses-dir", default="responses")
    ap.add_argument("--gt", default="data/outputs/ground_truth_7class.csv")
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="data/outputs/routing_stability_dev.csv")
    args = ap.parse_args()

    # Load GT (dev only)
    gt = pd.read_csv(args.gt)
    gt = gt[gt["split"] == "dev"].set_index("accession_number")
    for c in V4_CLASSES:
        gt[c] = gt[c].fillna(0).astype(int)
    print(f"Dev GT loaded: N={len(gt)}")

    # Load per-vendor predictions on dev
    vendors = []
    preds = {}
    for f in sorted(glob(str(Path(args.responses_dir) / "*_dev.csv"))):
        name = Path(f).name
        if "smoke" in name.lower(): continue
        v = name.replace("_dev.csv", "")
        df = pd.read_csv(f)
        df = df[df["split"] == "dev"].set_index("accession_number")
        for c in V4_CLASSES:
            df[c] = df[c].fillna(0).astype(int)
        preds[v] = df[V4_CLASSES]
        vendors.append(v)
    vendors = sorted(vendors)
    print(f"Vendors: {vendors}")

    # Align
    common = gt.index
    for v in vendors:
        common = common.intersection(preds[v].index)
    common = sorted(common)
    print(f"Aligned N: {len(common)}")
    gt_mat = gt.loc[common, V4_CLASSES].values.astype(int)
    pred_mats = {v: preds[v].loc[common, V4_CLASSES].values.astype(int) for v in vendors}

    # Bootstrap routing
    rng = np.random.default_rng(args.seed)
    n = len(common)
    # winners[class] = Counter({vendor: count})
    winners = {cls: Counter() for cls in V4_CLASSES}
    f1_means = {(v, cls): [] for v in vendors for cls in V4_CLASSES}

    for b in range(args.n_bootstrap):
        idx = rng.integers(0, n, size=n)
        gt_b = gt_mat[idx]
        # per-class F1 per vendor
        f1_table = np.zeros((len(vendors), len(V4_CLASSES)))
        for vi, v in enumerate(vendors):
            pv = pred_mats[v][idx]
            for cj, cls in enumerate(V4_CLASSES):
                f1 = f1_binary(gt_b[:, cj], pv[:, cj])
                f1_table[vi, cj] = f1
                f1_means[(v, cls)].append(f1)
        # winner per class (alphabetical tie-break by argmax with sorted vendors)
        for cj, cls in enumerate(V4_CLASSES):
            col = f1_table[:, cj]
            top = np.flatnonzero(col == col.max())
            winner = vendors[top[0]]   # vendors sorted alphabetically; argmax stable
            winners[cls][winner] += 1
        if (b + 1) % 500 == 0:
            print(f"  done {b+1}/{args.n_bootstrap}")

    # Build long-format output
    rows = []
    for cls in V4_CLASSES:
        for v in vendors:
            n_win = winners[cls][v]
            pct = 100.0 * n_win / args.n_bootstrap
            f1_b = np.array(f1_means[(v, cls)])
            rows.append({
                "classe": cls,
                "vendor": v,
                "n_bootstraps_won": n_win,
                "selection_probability_pct": round(pct, 2),
                "mean_f1": round(float(f1_b.mean()), 4),
                "sd_f1": round(float(f1_b.std(ddof=1)), 4),
                "n_total_bootstraps": args.n_bootstrap,
            })
    out = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"\nWrote: {args.output}  ({len(out)} rows)")

    # Console summary: top winner per class with stability
    print(f"\n=== Routing stability summary ({args.n_bootstrap} bootstraps) ===")
    print(f"{'class':14s} {'top winner':22s} {'pct':>7s}  {'runner-up':22s} {'pct':>7s}  STABLE?")
    for cls in V4_CLASSES:
        top = winners[cls].most_common(2)
        if len(top) < 2:
            top.append(("--", 0))
        (v1, n1), (v2, n2) = top
        p1 = 100.0 * n1 / args.n_bootstrap
        p2 = 100.0 * n2 / args.n_bootstrap
        stable = "STABLE" if p1 >= 80 else ("MIXED" if p1 >= 50 else "FRAGILE")
        print(f"  {cls:14s} {v1:22s} {p1:6.1f}%  {v2:22s} {p2:6.1f}%   {stable}")


if __name__ == "__main__":
    main()
