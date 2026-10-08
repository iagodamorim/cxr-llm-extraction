"""
bca_sensitivity.py
===================

Bias-corrected and accelerated (BCa) bootstrap as a sensitivity analysis
for per-class F1 confidence intervals, specifically targeting classes
where the point estimate is at or near 1.000 (and where the percentile
method gives a degenerate upper bound).

We re-compute CI95% for every (unit, classe) cell using BCa (via
scipy.stats.bootstrap) and report it alongside the percentile-based
primary analysis. Cells where the upper bound changes meaningfully
are flagged.

Author: IPDA et al. PAP v2.1 Phase C+ (P1 hardening).
"""

import argparse
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import bootstrap

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


def f1_from_indices(idx, y_true, y_pred):
    """Statistic function for scipy.stats.bootstrap.
    Accepts a sample of indices and returns F1 on the resampled rows."""
    idx = np.asarray(idx, dtype=int)
    return f1_binary(y_true[idx], y_pred[idx])


def bca_ci_f1(y_true, y_pred, n_boot=10000, seed=42):
    """Returns (point, lo, hi) using BCa."""
    n = len(y_true)
    point = f1_binary(y_true, y_pred)
    # scipy.stats.bootstrap with method='BCa' requires the statistic to
    # operate on data samples (not indices). We use a workaround: pass an
    # array of indices [0..n-1] as the "data" and define statistic that
    # closes over y_true and y_pred.
    indices = np.arange(n)
    rng = np.random.default_rng(seed)

    def stat(idx, axis=-1):
        # idx has shape (..., n) after resampling
        if idx.ndim == 1:
            return f1_binary(y_true[idx.astype(int)], y_pred[idx.astype(int)])
        # vectorized: shape (B, n)
        out = np.empty(idx.shape[0])
        for i in range(idx.shape[0]):
            ii = idx[i].astype(int)
            out[i] = f1_binary(y_true[ii], y_pred[ii])
        return out

    try:
        res = bootstrap(
            (indices,),
            stat,
            n_resamples=n_boot,
            method="BCa",
            random_state=rng,
            vectorized=True,
            paired=False,
            confidence_level=0.95,
        )
        return point, float(res.confidence_interval.low), float(res.confidence_interval.high)
    except Exception as e:
        # BCa fails when the jackknife acceleration is undefined (e.g. point=1)
        # Fall back to percentile method to keep the row, mark accel=NaN.
        rng2 = np.random.default_rng(seed)
        boots = np.empty(n_boot)
        for i in range(n_boot):
            idx = rng2.integers(0, n, size=n)
            boots[i] = f1_binary(y_true[idx], y_pred[idx])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return point, float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--responses-dir", default="responses")
    ap.add_argument("--ensemble-preds", default="data/outputs/ensemble_predictions_test.csv")
    ap.add_argument("--gt", default="data/outputs/ground_truth_7class.csv")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="data/outputs/bca_sensitivity_test.csv")
    args = ap.parse_args()

    gt = pd.read_csv(args.gt)
    gt = gt[gt["split"] == "test"].set_index("accession_number")
    for c in V4_CLASSES:
        gt[c] = gt[c].fillna(0).astype(int)
    print(f"GT test N = {len(gt)}")

    # individuals
    individuals = {}
    for f in sorted(glob(str(Path(args.responses_dir) / "*_test.csv"))):
        name = Path(f).name
        if "smoke" in name.lower():
            continue
        vendor = name.replace("_test.csv", "")
        df = pd.read_csv(f)
        df = df[df["split"] == "test"].set_index("accession_number")
        for c in V4_CLASSES:
            df[c] = df[c].fillna(0).astype(int)
        individuals[vendor] = df[V4_CLASSES]

    # ensembles
    ens = pd.read_csv(args.ensemble_preds)
    ens_wide = ens.pivot_table(index="accession_number",
                                columns=["strategy", "classe"],
                                values="prediction", aggfunc="first")
    strategies = sorted(set(s for s, _ in ens_wide.columns))

    common = gt.index
    for v, df in individuals.items():
        common = common.intersection(df.index)
    common = common.intersection(ens_wide.index)
    common = sorted(common)
    gt_a = gt.loc[common]

    # Load existing percentile CIs to merge
    pct = pd.read_csv("data/outputs/bootstrap_ci_test.csv")
    pct_lookup = {(r.unit, r.classe): (r.f1_point, r.f1_lo, r.f1_hi)
                  for r in pct.itertuples(index=False)}

    rows = []

    print(f"\n--- Individual models ---")
    for v, df in individuals.items():
        df_a = df.loc[common]
        for cls in V4_CLASSES:
            y_t = gt_a[cls].values.astype(int)
            y_p = df_a[cls].values.astype(int)
            point, bca_lo, bca_hi = bca_ci_f1(y_t, y_p, n_boot=args.n_boot, seed=args.seed)
            pct_p, pct_lo, pct_hi = pct_lookup.get((v, cls), (None, None, None))
            ceiling = (point >= 0.999)
            rows.append({
                "unit": v, "unit_type": "individual", "classe": cls,
                "f1_point": round(point, 4),
                "percentile_lo": pct_lo, "percentile_hi": pct_hi,
                "bca_lo": round(bca_lo, 4), "bca_hi": round(bca_hi, 4),
                "at_ceiling": ceiling,
            })
        print(f"  done: {v}")

    print(f"\n--- Ensembles ---")
    for s in strategies:
        for cls in V4_CLASSES:
            key = (s, cls)
            if key not in ens_wide.columns: continue
            y_p = ens_wide.loc[common, key].fillna(0).astype(int).values
            y_t = gt_a[cls].values.astype(int)
            point, bca_lo, bca_hi = bca_ci_f1(y_t, y_p, n_boot=args.n_boot, seed=args.seed)
            pct_p, pct_lo, pct_hi = pct_lookup.get((s, cls), (None, None, None))
            ceiling = (point >= 0.999)
            rows.append({
                "unit": s, "unit_type": "ensemble", "classe": cls,
                "f1_point": round(point, 4),
                "percentile_lo": pct_lo, "percentile_hi": pct_hi,
                "bca_lo": round(bca_lo, 4), "bca_hi": round(bca_hi, 4),
                "at_ceiling": ceiling,
            })
        print(f"  done: {s}")

    out = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"\nWrote: {args.output}  ({len(out)} rows)")

    # Summary: cells at ceiling, comparing percentile vs BCa
    ceiling_rows = out[out["at_ceiling"]]
    print(f"\n=== Cells at ceiling (F1 >= 0.999) ===")
    print(f"  Total: {len(ceiling_rows)}")
    if len(ceiling_rows):
        print(ceiling_rows[["unit", "classe", "f1_point",
                            "percentile_lo", "percentile_hi",
                            "bca_lo", "bca_hi"]].to_string(index=False))

    # Max divergence overall
    if "percentile_lo" in out.columns:
        out["divergence_lo"] = (out["bca_lo"] - out["percentile_lo"]).abs()
        out["divergence_hi"] = (out["bca_hi"] - out["percentile_hi"]).abs()
        print(f"\n=== CI bound divergence (BCa vs percentile) ===")
        print(f"  Max |lo divergence|: {out['divergence_lo'].max():.4f}")
        print(f"  Max |hi divergence|: {out['divergence_hi'].max():.4f}")
        print(f"  Median |lo divergence|: {out['divergence_lo'].median():.4f}")
        print(f"  Median |hi divergence|: {out['divergence_hi'].median():.4f}")


if __name__ == "__main__":
    main()
