"""
bootstrap_ci.py
================

Computes 95% confidence intervals (percentile bootstrap, 2000 resamples)
for per-class F1 of each individual model AND each ensemble strategy on
the test split. Output is in the format expected by paper Tables 2-4.

Inputs:
  --responses-dir         responses/*_test.csv files (6 individual models)
  --ensemble-preds        ensemble_predictions_test.csv (from compute_ensembles.py)
  --gt                    ground_truth_7class.csv
  --n-bootstrap           number of bootstrap resamples (default 2000)
  --output                bootstrap_ci_test.csv

Output format (long):
  unit, unit_type, classe, f1_point, f1_lo, f1_hi, n

unit_type ∈ {individual, ensemble}

PAP §8 (statistical analysis): bootstrap, 2000 resamples, percentile method.

Author: IPDA et al. PAP v2.1 Phase C.
"""

import argparse
import sys
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

V4_CLASSES = ["opacity", "pneumonia", "cardiomegaly", "effusion",
              "pneumothorax", "mass", "yesfinding"]


def f1_score_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    if tp + fp == 0 or tp + fn == 0:
        if tp == 0 and fp == 0 and fn == 0:
            return 1.0  # all-zero gt and pred -> degenerate, perfect
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def bootstrap_f1_ci(y_true: np.ndarray, y_pred: np.ndarray,
                     n_boot: int = 2000, seed: int = 42) -> tuple[float, float, float]:
    """Return (point, lo95, hi95) bootstrap F1."""
    n = len(y_true)
    rng = np.random.default_rng(seed)
    point = f1_score_binary(y_true, y_pred)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = f1_score_binary(y_true[idx], y_pred[idx])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def load_individual_preds(responses_dir: Path) -> dict[str, pd.DataFrame]:
    """Return {vendor: DataFrame indexed by accession_number with class cols}."""
    files = sorted(glob(str(responses_dir / "*_test.csv")))
    out = {}
    for f in files:
        name = Path(f).name
        if "smoke" in name.lower(): continue
        vendor = name.replace("_test.csv", "")
        df = pd.read_csv(f)
        df = df[df["split"] == "test"].copy()
        df = df.set_index("accession_number")
        # Fill NaN with 0 (defensive)
        for c in V4_CLASSES:
            if c in df.columns:
                df[c] = df[c].fillna(0).astype(int)
        out[vendor] = df
    return out


def load_ensemble_preds(path: Path) -> pd.DataFrame:
    """Long form: accession_number, strategy, classe, prediction."""
    return pd.read_csv(path)


def load_gt(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["split"] == "test"].copy().set_index("accession_number")
    for c in V4_CLASSES:
        df[c] = df[c].fillna(0).astype(int)
    return df[V4_CLASSES]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--responses-dir", default="responses")
    ap.add_argument("--ensemble-preds", default="data/outputs/ensemble_predictions_test.csv")
    ap.add_argument("--gt", default="data/outputs/ground_truth_7class.csv")
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="data/outputs/bootstrap_ci_test.csv")
    args = ap.parse_args()

    print(f"Loading GT from {args.gt} ...")
    gt = load_gt(Path(args.gt))
    print(f"  test reports: {len(gt)}")

    print(f"Loading individual predictions from {args.responses_dir}/*_test.csv ...")
    individuals = load_individual_preds(Path(args.responses_dir))
    print(f"  vendors: {list(individuals.keys())}")

    print(f"Loading ensemble predictions from {args.ensemble_preds} ...")
    if Path(args.ensemble_preds).exists():
        ens = load_ensemble_preds(Path(args.ensemble_preds))
        ens_wide = ens.pivot_table(index="accession_number",
                                     columns=["strategy", "classe"],
                                     values="prediction", aggfunc="first")
    else:
        print(f"  WARN: not found; will only bootstrap individuals")
        ens_wide = None

    rows = []

    # Individuals
    print(f"\nBootstrapping individuals ({args.n_bootstrap} resamples each)...")
    for vendor, df in individuals.items():
        common = gt.index.intersection(df.index)
        gt_aligned = gt.loc[common]
        df_aligned = df.loc[common]
        for cls in V4_CLASSES:
            if cls not in df_aligned.columns: continue
            point, lo, hi = bootstrap_f1_ci(
                gt_aligned[cls].values.astype(int),
                df_aligned[cls].values.astype(int),
                n_boot=args.n_bootstrap, seed=args.seed)
            rows.append({
                "unit": vendor, "unit_type": "individual", "classe": cls,
                "f1_point": round(point, 4),
                "f1_lo": round(lo, 4),
                "f1_hi": round(hi, 4),
                "n": int(len(common)),
            })
        print(f"  done: {vendor}")

    # Ensembles
    if ens_wide is not None:
        print(f"\nBootstrapping ensembles ({args.n_bootstrap} resamples each)...")
        strategies = sorted(set(s for s, _ in ens_wide.columns))
        for strat in strategies:
            common = gt.index.intersection(ens_wide.index)
            gt_aligned = gt.loc[common]
            for cls in V4_CLASSES:
                key = (strat, cls)
                if key not in ens_wide.columns: continue
                pred = ens_wide.loc[common, key].fillna(0).astype(int).values
                point, lo, hi = bootstrap_f1_ci(
                    gt_aligned[cls].values.astype(int),
                    pred,
                    n_boot=args.n_bootstrap, seed=args.seed)
                rows.append({
                    "unit": strat, "unit_type": "ensemble", "classe": cls,
                    "f1_point": round(point, 4),
                    "f1_lo": round(lo, 4),
                    "f1_hi": round(hi, 4),
                    "n": int(len(common)),
                })
            print(f"  done: {strat}")

    out = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"\nWrote: {args.output}  ({len(out)} rows)")
    print(f"\nQuick view (macro F1 with CI per unit):")
    macro = out.groupby(["unit_type", "unit"]).agg(
        macro_f1=("f1_point", "mean"),
        macro_lo=("f1_lo", "mean"),
        macro_hi=("f1_hi", "mean")
    ).round(4).reset_index().sort_values("macro_f1", ascending=False)
    print(macro.to_string(index=False))


if __name__ == "__main__":
    main()
