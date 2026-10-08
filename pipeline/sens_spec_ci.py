"""
sens_spec_ci.py
================

Tier 2D peer-review armor: per-class sensitivity and specificity with
bootstrap CI95% for every individual model AND every ensemble strategy
on the test split.

Radiologist reviewers expect sens/spec, not just F1. This script outputs
them parallel to bootstrap_ci_test.csv.

Inputs:
  --responses-dir         responses/*_test.csv
  --ensemble-preds        data/outputs/ensemble_predictions_test.csv
  --gt                    data/outputs/ground_truth_7class.csv
  --n-bootstrap           default 10000
  --output                data/outputs/sens_spec_ci_test.csv

Output: long form
  unit, unit_type, classe, n, n_pos, n_neg,
  sens, sens_lo, sens_hi,
  spec, spec_lo, spec_hi,
  ppv,  ppv_lo,  ppv_hi,
  npv,  npv_lo,  npv_hi

Author: IPDA et al. PAP v2.1 Phase C+ (statistical strengthening).
"""

import argparse
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

V4_CLASSES = ["opacity", "pneumonia", "cardiomegaly", "effusion",
              "pneumothorax", "mass", "yesfinding"]


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return tp, fp, tn, fn


def metrics_from_conf(tp, fp, tn, fn):
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    ppv  = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    npv  = tn / (tn + fn) if (tn + fn) > 0 else float("nan")
    return sens, spec, ppv, npv


def bootstrap_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                       n_boot: int = 10000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    tp, fp, tn, fn = confusion(y_true, y_pred)
    point = metrics_from_conf(tp, fp, tn, fn)
    boots = np.empty((n_boot, 4))
    boots[:] = np.nan
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        b = metrics_from_conf(*confusion(y_true[idx], y_pred[idx]))
        boots[i] = b
    out = {}
    names = ["sens", "spec", "ppv", "npv"]
    for j, name in enumerate(names):
        valid = boots[:, j][~np.isnan(boots[:, j])]
        if len(valid) == 0:
            out[name] = (float("nan"), float("nan"), float("nan"))
            continue
        lo, hi = np.percentile(valid, [2.5, 97.5])
        out[name] = (point[j], float(lo), float(hi))
    out["n"] = int(n)
    out["n_pos"] = int(y_true.sum())
    out["n_neg"] = int(n - y_true.sum())
    out["tp"], out["fp"], out["tn"], out["fn"] = tp, fp, tn, fn
    return out


def load_individuals(responses_dir: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for f in sorted(glob(str(responses_dir / "*_test.csv"))):
        name = Path(f).name
        if "smoke" in name.lower(): continue
        v = name.replace("_test.csv", "")
        df = pd.read_csv(f)
        df = df[df["split"] == "test"].set_index("accession_number")
        for c in V4_CLASSES:
            df[c] = df[c].fillna(0).astype(int)
        out[v] = df[V4_CLASSES]
    return out


def load_ensembles(path: Path) -> pd.DataFrame:
    ens = pd.read_csv(path)
    return ens.pivot_table(index="accession_number",
                            columns=["strategy", "classe"],
                            values="prediction", aggfunc="first")


def load_gt(path: Path) -> pd.DataFrame:
    gt = pd.read_csv(path)
    gt = gt[gt["split"] == "test"].set_index("accession_number")
    for c in V4_CLASSES:
        gt[c] = gt[c].fillna(0).astype(int)
    return gt[V4_CLASSES]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--responses-dir", default="responses")
    ap.add_argument("--ensemble-preds", default="data/outputs/ensemble_predictions_test.csv")
    ap.add_argument("--gt", default="data/outputs/ground_truth_7class.csv")
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="data/outputs/sens_spec_ci_test.csv")
    args = ap.parse_args()

    gt = load_gt(Path(args.gt))
    print(f"GT test N={len(gt)}")
    inds = load_individuals(Path(args.responses_dir))
    ens = load_ensembles(Path(args.ensemble_preds))
    strategies = sorted(set(s for s, _ in ens.columns))

    common = gt.index
    for v in inds:
        common = common.intersection(inds[v].index)
    common = common.intersection(ens.index)
    common = sorted(common)
    gt_a = gt.loc[common]

    rows = []

    # Individuals
    print(f"\nBootstrapping individuals (B={args.n_bootstrap})...")
    for v, df in inds.items():
        df_a = df.loc[common]
        for cls in V4_CLASSES:
            m = bootstrap_metrics(gt_a[cls].values.astype(int),
                                   df_a[cls].values.astype(int),
                                   n_boot=args.n_bootstrap, seed=args.seed)
            rows.append({
                "unit": v, "unit_type": "individual", "classe": cls,
                "n": m["n"], "n_pos": m["n_pos"], "n_neg": m["n_neg"],
                "tp": m["tp"], "fp": m["fp"], "tn": m["tn"], "fn": m["fn"],
                "sens": round(m["sens"][0], 4),
                "sens_lo": round(m["sens"][1], 4),
                "sens_hi": round(m["sens"][2], 4),
                "spec": round(m["spec"][0], 4),
                "spec_lo": round(m["spec"][1], 4),
                "spec_hi": round(m["spec"][2], 4),
                "ppv": round(m["ppv"][0], 4),
                "ppv_lo": round(m["ppv"][1], 4),
                "ppv_hi": round(m["ppv"][2], 4),
                "npv": round(m["npv"][0], 4),
                "npv_lo": round(m["npv"][1], 4),
                "npv_hi": round(m["npv"][2], 4),
            })
        print(f"  done: {v}")

    # Ensembles
    print(f"\nBootstrapping ensembles (B={args.n_bootstrap})...")
    for s in strategies:
        for cls in V4_CLASSES:
            key = (s, cls)
            if key not in ens.columns: continue
            pred = ens.loc[common, key].fillna(0).astype(int).values
            m = bootstrap_metrics(gt_a[cls].values.astype(int), pred,
                                   n_boot=args.n_bootstrap, seed=args.seed)
            rows.append({
                "unit": s, "unit_type": "ensemble", "classe": cls,
                "n": m["n"], "n_pos": m["n_pos"], "n_neg": m["n_neg"],
                "tp": m["tp"], "fp": m["fp"], "tn": m["tn"], "fn": m["fn"],
                "sens": round(m["sens"][0], 4),
                "sens_lo": round(m["sens"][1], 4),
                "sens_hi": round(m["sens"][2], 4),
                "spec": round(m["spec"][0], 4),
                "spec_lo": round(m["spec"][1], 4),
                "spec_hi": round(m["spec"][2], 4),
                "ppv": round(m["ppv"][0], 4),
                "ppv_lo": round(m["ppv"][1], 4),
                "ppv_hi": round(m["ppv"][2], 4),
                "npv": round(m["npv"][0], 4),
                "npv_lo": round(m["npv"][1], 4),
                "npv_hi": round(m["npv"][2], 4),
            })
        print(f"  done: {s}")

    out = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"\nWrote: {args.output}  ({len(out)} rows)")
    # Quick console summary: macro sens/spec per unit
    print(f"\nMacro sens / spec per unit:")
    macro = out.groupby(["unit_type", "unit"]).agg(
        macro_sens=("sens", "mean"),
        macro_spec=("spec", "mean"),
    ).round(4).reset_index().sort_values("macro_sens", ascending=False)
    print(macro.to_string(index=False))


if __name__ == "__main__":
    main()
