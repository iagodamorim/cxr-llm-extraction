"""
paired_delta_analysis.py
=========================

Tier 1 statistical strengthening (peer-review armor):

  (A) Paired bootstrap of the delta F1(Hierarchical) - F1(best individual)
      with B=10000 resamples. Reports CI95% of the delta itself.
      If CI lower bound > 0  ->  superiority is statistically supported.
      If CI crosses 0        ->  reframe as "non-inferior with practical gain".

  (B) Exact binomial McNemar test for paired classifier comparison,
      per-class, comparing Hierarchical Decision vs every individual model
      and vs Specialist Routing. Uses statsmodels mcnemar(exact=True) when
      n_discordant < 25, else continuity-corrected chi-square.

Inputs:
  --responses-dir         responses/*_test.csv
  --ensemble-preds        data/outputs/ensemble_predictions_test.csv
  --gt                    data/outputs/ground_truth_7class.csv
  --n-bootstrap           default 10000
  --reference-individual  default 'sonnet-4-6'  (the best individual on test)
  --output-prefix         data/outputs/paired_delta

Outputs:
  <prefix>_bootstrap.csv   long format: comparison, classe, delta_point, delta_lo, delta_hi, p_one_sided
  <prefix>_mcnemar.csv     long format: comparison, classe, n_discordant, b, c, p_exact, test_kind

Author: IPDA et al. PAP v2.1 Phase C+ (statistical strengthening, 2026-05-19).
"""

import argparse
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

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


def macro_f1(y_true_matrix: np.ndarray, y_pred_matrix: np.ndarray) -> float:
    # both: (N, C) binary
    f1s = [f1_binary(y_true_matrix[:, c], y_pred_matrix[:, c])
           for c in range(y_true_matrix.shape[1])]
    return float(np.mean(f1s))


def load_individual(responses_dir: Path, vendor: str) -> pd.DataFrame:
    f = responses_dir / f"{vendor}_test.csv"
    df = pd.read_csv(f)
    df = df[df["split"] == "test"].set_index("accession_number")
    for c in V4_CLASSES:
        df[c] = df[c].fillna(0).astype(int)
    return df[V4_CLASSES]


def load_ensemble_wide(ens_path: Path) -> pd.DataFrame:
    """Return DataFrame indexed by accession, columns (strategy, classe)."""
    ens = pd.read_csv(ens_path)
    wide = ens.pivot_table(index="accession_number",
                            columns=["strategy", "classe"],
                            values="prediction", aggfunc="first")
    return wide


def load_gt(path: Path) -> pd.DataFrame:
    gt = pd.read_csv(path)
    gt = gt[gt["split"] == "test"].set_index("accession_number")
    for c in V4_CLASSES:
        gt[c] = gt[c].fillna(0).astype(int)
    return gt[V4_CLASSES]


def paired_bootstrap_delta(
    y_true_mat: np.ndarray,
    y_a_mat: np.ndarray,
    y_b_mat: np.ndarray,
    n_boot: int = 10000,
    seed: int = 42,
) -> tuple[float, float, float, float]:
    """
    Returns (delta_point, delta_lo, delta_hi, p_one_sided)
    where delta = macro_F1(A) - macro_F1(B), computed pairwise on each resample.
    p_one_sided = fraction of bootstrap deltas <= 0 (testing A > B).
    """
    n = y_true_mat.shape[0]
    rng = np.random.default_rng(seed)
    point = macro_f1(y_true_mat, y_a_mat) - macro_f1(y_true_mat, y_b_mat)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = (macro_f1(y_true_mat[idx], y_a_mat[idx])
                    - macro_f1(y_true_mat[idx], y_b_mat[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p_one_sided = float((boots <= 0).mean())
    return float(point), float(lo), float(hi), p_one_sided


def paired_bootstrap_delta_per_class(
    y_true: np.ndarray, y_a: np.ndarray, y_b: np.ndarray,
    n_boot: int = 10000, seed: int = 42,
) -> tuple[float, float, float, float]:
    n = len(y_true)
    rng = np.random.default_rng(seed)
    point = f1_binary(y_true, y_a) - f1_binary(y_true, y_b)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = f1_binary(y_true[idx], y_a[idx]) - f1_binary(y_true[idx], y_b[idx])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p_one_sided = float((boots <= 0).mean())
    return float(point), float(lo), float(hi), p_one_sided


def mcnemar_paired(y_true: np.ndarray, y_a: np.ndarray, y_b: np.ndarray) -> dict:
    """
    McNemar comparing two classifiers A and B on the SAME labels.
    - b = cases where A correct, B wrong
    - c = cases where A wrong, B correct
    - exact binomial used when n_discordant = b+c < 25, else chi2 with continuity correction
    """
    correct_a = (y_a == y_true).astype(int)
    correct_b = (y_b == y_true).astype(int)
    b = int(((correct_a == 1) & (correct_b == 0)).sum())
    c = int(((correct_a == 0) & (correct_b == 1)).sum())
    table = np.array([[0, b], [c, 0]])
    n_disc = b + c
    if n_disc == 0:
        return {"b": b, "c": c, "n_discordant": 0, "p": 1.0, "test_kind": "no_discordance"}
    use_exact = n_disc < 25
    result = mcnemar(table, exact=use_exact, correction=not use_exact)
    return {
        "b": b, "c": c, "n_discordant": n_disc,
        "p": float(result.pvalue),
        "test_kind": "exact_binomial" if use_exact else "chi2_continuity",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--responses-dir", default="responses")
    ap.add_argument("--ensemble-preds", default="data/outputs/ensemble_predictions_test.csv")
    ap.add_argument("--gt", default="data/outputs/ground_truth_7class.csv")
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    ap.add_argument("--reference-individual", default="sonnet-4-6",
                    help="Vendor key of the best individual model (will be compared to Hierarchical).")
    ap.add_argument("--output-prefix", default="data/outputs/paired_delta")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"Loading data ...")
    gt = load_gt(Path(args.gt))
    print(f"  GT test reports: {len(gt)}")

    # Individuals
    individuals = {}
    for f in sorted(glob(str(Path(args.responses_dir) / "*_test.csv"))):
        name = Path(f).name
        if "smoke" in name.lower():
            continue
        vendor = name.replace("_test.csv", "")
        individuals[vendor] = load_individual(Path(args.responses_dir), vendor)
    print(f"  individual models: {list(individuals.keys())}")

    # Ensembles
    ens_wide = load_ensemble_wide(Path(args.ensemble_preds))
    strategies = sorted(set(s for s, _ in ens_wide.columns))
    print(f"  ensemble strategies: {strategies}")

    # Align indices
    common = gt.index.intersection(ens_wide.index)
    for v, df in individuals.items():
        common = common.intersection(df.index)
    common = sorted(common)
    print(f"  aligned N: {len(common)}")
    gt_aligned = gt.loc[common]
    gt_mat = gt_aligned[V4_CLASSES].values.astype(int)

    # Build prediction matrices for every "unit" of interest
    units: dict[str, np.ndarray] = {}
    for v, df in individuals.items():
        units[v] = df.loc[common, V4_CLASSES].values.astype(int)
    for s in strategies:
        mat = np.zeros((len(common), len(V4_CLASSES)), dtype=int)
        for j, cls in enumerate(V4_CLASSES):
            key = (s, cls)
            if key in ens_wide.columns:
                mat[:, j] = ens_wide.loc[common, key].fillna(0).astype(int).values
        units[s] = mat

    # Comparisons of interest:
    #   Hierarchical vs every individual
    #   Hierarchical vs Specialist Routing
    #   Specialist Routing vs reference individual (sanity check)
    ref_ind = args.reference_individual
    assert ref_ind in individuals, f"Reference individual {ref_ind} not found among {list(individuals.keys())}"

    comparisons = []
    for v in individuals:
        comparisons.append(("hierarchical_decision", v))
    comparisons.append(("hierarchical_decision", "specialist_routing"))
    comparisons.append(("specialist_routing", ref_ind))

    # ---------- A. Paired bootstrap ----------
    print(f"\n=== Tier 1A: Paired bootstrap (B={args.n_bootstrap}) ===")
    boot_rows = []
    for a, b in comparisons:
        a_mat = units[a]
        b_mat = units[b]
        # macro delta
        d, lo, hi, p1 = paired_bootstrap_delta(gt_mat, a_mat, b_mat,
                                                n_boot=args.n_bootstrap, seed=args.seed)
        boot_rows.append({
            "comparison": f"{a} vs {b}", "classe": "macro",
            "delta_f1": round(d, 4),
            "delta_lo": round(lo, 4), "delta_hi": round(hi, 4),
            "p_one_sided_A_gt_B": round(p1, 5),
            "n": int(len(common)),
        })
        # per-class deltas (useful for showing where the gain comes from)
        for j, cls in enumerate(V4_CLASSES):
            d, lo, hi, p1 = paired_bootstrap_delta_per_class(
                gt_mat[:, j], a_mat[:, j], b_mat[:, j],
                n_boot=args.n_bootstrap, seed=args.seed)
            boot_rows.append({
                "comparison": f"{a} vs {b}", "classe": cls,
                "delta_f1": round(d, 4),
                "delta_lo": round(lo, 4), "delta_hi": round(hi, 4),
                "p_one_sided_A_gt_B": round(p1, 5),
                "n": int(len(common)),
            })
        sig = "***" if boot_rows[-8]["delta_lo"] > 0 else (
              " * " if boot_rows[-8]["delta_lo"] >= -0.001 else "   ")
        print(f"  {sig} {a} vs {b}  macro Δ = {boot_rows[-8]['delta_f1']:+.4f} "
              f"[{boot_rows[-8]['delta_lo']:+.4f}, {boot_rows[-8]['delta_hi']:+.4f}]  "
              f"p1={boot_rows[-8]['p_one_sided_A_gt_B']:.4f}")

    boot_df = pd.DataFrame(boot_rows)
    boot_path = Path(args.output_prefix + "_bootstrap.csv")
    boot_path.parent.mkdir(parents=True, exist_ok=True)
    boot_df.to_csv(boot_path, index=False)
    print(f"\nWrote: {boot_path}  ({len(boot_df)} rows)")

    # ---------- B. McNemar exact ----------
    print(f"\n=== Tier 1B: McNemar paired (exact when n_disc<25) ===")
    mc_rows = []
    for a, b in comparisons:
        a_mat = units[a]
        b_mat = units[b]
        for j, cls in enumerate(V4_CLASSES):
            res = mcnemar_paired(gt_mat[:, j], a_mat[:, j], b_mat[:, j])
            res.update({"comparison": f"{a} vs {b}", "classe": cls})
            mc_rows.append(res)

    mc_df = pd.DataFrame(mc_rows)[
        ["comparison", "classe", "n_discordant", "b", "c", "p", "test_kind"]
    ]
    mc_df["p"] = mc_df["p"].round(5)
    mc_path = Path(args.output_prefix + "_mcnemar.csv")
    mc_df.to_csv(mc_path, index=False)
    print(f"Wrote: {mc_path}  ({len(mc_df)} rows)")

    # Console summary: per-comparison number of classes with p<0.05 raw
    print(f"\nSummary (raw p<0.05, no multiple-testing correction):")
    for (cmp,), grp in mc_df.groupby(["comparison"]):
        n_sig = int((grp["p"] < 0.05).sum())
        print(f"  {cmp:60s}  {n_sig}/{len(grp)} classes raw-significant")


if __name__ == "__main__":
    main()
