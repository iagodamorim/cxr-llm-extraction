"""
fdr_sensitivity.py
===================

Tier 2E peer-review armor: Benjamini-Hochberg FDR sensitivity analysis
on top of the existing McNemar grid (which uses Bonferroni in the
primary analysis).

Bonferroni at alpha = 2.6e-6 (385 tests) is extremely conservative;
some reviewers will request a less-restrictive multiple-testing correction.
This script adds a `p_bh` column with BH-adjusted q-values and a
`significant_bh` flag, computed across all rows of the input grid.

Inputs:
  --input    data/outputs/mcnemar_grid_test.csv
  --output   data/outputs/mcnemar_grid_test_with_fdr.csv

Output: same columns as input plus p_bh (q-value), significant_bh (q<0.05).

Author: IPDA et al. PAP v2.1 Phase C+ (statistical strengthening).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="data/outputs/mcnemar_grid_test.csv")
    ap.add_argument("--output", default="data/outputs/mcnemar_grid_test_with_fdr.csv")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")

    pvals = df["p_value"].astype(float).values
    rej_bh, q_bh, _, _ = multipletests(pvals, alpha=args.alpha, method="fdr_bh")
    df["p_bh"] = np.round(q_bh, 5)
    df["significant_bh"] = rej_bh

    # Also recompute the Bonferroni flag for sanity (keep p_bonferroni from input)
    n_tests = len(df)
    alpha_bonf = args.alpha / n_tests
    df["alpha_bonferroni"] = alpha_bonf
    df["significant_bonferroni"] = df["p_value"] < alpha_bonf

    df.to_csv(args.output, index=False)
    print(f"Wrote: {args.output}")
    print(f"\nMultiple-testing summary (alpha={args.alpha}, n_tests={n_tests}):")
    print(f"  Bonferroni alpha = {alpha_bonf:.3g}")
    print(f"  significant (Bonferroni): {int(df['significant_bonferroni'].sum())} / {n_tests}")
    print(f"  significant (BH-FDR q<{args.alpha}): {int(df['significant_bh'].sum())} / {n_tests}")

    # Per-class top hits under FDR
    print(f"\nClasses with at least one BH-significant comparison:")
    sig = df[df["significant_bh"]]
    if len(sig) == 0:
        print("  (none)")
    else:
        by_cls = sig.groupby("classe").size().sort_values(ascending=False)
        for cls, n in by_cls.items():
            print(f"  {cls:15s}  {n} comparisons")
        print(f"\nTop 15 most-significant BH hits:")
        top = sig.sort_values("p_bh").head(15)[
            ["unit_a", "unit_b", "classe", "p_value", "p_bh"]
        ]
        print(top.to_string(index=False))


if __name__ == "__main__":
    main()
