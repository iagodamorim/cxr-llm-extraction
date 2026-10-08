"""
rename_pneumonia_in_gt.py
==========================

Rename column `bacterial_pneumonia` -> `pneumonia` in ground_truth_v1.csv,
producing ground_truth_v4.csv ready to compare against the 7-class prompt
v4 outputs.

Per the v3->v4 schema decision (May 2026): the v1 `bacterial_pneumonia`
class is RENAMED (not redefined) to `pneumonia`. Definition broadens in
the prompt, but the GT positives — laudos that explicitly framed a
pneumonia process — were ALREADY annotated under bacterial_pneumonia
by IPDA. No re-annotation needed.

Usage:
    python pipeline/rename_pneumonia_in_gt.py \\
        --in  data/outputs/ground_truth_v1.csv \\
        --out data/outputs/ground_truth_v4.csv

Author: IPDA et al. PAP v2.1 amendment (schema 8->7, May 2026).
"""

import argparse
import sys
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", required=True, help="ground_truth_v1.csv")
    ap.add_argument("--out", required=True, help="output csv (v4 schema)")
    args = ap.parse_args()

    df = pd.read_csv(args.inp)

    if "bacterial_pneumonia" not in df.columns:
        sys.exit(f"ERROR: column 'bacterial_pneumonia' not found in {args.inp}; "
                 f"got columns: {list(df.columns)}")
    if "pneumonia" in df.columns:
        sys.exit(f"ERROR: column 'pneumonia' already exists in {args.inp}; "
                 f"refusing to overwrite.")

    df = df.rename(columns={"bacterial_pneumonia": "pneumonia"})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    n_pos = int(df["pneumonia"].sum())
    print(f"Wrote: {args.out}")
    print(f"  rows: {len(df)}")
    print(f"  pneumonia positives: {n_pos} ({100*n_pos/len(df):.2f}%)")


if __name__ == "__main__":
    main()
