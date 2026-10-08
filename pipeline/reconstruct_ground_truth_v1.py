"""
reconstruct_ground_truth_v1.py
================================

Reconstructs the v1 7-class ground truth for all 1075 reports by parsing
the existing annotated spreadsheets. Both source files contain LLM
extractions (Gemini) with the user's manual annotations of false positives
(FP) and false negatives (FN). The TRUE ground truth is recovered by:

    GT[class] = 1  if (extracted[class] == 1 AND class NOT in FP)
                 OR (class in FN)
              = 0  otherwise

The two source files use different formats:
  - 295-report file: "ground_truth" column is a JSON string of the LLM
    extraction. Class names: opacity, cardiomegaly, effusion, mass,
    pneumothorax, bacterial_pneumonia, yesfinding.
  - 780-report file: "extracted" column is a pipe-separated string of
    "class:0|1". Class names INCONSISTENT: 423 rows use "bacterial_pneumonia",
    357 rows use "consolidation" (schema partially migrated mid-run).
    This script normalizes ALL of them to "bacterial_pneumonia" for the v1
    schema; the v3 8-class split (consolidation + pneumonia separated) is
    handled by re_annotate_helper.py.

OUTPUT:
  ground_truth_v1.csv with columns:
    accession_number, split, opacity, cardiomegaly, effusion, mass,
    pneumothorax, bacterial_pneumonia, yesfinding, abnormalities_found,
    annotation_status (OK / corrected), n_corrections

Author: I.P. D'Amorim et al. PAP v2.1.
"""

import argparse
import json
import re
import sys
from pathlib import Path
import pandas as pd
import openpyxl

V1_CLASSES = ["opacity", "cardiomegaly", "effusion", "mass", "pneumothorax",
              "bacterial_pneumonia", "yesfinding"]


def parse_class_set(s):
    """Parse FP/FN strings like 'opacity;\\nyesfinding;' into a set of canonical class names."""
    if not s:
        return set()
    out = set()
    for tok in re.split(r"[;,\n]", str(s)):
        tok = tok.strip().lower()
        if not tok:
            continue
        # normalize: cadiomegaly typo -> cardiomegaly; consolidation -> bacterial_pneumonia (v1 schema)
        tok = tok.replace("cadiomegaly", "cardiomegaly")
        if tok == "consolidation":
            tok = "bacterial_pneumonia"
        if tok in V1_CLASSES:
            out.add(tok)
        elif tok.replace(" ", "") in V1_CLASSES:
            out.add(tok.replace(" ", ""))
    return out


def parse_json_extraction(s):
    """Parse the JSON 'ground_truth' column from the 295-report file."""
    try:
        d = json.loads(s)
    except Exception:
        return None, ""
    findings = d.get("findings", [])
    out = {}
    abn_found = ""
    for f in findings:
        cname = f.get("class_name", "").strip().lower()
        if cname == "consolidation":
            cname = "bacterial_pneumonia"  # normalize to v1
        presence = f.get("presence", "0")
        try:
            out[cname] = int(presence)
        except (ValueError, TypeError):
            out[cname] = 0
        if cname == "yesfinding":
            abn_found = f.get("abnormalities_found", "") or ""
    return out, abn_found


def parse_pipe_extraction(s):
    """Parse the pipe-separated 'extracted' column from the 780-report file."""
    if not s:
        return None, ""
    s = str(s)
    out = {}
    abn_found = ""
    parts = s.split("|")
    for p in parts:
        p = p.strip()
        if ":" not in p:
            continue
        cname, rest = p.split(":", 1)
        cname = cname.strip().lower()
        if cname == "consolidation":
            cname = "bacterial_pneumonia"  # normalize to v1
        rest = rest.strip()
        # rest might be just "0" or "1" or "1 (abnormalities list)"
        m = re.match(r"^([01])(?:\s*\((.*)\))?\s*$", rest, flags=re.DOTALL)
        if m:
            out[cname] = int(m.group(1))
            if cname == "yesfinding" and m.group(2):
                abn_found = m.group(2).strip()
        else:
            # Fallback: take first digit
            digits = re.findall(r"[01]", rest)
            if digits:
                out[cname] = int(digits[0])
    return out, abn_found


def reconstruct_one(extracted: dict, fp: set, fn: set) -> dict:
    """Apply: GT = (ext AND not FP) OR FN."""
    gt = {}
    for c in V1_CLASSES:
        ext = extracted.get(c, 0)
        was_fp = c in fp
        was_fn = c in fn
        if was_fn:
            gt[c] = 1
        elif was_fp:
            gt[c] = 0
        else:
            gt[c] = ext
    return gt


def process_dev_file(path: Path, sheet: str) -> pd.DataFrame:
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    sh = wb[sheet]
    rows = []
    seen = set()
    for row in sh.iter_rows(min_row=2, values_only=True):
        if row[0] is None: continue
        acc = str(row[0]).strip()
        if acc in seen: continue
        seen.add(acc)
        # Columns: accession, report, ground_truth (= extraction), VP, FP, FN, Observações
        gt_json = row[2]
        vp, fp_str, fn_str, obs = row[3], row[4], row[5], row[6]
        ext, abn_found = parse_json_extraction(gt_json)
        if ext is None:
            print(f"  WARN: could not parse JSON for {acc}")
            continue
        fp = parse_class_set(fp_str)
        fn = parse_class_set(fn_str)
        gt = reconstruct_one(ext, fp, fn)
        # If user wrote VP="OK" only, no corrections (fp+fn empty); else status="corrected"
        is_ok = vp is not None and str(vp).strip().upper() == "OK" and not fp and not fn
        status = "OK_as_extracted" if is_ok else "corrected"
        n_corr = len(fp) + len(fn)
        rows.append({
            "accession_number": acc,
            "split": "dev",
            **gt,
            "abnormalities_found": abn_found,
            "annotation_status": status,
            "n_corrections": n_corr,
            "obs": obs or "",
        })
    return pd.DataFrame(rows)


def process_test_file(path: Path, sheet: str) -> pd.DataFrame:
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    sh = wb[sheet]
    rows = []
    seen = set()
    for row in sh.iter_rows(min_row=2, values_only=True):
        if row[0] is None: continue
        acc = str(row[0]).strip()
        if acc in seen: continue
        seen.add(acc)
        # Columns: accession, report, extracted (pipe), TRUE, FP, FN, Observações
        ext_str = row[2]
        true_col, fp_str, fn_str, obs = row[3], row[4], row[5], row[6]
        ext, abn_found = parse_pipe_extraction(ext_str)
        if ext is None:
            print(f"  WARN: could not parse pipe for {acc}")
            continue
        fp = parse_class_set(fp_str)
        fn = parse_class_set(fn_str)
        gt = reconstruct_one(ext, fp, fn)
        is_ok = true_col is not None and str(true_col).strip().upper() == "OK" and not fp and not fn
        status = "OK_as_extracted" if is_ok else "corrected"
        n_corr = len(fp) + len(fn)
        rows.append({
            "accession_number": acc,
            "split": "test",
            **gt,
            "abnormalities_found": abn_found,
            "annotation_status": status,
            "n_corrections": n_corr,
            "obs": obs or "",
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dev-xlsx", required=True)
    ap.add_argument("--test-xlsx", required=True)
    ap.add_argument("--dev-sheet", default="sampled_300_reports_with_ground")
    ap.add_argument("--test-sheet", default="Validacao")
    ap.add_argument("--output", default="ground_truth_v1.csv")
    args = ap.parse_args()

    print(f"Processing dev (295): {args.dev_xlsx}")
    dev = process_dev_file(Path(args.dev_xlsx), args.dev_sheet)
    print(f"  -> {len(dev)} rows; corrected={int((dev['annotation_status']=='corrected').sum())}")

    print(f"Processing test (780): {args.test_xlsx}")
    test = process_test_file(Path(args.test_xlsx), args.test_sheet)
    print(f"  -> {len(test)} rows; corrected={int((test['annotation_status']=='corrected').sum())}")

    full = pd.concat([dev, test], ignore_index=True)
    full.to_csv(args.output, index=False)

    print(f"\nWrote: {args.output} ({len(full)} rows)")
    print("\nClass prevalence in reconstructed v1 GT:")
    n = len(full)
    for c in V1_CLASSES:
        pos = int(full[c].sum())
        print(f"  {c:>20s}  {pos:>5d} / {n}  ({100*pos/n:5.2f}%)")


if __name__ == "__main__":
    main()
