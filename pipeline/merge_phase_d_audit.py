"""
merge_phase_d_audit.py
=======================

Aplica as correções da auditoria humana (Phase D) ao ground truth.

Entrada:
  - ground_truth_7class.csv (GT v1, antes da auditoria)
  - revisão_exportada.xlsx  (output do validator /audit, com FP/FN preenchidos)

Lógica de merge:
  Para cada case auditado:
    - Para cada classe em FP  → GT[classe] = 0
    - Para cada classe em FN  → GT[classe] = 1
    - "OK" / vazio em TRUE com FP/FN vazios → sem mudança (auditor manteve GT)
    - Se yesfinding vai pra 0 → limpa abnormalities_found

Nomenclatura:
  O validator usa "bacterial_pneumonia" no campo extracted (compat legacy).
  Mapeamos pra "pneumonia" (schema v4) ao escrever o GT.

Saídas:
  ground_truth_7class_v2.csv       — GT corrigido pós-Phase D
  ground_truth_7class_v2.xlsx      — versão formatada
  phase_d_changelog.csv            — audit trail (1 linha por mudança)
  phase_d_integrity_warnings.xlsx  — só os cases pós-merge que violam Rule 8 / entailment

Author: I.P. D'Amorim. PAP v2.2 (após Phase D).
"""

import argparse
import csv
import re
from pathlib import Path
from collections import Counter, defaultdict

import openpyxl
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

CLASSES_V4 = ["opacity", "pneumonia", "cardiomegaly", "effusion",
              "pneumothorax", "mass", "yesfinding"]
BINARY = ["opacity", "pneumonia", "cardiomegaly", "effusion",
          "pneumothorax", "mass"]

# Validator name -> v4 name
VALIDATOR_TO_V4 = {
    "bacterial_pneumonia": "pneumonia",
    "cadiomegaly": "cardiomegaly",  # typo
}

def normalize_class_name(s):
    s = (s or "").strip().lower()
    return VALIDATOR_TO_V4.get(s, s)


def parse_class_list(s):
    """Parse FP/FN string like 'opacity;\\nyesfinding' into set of v4 names."""
    if not s: return set()
    out = set()
    for tok in re.split(r"[;,\n]", str(s)):
        n = normalize_class_name(tok)
        if n in CLASSES_V4:
            out.add(n)
    return out


def load_gt(path):
    """Returns {acc: {classe: 0/1, ..., '_split', '_report', '_abn', '_extra'}}."""
    out = {}
    fieldnames = []
    with open(path, encoding='utf-8') as f:
        r = csv.DictReader(f)
        fieldnames = list(r.fieldnames or [])
        for row in r:
            acc = row["accession_number"]
            out[acc] = {
                **{c: int(row[c]) for c in CLASSES_V4},
                "_split": row["split"],
                "_report": row.get("report", ""),
                "_abn": row.get("abnormalities_found", "") or "",
                "_extra": {k: row.get(k, "") for k in fieldnames
                           if k not in (["accession_number","split","report"]
                                         + CLASSES_V4 + ["abnormalities_found"])},
            }
    return out, fieldnames


def load_audit(path):
    """Returns {acc: {'fp': set, 'fn': set, 'true': str, 'obs': str}}."""
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    ws = wb.active
    hdrs = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1))]
    idx = {h: i for i, h in enumerate(hdrs)}
    out = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row or row[idx["accession_number"]] is None: continue
        acc = str(row[idx["accession_number"]]).strip()
        out[acc] = {
            "fp": parse_class_list(row[idx.get("FP", -1)] if "FP" in idx else None),
            "fn": parse_class_list(row[idx.get("FN", -1)] if "FN" in idx else None),
            "true": (row[idx.get("TRUE", -1)] if "TRUE" in idx else "") or "",
            "obs": (row[idx.get("Observações", -1)] if "Observações" in idx else "") or "",
        }
    return out


def check_integrity(gt_row, abn):
    """Detecta violações pós-merge."""
    flags = []
    # Rule 8: yesfinding=1 deve ter binária OU abn
    if gt_row['yesfinding'] == 1:
        any_bin = any(gt_row[c] == 1 for c in BINARY)
        has_abn = bool((abn or "").strip())
        if not any_bin and not has_abn:
            flags.append("yesfinding=1 sem nenhuma evidência (viola Rule 8)")
    # Hierarquia: pneumonia=1 deve ter opacity=1
    if gt_row['pneumonia'] == 1 and gt_row['opacity'] == 0:
        flags.append("pneumonia=1 com opacity=0 (viola hierarquia v4)")
    # Inverso Rule 8: se yesfinding=0 mas alguma binária=1, viola
    if gt_row['yesfinding'] == 0:
        if any(gt_row[c] == 1 for c in BINARY):
            cls = [c for c in BINARY if gt_row[c] == 1]
            flags.append(f"yesfinding=0 mas binária(s)=1: {cls} (viola Rule 8 entailment)")
    return flags


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt-csv", default="data/outputs/ground_truth_7class.csv")
    ap.add_argument("--audit-xlsx", default="revisão_exportada.xlsx")
    ap.add_argument("--out-gt-csv", default="data/outputs/ground_truth_7class_v2.csv")
    ap.add_argument("--out-gt-xlsx", default="data/outputs/ground_truth_7class_v2.xlsx")
    ap.add_argument("--out-changelog", default="data/outputs/phase_d_changelog.csv")
    ap.add_argument("--out-integrity", default="data/outputs/phase_d_integrity_warnings.xlsx")
    args = ap.parse_args()

    gt, gt_fields = load_gt(args.gt_csv)
    audit = load_audit(args.audit_xlsx)

    print(f"GT original    : {len(gt)} rows")
    print(f"Audit revisados: {len(audit)} cases")
    print()

    changelog_rows = []
    changes_per_class = Counter()
    flips_1to0 = Counter()
    flips_0to1 = Counter()
    affected_accs = set()

    for acc, a in audit.items():
        if acc not in gt:
            print(f"WARN: {acc} no audit não existe no GT (pulado)")
            continue
        original = {c: gt[acc][c] for c in CLASSES_V4}
        # FP → 0
        for c in a['fp']:
            if c not in CLASSES_V4: continue
            old = gt[acc][c]
            if old != 0:
                gt[acc][c] = 0
                changes_per_class[c] += 1
                flips_1to0[c] += 1
                changelog_rows.append({
                    "accession_number": acc, "classe": c,
                    "from": old, "to": 0, "type": "FP_corrected",
                    "audit_obs": a['obs'],
                })
                affected_accs.add(acc)
        # FN → 1
        for c in a['fn']:
            if c not in CLASSES_V4: continue
            old = gt[acc][c]
            if old != 1:
                gt[acc][c] = 1
                changes_per_class[c] += 1
                flips_0to1[c] += 1
                changelog_rows.append({
                    "accession_number": acc, "classe": c,
                    "from": old, "to": 1, "type": "FN_corrected",
                    "audit_obs": a['obs'],
                })
                affected_accs.add(acc)
        # se yesfinding foi pra 0, limpa abn
        if gt[acc]['yesfinding'] == 0 and original['yesfinding'] == 1:
            gt[acc]['_abn'] = ""

    print(f"Cases com mudanças: {len(affected_accs)} / {len(audit)}")
    print(f"Total de flips:    {sum(changes_per_class.values())}")
    print()
    print("Mudanças por classe:")
    for c in CLASSES_V4:
        n1to0 = flips_1to0[c]
        n0to1 = flips_0to1[c]
        print(f"  {c:<16}  -1→0: {n1to0:>3}  -0→1: {n0to1:>3}")

    # ---- Integrity check ----
    warnings = []
    for acc, g in gt.items():
        flags = check_integrity(g, g['_abn'])
        if flags:
            warnings.append({"accession_number": acc, "flags": flags, "gt": g})
    print(f"\nIntegrity warnings pós-merge: {len(warnings)} cases")

    # ---- Write GT v2 csv ----
    Path(args.out_gt_csv).parent.mkdir(parents=True, exist_ok=True)
    out_cols = ["accession_number", "split", "report",
                *CLASSES_V4, "abnormalities_found"]
    # Preserva colunas extras do GT original
    extra_cols = list(gt[next(iter(gt))]['_extra'].keys()) if gt else []
    out_cols += extra_cols
    with open(args.out_gt_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_cols)
        w.writeheader()
        for acc, g in gt.items():
            row = {
                "accession_number": acc,
                "split": g["_split"],
                "report": g["_report"],
                "abnormalities_found": g["_abn"],
                **{c: g[c] for c in CLASSES_V4},
                **g["_extra"],
            }
            w.writerow(row)
    print(f"\nWrote: {args.out_gt_csv}")

    # ---- Write changelog ----
    with open(args.out_changelog, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["accession_number","classe","from","to","type","audit_obs"])
        w.writeheader()
        for r in changelog_rows:
            w.writerow(r)
    print(f"Wrote: {args.out_changelog}  ({len(changelog_rows)} mudanças)")

    # ---- Write GT v2 xlsx (formatado) ----
    wb = Workbook()
    ws = wb.active
    ws.title = "ground_truth_v2"
    NAVY, GREEN, AMBER, LIGHT = "1F3864", "C6EFCE", "FFEB9C", "F2F2F2"
    for ci, c in enumerate(out_cols, 1):
        cell = ws.cell(row=1, column=ci, value=c)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", start_color=NAVY)
        cell.alignment = Alignment(horizontal="center", wrap_text=True)
    ws.row_dimensions[1].height = 28
    for ri, (acc, g) in enumerate(gt.items(), 2):
        row = {
            "accession_number": acc,
            "split": g["_split"],
            "report": g["_report"],
            "abnormalities_found": g["_abn"],
            **{c: g[c] for c in CLASSES_V4},
            **g["_extra"],
        }
        was_audited = acc in affected_accs
        for ci, c in enumerate(out_cols, 1):
            v = row.get(c, "")
            cell = ws.cell(row=ri, column=ci, value=v)
            cell.font = Font(size=10)
            if c in CLASSES_V4 and v == 1:
                cell.fill = PatternFill("solid", start_color=GREEN)
                cell.alignment = Alignment(horizontal="center")
            elif c in CLASSES_V4:
                cell.alignment = Alignment(horizontal="center")
            elif c == "accession_number" and was_audited:
                cell.fill = PatternFill("solid", start_color=AMBER)
                cell.font = Font(size=10, bold=True)
    widths = {"accession_number": 16, "split": 8, "report": 60,
              "abnormalities_found": 50}
    for ci, c in enumerate(out_cols, 1):
        ws.column_dimensions[get_column_letter(ci)].width = widths.get(c, 12)
    ws.freeze_panes = "D2"

    # Summary sheet
    ws2 = wb.create_sheet("summary")
    info = [
        ("GT rows total", len(gt)),
        ("Cases auditados (Phase D)", len(audit)),
        ("Cases com mudanças", len(affected_accs)),
        ("Total de flips de classe", sum(changes_per_class.values())),
        ("", ""),
    ]
    for c in CLASSES_V4:
        info.append((f"{c} (1→0)", flips_1to0[c]))
        info.append((f"{c} (0→1)", flips_0to1[c]))
    info += [("", ""),
             ("Integrity warnings pós-merge", len(warnings))]
    for ri, (k, v) in enumerate(info, 1):
        ws2.cell(row=ri, column=1, value=k).font = Font(bold=True)
        ws2.cell(row=ri, column=2, value=v)
    ws2.column_dimensions["A"].width = 32
    ws2.column_dimensions["B"].width = 16

    # Prevalência antes/depois
    ws3 = wb.create_sheet("prevalencia_antes_depois")
    ws3.cell(row=1, column=1, value="classe").font = Font(bold=True, color="FFFFFF")
    ws3.cell(row=1, column=1).fill = PatternFill("solid", start_color=NAVY)
    for ci, lbl in enumerate(["classe","n_pos_v2","total","%_v2"], 1):
        c = ws3.cell(row=1, column=ci, value=lbl)
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", start_color=NAVY)
    for ri, c in enumerate(CLASSES_V4, 2):
        pos = sum(1 for g in gt.values() if g[c] == 1)
        ws3.cell(row=ri, column=1, value=c)
        ws3.cell(row=ri, column=2, value=pos)
        ws3.cell(row=ri, column=3, value=len(gt))
        ws3.cell(row=ri, column=4, value=f"{100*pos/len(gt):.2f}%")
    for col, w in [(1,18),(2,12),(3,8),(4,10)]:
        ws3.column_dimensions[get_column_letter(col)].width = w

    wb.save(args.out_gt_xlsx)
    print(f"Wrote: {args.out_gt_xlsx}")

    # ---- Integrity warnings xlsx ----
    if warnings:
        wbw = Workbook()
        wsw = wbw.active
        wsw.title = "integrity_warnings"
        hdrs = ["accession_number", "split", "flags", *CLASSES_V4, "abnormalities_found"]
        for ci, h in enumerate(hdrs, 1):
            c = wsw.cell(row=1, column=ci, value=h)
            c.font = Font(bold=True, color="FFFFFF")
            c.fill = PatternFill("solid", start_color=NAVY)
            c.alignment = Alignment(horizontal="center", wrap_text=True)
        wsw.row_dimensions[1].height = 28
        for ri, w in enumerate(warnings, 2):
            g = w["gt"]
            wsw.cell(row=ri, column=1, value=w["accession_number"])
            wsw.cell(row=ri, column=2, value=g["_split"])
            wsw.cell(row=ri, column=3, value=" | ".join(w["flags"]))
            for ci, c in enumerate(CLASSES_V4, 4):
                v = g[c]
                cc = wsw.cell(row=ri, column=ci, value=v)
                if v == 1:
                    cc.fill = PatternFill("solid", start_color=GREEN)
                cc.alignment = Alignment(horizontal="center")
            wsw.cell(row=ri, column=4 + len(CLASSES_V4), value=g["_abn"])
        widths = [16, 8, 60] + [11]*len(CLASSES_V4) + [50]
        for ci, w in enumerate(widths, 1):
            wsw.column_dimensions[get_column_letter(ci)].width = w
        wbw.save(args.out_integrity)
        print(f"Wrote: {args.out_integrity}  ({len(warnings)} cases pra checar)")

    # Prevalência antes/depois
    print(f"\nPrevalência por classe no GT v2:")
    for c in CLASSES_V4:
        pos = sum(1 for g in gt.values() if g[c] == 1)
        print(f"  {c:<16}  {pos:>5d} / {len(gt)}  ({100*pos/len(gt):5.2f}%)")


if __name__ == "__main__":
    main()
