"""
re_annotate_v3_helper.py
==========================

Generates a review spreadsheet for the consolidation/pneumonia split
introduced in prompt v3 (PAP v2.1). The user reviews ONLY the cases
where heuristic keyword matching flags a likely consolidation or
pneumonia mention; the remaining ~875 cases are auto-assigned (both = 0)
and do not require manual review.

INPUT:
  --gt-csv         output of reconstruct_ground_truth_v1.py
                   (one row per report with v1 7-class GT)
  --reports-csv    output of build_master_input.py
                   (one row per report with full Brazilian Portuguese text)

LOGIC:
  For each report, look for keyword evidence of:
    (a) FOCAL CONSOLIDATION (descriptive)
    (b) EXPLICIT PNEUMONIA FRAMING (interpretive)

  CONSOLIDATION keywords (substring, case-insensitive):
    consolidação, consolidacao, consolidação alveolar,
    preenchimento alveolar, broncograma aéreo, broncograma aereo,
    opacidade homogênea com broncograma

  PNEUMONIA keywords:
    pneumonia, pneumônica, pneumonica, processo pneumônico,
    processo pneumonico, broncopneumonia, infiltrado pneumônico,
    infiltrado pneumonico, compatível com pneumonia,
    compativel com pneumonia, sugestivo de pneumonia

  AUTO-SUGGESTIONS (suggested_consolidation, suggested_pneumonia):
    suggested_consolidation = 1 if ANY consolidation OR pneumonia keyword matched
    suggested_pneumonia     = 1 if ANY pneumonia keyword matched
    suggested_*             = 0 otherwise

  RATIONALE: pneumonia is a sub-type of consolidation in radiologic terms,
  so a positive pneumonia mention implies consolidation as well. A
  descriptive consolidation without pneumonic framing is consolidation=1
  but pneumonia=0.

OUTPUT:
  consolidation_pneumonia_review.xlsx with two sheets:

  Sheet 1 ("REVIEW_NEEDED"): rows where at least one keyword matched.
    Columns:
      accession_number, split, report, v1_bacterial_pneumonia,
      keywords_consolidation, keywords_pneumonia,
      suggested_consolidation, suggested_pneumonia,
      USER_consolidation, USER_pneumonia, USER_notes
    The two USER_* columns are pre-filled with the suggestions; user
    edits them where necessary and saves.

  Sheet 2 ("AUTO_BOTH_ZERO"): rows where no keyword matched.
    Columns: accession_number, split, report
    These rows are auto-assigned consolidation = pneumonia = 0 and do
    NOT require user review unless the user wants to spot-check.

After review, run merge_v3_groundtruth.py to combine USER_* columns from
Sheet 1 with the AUTO_BOTH_ZERO sheet and produce ground_truth_v3.csv.

Author: I.P. D'Amorim et al. PAP v2.1.
"""

import argparse
import re
import sys
from pathlib import Path
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter


CONSOLIDATION_PATTERNS = [
    r"consolida[çc][ãa]o",
    r"preenchimento alveolar",
    r"broncograma a[ée]reo",
    r"opacidade homog[êe]nea (com|associada a) broncograma",
]

PNEUMONIA_PATTERNS = [
    r"pneumonia",            # catches: pneumonia, pneumonia comunitária, pneumonia atípica
    r"pneum[ôo]nic[ao]",     # catches: pneumônico/pneumônica
    r"broncopneumonia",
    r"infiltrado pneum[ôo]nico",
    r"compat[íi]vel com pneumonia",
    r"sugestivo de pneumonia",
    r"sugest[ãa]o de pneumonia",
    r"processo pneum[ôo]nico",
]

# Filter out negated mentions (e.g., "sem consolidação", "ausência de pneumonia")
NEGATION_PREFIXES = [
    r"sem ",
    r"aus[êe]ncia de ",
    r"sem evid[êe]ncia de ",
    r"sem sinais de ",
    r"n[ãa]o h[áa] ",
    r"descartad[ao] ",
    r"afastad[ao] ",
]

# A simple negation check: if a matched keyword is immediately preceded by
# one of the negation prefixes within ~25 chars, ignore that match.
NEG_RE = re.compile(
    r"(?:" + "|".join(NEGATION_PREFIXES) + r")[\w\s,]{0,25}$",
    re.IGNORECASE,
)


def find_matches(text: str, patterns: list[str]) -> list[str]:
    """Return list of distinct matched substrings (lowercased)."""
    text_low = text.lower()
    hits = []
    for pat in patterns:
        for m in re.finditer(pat, text_low):
            start = m.start()
            preceding = text_low[max(0, start - 40):start]
            if NEG_RE.search(preceding):
                continue
            hits.append(m.group(0))
    return sorted(set(hits))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt-csv", required=True, help="reconstructed v1 GT csv")
    ap.add_argument("--reports-csv", required=True, help="master reports csv")
    ap.add_argument("--output", default="consolidation_pneumonia_review.xlsx")
    args = ap.parse_args()

    gt = pd.read_csv(args.gt_csv)
    rep = pd.read_csv(args.reports_csv)

    merged = gt.merge(rep[["accession_number", "report"]], on="accession_number", how="left")
    if merged["report"].isna().any():
        print(f"WARN: {int(merged['report'].isna().sum())} reports missing text in master")

    review_rows = []
    auto_rows = []

    for _, row in merged.iterrows():
        text = str(row.get("report") or "")
        c_hits = find_matches(text, CONSOLIDATION_PATTERNS)
        p_hits = find_matches(text, PNEUMONIA_PATTERNS)

        if not c_hits and not p_hits:
            auto_rows.append({
                "accession_number": row["accession_number"],
                "split": row["split"],
                "report": text,
            })
            continue

        suggested_pneumonia = 1 if p_hits else 0
        suggested_consolidation = 1 if (c_hits or p_hits) else 0
        review_rows.append({
            "accession_number": row["accession_number"],
            "split": row["split"],
            "report": text,
            "v1_bacterial_pneumonia": int(row.get("bacterial_pneumonia", 0)),
            "keywords_consolidation": ", ".join(c_hits),
            "keywords_pneumonia": ", ".join(p_hits),
            "suggested_consolidation": suggested_consolidation,
            "suggested_pneumonia": suggested_pneumonia,
            "USER_consolidation": suggested_consolidation,
            "USER_pneumonia": suggested_pneumonia,
            "USER_notes": "",
        })

    review_df = pd.DataFrame(review_rows)
    auto_df = pd.DataFrame(auto_rows)

    print(f"Review needed:    {len(review_df)} reports")
    print(f"Auto both-zero:   {len(auto_df)} reports")
    print(f"Total:            {len(review_df) + len(auto_df)}")

    # Build xlsx with formatting
    wb = Workbook()
    ws1 = wb.active
    ws1.title = "REVIEW_NEEDED"

    # Sheet 1: REVIEW_NEEDED
    NAVY = "1F3864"; AMBER = "FFEB9C"; GREY_BG = "F2F2F2"
    cols = list(review_df.columns)
    for ci, c in enumerate(cols, 1):
        cell = ws1.cell(row=1, column=ci, value=c)
        cell.font = Font(name="Arial", size=10, bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", start_color=NAVY)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws1.row_dimensions[1].height = 36

    for ri, row in enumerate(review_df.itertuples(index=False), 2):
        for ci, val in enumerate(row, 1):
            cell = ws1.cell(row=ri, column=ci, value=val)
            cell.font = Font(name="Arial", size=10)
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            colname = cols[ci - 1]
            # Highlight USER_* columns in amber
            if colname.startswith("USER_"):
                cell.fill = PatternFill("solid", start_color=AMBER)
        if ri % 2 == 0:
            for ci in range(1, len(cols) + 1):
                cc = ws1.cell(row=ri, column=ci)
                if not cc.fill.start_color.rgb or cc.fill.start_color.rgb == "00000000":
                    cc.fill = PatternFill("solid", start_color=GREY_BG)
        ws1.row_dimensions[ri].height = 80

    widths = {
        "accession_number": 16, "split": 8, "report": 60,
        "v1_bacterial_pneumonia": 12,
        "keywords_consolidation": 22, "keywords_pneumonia": 22,
        "suggested_consolidation": 12, "suggested_pneumonia": 12,
        "USER_consolidation": 12, "USER_pneumonia": 12, "USER_notes": 28,
    }
    for ci, c in enumerate(cols, 1):
        ws1.column_dimensions[get_column_letter(ci)].width = widths.get(c, 14)
    ws1.freeze_panes = "D2"

    # Sheet 2: AUTO_BOTH_ZERO
    ws2 = wb.create_sheet("AUTO_BOTH_ZERO")
    cols2 = list(auto_df.columns)
    for ci, c in enumerate(cols2, 1):
        cell = ws2.cell(row=1, column=ci, value=c)
        cell.font = Font(name="Arial", size=10, bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", start_color=NAVY)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for ri, row in enumerate(auto_df.itertuples(index=False), 2):
        for ci, val in enumerate(row, 1):
            cell = ws2.cell(row=ri, column=ci, value=val)
            cell.font = Font(name="Arial", size=10)
            cell.alignment = Alignment(vertical="top", wrap_text=True)
        ws2.row_dimensions[ri].height = 70
    widths2 = {"accession_number": 16, "split": 8, "report": 80}
    for ci, c in enumerate(cols2, 1):
        ws2.column_dimensions[get_column_letter(ci)].width = widths2.get(c, 14)
    ws2.freeze_panes = "A2"

    wb.save(args.output)
    print(f"\nWrote: {args.output}")
    print("\nNext step: open the file, review REVIEW_NEEDED rows, edit the")
    print("USER_consolidation and USER_pneumonia columns where the suggestion")
    print("is wrong, then run merge_v3_groundtruth.py to consolidate.")


if __name__ == "__main__":
    main()
