"""
Build GT audit spreadsheet (compatível com llm-extraction-validator,
modo validação).

Output: data/outputs/gt_audit_review.xlsx (sheet 'Validacao')

Inclui todos os casos em ground_truth_7class.csv com QUALQUER anotação
(yesfinding=1 OU qualquer das 6 classes específicas=1), EXCLUINDO casos
já auditados em:
  - revisão_exportada.xlsx (sheet Validacao)
  - data/outputs/pneumonia_errors.xlsx (todas as sheets)

Formato (colunas do validator, modo validação):
  accession_number, report, extracted, TRUE, FP, FN, Observações

A coluna `extracted` carrega o GT atual no formato pipe:
  'opacity:1 | bacterial_pneumonia:0 | cardiomegaly:1 | effusion:0 |
   pneumothorax:0 | mass:0 | yesfinding:1 (abnormalities_found...)'

NB: o validator ainda usa o label legado `bacterial_pneumonia`, então
mapeamos `pneumonia` (7-class atual) → `bacterial_pneumonia` no string
para o app reconhecer. As predições por modelo são omitidas para
reduzir anchoring bias.
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GT_PATH = ROOT / "data" / "outputs" / "ground_truth_7class.csv"
REVISAO_PATH = ROOT / "revisão_exportada.xlsx"
PNEUMO_PATH = ROOT / "data" / "outputs" / "pneumonia_errors.xlsx"
OUT_PATH = ROOT / "data" / "outputs" / "gt_audit_review.xlsx"

CLASSES_GT = ["opacity", "pneumonia", "cardiomegaly", "effusion",
              "pneumothorax", "mass", "yesfinding"]

# Ordem e nomes esperados pelo validator (lib/types.ts FINDINGS)
EXTRACTED_ORDER = [
    ("opacity", "opacity"),
    ("pneumonia", "bacterial_pneumonia"),
    ("cardiomegaly", "cardiomegaly"),
    ("effusion", "effusion"),
    ("pneumothorax", "pneumothorax"),
    ("mass", "mass"),
    ("yesfinding", "yesfinding"),
]


def already_audited() -> set[str]:
    audited: set[str] = set()
    r = pd.read_excel(REVISAO_PATH, sheet_name="Validacao")
    audited |= set(r["accession_number"].astype(str))
    xl = pd.ExcelFile(PNEUMO_PATH)
    for s in xl.sheet_names:
        d = pd.read_excel(PNEUMO_PATH, sheet_name=s)
        if "accession_number" in d.columns:
            audited |= set(d["accession_number"].astype(str))
    return audited


def build_extracted(row) -> str:
    parts = [f"{validator_name}:{int(row[gt_col])}"
             for gt_col, validator_name in EXTRACTED_ORDER]
    base = " | ".join(parts)
    findings = row.get("abnormalities_found")
    if pd.notna(findings) and str(findings).strip():
        return f"{base} ({str(findings).strip()})"
    return base


def main():
    gt = pd.read_csv(GT_PATH)
    audited = already_audited()
    print(f"Já auditados: {len(audited)}")

    mask_any = (gt[CLASSES_GT].sum(axis=1) > 0)
    not_done = ~gt["accession_number"].astype(str).isin(audited)
    sub = gt[mask_any & not_done].copy()

    print(f"Casos com anotação: {mask_any.sum()}")
    print(f"A auditar: {len(sub)} (dev={(sub.split=='dev').sum()}, "
          f"test={(sub.split=='test').sum()})")

    # Formato esperado pelo validator. A coluna `_meta_reaudit=1` sinaliza
    # ao app que esta é uma re-auditoria do GT: usa a UI de /validate
    # (TRUE/FP/FN tradicional) mas em modo reaudit (sem hidratar GT antigo,
    # persiste na tabela `reaudit_validations` separada).
    out = pd.DataFrame()
    out["accession_number"] = sub["accession_number"].values
    out["report"] = sub["report"].values
    out["extracted"] = sub.apply(build_extracted, axis=1).values
    out["TRUE"] = ""
    out["FP"] = ""
    out["FN"] = ""
    out["Observações"] = ""
    out["_meta_reaudit"] = 1
    # Mantém split como coluna auxiliar (validator ignora colunas extras)
    out["split"] = sub["split"].values

    # Ordem: dev primeiro (cohort menor → fácil fechar um split), depois test
    out["_split_order"] = out["split"].map({"dev": 0, "test": 1})
    out = (out.sort_values(["_split_order", "accession_number"])
              .drop(columns="_split_order")
              .reset_index(drop=True))

    with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as w:
        out.to_excel(w, sheet_name="Validacao", index=False)

    print(f"\n→ {OUT_PATH.relative_to(ROOT)}")
    print(f"  {len(out)} linhas | colunas: {list(out.columns)}")
    print(f"  primeiro extracted: {out['extracted'].iloc[0][:120]}...")


if __name__ == "__main__":
    main()
