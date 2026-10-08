"""
build_phase_d_review.py
========================

Identifica cases onde o consenso forte dos LLMs discorda do ground truth (GT)
e gera uma planilha no formato do llm-extraction-validator pra revisão por
um segundo radiologista.

Critério de "alta suspeita":
    - Discordância UNÂNIME ou QUASE-UNÂNIME (N-1 de N modelos) com o GT
    - Onde N = todos os modelos que responderam pra aquele case
    - Por exemplo, se 6 modelos responderam e 5+ discordam → flag
    - Se só 5 responderam (gemini-pro ainda rodando) e todos discordam → flag

Formato de saída:
    Coluna `extracted` = GT atual em formato pipe (compatível com validator)
    Coluna `Observações` = quais classes têm discordância e a direção
        Ex: "5/6 LLMs: pneumonia GT=1 mas consenso=0 (provável GT FP)"
    O revisor marca FP/FN normalmente como no validator → se discordar do GT,
    a marcação corrige o GT.

Schema usado: v4 7 classes, mas com `bacterial_pneumonia` como nome de classe
no campo `extracted` pra manter compatibilidade com o validator atual (que
ainda usa esse naming). Uma nota no Observações esclarece o equivalente.
"""

import argparse
import csv
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

V4_CLASSES = ["opacity", "pneumonia", "cardiomegaly", "effusion",
              "pneumothorax", "mass", "yesfinding"]

# Mapeamento pro formato do validator (que usa bacterial_pneumonia em vez de pneumonia)
VALIDATOR_NAME = {"pneumonia": "bacterial_pneumonia"}
def to_validator(c): return VALIDATOR_NAME.get(c, c)

# Prioridade clínica das classes (mais grave / mais difícil primeiro)
PRIORITY = {"pneumonia": 1, "mass": 2, "pneumothorax": 3,
            "opacity": 4, "effusion": 5, "cardiomegaly": 6, "yesfinding": 7}

MODELS = ['sonnet-4-6', 'haiku-4-5', 'gpt-4-1', 'gpt-4-1-mini',
          'gemini-2-5-flash', 'gemini-2-5-pro']


def load_gt(path):
    out = {}
    reports = {}
    abn = {}
    with open(path, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['split'] != 'test':
                continue
            acc = r['accession_number']
            out[acc] = {c: int(r[c]) for c in V4_CLASSES}
            reports[acc] = r['report']
            abn[acc] = r.get('abnormalities_found', '') or ''
    return out, reports, abn


def load_preds(model):
    out = {}
    p = str(Path(__file__).resolve().parent.parent / 'responses' / f'{model}_test.csv')
    if not Path(p).exists():
        return out
    with open(p, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            acc = r['accession_number']
            d = {}
            for c in V4_CLASSES:
                v = r.get(c)
                try:
                    d[c] = int(v) if v not in (None, '') else None
                except (ValueError, TypeError):
                    d[c] = None
            out[acc] = d
    return out


def gt_to_pipe(gt_row, abn_text):
    """Converte GT em string formato pipe igual ao do test_780_validated."""
    parts = []
    for c in V4_CLASSES:
        name = to_validator(c)
        v = gt_row[c]
        if c == "yesfinding" and v == 1 and abn_text:
            parts.append(f"{name}:{v} ({abn_text})")
        else:
            parts.append(f"{name}:{v}")
    return " | ".join(parts)


def detect_suspect_cases(gt, all_preds, strict_threshold=1):
    """
    Para cada case e cada classe, conta votos dos modelos disponíveis.
    Retorna lista de dicts com info da discordância.

    strict_threshold: número máximo de modelos que podem CONCORDAR com o GT
                       pra o case ser flag (ex: 1 = pelo menos N-1 discordam).
    """
    suspects = []
    for acc, gt_row in gt.items():
        case_issues = []
        for c in V4_CLASSES:
            gt_val = gt_row[c]
            votes = []
            for m in MODELS:
                v = all_preds[m].get(acc, {}).get(c)
                if v is not None:
                    votes.append((m, v))
            if not votes:
                continue
            n = len(votes)
            n_agree_with_gt = sum(1 for _, v in votes if v == gt_val)
            n_disagree = n - n_agree_with_gt
            # Flag se MUITOS discordam (≥ N - strict_threshold)
            if n_disagree >= max(n - strict_threshold, n):
                # i.e., todos exceto strict_threshold concordam → discordam
                if n_disagree >= n - strict_threshold:
                    direction = "GT=1 mas consenso=0 (provável FP do GT)" if gt_val == 1 \
                                else "GT=0 mas consenso=1 (provável FN do GT)"
                    case_issues.append({
                        'classe': c,
                        'gt_val': gt_val,
                        'n_models': n,
                        'n_disagree': n_disagree,
                        'direction': direction,
                        'votes': votes,
                    })
        if case_issues:
            # ordenar por prioridade clinica
            case_issues.sort(key=lambda x: PRIORITY[x['classe']])
            suspects.append({'accession': acc, 'issues': case_issues})
    return suspects


def make_obs(issues):
    """Constrói o texto da coluna Observações pro revisor."""
    lines = []
    for it in issues:
        cls_name = to_validator(it['classe'])
        v = '/'.join(f"{m.replace('-', '')}={vv}" for m, vv in it['votes'])
        lines.append(
            f"[{cls_name}] {it['n_disagree']}/{it['n_models']} LLMs discordam: "
            f"{it['direction']}  •  votos: {v}"
        )
    return " \n".join(lines)


def write_xlsx(path, suspects, gt, reports, abn, sheet_name="Validacao", title_note=""):
    wb = Workbook()
    ws = wb.active
    ws.title = sheet_name
    NAVY = "1F3864"
    AMBER = "FFEB9C"
    LIGHT = "F2F2F2"

    hdr = ["accession_number", "report", "extracted",
           "TRUE", "FP", "FN", "Observações"]
    for ci, c in enumerate(hdr, 1):
        cell = ws.cell(row=1, column=ci, value=c)
        cell.font = Font(name="Arial", size=10, bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", start_color=NAVY)
        cell.alignment = Alignment(horizontal="center", vertical="center",
                                    wrap_text=True)
    ws.row_dimensions[1].height = 32

    for ri, s in enumerate(suspects, 2):
        acc = s['accession']
        extracted = gt_to_pipe(gt[acc], abn[acc])
        obs = make_obs(s['issues'])
        if title_note:
            obs = f"{title_note}\n{obs}"

        row_values = [acc, reports[acc], extracted, "", "", "", obs]
        for ci, v in enumerate(row_values, 1):
            cell = ws.cell(row=ri, column=ci, value=v)
            cell.font = Font(name="Arial", size=10)
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            if ci == 7:  # Observações em amber pra destacar
                cell.fill = PatternFill("solid", start_color=AMBER)
        if ri % 2 == 0:
            for ci in range(1, len(hdr) + 1):
                cc = ws.cell(row=ri, column=ci)
                if not cc.fill.start_color.rgb or cc.fill.start_color.rgb == "00000000":
                    cc.fill = PatternFill("solid", start_color=LIGHT)
        ws.row_dimensions[ri].height = 120

    widths = {"accession_number": 16, "report": 70, "extracted": 50,
              "TRUE": 12, "FP": 18, "FN": 18, "Observações": 65}
    for ci, c in enumerate(hdr, 1):
        ws.column_dimensions[get_column_letter(ci)].width = widths.get(c, 16)
    ws.freeze_panes = "D2"
    wb.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt-csv", default="data/outputs/ground_truth_7class.csv")
    ap.add_argument("--output-high", default="data/outputs/phase_d_review_high_suspicion.xlsx",
                    help="cases de alta suspeita (não-yesfinding + yesfinding raros)")
    ap.add_argument("--output-yes", default="data/outputs/phase_d_review_yesfinding_sample.xlsx",
                    help="sample de cases yesfinding (muito numerosos → amostra)")
    ap.add_argument("--yesfinding-sample-size", type=int, default=25)
    args = ap.parse_args()

    gt, reports, abn = load_gt(args.gt_csv)
    print(f"GT test rows: {len(gt)}")

    all_preds = {}
    for m in MODELS:
        all_preds[m] = load_preds(m)
        n = len(all_preds[m])
        print(f"  {m}: {n} respostas")

    # Detecta suspeitos (≥ N-1 modelos discordando)
    suspects = detect_suspect_cases(gt, all_preds, strict_threshold=1)

    # Separa yesfinding-only dos demais
    yes_only_suspects = []
    high_suspects = []
    for s in suspects:
        non_yes = [i for i in s['issues'] if i['classe'] != 'yesfinding']
        if non_yes:
            # tem pelo menos 1 issue não-yesfinding → alta suspeita
            high_suspects.append(s)
        else:
            yes_only_suspects.append(s)

    # Sort alta-suspeita por prioridade clínica (classes mais sérias primeiro,
    # discordância mais forte primeiro)
    def sort_key(s):
        # menor PRIORITY = mais sério
        min_pri = min(PRIORITY[i['classe']] for i in s['issues'])
        max_disagree_frac = max(i['n_disagree'] / i['n_models']
                                  for i in s['issues'])
        return (min_pri, -max_disagree_frac)
    high_suspects.sort(key=sort_key)

    # Sample do yesfinding
    yes_sample = yes_only_suspects[:args.yesfinding_sample_size]

    Path(args.output_high).parent.mkdir(parents=True, exist_ok=True)
    write_xlsx(args.output_high, high_suspects, gt, reports, abn,
               sheet_name="Validacao",
               title_note="REVISÃO PHASE D — Cases de alta suspeita (não-yesfinding)")
    write_xlsx(args.output_yes, yes_sample, gt, reports, abn,
               sheet_name="Validacao",
               title_note="REVISÃO PHASE D — Sample de cases yesfinding (entender critério)")

    print()
    print(f"Cases alta suspeita (não-yesfinding):  {len(high_suspects)}")
    print(f"Cases yesfinding-only:                  {len(yes_only_suspects)} (sample: {len(yes_sample)})")
    print(f"\nWrote: {args.output_high}")
    print(f"Wrote: {args.output_yes}")

    # Breakdown por classe principal
    from collections import Counter
    cls_counter = Counter()
    for s in high_suspects:
        for i in s['issues']:
            if i['classe'] != 'yesfinding':
                cls_counter[i['classe']] += 1
    print(f"\nBreakdown alta-suspeita por classe:")
    for c, n in sorted(cls_counter.items(), key=lambda x: PRIORITY[x[0]]):
        print(f"  {c:<16}  {n} discordâncias")


if __name__ == "__main__":
    main()
