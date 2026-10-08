"""
build_phase_d_consolidated.py
==============================

Gera uma planilha ÚNICA com TODOS os cases que precisam re-análise humana —
combinando discordâncias em qualquer classe (yesfinding ou não).

Critério: pelo menos N-1 de N modelos discordam do GT em pelo menos uma classe
(N = quantos modelos responderam pra aquele case; pode ser 5 ou 6).

Ordenação:
    1. Cases com MÚLTIPLAS classes discordantes primeiro (mais urgentes)
    2. Depois por menor PRIORITY (pneumonia/mass/pneumothorax > opacity/effusion > yesfinding)
    3. Depois por força do consenso (5/5 unânime > 4/5 quase-unânime)

Formato: 100% compatível com llm-extraction-validator.
    Coluna `extracted` = GT atual em pipe format (bacterial_pneumonia naming)
    Coluna `Observações` = todas as discordâncias detectadas + categoria

Cada row tem também colunas extras (após Observações) que o validator NÃO usa
mas o reviewer pode consultar:
    _gt_yesfinding_abn  — abnormalities_found do GT
    _consensus_summary  — voto-a-voto de cada modelo em cada classe
"""

import argparse
import csv
from pathlib import Path
from collections import Counter

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

V4_CLASSES = ["opacity", "pneumonia", "cardiomegaly", "effusion",
              "pneumothorax", "mass", "yesfinding"]
VALIDATOR_NAME = {"pneumonia": "bacterial_pneumonia"}
PRIORITY = {"pneumonia": 1, "mass": 2, "pneumothorax": 3,
            "opacity": 4, "effusion": 5, "cardiomegaly": 6, "yesfinding": 7}

MODELS = ['sonnet-4-6', 'haiku-4-5', 'gpt-4-1', 'gpt-4-1-mini',
          'gemini-2-5-flash', 'gemini-2-5-pro']

# Keywords pra categorizar yesfinding discordâncias
SUPPORT_KW = ['catéter', 'cateter', 'dreno', 'tubo', 'pigtail', 'marcapasso',
              'prótese', 'clipe', 'fio', 'sutura', 'porto', 'portocath',
              'órtese', 'traqueostomia']
EXPECTED_KW = ['nódulo', 'massa', 'consolidação', 'derrame', 'pneumotórax',
               'opacidade', 'infiltrado', 'cardiomegalia', 'pneumonia']

def to_validator(c):
    return VALIDATOR_NAME.get(c, c)


def load_data(gt_csv):
    gt = {}; abn = {}; reports = {}
    with open(gt_csv, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['split'] != 'test':
                continue
            acc = r['accession_number']
            gt[acc] = {c: int(r[c]) for c in V4_CLASSES}
            abn[acc] = r.get('abnormalities_found', '') or ''
            reports[acc] = r['report']
    return gt, abn, reports


def load_preds(model):
    p = Path(str(Path(__file__).resolve().parent.parent / 'responses' / f'{model}_test.csv'))
    out = {}
    if not p.exists():
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


CRITICAL_CLASSES = {"pneumonia", "mass", "pneumothorax"}


def detect_discordances(gt, all_preds, threshold=1, include_critical_splits=True):
    """Retorna {acc: [issues...]} para cases com discordância significativa.

    Critério primário: ≥ N-threshold modelos discordam do GT.
    Critério adicional (opcional, só pra classes críticas): split ~50/50.
    Split em opacity/effusion/cardiomegaly/yesfinding NÃO é incluído pra
    evitar inflar a lista (muito ruído de borderline genuíno em opacity).
    """
    out = {}
    for acc, gt_row in gt.items():
        issues = []
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
            n_disagree = sum(1 for _, v in votes if v != gt_val)

            include = False
            strength = None
            kind = None

            if n_disagree >= n - threshold:
                strength = 'unanime' if n_disagree == n else f'quase-unanime ({n_disagree}/{n})'
                kind = 'consenso'
                include = True
            elif include_critical_splits and c in CRITICAL_CLASSES:
                # Split em classe crítica: 2-4 dos 5+ modelos disagree
                n_agree = n - n_disagree
                if 2 <= n_disagree <= n - 2 and abs(n_disagree - n_agree) <= 1:
                    strength = f'split ({n_disagree}vs{n_agree})'
                    kind = 'split_critico'
                    include = True

            if include:
                issues.append({
                    'classe': c, 'gt_val': gt_val,
                    'n_models': n, 'n_disagree': n_disagree,
                    'strength': strength, 'kind': kind,
                    'votes': votes,
                })
        if issues:
            out[acc] = issues
    return out


def detect_internal_inconsistencies(gt_row, abn_text):
    """Detecta violações de regras internas do prompt v4 no GT."""
    flags = []
    # Rule 8: yesfinding=1 deve ter pelo menos uma binária=1 OU abn não-vazio
    if gt_row['yesfinding'] == 1:
        any_bin = any(gt_row[c] == 1 for c in
                       ['opacity','pneumonia','cardiomegaly','effusion','pneumothorax','mass'])
        has_abn = bool((abn_text or '').strip())
        if not any_bin and not has_abn:
            flags.append("violação Rule 8: yesfinding=1 sem nenhuma evidência (bin todas 0 e abn vazio)")
    # Hierarquia v4: pneumonia=1 entails opacity=1
    if gt_row['pneumonia'] == 1 and gt_row['opacity'] == 0:
        flags.append("violação hierarquia: pneumonia=1 mas opacity=0 (deveria ser 1)")
    return flags


def categorize_yesfinding(abn_text):
    a = (abn_text or '').lower().strip()
    if not a:
        return 'empty_abn'
    has_support = any(k in a for k in SUPPORT_KW)
    has_expected = any(k in a for k in EXPECTED_KW)
    if has_support and not has_expected:
        return 'support_only'
    if not has_expected:
        return 'subtle_or_other'
    return 'has_expected'


def gt_to_pipe(gt_row, abn_text):
    parts = []
    for c in V4_CLASSES:
        n = to_validator(c)
        v = gt_row[c]
        if c == "yesfinding" and v == 1 and abn_text:
            parts.append(f"{n}:{v} ({abn_text})")
        else:
            parts.append(f"{n}:{v}")
    return " | ".join(parts)


def make_observations(issues, abn_text, gt_row):
    """Texto consolidado de Observações para o revisor."""
    lines = ["REVISÃO PHASE D — re-validação do GT por consenso LLM"]

    # Internal consistency flags primeiro (mais sério)
    inconsistencies = detect_internal_inconsistencies(gt_row, abn_text)
    for f in inconsistencies:
        lines.append(f"🚨 INCONSISTÊNCIA INTERNA: {f}")

    # Lista geral de discordâncias
    for it in issues:
        cls = to_validator(it['classe'])
        votes_str = '/'.join(f"{m.replace('-','')}={v}" for m, v in it['votes'])
        if it.get('kind') == 'split_critico':
            tag = "🟠 SPLIT em classe crítica"
        else:
            direction = ("→ provável FP do GT" if it['gt_val'] == 1
                         else "→ provável FN do GT")
            tag = direction
        lines.append(
            f"• [{cls}] GT={it['gt_val']}, {it['n_disagree']}/{it['n_models']} LLMs discordam ({it['strength']}) {tag}"
        )
        lines.append(f"  votos: {votes_str}")

    # Anota categoria se for problema yesfinding
    for it in issues:
        if it['classe'] == 'yesfinding' and it['gt_val'] == 1:
            cat = categorize_yesfinding(abn_text)
            cat_label = {
                'empty_abn': 'abnormalities_found vazio — sem justificativa documentada',
                'support_only': 'GT marcou yesfinding por SUPPORT DEVICE (viola Rule 9 do prompt v4.1)',
                'subtle_or_other': 'GT marcou por achado sutil ou extra-pulmonar discutível',
                'has_expected': 'há achado esperado em outras classes',
            }.get(cat, cat)
            lines.append(f"⚠️ yesfinding: {cat_label}")
            if abn_text:
                lines.append(f"   abn no GT: \"{abn_text}\"")

    return "\n".join(lines)


def sort_key(item):
    """Ordena: + classes discordantes primeiro, + grave primeiro, + consenso primeiro."""
    issues = item[1]
    n_classes = len(issues)
    min_pri = min(PRIORITY[i['classe']] for i in issues)
    max_disagree_frac = max(i['n_disagree'] / i['n_models'] for i in issues)
    # Tuple ordenação: queremos descrescente em (n_classes, max_disagree_frac), crescente em min_pri
    return (-n_classes, min_pri, -max_disagree_frac)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt-csv", default="data/outputs/ground_truth_7class.csv")
    ap.add_argument("--output",
                    default="data/outputs/phase_d_review_CONSOLIDATED.xlsx")
    args = ap.parse_args()

    gt, abn, reports = load_data(args.gt_csv)
    all_preds = {m: load_preds(m) for m in MODELS}
    for m in MODELS:
        print(f"  {m}: {len(all_preds[m])} respostas")

    suspects = detect_discordances(gt, all_preds, threshold=1)
    print(f"\nCases com discordância (≥N-1 de N modelos): {len(suspects)}")

    # Sort
    sorted_cases = sorted(suspects.items(), key=sort_key)

    # Stats
    multi_class = sum(1 for _, issues in sorted_cases if len(issues) > 1)
    print(f"  com discordância em múltiplas classes: {multi_class}")
    cls_counter = Counter()
    for _, issues in sorted_cases:
        for it in issues:
            cls_counter[it['classe']] += 1
    print(f"\nDiscordâncias por classe:")
    for c, n in sorted(cls_counter.items(), key=lambda x: PRIORITY[x[0]]):
        print(f"  {c:<16}  {n}")

    # Build xlsx
    wb = Workbook()
    ws = wb.active
    ws.title = "Validacao"
    NAVY = "1F3864"; AMBER = "FFEB9C"; LIGHT = "F2F2F2"
    RED = "FFC7CE"; ORANGE = "FCD5B4"; GREEN = "C6EFCE"; BLUE = "D9E1F2"

    hdr = ["accession_number", "report", "extracted", "TRUE", "FP", "FN",
           "Observações", "_meta_n_classes", "_meta_top_class", "_meta_strength",
           "_meta_consensus_json"]
    for ci, c in enumerate(hdr, 1):
        cell = ws.cell(row=1, column=ci, value=c)
        cell.font = Font(name="Arial", size=10, bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", start_color=NAVY)
        cell.alignment = Alignment(horizontal="center", vertical="center",
                                    wrap_text=True)
    ws.row_dimensions[1].height = 32

    for ri, (acc, issues) in enumerate(sorted_cases, 2):
        extracted = gt_to_pipe(gt[acc], abn[acc])
        obs = make_observations(issues, abn[acc], gt[acc])

        # Meta tags (não usadas pelo validator, mas úteis pra filtragem)
        n_classes = len(issues)
        # Top class = a com menor PRIORITY (mais grave)
        top = min(issues, key=lambda i: PRIORITY[i['classe']])
        top_class = to_validator(top['classe'])
        strength = top['strength']

        # JSON estruturado com o consenso (consumido pela UI /audit)
        import json
        consensus_struct = {
            "issues": [
                {
                    "classe": to_validator(i['classe']),
                    "gt_val": i['gt_val'],
                    "n_disagree": i['n_disagree'],
                    "n_models": i['n_models'],
                    "strength": i['strength'],
                    "kind": i.get('kind', 'consenso'),
                    "direction": ("fp_do_gt" if i['gt_val'] == 1 else "fn_do_gt"),
                    "votes": [{"model": m, "vote": v} for m, v in i['votes']],
                    "yesfinding_category": (categorize_yesfinding(abn[acc])
                                              if i['classe'] == 'yesfinding' else None),
                }
                for i in issues
            ],
            "abn_in_gt": abn[acc],
            "internal_inconsistencies": detect_internal_inconsistencies(gt[acc], abn[acc]),
        }
        consensus_json = json.dumps(consensus_struct, ensure_ascii=False)

        row_values = [acc, reports[acc], extracted, "", "", "", obs,
                      n_classes, top_class, strength, consensus_json]
        for ci, v in enumerate(row_values, 1):
            cell = ws.cell(row=ri, column=ci, value=v)
            cell.font = Font(name="Arial", size=10)
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            if ci == 7:  # Observações
                # Cor por severidade
                if n_classes >= 2:
                    color = RED
                elif PRIORITY[top['classe']] <= 3:  # pneumonia/mass/pneumothorax
                    color = ORANGE
                elif top['classe'] == 'yesfinding':
                    color = GREEN
                else:
                    color = AMBER
                cell.fill = PatternFill("solid", start_color=color)
        if ri % 2 == 0:
            for ci in range(1, len(hdr) + 1):
                cc = ws.cell(row=ri, column=ci)
                if not cc.fill.start_color.rgb or cc.fill.start_color.rgb == "00000000":
                    cc.fill = PatternFill("solid", start_color=LIGHT)
        ws.row_dimensions[ri].height = 130

    widths = {"accession_number": 16, "report": 70, "extracted": 50,
              "TRUE": 12, "FP": 18, "FN": 18, "Observações": 75,
              "_meta_n_classes": 10, "_meta_top_class": 18,
              "_meta_strength": 18, "_meta_consensus_json": 30}
    for ci, c in enumerate(hdr, 1):
        ws.column_dimensions[get_column_letter(ci)].width = widths.get(c, 16)
    ws.freeze_panes = "D2"

    # Sheet 2 — guia de cores
    ws2 = wb.create_sheet("guia_de_cores")
    legend = [
        ["Cor", "Significado", "Prioridade"],
        ["🔴 Vermelho", "≥ 2 classes discordantes no mesmo case", "URGENTE"],
        ["🟠 Laranja", "1 discordância em pneumonia / mass / pneumothorax", "Alta"],
        ["🟡 Amber", "1 discordância em opacity / effusion / cardiomegaly", "Média"],
        ["🟢 Verde", "Apenas yesfinding discordante (provável FP do GT)", "Sistematica"],
        ["", "", ""],
        ["Categoria yesfinding", "O que significa", ""],
        ["empty_abn", "GT yesfinding=1 mas sem justificativa documentada (mais provável GT errado)", ""],
        ["support_only", "GT marcou yesfinding por dreno/cateter/etc (viola Rule 9 do prompt v4.1)", ""],
        ["subtle_or_other", "GT marcou por achado sutil ou extra-pulmonar discutível", ""],
        ["has_expected", "GT marcou + tem achado esperado em outras classes (revisar)", ""],
    ]
    for ri, row in enumerate(legend, 1):
        for ci, v in enumerate(row, 1):
            cell = ws2.cell(row=ri, column=ci, value=v)
            if ri == 1 or (ri == 7):
                cell.font = Font(bold=True)
                cell.fill = PatternFill("solid", start_color=NAVY)
                cell.font = Font(bold=True, color="FFFFFF")
    ws2.column_dimensions['A'].width = 24
    ws2.column_dimensions['B'].width = 80
    ws2.column_dimensions['C'].width = 12

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    wb.save(args.output)
    print(f"\nWrote: {args.output}")
    print(f"Total cases: {len(sorted_cases)}")


if __name__ == "__main__":
    main()
