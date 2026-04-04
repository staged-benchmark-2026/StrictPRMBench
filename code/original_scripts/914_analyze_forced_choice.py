#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import openpyxl
from scipy.stats import beta, binomtest

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / 'outputs/reports/forced_choice'
COMPLETED_DIR = OUT_DIR / 'completed'
DEBLIND_DIR = OUT_DIR / 'deblind_keys'
RESULTS_CSV = OUT_DIR / 'forced_choice_results.csv'
PER_ANNOTATOR_CSV = OUT_DIR / 'forced_choice_per_annotator.csv'
SUMMARY_TXT = OUT_DIR / 'forced_choice_summary.txt'

CELLS = [
    ('skywork_prm', 'gsm8k', 'Skywork-7B'),
    ('skywork_prm', 'math500', 'Skywork-7B'),
    ('internlm2_7b_reward', 'gsm8k', 'InternLM2-7B'),
    ('internlm2_7b_reward', 'math500', 'InternLM2-7B'),
]


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lo, hi


def fleiss_kappa(card_votes: list[tuple[int, int]]) -> float:
    if not card_votes:
        return float('nan')
    n_ann = sum(card_votes[0])
    if n_ann <= 1:
        return float('nan')
    n_cards = len(card_votes)
    p_gm = sum(gm for gm, _ in card_votes) / (n_cards * n_ann)
    p_prod = sum(prod for _, prod in card_votes) / (n_cards * n_ann)
    p_e = p_gm * p_gm + p_prod * p_prod
    p_bar = 0.0
    for gm, prod in card_votes:
        p_bar += (gm * (gm - 1) + prod * (prod - 1)) / (n_ann * (n_ann - 1))
    p_bar /= n_cards
    if p_e >= 1.0:
        return float('nan')
    return float((p_bar - p_e) / (1.0 - p_e))


def load_deblind(prm: str, benchmark: str) -> dict[str, dict[str, Any]]:
    out = {}
    with (DEBLIND_DIR / f'{prm}_{benchmark}_deblind.jsonl').open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                out[str(rec['card_id'])] = rec
    return out


def load_sheet(path: Path) -> list[dict[str, str]]:
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        return []
    headers = [str(x).strip() if x is not None else '' for x in rows[0]]
    out = []
    for row in rows[1:]:
        rec = {headers[i]: ('' if i >= len(row) or row[i] is None else str(row[i]).strip()) for i in range(len(headers))}
        if any(v for v in rec.values()):
            out.append(rec)
    return out


def choice_to_source(choice: str, deblind: dict[str, Any]) -> str | None:
    choice = choice.strip().upper()
    if choice == 'A':
        return str(deblind['trace_A_source'])
    if choice == 'B':
        return str(deblind['trace_B_source'])
    return None


def fmt_p(p: float) -> str:
    return '<0.001' if p < 0.001 else f'{p:.3f}'


def main() -> int:
    if not COMPLETED_DIR.exists():
        raise SystemExit(f'missing completed dir: {COMPLETED_DIR}')

    rows_out: list[dict[str, Any]] = []
    per_annotator_rows: list[dict[str, Any]] = []
    pooled_card_votes: list[tuple[int, int]] = []
    summary_lines = ['Stress-condition forced-choice confirmation', '']

    for prm, benchmark, label in CELLS:
        deblind = load_deblind(prm, benchmark)
        sheet_paths = sorted(COMPLETED_DIR.glob(f'{prm}_{benchmark}_annotator*.xlsx'))
        if not sheet_paths:
            summary_lines.append(f'{label} / {benchmark}: no completed sheets found')
            continue

        card_votes: dict[str, list[str]] = defaultdict(list)
        for path in sheet_paths:
            ann_name = path.stem
            ann_choices: list[str] = []
            for row in load_sheet(path):
                card_id = row.get('Card ID', '')
                if not card_id or card_id not in deblind:
                    continue
                chosen = choice_to_source(row.get('Annotator Choice (A/B)', ''), deblind[card_id])
                if chosen not in {'gm', 'product'}:
                    continue
                card_votes[card_id].append(chosen)
                ann_choices.append(chosen)
            if ann_choices:
                gm_count = sum(1 for x in ann_choices if x == 'gm')
                per_annotator_rows.append({
                    'prm': prm,
                    'benchmark': benchmark,
                    'annotator': ann_name,
                    'n_cards': len(ann_choices),
                    'gm_preferred': gm_count,
                    'gm_rate': gm_count / len(ann_choices),
                })

        majority_gm = 0
        analyzed_cards = 0
        cell_votes: list[tuple[int, int]] = []
        for card_id in sorted(card_votes):
            votes = card_votes[card_id]
            if len(votes) < 3:
                continue
            gm = sum(1 for x in votes if x == 'gm')
            prod = sum(1 for x in votes if x == 'product')
            analyzed_cards += 1
            if gm >= 2:
                majority_gm += 1
            cell_votes.append((gm, prod))
            pooled_card_votes.append((gm, prod))

        if analyzed_cards == 0:
            summary_lines.append(f'{label} / {benchmark}: no complete 3-vote cards')
            continue

        gm_rate = majority_gm / analyzed_cards
        ci_lo, ci_hi = clopper_pearson(majority_gm, analyzed_cards)
        p_value = float(binomtest(majority_gm, analyzed_cards, 0.5).pvalue)
        kappa = fleiss_kappa(cell_votes)
        rows_out.append({
            'prm': prm,
            'benchmark': benchmark,
            'n': analyzed_cards,
            'gm_preferred': majority_gm,
            'gm_rate': gm_rate,
            'ci_lo': ci_lo,
            'ci_hi': ci_hi,
            'p_value': p_value,
            'fleiss_kappa': kappa,
        })
        summary_lines.append(
            f'{label:15s} {benchmark:8s} n={analyzed_cards:3d} GM pref={gm_rate*100:5.1f}% [{ci_lo*100:5.1f}%, {ci_hi*100:5.1f}%] p={fmt_p(p_value)} kappa={kappa:.2f}'
        )

    if rows_out:
        total_n = sum(int(r['n']) for r in rows_out)
        total_gm = sum(int(r['gm_preferred']) for r in rows_out)
        ci_lo, ci_hi = clopper_pearson(total_gm, total_n)
        p_value = float(binomtest(total_gm, total_n, 0.5).pvalue)
        kappa = fleiss_kappa(pooled_card_votes)
        rows_out.append({
            'prm': 'pooled',
            'benchmark': 'all',
            'n': total_n,
            'gm_preferred': total_gm,
            'gm_rate': total_gm / total_n,
            'ci_lo': ci_lo,
            'ci_hi': ci_hi,
            'p_value': p_value,
            'fleiss_kappa': kappa,
        })
        summary_lines.extend([
            '',
            f'Pooled           all      n={total_n:3d} GM pref={(total_gm/total_n)*100:5.1f}% [{ci_lo*100:5.1f}%, {ci_hi*100:5.1f}%] p={fmt_p(p_value)}',
            f'Per-annotator agreement: Fleiss kappa = {kappa:.2f}',
        ])

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_CSV.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['prm', 'benchmark', 'n', 'gm_preferred', 'gm_rate', 'ci_lo', 'ci_hi', 'p_value', 'fleiss_kappa'])
        writer.writeheader()
        writer.writerows(rows_out)
    with PER_ANNOTATOR_CSV.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['prm', 'benchmark', 'annotator', 'n_cards', 'gm_preferred', 'gm_rate'])
        writer.writeheader()
        writer.writerows(per_annotator_rows)
    SUMMARY_TXT.write_text('\n'.join(summary_lines) + '\n', encoding='utf-8')
    print(f'[saved] {RESULTS_CSV}')
    print(f'[saved] {PER_ANNOTATOR_CSV}')
    print(f'[saved] {SUMMARY_TXT}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
