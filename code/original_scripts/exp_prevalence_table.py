#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / 'outputs/reports/prevalence'
NAT_DIR = REPO_ROOT / 'outputs/reports/natural_pool'

DISPLAY = {
    'math_shepherd': 'Math-Shepherd',
    'skywork_prm': 'Skywork-7B',
    'skywork_prm_1_5b': 'Skywork-1.5B',
    'qwen_prm': 'Qwen',
    'rlhflow_prm': 'RLHFlow',
    'internlm2_7b_reward': 'InternLM2-7B',
    'internlm2_1_8b_reward': 'InternLM2-1.8B',
}
DISPLAY_BENCH = {'gsm8k': 'GSM8K', 'math500': 'MATH-500'}


def read_csv(path: Path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def frac(num: float, den: float) -> float:
    return 0.0 if den == 0 else float(num / den)


def fmt_pct(x: float) -> str:
    return f'{100*x:.1f}%'


def main() -> int:
    pg = read_csv(NAT_DIR / 'phase1b_disagreement_pg.csv')
    pn = read_csv(NAT_DIR / 'phase1b_disagreement_pn.csv')

    by_key = {}
    for row in pg:
        key = (row['prm'], row['benchmark'])
        by_key.setdefault(key, {})['pg'] = row
    for row in pn:
        key = (row['prm'], row['benchmark'])
        by_key.setdefault(key, {})['pn'] = row

    rows = []
    for key in sorted(by_key, key=lambda k: (k[1], k[0])):
        prm, benchmark = key
        pg_row = by_key[key]['pg']
        pn_row = by_key[key]['pn']

        n_total = int(pg_row['n_total'])
        n_disagree_pg = int(pg_row['n_disagree'])
        both_right_pg = int(pg_row['both_right'])
        both_wrong_pg = int(pg_row['both_wrong'])
        mixed_pg = n_disagree_pg - both_right_pg - both_wrong_pg

        n_disagree_pn = int(pn_row['n_disagree'])
        both_right_pn = int(pn_row['both_right'])
        both_wrong_pn = int(pn_row['both_wrong'])
        mixed_pn = n_disagree_pn - both_right_pn - both_wrong_pn

        rows.append({
            'prm': prm,
            'benchmark': benchmark,
            'prm_display': DISPLAY[prm],
            'benchmark_display': DISPLAY_BENCH[benchmark],
            'n_total': n_total,
            'disagree_pg_rate': float(pg_row['disagree_rate']),
            'both_correct_pg_rate_total': frac(both_right_pg, n_total),
            'both_correct_pg_rate_given_disagree': frac(both_right_pg, n_disagree_pg),
            'both_wrong_pg_rate_given_disagree': frac(both_wrong_pg, n_disagree_pg),
            'mixed_pg_rate_given_disagree': frac(mixed_pg, n_disagree_pg),
            'product_shorter_chars_pg': float(pg_row['prod_shorter_chars']),
            'product_shorter_steps_pg': float(pg_row['prod_shorter_steps']),
            'shorter_and_wrong_pg_rate_total': frac(int(pg_row['shorter_and_wrong']), n_total),
            'disagree_pn_rate': float(pn_row['disagree_rate']),
            'both_correct_pn_rate_total': frac(both_right_pn, n_total),
            'both_correct_pn_rate_given_disagree': frac(both_right_pn, n_disagree_pn),
            'both_wrong_pn_rate_given_disagree': frac(both_wrong_pn, n_disagree_pn),
            'mixed_pn_rate_given_disagree': frac(mixed_pn, n_disagree_pn),
            'product_shorter_chars_pn': float(pn_row['prod_shorter_chars']),
            'product_shorter_steps_pn': float(pn_row['prod_shorter_steps']),
            'shorter_and_wrong_pn_rate_total': frac(int(pn_row['shorter_and_wrong']), n_total),
        })

    write_csv(OUT_DIR / 'prevalence_table.csv', rows)

    tex_rows = []
    tex_rows.append('\\begin{tabular}{llrrrrrr}')
    tex_rows.append('\\toprule')
    tex_rows.append('PRM & Bench & Disagree(P,GM) & Both-correct & Both-correct|disagree & P shorter(chars) & Shorter+wrong & Disagree(P,N) \\\\')
    tex_rows.append('\\midrule')
    for r in rows:
        tex_rows.append(
            f"{r['prm_display']} & {r['benchmark_display']} & {fmt_pct(r['disagree_pg_rate'])} & {fmt_pct(r['both_correct_pg_rate_total'])} & {fmt_pct(r['both_correct_pg_rate_given_disagree'])} & {fmt_pct(r['product_shorter_chars_pg'])} & {fmt_pct(r['shorter_and_wrong_pg_rate_total'])} & {fmt_pct(r['disagree_pn_rate'])} \\\\"
        )
    tex_rows.append('\\bottomrule')
    tex_rows.append('\\end{tabular}')
    (OUT_DIR / 'prevalence_table.tex').write_text('\n'.join(tex_rows) + '\n', encoding='utf-8')

    mean_pg = mean(r['disagree_pg_rate'] for r in rows)
    mean_bc_total_pg = mean(r['both_correct_pg_rate_total'] for r in rows)
    mean_bc_cond_pg = mean(r['both_correct_pg_rate_given_disagree'] for r in rows)
    mean_shorter_chars_pg = mean(r['product_shorter_chars_pg'] for r in rows)
    mean_shorter_wrong_pg = mean(r['shorter_and_wrong_pg_rate_total'] for r in rows)

    top_pg = sorted(rows, key=lambda r: r['disagree_pg_rate'], reverse=True)
    summary = [
        'Prevalence table from existing natural-pool data',
        '',
        'Primary prevalence notion: product vs GM disagreement on natural 32-trace pools.',
        '',
        f"Mean product-vs-GM disagreement rate across 14 cells: {fmt_pct(mean_pg)}",
        f"Mean both-correct rate (of all problems) for product-vs-GM: {fmt_pct(mean_bc_total_pg)}",
        f"Mean both-correct share conditional on disagreement: {fmt_pct(mean_bc_cond_pg)}",
        f"Mean Pr(product shorter chars | product-vs-GM disagreement): {fmt_pct(mean_shorter_chars_pg)}",
        f"Mean shorter+wrong rate (of all problems) for product-vs-GM: {fmt_pct(mean_shorter_wrong_pg)}",
        '',
        'Highest product-vs-GM disagreement cells:',
    ]
    for r in top_pg[:5]:
        summary.append(
            f"- {r['prm_display']} / {r['benchmark_display']}: disagree={fmt_pct(r['disagree_pg_rate'])}, both-correct={fmt_pct(r['both_correct_pg_rate_total'])}, both-correct|disagree={fmt_pct(r['both_correct_pg_rate_given_disagree'])}"
        )
    summary += [
        '',
        'Interpretation:',
        '- On natural pools, the phenomenon is common rather than rare: product vs GM disagree on a substantial fraction of problems in every cell.',
        '- A large share of these disagreements are both-correct, so the practical issue is often trace selection among correct answers rather than outright answer failure.',
        '- Product-selected traces are usually shorter on disagreement cases, which matches the omission-bias story.',
    ]
    (OUT_DIR / 'summary.txt').write_text('\n'.join(summary) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
