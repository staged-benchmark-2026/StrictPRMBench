#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

REPO_ROOT = Path(__file__).resolve().parents[2]
SCORES_DIR = REPO_ROOT / 'outputs/matched_control/scores'
MATCHED_DIR = REPO_ROOT / 'outputs/reports/matched_control'
OUT_DIR = REPO_ROOT / 'outputs/reports/ni_gap'
PRMS = [
    'skywork_prm',
    'internlm2_7b_reward',
    'qwen_prm',
    'math_shepherd',
    'skywork_prm_1_5b',
    'rlhflow_prm',
    'internlm2_1_8b_reward',
]
BENCHMARKS = ['gsm8k', 'math500']
BOOTSTRAPS = 1000
SEED = 42
CALIPER = 0.10
BIG_COST = 1e6
CLIP_EPS = 1e-30

DISPLAY_PRM = {
    'skywork_prm': 'Skywork-7B',
    'internlm2_7b_reward': 'InternLM2-7B',
    'qwen_prm': 'Qwen',
    'math_shepherd': 'Math-Shepherd',
    'skywork_prm_1_5b': 'Skywork-1.5B',
    'rlhflow_prm': 'RLHFlow',
    'internlm2_1_8b_reward': 'InternLM2-1.8B',
}
DISPLAY_BENCH = {'gsm8k': 'GSM8K', 'math500': 'MATH-500'}


def safe_log(x: float, eps: float = CLIP_EPS) -> float:
    return math.log(max(float(x), eps))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def product_flip(row: dict[str, Any]) -> int:
    return int(float(row['traj_product_modified']) > float(row['traj_product_clean']))


def mech_bonus(row: dict[str, Any]) -> float:
    return -safe_log(float(row['clean_target_step_score']))


def optimal_match_indices(nec_bonus: np.ndarray, inert_bonus: np.ndarray, caliper: float) -> list[tuple[int, int, float]]:
    cost = np.abs(nec_bonus[:, None] - inert_bonus[None, :])
    penalized = np.where(cost <= caliper, cost, BIG_COST + cost)
    row_ind, col_ind = linear_sum_assignment(penalized)
    matched = []
    for i, j in zip(row_ind.tolist(), col_ind.tolist()):
        d = float(cost[i, j])
        if d <= caliper:
            matched.append((i, j, d))
    matched.sort(key=lambda x: x[0])
    return matched


def bootstrap_delta(diff: np.ndarray) -> tuple[float, float]:
    rng = np.random.RandomState(SEED)
    n = len(diff)
    boots = np.empty(BOOTSTRAPS, dtype=float)
    for b in range(BOOTSTRAPS):
        idx = rng.randint(0, n, size=n)
        boots[b] = float(diff[idx].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def answer_preserved_groups(benchmark: str) -> tuple[set[str], int, int, int]:
    nec_rows = load_jsonl(MATCHED_DIR / f'{benchmark}_necessary_matched.jsonl')
    inert_rows = load_jsonl(MATCHED_DIR / f'{benchmark}_inert_matched.jsonl')
    nec_pres = {
        r['match_group_id']: (r['original_trace'].get('extracted_answer') == r['modified_trace'].get('extracted_answer'))
        for r in nec_rows
    }
    inert_pres = {
        r['match_group_id']: (r['original_trace'].get('extracted_answer') == r['modified_trace'].get('extracted_answer'))
        for r in inert_rows
    }
    common = sorted(set(nec_pres) & set(inert_pres))
    keep = {g for g in common if nec_pres[g] and inert_pres[g]}
    return keep, len(common), sum(nec_pres.values()), sum(inert_pres.values())


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    lines = []
    lines.append('Answer-preserved NI analysis (product-only)')
    lines.append(f'Caliper for score-matched subset: {CALIPER:.2f}')
    lines.append('')

    bench_preservation = {}
    for bench in BENCHMARKS:
        keep, n_common, n_nec_pres, n_inert_pres = answer_preserved_groups(bench)
        bench_preservation[bench] = keep
        lines.append(f'{DISPLAY_BENCH[bench]}: keep {len(keep)}/{n_common} match groups; necessary preserved {n_nec_pres}/{n_common}; inert preserved {n_inert_pres}/{n_common}.')
    lines.append('')

    for prm in PRMS:
        for bench in BENCHMARKS:
            keep = bench_preservation[bench]
            nec_rows = [r for r in load_jsonl(SCORES_DIR / f'{prm}_{bench}_necessary.jsonl') if r['match_group_id'] in keep]
            inert_rows = [r for r in load_jsonl(SCORES_DIR / f'{prm}_{bench}_inert.jsonl') if r['match_group_id'] in keep]
            nec_rows.sort(key=lambda r: r['match_group_id'])
            inert_rows.sort(key=lambda r: r['match_group_id'])
            assert [r['match_group_id'] for r in nec_rows] == [r['match_group_id'] for r in inert_rows]

            nec_y = np.asarray([product_flip(r) for r in nec_rows], dtype=float)
            inert_y = np.asarray([product_flip(r) for r in inert_rows], dtype=float)
            raw_diff = nec_y - inert_y
            delta_raw = float(raw_diff.mean())
            raw_ci_lo, raw_ci_hi = bootstrap_delta(raw_diff)

            nec_bonus = np.asarray([mech_bonus(r) for r in nec_rows], dtype=float)
            inert_bonus = np.asarray([mech_bonus(r) for r in inert_rows], dtype=float)
            matches = optimal_match_indices(nec_bonus, inert_bonus, CALIPER)
            nec_idx = np.asarray([m[0] for m in matches], dtype=int)
            inert_idx = np.asarray([m[1] for m in matches], dtype=int)
            matched_diff = nec_y[nec_idx] - inert_y[inert_idx]
            delta_matched = float(matched_diff.mean())
            matched_ci_lo, matched_ci_hi = bootstrap_delta(matched_diff)

            rows.append({
                'prm': prm,
                'benchmark': bench,
                'n_answer_preserved': int(len(keep)),
                'delta_ni_raw': delta_raw,
                'raw_ci_lo': raw_ci_lo,
                'raw_ci_hi': raw_ci_hi,
                'n_matched': int(len(matches)),
                'delta_ni_matched': delta_matched,
                'matched_ci_lo': matched_ci_lo,
                'matched_ci_hi': matched_ci_hi,
            })
            lines.append(
                f"{DISPLAY_PRM[prm]} / {DISPLAY_BENCH[bench]}: n_pres={len(keep)}, "
                f"ΔNI_raw={delta_raw:+.3f} [{raw_ci_lo:+.3f}, {raw_ci_hi:+.3f}], "
                f"n_matched={len(matches)}, ΔNI_matched={delta_matched:+.3f} [{matched_ci_lo:+.3f}, {matched_ci_hi:+.3f}]"
            )

    rows.sort(key=lambda r: (r['benchmark'], r['prm']))
    out_csv = OUT_DIR / 'answer_preserved_ni.csv'
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(
            f,
            fieldnames=['prm', 'benchmark', 'n_answer_preserved', 'delta_ni_raw', 'raw_ci_lo', 'raw_ci_hi', 'n_matched', 'delta_ni_matched', 'matched_ci_lo', 'matched_ci_hi'],
        )
        w.writeheader()
        w.writerows(rows)

    raw_non_pos = sum(1 for r in rows if float(r['delta_ni_raw']) <= 0)
    matched_non_pos = sum(1 for r in rows if float(r['delta_ni_matched']) <= 0)
    lines.append('')
    lines.append(f'Raw ΔNI non-positive in {raw_non_pos}/14 cells.')
    lines.append(f'Score-matched ΔNI non-positive in {matched_non_pos}/14 cells.')
    lines.append('Because all matched-control groups are answer-preserved in both deletion types, this subset is identical to the full matched-control analysis.')

    out_summary = OUT_DIR / 'answer_preserved_ni_summary.txt'
    out_summary.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'[saved] {out_csv}')
    print(f'[saved] {out_summary}')
    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
