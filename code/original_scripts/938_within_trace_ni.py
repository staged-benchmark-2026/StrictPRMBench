#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
UNION_INPUT_DIR = REPO_ROOT / 'outputs/reports/matched_control_multiseed/union'
ALL_ELIGIBLE_INPUT_DIR = REPO_ROOT / 'outputs/reports/matched_control_all_eligible'
SCORES_DIR = REPO_ROOT / 'outputs/matched_control/multiseed_scores'
OUT_DIR = REPO_ROOT / 'outputs/reports/within_trace'
ELIGIBLE_COUNTS_CSV = OUT_DIR / 'eligible_counts.csv'
WITHIN_TRACE_CSV = OUT_DIR / 'within_trace_ni.csv'
SUMMARY_TXT = OUT_DIR / 'summary.txt'
SEED = 42
BOOT_ITERS = 1000
CLIP_EPS = 1e-30

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
PRETTY = {
    'skywork_prm': 'Skywork-7B',
    'internlm2_7b_reward': 'InternLM2-7B',
    'qwen_prm': 'Qwen',
    'math_shepherd': 'Math-Shepherd',
    'skywork_prm_1_5b': 'Skywork-1.5B',
    'rlhflow_prm': 'RLHFlow',
    'internlm2_1_8b_reward': 'InternLM2-1.8B',
}
PRETTY_BENCH = {'gsm8k': 'GSM8K', 'math500': 'MATH-500'}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open('r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def trace_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row['problem_id']), int(row['candidate_index'])


def safe_float(x: Any) -> float:
    return float(x)


def product_flip(row: dict[str, Any]) -> int:
    return int(float(row['traj_logprod_modified']) > float(row['traj_logprod_clean']))


def bootstrap_paired_delta(diffs: list[float], seed_text: str, iters: int = BOOT_ITERS) -> tuple[float, float, float]:
    if not diffs:
        return float('nan'), float('nan'), float('nan')
    point = sum(diffs) / len(diffs)
    rng = random.Random(seed_text)
    boots: list[float] = []
    n = len(diffs)
    for _ in range(iters):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        boots.append(s / n)
    boots.sort()
    lo = boots[int(0.025 * len(boots))]
    hi = boots[int(0.975 * len(boots))]
    return float(point), float(lo), float(hi)


def fmt_pct(x: float) -> str:
    return f'{100.0 * x:.1f}%'


def fmt_signed(x: float) -> str:
    return f'{x:+.3f}'


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    eligible_rows: list[dict[str, Any]] = []
    overlap_meta: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}

    for bench in BENCHMARKS:
        all_nec = load_jsonl(ALL_ELIGIBLE_INPUT_DIR / f'{bench}_necessary_matched.jsonl')
        all_inert = load_jsonl(ALL_ELIGIBLE_INPUT_DIR / f'{bench}_inert_matched.jsonl')
        union_nec = load_jsonl(UNION_INPUT_DIR / f'{bench}_necessary_matched.jsonl')
        union_inert = load_jsonl(UNION_INPUT_DIR / f'{bench}_inert_matched.jsonl')

        all_nec_keys = {trace_key(r) for r in all_nec}
        all_inert_keys = {trace_key(r) for r in all_inert}
        union_nec_by_key = {trace_key(r): r for r in union_nec}
        union_inert_by_key = {trace_key(r): r for r in union_inert}

        all_overlap = sorted(all_nec_keys & all_inert_keys)
        union_overlap = sorted(set(union_nec_by_key) & set(union_inert_by_key))

        overlap_meta[bench] = {}
        for key in union_overlap:
            nr = union_nec_by_key[key]
            ir = union_inert_by_key[key]
            overlap_meta[bench][key] = {
                'problem_id': key[0],
                'candidate_index': key[1],
                'nec_pair_id': nr['pair_id'],
                'inert_pair_id': ir['pair_id'],
                'nec_deleted_step_index_0based': int(nr['deleted_step_index_0based']),
                'inert_deleted_step_index_0based': int(ir['deleted_step_index_0based']),
                'position_gap_abs': abs(int(nr['deleted_step_index_0based']) - int(ir['deleted_step_index_0based'])),
                'n_steps_clean': int(nr['n_steps_clean']),
            }

        eligible_rows.append({
            'benchmark': bench,
            'all_eligible_nec_rows': len(all_nec),
            'all_eligible_inert_rows': len(all_inert),
            'all_eligible_overlap_traces': len(all_overlap),
            'scored_union_nec_rows': len(union_nec),
            'scored_union_inert_rows': len(union_inert),
            'scored_union_overlap_traces': len(union_overlap),
            'mean_abs_position_gap_union': (
                sum(v['position_gap_abs'] for v in overlap_meta[bench].values()) / len(overlap_meta[bench])
                if overlap_meta[bench] else float('nan')
            ),
        })

    with ELIGIBLE_COUNTS_CSV.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(eligible_rows[0].keys()))
        w.writeheader()
        w.writerows(eligible_rows)

    within_rows: list[dict[str, Any]] = []
    total_traces = 0
    non_positive = 0
    ci_ex_zero = 0

    for prm in PRMS:
        for bench in BENCHMARKS:
            nec_scores = load_jsonl(SCORES_DIR / f'{prm}_{bench}_necessary.jsonl')
            inert_scores = load_jsonl(SCORES_DIR / f'{prm}_{bench}_inert.jsonl')
            nec_by_pair = {str(r['pair_id']): r for r in nec_scores}
            inert_by_pair = {str(r['pair_id']): r for r in inert_scores}

            apr_nec_bits: list[int] = []
            apr_inert_bits: list[int] = []
            diffs: list[float] = []
            clean_logprod_diffs: list[float] = []
            clean_gm_diffs: list[float] = []

            for key, meta in overlap_meta[bench].items():
                nr = nec_by_pair.get(meta['nec_pair_id'])
                ir = inert_by_pair.get(meta['inert_pair_id'])
                if nr is None or ir is None:
                    continue
                apr_nec = product_flip(nr)
                apr_inert = product_flip(ir)
                apr_nec_bits.append(apr_nec)
                apr_inert_bits.append(apr_inert)
                diffs.append(float(apr_nec - apr_inert))
                clean_logprod_diffs.append(abs(float(nr['traj_logprod_clean']) - float(ir['traj_logprod_clean'])))
                clean_gm_diffs.append(abs(float(nr['traj_gm_clean']) - float(ir['traj_gm_clean'])))

            n = len(diffs)
            apr_nec = sum(apr_nec_bits) / n if n else float('nan')
            apr_inert = sum(apr_inert_bits) / n if n else float('nan')
            delta, ci_lo, ci_hi = bootstrap_paired_delta(diffs, f'{SEED}:{prm}:{bench}:within_trace')
            row = {
                'prm': prm,
                'benchmark': bench,
                'n_eligible': n,
                'apr_necessary': apr_nec,
                'apr_inert': apr_inert,
                'delta_ni_within': delta,
                'ci_lo': ci_lo,
                'ci_hi': ci_hi,
                'mean_abs_clean_logprod_diff': (sum(clean_logprod_diffs) / len(clean_logprod_diffs)) if clean_logprod_diffs else float('nan'),
                'max_abs_clean_logprod_diff': max(clean_logprod_diffs) if clean_logprod_diffs else float('nan'),
                'mean_abs_clean_gm_diff': (sum(clean_gm_diffs) / len(clean_gm_diffs)) if clean_gm_diffs else float('nan'),
                'max_abs_clean_gm_diff': max(clean_gm_diffs) if clean_gm_diffs else float('nan'),
                'source_scope': 'matched_control_multiseed_union_rescored',
            }
            within_rows.append(row)
            total_traces += n
            if delta <= 0:
                non_positive += 1
            if ci_hi < 0 or ci_lo > 0:
                ci_ex_zero += 1

    within_rows.sort(key=lambda r: (r['benchmark'], r['prm']))
    with WITHIN_TRACE_CSV.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(within_rows[0].keys()))
        w.writeheader()
        w.writerows(within_rows)

    mean_delta = sum(float(r['delta_ni_within']) for r in within_rows) / len(within_rows)

    lines = []
    lines.append('Within-trace NI verification (strict rescored, zero new inference)')
    lines.append('')
    lines.append('Source scope')
    lines.append('- Strict within-trace analysis uses outputs/reports/matched_control_multiseed/union with rescored traces from outputs/matched_control/multiseed_scores.')
    lines.append('- This is the largest same-trace subset with cached full rescoring available at zero compute.')
    lines.append('')
    lines.append('Eligible overlap counts')
    for r in eligible_rows:
        lines.append(
            f"- {PRETTY_BENCH[r['benchmark']]}: all-eligible overlap={r['all_eligible_overlap_traces']}, "
            f"rescored union overlap={r['scored_union_overlap_traces']}, mean |position gap|={r['mean_abs_position_gap_union']:.2f}"
        )
    lines.append('')
    lines.append('Per-cell results (product APR within same trace)')
    for r in within_rows:
        lines.append(
            f"- {PRETTY[r['prm']]} / {PRETTY_BENCH[r['benchmark']]}: "
            f"n={r['n_eligible']}, APR_nec={r['apr_necessary']:.3f}, APR_inert={r['apr_inert']:.3f}, "
            f"ΔNI_within={fmt_signed(float(r['delta_ni_within']))} "
            f"[{fmt_signed(float(r['ci_lo']))}, {fmt_signed(float(r['ci_hi']))}]"
        )
    lines.append('')
    lines.append(f'- Total within-trace pairs analyzed across 14 cells: {total_traces}')
    lines.append(f'- Non-positive ΔNI_within: {non_positive}/{len(within_rows)}')
    lines.append(f'- CIs excluding zero: {ci_ex_zero}/{len(within_rows)}')
    lines.append(f'- Mean ΔNI_within across 14 cells: {fmt_signed(mean_delta)}')
    lines.append('')
    lines.append('Clean-score consistency check')
    lines.append('- Same-trace necessary/inert rows should have identical clean scores up to numerical noise; per-cell mean/max absolute clean-score differences are written to within_trace_ni.csv.')

    SUMMARY_TXT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'[saved] {ELIGIBLE_COUNTS_CSV}')
    print(f'[saved] {WITHIN_TRACE_CSV}')
    print(f'[saved] {SUMMARY_TXT}')
    print(f'[check] non_positive={non_positive}/{len(within_rows)} ci_ex_zero={ci_ex_zero}/{len(within_rows)} total_pairs={total_traces}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
