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
SCORES_DIR = REPO_ROOT / 'outputs/matched_control/multiseed_scores'
WITHIN_TRACE_CSV = REPO_ROOT / 'outputs/reports/within_trace/within_trace_ni.csv'
OUT_DIR = REPO_ROOT / 'outputs/reports/mechanical_null'
OUT_CSV = OUT_DIR / 'null_ni_gap.csv'
OUT_TXT = OUT_DIR / 'summary.txt'
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


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open('r', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def trace_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row['problem_id']), int(row['candidate_index'])


def safe_log(x: float) -> float:
    return math.log(max(CLIP_EPS, min(1.0, float(x))))


def product_flip_observed(row: dict[str, Any]) -> int:
    return int(float(row['traj_logprod_modified']) > float(row['traj_logprod_clean']))


def product_flip_null(row: dict[str, Any]) -> int:
    clean_scores = [float(x) for x in row['clean_step_scores']]
    k = int(row['deleted_step_index_0based'])
    clean_logprod = sum(safe_log(x) for x in clean_scores)
    null_logprod = sum(safe_log(x) for i, x in enumerate(clean_scores) if i != k)
    return int(null_logprod > clean_logprod)


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


def build_overlap_meta() -> dict[str, dict[tuple[str, int], dict[str, Any]]]:
    out: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    for bench in BENCHMARKS:
        union_nec = load_jsonl(UNION_INPUT_DIR / f'{bench}_necessary_matched.jsonl')
        union_inert = load_jsonl(UNION_INPUT_DIR / f'{bench}_inert_matched.jsonl')
        union_nec_by_key = {trace_key(r): r for r in union_nec}
        union_inert_by_key = {trace_key(r): r for r in union_inert}
        overlap = sorted(set(union_nec_by_key) & set(union_inert_by_key))
        out[bench] = {}
        for key in overlap:
            nr = union_nec_by_key[key]
            ir = union_inert_by_key[key]
            out[bench][key] = {
                'nec_pair_id': str(nr['pair_id']),
                'inert_pair_id': str(ir['pair_id']),
            }
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    observed_rows = {
        (r['prm'], r['benchmark']): r
        for r in read_csv_rows(WITHIN_TRACE_CSV)
    }
    overlap_meta = build_overlap_meta()

    rows_out: list[dict[str, Any]] = []
    null_nonpos = 0
    null_near_zero = 0
    observed_more_negative = 0

    for prm in PRMS:
        for bench in BENCHMARKS:
            nec_scores = load_jsonl(SCORES_DIR / f'{prm}_{bench}_necessary.jsonl')
            inert_scores = load_jsonl(SCORES_DIR / f'{prm}_{bench}_inert.jsonl')
            nec_by_pair = {str(r['pair_id']): r for r in nec_scores}
            inert_by_pair = {str(r['pair_id']): r for r in inert_scores}

            null_nec_bits: list[int] = []
            null_inert_bits: list[int] = []
            null_diffs: list[float] = []
            obs_diffs: list[float] = []
            nec_s_k: list[float] = []
            inert_s_k: list[float] = []

            for meta in overlap_meta[bench].values():
                nr = nec_by_pair.get(meta['nec_pair_id'])
                ir = inert_by_pair.get(meta['inert_pair_id'])
                if nr is None or ir is None:
                    continue
                nnull = product_flip_null(nr)
                inull = product_flip_null(ir)
                nobs = product_flip_observed(nr)
                iobs = product_flip_observed(ir)
                null_nec_bits.append(nnull)
                null_inert_bits.append(inull)
                null_diffs.append(float(nnull - inull))
                obs_diffs.append(float(nobs - iobs))
                nec_s_k.append(float(nr['clean_target_step_score']))
                inert_s_k.append(float(ir['clean_target_step_score']))

            n = len(null_diffs)
            apr_nec_null = sum(null_nec_bits) / n if n else float('nan')
            apr_inert_null = sum(null_inert_bits) / n if n else float('nan')
            delta_null, null_lo, null_hi = bootstrap_paired_delta(null_diffs, f'{SEED}:{prm}:{bench}:null')
            delta_obs, _, _ = bootstrap_paired_delta(obs_diffs, f'{SEED}:{prm}:{bench}:obs')
            obs = observed_rows[(prm, bench)]
            delta_obs_ref = float(obs['delta_ni_within'])
            ci_obs_lo = float(obs['ci_lo'])
            ci_obs_hi = float(obs['ci_hi'])
            diff_obs_minus_null = delta_obs_ref - delta_null
            rows_out.append({
                'prm': prm,
                'benchmark': bench,
                'n_eligible': n,
                'apr_necessary_null': apr_nec_null,
                'apr_inert_null': apr_inert_null,
                'delta_ni_null': delta_null,
                'null_ci_lo': null_lo,
                'null_ci_hi': null_hi,
                'apr_necessary_observed': float(obs['apr_necessary']),
                'apr_inert_observed': float(obs['apr_inert']),
                'delta_ni_observed': delta_obs_ref,
                'obs_ci_lo': ci_obs_lo,
                'obs_ci_hi': ci_obs_hi,
                'delta_observed_minus_null': diff_obs_minus_null,
                'mean_deleted_step_score_nec': sum(nec_s_k) / len(nec_s_k) if nec_s_k else float('nan'),
                'mean_deleted_step_score_inert': sum(inert_s_k) / len(inert_s_k) if inert_s_k else float('nan'),
            })
            if delta_null <= 0:
                null_nonpos += 1
            if abs(delta_null) <= 0.02:
                null_near_zero += 1
            if diff_obs_minus_null < 0:
                observed_more_negative += 1

    rows_out.sort(key=lambda r: (r['benchmark'], r['prm']))
    with OUT_CSV.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)

    mean_null = sum(float(r['delta_ni_null']) for r in rows_out) / len(rows_out)
    mean_obs = sum(float(r['delta_ni_observed']) for r in rows_out) / len(rows_out)
    mean_diff = sum(float(r['delta_observed_minus_null']) for r in rows_out) / len(rows_out)

    lines = []
    lines.append('Mechanical-only null vs observed within-trace NI')
    lines.append('')
    lines.append('Definition')
    lines.append('- Null APR removes the deleted step from the clean product score while freezing all remaining step scores at their clean values.')
    lines.append('- Observed APR uses the fully rescored deleted trace from the PRM.')
    lines.append('- Both analyses are computed on the same strict within-trace subset from experiment 938.')
    lines.append('')
    lines.append('Per-cell comparison')
    for r in rows_out:
        lines.append(
            f"- {PRETTY[r['prm']]} / {PRETTY_BENCH[r['benchmark']]}: "
            f"ΔNI_null={r['delta_ni_null']:+.3f} [{r['null_ci_lo']:+.3f}, {r['null_ci_hi']:+.3f}], "
            f"ΔNI_observed={r['delta_ni_observed']:+.3f} [{r['obs_ci_lo']:+.3f}, {r['obs_ci_hi']:+.3f}], "
            f"obs-null={r['delta_observed_minus_null']:+.3f}"
        )
    lines.append('')
    lines.append(f'- Null ΔNI non-positive: {null_nonpos}/{len(rows_out)}')
    lines.append(f'- Null ΔNI near zero (|Δ| <= 0.02): {null_near_zero}/{len(rows_out)}')
    lines.append(f'- Observed ΔNI more negative than null: {observed_more_negative}/{len(rows_out)}')
    lines.append(f'- Mean ΔNI_null across 14 cells: {mean_null:+.3f}')
    lines.append(f'- Mean ΔNI_observed across 14 cells: {mean_obs:+.3f}')
    lines.append(f'- Mean (observed - null): {mean_diff:+.3f}')

    OUT_TXT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'[saved] {OUT_CSV}')
    print(f'[saved] {OUT_TXT}')
    print(f'[check] null_nonpos={null_nonpos}/{len(rows_out)} null_near_zero={null_near_zero}/{len(rows_out)} observed_more_negative={observed_more_negative}/{len(rows_out)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
