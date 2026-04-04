#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import random
from functools import lru_cache
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
STEP_CACHE_DIR = REPO_ROOT / 'outputs/reports/aggregation_sweep/step_scores_cache'
ATTACK_DIR = REPO_ROOT / 'outputs/reports/matched_trace'
PER_PROBLEM_DIR = REPO_ROOT / 'outputs/reports/aggregation_sweep/per_problem'
MATCHED_CONTROL_DIR = REPO_ROOT / 'outputs/matched_control/scores'
MATCHED_FLIP_REF = ATTACK_DIR / 'matched_trace_flip.csv'
PARETO_REF = REPO_ROOT / 'outputs/reports/figures/pareto_data.csv'
NI_REF = REPO_ROOT / 'outputs/reports/ni_gap/ni_gap_7prm_product.csv'
OUT_DIR = REPO_ROOT / 'outputs/reports/alpha_sweep'
FLIP_OUT = OUT_DIR / 'flip32_by_alpha.csv'
NI_OUT = OUT_DIR / 'ni_gap_by_alpha.csv'
GAIN_OUT = OUT_DIR / 'gain32_by_alpha.csv'
SUMMARY_OUT = OUT_DIR / 'summary.txt'

PRMS = [
    'math_shepherd',
    'skywork_prm',
    'skywork_prm_1_5b',
    'qwen_prm',
    'rlhflow_prm',
    'internlm2_7b_reward',
    'internlm2_1_8b_reward',
]
BENCHMARKS = ['gsm8k', 'math500']
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
SEED = 42
BOOT_ITERS = 1000
CLIP_EPS = 1e-30
TOL = 1e-12
PRETTY = {
    'math_shepherd': 'Math-Shepherd',
    'skywork_prm': 'Skywork-7B',
    'skywork_prm_1_5b': 'Skywork-1.5B',
    'qwen_prm': 'Qwen',
    'rlhflow_prm': 'RLHFlow',
    'internlm2_7b_reward': 'InternLM2-7B',
    'internlm2_1_8b_reward': 'InternLM2-1.8B',
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def safe_log(x: float) -> float:
    return math.log(max(float(x), CLIP_EPS))


def alpha_log_score(step_scores: list[float], alpha: float) -> float:
    scores = [float(s) for s in step_scores]
    return float(sum(safe_log(s) for s in scores) / (len(scores) ** alpha))


def bootstrap_mean(values: list[float], iters: int = BOOT_ITERS, seed_text: str = '42') -> tuple[float, float, float]:
    if not values:
        return float('nan'), float('nan'), float('nan')
    rng = random.Random(seed_text)
    n = len(values)
    point = sum(values) / n
    boots = []
    for _ in range(iters):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(sample) / n)
    boots.sort()
    return float(point), float(boots[int(0.025 * len(boots))]), float(boots[int(0.975 * len(boots))])


def bootstrap_delta_binary(a: list[int], b: list[int], iters: int = BOOT_ITERS, seed_text: str = '42') -> tuple[float, float, float]:
    if not a or not b:
        return float('nan'), float('nan'), float('nan')
    rng = random.Random(seed_text)
    point = sum(a) / len(a) - sum(b) / len(b)
    boots = []
    for _ in range(iters):
        sa = [a[rng.randrange(len(a))] for _ in range(len(a))]
        sb = [b[rng.randrange(len(b))] for _ in range(len(b))]
        boots.append(sum(sa) / len(sa) - sum(sb) / len(sb))
    boots.sort()
    return float(point), float(boots[int(0.025 * len(boots))]), float(boots[int(0.975 * len(boots))])


def select_top1(pool_scores: dict[int, list[float]], alpha: float) -> int:
    best_idx = None
    best_score = None
    for idx in sorted(pool_scores.keys()):
        score = alpha_log_score(pool_scores[idx], alpha)
        if best_idx is None or score > best_score:
            best_idx = idx
            best_score = score
    assert best_idx is not None
    return int(best_idx)


def pick_attack_idx(n: int, top_idx: int) -> int:
    idx = n - 1
    if idx == top_idx:
        idx = n - 2
    return max(0, idx)


@lru_cache(maxsize=None)
def load_clean_scores(prm: str, benchmark: str) -> dict[str, dict[int, dict[str, Any]]]:
    path = STEP_CACHE_DIR / f'{prm}_{benchmark}.jsonl'
    grouped: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for rec in read_jsonl(path):
        grouped[str(rec['problem_id'])][int(rec['candidate_index'])] = {
            'step_scores': [float(x) for x in rec['step_scores']],
            'is_correct': bool(rec.get('is_correct', False)),
        }
    return grouped


@lru_cache(maxsize=None)
def load_attack_rows(prm: str, benchmark: str) -> dict[str, dict[str, Any]]:
    path = ATTACK_DIR / f'attack_scores_{prm}_{benchmark}.jsonl'
    return {str(r['problem_id']): r for r in read_jsonl(path)}


@lru_cache(maxsize=None)
def load_matched_per_problem(prm: str, benchmark: str) -> list[dict[str, Any]]:
    path = PER_PROBLEM_DIR / f'{benchmark}_{prm}_flip_gain_6rule_per_problem.jsonl'
    return read_jsonl(path)


def matched_flip_rows_for_alpha(prm: str, benchmark: str, alpha: float) -> list[dict[str, Any]]:
    clean = load_clean_scores(prm, benchmark)
    attacks = load_attack_rows(prm, benchmark)
    per_problem = load_matched_per_problem(prm, benchmark)
    rows = []
    for rec in per_problem:
        if int(rec.get('eligible_native', 0)) != 1:
            continue
        pid = str(rec['problem_id'])
        attack = attacks.get(pid)
        if attack is None or pid not in clean:
            continue
        pool_meta = clean[pid]
        pool_scores = {idx: meta['step_scores'][:] for idx, meta in pool_meta.items()}
        top_native_idx = int(rec.get('native_selected_idx', attack.get('native_top1_idx')))
        attack_idx = pick_attack_idx(len(pool_scores), top_native_idx)
        injected_scores = {idx: scores[:] for idx, scores in pool_scores.items()}
        injected_scores[attack_idx] = [float(x) for x in attack['attack_step_scores']]
        clean_top_idx = select_top1(pool_scores, alpha)
        inj_top_idx = select_top1(injected_scores, alpha)
        rows.append({
            'problem_id': pid,
            'clean_top_idx': clean_top_idx,
            'injected_top_idx': inj_top_idx,
            'flip': int(clean_top_idx != inj_top_idx),
        })
    return rows


def gain_rows_for_alpha(prm: str, benchmark: str, alpha: float) -> list[dict[str, Any]]:
    clean = load_clean_scores(prm, benchmark)
    pids = sorted(clean.keys())
    rng = random.Random(SEED)
    random_idx_by_pid = {pid: rng.randrange(len(clean[pid])) for pid in pids}
    rows = []
    for pid in pids:
        pool_meta = clean[pid]
        pool_scores = {idx: meta['step_scores'][:] for idx, meta in pool_meta.items()}
        best_idx = select_top1(pool_scores, alpha)
        random_idx = random_idx_by_pid[pid]
        rows.append({
            'problem_id': pid,
            'selected_idx': best_idx,
            'selected_correct': int(pool_meta[best_idx]['is_correct']),
            'random_idx': random_idx,
            'random_correct': int(pool_meta[random_idx]['is_correct']),
            'gain_indicator': int(pool_meta[best_idx]['is_correct']) - int(pool_meta[random_idx]['is_correct']),
        })
    return rows


@lru_cache(maxsize=None)
def load_ni_rows(prm: str, benchmark: str, variant: str) -> list[dict[str, Any]]:
    path = MATCHED_CONTROL_DIR / f'{prm}_{benchmark}_{variant}.jsonl'
    return read_jsonl(path)


def apr_bits_for_alpha(rows: list[dict[str, Any]], alpha: float) -> list[int]:
    bits = []
    for r in rows:
        clean_score = alpha_log_score(r['clean_step_scores'], alpha)
        modified_score = alpha_log_score(r['modified_step_scores'], alpha)
        bits.append(1 if modified_score > clean_score else 0)
    return bits


def load_flip_reference() -> dict[tuple[str, str, float], float]:
    out = {}
    with MATCHED_FLIP_REF.open('r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['agg_rule'] == 'product':
                out[(row['prm_name'], row['benchmark'], 0.0)] = float(row['matched_flip_at_32'])
            elif row['agg_rule'] in {'gm', 'mean_log'}:
                out[(row['prm_name'], row['benchmark'], 1.0)] = float(row['matched_flip_at_32'])
    return out


def load_gain_reference() -> dict[tuple[str, str, float], float]:
    out = {}
    with PARETO_REF.open('r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['rule'] == 'product':
                out[(row['prm'], row['benchmark'], 0.0)] = float(row['gain_at_32'])
            elif row['rule'] == 'gm':
                out[(row['prm'], row['benchmark'], 1.0)] = float(row['gain_at_32'])
    return out


def load_ni_reference() -> dict[tuple[str, str, float], float]:
    out = {}
    with NI_REF.open('r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['aggregation'] == 'product':
                out[(row['prm'], row['benchmark'], 0.0)] = float(row['delta_ni'])
            elif row['aggregation'] == 'gm':
                out[(row['prm'], row['benchmark'], 1.0)] = float(row['delta_ni'])
    return out


def fmt_alpha(alpha: float) -> str:
    return f'{alpha:.2f}'


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float('nan')


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    flip_ref = load_flip_reference()
    gain_ref = load_gain_reference()
    ni_ref = load_ni_reference()

    flip_rows_out: list[dict[str, Any]] = []
    gain_rows_out: list[dict[str, Any]] = []
    ni_rows_out: list[dict[str, Any]] = []

    for prm in PRMS:
        for benchmark in BENCHMARKS:
            print(f'[cell] {prm} / {benchmark}', flush=True)
            nec_rows = load_ni_rows(prm, benchmark, 'necessary')
            inert_rows = load_ni_rows(prm, benchmark, 'inert')
            for alpha in ALPHAS:
                flip_rows = matched_flip_rows_for_alpha(prm, benchmark, alpha)
                flip_bits = [int(r['flip']) for r in flip_rows]
                flip_point, flip_lo, flip_hi = bootstrap_mean([float(x) for x in flip_bits], seed_text=f'flip:{SEED}:{prm}:{benchmark}:{alpha}')
                flip_rows_out.append({
                    'checkpoint': prm,
                    'benchmark': benchmark,
                    'alpha': fmt_alpha(alpha),
                    'flip_rate': flip_point,
                    'ci_lo': flip_lo,
                    'ci_hi': flip_hi,
                    'n_eligible': len(flip_bits),
                })
                if alpha in {0.0, 1.0}:
                    ref = flip_ref[(prm, benchmark, alpha)]
                    if abs(flip_point - ref) > TOL:
                        raise RuntimeError(f'flip validation mismatch for {prm}/{benchmark}/alpha={alpha}: {flip_point} vs {ref}')

                gain_rows = gain_rows_for_alpha(prm, benchmark, alpha)
                gain_indicators = [float(r['gain_indicator']) for r in gain_rows]
                gain_point, gain_lo, gain_hi = bootstrap_mean(gain_indicators, seed_text=f'gain:{SEED}:{prm}:{benchmark}:{alpha}')
                pass_at_1 = mean([float(r['selected_correct']) for r in gain_rows])
                random_acc = mean([float(r['random_correct']) for r in gain_rows])
                gain_rows_out.append({
                    'checkpoint': prm,
                    'benchmark': benchmark,
                    'alpha': fmt_alpha(alpha),
                    'gain': gain_point,
                    'ci_lo': gain_lo,
                    'ci_hi': gain_hi,
                    'pass_at_1_alpha': pass_at_1,
                    'random_accuracy': random_acc,
                    'n_problems': len(gain_rows),
                })
                if alpha in {0.0, 1.0}:
                    ref = gain_ref[(prm, benchmark, alpha)]
                    if abs(gain_point - ref) > TOL:
                        raise RuntimeError(f'gain validation mismatch for {prm}/{benchmark}/alpha={alpha}: {gain_point} vs {ref}')

                nec_bits = apr_bits_for_alpha(nec_rows, alpha)
                inert_bits = apr_bits_for_alpha(inert_rows, alpha)
                delta_point, delta_lo, delta_hi = bootstrap_delta_binary(nec_bits, inert_bits, seed_text=f'ni:{SEED}:{prm}:{benchmark}:{alpha}')
                ni_rows_out.append({
                    'checkpoint': prm,
                    'benchmark': benchmark,
                    'alpha': fmt_alpha(alpha),
                    'apr_nec': mean([float(x) for x in nec_bits]),
                    'apr_inert': mean([float(x) for x in inert_bits]),
                    'delta_ni': delta_point,
                    'ci_lo': delta_lo,
                    'ci_hi': delta_hi,
                    'n_nec': len(nec_bits),
                    'n_inert': len(inert_bits),
                })
                if alpha in {0.0, 1.0}:
                    ref = ni_ref[(prm, benchmark, alpha)]
                    if abs(delta_point - ref) > 1e-6:
                        raise RuntimeError(f'NI validation mismatch for {prm}/{benchmark}/alpha={alpha}: {delta_point} vs {ref}')

    flip_rows_out.sort(key=lambda r: (r['benchmark'], r['checkpoint'], float(r['alpha'])))
    gain_rows_out.sort(key=lambda r: (r['benchmark'], r['checkpoint'], float(r['alpha'])))
    ni_rows_out.sort(key=lambda r: (r['benchmark'], r['checkpoint'], float(r['alpha'])))
    write_csv(FLIP_OUT, flip_rows_out)
    write_csv(GAIN_OUT, gain_rows_out)
    write_csv(NI_OUT, ni_rows_out)

    # summary
    flip_by_cell = defaultdict(list)
    for row in flip_rows_out:
        flip_by_cell[(row['checkpoint'], row['benchmark'])].append(row)
    monotonic = 0
    alpha1_optimal = 0
    recov50 = 0
    recov80 = 0
    valid_gap = 0
    for key, rows in flip_by_cell.items():
        rows = sorted(rows, key=lambda r: float(r['alpha']))
        vals = [float(r['flip_rate']) for r in rows]
        if all(vals[i] >= vals[i+1] - TOL for i in range(len(vals)-1)):
            monotonic += 1
        if vals[-1] <= min(vals) + TOL:
            alpha1_optimal += 1
        denom = vals[0] - vals[-1]
        if denom > TOL:
            valid_gap += 1
            recovered = (vals[0] - vals[2]) / denom
            if recovered >= 0.5 - TOL:
                recov50 += 1
            if recovered >= 0.8 - TOL:
                recov80 += 1

    mean_flip_by_alpha = {}
    mean_gain_by_alpha = {}
    for alpha in ALPHAS:
        a = fmt_alpha(alpha)
        mean_flip_by_alpha[a] = mean([float(r['flip_rate']) for r in flip_rows_out if r['alpha'] == a])
        mean_gain_by_alpha[a] = mean([float(r['gain']) for r in gain_rows_out if r['alpha'] == a])

    ni_summary = {}
    for alpha in [0.0, 0.5, 1.0]:
        a = fmt_alpha(alpha)
        vals = [float(r['delta_ni']) for r in ni_rows_out if r['alpha'] == a]
        ni_summary[a] = {
            'mean': mean(vals),
            'nonpos': sum(1 for v in vals if v <= TOL),
            'n': len(vals),
        }

    gain_range = (min(mean_gain_by_alpha.values()), max(mean_gain_by_alpha.values()))
    if monotonic >= 12 and alpha1_optimal >= 12:
        conclusion = 'alpha=1.0 is the robustness optimum'
    elif recov50 >= 12:
        conclusion = 'any normalization helps, but alpha=1.0 remains the clean endpoint'
    else:
        conclusion = 'intermediate alpha helps, but the sweep is not purely monotone'

    lines = [
        '=== α-Sweep Length Normalization ===',
        '',
        f'Flip@32 monotonically decreasing with α: {monotonic}/14 cells',
        f'α=1.0 is Flip-optimal: {alpha1_optimal}/14 cells',
        f'α=0.5 recovers ≥50% of product→GM gap: {recov50}/{valid_gap} valid-gap cells',
        f'α=0.5 recovers ≥80% of product→GM gap: {recov80}/{valid_gap} valid-gap cells',
        '',
        'Mean Flip@32 across 14 cells:',
    ]
    for alpha in ALPHAS:
        a = fmt_alpha(alpha)
        label = ' (product)' if alpha == 0.0 else ' (GM)' if alpha == 1.0 else ''
        lines.append(f'  α={a}{label}: {mean_flip_by_alpha[a]:.3f}')
    lines.extend([
        '',
        f"Δ_NI at α=0.00 (product): mean {ni_summary['0.00']['mean']:+.3f}, {ni_summary['0.00']['nonpos']}/{ni_summary['0.00']['n']} ≤ 0",
        f"Δ_NI at α=0.50:           mean {ni_summary['0.50']['mean']:+.3f}, {ni_summary['0.50']['nonpos']}/{ni_summary['0.50']['n']} ≤ 0",
        f"Δ_NI at α=1.00 (GM):      mean {ni_summary['1.00']['mean']:+.3f}, {ni_summary['1.00']['nonpos']}/{ni_summary['1.00']['n']} ≤ 0",
        '',
        f'Gain@32 mean range across α: [{gain_range[0]:+.3f}, {gain_range[1]:+.3f}]',
        '',
        f'Conclusion: {conclusion}',
        '',
        'Validation',
        '- α=0 flip matches matched-trace product rows exactly.',
        '- α=1 flip matches matched-trace GM rows exactly.',
        '- α=0/1 gain matches clean-pool product/GM pareto rows exactly.',
        '- α=0/1 Δ_NI matches 7-PRM product/GM NI tables exactly.',
    ])
    SUMMARY_OUT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'[saved] {FLIP_OUT}')
    print(f'[saved] {NI_OUT}')
    print(f'[saved] {GAIN_OUT}')
    print(f'[saved] {SUMMARY_OUT}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
