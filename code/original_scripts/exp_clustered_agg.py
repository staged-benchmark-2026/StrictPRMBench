#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.answer_extraction import extract_answer
from src.utils.numeric_equiv import normalize_numeric_text, parse_number

OUT_DIR = REPO_ROOT / 'outputs/reports/clustered_agg'
MATCHED_DIR = REPO_ROOT / 'outputs/reports/matched_trace'
AGG_SWEEP_DIR = REPO_ROOT / 'outputs/reports/aggregation_sweep'
CLIP_EPS = 1e-30
BOOTSTRAPS = 1000
SEED = 42
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
DISPLAY = {
    'math_shepherd': 'Math-Shepherd',
    'skywork_prm': 'Skywork-7B',
    'skywork_prm_1_5b': 'Skywork-1.5B',
    'qwen_prm': 'Qwen',
    'rlhflow_prm': 'RLHFlow',
    'internlm2_7b_reward': 'InternLM2-7B',
    'internlm2_1_8b_reward': 'InternLM2-1.8B',
}
OFFICIAL_MAP = {
    'qwen_prm': 'product',
    'skywork_prm': 'official_mean',
    'skywork_prm_1_5b': 'official_mean',
    'rlhflow_prm': 'official_mean',
    'math_shepherd': 'min',
    'internlm2_7b_reward': 'last',
    'internlm2_1_8b_reward': 'last',
}
ALL_METHODS = ['product', 'gm', 'official', 'cluster_sum_gm', 'cluster_majority_weighted_gm', 'cluster_majority_vote']
PRIMARY_METHODS = ['product', 'gm', 'cluster_sum_gm', 'official']


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def default_pool_dir(benchmark: str) -> Path:
    if benchmark == 'gsm8k':
        p = REPO_ROOT / 'outputs/openclaw_run/bon_candidates/gsm8k_full_500'
        return p if p.exists() else REPO_ROOT / 'outputs/openclaw_run/bon_candidates/gsm8k'
    p = REPO_ROOT / 'outputs/openclaw_run/bon_candidates/math500_full_500'
    return p if p.exists() else REPO_ROOT / 'outputs/openclaw_run/bon_candidates/math500'


def read_step_cache(prm: str, benchmark: str) -> dict[str, dict[int, list[float]]]:
    path = AGG_SWEEP_DIR / 'step_scores_cache' / f'{prm}_{benchmark}.jsonl'
    out: dict[str, dict[int, list[float]]] = {}
    for row in load_jsonl(path):
        out.setdefault(str(row['problem_id']), {})[int(row['candidate_index'])] = [float(x) for x in row['step_scores']]
    return out


def read_random_map(prm: str, benchmark: str) -> dict[str, int]:
    path = AGG_SWEEP_DIR / 'per_problem' / f'{benchmark}_{prm}_flip_gain_6rule_per_problem.jsonl'
    return {str(row['problem_id']): int(row['random_correct']) for row in load_jsonl(path)}


def read_attack_scores(prm: str, benchmark: str) -> dict[str, dict[str, Any]]:
    path = MATCHED_DIR / f'attack_scores_{prm}_{benchmark}.jsonl'
    return {str(row['problem_id']): row for row in load_jsonl(path)}


def read_deleted_rows(prm: str, benchmark: str) -> dict[str, dict[str, Any]]:
    path = MATCHED_DIR / f'native_top1_deleted_variants_{prm}_{benchmark}.jsonl'
    return {str(row['problem_id']): row for row in load_jsonl(path)}


def bootstrap_binary(vals: list[int], seed: int, iters: int = BOOTSTRAPS) -> tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(vals)
    arr = []
    for _ in range(iters):
        s = 0
        for _ in range(n):
            s += vals[rng.randrange(n)]
        arr.append(s / n)
    arr.sort()
    return float(arr[int(0.025 * len(arr))]), float(arr[int(0.975 * len(arr))])


def canonical_answer(text: str | None) -> str:
    if text is None:
        return '__NONE__'
    value = str(text).strip()
    if not value:
        return '__NONE__'
    num = parse_number(value)
    if num is not None:
        return f'NUM:{num:.12g}'
    norm = normalize_numeric_text(value)
    return f'STR:{norm}' if norm else '__NONE__'


def answer_from_row(row: dict[str, Any]) -> str:
    ans = row.get('extracted_answer')
    if ans is not None and str(ans).strip():
        return str(ans).strip()
    raw = str(row.get('raw_text', ''))
    extracted = extract_answer(raw)
    return '' if extracted is None else str(extracted).strip()


def gm_score(step_scores: list[float]) -> float:
    vals = [max(CLIP_EPS, float(x)) for x in step_scores]
    return float(math.exp(sum(math.log(v) for v in vals) / len(vals)))


def product_log(step_scores: list[float]) -> float:
    return float(sum(math.log(max(CLIP_EPS, float(x))) for x in step_scores))


def mean_score(step_scores: list[float]) -> float:
    return float(sum(step_scores) / len(step_scores))


def min_score(step_scores: list[float]) -> float:
    return float(min(step_scores))


def last_score(step_scores: list[float]) -> float:
    return float(step_scores[-1])


def base_method_score(step_scores: list[float], prm: str, method: str) -> float:
    if method == 'product':
        return product_log(step_scores)
    if method == 'gm':
        return gm_score(step_scores)
    if method == 'official':
        rule = OFFICIAL_MAP[prm]
        if rule == 'product':
            return product_log(step_scores)
        if rule == 'official_mean':
            return mean_score(step_scores)
        if rule == 'min':
            return min_score(step_scores)
        if rule == 'last':
            return last_score(step_scores)
    raise KeyError(method)


def select_flat(cands: list[dict[str, Any]], prm: str, method: str) -> dict[str, Any]:
    return max(cands, key=lambda c: (base_method_score(c['step_scores'], prm, method), -c['idx']))


def select_cluster(cands: list[dict[str, Any]], method: str) -> dict[str, Any]:
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cand in cands:
        clusters[cand['answer_key']].append(cand)
    scored_clusters = []
    for answer_key, members in clusters.items():
        gm_vals = [gm_score(m['step_scores']) for m in members]
        gm_sum = float(sum(gm_vals))
        gm_best = float(max(gm_vals))
        if method in {'cluster_sum_gm', 'cluster_majority_weighted_gm'}:
            cluster_score = gm_sum
        elif method == 'cluster_majority_vote':
            cluster_score = float(len(members))
        else:
            raise KeyError(method)
        scored_clusters.append((answer_key, members, cluster_score, len(members), gm_sum, gm_best))
    _, members, _, _, _, _ = max(scored_clusters, key=lambda x: (x[2], x[3], x[4], x[5], x[0]))
    return max(members, key=lambda c: (gm_score(c['step_scores']), -c['idx']))


def select_method(cands: list[dict[str, Any]], prm: str, method: str) -> dict[str, Any]:
    if method in {'product', 'gm', 'official'}:
        return select_flat(cands, prm, method)
    return select_cluster(cands, method)


def attack_slot(n: int, native_top1_idx: int) -> int:
    idx = n - 1
    if idx == native_top1_idx:
        idx = n - 2
    return max(0, idx)


def latex_escape(text: str) -> str:
    return text.replace('&', '\\&').replace('_', '\\_')


def prepare_clean_pool(payload: dict[str, Any], pid: str, step_cache: dict[str, dict[int, list[float]]]) -> list[dict[str, Any]] | None:
    cands = payload['candidates'][:32]
    if len(cands) < 32:
        return None
    if pid not in step_cache:
        return None
    out = []
    for idx, cand in enumerate(cands):
        scores = step_cache[pid].get(idx)
        if scores is None:
            return None
        ans = answer_from_row(cand)
        out.append({
            'identity': f'orig:{idx}',
            'idx': idx,
            'step_scores': scores,
            'answer_text': ans,
            'answer_key': canonical_answer(ans),
            'is_correct': bool(cand.get('is_correct', False)),
        })
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    detailed_rows: list[dict[str, Any]] = []
    primary_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []

    gm_best = 0
    cluster_best = 0
    product_worst = 0
    cell_count = 0

    for prm in PRMS:
        for benchmark in BENCHMARKS:
            step_cache = read_step_cache(prm, benchmark)
            random_map = read_random_map(prm, benchmark)
            attack_scores = read_attack_scores(prm, benchmark)
            deleted_rows = read_deleted_rows(prm, benchmark)
            pool_dir = default_pool_dir(benchmark)

            clean_pools: dict[str, list[dict[str, Any]]] = {}
            for pool_file in sorted(pool_dir.glob('*.json')):
                payload = load_json(pool_file)
                pid = str(payload['problem_id'])
                prepared = prepare_clean_pool(payload, pid, step_cache)
                if prepared is not None:
                    clean_pools[pid] = prepared

            utility_map: dict[str, dict[str, int]] = {m: {} for m in ALL_METHODS}
            for pid, pool in clean_pools.items():
                if pid not in random_map:
                    continue
                for method in ALL_METHODS:
                    utility_map[method][pid] = 1 if select_method(pool, prm, method)['is_correct'] else 0

            method_flip_vals: dict[str, list[int]] = {m: [] for m in ALL_METHODS}
            method_wrong_vals: dict[str, list[int]] = {m: [] for m in ALL_METHODS}
            method_eligible: dict[str, int] = {m: 0 for m in ALL_METHODS}
            n_attack_base = 0
            for pid, pool in clean_pools.items():
                if pid not in attack_scores or pid not in deleted_rows:
                    continue
                meta = deleted_rows[pid]
                if not meta.get('ok'):
                    continue
                atk = attack_scores[pid]
                attack_idx = attack_slot(len(pool), int(atk['native_top1_idx']))
                attack_row = meta['deleted_variant_row']
                attack_ans = answer_from_row(attack_row)
                injected_pool = [dict(c) for c in pool]
                injected_pool[attack_idx] = {
                    'identity': f'attack:{attack_idx}',
                    'idx': attack_idx,
                    'step_scores': [float(x) for x in atk['attack_step_scores']],
                    'answer_text': attack_ans,
                    'answer_key': canonical_answer(attack_ans),
                    'is_correct': bool(atk['attack_is_correct']),
                }
                n_attack_base += 1
                for method in ALL_METHODS:
                    clean_sel = select_method(pool, prm, method)
                    if not clean_sel['is_correct']:
                        continue
                    method_eligible[method] += 1
                    inj_sel = select_method(injected_pool, prm, method)
                    flip = 1 if inj_sel['identity'] != clean_sel['identity'] else 0
                    wrong = 1 if (flip and not inj_sel['is_correct']) else 0
                    method_flip_vals[method].append(flip)
                    method_wrong_vals[method].append(wrong)

            method_to_flip: dict[str, float] = {}
            method_to_wrong: dict[str, float] = {}
            method_to_gain: dict[str, float] = {}
            method_to_pass: dict[str, float] = {}
            for method in ALL_METHODS:
                pids = sorted(utility_map[method].keys())
                deltas = [utility_map[method][pid] - random_map[pid] for pid in pids]
                gain = sum(deltas) / len(deltas) if deltas else 0.0
                pass_at_1 = sum(utility_map[method].values()) / len(pids) if pids else 0.0
                flip_vals = method_flip_vals[method]
                wrong_vals = method_wrong_vals[method]
                flip_rate = sum(flip_vals) / len(flip_vals) if flip_vals else 0.0
                wrong_rate = sum(wrong_vals) / len(flip_vals) if flip_vals else 0.0
                flip_lo, flip_hi = bootstrap_binary(flip_vals, seed=SEED + hash((prm, benchmark, method, 'flip')) % 10000)
                wrong_lo, wrong_hi = bootstrap_binary(wrong_vals, seed=SEED + hash((prm, benchmark, method, 'wrong')) % 10000)
                detailed_rows.append({
                    'prm': prm,
                    'benchmark': benchmark,
                    'method': method,
                    'n_total_clean': len(pids),
                    'n_attack_base': n_attack_base,
                    'n_method_eligible': method_eligible[method],
                    'flip_rate': flip_rate,
                    'flip_ci_lo': flip_lo,
                    'flip_ci_hi': flip_hi,
                    'wrong_rate': wrong_rate,
                    'wrong_ci_lo': wrong_lo,
                    'wrong_ci_hi': wrong_hi,
                    'pass_at_1': pass_at_1,
                    'gain_at_32': gain,
                    'random_accuracy': sum(random_map[pid] for pid in pids) / len(pids) if pids else 0.0,
                })
                method_to_flip[method] = flip_rate
                method_to_wrong[method] = wrong_rate
                method_to_gain[method] = gain
                method_to_pass[method] = pass_at_1

            sensitivity_rows.append({
                'prm': prm,
                'benchmark': benchmark,
                'cluster_sum_gm_flip': method_to_flip['cluster_sum_gm'],
                'cluster_majority_weighted_gm_flip': method_to_flip['cluster_majority_weighted_gm'],
                'cluster_majority_vote_flip': method_to_flip['cluster_majority_vote'],
                'sum_equals_weighted_exactly': int(abs(method_to_flip['cluster_sum_gm'] - method_to_flip['cluster_majority_weighted_gm']) < 1e-12),
            })

            primary = {
                'prm': prm,
                'benchmark': benchmark,
                'n_attack_base': n_attack_base,
                'n_elig_product': method_eligible['product'],
                'n_elig_gm': method_eligible['gm'],
                'n_elig_cluster': method_eligible['cluster_sum_gm'],
                'n_elig_official': method_eligible['official'],
                'Flip_prod': method_to_flip['product'],
                'Flip_GM': method_to_flip['gm'],
                'Flip_cluster': method_to_flip['cluster_sum_gm'],
                'Flip_official': method_to_flip['official'],
                'Wrong_prod': method_to_wrong['product'],
                'Wrong_GM': method_to_wrong['gm'],
                'Wrong_cluster': method_to_wrong['cluster_sum_gm'],
                'Wrong_official': method_to_wrong['official'],
                'Gain_prod': method_to_gain['product'],
                'Gain_GM': method_to_gain['gm'],
                'Gain_cluster': method_to_gain['cluster_sum_gm'],
                'Gain_official': method_to_gain['official'],
                'Pass_prod': method_to_pass['product'],
                'Pass_GM': method_to_pass['gm'],
                'Pass_cluster': method_to_pass['cluster_sum_gm'],
                'Pass_official': method_to_pass['official'],
            }
            flip_primary = {
                'product': primary['Flip_prod'],
                'gm': primary['Flip_GM'],
                'cluster_sum_gm': primary['Flip_cluster'],
                'official': primary['Flip_official'],
            }
            gain_primary = {
                'product': primary['Gain_prod'],
                'gm': primary['Gain_GM'],
                'cluster_sum_gm': primary['Gain_cluster'],
                'official': primary['Gain_official'],
            }
            min_flip = min(flip_primary.values())
            max_flip = max(flip_primary.values())
            max_gain = max(gain_primary.values())
            stab_winners = [k for k, v in flip_primary.items() if abs(v - min_flip) < 1e-12]
            util_winners = [k for k, v in gain_primary.items() if abs(v - max_gain) < 1e-12]
            primary['Stab_Winner'] = '|'.join(stab_winners)
            primary['Util_Winner'] = '|'.join(util_winners)
            primary_rows.append(primary)

            cell_count += 1
            if 'gm' in stab_winners:
                gm_best += 1
            if 'cluster_sum_gm' in stab_winners:
                cluster_best += 1
            if 'product' in [k for k, v in flip_primary.items() if abs(v - max_flip) < 1e-12]:
                product_worst += 1

    detailed_rows.sort(key=lambda r: (BENCHMARKS.index(r['benchmark']), PRMS.index(r['prm']), ALL_METHODS.index(r['method'])))
    primary_rows.sort(key=lambda r: (BENCHMARKS.index(r['benchmark']), PRMS.index(r['prm'])))
    sensitivity_rows.sort(key=lambda r: (BENCHMARKS.index(r['benchmark']), PRMS.index(r['prm'])))

    write_csv(OUT_DIR / 'flip32_comparison.csv', detailed_rows)
    write_csv(OUT_DIR / 'gain32_comparison.csv', [{k: r[k] for k in ['prm','benchmark','method','n_total_clean','pass_at_1','gain_at_32','random_accuracy']} for r in detailed_rows])
    write_csv(OUT_DIR / 'primary_table.csv', primary_rows)
    write_csv(OUT_DIR / 'sensitivity_cluster_variants.csv', sensitivity_rows)

    tex = [
        '\\begin{tabular}{llrcccccc}',
        '\\toprule',
        'PRM & Bench & $n_{base}$ & Flip-Prod & Flip-GM & Flip-Cluster & Flip-Off & Gain-Cluster & Stab Winner \\\\',
        '\\midrule',
    ]
    for r in primary_rows:
        tex.append(
            f"{latex_escape(DISPLAY[r['prm']])} & {latex_escape(r['benchmark'].upper())} & {r['n_attack_base']} & {r['Flip_prod']:.3f} & {r['Flip_GM']:.3f} & {r['Flip_cluster']:.3f} & {r['Flip_official']:.3f} & {r['Gain_cluster']:.3f} & {latex_escape(r['Stab_Winner'])} \\\\"
        )
    tex.extend(['\\bottomrule', '\\end{tabular}', ''])
    (OUT_DIR / 'primary_table.tex').write_text('\n'.join(tex), encoding='utf-8')

    summary = [
        '=== Answer-Clustered Aggregation Audit ===',
        '',
        'Primary clustered method: cluster_sum_gm.',
        'Note: cluster_majority_weighted_gm is exactly identical to cluster_sum_gm because |cluster| × mean(GM) = sum(GM).',
        'Sensitivity method: cluster_majority_vote.',
        '',
        f'Product worst Flip@32 (primary four-rule comparison): {product_worst}/{cell_count} cells',
        f'GM stability-best (ties allowed): {gm_best}/{cell_count} cells',
        f'Cluster_sum_gm stability-best (ties allowed): {cluster_best}/{cell_count} cells',
        '',
        'Per-cell primary results:',
    ]
    for r in primary_rows:
        summary.append(
            f"- {DISPLAY[r['prm']]} / {r['benchmark'].upper()}: n_base={r['n_attack_base']}, Flip product={r['Flip_prod']:.3f}, GM={r['Flip_GM']:.3f}, cluster={r['Flip_cluster']:.3f}, official={r['Flip_official']:.3f}; Wrong cluster={r['Wrong_cluster']:.3f}; Gain cluster={r['Gain_cluster']:.3f}; winners: stab={r['Stab_Winner']} util={r['Util_Winner']}"
        )
    summary.extend([
        '',
        'Interpretation:',
        '- This audit uses the same native-top1 deleted-trace injection as the matched-trace product-vs-GM experiment, so all methods are evaluated on the same perturbation.',
        ('- GM already reaches or matches the stability frontier on most cells; clustered answer aggregation does not consistently outperform GM.' if gm_best >= cluster_best else '- Clustered answer aggregation improves over product on many cells and can outperform GM in a subset.'),
        '- Wrong@32 remains low under clustered aggregation on this protocol.',
    ])
    (OUT_DIR / 'summary.txt').write_text('\n'.join(summary) + '\n', encoding='utf-8')
    print(OUT_DIR / 'summary.txt')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
