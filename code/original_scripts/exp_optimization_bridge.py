#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.common import Problem, Step, Trace
from src.models.prm_wrapper import PRMWrapper
from src.utils.answer_extraction import check_answer, extract_answer
from src.utils.numeric_equiv import normalize_numeric_text, parse_number

CLIP_EPS = 1e-30
BOOTSTRAPS = 1000
PRMS = ['skywork_prm', 'qwen_prm']
BENCHMARKS = ['gsm8k', 'math500']
DISPLAY_PRM = {
    'skywork_prm': 'Skywork-7B',
    'qwen_prm': 'Qwen',
}
DISPLAY_BENCH = {'gsm8k': 'GSM8K', 'math500': 'MATH-500'}
OFFICIAL_MAP = {
    'qwen_prm': 'product',
    'skywork_prm': 'official_mean',
}
RULES = ['product', 'gm', 'cluster_sum_gm', 'official']
OPT_DIRECTIONS = ['product', 'gm']


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Optimization bridge experiment')
    p.add_argument('--output_dir', default='outputs/reports/optimization_bridge')
    p.add_argument('--max_gsm8k', type=int, default=200)
    p.add_argument('--max_math500', type=int, default=200)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def choose_problem_ids(candidate_dir: Path, limit: int, seed: int) -> list[str]:
    ids = sorted(p.stem for p in candidate_dir.glob('*.json'))
    if len(ids) <= limit:
        return ids
    rng = random.Random(seed)
    return sorted(rng.sample(ids, limit))


def default_pool_dir(benchmark: str) -> Path:
    if benchmark == 'gsm8k':
        p = REPO_ROOT / 'outputs/openclaw_run/bon_candidates/gsm8k_full_500'
        return p if p.exists() else REPO_ROOT / 'outputs/openclaw_run/bon_candidates/gsm8k'
    p = REPO_ROOT / 'outputs/openclaw_run/bon_candidates/math500_full_500'
    return p if p.exists() else REPO_ROOT / 'outputs/openclaw_run/bon_candidates/math500_full_200'


def load_step_cache(path: Path, wanted_ids: set[str]) -> dict[str, dict[int, dict[str, Any]]]:
    out: dict[str, dict[int, dict[str, Any]]] = {}
    for row in load_jsonl(path):
        pid = str(row['problem_id'])
        if pid not in wanted_ids:
            continue
        out.setdefault(pid, {})[int(row['candidate_index'])] = row
    return out


def agg_value(step_scores: list[float], rule: str) -> float:
    vals = [max(CLIP_EPS, float(s)) for s in step_scores]
    if rule == 'product':
        out = 1.0
        for v in vals:
            out *= v
        return float(out)
    if rule == 'gm':
        return float(math.exp(sum(math.log(v) for v in vals) / len(vals)))
    raise KeyError(rule)


def product_log(step_scores: list[float]) -> float:
    return float(sum(math.log(max(CLIP_EPS, float(x))) for x in step_scores))


def gm_score(step_scores: list[float]) -> float:
    vals = [max(CLIP_EPS, float(x)) for x in step_scores]
    return float(math.exp(sum(math.log(v) for v in vals) / len(vals)))


def mean_score(step_scores: list[float]) -> float:
    return float(sum(step_scores) / len(step_scores))


def last_score(step_scores: list[float]) -> float:
    return float(step_scores[-1])


def build_problem(problem_blob: dict[str, Any]) -> Problem:
    return Problem(
        id=str(problem_blob['id']),
        dataset=str(problem_blob['dataset']),
        question=str(problem_blob['question']),
        ground_truth=str(problem_blob['ground_truth']),
        domain=problem_blob.get('domain'),
        difficulty=None,
    )


def build_trace(problem: Problem, step_texts: list[str], problem_id: str, raw_text: str | None = None) -> Trace:
    steps = [Step(index=i, text=text) for i, text in enumerate(step_texts)]
    raw = raw_text if raw_text is not None else render_step_texts(step_texts)
    return Trace(
        problem_id=problem_id,
        problem=problem,
        raw_text=raw,
        steps=steps,
        extracted_answer=extract_answer(raw),
        is_correct=None,
        num_tokens=None,
        model_name='QwQ-32B',
    )


def render_step_texts(step_texts: list[str]) -> str:
    return "\n\n".join(step_texts)


def answer_preserved(step_texts: list[str], original_answer: str | None, ground_truth: str) -> bool:
    raw = render_step_texts(step_texts)
    extracted = extract_answer(raw)
    if extracted is None:
        return False
    if original_answer:
        try:
            if check_answer(extracted, original_answer):
                return True
        except Exception:
            if extracted.strip() == str(original_answer).strip():
                return True
    try:
        return bool(check_answer(extracted, ground_truth))
    except Exception:
        return extracted.strip() == str(ground_truth).strip()


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


def answer_from_candidate(cand: dict[str, Any]) -> str:
    ans = cand.get('extracted_answer')
    if ans is not None and str(ans).strip():
        return str(ans).strip()
    raw = str(cand.get('raw_text', ''))
    extracted = extract_answer(raw)
    return '' if extracted is None else str(extracted).strip()


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
        if rule == 'last':
            return last_score(step_scores)
        raise KeyError(rule)
    raise KeyError(method)


def select_flat(cands: list[dict[str, Any]], prm: str, method: str) -> dict[str, Any]:
    return max(cands, key=lambda c: (base_method_score(c['step_scores'], prm, method), -c['idx']))


def select_cluster(cands: list[dict[str, Any]]) -> dict[str, Any]:
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cand in cands:
        clusters[cand['answer_key']].append(cand)
    scored_clusters = []
    for answer_key, members in clusters.items():
        gm_vals = [gm_score(m['step_scores']) for m in members]
        gm_sum = float(sum(gm_vals))
        gm_best = float(max(gm_vals))
        cluster_score = gm_sum
        scored_clusters.append((answer_key, members, cluster_score, len(members), gm_sum, gm_best))
    _, members, _, _, _, _ = max(scored_clusters, key=lambda x: (x[2], x[3], x[4], x[5], x[0]))
    return max(members, key=lambda c: (gm_score(c['step_scores']), -c['idx']))


def select_method(cands: list[dict[str, Any]], prm: str, method: str) -> dict[str, Any]:
    if method in {'product', 'gm', 'official'}:
        return select_flat(cands, prm, method)
    if method == 'cluster_sum_gm':
        return select_cluster(cands)
    raise KeyError(method)


def prepare_clean_pool(payload: dict[str, Any], pid: str, step_cache: dict[str, dict[int, dict[str, Any]]]) -> list[dict[str, Any]] | None:
    cands = payload['candidates'][:32]
    if len(cands) < 32 or pid not in step_cache:
        return None
    out = []
    for idx, cand in enumerate(cands):
        row = step_cache[pid].get(idx)
        if row is None:
            return None
        scores = [float(x) for x in row['step_scores']]
        ans = answer_from_candidate(cand)
        out.append({
            'identity': f'orig:{idx}',
            'idx': idx,
            'step_scores': scores,
            'answer_text': ans,
            'answer_key': canonical_answer(ans),
            'is_correct': bool(cand.get('is_correct', False)),
        })
    return out


def greedy_optimize_with_trace(
    wrapper: PRMWrapper,
    problem: Problem,
    problem_id: str,
    original_answer: str | None,
    init_step_texts: list[str],
    init_scores: list[float],
    rule: str,
    memo: dict[tuple[str, ...], list[float]],
) -> dict[str, Any]:
    current_steps = list(init_step_texts)
    current_scores = [float(x) for x in init_scores]
    initial_score = agg_value(current_scores, rule)
    current_score = initial_score
    history: list[dict[str, Any]] = []
    trajectory = [{'deletions': 0, 'score': current_score, 'n_steps': len(current_steps)}]
    memo.setdefault(tuple(current_steps), current_scores)

    while len(current_steps) > 2:
        best = None
        best_score = current_score
        for k in range(len(current_steps)):
            cand_steps = current_steps[:k] + current_steps[k + 1:]
            if not answer_preserved(cand_steps, original_answer, problem.ground_truth):
                continue
            key = tuple(cand_steps)
            if key not in memo:
                trace = build_trace(problem, cand_steps, problem_id, raw_text=render_step_texts(cand_steps))
                scored = wrapper.score_trace(problem, trace)
                memo[key] = [float(x) for x in scored.step_scores]
            new_scores = memo[key]
            new_score = agg_value(new_scores, rule)
            if new_score > best_score + 1e-12:
                best_score = new_score
                best = (k, cand_steps, new_scores)

        if best is None:
            break

        deleted_idx, next_steps, next_scores = best
        history.append({
            'deleted_index': int(deleted_idx),
            'n_steps_before': len(current_steps),
            'score_before': current_score,
            'score_after': best_score,
        })
        current_steps = list(next_steps)
        current_scores = [float(x) for x in next_scores]
        current_score = best_score
        trajectory.append({'deletions': len(history), 'score': current_score, 'n_steps': len(current_steps)})

    final_raw = render_step_texts(current_steps)
    final_answer = extract_answer(final_raw)
    preserved = answer_preserved(current_steps, original_answer, problem.ground_truth)
    return {
        'original_n_steps': len(init_step_texts),
        'final_n_steps': len(current_steps),
        'n_deletions': len(history),
        'score_increase': float(current_score - initial_score),
        'deletion_history': history,
        'trajectory': trajectory,
        'final_steps': current_steps,
        'final_scores': current_scores,
        'final_raw_text': final_raw,
        'final_answer': final_answer,
        'answer_preserved': bool(preserved),
    }


def bootstrap_diff(a_vals: list[int], b_vals: list[int], seed: int) -> tuple[float, float, float]:
    n = len(a_vals)
    diffs = [float(a_vals[i] - b_vals[i]) for i in range(n)]
    point = float(sum(diffs) / n) if n else float('nan')
    rng = random.Random(seed)
    boots = []
    for _ in range(BOOTSTRAPS):
        total = 0.0
        for _ in range(n):
            total += diffs[rng.randrange(n)]
        boots.append(total / n)
    boots.sort()
    return point, float(boots[int(0.025 * len(boots))]), float(boots[int(0.975 * len(boots))])


def main() -> int:
    args = parse_args()
    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    with (REPO_ROOT / 'configs/models.yaml').open('r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    prms_cfg = cfg['prms']

    problem_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for benchmark in BENCHMARKS:
        candidate_dir = default_pool_dir(benchmark)
        limit = args.max_gsm8k if benchmark == 'gsm8k' else args.max_math500
        problem_ids = choose_problem_ids(candidate_dir, limit, args.seed)
        wanted = set(problem_ids)
        caches = {
            prm: load_step_cache(REPO_ROOT / f'outputs/reports/aggregation_sweep/step_scores_cache/{prm}_{benchmark}.jsonl', wanted)
            for prm in PRMS
        }

        for prm in PRMS:
            wrapper = PRMWrapper(prms_cfg[prm])
            wrapper.config['mock'] = False
            wrapper.config['mock_on_error'] = False
            wrapper.config['local_files_only'] = True
            wrapper.load()

            cell_rows: list[dict[str, Any]] = []
            print(f'[cell-start] {prm} {benchmark}')
            for j, pid in enumerate(problem_ids, start=1):
                payload_path = candidate_dir / f'{pid}.json'
                if not payload_path.exists():
                    continue
                payload = load_json(payload_path)
                clean_pool = prepare_clean_pool(payload, pid, caches[prm])
                if clean_pool is None:
                    continue

                product_top1 = select_method(clean_pool, prm, 'product')
                if not bool(product_top1['is_correct']):
                    continue
                start_idx = int(product_top1['idx'])
                cand = payload['candidates'][start_idx]
                step_texts = [str(s.get('text', '')) for s in cand.get('steps', [])]
                if len(step_texts) <= 2:
                    continue
                problem = build_problem(cand['problem'])
                original_answer = cand.get('extracted_answer') or extract_answer(str(cand.get('raw_text', '')))
                init_scores = list(product_top1['step_scores'])
                memo: dict[tuple[str, ...], list[float]] = {tuple(step_texts): init_scores}

                baseline_selected = {rule: select_method(clean_pool, prm, rule)['identity'] for rule in RULES}

                opt_results = {
                    rule: greedy_optimize_with_trace(wrapper, problem, pid, original_answer, step_texts, init_scores, rule, memo)
                    for rule in OPT_DIRECTIONS
                }

                for optimize_for, opt in opt_results.items():
                    if not opt['answer_preserved']:
                        continue
                    injected_pool = [dict(x) for x in clean_pool]
                    injected_pool[start_idx] = {
                        'identity': f'opt:{optimize_for}',
                        'idx': start_idx,
                        'step_scores': list(opt['final_scores']),
                        'answer_text': '' if opt['final_answer'] is None else str(opt['final_answer']),
                        'answer_key': canonical_answer(opt['final_answer']),
                        'is_correct': True,
                    }
                    selected = {rule: select_method(injected_pool, prm, rule)['identity'] for rule in RULES}
                    row = {
                        'prm': prm,
                        'benchmark': benchmark,
                        'problem_id': pid,
                        'optimize_for': optimize_for,
                        'start_candidate_index': start_idx,
                        'original_n_steps': len(step_texts),
                        'final_n_steps': int(opt['final_n_steps']),
                        'n_deletions': int(opt['n_deletions']),
                        'score_gain': float(opt['score_increase']),
                        'answer_preserved': int(bool(opt['answer_preserved'])),
                        'final_answer': '' if opt['final_answer'] is None else str(opt['final_answer']),
                        'baseline_product_identity': baseline_selected['product'],
                        'baseline_gm_identity': baseline_selected['gm'],
                        'baseline_cluster_identity': baseline_selected['cluster_sum_gm'],
                        'baseline_official_identity': baseline_selected['official'],
                        'selected_product_identity': selected['product'],
                        'selected_gm_identity': selected['gm'],
                        'selected_cluster_identity': selected['cluster_sum_gm'],
                        'selected_official_identity': selected['official'],
                        'hijack_product': int(selected['product'] == f'opt:{optimize_for}'),
                        'hijack_gm': int(selected['gm'] == f'opt:{optimize_for}'),
                        'hijack_cluster': int(selected['cluster_sum_gm'] == f'opt:{optimize_for}'),
                        'hijack_official': int(selected['official'] == f'opt:{optimize_for}'),
                        'optimized_steps_json': json.dumps(opt['final_steps'], ensure_ascii=False),
                        'optimized_scores_json': json.dumps(opt['final_scores']),
                        'optimized_raw_text': opt['final_raw_text'],
                        'deletion_history_json': json.dumps(opt['deletion_history']),
                        'trajectory_json': json.dumps(opt['trajectory']),
                    }
                    problem_rows.append(row)
                    cell_rows.append(row)

                if j % 10 == 0:
                    prod_done = sum(1 for r in cell_rows if r['optimize_for'] == 'product')
                    gm_done = sum(1 for r in cell_rows if r['optimize_for'] == 'gm')
                    print(f'[progress] {prm} {benchmark} sampled={j}/{len(problem_ids)} rows_product={prod_done} rows_gm={gm_done}')

            wrapper.unload()

            for optimize_for in OPT_DIRECTIONS:
                subset = [r for r in cell_rows if r['optimize_for'] == optimize_for]
                n_start = len(subset)
                nontrivial = [r for r in subset if int(r['n_deletions']) > 0]
                if not nontrivial:
                    summary_rows.append({
                        'prm': prm,
                        'benchmark': benchmark,
                        'optimize_for': optimize_for,
                        'n_start': n_start,
                        'n_nontrivial': 0,
                        'mean_n_deletions': 0.0,
                        'mean_score_gain': 0.0,
                        'hijack_product': 0.0,
                        'hijack_gm': 0.0,
                        'hijack_cluster': 0.0,
                        'hijack_official': 0.0,
                        'delta_product_minus_gm': 0.0,
                        'delta_ci_lo': 0.0,
                        'delta_ci_hi': 0.0,
                    })
                    continue
                prod_vals = [int(r['hijack_product']) for r in nontrivial]
                gm_vals = [int(r['hijack_gm']) for r in nontrivial]
                delta, ci_lo, ci_hi = bootstrap_diff(prod_vals, gm_vals, args.seed + hash((prm, benchmark, optimize_for)) % 10000)
                summary_rows.append({
                    'prm': prm,
                    'benchmark': benchmark,
                    'optimize_for': optimize_for,
                    'n_start': n_start,
                    'n_nontrivial': len(nontrivial),
                    'mean_n_deletions': float(mean([int(r['n_deletions']) for r in nontrivial])),
                    'mean_score_gain': float(mean([float(r['score_gain']) for r in nontrivial])),
                    'hijack_product': float(mean(prod_vals)),
                    'hijack_gm': float(mean(gm_vals)),
                    'hijack_cluster': float(mean([int(r['hijack_cluster']) for r in nontrivial])),
                    'hijack_official': float(mean([int(r['hijack_official']) for r in nontrivial])),
                    'delta_product_minus_gm': float(delta),
                    'delta_ci_lo': float(ci_lo),
                    'delta_ci_hi': float(ci_hi),
                })

            print(f'[cell-done] {prm} {benchmark}')

    if not problem_rows:
        raise SystemExit('No rows produced')

    problem_rows.sort(key=lambda r: (r['benchmark'], r['prm'], r['optimize_for'], r['problem_id']))
    summary_rows.sort(key=lambda r: (r['benchmark'], r['prm'], r['optimize_for']))

    problem_path = out_dir / 'optimized_traces.csv'
    with problem_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(problem_rows[0].keys()))
        w.writeheader()
        w.writerows(problem_rows)

    summary_path = out_dir / 'hijack_results.csv'
    with summary_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    go_cells = 0
    lines = [
        'Optimization bridge (minimum configuration)',
        '',
        'Start trace: clean product top-1, restricted to correct traces.',
        'Injected rules: product, GM, cluster_sum_gm, official.',
        'Optimizers: product and GM greedy answer-preserving deletion.',
        '',
    ]
    for r in summary_rows:
        if float(r['hijack_product']) > 0.20 and float(r['hijack_product']) > float(r['hijack_gm']):
            go_cells += 1
        lines.append(
            f"{DISPLAY_PRM[r['prm']]} / {DISPLAY_BENCH[r['benchmark']]} / optimize={r['optimize_for']}: "
            f"n_start={int(r['n_start'])}, n_nontrivial={int(r['n_nontrivial'])}, "
            f"mean_del={float(r['mean_n_deletions']):.2f}, mean_gain={float(r['mean_score_gain']):+.4f}, "
            f"hijack product={float(r['hijack_product']):.3f}, gm={float(r['hijack_gm']):.3f}, "
            f"cluster={float(r['hijack_cluster']):.3f}, official={float(r['hijack_official']):.3f}, "
            f"Δ(product-gm)={float(r['delta_product_minus_gm']):+.3f} "
            f"[{float(r['delta_ci_lo']):+.3f}, {float(r['delta_ci_hi']):+.3f}]"
        )
    lines += [
        '',
        f'Cells meeting provisional GO threshold (product hijack > 20% and > GM): {go_cells}/{len(summary_rows)}',
    ]
    (out_dir / 'summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    print(f'[saved] {problem_path}')
    print(f'[saved] {summary_path}')
    print(f'[saved] {out_dir / "summary.txt"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
