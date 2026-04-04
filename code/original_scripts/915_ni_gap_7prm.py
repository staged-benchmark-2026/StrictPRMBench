#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SCORES_DIR = REPO_ROOT / 'outputs/matched_control/scores'
OUT_DIR = REPO_ROOT / 'outputs/reports/ni_gap'
OUT_CSV = OUT_DIR / 'ni_gap_7prm_product.csv'
OUT_TXT = OUT_DIR / 'ni_gap_7prm_product_summary.txt'
REFERENCE_CSV = OUT_DIR / 'ni_gap_product.csv'
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
AGGS = ['product', 'gm']
PRETTY = {
    'skywork_prm': 'Skywork-7B',
    'internlm2_7b_reward': 'InternLM2-7B',
    'qwen_prm': 'Qwen',
    'math_shepherd': 'Math-Shepherd',
    'skywork_prm_1_5b': 'Skywork-1.5B',
    'rlhflow_prm': 'RLHFlow',
    'internlm2_1_8b_reward': 'InternLM2-1.8B',
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float('nan')


def percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float('nan')
    idx = int(q * (len(sorted_vals) - 1))
    return float(sorted_vals[idx])


def safe_log(x: float) -> float:
    return math.log(max(float(x), CLIP_EPS))


def agg_score(step_scores: list[float], aggregation: str) -> float:
    xs = [float(x) for x in step_scores]
    if aggregation == 'product':
        return float(sum(safe_log(x) for x in xs))
    if aggregation == 'gm':
        return float(math.exp(sum(safe_log(x) for x in xs) / len(xs)))
    raise KeyError(aggregation)


def apr_bits(rows: list[dict[str, Any]], aggregation: str) -> list[int]:
    bits: list[int] = []
    for r in rows:
        clean = agg_score(r['clean_step_scores'], aggregation)
        modified = agg_score(r['modified_step_scores'], aggregation)
        bits.append(1 if modified > clean else 0)
    return bits


def bootstrap_delta(nec: list[int], inert: list[int], seed_text: str, iters: int = BOOT_ITERS) -> tuple[float, float, float]:
    point = mean([float(x) for x in nec]) - mean([float(x) for x in inert])
    rng = random.Random(seed_text)
    boots: list[float] = []
    for _ in range(iters):
        nec_s = [nec[rng.randrange(len(nec))] for _ in range(len(nec))]
        inert_s = [inert[rng.randrange(len(inert))] for _ in range(len(inert))]
        boots.append(mean([float(x) for x in nec_s]) - mean([float(x) for x in inert_s]))
    boots.sort()
    return point, percentile(boots, 0.025), percentile(boots, 0.975)


def load_reference() -> dict[tuple[str, str], dict[str, str]]:
    out: dict[tuple[str, str], dict[str, str]] = {}
    if not REFERENCE_CSV.exists():
        return out
    with REFERENCE_CSV.open('r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            out[(row['prm'], row['benchmark'])] = row
    return out


def fmt(x: float) -> str:
    return f'{x:.6f}'


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reference = load_reference()
    rows_out: list[dict[str, Any]] = []
    mismatches: list[str] = []

    for prm in PRMS:
        for benchmark in BENCHMARKS:
            nec_path = SCORES_DIR / f'{prm}_{benchmark}_necessary.jsonl'
            inert_path = SCORES_DIR / f'{prm}_{benchmark}_inert.jsonl'
            if not nec_path.exists() or not inert_path.exists():
                raise FileNotFoundError(f'missing score files for {prm}/{benchmark}: {nec_path} {inert_path}')
            nec_rows = load_jsonl(nec_path)
            inert_rows = load_jsonl(inert_path)
            for aggregation in AGGS:
                nec_bits = apr_bits(nec_rows, aggregation)
                inert_bits = apr_bits(inert_rows, aggregation)
                apr_nec = mean([float(x) for x in nec_bits])
                apr_inert = mean([float(x) for x in inert_bits])
                delta, lo, hi = bootstrap_delta(nec_bits, inert_bits, f'{SEED}:{prm}:{benchmark}:{aggregation}')
                rows_out.append({
                    'prm': prm,
                    'benchmark': benchmark,
                    'aggregation': aggregation,
                    'apr_nec': fmt(apr_nec),
                    'apr_inert': fmt(apr_inert),
                    'delta_ni': fmt(delta),
                    'ci_lo': fmt(lo),
                    'ci_hi': fmt(hi),
                    'n_nec': len(nec_bits),
                    'n_inert': len(inert_bits),
                })
                if aggregation == 'product' and (prm, benchmark) in reference:
                    ref = reference[(prm, benchmark)]
                    triple_new = (fmt(apr_nec), fmt(apr_inert), fmt(delta))
                    triple_ref = (ref['apr_necessary'], ref['apr_inert'], ref['delta_ni'])
                    if triple_new != triple_ref:
                        mismatches.append(f'{prm}/{benchmark}: new={triple_new} ref={triple_ref}')

    rows_out.sort(key=lambda r: (r['benchmark'], r['prm'], AGGS.index(r['aggregation'])))
    with OUT_CSV.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['prm', 'benchmark', 'aggregation', 'apr_nec', 'apr_inert', 'delta_ni', 'ci_lo', 'ci_hi', 'n_nec', 'n_inert'])
        writer.writeheader()
        writer.writerows(rows_out)

    product_rows = [r for r in rows_out if r['aggregation'] == 'product']
    gm_rows = [r for r in rows_out if r['aggregation'] == 'gm']
    product_nonpos = sum(1 for r in product_rows if float(r['delta_ni']) <= 0)
    gm_nonpos = sum(1 for r in gm_rows if float(r['delta_ni']) <= 0)

    lines = [
        'ΔNI on 7 PRMs (product + GM)',
        '',
        'Product rows',
        '',
        'PRM              Bench      ΔNI      95% CI',
    ]
    for r in product_rows:
        lines.append(f"{PRETTY[r['prm']]:16s} {r['benchmark']:8s} {float(r['delta_ni']):+0.3f}   [{float(r['ci_lo']):+0.3f}, {float(r['ci_hi']):+0.3f}]")
    lines.extend([
        '',
        f'Product non-positive: {product_nonpos}/{len(product_rows)}',
        f'GM non-positive: {gm_nonpos}/{len(gm_rows)}',
    ])
    if mismatches:
        lines.extend(['', 'Product mismatches vs 911 reference:'] + mismatches)
    else:
        lines.extend(['', 'Product rows for the original 3 PRMs match 911 exactly.'])
    OUT_TXT.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    print(f'[saved] {OUT_CSV}')
    print(f'[saved] {OUT_TXT}')
    print(f'[check] product_non_positive={product_nonpos}/{len(product_rows)}')
    print(f'[check] gm_non_positive={gm_nonpos}/{len(gm_rows)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
