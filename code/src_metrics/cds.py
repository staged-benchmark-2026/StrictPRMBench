from __future__ import annotations

import random
from typing import Dict, List

from src.data.common import InterventionLabel, PRMEvalResult


def _auroc(labels: List[int], scores: List[float]) -> float:
    if len(labels) < 2 or len(set(labels)) < 2:
        return 0.5
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(labels, scores))
    except Exception:
        # Mann-Whitney U equivalent for AUROC.
        pos = [s for s, y in zip(scores, labels) if y == 1]
        neg = [s for s, y in zip(scores, labels) if y == 0]
        if not pos or not neg:
            return 0.5
        wins = 0.0
        total = 0
        for p in pos:
            for n in neg:
                total += 1
                if p > n:
                    wins += 1.0
                elif p == n:
                    wins += 0.5
        return wins / total if total else 0.5


def compute_cds(eval_results: List[PRMEvalResult]) -> float:
    labels = [1 if r.label == InterventionLabel.CAUSAL_IMPORTANT else 0 for r in eval_results]
    deltas = [float(r.delta) for r in eval_results]
    return _auroc(labels, deltas)


def bootstrap_cds_ci(eval_results: List[PRMEvalResult], n_bootstrap: int = 1000, seed: int = 42) -> Dict[str, float]:
    if not eval_results:
        return {"mean": 0.5, "low": 0.5, "high": 0.5}

    rnd = random.Random(seed)
    values = []
    n = len(eval_results)
    for _ in range(n_bootstrap):
        sample = [eval_results[rnd.randrange(n)] for _ in range(n)]
        values.append(compute_cds(sample))

    values_sorted = sorted(values)
    low_idx = int(0.025 * (len(values_sorted) - 1))
    high_idx = int(0.975 * (len(values_sorted) - 1))
    return {
        "mean": float(sum(values) / len(values)),
        "low": float(values_sorted[low_idx]),
        "high": float(values_sorted[high_idx]),
    }


def compute_cds_by_family(eval_results: List[PRMEvalResult]) -> Dict[str, float]:
    families = sorted({r.family.value for r in eval_results})
    out: Dict[str, float] = {}
    for family in families:
        subset = [r for r in eval_results if r.family.value == family]
        out[family] = compute_cds(subset)
    return out


def compute_cds_by_position(eval_results: List[PRMEvalResult], bins: int = 5) -> Dict[str, float]:
    if bins <= 0:
        bins = 5

    out: Dict[str, float] = {}
    for i in range(bins):
        lo = i / bins
        hi = (i + 1) / bins
        subset = [r for r in eval_results if lo <= r.step_position_normalized < hi or (i == bins - 1 and r.step_position_normalized <= hi)]
        key = f"{lo:.1f}-{hi:.1f}"
        out[key] = compute_cds(subset)
    return out
