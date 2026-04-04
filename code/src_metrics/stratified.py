from __future__ import annotations

from typing import Dict, List

from src.data.common import PRMEvalResult
from src.metrics.cds import compute_cds
from src.metrics.ssi import compute_ssi


def _quantile_bins(values: List[float]) -> Dict[str, tuple[float, float]]:
    if not values:
        return {"low": (0.0, 0.0), "mid": (0.0, 0.0), "high": (0.0, 0.0)}
    sv = sorted(values)
    q1 = sv[len(sv) // 3]
    q2 = sv[(2 * len(sv)) // 3]
    return {
        "low": (float("-inf"), q1),
        "mid": (q1, q2),
        "high": (q2, float("inf")),
    }


def stratified_by_position(eval_results: List[PRMEvalResult]) -> Dict[str, Dict[str, float]]:
    bins = {
        "early": (0.0, 1 / 3),
        "middle": (1 / 3, 2 / 3),
        "late": (2 / 3, 1.0 + 1e-9),
    }
    out: Dict[str, Dict[str, float]] = {}
    for name, (lo, hi) in bins.items():
        subset = [r for r in eval_results if lo <= r.step_position_normalized < hi]
        out[name] = {"cds": compute_cds(subset), "ssi": compute_ssi(subset)}
    return out


def stratified_by_step_length(eval_results: List[PRMEvalResult]) -> Dict[str, Dict[str, float]]:
    lengths = [r.step_length_original for r in eval_results]
    bins = _quantile_bins(lengths)
    out: Dict[str, Dict[str, float]] = {}
    for name, (lo, hi) in bins.items():
        subset = [r for r in eval_results if lo <= r.step_length_original < hi]
        out[name] = {"cds": compute_cds(subset), "ssi": compute_ssi(subset)}
    return out


def stratified_by_difficulty(eval_results: List[PRMEvalResult]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for diff in ["easy", "medium", "hard", "unknown"]:
        subset = [r for r in eval_results if (r.problem_difficulty or "unknown") == diff]
        out[diff] = {"cds": compute_cds(subset), "ssi": compute_ssi(subset)}
    return out
