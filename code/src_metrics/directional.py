from __future__ import annotations

from typing import Dict, List

from src.data.common import InterventionLabel, PRMEvalResult


def compute_dir_acc(eval_results: List[PRMEvalResult]) -> float:
    causal = [r for r in eval_results if r.label == InterventionLabel.CAUSAL_IMPORTANT]
    if not causal:
        return 0.0
    correct = sum(1 for r in causal if r.signed_delta > 0)
    return correct / len(causal)


def compute_dir_acc_by_family(eval_results: List[PRMEvalResult]) -> Dict[str, float]:
    families = sorted({r.family.value for r in eval_results if r.label == InterventionLabel.CAUSAL_IMPORTANT})
    out: Dict[str, float] = {}
    for family in families:
        subset = [r for r in eval_results if r.family.value == family and r.label == InterventionLabel.CAUSAL_IMPORTANT]
        out[family] = compute_dir_acc(subset)
    return out
