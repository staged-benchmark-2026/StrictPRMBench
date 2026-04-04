from __future__ import annotations

import math
from typing import Dict, List

from src.data.common import InterventionLabel, PRMEvalResult


def compute_ssi(eval_results: List[PRMEvalResult]) -> float:
    inert = [r.delta for r in eval_results if r.label == InterventionLabel.INERT]
    if not inert:
        return 0.0
    return float(sum(abs(x) for x in inert) / len(inert))


def compute_ssi_details(eval_results: List[PRMEvalResult]) -> Dict[str, object]:
    inert_rows = [r for r in eval_results if r.label == InterventionLabel.INERT]
    values = [abs(r.delta) for r in inert_rows]
    if not values:
        return {"ssi": 0.0, "stderr": 0.0, "median": 0.0, "by_family": {}}

    mean_val = float(sum(values) / len(values))
    variance = sum((v - mean_val) ** 2 for v in values) / max(1, len(values) - 1)
    stderr = math.sqrt(variance / len(values))
    values_sorted = sorted(values)
    mid = len(values_sorted) // 2
    median = values_sorted[mid] if len(values_sorted) % 2 == 1 else 0.5 * (values_sorted[mid - 1] + values_sorted[mid])

    by_family: Dict[str, float] = {}
    families = sorted({r.family.value for r in inert_rows})
    for f in families:
        fv = [abs(r.delta) for r in inert_rows if r.family.value == f]
        by_family[f] = float(sum(fv) / len(fv)) if fv else 0.0

    return {
        "ssi": mean_val,
        "stderr": float(stderr),
        "median": float(median),
        "by_family": by_family,
    }
