from __future__ import annotations

from typing import Dict, List

import numpy as np

from src.data.common import InterventionLabel, PRMEvalResult


def attribute_effect_regression(eval_results: List[PRMEvalResult]) -> dict:
    if not eval_results:
        return {
            "coefficients": {},
            "p_values": {},
            "r_squared": 0.0,
            "feature_order": [],
        }

    families = sorted({r.family.value for r in eval_results})
    family_to_idx = {f: i for i, f in enumerate(families)}

    X = []
    y = []
    feature_order = ["intercept", "length_change", "position", "has_math_symbols", "label"] + [f"family_{f}" for f in families]

    for r in eval_results:
        length_change = abs(r.step_length_modified - r.step_length_original) / max(1, r.step_length_original)
        position = r.step_position_normalized
        # Proxy for math symbols from length delta signal.
        has_math_symbols = 1.0 if (r.step_length_original > 0 and abs(r.signed_delta) > 0.01) else 0.0
        label = 1.0 if r.label == InterventionLabel.CAUSAL_IMPORTANT else 0.0

        row = [1.0, length_change, position, has_math_symbols, label]
        fam_oh = [0.0] * len(families)
        fam_oh[family_to_idx[r.family.value]] = 1.0
        row.extend(fam_oh)

        X.append(row)
        y.append(abs(r.delta))

    X_arr = np.array(X, dtype=float)
    y_arr = np.array(y, dtype=float)

    # OLS with pseudo inverse.
    beta = np.linalg.pinv(X_arr.T @ X_arr) @ (X_arr.T @ y_arr)
    y_hat = X_arr @ beta
    ss_res = float(np.sum((y_arr - y_hat) ** 2))
    ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    coefficients = {k: float(v) for k, v in zip(feature_order, beta)}

    # Optional p-values if statsmodels is available.
    pvals: Dict[str, float] = {}
    try:
        import statsmodels.api as sm

        model = sm.OLS(y_arr, X_arr)
        fit = model.fit()
        pvals = {k: float(v) for k, v in zip(feature_order, fit.pvalues)}
    except Exception:
        pvals = {k: float("nan") for k in feature_order}

    return {
        "coefficients": coefficients,
        "p_values": pvals,
        "r_squared": float(r2),
        "feature_order": feature_order,
    }
