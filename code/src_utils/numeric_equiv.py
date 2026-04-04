from __future__ import annotations

import re
from fractions import Fraction
from typing import Optional


_NUMBER_CLEAN_PATTERNS = [
    re.compile(r"\\text\{([^}]*)\}"),
    re.compile(r"\$"),
    re.compile(r",(?=\d{3}(\D|$))"),
]


def _strip_latex_frac(value: str) -> str:
    value = value.strip()
    m = re.fullmatch(r"\\frac\{\s*([-+]?\d+)\s*\}\{\s*([-+]?\d+)\s*\}", value)
    if m:
        num, den = m.group(1), m.group(2)
        if den == "0":
            return value
        return str(Fraction(int(num), int(den)))
    return value


def normalize_numeric_text(value: str) -> str:
    out = value.strip()
    for pattern in _NUMBER_CLEAN_PATTERNS:
        out = pattern.sub(r"\1" if "text" in pattern.pattern else "", out)
    out = out.strip()
    out = _strip_latex_frac(out)
    return out


def parse_number(value: str) -> Optional[float]:
    v = normalize_numeric_text(value)
    if not v:
        return None
    # Convert common fraction forms.
    if re.fullmatch(r"[-+]?\d+\s*/\s*[-+]?\d+", v):
        num, den = [x.strip() for x in v.split("/", 1)]
        if den == "0":
            return None
        return float(Fraction(int(num), int(den)))
    try:
        return float(v)
    except ValueError:
        pass

    # Last fallback: pick last numeric token.
    nums = re.findall(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", v, flags=re.IGNORECASE)
    if nums:
        try:
            return float(nums[-1])
        except ValueError:
            return None
    return None


def numeric_equivalent(a: str, b: str, atol: float = 1e-6) -> bool:
    pa = parse_number(a)
    pb = parse_number(b)
    if pa is None or pb is None:
        return normalize_numeric_text(a) == normalize_numeric_text(b)
    return abs(pa - pb) <= atol
