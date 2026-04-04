from __future__ import annotations

import re
from typing import Optional

from src.utils.numeric_equiv import numeric_equivalent


_BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}")
_FINAL_ANSWER_PATTERN = re.compile(
    r"(?:final\s*answer\s*[:：]|the\s*answer\s*is\s*[:：])\s*(.+)",
    flags=re.IGNORECASE,
)
_NUMBER_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:/\d+)?")


def _extract_last_boxed(text: str) -> Optional[str]:
    matches = _BOXED_PATTERN.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def extract_answer(text: str) -> Optional[str]:
    """
    Extract final answer from model output.

    Priority:
    1) boxed after </think>
    2) boxed anywhere
    3) explicit final-answer pattern
    4) last number fallback
    """
    post_think = ""
    if "</think>" in text:
        post_think = text.split("</think>", 1)[1]
        boxed_post = _extract_last_boxed(post_think)
        if boxed_post:
            return boxed_post

    boxed_anywhere = _extract_last_boxed(text)
    if boxed_anywhere:
        return boxed_anywhere

    search_scope = post_think if post_think else text
    fa_match = None
    for line in reversed(search_scope.splitlines()):
        m = _FINAL_ANSWER_PATTERN.search(line)
        if m:
            fa_match = m.group(1).strip()
            break
    if fa_match:
        number_in_answer = _NUMBER_PATTERN.findall(fa_match)
        if number_in_answer:
            return number_in_answer[-1]
        return fa_match

    nums = _NUMBER_PATTERN.findall(search_scope)
    if nums:
        return nums[-1]
    return None


def check_answer(extracted: str, ground_truth: str) -> bool:
    return numeric_equivalent(extracted, ground_truth)
