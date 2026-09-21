"""Арбитраж между двумя движками на уровне строк.

Правило для цифр отличается от правила для слов и это главное здесь: слово
можно выбрать по уверенности, а цифру — нет. Расхождение движков в токене,
содержащем цифры, означает, что как минимум один из них ошибся, и молча
выбрать «более уверенного» значит с вероятностью около половины записать в
документ неверный БИН или сумму. Поэтому такое расхождение поднимается
наружу как `engine_disagreement`, а окончательное решение остаётся за
доменным слоем (контрольная сумма) или за человеком.
"""

from __future__ import annotations

import re

from rapidfuzz import fuzz

from .model import Line

_DIGIT_RE = re.compile(r"\d")
MIN_VERTICAL_OVERLAP = 0.4
ORPHAN_MIN_CONF = 0.6


def _overlap_ratio(a: Line, b: Line) -> float:
    ay0, ay1 = a.bbox[1], a.bbox[3]
    by0, by1 = b.bbox[1], b.bbox[3]
    overlap = min(ay1, by1) - max(ay0, by0)
    if overlap <= 0:
        return 0.0
    height = min(ay1 - ay0, by1 - by0) or 1
    return overlap / height


def _digit_tokens(text: str) -> list[str]:
    return [t for t in text.split() if _DIGIT_RE.search(t)]


def arbitrate(primary: list[Line], secondary: list[Line]) -> tuple[list[Line], list[str]]:
    """Возвращает согласованные строки и список сообщений о расхождениях.

    Геометрия берётся из primary — он же определяет порядок чтения. Secondary
    влияет только на текст строки и на сигналы о расхождениях.
    """
    disagreements: list[str] = []
    if not secondary:
        return primary, disagreements

    used: set[int] = set()
    result: list[Line] = []

    for line in primary:
        best: Line | None = None
        best_score = 0.0
        best_index = -1
        for i, cand in enumerate(secondary):
            if i in used:
                continue
            if _overlap_ratio(line, cand) < MIN_VERTICAL_OVERLAP:
                continue
            score = fuzz.ratio(line.text, cand.text) / 100.0
            if score > best_score:
                best_score, best, best_index = score, cand, i

        if best is None:
            result.append(line)
            continue

        used.add(best_index)

        if line.text == best.text:
            result.append(line)
            continue

        primary_digits = _digit_tokens(line.text)
        secondary_digits = _digit_tokens(best.text)
        if primary_digits != secondary_digits:
            disagreements.append(
                f"цифры расходятся: {line.engine}={primary_digits} vs {best.engine}={secondary_digits}"
            )
            result.append(line)
            continue

        # Буквенное расхождение — решаем уверенностью.
        result.append(line if line.conf >= best.conf else best)

    for i, cand in enumerate(secondary):
        if i in used or cand.conf < ORPHAN_MIN_CONF:
            continue
        if any(_overlap_ratio(cand, ln) >= MIN_VERTICAL_OVERLAP for ln in primary):
            continue
        result.append(cand)

    return result, disagreements
