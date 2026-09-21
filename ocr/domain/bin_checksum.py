"""Контрольный разряд и repair БИН/ИИН РК (12 цифр).

Алгоритм контрольной цифры: для ``a1..a11``
``S = (Σ i*a_i, i=1..11) mod 11``. Если ``S != 10`` — контрольная цифра ``S``.
Если ``S == 10`` — вторая серия весов ``(3,4,5,6,7,8,9,10,11,1,2)``:
``S2 = (Σ w_i*a_i) mod 11``; ``S2 == 10`` — номер невалиден, иначе цифра ``S2``.

Известное слепое пятно: вес 11-й позиции в первой серии равен 11 ≡ 0 (mod 11),
поэтому ошибка ровно в 11-й цифре не детектируется первой серией.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Literal

# Веса второй серии для позиций 1..11.
_WEIGHTS2 = (3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2)

# Типичные OCR-конфузии цифр в обе стороны:
# 0↔8, 0↔6, 0↔9, 1↔7, 3↔8, 5↔6, 6↔8, 9↔4, 2↔7, 5↔8.
CONFUSIONS: dict[str, tuple[str, ...]] = {
    "0": ("8", "6", "9"),
    "1": ("7",),
    "2": ("7",),
    "3": ("8",),
    "4": ("9",),
    "5": ("6", "8"),
    "6": ("0", "5", "8"),
    "7": ("1", "2"),
    "8": ("0", "3", "6", "5"),
    "9": ("0", "4"),
}


def control_digit(first11: str) -> int | None:
    """Контрольная цифра для первых 11 разрядов; ``None`` — номер невалиден."""
    if len(first11) != 11 or not first11.isdigit():
        return None
    digits = [int(c) for c in first11]
    s = sum((i + 1) * d for i, d in enumerate(digits)) % 11
    if s != 10:
        return s
    s2 = sum(w * d for w, d in zip(_WEIGHTS2, digits)) % 11
    if s2 == 10:
        return None
    return s2


def is_valid(number: str) -> bool:
    """12 цифр, и 12-я совпадает с вычисленной контрольной."""
    if len(number) != 12 or not number.isdigit():
        return False
    cd = control_digit(number[:11])
    return cd is not None and cd == int(number[11])


def looks_like_bin(number: str) -> bool:
    """Структура БИН: разряды 1-4 = YYMM (месяц 01..12), 5-й ∈ {4,5,6}, 6-й ∈ {0,1,2,3}."""
    if len(number) != 12 or not number.isdigit():
        return False
    month = int(number[2:4])
    if not 1 <= month <= 12:
        return False
    return number[4] in "456" and number[5] in "0123"


def looks_like_iin(number: str) -> bool:
    """Структура ИИН: разряды 1-6 = YYMMDD с валидной датой, 7-й ∈ {1..6}.

    7-й разряд кодирует пол и век рождения: 1,2 → XIX век, 3,4 → XX, 5,6 → XXI.
    """
    if len(number) != 12 or not number.isdigit():
        return False
    seventh = int(number[6])
    if seventh not in (1, 2, 3, 4, 5, 6):
        return False
    year = 1800 + ((seventh - 1) // 2) * 100 + int(number[0:2])
    try:
        datetime.date(year, int(number[2:4]), int(number[4:6]))
    except ValueError:
        return False
    return True

def looks_structural(number: str) -> bool:
    """Дополнительный фильтр кандидатов: похоже на БИН или ИИН."""
    return looks_like_bin(number) or looks_like_iin(number)


@dataclass(slots=True)
class RepairResult:
    """Результат попытки восстановления БИН/ИИН."""

    status: Literal["valid", "fixed", "ambiguous", "invalid"]
    value: str  # для fixed — исправленный номер, иначе исходный
    candidates: list[str] = field(default_factory=list)
    detail: str = ""


def repair(number: str, char_confs: list[float] | None = None) -> RepairResult:
    """Попытка исправить номер одиночной OCR-конфузией цифры.

    Правило владельца: исправлять только если валидный кандидат ровно один;
    при нескольких — не исправлять (``ambiguous``). ``char_confs`` (длина 12,
    0..1) влияет только на порядок ``candidates``: первыми идут замены
    в позициях с наименьшей уверенностью.
    """
    if len(number) != 12 or not number.isdigit():
        return RepairResult("invalid", number, [], f"{number!r}: не 12 цифр")
    if is_valid(number):
        return RepairResult("valid", number, [number], "номер валиден")

    # Все одиночные замены по таблице конфузий, проходящие checksum.
    found: dict[str, int] = {}  # кандидат -> позиция замены
    for pos in range(12):
        for alt in CONFUSIONS.get(number[pos], ()):
            cand = number[:pos] + alt + number[pos + 1:]
            if is_valid(cand) and cand not in found:
                found[cand] = pos

    candidates = list(found)
    if len(candidates) > 1:
        # Структурный фильтр — дополнительный, не самостоятельный отказ:
        # если он обнуляет список, остаёмся на checksum-кандидатах.
        structural = [c for c in candidates if looks_structural(c)]
        if structural:
            candidates = structural
            found = {c: found[c] for c in candidates}

    if char_confs is not None and len(char_confs) == 12:
        candidates.sort(key=lambda c: (char_confs[found[c]], found[c], c))
    else:
        candidates.sort(key=lambda c: (found[c], c))

    if len(candidates) == 1:
        return RepairResult(
            "fixed", candidates[0], candidates,
            f"{number}: исправлено -> {candidates[0]}",
        )
    if candidates:
        return RepairResult(
            "ambiguous", number, candidates,
            f"{number}: несколько кандидатов {candidates}, не исправляем",
        )
    return RepairResult(
        "invalid", number, [],
        f"{number}: контрольный разряд не сходится, кандидатов нет",
    )
