"""Извлечение и валидация критичных полей: БИН/ИИН, суммы, даты.

Единая точка для пайплайна и бенчмарка:

``extract_fields(text) -> {"bin": [...], "amounts": [...], "dates": [...]}``
``validate_text(text) -> [(warning_type, detail), ...]``
"""

from __future__ import annotations

import re

from .amounts_ru import find_amount_pairs, find_amounts
from .bin_checksum import repair
from .dates_ru_kk import find_dates

# 12 цифр с возможными пробелами внутри — OCR часто рвёт группу
# («1234 5678 9012»). Границы длинных цифровых рядов проверяются кодом.
BIN_RE = re.compile(r"(?<!\d)(?:\d[ \u00a0\u202f]?){11}\d(?!\d)")

_SPACES = " \u00a0\u202f"


def _bin_candidates(text: str) -> list[str]:
    """Склеенные 12-значные номера; отсекает совпадения внутри длинных рядов."""
    out: list[str] = []
    for m in BIN_RE.finditer(text):
        start, end = m.start(), m.end()
        # Не часть более длинного цифрового ряда (слева цифра+пробелы).
        j = start - 1
        while j >= 0 and text[j] in _SPACES:
            j -= 1
        if j >= 0 and text[j].isdigit():
            continue
        # Справа: цифра сразу или после пробелов.
        j = end
        while j < len(text) and text[j] in _SPACES:
            j += 1
        if j < len(text) and text[j].isdigit():
            continue
        out.append(re.sub(rf"[{re.escape(_SPACES)}]", "", m.group(0)))
    return out


def extract_fields(text: str) -> dict[str, list[str]]:
    """Нормализованные поля документа.

    ``bin`` — только номера, прошедшие контрольную сумму сами или после
    однозначного исправления; ``amounts`` — целые тенге строками без пробелов;
    ``dates`` — ISO ``YYYY-MM-DD`` для валидных, исходная подстрока иначе.

    Номера, не прошедшие проверку, в поля НЕ попадают — они уходят в
    ``bin_checksum_failed``. На сильно деградированных сканах регулярное
    выражение цепляет мусорные 12-значные последовательности: замер на наборе
    с деградацией `heavy` дал 13 найденных «БИН» против 6 настоящих, то есть
    precision 0.46. Контрольная сумма отсеивает такой мусор почти полностью:
    случайная последовательность проходит её примерно в одном случае из
    одиннадцати. Выдать меньше полей и честно пометить проблему лучше, чем
    подсунуть в проверку госдокумента правдоподобный несуществующий номер.
    """
    bins: list[str] = []
    for num in _bin_candidates(text):
        res = repair(num)
        if res.status not in ("valid", "fixed"):
            continue
        if res.value not in bins:
            bins.append(res.value)

    amounts = sorted({str(v) for v in find_amounts(text)}, key=int)

    dates: list[str] = []
    for hit in find_dates(text):
        value = hit.iso if hit.valid else hit.raw
        if value not in dates:
            dates.append(value)

    return {"bin": bins, "amounts": amounts, "dates": dates}


def validate_text(text: str) -> list[tuple[str, str]]:
    """Проблемы полей: ``(warning_type, detail)`` для каждой находки."""
    warnings: list[tuple[str, str]] = []
    for num in _bin_candidates(text):
        res = repair(num)
        if res.status == "invalid":
            warnings.append(("bin_checksum_failed", res.detail))
        elif res.status == "ambiguous":
            warnings.append(("bin_ambiguous_fix", res.detail))
        elif res.status == "fixed":
            # Исправление обязано быть видимым: в документе, где ошибка в
            # цифре недопустима, подмена номера по одной правдоподобной
            # гипотезе должна доходить до оператора, а не растворяться.
            warnings.append((
                "bin_repaired",
                f"{num} → {res.value}: исправлено по контрольной сумме, проверьте оригинал",
            ))
    for check in find_amount_pairs(text):
        if not check.ok:
            warnings.append((
                "amount_mismatch",
                f"{check.raw}: цифрами {check.digits}, прописью {check.words}",
            ))
    for hit in find_dates(text):
        if not hit.valid:
            warnings.append(("date_invalid", f"{hit.raw}: несуществующая дата"))
    # Дедупликация одинаковых предупреждений (повторы того же номера/даты).
    seen: set[tuple[str, str]] = set()
    unique: list[tuple[str, str]] = []
    for w in warnings:
        if w not in seen:
            seen.add(w)
            unique.append(w)
    return unique
