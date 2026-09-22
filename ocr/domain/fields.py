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


def extract_fields(text: str) -> dict[str, list[dict]]:
    """Поля документа со статусом проверки у каждого значения.

    Плоский список строк здесь был бы обманом. Номер, прошедший контрольную
    сумму как прочитан, и номер, исправленный перебором по одной правдоподобной
    гипотезе, выглядели бы одинаково, хотя доверие к ним разное. Поэтому
    каждое значение несёт собственный статус, а потребитель решает, что с ним
    делать.

    ``bin`` — ``status``:

    * ``valid`` — контрольный разряд сошёлся на прочитанном номере;
    * ``repaired`` — номер исправлен, валидный кандидат был ровно один;
    * ``unverified`` — контрольный разряд не сошёлся либо кандидатов
      несколько; ``value`` равен прочитанному, подтверждением не является.

    ``requires_review`` истинно для всего, кроме ``valid``. Даже ``valid`` не
    означает, что номер верен: контрольный разряд не замечает ошибку в 11-й
    позиции (вес 11 ≡ 0 mod 11), а случайная 12-значная последовательность
    проходит проверку примерно в одном случае из одиннадцати.

    ``dates`` — ``status`` ``valid`` (``value`` в ISO) либо ``invalid``
    (``value`` равен ``None``, дата не существует в календаре).

    ``amounts`` — ``words_match``: ``True``, если сумма прописью рядом совпала
    с цифровой записью, ``False`` при расхождении, ``None``, если прописи не
    было и сверять было не с чем.
    """
    bins: list[dict] = []
    seen_bins: set[str] = set()
    for num in _bin_candidates(text):
        if num in seen_bins:
            continue
        seen_bins.add(num)
        res = repair(num)
        if res.status == "valid":
            status, value, review = "valid", res.value, False
        elif res.status == "fixed":
            status, value, review = "repaired", res.value, True
        else:
            status, value, review = "unverified", num, True
        bins.append(
            {
                "value": value,
                "raw": num,
                "status": status,
                "candidates": list(res.candidates),
                "requires_review": review,
            }
        )

    pairs = {check.digits: check for check in find_amount_pairs(text)}
    amounts: list[dict] = []
    for value in sorted(set(find_amounts(text))):
        check = pairs.get(value)
        amounts.append(
            {
                "value": str(value),
                "raw": check.raw if check else str(value),
                "words_match": check.ok if check else None,
            }
        )

    dates: list[dict] = []
    seen_dates: set[str] = set()
    for hit in find_dates(text):
        if hit.raw in seen_dates:
            continue
        seen_dates.add(hit.raw)
        dates.append(
            {
                "value": hit.iso if hit.valid else None,
                "raw": hit.raw,
                "status": "valid" if hit.valid else "invalid",
            }
        )

    return {"bin": bins, "amounts": amounts, "dates": dates}


def confirmed_values(fields: dict[str, list[dict]], key: str) -> list[str]:
    """Значения, которые прошли проверку и пригодны для автоматического разбора.

    Для ``bin`` это ``valid`` и ``repaired``; для ``dates`` — только
    календарно валидные; для ``amounts`` — все найденные. Отдельная функция
    нужна, чтобы потребители не повторяли правило фильтрации у себя и не
    разошлись в нём.
    """
    items = fields.get(key) or []
    if key == "bin":
        return [i["value"] for i in items if i["status"] in ("valid", "repaired")]
    if key == "dates":
        return [i["value"] for i in items if i["status"] == "valid" and i["value"]]
    return [i["value"] for i in items]


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
