"""Даты в русских и казахских юридических документах.

Форматы: ``dd.mm.yyyy``, ``dd/mm/yyyy``, ``yyyy-mm-dd``,
``«12» января 2026 года``, ``12 қаңтар 2026 жыл``.
Названия месяцев сопоставляются по основе — падежные окончания
(``января``, ``қаңтарда``) поддерживаются.
"""

from __future__ import annotations

import datetime
import re
from dataclasses import dataclass

# Названия месяцев -> номер. Ключи — основы слов (падежи режутся при поиске).
MONTHS_RU: dict[str, int] = {
    "январ": 1, "феврал": 2, "март": 3, "апрел": 4, "ма": 5, "июн": 6,
    "июл": 7, "август": 8, "сентябр": 9, "октябр": 10, "ноябр": 11, "декабр": 12,
}
MONTHS_KK: dict[str, int] = {
    "қаңтар": 1, "ақпан": 2, "наурыз": 3, "сәуір": 4, "мамыр": 5,
    "маусым": 6, "шілде": 7, "тамыз": 8, "қыркүйек": 9, "қазан": 10,
    "қараша": 11, "желтоқсан": 12,
}

# Самая длинная основа — первой (иначе «ма» съест «март»/«мамыр»).
_MONTH_STEMS: list[tuple[str, int]] = sorted(
    [("май", 5)]
    + list(MONTHS_RU.items())
    + list(MONTHS_KK.items()),
    key=lambda kv: -len(kv[0]),
)

# Цифровые даты: dd.mm.yyyy, dd/mm/yyyy, dd-mm-yyyy, yyyy-mm-dd.
_NUMERIC_RE = re.compile(
    r"(?<!\d)(?P<d1>\d{1,2})[./](?P<m1>\d{1,2})[./](?P<y1>\d{4})(?!\d)"
    r"|(?<!\d)(?P<y2>\d{4})-(?P<m2>\d{1,2})-(?P<d2>\d{1,2})(?!\d)"
)

# Текстовые даты: «12» января 2026 года / 12 қаңтар 2026 жыл.
_TEXTUAL_RE = re.compile(
    r"(?:[«\"']?\s*)"
    r"(?P<d>\d{1,2})"
    r"(?:\s*[»\"']?)\s+"
    r"(?P<mon>[A-Za-zА-Яа-яЁёӘәІіҢңҒғҮүҰұҚқӨөҺһ]+)"
    r"\s+(?P<y>\d{4})"
    r"(?:\s*(?:года|г\.|жылы|жыл|ж\.))?",
    re.IGNORECASE,
)


def _month_number(word: str) -> int | None:
    """Номер месяца по основе слова (падежные окончания отсекаются)."""
    w = word.lower()
    for stem, num in _MONTH_STEMS:
        if w.startswith(stem):
            return num
    return None


@dataclass(slots=True)
class DateHit:
    """Найденная дата: исходная подстрока, ISO и валидность по календарю."""

    raw: str
    iso: str | None
    valid: bool


def _hit(raw: str, y: int, m: int, d: int) -> DateHit:
    try:
        dt = datetime.date(y, m, d)
    except ValueError:
        return DateHit(raw=raw, iso=None, valid=False)
    return DateHit(raw=raw, iso=dt.isoformat(), valid=True)


def find_dates(text: str) -> list[DateHit]:
    """Все даты в тексте в порядке следования; невалидные — ``valid=False``."""
    hits: list[tuple[int, DateHit]] = []
    for m in _NUMERIC_RE.finditer(text):
        if m.group("d1") is not None:
            hits.append((m.start(), _hit(
                m.group(0), int(m.group("y1")),
                int(m.group("m1")), int(m.group("d1")),
            )))
        else:
            hits.append((m.start(), _hit(
                m.group(0), int(m.group("y2")),
                int(m.group("m2")), int(m.group("d2")),
            )))
    for m in _TEXTUAL_RE.finditer(text):
        mon = _month_number(m.group("mon"))
        if mon is None:
            continue
        hits.append((m.start(), _hit(
            m.group(0), int(m.group("y")), mon, int(m.group("d")),
        )))
    hits.sort(key=lambda t: t[0])
    return [h for _, h in hits]
