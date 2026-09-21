"""Нормализация Unicode и исправление гомоглифов латиница/кириллица.

OCR часто подставляет визуально identical буквы из другого алфавита
(латинская ``a`` вместо кириллической ``а`` и т.п.). Для метрик CER и для
поиска БИН/сумм такие подмены критичны, поэтому смешанные токены приводятся
к доминирующему скрипту.
"""

from __future__ import annotations

import re
import unicodedata

# Таблица гомоглифов в обе стороны. Критично: латинская ``i`` (U+0069)
# соответствует казахской ``і`` (U+0456), а не кириллической ``и``.
LAT2CYR: dict[str, str] = {
    "a": "а", "A": "А", "B": "В", "c": "с", "C": "С", "e": "е", "E": "Е",
    "H": "Н", "i": "і", "K": "К", "M": "М", "o": "о", "O": "О", "p": "р",
    "P": "Р", "T": "Т", "x": "х", "X": "Х", "y": "у",
}
CYR2LAT: dict[str, str] = {v: k for k, v in LAT2CYR.items()}

# Токен — максимальная последовательность букв/цифр (без underscore).
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def normalize_unicode(s: str) -> str:
    """NFC-нормализация: казахские литеры в сканах приходят декомпозированными."""
    return unicodedata.normalize("NFC", s)


def _is_latin(ch: str) -> bool:
    return "a" <= ch <= "z" or "A" <= ch <= "Z"


def _is_cyrillic(ch: str) -> bool:
    return "Ѐ" <= ch <= "ӿ"  # U+0400..U+04FF: кириллица + казахские литеры


def _fix_token(m: re.Match[str]) -> str:
    tok = m.group(0)
    # Токены с цифрами (БИН, суммы, даты) буквенно не конвертируем —
    # ими занимаются bin_checksum/amounts.
    if any(ch.isdigit() for ch in tok):
        return tok
    lat = sum(1 for ch in tok if _is_latin(ch))
    cyr = sum(1 for ch in tok if _is_cyrillic(ch))
    # Чистые токены (ISO, SWIFT, e-mail после разбиения) не трогаем.
    if lat == 0 or cyr == 0:
        return tok
    # Минорный скрипт приводим к доминирующему; при равенстве — к кириллице,
    # т.к. корпус документов русско/казахскоязычный.
    if lat > cyr:
        return "".join(CYR2LAT.get(ch, ch) for ch in tok)
    return "".join(LAT2CYR.get(ch, ch) for ch in tok)


def fix_homoglyphs(s: str) -> str:
    """Привести смешанные латиница/кириллица токены к доминирующему скрипту."""
    return _TOKEN_RE.sub(_fix_token, s)


def normalize(s: str) -> str:
    """Композиция: NFC + исправление гомоглифов."""
    return fix_homoglyphs(normalize_unicode(s))
