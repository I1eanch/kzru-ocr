"""Денежные суммы в документах РК: цифры, пропись, сверка пар.

Характерный шаблон договоров РК: ``500 000 (пятьсот тысяч) тенге`` —
число, затем в скобках сумма прописью. Расхождение цифр и прописи —
типичная OCR-ошибка, которую надо ловить.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_UNITS = {
    "ноль": 0,
    "один": 1, "одна": 1, "одно": 1, "одни": 1,
    "два": 2, "две": 2,
    "три": 3, "четыре": 4, "пять": 5, "шесть": 6, "семь": 7,
    "восемь": 8, "девять": 9, "десять": 10,
    "одиннадцать": 11, "двенадцать": 12, "тринадцать": 13,
    "четырнадцать": 14, "пятнадцать": 15, "шестнадцать": 16,
    "семнадцать": 17, "восемнадцать": 18, "девятнадцать": 19,
}
_TENS = {
    "двадцать": 20, "тридцать": 30, "сорок": 40, "пятьдесят": 50,
    "шестьдесят": 60, "семьдесят": 70, "восемьдесят": 80, "девяносто": 90,
}
_HUNDREDS = {
    "сто": 100, "двести": 200, "триста": 300, "четыреста": 400,
    "пятьсот": 500, "шестьсот": 600, "семьсот": 700,
    "восемьсот": 800, "девятьсот": 900,
}
_MULT = {
    "тысяча": 10**3, "тысячи": 10**3, "тысяч": 10**3, "тысячей": 10**3,
    "миллион": 10**6, "миллиона": 10**6, "миллионов": 10**6, "миллионе": 10**6,
    "миллиард": 10**9, "миллиарда": 10**9, "миллиардов": 10**9, "миллиарде": 10**9,
    "триллион": 10**12, "триллиона": 10**12, "триллионов": 10**12,
}
# Служебные слова, которые не ломают числовую группу.
_IGNORE = {"и", "тенге", "теңге", "тг", "kzt", "₸", "тиын", "тийын"}

# Число: разрядные группы через пробел/точку либо сплошные цифры + копейки.
# Разделители разрядов и пробелы вокруг валюты: space, NBSP, narrow NBSP,
# thin space; в разрядных группах допускается также точка.
_SP = "    "
_NUM = rf"\d{{1,3}}(?:[{re.escape(_SP)}.]\d{{3}})+|\d+"
_DEC = r"(?:,\d{1,2})?"
_CCY = r"(?:тенге|теңге|тг|kzt|₸)"
_LETTER = r"[A-Za-zА-Яа-яЁёӘәІіҢңҒғҮүҰұҚқӨөҺһ]"

# «число [валюта] (пропись)» — пропись обязана содержать хотя бы одну букву.
_PAIR_RE = re.compile(
    rf"(?P<digits>{_NUM}{_DEC})"
    rf"(?:[{re.escape(_SP)}]+{_CCY})?"
    rf"[{re.escape(_SP)}]*\((?=[^)]*{_LETTER})(?P<words>[^)]*)\)",
    re.IGNORECASE,
)

# «число [(пропись)] валюта» — все денежные величины.
_AMOUNT_RE = re.compile(
    rf"(?P<digits>{_NUM}{_DEC})"
    rf"(?:[{re.escape(_SP)}]*\([^)]*\))?"
    rf"[{re.escape(_SP)}]*{_CCY}",
    re.IGNORECASE,
)


def words_to_number(s: str) -> int | None:
    """Русские числительные до миллиардов -> int; неизвестное слово -> None."""
    total = 0
    current = 0
    seen = False
    for raw in re.split(r"[\s\-–—]+", s.lower().replace("ё", "е")):
        w = raw.strip(".,;:!?()«»\"'")
        if not w or w in _IGNORE:
            continue
        if w in _UNITS:
            current += _UNITS[w]
        elif w in _TENS:
            current += _TENS[w]
        elif w in _HUNDREDS:
            current += _HUNDREDS[w]
        elif w in _MULT:
            total += (current if current else 1) * _MULT[w]
            current = 0
        else:
            return None
        seen = True
    if not seen:
        return None
    return total + current


def parse_digits(s: str) -> int | None:
    """``"1 250 000"``, ``"1250000"``, ``"1 250 000,00"``, ``"1.250.000"`` -> int тенге.

    Копейки отбрасываются.
    """
    t = s.strip()
    for sp in ("\u00a0", "\u202f", "\u2009", "\u2007"):
        t = t.replace(sp, " ")
    # Копейки после запятой отсекаем.
    if "," in t:
        t = t.split(",", 1)[0]
    t = t.replace(" ", "")
    if "." in t:
        head, _, tail = t.rpartition(".")
        if tail.isdigit() and len(tail) == 3 and head.replace(".", "").isdigit():
            t = t.replace(".", "")  # точки — разрядные разделители
        else:
            t = t.split(".", 1)[0]  # десятичная точка — копейки
    if not t.isdigit():
        return None
    return int(t)


@dataclass(slots=True)
class AmountCheck:
    """Пара «цифрами (прописью)» и результат сверки."""

    digits: int
    words: int | None
    raw: str
    ok: bool


def find_amount_pairs(text: str) -> list[AmountCheck]:
    """Найти шаблоны «число (пропись)» и сверить цифры с прописью."""
    out: list[AmountCheck] = []
    for m in _PAIR_RE.finditer(text):
        digits = parse_digits(m.group("digits")) or 0
        words = words_to_number(m.group("words"))
        out.append(AmountCheck(
            digits=digits,
            words=words,
            raw=m.group(0),
            ok=words is not None and words == digits,
        ))
    return out


def find_amounts(text: str) -> list[int]:
    """Все денежные величины: число рядом с ``тенге|теңге|тг|KZT|₸``."""
    out: list[int] = []
    for m in _AMOUNT_RE.finditer(text):
        v = parse_digits(m.group("digits"))
        if v is not None:
            out.append(v)
    return out
