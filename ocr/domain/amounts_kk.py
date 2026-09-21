"""Казахские суммы прописью: парсер числительных и обозначения валюты.

Казахские числительные регулярны — нет родов и падежных форм числа,
поэтому парсер проще русского. OCR и клавиатурный ввод подменяют
казахские литеры русскими аналогами (``мың`` -> ``мын``, ``жүз`` ->
``жуз``, ``тоғыз`` -> ``тогыз`` и т.д.), поэтому токены нормализуются
перед поиском, а словари хранятся уже в нормализованном виде — одна
запись покрывает все варианты написания.
"""

from __future__ import annotations

import re

# Замена казахских литер русскими аналогами. Покрывает варианты
# «мын», «жуз», «тогыз», «елик», «жети», «сегиз», «бир», «кырык»,
# «токсан» и т.п. одной записью в словаре.
_KK_TO_RU = str.maketrans({
    "ә": "а", "ғ": "г", "қ": "к", "ң": "н", "ө": "о",
    "ұ": "у", "ү": "у", "һ": "х", "і": "и",
})


def normalize_token(w: str) -> str:
    """Нижний регистр + замена казахских литер русскими аналогами."""
    return w.lower().translate(_KK_TO_RU)


# Канонические написания -> значение. Рабочие словари собираются из них
# нормализацией ключей; коллизия разных значений под одним ключом —
# ошибка сборки и отдельно зафиксирована тестом.
UNITS = {
    "бір": 1, "екі": 2, "үш": 3, "төрт": 4, "бес": 5,
    "алты": 6, "жеті": 7, "сегіз": 8, "тоғыз": 9,
}
TENS = {
    "он": 10, "жиырма": 20, "отыз": 30, "қырық": 40, "елік": 50,
    "алпыс": 60, "жетпіс": 70, "сексен": 80, "тоқсан": 90,
}
# «жүз» — множитель разряда: «бес жүз» = 500, «жүз» = 100.
HUNDRED = {"жүз": 100}
MULT = {
    "мың": 10**3, "миллион": 10**6, "миллиард": 10**9, "триллион": 10**12,
}
# Все числительные одним отображением — для проверки коллизий.
NUMBER_WORDS = {**UNITS, **TENS, **HUNDRED, **MULT}

# Казахские обозначения валюты.
CURRENCY_WORDS = {"теңге", "тенге", "тг", "тиын"}

# Служебные слова, которые не ломают числовую группу: валюта,
# OCR-вариант «тийын», латинское/символьное обозначение и союз «и»
# для смешанных ru/kk документов.
_IGNORE_SRC = CURRENCY_WORDS | {"тийын", "kzt", "₸", "и"}


def _build(src: dict[str, int]) -> dict[str, int]:
    """Словарь по нормализованным ключам; коллизия значений — ошибка."""
    out: dict[str, int] = {}
    for word, value in src.items():
        key = normalize_token(word)
        if key in out and out[key] != value:
            raise ValueError(
                f"нормализация склеила {word!r} с другим числительным")
        out[key] = value
    return out


_UNITS = _build(UNITS)
_TENS = _build(TENS)
_HUNDRED = _build(HUNDRED)
_MULT = _build(MULT)
_IGNORE = {normalize_token(w) for w in _IGNORE_SRC}


def words_to_number(s: str) -> int | None:
    """Казахские числительные до триллионов -> int; неизвестное слово -> None."""
    total = 0
    current = 0
    seen = False
    for raw in re.split(r"[\s\-–—]+", s):
        w = normalize_token(raw.strip(".,;:!?()«»\"'"))
        if not w or w in _IGNORE:
            continue
        if w in _UNITS:
            current += _UNITS[w]
        elif w in _TENS:
            current += _TENS[w]
        elif w in _HUNDRED:
            current = (current if current else 1) * 100
        elif w in _MULT:
            total += (current if current else 1) * _MULT[w]
            current = 0
        else:
            return None
        seen = True
    if not seen:
        return None
    return total + current
