#!/usr/bin/env python3
"""Автономные проверки доменного слоя kzru-ocr (stdlib-only).

Запуск: ``python3 tests_domain.py`` — печатает PASS/FAIL по каждому кейсу
и завершается кодом 1 при любом провале.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ocr.domain.amounts_ru import (
    find_amount_pairs,
    find_amounts,
    parse_digits,
    words_to_number,
)
from ocr.domain.bin_checksum import (
    CONFUSIONS,
    control_digit,
    is_valid,
    looks_like_bin,
    looks_like_iin,
    repair,
)
from ocr.domain.dates_ru_kk import find_dates
from ocr.domain.fields import extract_fields, validate_text
from ocr.domain.homoglyphs import fix_homoglyphs, normalize, normalize_unicode

_FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"PASS {name}")
    else:
        print(f"FAIL {name}  {detail}")
        _FAILURES.append(name)


def make_valid(first11: str) -> str:
    cd = control_digit(first11)
    assert cd is not None, f"{first11}: контрольная цифра не вычисляется"
    return first11 + str(cd)


def checksum_candidates(number: str) -> list[str]:
    """Все одиночные замены по CONFUSIONS, проходящие checksum (как в repair)."""
    out = []
    for pos in range(12):
        for alt in CONFUSIONS.get(number[pos], ()):
            cand = number[:pos] + alt + number[pos + 1:]
            if is_valid(cand) and cand not in out:
                out.append(cand)
    return out


def diff_pos(a: str, b: str) -> int:
    return next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)


# --- контрольная цифра: обратный ход -------------------------------------

for p in ("12345678901", "00000000000", "99074000123", "05011512345"):
    num = make_valid(p)
    check(f"control_digit round-trip {p}", is_valid(num), num)

check("is_valid rejects wrong check digit",
      not is_valid("12345678901" + str((control_digit("12345678901") + 1) % 11
                                       if control_digit("12345678901") != 10
                                       else 0)))
check("is_valid rejects short", not is_valid("1234567890"))
check("is_valid rejects non-digit", not is_valid("12345678901a"))

# --- вторая серия весов ---------------------------------------------------

second_series = None
for i in range(200000):
    p = f"{i:011d}"
    s1 = sum((k + 1) * int(c) for k, c in enumerate(p)) % 11
    if s1 == 10:
        s2 = sum(w * int(c) for w, c in zip((3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2), p)) % 11
        if s2 != 10:
            second_series = (p, s2)
            break
check("second weight series prefix found", second_series is not None)
if second_series:
    p, s2 = second_series
    check(f"control_digit uses 2nd series ({p})",
          control_digit(p) == s2, f"got {control_digit(p)}")
    check(f"is_valid 2nd series ({p}{s2})", is_valid(p + str(s2)))

# --- слепое пятно 11-й позиции (известное поведение) ----------------------

# Вес 11-й позиции в первой серии = 11 ≡ 0 (mod 11): порча этой цифры
# не детектируется. Фиксируем как известное поведение, НЕ «исправляем».
blind = None
for i in range(1000):
    p = f"{i:011d}"
    s1 = sum((k + 1) * int(c) for k, c in enumerate(p)) % 11
    if s1 != 10:
        cd = control_digit(p)
        if cd is not None:
            blind = p + str(cd)
            break
if blind:
    pos = 10  # 11-я цифра
    alt = "5" if blind[pos] != "5" else "6"
    corrupted = blind[:pos] + alt + blind[pos + 1:]
    check(f"blind spot pos11 stays valid ({blind} -> {corrupted})",
          is_valid(corrupted))
else:
    check("blind spot fixture", False, "no valid number found")

# --- repair: fixed / ambiguous / invalid ----------------------------------

fixed_case = ambiguous_case = invalid_case = None
for i in range(300000):
    if fixed_case and ambiguous_case and invalid_case:
        break
    p = f"{i:011d}"
    cd = control_digit(p)
    if cd is None:
        continue
    num = p + str(cd)
    for pos in range(12):
        for alt in CONFUSIONS.get(num[pos], ()):
            corrupted = num[:pos] + alt + num[pos + 1:]
            if is_valid(corrupted):
                continue  # слепое пятно — repair вернёт valid
            cands = checksum_candidates(corrupted)
            if len(cands) == 1 and fixed_case is None:
                fixed_case = (corrupted, cands[0])
            elif len(cands) >= 2 and ambiguous_case is None:
                res = repair(corrupted)
                if res.status == "ambiguous":
                    positions = {diff_pos(corrupted, c) for c in res.candidates}
                    if len(positions) == len(res.candidates):
                        ambiguous_case = corrupted
    if invalid_case is None:
        # Случайный на вид номер без единого checksum-кандидата.
        probe = f"{(i * 7919) % 10**12:012d}"
        if not is_valid(probe) and not checksum_candidates(probe):
            invalid_case = probe

check("repair fixed fixture found", fixed_case is not None)
if fixed_case:
    corrupted, expected = fixed_case
    res = repair(corrupted)
    check(f"repair fixed {corrupted} -> {expected}",
          res.status == "fixed" and res.value == expected,
          f"{res.status} {res.value}")

check("repair ambiguous fixture found", ambiguous_case is not None)
if ambiguous_case:
    res = repair(ambiguous_case)
    check(f"repair ambiguous {ambiguous_case}",
          res.status == "ambiguous" and res.value == ambiguous_case
          and len(res.candidates) >= 2,
          f"{res.status} {res.value} {res.candidates}")
    # char_confs влияет только на порядок: кандидат в позиции с минимальной
    # уверенностью идёт первым, выбор при неоднозначности не меняется.
    last = res.candidates[-1]
    confs = [0.9] * 12
    confs[diff_pos(ambiguous_case, last)] = 0.05
    res2 = repair(ambiguous_case, confs)
    check("char_confs reorders candidates",
          res2.status == "ambiguous" and res2.candidates[0] == last
          and res2.value == ambiguous_case,
          f"{res2.candidates}")

check("repair invalid fixture found", invalid_case is not None)
if invalid_case:
    res = repair(invalid_case)
    check(f"repair invalid {invalid_case}",
          res.status == "invalid" and res.value == invalid_case
          and res.candidates == [],
          f"{res.status} {res.candidates}")

valid_num = make_valid("12345678901")
res = repair(valid_num)
check("repair valid passthrough", res.status == "valid" and res.value == valid_num)

# --- структурные фильтры ---------------------------------------------------

check("looks_like_bin", looks_like_bin("990740001234") or
      looks_like_bin(make_valid("99074000123")))
check("looks_like_iin", looks_like_iin("900115300000") or
      looks_like_iin(make_valid("90011530000")))
check("looks_like_bin bad month", not looks_like_bin("991340001234"))

# --- гомоглифы -------------------------------------------------------------

check("homoglyph Aкт", fix_homoglyphs("Aкт") == "Акт")
check("homoglyph қaзақ", fix_homoglyphs("қaзақ") == "қазақ")
check("homoglyph ISO untouched", fix_homoglyphs("ISO") == "ISO")
check("homoglyph e-mail untouched", fix_homoglyphs("e-mail") == "e-mail")
check("homoglyph i -> і (қiлi)", fix_homoglyphs("қiлi") == "қілі")
check("homoglyph digits untouched", fix_homoglyphs("БИН123a") == "БИН123a")
check("normalize NFC", normalize_unicode("й") == "й")
check("normalize composes", normalize("Aкт қiлi") == "Акт қілі")

# --- числительные и суммы --------------------------------------------------

check("words пятьсот тысяч", words_to_number("пятьсот тысяч") == 500000)
check("words 1 250 000",
      words_to_number("один миллион двести пятьдесят тысяч") == 1250000)
check("words миллиард",
      words_to_number("два миллиарда триста миллионов") == 2300000000)
check("words garbage -> None",
      words_to_number("пятьсот квадрокоптеров") is None)
check("words empty -> None", words_to_number("") is None)

check("parse_digits spaced", parse_digits("1 250 000") == 1250000)
check("parse_digits plain", parse_digits("1250000") == 1250000)
check("parse_digits kop", parse_digits("1 250 000,00") == 1250000)
check("parse_digits dots", parse_digits("1.250.000") == 1250000)
check("parse_digits garbage", parse_digits("abc") is None)

pairs = find_amount_pairs("капитал 500 000 (пятьсот тысяч) тенге")
check("amount pair ok", len(pairs) == 1 and pairs[0].ok
      and pairs[0].digits == 500000 and pairs[0].words == 500000,
      repr(pairs))
pairs = find_amount_pairs("капитал 500 000 (шестьсот тысяч) тенге")
check("amount pair mismatch", len(pairs) == 1 and not pairs[0].ok
      and pairs[0].words == 600000, repr(pairs))
pairs = find_amount_pairs(
    "1 250 000 (один миллион двести пятьдесят тысяч) тенге")
check("amount pair 1.25M ok", len(pairs) == 1 and pairs[0].ok
      and pairs[0].digits == 1250000, repr(pairs))

amts = find_amounts("сумма 1 250 000 тенге и ещё 300 тг, 700 KZT")
check("find_amounts", amts == [1250000, 300, 700], repr(amts))
amts = find_amounts("500 000 (пятьсот тысяч) тенге")
check("find_amounts skips parens", amts == [500000], repr(amts))

# --- даты ------------------------------------------------------------------

hits = find_dates("от 12.01.2026 подписано")
check("date dd.mm.yyyy", len(hits) == 1 and hits[0].valid
      and hits[0].iso == "2026-01-12", repr(hits))
hits = find_dates("срок 31.02.2026 истёк")
check("date invalid 31.02", len(hits) == 1 and not hits[0].valid
      and hits[0].iso is None, repr(hits))
hits = find_dates("12 қаңтар 2026 жыл")
check("date kazakh", len(hits) == 1 and hits[0].valid
      and hits[0].iso == "2026-01-12", repr(hits))
hits = find_dates("«15» марта 2024 года")
check("date quoted ru", len(hits) == 1 and hits[0].iso == "2024-03-15",
      repr(hits))
hits = find_dates("выгрузка 2026-09-21 готова")
check("date iso", len(hits) == 1 and hits[0].iso == "2026-09-21", repr(hits))

# --- fields: контракт ------------------------------------------------------

bin_ok = make_valid("99074000123")
text = (f"Устав ТОО «Пример», БИН {bin_ok}, уставный капитал "
        f"500 000 (пятьсот тысяч) тенге, дата составления 12.01.2026")
f = extract_fields(text)
check("extract_fields keys", set(f) == {"bin", "amounts", "dates"}, repr(f))
check("extract_fields bin", f["bin"] == [bin_ok], repr(f["bin"]))
check("extract_fields amounts", f["amounts"] == ["500000"], repr(f["amounts"]))
check("extract_fields dates", f["dates"] == ["2026-01-12"], repr(f["dates"]))

spaced = f"{bin_ok[:4]} {bin_ok[4:8]} {bin_ok[8:]}"
f = extract_fields(f"БИН {spaced} компании")
check("extract_fields spaced BIN glued", f["bin"] == [bin_ok], repr(f["bin"]))

w = validate_text(text)
check("validate_text clean", w == [], repr(w))

if invalid_case:
    w = validate_text(f"БИН {invalid_case} указан")
    check("validate_text bin_checksum_failed",
          ("bin_checksum_failed", f"{invalid_case}: контрольный разряд "
           "не сходится, кандидатов нет") in w
          or any(t == "bin_checksum_failed" for t, _ in w), repr(w))
if ambiguous_case:
    w = validate_text(f"БИН {ambiguous_case} указан")
    check("validate_text bin_ambiguous_fix",
          any(t == "bin_ambiguous_fix" for t, _ in w), repr(w))
w = validate_text("капитал 500 000 (шестьсот тысяч) тенге")
check("validate_text amount_mismatch",
      any(t == "amount_mismatch" for t, _ in w), repr(w))
w = validate_text("дата 31.02.2026 неверна")
check("validate_text date_invalid",
      any(t == "date_invalid" for t, _ in w), repr(w))

allowed = {"bin_checksum_failed", "bin_ambiguous_fix", "amount_mismatch",
           "date_invalid", "low_confidence_page", "engine_disagreement",
           "text_layer_rejected", "page_failed"}
w = validate_text(f"БИН {invalid_case or '000000000000'}, "
                  "500 000 (шестьсот тысяч) тенге, 31.02.2026")
check("validate_text types in WarningType",
      all(t in allowed for t, _ in w), repr(w))

# --- итог ------------------------------------------------------------------

print()
if _FAILURES:
    print(f"FAILED: {len(_FAILURES)} case(s): {', '.join(_FAILURES)}")
    sys.exit(1)
print("ALL PASS")
sys.exit(0)
