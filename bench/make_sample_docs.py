"""Генератор синтетических текстослойных PDF для dev-набора бенчмарка.

Реальных документов РК в репозитории нет, поэтому набор воспроизводимый:
один и тот же ``--seed`` даёт одинаковые документы.

Использование::

    python -m bench.make_sample_docs --out-dir bench/data --count 8 [--seed 42]

Каждый документ — 1-3 страницы одного из жанров: устав ТОО, договор,
справка. Содержимое включает валидный по контрольной сумме БИН, ИИН,
суммы прописью, даты в русском и казахском форматах, абзац на казахском
со всеми национальными литерами, таблицу и блок подписи.

Рядом с каждым PDF пишется ``<name>.fields.json`` — истинные значения
полей (БИН/ИИН, суммы, даты ISO), записанные в момент подстановки в
текст. Это независимый от экстрактора ground truth для ``bench.metrics``.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import sys
import textwrap
from pathlib import Path

import pymupdf as fitz

# --- Геометрия страницы (A4) -------------------------------------------------

PAGE_W, PAGE_H = 595.0, 842.0
MARGIN = 56.0
FONT_SIZE = 11.0
LINE_H = 16.0
BOTTOM_LIMIT = PAGE_H - MARGIN - 3 * LINE_H  # запас под блок подписи
WRAP_WIDTH = 78

# --- Данные для генерации ----------------------------------------------------

ORGS = [
    "Алтын Дан", "Барыс Групп", "Есіл Строй", "Қазына Логистик",
    "Сарыарка Трейд", "Тенгиз Сервис", "Жібек Жолы", "Нұрлы Құрылыс",
]
SURNAMES = [
    "Абдрахманов А.Е.", "Сейтқазиева Г.Н.", "Мұратбеков Д.С.",
    "Ким В.П.", "Оспанова Ж.Т.", "Ибрагимов Н.К.",
]
CITIES = ["Алматы", "Астана", "Шымкент", "Караганда"]
AMOUNTS = [
    (500_000, "пятьсот тысяч"),
    (1_250_000, "один миллион двести пятьдесят тысяч"),
    (3_700_000, "три миллиона семьсот тысяч"),
    (750_000, "семьсот пятьдесят тысяч"),
    (12_000_000, "двенадцать миллионов"),
    (98_500, "девяносто восемь тысяч пятьсот"),
]
RU_MONTHS = [
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
]
KZ_MONTHS = [
    "қаңтар", "ақпан", "наурыз", "сәуір", "мамыр", "маусым",
    "шілде", "тамыз", "қыркүйек", "қазан", "қараша", "желтоқсан",
]
KZ_PARAGRAPH = (
    "Осы құжаттың мәтіні қазақ тілінде де жазылады: ұлттық әліпби "
    "(ә, ғ, қ, ң, ө, ұ, ү, һ, і) толық қамтылады. Тараптар өзара "
    "келісімге қол жеткізді және міндеттемелерді орындауға кірісті. "
    "Ұлттық валютадағы төлемдер белгіленген мерзімде жүзеге асырылады."
)
FILLER_RU = [
    "Стороны обязуются соблюдать условия настоящего документа и нести "
    "ответственность за их нарушение в соответствии с законодательством "
    "Республики Казахстан.",
    "Все споры и разногласия разрешаются путём переговоров, а при "
    "недостижении согласия — в судебном порядке по месту нахождения "
    "ответчика.",
    "Настоящий документ составлен в двух экземплярах, имеющих одинаковую "
    "юридическую силу, по одному для каждой из сторон.",
    "Стороны подтверждают, что ознакомлены со всеми условиями, понимают "
    "их содержание и действуют добровольно, без принуждения.",
    "Изменения и дополнения действительны при условии оформления в "
    "письменном виде и подписания обеими сторонами.",
    "При изменении реквизитов сторона обязана уведомить другую сторону "
    "в течение десяти рабочих дней с момента такого изменения.",
]
GENRES = ["ustav", "dogovor", "spravka"]


# --- Контрольная цифра БИН/ИИН ------------------------------------------------

def _control_digit_local(first11: str) -> int:
    """Контрольный разряд ИИН/БИН РК по первым 11 цифрам.

    S = (Σ i·a_i, i=1..11) mod 11; при S == 10 пересчёт со второй серией
    весов (3,4,5,6,7,8,9,10,11,1,2). Если и там остаток 10 — номер
    невалиден, возвращается -1.
    """
    digits = [int(c) for c in first11]
    s = sum((i + 1) * d for i, d in enumerate(digits)) % 11
    if s != 10:
        return s
    weights = (3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2)
    s2 = sum(w * d for w, d in zip(weights, digits)) % 11
    return s2 if s2 != 10 else -1


def _control_digit(first11: str) -> int:
    """Обёртка: предпочитает ocr.domain.bin_checksum.control_digit.

    Каноническая реализация (DomainLayer): control_digit(first11) -> int|None,
    None — если номер невалиден. При отсутствии модуля или ином результате
    используется локальная копия алгоритма.
    """
    try:
        from ocr.domain.bin_checksum import control_digit

        cd = control_digit(first11)
        if cd is not None:
            return int(cd)
    except Exception:
        pass
    return _control_digit_local(first11)


def _gen_number12(rng: random.Random) -> str:
    """12-значный номер с валидной контрольной суммой."""
    while True:
        first11 = "".join(rng.choice("0123456789") for _ in range(11))
        cd = _control_digit(first11)
        if 0 <= cd <= 9:
            return first11 + str(cd)


# --- Шрифт --------------------------------------------------------------------

def _find_font() -> str:
    """Путь к DejaVuSans.ttf с покрытием казахских литер."""
    candidates = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    candidates += glob.glob("/usr/share/fonts/**/DejaVuSans.ttf", recursive=True)
    candidates += glob.glob("/System/Library/Fonts/**/DejaVuSans.ttf", recursive=True)
    for path in candidates:
        if Path(path).is_file():
            return path
    sys.exit(
        "ошибка: не найден DejaVuSans.ttf — нужен шрифт с полным покрытием "
        "казахских литер.\n"
        "В Debian-образе установите пакет fonts-dejavu-core "
        "(файл /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf) "
        "или положите DejaVuSans.ttf в /usr/share/fonts."
    )


# --- Построитель страниц -------------------------------------------------------

class _DocBuilder:
    """Пишет текст построчно, сам открывает новые страницы."""

    def __init__(self, doc: fitz.Document, fontfile: str) -> None:
        self.doc = doc
        self.fontfile = fontfile
        self.page: fitz.Page | None = None
        self.y = 0.0
        self._new_page()

    def _new_page(self) -> None:
        self.page = self.doc.new_page(width=PAGE_W, height=PAGE_H)
        self.page.insert_font(fontname="djv", fontfile=self.fontfile)
        self.y = MARGIN
    def line(self, text: str = "", size: float = FONT_SIZE, height: float = LINE_H) -> None:
        if self.y + height > BOTTOM_LIMIT:
            self._new_page()
        if text:
            self.page.insert_text(
                (MARGIN, self.y + height - 4), text,
                fontname="djv", fontsize=size,
            )
        self.y += height

    def para(self, text: str, size: float = FONT_SIZE) -> None:
        for chunk in textwrap.wrap(text, WRAP_WIDTH) or [""]:
            self.line(chunk, size=size)
        self.line()  # пустая строка после абзаца

    def title(self, text: str) -> None:
        # 15pt: ~55 символов на строку при ширине текста 483pt.
        for chunk in textwrap.wrap(text, 55) or [""]:
            self.line(chunk, size=15.0, height=22.0)
        self.line()

    def table(self, rows: list[list[str]], col_x: list[float]) -> None:
        """Простая таблица с рамкой; col_x — x-координаты колонок."""
        top = self.y
        for row in rows:
            if self.y + LINE_H > BOTTOM_LIMIT:
                self._new_page()
                top = self.y
            baseline = self.y + LINE_H - 4
            for x, cell in zip(col_x, row):
                self.page.insert_text(
                    (x + 4, baseline), cell, fontname="djv", fontsize=FONT_SIZE
                )
            self.y += LINE_H
        bottom = self.y
        xs = [MARGIN, *col_x[1:], PAGE_W - MARGIN]
        for x in xs:
            self.page.draw_line((x, top), (x, bottom))
        for yy in [top, *[top + LINE_H * (i + 1) for i in range(len(rows))]]:
            self.page.draw_line((xs[0], yy), (xs[-1], yy))
        self.line()

    def signature(self, left: str, right: str) -> None:
        if self.y + 3 * LINE_H > BOTTOM_LIMIT:
            self._new_page()
        baseline = self.y + LINE_H - 4
        self.page.insert_text((MARGIN, baseline), left, fontname="djv", fontsize=FONT_SIZE)
        self.page.insert_text((PAGE_W - MARGIN - 170, baseline), right,
                              fontname="djv", fontsize=FONT_SIZE)
        self.y += LINE_H
        self.page.insert_text(
            (PAGE_W - MARGIN - 170, self.y + LINE_H - 4), "М.П.",
            fontname="djv", fontsize=FONT_SIZE,
        )
        self.y += LINE_H


# --- Контекст документа ---------------------------------------------------------

def _make_ctx(rng: random.Random) -> dict:
    org1, org2 = rng.sample(ORGS, 2)
    day, month, year = rng.randint(1, 28), rng.randrange(12), rng.randint(2020, 2025)
    amount, amount_words = rng.choice(AMOUNTS)
    return {
        "org1": org1,
        "org2": org2,
        "bin1": _gen_number12(rng),
        "bin2": _gen_number12(rng),
        "iin": _gen_number12(rng),
        "amount_int": amount,
        "amount": f"{amount:,}".replace(",", " "),
        "amount_words": amount_words,
        "date_ru": f"«{day:02d}» {RU_MONTHS[month]} {year} года",
        "date_kz": f"{year} жылғы {day} {KZ_MONTHS[month]}",
        "date_iso": f"{year}-{month + 1:02d}-{day:02d}",
        "city": rng.choice(CITIES),
        "person1": rng.choice(SURNAMES),
        "person2": rng.choice(SURNAMES),
        "doc_no": rng.randint(1, 999),
    }


def _record(truth: dict[str, list[str]], key: str, value: str) -> None:
    """Истинное значение поля — в момент подстановки в текст, без разбора."""
    if value not in truth[key]:
        truth[key].append(value)


# --- Жанры ----------------------------------------------------------------------

def _build_ustav(b: _DocBuilder, c: dict, truth: dict[str, list[str]]) -> None:
    b.title(f"УСТАВ ТОВАРИЩЕСТВА С ОГРАНИЧЕННОЙ ОТВЕТСТВЕННОСТЬЮ «{c['org1']}»")
    b.para(
        f"1. Общие положения. Товарищество с ограниченной ответственностью "
        f"«{c['org1']}» (далее — Товарищество), БИН {c['bin1']}, создано и "
        f"действует в соответствии с законодательством Республики Казахстан. "
        f"Место нахождения: город {c['city']}."
    )
    _record(truth, "bin", c["bin1"])
    b.para(
        f"2. Уставный капитал Товарищества составляет {c['amount']} "
        f"({c['amount_words']}) тенге и формируется вкладами участников."
    )
    _record(truth, "amounts", str(c["amount_int"]))
    b.para(
        f"3. Участник Товарищества: {c['person1']}, ИИН {c['iin']}. "
        f"Дата утверждения устава: {c['date_ru']} ({c['date_kz']})."
    )
    _record(truth, "bin", c["iin"])
    _record(truth, "dates", c["date_iso"])
    b.para("4. Распределение долей участников:")
    b.table(
        [
            ["№", "Участник", "Доля, %", "Вклад, тенге"],
            ["1", c["person1"], "60", "300 000"],
            ["2", c["person2"], "40", "200 000"],
            ["", "Итого", "100", c["amount"]],
        ],
        [MARGIN, MARGIN + 30, MARGIN + 260, MARGIN + 340],
    )
    b.para(KZ_PARAGRAPH)


def _build_dogovor(b: _DocBuilder, c: dict, truth: dict[str, list[str]]) -> None:
    b.title(f"ДОГОВОР № {c['doc_no']} купли-продажи")
    b.para(f"г. {c['city']}                                                    {c['date_ru']}")
    _record(truth, "dates", c["date_iso"])
    b.para(
        f"ТОО «{c['org1']}», БИН {c['bin1']}, именуемое «Продавец», в лице "
        f"директора {c['person1']}, и ТОО «{c['org2']}», БИН {c['bin2']}, "
        f"именуемое «Покупатель», в лице директора {c['person2']}, заключили "
        f"настоящий договор о нижеследующем."
    )
    _record(truth, "bin", c["bin1"])
    _record(truth, "bin", c["bin2"])
    b.para(
        f"1. Предмет договора. Продавец обязуется передать, а Покупатель — "
        f"принять и оплатить товар на сумму {c['amount']} ({c['amount_words']}) "
        f"тенге. Оплата производится до {c['date_iso']}."
    )
    _record(truth, "amounts", str(c["amount_int"]))
    _record(truth, "dates", c["date_iso"])
    b.para(
        f"2. Ответственность сторон. За нарушение сроков оплаты Покупатель "
        f"уплачивает пеню в размере 0,1% от суммы задолженности за каждый "
        f"день просрочки. Дата в казахском формате: {c['date_kz']}."
    )
    b.para("3. Спецификация товара:")
    b.table(
        [
            ["№", "Наименование", "Кол-во", "Сумма, тенге"],
            ["1", "Оборудование", "1", "350 000"],
            ["2", "Материалы", "10", "120 000"],
            ["3", "Услуги монтажа", "1", "30 000"],
            ["", "Итого", "", c["amount"]],
        ],
        [MARGIN, MARGIN + 30, MARGIN + 260, MARGIN + 340],
    )
    b.para(KZ_PARAGRAPH)


def _build_spravka(b: _DocBuilder, c: dict, truth: dict[str, list[str]]) -> None:
    b.title(f"СПРАВКА № {c['doc_no']}")
    b.para(f"Дата выдачи: {c['date_ru']} / {c['date_kz']}")
    _record(truth, "dates", c["date_iso"])
    b.para(
        f"Настоящая справка выдана {c['person1']}, ИИН {c['iin']}, в "
        f"подтверждение того, что он(а) является работником ТОО «{c['org1']}» "
        f"(БИН {c['bin1']}), расположенного по адресу: город {c['city']}."
    )
    _record(truth, "bin", c["iin"])
    _record(truth, "bin", c["bin1"])
    b.para(
        f"Среднемесячная заработная плата за последние шесть месяцев "
        f"составляет {c['amount']} ({c['amount_words']}) тенге."
    )
    _record(truth, "amounts", str(c["amount_int"]))
    b.para("Начисления по месяцам:")
    b.table(
        [
            ["Месяц", "Начислено, тенге", "Удержано, тенге"],
            ["Январь", "500 000", "50 000"],
            ["Февраль", "500 000", "50 000"],
            ["Март", "500 000", "50 000"],
        ],
        [MARGIN, MARGIN + 140, MARGIN + 330],
    )
    b.para("Справка выдана для предъявления по месту требования.")
    b.para(KZ_PARAGRAPH)


_BUILDERS = {
    "ustav": _build_ustav,
    "dogovor": _build_dogovor,
    "spravka": _build_spravka,
}
_SIGNATURES = {
    "ustav": ("Участник _______________", "Директор _______________"),
    "dogovor": ("Продавец _______________", "Покупатель _______________"),
    "spravka": ("Директор _______________", "Гл. бухгалтер _______________"),
}


def _fill_to_pages(b: _DocBuilder, rng: random.Random, target_pages: int) -> None:
    """Добивает документ абзацами до целевого числа страниц."""
    target_y = (target_pages - 1) * (PAGE_H - 2 * MARGIN) + 0.55 * (PAGE_H - 2 * MARGIN)
    i = 0
    while True:
        filled = (len(b.doc) - 1) * (PAGE_H - 2 * MARGIN) + (b.y - MARGIN)
        if filled >= target_y:
            break
        b.para(FILLER_RU[i % len(FILLER_RU)])
        i += 1


def make_doc(
    rng: random.Random, genre: str, fontfile: str
) -> tuple[fitz.Document, dict[str, list[str]]]:
    """Один документ заданного жанра, 1-3 страницы, и истинные поля."""
    doc = fitz.open()
    b = _DocBuilder(doc, fontfile)
    ctx = _make_ctx(rng)
    truth: dict[str, list[str]] = {"bin": [], "amounts": [], "dates": []}
    _BUILDERS[genre](b, ctx, truth)
    _fill_to_pages(b, rng, rng.randint(1, 3))
    left, right = _SIGNATURES[genre]
    b.signature(left, right)
    return doc, truth


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="bench.make_sample_docs",
        description="Генерирует синтетические текстослойные PDF для dev-набора.",
    )
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="каталог для PDF")
    ap.add_argument("--count", type=int, default=8, help="число документов")
    ap.add_argument("--seed", type=int, default=42, help="seed генератора")
    args = ap.parse_args(argv)

    fontfile = _find_font()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    for i in range(args.count):
        genre = GENRES[i % len(GENRES)]
        name = f"doc_{i:03d}_{genre}"
        doc, truth = make_doc(rng, genre, fontfile)
        path = args.out_dir / f"{name}.pdf"
        doc.save(path, deflate=True)
        doc.close()
        sidecar = args.out_dir / f"{name}.fields.json"
        sidecar.write_text(
            json.dumps(truth, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"{path}  ({genre}) + {sidecar.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
