"""Порядок чтения через рекурсивный XY-cut.

Число колонок во входящих документах не гарантировано, поэтому порядок не
зашит под «одну колонку»: страница рекурсивно режется по самым широким пустым
полосам — сначала горизонтальным (полосы), затем вертикальным (колонки внутри
полосы). Одноколоночный текст даёт одну колонку и деградирует в обычную
сортировку сверху вниз; двухколоночный разделяется корректно.

Побочная выгода: полоса, распавшаяся на 2+ вертикальные части с несколькими
строками в каждой, — это таблица. Она помечается kind="table", а границы
колонок сохраняются, чтобы рендерер расставил разделители ячеек.
"""

from __future__ import annotations

import numpy as np

from .model import Block, Line

MAX_DEPTH = 6
MIN_GAP_Y_FACTOR = 1.8
"""Вертикальный пропуск считается разрывом, если он шире 1.8 высоты строки."""
MIN_GAP_X_FACTOR = 0.04
"""Горизонтальный — если шире 4% ширины занятой области."""
MIN_GAP_X_ABS = 18


def _median_line_height(lines: list[Line]) -> float:
    heights = [ln.bbox[3] - ln.bbox[1] for ln in lines if ln.words]
    if not heights:
        return 0.0
    return float(np.median(heights))


def _find_gaps(intervals: list[tuple[int, int]], min_gap: float) -> list[int]:
    """Точки разреза между непересекающимися промежутками занятости."""
    if len(intervals) < 2:
        return []
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    cuts: list[int] = []
    for prev, nxt in zip(merged, merged[1:]):
        gap = nxt[0] - prev[1]
        if gap >= min_gap:
            cuts.append((prev[1] + nxt[0]) // 2)
    return cuts


def _split(lines: list[Line], cuts: list[int], axis: int) -> list[list[Line]]:
    """axis=1 — делим по y, axis=0 — по x."""
    if not cuts:
        return [lines]
    bounds = [-10**9, *cuts, 10**9]
    groups: list[list[Line]] = [[] for _ in range(len(bounds) - 1)]
    for ln in lines:
        bbox = ln.bbox
        center = (bbox[1] + bbox[3]) / 2.0 if axis == 1 else (bbox[0] + bbox[2]) / 2.0
        for i in range(len(bounds) - 1):
            if bounds[i] <= center < bounds[i + 1]:
                groups[i].append(ln)
                break
    return [g for g in groups if g]


def _split_words(lines: list[Line], cuts: list[int]) -> list[list[tuple[int, int]]]:
    """Раскладывает пролёты слов по колонкам, заданным точками разреза."""
    bounds = [-10**9, *cuts, 10**9]
    groups: list[list[tuple[int, int]]] = [[] for _ in range(len(bounds) - 1)]
    for line in lines:
        for word in line.words:
            for i in range(len(bounds) - 1):
                if bounds[i] <= word.cx < bounds[i + 1]:
                    groups[i].append((word.x0, word.x1))
                    break
    return groups


def _rows_from_columns(columns: list[list[Line]]) -> list[Line]:
    """Сшивает строки колонок в ряды по пересечению по вертикали."""
    anchor = max(columns, key=len)
    rows: list[Line] = []
    used: set[int] = set()

    for base in sorted(anchor, key=lambda ln: ln.bbox[1]):
        by0, by1 = base.bbox[1], base.bbox[3]
        words = list(base.words)
        for col in columns:
            if col is anchor:
                continue
            for ln in col:
                key = id(ln)
                if key in used:
                    continue
                y0, y1 = ln.bbox[1], ln.bbox[3]
                overlap = min(by1, y1) - max(by0, y0)
                height = min(by1 - by0, y1 - y0) or 1
                if overlap > 0.4 * height:
                    words.extend(ln.words)
                    used.add(key)
        rows.append(Line(words=sorted(words, key=lambda w: w.x0), engine=base.engine))

    # Строки колонок, не попавшие ни в один ряд, не теряем.
    for col in columns:
        if col is anchor:
            continue
        for ln in col:
            if id(ln) not in used:
                rows.append(ln)

    return sorted(rows, key=lambda ln: ln.bbox[1])


def _columns_parallel(columns: list[list[Line]]) -> bool:
    """Идут ли части вертикально рядом, а не одна под другой."""
    spans = [(min(ln.bbox[1] for ln in c), max(ln.bbox[3] for ln in c)) for c in columns]
    for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
        overlap = min(a1, b1) - max(a0, b0)
        shorter = min(a1 - a0, b1 - b0) or 1
        if overlap < 0.5 * shorter:
            return False
    return True


def _pairing_ratio(columns: list[list[Line]]) -> float:
    """Доля строк самой длинной колонки, имеющих пару по вертикали в остальных.

    Высокая доля означает выровненные ряды, то есть таблицу; низкая —
    независимые текстовые колонки.
    """
    anchor = max(columns, key=len)
    others = [c for c in columns if c is not anchor]
    if not others or not anchor:
        return 0.0

    paired = 0
    for base in anchor:
        by0, by1 = base.bbox[1], base.bbox[3]
        if all(
            any(
                (min(by1, ln.bbox[3]) - max(by0, ln.bbox[1])) > 0.4 * (min(by1 - by0, ln.bbox[3] - ln.bbox[1]) or 1)
                for ln in col
            )
            for col in others
        ):
            paired += 1
    return paired / len(anchor)



def _column_bounds(columns: list[list[Line]]) -> list[tuple[int, int]]:
    bounds = []
    for col in columns:
        xs0 = min(ln.bbox[0] for ln in col)
        xs1 = max(ln.bbox[2] for ln in col)
        bounds.append((int(xs0), int(xs1)))
    return sorted(bounds)


def _order_lines(lines: list[Line]) -> list[Line]:
    return sorted(lines, key=lambda ln: (ln.bbox[1], ln.bbox[0]))


def xy_cut(lines: list[Line], depth: int = 0) -> list[Block]:
    lines = [ln for ln in lines if ln.words]
    if not lines:
        return []
    if len(lines) == 1 or depth >= MAX_DEPTH:
        return [Block(lines=_order_lines(lines))]

    line_h = _median_line_height(lines) or 10.0

    y_cuts = _find_gaps([(ln.bbox[1], ln.bbox[3]) for ln in lines], line_h * MIN_GAP_Y_FACTOR)
    if y_cuts:
        blocks: list[Block] = []
        for group in sorted(_split(lines, y_cuts, axis=1), key=lambda g: min(ln.bbox[1] for ln in g)):
            blocks.extend(xy_cut(group, depth + 1))
        return blocks

    x_min = min(ln.bbox[0] for ln in lines)
    x_max = max(ln.bbox[2] for ln in lines)
    min_gap_x = max(MIN_GAP_X_ABS, (x_max - x_min) * MIN_GAP_X_FACTOR)
    # Занятость считается по словам, а не по bbox строк: bbox строки таблицы
    # тянется через все колонки и затирает межколоночные коридоры. Слова же
    # дают разрыв только там, где он пуст во всех строках сразу, поэтому
    # случайный широкий пробел в одной строке колонку не создаёт.
    word_spans = [(w.x0, w.x1) for ln in lines for w in ln.words]
    x_cuts = _find_gaps(word_spans, min_gap_x)

    if x_cuts:
        # Два принципиально разных случая при одинаковых коридорах.
        #
        # Если строки пересекают коридоры, значит движок отдал ряд таблицы
        # одной строкой (Tesseract с psm 6): ряды уже готовы, остаётся лишь
        # запомнить границы колонок для рендерера.
        #
        # Если не пересекают — это либо текстовые колонки, либо построчный
        # вывод детектора (PaddleOCR отдаёт каждую ячейку отдельной строкой),
        # и тогда ряды надо сшить по вертикальному перекрытию.
        crossing = sum(1 for ln in lines if any(ln.bbox[0] < c < ln.bbox[2] for c in x_cuts))
        if crossing >= max(2, int(0.6 * len(lines))):
            word_groups = _split_words(lines, x_cuts)
            return [
                Block(
                    lines=_order_lines(lines),
                    kind="table",
                    columns=[(min(s for s, _ in g), max(e for _, e in g)) for g in word_groups if g],
                )
            ]

        columns = sorted(_split(lines, x_cuts, axis=0), key=lambda g: min(ln.bbox[0] for ln in g))

        # Разрез по x переупорядочивает строки, поэтому он допустим только для
        # настоящей параллельной структуры: в каждой части несколько строк и
        # части идут вертикально рядом. Иначе абзац, где короткий перенос
        # («года») оказался слева под широкой строкой, будет разрезан на две
        # «колонки» и выведен в обратном порядке.
        if len(columns) < 2 or not all(len(c) >= 2 for c in columns) or not _columns_parallel(columns):
            return [Block(lines=_order_lines(lines))]

        # Ряды выровнены по вертикали — это таблица, отданная построчно
        # (так делает детектор PaddleOCR). Иначе это текст в колонках.
        if _pairing_ratio(columns) >= 0.6:
            return [
                Block(
                    lines=_rows_from_columns(columns),
                    kind="table",
                    columns=_column_bounds(columns),
                )
            ]

        blocks = []
        for col in columns:
            blocks.extend(xy_cut(col, depth + 1))
        return blocks

    return [Block(lines=_order_lines(lines))]
