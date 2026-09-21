"""Сериализация структуры документа в текст.

Формат эталона заказчика на момент разработки неизвестен: непонятно, чем
разделены ячейки таблиц, склеены ли переносы по дефису, сохранены ли
колонтитулы. Поэтому распознавание не знает о формате вывода вообще, а
конвенция задаётся профилем. Подбор профиля под чужой эталон — это прогон
`bench.run_bench` по решётке профилей, а не переделка пайплайна.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..domain.homoglyphs import normalize
from ..model import Block, Document, Page

HEADER_ZONE = 0.07
"""Доля высоты страницы сверху и снизу, где ищутся колонтитулы."""
_PAGE_NO_RE = re.compile(r"^[\s\-—–]*(?:стр\.?|бет)?[\s\-—–]*\d{1,4}[\s\-—–.]*$", re.IGNORECASE)


@dataclass(slots=True)
class RenderProfile:
    name: str = "default"
    table_sep: str = "\t"
    hyphen_join: bool = False
    keep_headers: bool = True
    blank_line_between_blocks: bool = True
    page_separator: str = "\n"
    normalize_text: bool = True


PROFILES: dict[str, RenderProfile] = {
    "default": RenderProfile(name="default"),
    "pipe": RenderProfile(name="pipe", table_sep=" | "),
    "cells": RenderProfile(name="cells", table_sep="\n"),
    "space": RenderProfile(name="space", table_sep="  "),
    "joined": RenderProfile(name="joined", hyphen_join=True),
    "no-headers": RenderProfile(name="no-headers", keep_headers=False),
    "flat": RenderProfile(
        name="flat",
        table_sep=" ",
        hyphen_join=True,
        keep_headers=False,
        blank_line_between_blocks=False,
    ),
}


def _is_header_block(block: Block, page_height: int) -> bool:
    if page_height <= 0 or len(block.lines) != 1:
        return False
    y0, y1 = block.bbox[1], block.bbox[3]
    in_top = y1 < page_height * HEADER_ZONE
    in_bottom = y0 > page_height * (1.0 - HEADER_ZONE)
    if not (in_top or in_bottom):
        return False
    text = block.lines[0].text.strip()
    return bool(_PAGE_NO_RE.match(text)) or len(text) <= 40


def _render_table_line(line, columns: list[tuple[int, int]], sep: str) -> str:
    if not columns:
        return line.text
    cells: list[list[str]] = [[] for _ in columns]
    last = len(columns) - 1
    for word in line.words:
        # Колонки не пересекаются и отсортированы, поэтому слово попадает в
        # первую, чья правая граница его не отсекает. Выбор «по ближайшему
        # центру» здесь ошибается: у колонок разная ширина, и слово у левого
        # края широкой колонки оказывается ближе к центру узкой соседней.
        target = last
        for i, (_, end) in enumerate(columns):
            if word.cx <= end:
                target = i
                break
        cells[target].append(word.text)
    return sep.join(" ".join(cell) for cell in cells if cell)


def _render_block(block: Block, profile: RenderProfile) -> list[str]:
    if block.kind == "table":
        return [_render_table_line(ln, block.columns, profile.table_sep) for ln in block.lines]
    return [ln.text for ln in block.lines]


def _join_hyphens(lines: list[str]) -> list[str]:
    out: list[str] = []
    for line in lines:
        if out and out[-1].endswith(("-", "\u00ad")) and line[:1].islower():
            out[-1] = out[-1].rstrip("-\u00ad") + line.lstrip()
        else:
            out.append(line)
    return out


def render_page(page: Page, profile: RenderProfile) -> str:
    if page.source == "text_layer" and page.raw_text is not None:
        text = page.raw_text
        return normalize(text) if profile.normalize_text else text

    chunks: list[str] = []
    for block in page.blocks:
        if not profile.keep_headers and _is_header_block(block, page.height):
            continue
        lines = [ln for ln in _render_block(block, profile) if ln.strip()]
        if not lines:
            continue
        if profile.hyphen_join:
            lines = _join_hyphens(lines)
        chunks.append("\n".join(lines))

    sep = "\n\n" if profile.blank_line_between_blocks else "\n"
    text = sep.join(chunks)
    return normalize(text) if profile.normalize_text else text


def render_document(doc: Document, profile: RenderProfile | str = "default") -> str:
    if isinstance(profile, str):
        profile = PROFILES[profile]
    pages = [render_page(p, profile) for p in doc.pages]
    return profile.page_separator.join(pages).strip() + "\n"
