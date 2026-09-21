"""Растеризация PDF и решение «брать текстовый слой или распознавать».

Два неочевидных места:

1. Часть входящих «сканов» несёт чужой плохой OCR-слой. Взять его как есть —
   значит унаследовать чужие ошибки. Поэтому слой проверяется на пригодность.
2. Tesseract LSTM чувствителен к масштабу: оптимум — высота строчных букв
   примерно 25-35 px. Фиксированные 300 DPI на документе, отсканированном в
   150 DPI с крупным кеглем, дают не тот масштаб. Поэтому после растеризации
   масштаб оценивается по связным компонентам и правится ресайзом.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import cv2
import pymupdf as fitz
import numpy as np

BASE_DPI = 300
TARGET_XHEIGHT_PX = 30
MIN_XHEIGHT_PX = 18
MAX_XHEIGHT_PX = 48
MAX_PAGE_PIXELS = 40_000_000
"""Защита от 600-DPI цветных A3: 40 Мпикс в grayscale — это 40 МБ на страницу."""

_CYRILLIC_RE = re.compile(r"[а-яёәғқңөұүһіА-ЯЁӘҒҚҢӨҰҮҺІ]")
_WORD_RE = re.compile(r"\S+")


@dataclass(slots=True)
class RasterPage:
    index: int
    image: np.ndarray
    """grayscale uint8."""
    dpi: int
    scale: float
    """Применённый ресайз относительно BASE_DPI."""


@dataclass(slots=True)
class TextLayer:
    index: int
    text: str
    usable: bool
    reason: str


def _page_text(page: fitz.Page) -> str:
    return page.get_text("text") or ""


def assess_text_layer(page: fitz.Page) -> TextLayer:
    """Пригоден ли текстовый слой страницы.

    Отвергаем в четырёх случаях: слоя нет; текста подозрительно мало для
    страницы, занятой полноразмерным изображением; нет кириллицы, хотя
    документ русско-казахский; слой похож на мусорный OCR (много однобуквенных
    токенов и мало алфавитных символов).
    """
    text = _page_text(page)
    stripped = text.strip()
    idx = page.number or 0

    if len(stripped) < 40:
        return TextLayer(idx, text, False, "слой пуст или почти пуст")

    words = _WORD_RE.findall(stripped)
    alpha = sum(ch.isalpha() for ch in stripped)
    alpha_ratio = alpha / len(stripped)

    if not _CYRILLIC_RE.search(stripped):
        return TextLayer(idx, text, False, "в слое нет кириллицы")

    if alpha_ratio < 0.45:
        return TextLayer(idx, text, False, f"мало алфавитных символов: {alpha_ratio:.2f}")

    single = sum(1 for w in words if len(w) == 1)
    if words and single / len(words) > 0.35:
        return TextLayer(idx, text, False, "много однобуквенных токенов — похоже на мусорный OCR-слой")

    # Страница, полностью закрытая изображением, при коротком тексте —
    # типичный «скан с подписью-водяным знаком».
    area = abs(page.rect.width * page.rect.height)
    if area > 0:
        covered = 0.0
        for img in page.get_images(full=True):
            for rect in page.get_image_rects(img[0]):
                covered = max(covered, abs(rect.width * rect.height) / area)
        if covered > 0.7 and len(stripped) < 400:
            return TextLayer(idx, text, False, f"страница закрыта изображением на {covered:.0%} при коротком тексте")

    return TextLayer(idx, text, True, "ok")


def estimate_xheight(gray: np.ndarray) -> float:
    """Медианная высота буквоподобных связных компонент, px.

    Отбор по площади и пропорциям отбрасывает точки, линии рамок и склейки
    целых строк, оставляя отдельные глифы.
    """
    if gray.size == 0:
        return 0.0
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    heights: list[int] = []
    for i in range(1, count):
        x0, y0, w, h, area = stats[i]
        if h < 4 or h > 200 or w < 2:
            continue
        if w > 6 * h:  # горизонтальная линия
            continue
        if h > 6 * w and h > 40:  # вертикальная линия
            continue
        if area < 0.15 * w * h:  # рамка, а не глиф
            continue
        heights.append(int(h))
    if len(heights) < 20:
        return 0.0
    return float(np.median(heights))


def _cap_zoom(page: fitz.Page, zoom: float) -> float:
    pixels = (page.rect.width * zoom) * (page.rect.height * zoom)
    if pixels <= MAX_PAGE_PIXELS:
        return zoom
    return zoom * math.sqrt(MAX_PAGE_PIXELS / pixels)


def rasterize_page(page: fitz.Page, base_dpi: int = BASE_DPI, adaptive: bool = True) -> RasterPage:
    zoom = _cap_zoom(page, base_dpi / 72.0)
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csGRAY, alpha=False)
    gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).copy()
    effective_dpi = int(round(zoom * 72.0))
    scale = 1.0

    if adaptive:
        xh = estimate_xheight(gray)
        if xh and not (MIN_XHEIGHT_PX <= xh <= MAX_XHEIGHT_PX):
            scale = TARGET_XHEIGHT_PX / xh
            scale = max(0.4, min(3.0, scale))
            new_w, new_h = int(gray.shape[1] * scale), int(gray.shape[0] * scale)
            if new_w * new_h <= MAX_PAGE_PIXELS and new_w > 0 and new_h > 0:
                interp = cv2.INTER_LANCZOS4 if scale > 1.0 else cv2.INTER_AREA
                gray = cv2.resize(gray, (new_w, new_h), interpolation=interp)
            else:
                scale = 1.0

    return RasterPage(index=page.number or 0, image=gray, dpi=effective_dpi, scale=scale)


def open_pdf(path: str) -> fitz.Document:
    return fitz.open(path)
