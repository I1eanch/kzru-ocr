"""Доступ к PDF: два взаимозаменяемых бэкенда.

Зачем два. Разрешённый стек ТЗ называет для растеризации `pdftoppm` или
Ghostscript, и это же снимает лицензионный вопрос: PyMuPDF распространяется
под AGPL-3.0, что для закрытого коммерческого сервиса означает либо раскрытие
исходников, либо коммерческую лицензию Artifex. Poppler (`pdftoppm`,
`pdftotext`, `pdfinfo`, `pdfimages`) — GPL-2 утилиты, вызываемые как внешние
процессы, без линковки с продуктом.

Поэтому по умолчанию работает `poppler`, а `pymupdf` остаётся опцией: он
быстрее (нет запуска процесса и обмена через файлы) и удобен в bench-скриптах,
где лицензионный режим не важен.

Выбор: параметр `backend=` или переменная окружения `KZRU_PDF_BACKEND`.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

DEFAULT_BACKEND = os.environ.get("KZRU_PDF_BACKEND", "poppler")
_PAGES_RE = re.compile(r"^Pages:\s+(\d+)", re.MULTILINE)
_SIZE_RE = re.compile(r"^Page\s+\d+\s+size:\s+([\d.]+)\s+x\s+([\d.]+)\s+pts", re.MULTILINE)
_SIZE_FALLBACK_RE = re.compile(r"^Page size:\s+([\d.]+)\s+x\s+([\d.]+)\s+pts", re.MULTILINE)


class BackendUnavailable(RuntimeError):
    """Бэкенд не установлен в этом окружении."""


@dataclass(slots=True)
class PageInfo:
    index: int
    width_pt: float
    height_pt: float


class PdfDocument(Protocol):
    """Минимум, который нужен пайплайну от PDF."""

    page_count: int

    def page_info(self, index: int) -> PageInfo: ...
    def page_text(self, index: int) -> str: ...
    def image_coverage(self, index: int) -> float: ...
    def rasterize(self, index: int, dpi: int) -> np.ndarray: ...
    def close(self) -> None: ...


def _run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"{cmd[0]} завершился с кодом {result.returncode}: {result.stderr.strip()[:200]}")
    return result.stdout


class PopplerDocument:
    """PDF через утилиты poppler. Каждая операция — отдельный процесс."""

    __slots__ = ("path", "page_count", "_sizes", "_tmp", "_texts")

    def __init__(self, path: str) -> None:
        for tool in ("pdfinfo", "pdftoppm", "pdftotext"):
            if shutil.which(tool) is None:
                raise BackendUnavailable(f"не найден {tool}: установите poppler-utils")

        self.path = path
        info = _run(["pdfinfo", path])
        match = _PAGES_RE.search(info)
        if not match:
            raise RuntimeError(f"pdfinfo не сообщил число страниц для {path}")
        self.page_count = int(match.group(1))
        self._sizes: dict[int, tuple[float, float]] = {}
        self._texts: list[str] | None = None
        self._tmp = tempfile.mkdtemp(prefix="kzru-poppler-")

        size = _SIZE_FALLBACK_RE.search(info)
        if size:
            default = (float(size.group(1)), float(size.group(2)))
            for i in range(self.page_count):
                self._sizes[i] = default

    def page_info(self, index: int) -> PageInfo:
        size = self._sizes.get(index)
        if size is None:
            # Документ со страницами разного формата: pdfinfo умеет отдать
            # размеры построчно только с -f/-l.
            info = _run(["pdfinfo", "-f", str(index + 1), "-l", str(index + 1), self.path])
            match = _SIZE_RE.search(info) or _SIZE_FALLBACK_RE.search(info)
            size = (float(match.group(1)), float(match.group(2))) if match else (595.0, 842.0)
            self._sizes[index] = size
        return PageInfo(index=index, width_pt=size[0], height_pt=size[1])

    def page_text(self, index: int) -> str:
        # Текст берётся одним вызовом на весь документ и кэшируется: страницы
        # разделены переводом формы. Вызов pdftotext на каждую страницу — это
        # N процессов вместо одного, что заметно на 30-страничных файлах.
        if self._texts is None:
            try:
                whole = _run(["pdftotext", "-layout", "-enc", "UTF-8", self.path, "-"])
            except RuntimeError:
                whole = ""
            self._texts = whole.split("\f")
        if 0 <= index < len(self._texts):
            return self._texts[index]
        return ""

    def image_coverage(self, index: int) -> float:
        if shutil.which("pdfimages") is None:
            return 0.0
        page = str(index + 1)
        try:
            listing = _run(["pdfimages", "-list", "-f", page, "-l", page, self.path])
        except RuntimeError:
            return 0.0

        info = self.page_info(index)
        area = info.width_pt * info.height_pt
        if area <= 0:
            return 0.0

        best = 0.0
        for line in listing.splitlines()[2:]:
            parts = line.split()
            if len(parts) < 14:
                continue
            try:
                width, height = int(parts[3]), int(parts[4])
                x_ppi, y_ppi = float(parts[12]), float(parts[13])
            except ValueError:
                continue
            if x_ppi <= 0 or y_ppi <= 0:
                continue
            # Размер изображения на странице в пунктах.
            covered = (width / x_ppi * 72.0) * (height / y_ppi * 72.0)
            best = max(best, covered / area)
        return best

    def rasterize(self, index: int, dpi: int) -> np.ndarray:
        page = str(index + 1)
        prefix = Path(self._tmp) / f"p{index}"
        # PGM вместо PNG: сжатие здесь бесполезно — файл живёт миллисекунды и
        # сразу читается, а кодирование PNG страницы A4/300 DPI стоит заметно
        # дороже самой растеризации. `-singlefile` убирает номер из имени.
        _run(
            [
                "pdftoppm",
                "-f", page, "-l", page,
                "-r", str(dpi),
                "-gray",
                "-singlefile",
                self.path,
                str(prefix),
            ]
        )

        produced = prefix.with_suffix(".pgm")
        if not produced.exists():
            candidates = sorted(Path(self._tmp).glob(f"p{index}*"))
            if not candidates:
                raise RuntimeError(f"pdftoppm не создал изображение для страницы {page}")
            produced = candidates[0]

        image = cv2.imread(str(produced), cv2.IMREAD_GRAYSCALE)
        produced.unlink(missing_ok=True)
        if image is None:
            raise RuntimeError(f"не удалось прочитать растр страницы {page}")
        return image

    def close(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)


class PyMuPDFDocument:
    """PDF через PyMuPDF. Быстрее, но библиотека под AGPL-3.0."""

    __slots__ = ("path", "page_count", "_doc")

    def __init__(self, path: str) -> None:
        try:
            import pymupdf
        except ImportError as exc:  # pragma: no cover
            raise BackendUnavailable("pymupdf не установлен") from exc

        self.path = path
        self._doc = pymupdf.open(path)
        self.page_count = self._doc.page_count

    def page_info(self, index: int) -> PageInfo:
        rect = self._doc[index].rect
        return PageInfo(index=index, width_pt=float(rect.width), height_pt=float(rect.height))

    def page_text(self, index: int) -> str:
        return self._doc[index].get_text("text") or ""

    def image_coverage(self, index: int) -> float:
        page = self._doc[index]
        area = abs(page.rect.width * page.rect.height)
        if area <= 0:
            return 0.0
        best = 0.0
        for img in page.get_images(full=True):
            for rect in page.get_image_rects(img[0]):
                best = max(best, abs(rect.width * rect.height) / area)
        return best

    def rasterize(self, index: int, dpi: int) -> np.ndarray:
        import pymupdf

        page = self._doc[index]
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csGRAY, alpha=False)
        return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).copy()

    def close(self) -> None:
        self._doc.close()


def open_document(path: str, backend: str | None = None) -> PdfDocument:
    name = (backend or DEFAULT_BACKEND).lower()
    if name == "poppler":
        return PopplerDocument(path)
    if name == "pymupdf":
        return PyMuPDFDocument(path)
    raise ValueError(f"неизвестный PDF-бэкенд: {name}")


def available_backends() -> list[str]:
    out: list[str] = []
    if all(shutil.which(t) for t in ("pdfinfo", "pdftoppm", "pdftotext")):
        out.append("poppler")
    try:
        import pymupdf  # noqa: F401

        out.append("pymupdf")
    except ImportError:
        pass
    return out
