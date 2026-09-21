"""Внутреннее представление документа.

Пайплайн не производит текст — он производит структуру. Текст рождается
только в ocr/render/. Это позволяет менять конвенцию вывода (разделители
таблиц, склейка переносов, колонтитулы) без переделки распознавания.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

PageSource = Literal["text_layer", "ocr"]
BlockKind = Literal["text", "table"]

WarningType = Literal[
    "bin_checksum_failed",
    "bin_repaired",
    "bin_ambiguous_fix",
    "amount_mismatch",
    "date_invalid",
    "low_confidence_page",
    "engine_disagreement",
    "text_layer_rejected",
    "page_failed",
]


@dataclass(slots=True)
class Word:
    text: str
    x0: int
    y0: int
    x1: int
    y1: int
    conf: float
    """0..1. Tesseract отдаёт 0..100, адаптер нормализует."""

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


@dataclass(slots=True)
class Line:
    words: list[Word] = field(default_factory=list)
    engine: str = ""

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words if w.text)

    @property
    def conf(self) -> float:
        if not self.words:
            return 0.0
        return sum(w.conf for w in self.words) / len(self.words)

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        if not self.words:
            return (0, 0, 0, 0)
        return (
            min(w.x0 for w in self.words),
            min(w.y0 for w in self.words),
            max(w.x1 for w in self.words),
            max(w.y1 for w in self.words),
        )


@dataclass(slots=True)
class Block:
    lines: list[Line] = field(default_factory=list)
    kind: BlockKind = "text"
    columns: list[tuple[int, int]] = field(default_factory=list)
    """Границы колонок по x. Заполняется только для kind == "table":
    рендерер по ним расставляет разделитель ячеек."""

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        boxes = [ln.bbox for ln in self.lines if ln.words]
        if not boxes:
            return (0, 0, 0, 0)
        return (
            min(b[0] for b in boxes),
            min(b[1] for b in boxes),
            max(b[2] for b in boxes),
            max(b[3] for b in boxes),
        )


@dataclass(slots=True)
class Warning:
    type: WarningType
    page: int
    detail: str


@dataclass(slots=True)
class Page:
    index: int
    """0-indexed."""
    width: int = 0
    height: int = 0
    blocks: list[Block] = field(default_factory=list)
    source: PageSource = "ocr"
    rotation: int = 0
    skew: float = 0.0
    dpi: int = 0
    engine: str = ""
    raw_text: str | None = None
    """Заполняется только при source == "text_layer"."""

    @property
    def lines(self) -> list[Line]:
        return [ln for b in self.blocks for ln in b.lines]

    @property
    def mean_conf(self) -> float:
        lines = [ln for ln in self.lines if ln.words]
        if not lines:
            return 1.0 if self.source == "text_layer" else 0.0
        return sum(ln.conf for ln in lines) / len(lines)


@dataclass(slots=True)
class Document:
    pages: list[Page] = field(default_factory=list)
    warnings: list[Warning] = field(default_factory=list)
    fields: dict[str, list[str]] = field(default_factory=dict)
    profile: str = "balanced"
    elapsed_s: float = 0.0

    def warn(self, type_: WarningType, page: int, detail: str) -> None:
        self.warnings.append(Warning(type=type_, page=page, detail=detail))

    @property
    def mean_conf(self) -> float:
        if not self.pages:
            return 0.0
        return sum(p.mean_conf for p in self.pages) / len(self.pages)
