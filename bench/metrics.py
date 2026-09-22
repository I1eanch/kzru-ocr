"""Метрики качества OCR для бенчмарка.

Два режима сравнения текста:
- ``strict`` — только универсальный перевод строк (``\\r\\n``, ``\\r`` → ``\\n``);
- ``normalized`` — NFC, унификация тире и кавычек, удаление soft hyphen,
  схлопывание пробелов и пустых строк, нижний регистр.

Точность извлечения полей (БИН, суммы, даты) считается по независимому
ground truth из ``<name>.fields.json`` (пишет ``make_sample_docs``), если
sidecar есть; иначе эталон разбирается тем же
``ocr.domain.fields.extract_fields`` — оценка циклична и завышена.
Гипотеза всегда разбирается экстрактором, сравниваются значения из
``confirmed_values`` — единой точки нормализации с продакшен-пайплайном.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import jiwer

from ocr.domain.fields import confirmed_values, extract_fields

# Символы тире и кавычек, приводимые к ASCII в normalized-режиме.
_DASHES = "‐‑‒–—―"
_QUOTES = "«»„“”‟‘’"
_TRANSLATE = str.maketrans(
    {c: "-" for c in _DASHES} | {c: '"' for c in _QUOTES}
)
_SOFT_HYPHEN = "\u00ad"
_WS_RUN = re.compile(r"[ \t]+")


def normalize_for_metric(s: str, mode: str = "normalized") -> str:
    """Приводит текст к канонической форме для сравнения.

    ``mode="strict"`` — только универсальный перевод строк.
    ``mode="normalized"`` — полная нормализация (см. docstring модуля).
    Пустые строки удаляются, пробельные серии схлопываются до одного пробела.
    """
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    if mode == "strict":
        return s
    if mode != "normalized":
        raise ValueError(f"неизвестный режим нормализации: {mode!r}")
    s = unicodedata.normalize("NFC", s)
    s = s.replace(_SOFT_HYPHEN, "")
    s = s.translate(_TRANSLATE)
    lines = [_WS_RUN.sub(" ", ln).strip() for ln in s.split("\n")]
    return "\n".join(ln for ln in lines if ln).lower()


def cer(gt: str, pred: str) -> float:
    """Character Error Rate через jiwer.

    Пустой эталон: 0.0, если предсказание тоже пусто, иначе 1.0.
    """
    if not gt.strip():
        return 0.0 if not pred.strip() else 1.0
    if not pred.strip():
        return 1.0
    return float(jiwer.cer(gt, pred))


def wer(gt: str, pred: str) -> float:
    """Word Error Rate через jiwer; пустой эталон — как в :func:`cer`."""
    if not gt.strip():
        return 0.0 if not pred.strip() else 1.0
    if not pred.strip():
        return 1.0
    return float(jiwer.wer(gt, pred))


@dataclass
class FieldStat:
    """Счётчики по одному типу поля в одном документе или агрегате."""

    expected: int = 0
    found: int = 0
    correct: int = 0

    #: Откуда взята эталонная сторона: sidecar ``truth`` или разбор
    #: эталонного текста экстрактором (циклическая оценка).
    source: Literal["truth", "extractor"] = "extractor"

    @property
    def precision(self) -> float:
        """Доля верных среди найденных; 0.0 при пустом found."""
        return self.correct / self.found if self.found else 0.0

    @property
    def recall(self) -> float:
        """Доля найденных среди ожидаемых; 0.0 при пустом expected."""
        return self.correct / self.expected if self.expected else 0.0

    @property
    def exact(self) -> float:
        """Строгая точность: correct / max(expected, found).

        Штрафует и за пропуски, и за ложные срабатывания.
        1.0, когда полей нет ни в эталоне, ни в предсказании.
        """
        denom = max(self.expected, self.found)
        return self.correct / denom if denom else 1.0


@dataclass
class DocMetrics:
    """Метрики одного документа."""

    name: str
    cer_strict: float
    wer_strict: float
    cer_norm: float
    wer_norm: float
    field_stats: dict[str, FieldStat] = field(default_factory=dict)
    chars_gt: int = 0
    chars_pred: int = 0
    #: Источник эталонной стороны полей: ``truth`` (sidecar) или
    #: ``extractor`` (циклическая оценка по разбору эталона).
    fields_source: Literal["truth", "extractor"] = "extractor"


def load_truth_fields(gt_dir: Path, name: str) -> dict[str, list[str]] | None:
    """Истинные поля документа из sidecar ``<name>.fields.json``, если есть."""
    path = gt_dir / f"{name}.fields.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return {key: [str(v) for v in data.get(key, [])] for key in data}


def compare_fields(
    gt_text: str,
    pred_text: str,
    truth: dict[str, list[str]] | None = None,
) -> dict[str, FieldStat]:
    """Сравнивает поля как множества значений по каждому ключу.

    Эталонная сторона: ``truth`` из sidecar, если передан, иначе разбор
    ``gt_text`` экстрактором — тогда оценка циклична (экстрактор меряет
    сам себя) и завышена. Гипотеза всегда разбирается ``extract_fields``;
    в сравнение идут только ``confirmed_values`` — подтверждённые значения.
    """
    pred_fields = extract_fields(pred_text)
    source: Literal["truth", "extractor"] = "extractor"
    if truth is not None:
        source = "truth"
        expected_map = {key: set(vals) for key, vals in truth.items()}
    else:
        gt_fields = extract_fields(gt_text)
        expected_map = {
            key: set(confirmed_values(gt_fields, key)) for key in gt_fields
        }
    stats: dict[str, FieldStat] = {}
    for key in sorted(set(expected_map) | set(pred_fields)):
        expected = expected_map.get(key, set())
        found = set(confirmed_values(pred_fields, key))
        stats[key] = FieldStat(
            expected=len(expected),
            found=len(found),
            correct=len(expected & found),
            source=source,
        )
    return stats
