"""Адаптер PaddleOCR (PP-OCRv5, модель cyrillic, lang="kk").

Проверено по словарю `ppocrv5_cyrillic_dict.txt`: он содержит все казахские
литеры `Ә Ғ Қ Ң Ө Ұ Ү Һ І` в обоих регистрах, поэтому `cyrillic_PP-OCRv5_mobile_rec`
пригоден для RU+KK без дообучения. Устаревший `cyrillic_dict.txt` (PP-OCRv3)
казахских литер НЕ содержит — на v3 переключаться нельзя.

Ловушка масштаба: детектор по умолчанию ограничивает сторону 960 px. Страница
A4 при 300 DPI — это 2480x3508, её ужатие в 960 уничтожает мелкий текст.
Поэтому `text_det_limit_side_len` поднят до 2000 и вынесен в параметр.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..model import Line, Word
from ..preprocess import Prepared
from .base import EngineUnavailable

DEFAULT_LIMIT_SIDE_LEN = 2000


@dataclass
class PaddleEngine:
    lang: str = "kk"
    limit_side_len: int = DEFAULT_LIMIT_SIDE_LEN
    name: str = field(default="paddle", init=False)
    _ocr: object | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:  # pragma: no cover - зависит от платформы
            raise EngineUnavailable(
                "paddleocr не установлен. PyPI отдаёт paddlepaddle только для manylinux x86_64; "
                "используйте таргет `full` образа на amd64-хосте"
            ) from exc

        self._ocr = PaddleOCR(
            lang=self.lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_det_limit_side_len=self.limit_side_len,
            text_det_limit_type="max",
        )

    def recognize(self, prepared: Prepared) -> list[Line]:
        # Нейросетевому распознавателю подаётся grayscale->RGB, не бинаризация:
        # сеть обучалась на естественных изображениях, порог ей только вредит.
        rgb = np.stack([prepared.gray] * 3, axis=-1)
        raw = self._predict(rgb)
        return self._lines_from_result(raw)

    def _predict(self, rgb: np.ndarray):
        ocr = self._ocr
        if hasattr(ocr, "predict"):
            return ocr.predict(rgb)
        return ocr.ocr(rgb)  # pragma: no cover - API 2.x

    def _lines_from_result(self, raw) -> list[Line]:
        lines: list[Line] = []
        if not raw:
            return lines

        for page_result in raw:
            payload = getattr(page_result, "json", None)
            if isinstance(page_result, dict):
                payload = page_result
            elif isinstance(payload, dict):
                payload = payload.get("res", payload)

            if isinstance(payload, dict) and "rec_texts" in payload:
                texts = payload.get("rec_texts") or []
                scores = payload.get("rec_scores") or []
                boxes = payload.get("rec_boxes")
                polys = payload.get("rec_polys") or payload.get("dt_polys") or []
                for i, text in enumerate(texts):
                    text = (text or "").strip()
                    if not text:
                        continue
                    conf = float(scores[i]) if i < len(scores) else 0.0
                    bbox = self._bbox(boxes, polys, i)
                    if bbox is None:
                        continue
                    x0, y0, x1, y1 = bbox
                    word = Word(text=text, x0=x0, y0=y0, x1=x1, y1=y1, conf=conf)
                    lines.append(Line(words=[word], engine=self.name))
                continue

            # API 2.x: [[poly, (text, score)], ...]
            for item in page_result or []:  # pragma: no cover
                try:
                    poly, (text, score) = item
                except (TypeError, ValueError):
                    continue
                text = (text or "").strip()
                if not text:
                    continue
                xs = [float(p[0]) for p in poly]
                ys = [float(p[1]) for p in poly]
                word = Word(
                    text=text,
                    x0=int(min(xs)),
                    y0=int(min(ys)),
                    x1=int(max(xs)),
                    y1=int(max(ys)),
                    conf=float(score),
                )
                lines.append(Line(words=[word], engine=self.name))

        return lines

    @staticmethod
    def _bbox(boxes, polys, i: int) -> tuple[int, int, int, int] | None:
        if boxes is not None and len(boxes) > i:
            box = boxes[i]
            return int(box[0]), int(box[1]), int(box[2]), int(box[3])
        if polys is not None and len(polys) > i:
            poly = polys[i]
            xs = [float(p[0]) for p in poly]
            ys = [float(p[1]) for p in poly]
            return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        return None
