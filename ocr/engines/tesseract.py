"""Адаптер Tesseract 5 (LSTM).

Особенности:
- `tessdata_best` для качества, `tessdata_fast` для профиля fast — переключение
  через `--tessdata-dir`, а не через переменную окружения процесса.
- `image_to_data` даёт per-word bbox и confidence; на них держатся арбитраж,
  XY-cut и цифровой доуточняющий проход.
- Отдельный проход по числовому кропу с `tessedit_char_whitelist` заметно
  снижает ошибку в цифрах: модель перестаёт путать `0`/`О` и `1`/`l`.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..model import Line, Word
from ..preprocess import Prepared
from .base import EngineUnavailable

try:  # pragma: no cover - зависит от окружения
    import pytesseract
    from pytesseract import Output
except ImportError:  # pragma: no cover
    pytesseract = None
    Output = None

TESSDATA_BEST = os.environ.get("TESSDATA_BEST", "/opt/tessdata_best")
TESSDATA_FAST = os.environ.get("TESSDATA_FAST", "/usr/share/tesseract-ocr/5/tessdata")

DIGIT_WHITELIST = "0123456789., "


@dataclass(slots=True)
class TesseractEngine:
    lang: str = "rus+kaz"
    psm: int = 6
    tessdata: str = "best"
    user_words: str | None = None
    user_patterns: str | None = None
    name: str = field(default="tesseract", init=False)

    def __post_init__(self) -> None:
        if pytesseract is None:
            raise EngineUnavailable("pytesseract не установлен")
        if shutil.which("tesseract") is None:
            raise EngineUnavailable("бинарь tesseract не найден в PATH")

        # Tesseract при несуществующем `--tessdata-dir` не сообщает об
        # ошибке, а молча берёт модели из вкомпилированного пути. Тогда
        # распознавание идёт на весах, которых профиль не объявлял, и
        # результат не соответствует заявленной конфигурации. Проверено
        # вживую: со сломанным каталогом `/ocr` продолжал отвечать 200.
        missing = [
            lang
            for lang in self.lang.split("+")
            if lang and not Path(self.tessdata_dir, f"{lang}.traineddata").is_file()
        ]
        if missing:
            raise EngineUnavailable(
                f"в каталоге {self.tessdata_dir} нет моделей {missing}; "
                "Tesseract подменил бы их другими без предупреждения"
            )

    @property
    def tessdata_dir(self) -> str:
        return TESSDATA_BEST if self.tessdata == "best" else TESSDATA_FAST

    def _config(self, psm: int | None = None, extra: str = "") -> str:
        parts = [
            f'--tessdata-dir "{self.tessdata_dir}"',
            "--oem 1",
            f"--psm {psm if psm is not None else self.psm}",
        ]
        if self.user_words:
            parts.append(f'--user-words "{self.user_words}"')
        if self.user_patterns:
            parts.append(f'--user-patterns "{self.user_patterns}"')
        if extra:
            parts.append(extra)
        return " ".join(parts)

    def detect_orientation(self, image: np.ndarray) -> int:
        """Угол, на который страница повёрнута (0/90/180/270)."""
        try:
            osd = pytesseract.image_to_osd(
                image, config=f'--tessdata-dir "{self.tessdata_dir}" --psm 0', output_type=Output.DICT
            )
        except Exception:
            return 0
        try:
            return int(osd.get("rotate", 0)) % 360
        except (TypeError, ValueError):
            return 0

    def recognize(self, prepared: Prepared) -> list[Line]:
        data = pytesseract.image_to_data(
            prepared.binary, lang=self.lang, config=self._config(), output_type=Output.DICT
        )
        return self._lines_from_data(data)

    def _lines_from_data(self, data: dict) -> list[Line]:
        groups: dict[tuple[int, int, int, int], list[Word]] = {}
        order: list[tuple[int, int, int, int]] = []

        for i, raw_text in enumerate(data["text"]):
            text = (raw_text or "").strip()
            if not text:
                continue
            try:
                conf = float(data["conf"][i])
            except (TypeError, ValueError):
                conf = -1.0
            if conf < 0:
                continue
            key = (
                int(data["page_num"][i]),
                int(data["block_num"][i]),
                int(data["par_num"][i]),
                int(data["line_num"][i]),
            )
            x, y, w, h = (int(data["left"][i]), int(data["top"][i]), int(data["width"][i]), int(data["height"][i]))
            word = Word(text=text, x0=x, y0=y, x1=x + w, y1=y + h, conf=conf / 100.0)
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(word)

        lines: list[Line] = []
        for key in order:
            words = sorted(groups[key], key=lambda w: w.x0)
            lines.append(Line(words=words, engine=self.name))
        return lines

    def read_digits(self, crop: np.ndarray, upscale: int = 3) -> str:
        """Проход по кропу с ограничением алфавита цифрами."""
        import cv2

        if crop.size == 0:
            return ""
        big = cv2.resize(crop, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
        config = self._config(psm=7, extra=f'-c tessedit_char_whitelist="{DIGIT_WHITELIST}"')
        text = pytesseract.image_to_string(big, lang=self.lang, config=config)
        return text.strip()
