"""Препроцессинг страницы: ориентация, перекос, шум, бинаризация.

Deskew сделан проекционным профилем, а не `cv2.minAreaRect` по всем чёрным
пикселям: minAreaRect на странице с малым количеством текста, рамкой или
печатью выдаёт угол по мусорному объекту. Проекционный профиль ищет угол, при
котором строки максимально «собраны», и на разреженной странице деградирует
мягко — возвращает 0, а не случайный угол.

Sauvola реализована на интегральных изображениях, чтобы не тащить
scikit-image: локальный порог T = m * (1 + k * (s / R - 1)).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

SAUVOLA_WINDOW = 25
SAUVOLA_K = 0.2
SAUVOLA_R = 128.0

DESKEW_COARSE_RANGE = 5.0
DESKEW_COARSE_STEP = 0.5
DESKEW_FINE_STEP = 0.05
DESKEW_WORK_WIDTH = 700
"""Угол ищется на уменьшенной копии: результат тот же, стоимость в разы ниже."""


@dataclass(slots=True)
class Prepared:
    gray: np.ndarray
    """Для нейросетевых движков — они работают по grayscale/RGB."""
    binary: np.ndarray
    """Для Tesseract — ему подаётся бинаризованное изображение."""
    rotation: int
    skew: float


def sauvola(gray: np.ndarray, window: int = SAUVOLA_WINDOW, k: float = SAUVOLA_K) -> np.ndarray:
    """Локальная бинаризация. Возвращает uint8 {0,255}, текст чёрный."""
    if window % 2 == 0:
        window += 1
    img = gray.astype(np.float64)
    pad = window // 2
    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    integral, integral_sq = cv2.integral2(padded)

    h, w = gray.shape
    ys, xs = np.mgrid[0:h, 0:w]
    y0, x0 = ys, xs
    y1, x1 = ys + window, xs + window
    area = float(window * window)

    total = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
    total_sq = integral_sq[y1, x1] - integral_sq[y0, x1] - integral_sq[y1, x0] + integral_sq[y0, x0]

    mean = total / area
    var = np.maximum(total_sq / area - mean * mean, 0.0)
    std = np.sqrt(var)
    threshold = mean * (1.0 + k * (std / SAUVOLA_R - 1.0))
    return np.where(img > threshold, 255, 0).astype(np.uint8)


def _projection_score(binary_inv: np.ndarray, angle: float) -> float:
    """Дисперсия горизонтальной проекции после поворота на angle."""
    if angle == 0.0:
        rotated = binary_inv
    else:
        h, w = binary_inv.shape
        m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        rotated = cv2.warpAffine(
            binary_inv, m, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
    profile = rotated.sum(axis=1, dtype=np.float64)
    return float(profile.var())


def find_skew(gray: np.ndarray) -> float:
    """Угол перекоса в градусах. Положительный — страница повёрнута по часовой."""
    h, w = gray.shape
    if w > DESKEW_WORK_WIDTH:
        scale = DESKEW_WORK_WIDTH / w
        small = cv2.resize(gray, (DESKEW_WORK_WIDTH, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    else:
        small = gray

    binary_inv = cv2.threshold(small, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    if binary_inv.sum() < 200:
        return 0.0

    coarse = np.arange(-DESKEW_COARSE_RANGE, DESKEW_COARSE_RANGE + DESKEW_COARSE_STEP, DESKEW_COARSE_STEP)
    scores = [(_projection_score(binary_inv, float(a)), float(a)) for a in coarse]
    best = max(scores)[1]

    fine = np.arange(best - DESKEW_COARSE_STEP, best + DESKEW_COARSE_STEP + DESKEW_FINE_STEP, DESKEW_FINE_STEP)
    scores = [(_projection_score(binary_inv, float(a)), float(a)) for a in fine]
    best_score, best_angle = max(scores)

    baseline = _projection_score(binary_inv, 0.0)
    # Не крутим страницу ради шума: выигрыш должен быть заметным.
    if baseline > 0 and best_score / baseline < 1.02:
        return 0.0
    return round(best_angle, 2)


def rotate(gray: np.ndarray, angle: float) -> np.ndarray:
    """Поворот с белым фоном и расширением канвы — текст не обрезается."""
    if angle == 0.0:
        return gray
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    m[0, 2] += (new_w - w) / 2.0
    m[1, 2] += (new_h - h) / 2.0
    return cv2.warpAffine(
        gray, m, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=255
    )


def apply_orientation(gray: np.ndarray, rotation: int) -> np.ndarray:
    """rotation — на сколько градусов страница повёрнута; возвращаем выпрямленную."""
    if rotation == 90:
        return cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotation == 180:
        return cv2.rotate(gray, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
    return gray


def remove_stamp(bgr_or_gray: np.ndarray) -> np.ndarray:
    """Заглушка удаления цветной печати.

    Вынесена в Roadmap: на grayscale-входе цветовая сегментация невозможна, а
    переход на цветную растеризацию удваивает память. Функция оставлена как
    явная точка расширения, а не как работающий шаг.
    """
    return bgr_or_gray


def remove_rules(binary: np.ndarray, min_fraction: float = 0.25) -> np.ndarray:
    """Убирает линии рамки таблиц с бинаризованного изображения.

    Линии сетки сливаются с глифами, и Tesseract на обрамлённой таблице теряет
    целые строки: в замере из пяти строк обрамлённой таблицы доходила одна, и
    та искажённой. Морфологическое открытие длинным ядром выделяет протяжённые
    штрихи, не затрагивая черты букв.

    `min_fraction` — какую долю стороны должна занимать линия, чтобы считаться
    линейкой; 0.25 отсекает подчёркивания отдельных слов.
    """
    inv = 255 - binary
    h, w = binary.shape
    h_len = max(20, int(w * min_fraction))
    v_len = max(20, int(h * min_fraction))

    horizontal = cv2.morphologyEx(inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1)))
    vertical = cv2.morphologyEx(inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len)))
    rules = cv2.bitwise_or(horizontal, vertical)
    if not rules.any():
        return binary

    # Небольшое расширение снимает сглаженные края штриха.
    rules = cv2.dilate(rules, np.ones((3, 3), np.uint8))
    out = binary.copy()
    out[rules > 0] = 255
    return out


def prepare(
    gray: np.ndarray,
    rotation: int = 0,
    denoise: bool = True,
    do_deskew: bool = True,
    drop_rules: bool = True,
) -> Prepared:
    img = apply_orientation(gray, rotation)

    if denoise:
        # medianBlur 3x3 вместо fastNlMeansDenoising: последний стоит
        # 0.5-2 с на страницу на CPU, что недопустимо в бюджете.
        img = cv2.medianBlur(img, 3)

    skew = find_skew(img) if do_deskew else 0.0
    if skew:
        img = rotate(img, skew)

    binary = sauvola(img)
    if drop_rules:
        binary = remove_rules(binary)

    return Prepared(gray=img, binary=binary, rotation=rotation, skew=skew)
