"""Превращает текстослойные PDF в «сканы» без текстового слоя + эталон.

Это бесплатный точный ground truth: эталон извлекается из исходного
текстового слоя, а скан — растеризованная и деградированная картинка.

Если рядом с исходным PDF лежит ``<name>.fields.json`` (истинные поля
от ``make_sample_docs``), он копируется в ``--gt-dir`` — метрики полей
тогда считаются по независимому ground truth, а не по разбору эталона.

Использование::

    python -m bench.make_pseudoscan --src-dir bench/data \
        --out-dir bench/scans --gt-dir bench/gt \
        [--dpi 200] [--degrade light|medium|heavy] [--seed 42]

Детерминированность: один и тот же ``--seed`` даёт побайтово одинаковый
результат — шум и повороты зависят только от seed, имени файла и номера
страницы.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pymupdf as fitz

# Параметры деградации по уровням (кумулятивно).
#   noise   — σ гауссова шума (0 = выкл)
#   angle   — макс. угол поворота, градусы (с белым фоном, без обрезки)
#   blur    — σ GaussianBlur (0 = выкл)
#   jpeg    — качество JPEG
#   sp      — доля salt-and-pepper шума (0 = выкл)
#   erode   — лёгкая эрозия 2x2 (утончает штрихи)
_LEVELS: dict[str, dict] = {
    "light": {"noise": 4.0, "angle": 0.0, "blur": 0.0, "jpeg": 75, "sp": 0.0, "erode": False},
    "medium": {"noise": 6.0, "angle": 1.2, "blur": 0.6, "jpeg": 55, "sp": 0.0, "erode": False},
    "heavy": {"noise": 9.0, "angle": 2.5, "blur": 0.6, "jpeg": 35, "sp": 0.004, "erode": True},
}


def _rotate_white(img: np.ndarray, angle: float) -> np.ndarray:
    """Поворот с расширением холста и белым фоном — текст не обрезается."""
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    m = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos) + 1, int(h * cos + w * sin) + 1
    m[0, 2] += nw / 2.0 - cx
    m[1, 2] += nh / 2.0 - cy
    return cv2.warpAffine(
        img, m, (nw, nh), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=255,
    )


def _degrade(img: np.ndarray, rng: np.random.Generator, level: str) -> bytes:
    """Применяет деградации и возвращает JPEG-байты."""
    p = _LEVELS[level]
    out = img.copy()  # np.frombuffer даёт read-only массив

    if p["noise"] > 0:
        noise = rng.normal(0.0, p["noise"], out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if p["angle"] > 0:
        out = _rotate_white(out, float(rng.uniform(-p["angle"], p["angle"])))
    if p["blur"] > 0:
        out = cv2.GaussianBlur(out, (0, 0), p["blur"])
    if p["sp"] > 0:
        mask = rng.random(out.shape)
        out[mask < p["sp"] / 2] = 0
        out[(mask >= p["sp"] / 2) & (mask < p["sp"])] = 255
    if p["erode"]:
        out = cv2.erode(out, np.ones((2, 2), np.uint8), iterations=1)

    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, p["jpeg"]])
    if not ok:
        raise RuntimeError("cv2.imencode(.jpg) вернул False")
    return buf.tobytes()


def process_pdf(
    src: Path,
    out_pdf: Path,
    gt_path: Path,
    dpi: int,
    degrade: str,
    seed: int,
) -> None:
    """Один файл: эталон из текстового слоя + скан-PDF из картинок."""
    src_doc = fitz.open(src)
    out_doc = fitz.open()
    gt_parts: list[str] = []

    for i, page in enumerate(src_doc):
        gt_parts.append(page.get_text("text"))

        # RNG создаётся на каждую страницу до любых вызовов — порядок
        # потребления случайности не зависит от уровня деградации.
        # SeedSequence принимает только int/последовательность int —
        # имя файла сворачивается в int через blake2b.
        page_seed = int.from_bytes(
            hashlib.blake2b(
                f"{seed}|{src.stem}|{i}".encode(), digest_size=8
            ).digest(),
            "big",
        )
        rng = np.random.default_rng(page_seed)
        pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width
        )
        jpg = _degrade(img, rng, degrade)

        new_page = out_doc.new_page(width=page.rect.width, height=page.rect.height)
        new_page.insert_image(new_page.rect, stream=jpg)

    src_doc.close()
    # PyMuPDF не пишет метаданные с текущим временем — вывод детерминирован.
    out_doc.save(out_pdf, deflate=True)
    out_doc.close()
    gt_path.write_text("\n".join(gt_parts), encoding="utf-8")

    # Sidecar с истинными полями — независимый от экстрактора ground truth.
    sidecar = src.with_suffix(".fields.json")
    if sidecar.is_file():
        shutil.copyfile(sidecar, gt_path.with_suffix(".fields.json"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="bench.make_pseudoscan",
        description="Растеризует текстослойные PDF в «сканы» и пишет эталон.",
    )
    ap.add_argument("--src-dir", required=True, type=Path,
                    help="каталог с исходными текстослойными PDF")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="каталог для скан-PDF")
    ap.add_argument("--gt-dir", required=True, type=Path,
                    help="каталог для эталонных <name>.txt")
    ap.add_argument("--dpi", type=int, default=200, help="dpi растеризации")
    ap.add_argument("--degrade", choices=sorted(_LEVELS), default="light",
                    help="уровень деградации")
    ap.add_argument("--seed", type=int, default=42, help="seed генератора шума")
    args = ap.parse_args(argv)

    if not args.src_dir.is_dir():
        print(f"ошибка: src-dir не существует: {args.src_dir}", file=sys.stderr)
        return 2
    pdfs = sorted(args.src_dir.glob("*.pdf"))
    if not pdfs:
        print(f"ошибка: в {args.src_dir} нет *.pdf", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.gt_dir.mkdir(parents=True, exist_ok=True)

    for src in pdfs:
        out_pdf = args.out_dir / src.name
        gt_path = args.gt_dir / f"{src.stem}.txt"
        process_pdf(src, out_pdf, gt_path, args.dpi, args.degrade, args.seed)
        print(f"{src.name} -> {out_pdf.name} + {gt_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
