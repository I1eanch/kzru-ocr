"""Замер масштабирования по страницам.

Нужен отдельным скриптом, потому что на этом пайплайне параллелизм один раз
уже деградировал суперлинейно (1 ГБ временных массивов на страницу в sauvola
плюс собственный пул потоков OpenCV в каждом воркере). Любое изменение
препроцессинга следует проверять этим замером, а не рассуждением.

    python -m bench.measure_parallel --pdf bench/scans/doc_001_dogovor.pdf --repeat 8
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pymupdf

from ocr.pipeline import available_cpus, process_pdf


def build_multipage(src: str, repeat: int, dest: str) -> str:
    source = pymupdf.open(src)
    out = pymupdf.open()
    for _ in range(repeat):
        out.insert_pdf(source)
    out.save(dest)
    out.close()
    source.close()
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Масштабирование OCR по страницам")
    parser.add_argument("--pdf", default="bench/scans/doc_001_dogovor.pdf")
    parser.add_argument("--repeat", type=int, default=8, help="во сколько раз размножить страницы")
    parser.add_argument("--profile", default="balanced")
    parser.add_argument("--workers", default="1,2,3,4", help="список через запятую")
    parser.add_argument("--dest", default="/tmp/kzru_multipage.pdf")
    args = parser.parse_args(argv)

    if not Path(args.pdf).exists():
        parser.error(f"нет файла {args.pdf}")

    path = build_multipage(args.pdf, args.repeat, args.dest)
    workers = [int(w) for w in args.workers.split(",") if w.strip()]

    print(f"available_cpus(): {available_cpus()}  профиль: {args.profile}  файл: {path}")
    header = f"{'workers':>8}{'итого, с':>12}{'с/стр':>9}{'ускорение':>11}"
    print(header)
    print("-" * len(header))

    baseline: float | None = None
    for count in workers:
        started = time.perf_counter()
        doc = process_pdf(path, profile=args.profile, workers=count)
        elapsed = time.perf_counter() - started
        if baseline is None:
            baseline = elapsed
        pages = len(doc.pages) or 1
        print(f"{count:>8}{elapsed:>12.1f}{elapsed / pages:>9.2f}{baseline / elapsed:>10.2f}x")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
