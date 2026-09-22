"""Bake-off движков и настроек по числам, а не по вкусу.

Сравнивает конфигурации на одном наборе псевдосканов и печатает CER, ошибку в
цифрах и секунды на страницу. Решение об основном движке принимается по этой
таблице.

Отдельная метрика digit-CER считается потому, что в требованиях ошибка в
цифре недопустима, а в общем CER цифры растворяются: их доля в тексте мала, и
движок с худшим CER может оказаться лучшим на числах.

    python -m bench.bakeoff --scans bench/scans --gt bench/gt
    python -m bench.bakeoff --scans bench/scans --gt bench/gt --only tesseract
"""

from __future__ import annotations

import argparse
import re
import time
from dataclasses import replace
from pathlib import Path

from bench.metrics import cer, normalize_for_metric
from ocr import raster
from ocr.pipeline import PIPELINE_PROFILES, PipelineProfile, process_pdf
from ocr.render import render_document

_DIGITS_RE = re.compile(r"\d+")


def digit_cer(gt: str, pred: str) -> float:
    """CER по последовательности только цифровых групп."""
    gt_digits = " ".join(_DIGITS_RE.findall(gt))
    pred_digits = " ".join(_DIGITS_RE.findall(pred))
    return cer(gt_digits, pred_digits)


def tesseract_matrix() -> list[PipelineProfile]:
    out: list[PipelineProfile] = []
    for tessdata in ("best", "fast"):
        for psm in (4, 6):
            for dpi in (300, 400):
                out.append(
                    PipelineProfile(
                        name=f"tess-{tessdata}-psm{psm}-{dpi}dpi",
                        engines=("tesseract",),
                        tessdata=tessdata,
                        psm=psm,
                        base_dpi=dpi,
                    )
                )
    return out


def paddle_matrix() -> list[PipelineProfile]:
    return [
        PipelineProfile(
            name=f"paddle-kk-side{side}-{dpi}dpi",
            engines=("paddle",),
            digit_pass=False,
            base_dpi=dpi,
            paddle_limit=side,
        )
        for side in (1280, 2000, 2500)
        for dpi in (300,)
    ]


def finetuned_matrix() -> list[PipelineProfile]:
    """Дообученная модель против stock на тех же документах.

    BCER, который печатает обучение, считается на той же синтетике, которой
    модель училась, и качеством на документах не является. Решение о замене
    stock-модели принимается только по этой таблице.
    """
    out: list[PipelineProfile] = []
    for lang in ("kzru_doc", "kzru_doc+rus"):
        for psm in (4, 6):
            out.append(
                PipelineProfile(
                    name=f"ft-{lang}-psm{psm}-300dpi",
                    engines=("tesseract",),
                    tessdata="best",
                    lang=lang,
                    psm=psm,
                    base_dpi=300,
                )
            )
    return out


def ensemble_matrix() -> list[PipelineProfile]:
    return [
        PipelineProfile(
            name="ensemble-tess+paddle",
            engines=("tesseract", "paddle"),
            tessdata="best",
            psm=6,
            base_dpi=300,
        )
    ]


def evaluate(profile: PipelineProfile, scans: list[Path], gt_dir: Path, render: str) -> dict:
    PIPELINE_PROFILES[profile.name] = profile

    total_pages = 0
    elapsed = 0.0
    cers: list[tuple[float, int]] = []
    digit_cers: list[float] = []
    failures = 0

    for pdf in scans:
        gt_path = gt_dir / f"{pdf.stem}.txt"
        if not gt_path.exists():
            continue
        gt_text = gt_path.read_text(encoding="utf-8")

        started = time.perf_counter()
        try:
            # workers=1: сравниваем качество и стоимость страницы, а не
            # масштабирование — иначе конкуренция за ядра искажает секунды.
            doc = process_pdf(str(pdf), profile=profile.name, render=render, workers=1)
        except Exception as exc:
            failures += 1
            print(f"    [fail] {pdf.name}: {type(exc).__name__}: {exc}")
            continue
        elapsed += time.perf_counter() - started
        total_pages += len(doc.pages)

        pred_text = render_document(doc, render)
        gt_norm = normalize_for_metric(gt_text)
        pred_norm = normalize_for_metric(pred_text)
        cers.append((cer(gt_norm, pred_norm), len(gt_norm)))
        digit_cers.append(digit_cer(gt_text, pred_text))

    chars = sum(n for _, n in cers) or 1
    micro = sum(value * n for value, n in cers) / chars
    digits = sum(digit_cers) / len(digit_cers) if digit_cers else float("nan")

    return {
        "profile": profile.name,
        "cer_norm": micro,
        "digit_cer": digits,
        "s_per_page": elapsed / total_pages if total_pages else float("nan"),
        "pages": total_pages,
        "failures": failures,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bake-off движков OCR")
    parser.add_argument("--scans", default="bench/scans")
    parser.add_argument("--gt", default="bench/gt")
    parser.add_argument("--render", default="cells")
    parser.add_argument(
        "--only",
        choices=["tesseract", "paddle", "ensemble", "finetuned"],
        action="append",
        help="какие группы конфигураций прогонять",
    )
    parser.add_argument("--limit", type=int, default=0, help="взять только N документов")
    args = parser.parse_args(argv)

    scans = sorted(Path(args.scans).glob("*.pdf"))
    if args.limit:
        scans = scans[: args.limit]
    if not scans:
        parser.error(f"не найдено PDF в {args.scans}")

    groups = args.only or ["tesseract", "paddle", "ensemble"]
    profiles: list[PipelineProfile] = []
    if "tesseract" in groups:
        profiles += tesseract_matrix()
    if "paddle" in groups:
        profiles += paddle_matrix()
    if "ensemble" in groups:
        profiles += ensemble_matrix()
    if "finetuned" in groups:
        profiles += finetuned_matrix()

    print(f"документов: {len(scans)}, конфигураций: {len(profiles)}, render={args.render}\n")
    header = f"{'конфигурация':<26}{'CER norm':>10}{'digit CER':>11}{'с/стр':>8}{'стр':>6}{'fail':>6}"
    print(header)
    print("-" * len(header))

    rows = []
    for profile in profiles:
        row = evaluate(profile, scans, Path(args.gt), args.render)
        rows.append(row)
        print(
            f"{row['profile']:<26}{row['cer_norm']:>10.4f}{row['digit_cer']:>11.4f}"
            f"{row['s_per_page']:>8.2f}{row['pages']:>6}{row['failures']:>6}"
        )

    print("\nлучшие:")
    usable = [r for r in rows if r["pages"]]
    if usable:
        by_cer = min(usable, key=lambda r: r["cer_norm"])
        by_digits = min(usable, key=lambda r: r["digit_cer"])
        by_speed = min(usable, key=lambda r: r["s_per_page"])
        print(f"  по CER:     {by_cer['profile']} ({by_cer['cer_norm']:.4f})")
        print(f"  по цифрам:  {by_digits['profile']} ({by_digits['digit_cer']:.4f})")
        print(f"  по скорости:{by_speed['profile']} ({by_speed['s_per_page']:.2f} с/стр)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
