"""CLI: распознать PDF или каталог PDF в текст.

    python -m ocr.cli scan.pdf
    python -m ocr.cli --in-dir bench/scans --out-dir out --profile balanced --render default
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

from .pipeline import PIPELINE_PROFILES, process_pdf
from .render import PROFILES, render_document


def _report(doc, path: Path) -> dict:
    return {
        "file": path.name,
        "profile": doc.profile,
        "pages": len(doc.pages),
        "elapsed_s": doc.elapsed_s,
        "mean_conf": round(doc.mean_conf, 4),
        "sources": {p.index: p.source for p in doc.pages},
        "fields": doc.fields,
        "warnings": [asdict(w) for w in doc.warnings],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Локальный OCR PDF-сканов RU/KK")
    parser.add_argument("pdf", nargs="?", help="путь к PDF")
    parser.add_argument("--in-dir", help="каталог с PDF (пакетный режим)")
    parser.add_argument("--out-dir", help="куда писать .txt (по умолчанию stdout)")
    parser.add_argument("--profile", default="balanced", choices=sorted(PIPELINE_PROFILES))
    parser.add_argument("--render", default="default", choices=sorted(PROFILES))
    parser.add_argument("--workers", type=int, default=None, help="процессов на страницы")
    parser.add_argument(
        "--engine",
        choices=["tesseract", "paddle"],
        help="переопределить движок профиля (paddle требует образ full)",
    )
    parser.add_argument(
        "--no-text-layer", action="store_true", help="игнорировать текстовый слой, всё через OCR"
    )
    parser.add_argument("--json", dest="json_path", help="куда писать отчёт с warnings и полями")
    args = parser.parse_args(argv)

    if not args.pdf and not args.in_dir:
        parser.error("нужен путь к PDF или --in-dir")

    targets: list[Path] = []
    if args.pdf:
        targets.append(Path(args.pdf))
    if args.in_dir:
        targets.extend(sorted(Path(args.in_dir).glob("*.pdf")))
    if not targets:
        parser.error("не найдено ни одного PDF")

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    profile_name = args.profile
    if args.engine:
        # Переопределение движка регистрируется отдельным профилем, а не
        # правкой существующего: базовые профили должны оставаться теми, для
        # которых опубликованы замеры bake-off.
        base = PIPELINE_PROFILES[args.profile]
        profile_name = f"{args.profile}+{args.engine}"
        PIPELINE_PROFILES[profile_name] = replace(base, name=profile_name, engines=(args.engine,))

    reports: list[dict] = []
    failed = 0

    for path in targets:
        try:
            doc = process_pdf(
                str(path),
                profile=profile_name,
                render=args.render,
                workers=args.workers,
                use_text_layer=not args.no_text_layer,
            )
        except Exception as exc:
            failed += 1
            print(f"[FAIL] {path.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue

        text = render_document(doc, args.render)
        if out_dir:
            (out_dir / f"{path.stem}.txt").write_text(text, encoding="utf-8")
            print(
                f"[ok] {path.name}: {len(doc.pages)} стр, {doc.elapsed_s}s, "
                f"conf={doc.mean_conf:.2f}, warnings={len(doc.warnings)}",
                file=sys.stderr,
            )
        else:
            print(text)
        reports.append(_report(doc, path))

    if args.json_path:
        Path(args.json_path).write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
