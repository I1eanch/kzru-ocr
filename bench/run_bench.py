"""CLI бенчмарка: сравнение OCR-вывода с эталонными текстами.

Использование::

    python -m bench.run_bench --pred-dir out/ --gt-dir bench/gt/ \
        [--csv report.csv] [--json report.json] [--worst 10]

Сопоставление файлов — по basename без расширения: ``<name>.txt`` в обеих
папках, в pred допускается также ``<name>.pdf.txt``. Эталон без предсказания
не пропускается: документ получает CER=1.0 и пометку ``MISSING``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import zip_longest
from pathlib import Path

from bench.metrics import (
    DocMetrics,
    FieldStat,
    cer,
    compare_fields,
    normalize_for_metric,
    wer,
)

# Порядок типов полей в отчётах.
_FIELD_KEYS = ("bin", "amounts", "dates")


def _pred_stem(p: Path) -> str:
    """Имя документа по pred-файлу: ``a.pdf.txt`` и ``a.txt`` → ``a``."""
    name = p.name
    if name.endswith(".pdf.txt"):
        return name[: -len(".pdf.txt")]
    return p.stem


def _collect(gt_dir: Path, pred_dir: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    gt = {p.stem: p for p in sorted(gt_dir.glob("*.txt"))}
    pred: dict[str, Path] = {}
    for p in sorted(pred_dir.glob("*.txt")):
        pred.setdefault(_pred_stem(p), p)
    return gt, pred


def evaluate(name: str, gt_text: str, pred_text: str) -> DocMetrics:
    """Считает все метрики одного документа."""
    gt_s = normalize_for_metric(gt_text, "strict")
    pr_s = normalize_for_metric(pred_text, "strict")
    gt_n = normalize_for_metric(gt_text)
    pr_n = normalize_for_metric(pred_text)
    return DocMetrics(
        name=name,
        cer_strict=cer(gt_s, pr_s),
        wer_strict=wer(gt_s, pr_s),
        cer_norm=cer(gt_n, pr_n),
        wer_norm=wer(gt_n, pr_n),
        field_stats=compare_fields(gt_text, pred_text),
        chars_gt=len(gt_n),
        chars_pred=len(pr_n),
    )


def _first_diff(gt_n: str, pr_n: str) -> tuple[int, str, str] | None:
    """Первая различающаяся строка (1-based) в normalized-текстах."""
    for i, (g, p) in enumerate(
        zip_longest(gt_n.split("\n"), pr_n.split("\n"), fillvalue=""), start=1
    ):
        if g != p:
            return i, g, p
    return None


def _aggregate(docs: list[DocMetrics]) -> dict:
    """Агрегаты: микро-CER (взвешен по символам эталона) и макро-средние."""
    total_chars = sum(d.chars_gt for d in docs)
    n = len(docs)

    def micro(attr: str) -> float:
        if not total_chars:
            return 0.0
        return sum(getattr(d, attr) * d.chars_gt for d in docs) / total_chars

    def macro(attr: str) -> float:
        return sum(getattr(d, attr) for d in docs) / n if n else 0.0

    fields: dict[str, FieldStat] = {}
    for d in docs:
        for key, st in d.field_stats.items():
            agg = fields.setdefault(key, FieldStat())
            agg.expected += st.expected
            agg.found += st.found
            agg.correct += st.correct

    return {
        "docs": n,
        "chars_gt": total_chars,
        "micro": {"cer_strict": micro("cer_strict"), "cer_norm": micro("cer_norm")},
        "macro": {
            "cer_strict": macro("cer_strict"),
            "wer_strict": macro("wer_strict"),
            "cer_norm": macro("cer_norm"),
            "wer_norm": macro("wer_norm"),
        },
        "fields": fields,
    }


def _field_dict(st: FieldStat) -> dict:
    return {
        "expected": st.expected,
        "found": st.found,
        "correct": st.correct,
        "precision": st.precision,
        "recall": st.recall,
        "exact": st.exact,
    }


def _print_report(
    docs: list[DocMetrics],
    missing: set[str],
    agg: dict,
    norm_texts: dict[str, tuple[str, str]],
    worst: int,
) -> None:
    print(f"{'document':<32} {'CER':>7} {'WER':>7} {'CERn':>7} {'WERn':>7} {'chars':>7}  flag")
    print("-" * 82)
    for d in docs:
        flag = "MISSING" if d.name in missing else ""
        print(
            f"{d.name:<32} {d.cer_strict:>7.4f} {d.wer_strict:>7.4f} "
            f"{d.cer_norm:>7.4f} {d.wer_norm:>7.4f} {d.chars_gt:>7}  {flag}"
        )
    print("-" * 82)
    print(
        f"docs: {agg['docs']}  (missing pred: {len(missing)})  "
        f"chars_gt: {agg['chars_gt']}"
    )
    print(
        f"micro CER (взвешен по символам эталона): "
        f"strict={agg['micro']['cer_strict']:.4f}  norm={agg['micro']['cer_norm']:.4f}"
    )
    m = agg["macro"]
    print(
        f"macro mean: CER strict={m['cer_strict']:.4f}  WER strict={m['wer_strict']:.4f}  "
        f"CER norm={m['cer_norm']:.4f}  WER norm={m['wer_norm']:.4f}"
    )

    print("\nfield accuracy:")
    print(f"{'field':<10} {'expected':>8} {'found':>6} {'correct':>7} {'P':>7} {'R':>7} {'exact':>7}")
    for key in _FIELD_KEYS:
        st = agg["fields"].get(key)
        if st is None:
            continue
        print(
            f"{key:<10} {st.expected:>8} {st.found:>6} {st.correct:>7} "
            f"{st.precision:>7.4f} {st.recall:>7.4f} {st.exact:>7.4f}"
        )

    if worst > 0:
        ranked = sorted(docs, key=lambda d: d.cer_norm, reverse=True)[:worst]
        print(f"\nworst {len(ranked)} по cer_norm:")
        for d in ranked:
            print(f"  {d.name}  cer_norm={d.cer_norm:.4f}")
            diff = _first_diff(*norm_texts[d.name])
            if diff is not None:
                lineno, g, p = diff
                print(f"    line {lineno}:")
                print(f"      gt:   {g[:120]}")
                print(f"      pred: {p[:120]}")


def _write_csv(path: Path, docs: list[DocMetrics], missing: set[str]) -> None:
    cols = [
        "name", "missing",
        "cer_strict", "wer_strict", "cer_norm", "wer_norm",
        "chars_gt", "chars_pred",
    ]
    for key in _FIELD_KEYS:
        cols += [f"{key}_expected", f"{key}_found", f"{key}_correct"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for d in docs:
            row = [
                d.name, int(d.name in missing),
                f"{d.cer_strict:.6f}", f"{d.wer_strict:.6f}",
                f"{d.cer_norm:.6f}", f"{d.wer_norm:.6f}",
                d.chars_gt, d.chars_pred,
            ]
            for key in _FIELD_KEYS:
                st = d.field_stats.get(key, FieldStat())
                row += [st.expected, st.found, st.correct]
            w.writerow(row)


def _write_json(
    path: Path,
    docs: list[DocMetrics],
    missing: set[str],
    agg: dict,
) -> None:
    payload = {
        "documents": [
            {
                "name": d.name,
                "missing": d.name in missing,
                "cer_strict": d.cer_strict,
                "wer_strict": d.wer_strict,
                "cer_norm": d.cer_norm,
                "wer_norm": d.wer_norm,
                "chars_gt": d.chars_gt,
                "chars_pred": d.chars_pred,
                "fields": {k: _field_dict(v) for k, v in d.field_stats.items()},
            }
            for d in docs
        ],
        "aggregate": {
            "docs": agg["docs"],
            "missing": len(missing),
            "chars_gt": agg["chars_gt"],
            "micro": agg["micro"],
            "macro": agg["macro"],
            "fields": {k: _field_dict(v) for k, v in agg["fields"].items()},
        },
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="bench.run_bench",
        description="Сравнивает OCR-вывод (pred) с эталоном (gt) и печатает метрики.",
    )
    ap.add_argument("--pred-dir", required=True, type=Path,
                    help="каталог с <name>.txt или <name>.pdf.txt предсказаниями")
    ap.add_argument("--gt-dir", required=True, type=Path,
                    help="каталог с эталонными <name>.txt")
    ap.add_argument("--csv", type=Path, default=None,
                    help="куда записать per-document CSV")
    ap.add_argument("--json", type=Path, default=None,
                    help="куда записать полный машинно-читаемый отчёт")
    ap.add_argument("--worst", type=int, default=0, metavar="N",
                    help="показать N худших документов по cer_norm с первым расхождением")
    args = ap.parse_args(argv)

    if not args.gt_dir.is_dir():
        print(f"ошибка: gt-dir не существует: {args.gt_dir}", file=sys.stderr)
        return 2
    if not args.pred_dir.is_dir():
        print(f"ошибка: pred-dir не существует: {args.pred_dir}", file=sys.stderr)
        return 2

    gt_files, pred_files = _collect(args.gt_dir, args.pred_dir)
    if not gt_files:
        print(f"ошибка: в {args.gt_dir} нет эталонных *.txt", file=sys.stderr)
        return 2

    extra = sorted(set(pred_files) - set(gt_files))
    if extra:
        print(
            "предупреждение: pred без эталона (игнорируются): " + ", ".join(extra),
            file=sys.stderr,
        )

    docs: list[DocMetrics] = []
    missing: set[str] = set()
    norm_texts: dict[str, tuple[str, str]] = {}
    for name in sorted(gt_files):
        gt_text = gt_files[name].read_text(encoding="utf-8")
        pred_path = pred_files.get(name)
        if pred_path is None:
            missing.add(name)
            pred_text = ""
        else:
            pred_text = pred_path.read_text(encoding="utf-8")
        docs.append(evaluate(name, gt_text, pred_text))
        norm_texts[name] = (
            normalize_for_metric(gt_text),
            normalize_for_metric(pred_text),
        )

    agg = _aggregate(docs)
    _print_report(docs, missing, agg, norm_texts, args.worst)

    if args.csv:
        _write_csv(args.csv, docs, missing)
        print(f"\nCSV записан: {args.csv}")
    if args.json:
        _write_json(args.json, docs, missing, agg)
        print(f"JSON записан: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
