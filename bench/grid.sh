#!/usr/bin/env sh
# Подбор профиля вывода под конвенцию эталона.
#
# Формат эталона заказчика заранее неизвестен: чем разделены ячейки таблиц,
# склеены ли переносы, сохранены ли колонтитулы. Вместо угадывания прогоняем
# решётку профилей и смотрим, какой ближе к их разметке. Распознавание при
# этом не меняется — меняется только сериализация.
#
#   sh bench/grid.sh bench/scans bench/gt balanced
set -eu

SCANS="${1:-bench/scans}"
GT="${2:-bench/gt}"
OCR_PROFILE="${3:-balanced}"
RENDERS="${RENDERS:-default cells pipe space joined no-headers flat}"

printf '%-12s %10s %10s %10s %10s\n' render CERstrict CERnorm WERstrict WERnorm

for r in $RENDERS; do
    out="bench/pred_$r"
    python -m ocr.cli --in-dir "$SCANS" --out-dir "$out" \
        --profile "$OCR_PROFILE" --render "$r" >/dev/null 2>&1

    python -m bench.run_bench --pred-dir "$out" --gt-dir "$GT" \
        --json "bench/grid_$r.json" >/dev/null 2>&1

    python - "$r" "bench/grid_$r.json" <<'PY'
import json
import sys

name, path = sys.argv[1], sys.argv[2]
with open(path, encoding="utf-8") as fh:
    report = json.load(fh)
agg = report.get("aggregate", report)


def pick(*keys):
    for key in keys:
        if key in agg:
            return agg[key]
    return float("nan")


print(
    f"{name:<12} "
    f"{pick('micro_cer_strict', 'cer_strict'):>10.4f} "
    f"{pick('micro_cer_norm', 'cer_norm'):>10.4f} "
    f"{pick('macro_wer_strict', 'wer_strict'):>10.4f} "
    f"{pick('macro_wer_norm', 'wer_norm'):>10.4f}"
)
PY
done
