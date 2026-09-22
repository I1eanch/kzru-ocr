#!/usr/bin/env sh
# Подбор профиля вывода под конвенцию эталона.
#
# Формат эталона заказчика заранее неизвестен: чем разделены ячейки таблиц,
# склеены ли переносы, сохранены ли колонтитулы. Вместо угадывания прогоняем
# решётку профилей и смотрим, какой ближе к их разметке. Распознавание при
# этом не меняется — меняется только сериализация.
#
#   sh bench/grid.sh bench/scans bench/gt balanced
#
# Промежуточные результаты (pred_*, grid_*.json) пишутся в OUT_DIR — по
# умолчанию bench/ рядом со скриптом. В контейнере каталог bench/
# принадлежит root и недоступен на запись uid 10001, поэтому там
# передавайте записываемый путь: OUT_DIR=/tmp/grid или точку монтирования.
set -eu

SCANS="${1:-bench/scans}"
GT="${2:-bench/gt}"
OCR_PROFILE="${3:-balanced}"
RENDERS="${RENDERS:-default cells pipe space joined no-headers flat}"
OUT_DIR="${OUT_DIR:-bench}"
mkdir -p "$OUT_DIR"

printf '%-12s %10s %10s %10s %10s\n' render CERstrict CERnorm WERstrict WERnorm

for r in $RENDERS; do
    out="$OUT_DIR/pred_$r"
    python -m ocr.cli --in-dir "$SCANS" --out-dir "$out" \
        --profile "$OCR_PROFILE" --render "$r" >/dev/null 2>&1

    python -m bench.run_bench --pred-dir "$out" --gt-dir "$GT" \
        --json "$OUT_DIR/grid_$r.json" >/dev/null 2>&1

    python - "$r" "$OUT_DIR/grid_$r.json" <<'PY'
import json
import sys

name, path = sys.argv[1], sys.argv[2]
with open(path, encoding="utf-8") as fh:
    agg = json.load(fh)["aggregate"]

micro, macro = agg["micro"], agg["macro"]
print(
    f"{name:<12} "
    f"{micro['cer_strict']:>10.4f} "
    f"{micro['cer_norm']:>10.4f} "
    f"{macro['wer_strict']:>10.4f} "
    f"{macro['wer_norm']:>10.4f}"
)
PY
done
