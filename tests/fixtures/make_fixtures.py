"""Генератор отслеживаемых тестовых сканов.

Тесты и дымовая проверка не должны зависеть от `bench/scans/`: этот каталог
в `.gitignore`, и в свежем клоне его нет — прогон молча опирался на локальные
артефакты, а доказательство «тесты проходят» становилось невоспроизводимым.

Поэтому два PDF лежат в репозитории рядом с этим скриптом, а скрипт позволяет
пересоздать их байт в байт: генератор детерминирован по seed.

    python3 tests/fixtures/make_fixtures.py

Рядом с каждым PDF пишется `.fields.json` — независимый эталон полей. Он взят
из генератора, который РИСОВАЛ документ, а не из того же экстрактора, которым
разбирается предсказание; иначе оценка полей замыкается сама на себя и всегда
показывает единицу.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Один документ на одну страницу и один на три: этого хватает, чтобы отличить
# полный отказ распознавания от сбоя отдельной страницы.
FIXTURES = {"one_page.pdf": 1, "three_pages.pdf": 3}

DPI = 200
DEGRADE = "medium"
SEED = 7


def main() -> int:
    import random

    from bench.make_pseudoscan import process_pdf as pseudoscan
    from bench.make_sample_docs import _find_font, make_doc

    font = _find_font()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for name, want_pages in FIXTURES.items():
            # Число страниц у генератора случайное (1-3), поэтому seed
            # подбирается до нужного объёма. Перебор детерминирован.
            for seed in range(SEED, SEED + 200):
                doc, truth = make_doc(random.Random(seed), "dogovor", font)
                if doc.page_count == want_pages:
                    break
                doc.close()
            else:
                raise SystemExit(f"не удалось получить документ на {want_pages} стр.")

            clean = tmp_dir / f"clean_{name}"
            doc.save(str(clean))
            doc.close()

            pseudoscan(clean, tmp_dir / name, tmp_dir / f"{name}.gt.txt",
                       dpi=DPI, degrade=DEGRADE, seed=SEED)

            shutil.copy(tmp_dir / name, HERE / name)
            (HERE / name).with_suffix(".fields.json").write_text(
                json.dumps(truth, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"{name}: {want_pages} стр., seed {seed}, "
                  f"{(HERE / name).stat().st_size} байт")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
