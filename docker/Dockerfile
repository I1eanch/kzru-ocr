# syntax=docker/dockerfile:1
#
# Три таргета:
#   runtime — production: Tesseract-ветка, только ocr/, service/, models/.
#             Собирается и на arm64, и на amd64.
#   dev     — runtime + bench/, тесты и dev-зависимости (PyMuPDF, jiwer,
#             httpx, pytest). Для разработки и CI, не для продакшена.
#   full    — dev + PaddleOCR. PyPI отдаёт paddlepaddle только для manylinux
#             x86_64, поэтому этот таргет собирается на amd64-хосте
#             (--platform linux/amd64).
#
# Training tools здесь сознательно отсутствуют: дообучение живёт в
# отдельном образе docker/Dockerfile.train, prod-образ только inference.
#
# Базовый образ закреплён по digest (python:3.12-slim-bookworm на 2026-09-22)
# — сборка воспроизводима, «подвижный» тег не подменит содержимое.
FROM python:3.12-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    TESSDATA_BEST=/opt/tessdata_best \
    TESSDATA_FAST=/usr/share/tesseract-ocr/5/tessdata

# tesseract-ocr-{rus,kaz,osd} из Debian — это tessdata_fast-класс модели,
# они остаются доступны для профиля fast. tessdata_best докачиваем ниже.
# poppler-utils (pdftoppm/pdftotext/pdfinfo/pdfimages) — растеризатор из
# разрешённого ТЗ стека и дефолтный бэкенд: внешние GPL-2 утилиты вместо
# линковки с AGPL-библиотекой.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-rus \
        tesseract-ocr-kaz \
        tesseract-ocr-osd \
        poppler-utils \
        libglib2.0-0 \
        fonts-dejavu-core \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# tessdata_best скачивается с GitHub; каждый файл проверяется по sha256 —
# сборка падает при несовпадении (подмена/битая загрузка). Суммы посчитаны
# по тегу 4.1.0 и сверены с содержимым предыдущего образа.
# curl нужен только на этом шаге — после скачивания он вычищается из образа.
ARG TESSDATA_BEST_REF=4.1.0
RUN set -eu; \
    mkdir -p "$TESSDATA_BEST"; \
    for lang in rus kaz osd; do \
      curl -fsSL -o "$TESSDATA_BEST/$lang.traineddata" \
        "https://github.com/tesseract-ocr/tessdata_best/raw/${TESSDATA_BEST_REF}/${lang}.traineddata"; \
    done; \
    cd "$TESSDATA_BEST"; \
    printf '%s  %s\n' \
      'b617eb6830ffabaaa795dd87ea7fd251adfe9cf0efe05eb9a2e8128b7728d6b6' 'rus.traineddata' \
      '34cbd9204b1ff3cc813d50b29e3c0ae3752bcc201f261a4a393ceca3895aea9d' 'kaz.traineddata' \
      '9cf5d576fcc47564f11265841e5ca839001e7e6f38ff7f7aacf46d15a96b00ff' 'osd.traineddata' \
      | sha256sum -c -; \
    apt-get purge -y curl && apt-get autoremove -y; \
    rm -rf /var/lib/apt/lists/*; \
    ls -la "$TESSDATA_BEST"

WORKDIR /app

# Ставим lock-файл: зафиксированы и прямые, и транзитивные зависимости.
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

COPY ocr ./ocr
COPY service ./service

# Дообученная модель kzru_doc поставляется в образе для экспериментальных
# профилей ft-balanced/ft-accurate. Дефолтом она НЕ является: профили
# fast/balanced/accurate работают на stock-моделях rus+kaz — по замеру
# (bench/bakeoff.py) дообученная проигрывает stock по CER и теряет один БИН.
# sha256: 1098082e374c42fcfebb2fed39484f679376d81352aa09591dfe21509565acc9
COPY models/kzru_doc.traineddata /opt/tessdata_best/

# Непривилегированный пользователь: сервис и CLI не должны работать от root.
# /data — точка монтирования входных файлов пользователя.
RUN useradd --create-home --uid 10001 ocr \
    && mkdir -p /data \
    && chown -R ocr:ocr /data /opt/tessdata_best

# PYTHONPATH прописан в образе, чтобы `python -m ocr.cli` работал из любого
# рабочего каталога: пользователь монтирует свои данные в /data и запускает
# команду оттуда, не думая про пути пакета.
ENV PYTHONPATH=/app
ENV TESSDATA_PREFIX=/opt/tessdata_best
EXPOSE 8099

USER ocr

# Readiness-проба без curl (curl удалён из образа после скачивания моделей).
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8099/readyz', timeout=4)"]

CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8099", "--workers", "1"]


FROM runtime AS dev

USER root
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

COPY bench ./bench
# Глоб: сборка не падает, пока есть хотя бы один tests_*.py.
COPY tests_*.py ./
# Фикстуры едут вместе с тестами: они отслеживаются git, и прогон в свежем
# клоне не должен зависеть от игнорируемого bench/scans.
COPY tests ./tests
USER ocr


FROM dev AS full

USER root
# PaddleOCR тянет полный opencv-python (не headless), которому нужен libGL.
# В runtime его нет сознательно: там headless-сборка и лишние 200 МБ не нужны.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-paddle.txt .
RUN pip install --no-cache-dir -r requirements-paddle.txt
USER ocr

# Прогрев: модели det+rec скачиваются в образ, чтобы первый запрос не ждал
# сети. Офлайн-требование распространяется и на первый вызов.
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(lang='kk', use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)"
