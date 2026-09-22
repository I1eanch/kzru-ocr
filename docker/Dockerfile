# syntax=docker/dockerfile:1
#
# Два таргета:
#   runtime (default) — Tesseract-ветка. Собирается и на arm64, и на amd64.
#   full              — + PaddleOCR. PyPI отдаёт paddlepaddle только для
#                       manylinux x86_64, поэтому этот таргет собирается
#                       на amd64-хосте (--platform linux/amd64).
#
# Training tools здесь сознательно отсутствуют: дообучение живёт в
# отдельном образе docker/Dockerfile.train, prod-образ только inference.

FROM python:3.12-slim-bookworm AS runtime

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

ARG TESSDATA_BEST_REF=4.1.0
RUN mkdir -p "$TESSDATA_BEST" \
    && for lang in rus kaz osd; do \
         curl -fsSL -o "$TESSDATA_BEST/$lang.traineddata" \
           "https://github.com/tesseract-ocr/tessdata_best/raw/${TESSDATA_BEST_REF}/${lang}.traineddata"; \
       done \
    && ls -la "$TESSDATA_BEST"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ocr ./ocr
COPY service ./service
COPY bench ./bench
COPY tests_domain.py ./

# Дообученная модель поставляется в образе: она выбрана по замеру
# (bench/bakeoff.py) и используется профилями по умолчанию.
COPY models/kzru_doc.traineddata /opt/tessdata_best/

# PYTHONPATH прописан в образе, чтобы `python -m ocr.cli` работал из любого
# рабочего каталога: пользователь монтирует свои данные в /data и запускает
# команду оттуда, не думая про пути пакета.
ENV PYTHONPATH=/app
ENV TESSDATA_PREFIX=/opt/tessdata_best
EXPOSE 8099

CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8099", "--workers", "1"]


FROM runtime AS full

# PaddleOCR тянет полный opencv-python (не headless), которому нужен libGL.
# В runtime его нет сознательно: там headless-сборка и лишние 200 МБ не нужны.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-paddle.txt .
RUN pip install --no-cache-dir -r requirements-paddle.txt

# Прогрев: модели det+rec скачиваются в образ, чтобы первый запрос не ждал
# сети. Офлайн-требование распространяется и на первый вызов.
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(lang='kk', use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)"
