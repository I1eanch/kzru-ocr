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
    TESSDATA_BEST=/opt/tessdata_best \
    TESSDATA_FAST=/usr/share/tesseract-ocr/5/tessdata

# tesseract-ocr-{rus,kaz,osd} из Debian — это tessdata_fast-класс модели,
# они остаются доступны для профиля fast. tessdata_best докачиваем ниже.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-rus \
        tesseract-ocr-kaz \
        tesseract-ocr-osd \
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

ENV TESSDATA_PREFIX=/opt/tessdata_best
EXPOSE 8099

CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8099", "--workers", "1"]


FROM runtime AS full

COPY requirements-paddle.txt .
RUN pip install --no-cache-dir -r requirements-paddle.txt

# Прогрев: скачиваем det+rec модели в образ, чтобы первый запрос не ждал сети.
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(lang='kk', use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)" || true
