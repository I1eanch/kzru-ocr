# Эксплуатация kzru-ocr

Документ для развёртывания и сопровождения Docker-образов сервиса OCR
PDF-сканов (RU/KK). Сборка частично воспроизводима: базовый образ закреплён
по digest, версии Python-зависимостей — по lock-файлу, языковые модели — по
sha256. Не зафиксированы версии системных пакетов Debian и хеши колёс —
см. раздел «Зафиксированные версии».

## Таргеты образов

`docker/Dockerfile` — multi-stage, три таргета:

| Таргет    | Назначение | Состав |
|-----------|------------|--------|
| `runtime` | production | `ocr/`, `service/`, `models/`, зависимости из `requirements.lock`. Без `bench/`, без тестов, без PyMuPDF/jiwer/pytest/httpx, без curl. |
| `dev`     | разработка и CI | `runtime` + `requirements-dev.txt` (PyMuPDF, jiwer, httpx, pytest) + `bench/` + `tests_*.py`. |
| `full`    | runtime + PaddleOCR | `dev` + `requirements-paddle.txt` + libGL + прогретые модели Paddle. Только amd64: PyPI не отдаёт колёса paddlepaddle под arm64. |

Сборка:

```bash
docker build --target runtime -t kzru-ocr:latest -f docker/Dockerfile .
docker build --target dev     -t kzru-ocr:dev    -f docker/Dockerfile .
docker build --platform linux/amd64 --target full -t kzru-ocr:full -f docker/Dockerfile .
```

Фактические размеры (arm64, 2026-09-22): `runtime` — 806 МБ, `dev` — 930 МБ.

## Зафиксированные версии

- Базовый образ: `python:3.12-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e` (Python 3.12.14).
- Python-пакеты production: `requirements.lock` (полный `pip freeze`,
  включая транзитивные). Прямые зависимости продублированы в
  `requirements.txt` с `==`-пинами.
- Dev-зависимости: `requirements-dev.txt` (`==`-пины).
- Paddle-таргет: `requirements-paddle.txt` (`paddlepaddle==3.0.0`,
  `paddleocr==3.0.0`; транзитивные зависимости не закреплены — отдельного
  lock для этого таргета нет). Сам таргет нужен только для экспериментов:
  по bake-off PaddleOCR ошибается на цифрах в 6–11 раз чаще Tesseract,
  поэтому в дефолтный путь он не входит.
- Системные пакеты Debian (tesseract-ocr 5.3.0, poppler-utils и пр.)
  **не закреплены**: `apt-get update` в Dockerfile тянет текущее состояние
  репозитория bookworm, и digest базового образа их версии не фиксирует —
  база фиксирует только саму базу. Точный список установленного —
  `docker run --rm <образ> dpkg -l`.
- Хеши wheel-файлов **не зафиксированы**: `requirements.lock` содержит
  версии, но установка идёт без `--require-hashes`, поэтому подмена колеса
  той же версии сборкой не отсекается.

### Как довести сборку до полной герметичности

Сейчас не реализовано; рецепт для будущего изменения:

1. Системные пакеты: либо перейти на snapshot.debian.org с зафиксированной
   датой снапшота в `sources.list`, либо прописать точные версии в
   `apt-get install` (`tesseract-ocr=5.3.0-2` и т.д. по выводу `dpkg -l`
   эталонной сборки).
2. Python-зависимости: перегенерировать lock с хешами
   (`pip-compile --generate-hashes` или `pip freeze` + `pip hash`) и ставить
   через `pip install --require-hashes -r requirements.lock`.

## Контрольные суммы моделей

`tessdata_best` (тег 4.1.0, скачивается при сборке, проверяется
`sha256sum -c` — сборка падает при несовпадении):

| Файл | sha256 |
|------|--------|
| `rus.traineddata` | `b617eb6830ffabaaa795dd87ea7fd251adfe9cf0efe05eb9a2e8128b7728d6b6` |
| `kaz.traineddata` | `34cbd9204b1ff3cc813d50b29e3c0ae3752bcc201f261a4a393ceca3895aea9d` |
| `osd.traineddata` | `9cf5d576fcc47564f11265841e5ca839001e7e6f38ff7f7aacf46d15a96b00ff` |

Дообученная модель из репозитория (`models/kzru_doc.traineddata`,
копируется в `/opt/tessdata_best/`):

```
1098082e374c42fcfebb2fed39484f679376d81352aa09591dfe21509565acc9
```

Проверка внутри образа:

```bash
docker run --rm kzru-ocr:latest sha256sum /opt/tessdata_best/kzru_doc.traineddata
```

## Запуск production-контейнера

Рекомендуемая команда (все флаги проверены фактически на arm64):

```bash
docker run -d --name kzru-ocr \
  --read-only --tmpfs /tmp \
  --pids-limit 256 --cpus 2 --memory 2g \
  --security-opt no-new-privileges \
  -p 127.0.0.1:8099:8099 \
  kzru-ocr:latest
```

Проверено в этой конфигурации: `/readyz` отвечает `{"status":"ready"}`,
`POST /ocr` распознаёт PDF, процесс работает от `uid=10001 (ocr)`,
HEALTHCHECK переходит в `healthy`.

Пояснения:

- `--read-only --tmpfs /tmp` — корневая ФС только для чтения; временные
  файлы (загрузки, растеризация poppler, tesseract) пишутся в tmpfs `/tmp`.
  Ограничение: для таргета `full` PaddleOCR пишет кэш в `$HOME/.paddleocr`
  — при `--read-only` добавьте `--tmpfs /home/ocr` (на arm64 не проверялось,
  full собирается только на amd64).
- `--pids-limit 256` — защита от fork-бомб; пайплайн порождает процессы
  tesseract/poppler, лимит с запасом.
- `--cpus 2 --memory 2g` — подберите под профиль нагрузки; 2 ГБ достаточно
  для профиля balanced на документах до 30 страниц.
- `--security-opt no-new-privileges` — процесс и так не root; флаг
  запрещает эскалацию через setuid-бинарники.
- Порт публикуйте на loopback (`127.0.0.1:8099`), если сервис вызывается
  только с этого хоста (PHP-FPM на той же машине).

CLI-режим (пакетная обработка без сервиса):

```bash
mkdir -p "$PWD/out"
docker run --rm --read-only --tmpfs /tmp \
  --user "$(id -u):$(id -g)" \
  -v "$PWD/scans:/data:ro" -v "$PWD/out:/out" \
  kzru-ocr:latest python -m ocr.cli --in-dir /data --out-dir /out
```

Про `--user` и права на bind-mount: контейнер работает от `uid=10001 (ocr)`,
поэтому каталог, в который он пишет, обязан быть доступен этому uid. На
Linux без `--user "$(id -u):$(id -g)"` запись в смонтированный каталог
упирается в `Permission denied`; на macOS Docker Desktop это не проявляется
(файлы отображаются на владельца хоста), поэтому проблема легко ускользает
при локальной проверке. Альтернатива `--user` — заранее отдать каталог
uid 10001: `sudo chown -R 10001:10001 out` (или `chmod 777 out`).
`--user` удобнее тем, что результаты сразу принадлежат вам.

## Граница размера запроса и обратный прокси

Приложение отвергает тело больше `MAX_REQUEST_BYTES` (16 777 216 байт =
`MAX_BYTES` 15 МБ + 1 МБ на multipart-обёртку, `service/app.py`), но только
по заголовку `Content-Length`. При `Transfer-Encoding: chunked` этого
заголовка нет: размер заранее неизвестен, и на уровне приложения дешёвого
способа отказать до приёма тела нет — **границу обязан ставить обратный
прокси**.

> [!WARNING]
> Без прокси и при chunked-загрузке тело запроса принимается целиком до
> проверки размера файла: лимит сработает только после того, как тело уже
> прочитано и разобрано.

Минимальный конфиг nginx перед сервисом (порт опубликован на loopback,
как в команде выше):

```nginx
server {
    listen 8080;

    # то же значение, что MAX_REQUEST_BYTES в service/app.py
    # (16m = 16777216 байт); держите их равными при изменении лимита
    client_max_body_size 16m;

    location / {
        proxy_pass http://127.0.0.1:8099;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;

        # OCR 30-страничного документа занимает десятки секунд;
        # дефолтные 60 с оборвут долгий запрос
        proxy_connect_timeout 10s;
        proxy_send_timeout    120s;
        proxy_read_timeout    300s;
        client_body_timeout   120s;
    }
}
```

nginx буферизует тело запроса и обрывает приём на `client_max_body_size`
с ответом `413` ещё до того, как тело уйдёт в сервис — это закрывает и
chunked-загрузки.

## Healthcheck и пробы

- `GET /healthz` — liveness: процесс жив, без внешних проверок.
- `GET /readyz` — readiness (fail-closed): проверяет tesseract, языковые
  модели, бэкенды PDF; 503, если чего-то не хватает. Именно этот endpoint
  используется в `HEALTHCHECK` образа (python urllib, curl в образе нет).

## Обновление зависимостей

1. Поднимите версии в `requirements.txt` (только `==`-пины).
2. Перегенерируйте lock в чистом базовом образе:

   ```bash
   docker run --rm -v "$PWD":/app -w /app \
     python:3.12-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e \
     sh -c 'pip install --no-cache-dir -r requirements.txt && pip freeze' \
     > requirements.lock
   ```

   (при смене базового образа сначала обновите digest: `docker pull
   python:3.12-slim-bookworm && docker inspect ... --format '{{index
   .RepoDigests 0}}'` и впишите его в `docker/Dockerfile` и сюда).
3. Dev-зависимости пинуются аналогично: установить в чистый контейнер,
   взять версии из `pip freeze`.
4. При обновлении `TESSDATA_BEST_REF` пересчитайте sha256 трёх файлов и
   обновите таблицу выше и `printf`-блок в Dockerfile.
5. Соберите `runtime` и `dev`, прогоните `scripts/smoke.sh kzru-ocr:latest`
   и `docker run --rm kzru-ocr:dev python tests_domain.py` +
   `python -m pytest tests_service.py -q`.

## Дымовая проверка

```bash
scripts/smoke.sh kzru-ocr:latest
```

Проверяет состав образа (нет dev-зависимостей и `bench/`, не root),
liveness/readiness, контракт полей, лимиты входа (413/422/400), очередь
заданий и отсутствие traceback в журнале.
