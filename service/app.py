"""Локальный HTTP-сервис для вызова из PHP (CodeIgniter 4).

Почему сервис, а не CLI на каждый файл: пул рабочих процессов переиспользуется
между запросами, и каждый документ не платит заново за старт воркеров
(замерено: 0.54 с на документ из двух страниц). Модели Tesseract при этом
резидентными не становятся — распознавание идёт вызовом внешнего бинаря,
который читает их заново; за повторное чтение отвечает страничный кэш ОС.

Контракт ответа включает `warnings` и статусы у каждого поля — это часть
продукта, а не отладочный вывод. Сервис проверки госдокументов должен знать,
где OCR не уверен: непрошедшая контрольная сумма БИН, исправленный номер,
расхождение суммы цифрами и прописью, низкая уверенность страницы.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from ocr import concurrency
from ocr.pdf_backend import DEFAULT_BACKEND, InvalidPdf, available_backends
from ocr.pipeline import MAX_BYTES, MAX_PAGES, PIPELINE_PROFILES, process_pdf
from ocr.render import PROFILES, render_document

log = logging.getLogger("kzru-ocr")

app = FastAPI(title="kzru-ocr", version="1.0.0")

# Границы очереди фоновых заданий. Без них словарь результатов и временные
# файлы растут, пока не кончится память или диск: клиент не обязан забирать
# результат, а сервис не имеет права ждать его вечно.
MAX_QUEUED_JOBS = int(os.environ.get("KZRU_MAX_QUEUED_JOBS", "8"))
JOB_TTL_SECONDS = float(os.environ.get("KZRU_JOB_TTL", "900"))
SLOT_TIMEOUT_SECONDS = float(os.environ.get("KZRU_SLOT_TIMEOUT", "30"))

# Профиль, применяемый без явного запроса. Одно имя на валидацию, проверку
# готовности и значения по умолчанию у ручек: рассинхрон этих трёх мест уже
# приводил к тому, что сервис рекламировал профиль, который отвергал.
DEFAULT_PROFILE = "balanced"
DEFAULT_RENDER = "default"

UPLOAD_CHUNK = 1024 * 1024

_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=max(1, concurrency.max_documents()))


# Запас поверх лимита самого файла: multipart добавляет границы и заголовки
# частей, поэтому тело запроса законно больше содержимого файла.
MAX_REQUEST_BYTES = MAX_BYTES + 1024 * 1024


@app.middleware("http")
async def _reject_oversized_body(request: Request, call_next):
    """Отказ по объявленному размеру тела — до разбора multipart.

    Без этого граница стоит слишком поздно: Starlette формирует `UploadFile`
    только после полного разбора multipart, то есть тело уже принято целиком
    и, превысив порог `SpooledTemporaryFile`, осело на диске. Проверка в
    `_save_upload` ограничивает лишь вторую копию, которую делает сервис, а
    не приём как таковой.

    Это заслон только для запросов с честным `Content-Length`. При
    `Transfer-Encoding: chunked` размер заранее неизвестен, и на уровне
    приложения дешёвого способа его узнать нет — там границу обязан ставить
    обратный прокси (`client_max_body_size`), см. docs/DEPLOY.md.
    """
    raw = request.headers.get("content-length")
    if raw and raw.isdigit() and int(raw) > MAX_REQUEST_BYTES:
        return JSONResponse(
            {"detail": f"тело запроса больше {MAX_REQUEST_BYTES} байт"},
            status_code=413,
        )
    return await call_next(request)


def _profile_names() -> list[str]:
    """Профили, которые сервис реально принимает.

    Единый источник и для валидации, и для readiness: раньше API объявлял
    жёсткий Literal из трёх имён, а `/healthz` рекламировал весь реестр, где
    профилей больше. Клиент видел имя, которое сервис отвергал.
    """
    return sorted(PIPELINE_PROFILES)


def _client_error(status: int, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail=message)


def _server_error(exc: Exception, context: str) -> HTTPException:
    """Наружу — идентификатор, в журнал — подробности.

    Тип и текст внутреннего исключения могут содержать пути файловой системы и
    детали окружения; клиенту они не нужны, а в журнале нужны обязательно.
    """
    error_id = uuid.uuid4().hex[:12]
    log.exception("[%s] %s: %s: %s", error_id, context, type(exc).__name__, exc)
    return HTTPException(status_code=500, detail=f"внутренняя ошибка, идентификатор {error_id}")


def _save_upload(upload: UploadFile) -> Path:
    """Пишет загрузку во временный файл, обрывая её на превышении лимита.

    Три вещи, которых не делала прежняя версия. Дескриптор от `mkstemp`
    закрывается — раньше он терялся, и каждый запрос стоил одного
    file descriptor (замерено: 20 утечек на 20 вызовов). Размер проверяется
    по ходу копирования, а не после: иначе клиент мог заставить сервис
    записать на диск файл любого размера, прежде чем получить 413. При любом
    исключении временный файл удаляется.
    """
    suffix = Path(upload.filename or "upload.pdf").suffix or ".pdf"
    if len(suffix) > 16 or "/" in suffix or "\\" in suffix:
        suffix = ".pdf"

    fd, name = tempfile.mkstemp(suffix=suffix, prefix="kzru-upload-")
    path = Path(name)
    written = 0
    try:
        with os.fdopen(fd, "wb") as fh:
            while True:
                chunk = upload.file.read(UPLOAD_CHUNK)
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_BYTES:
                    raise _client_error(413, f"файл больше {MAX_BYTES} байт")
                fh.write(chunk)
    except BaseException:
        path.unlink(missing_ok=True)
        raise

    if written == 0:
        path.unlink(missing_ok=True)
        raise _client_error(400, "пустой файл")
    return path


class RecognitionUnavailable(RuntimeError):
    """Ни одна страница не распозналась — сервис неисправен, а не документ плох."""


class JobCancelled(RuntimeError):
    """Задание отменено клиентом до начала работы."""


def _run(
    path: Path,
    name: str,
    profile: str,
    render: str,
    slot_timeout: float = SLOT_TIMEOUT_SECONDS,
    cancel: threading.Event | None = None,
) -> dict[str, Any]:
    """Обрабатывает документ, занимая слот общего бюджета.

    `slot_timeout` различается по способу вызова. Синхронный `/ocr` ждёт
    недолго: на том конце висит HTTP-клиент, и честнее быстро ответить 503.
    Принятое фоновое задание — наоборот: очередь существует ровно затем,
    чтобы поглощать всплески нагрузки, и ронять принятое задание из-за того,
    что слот занят полминуты, противоречит её смыслу. Поэтому фоновому
    заданию даётся весь его TTL.

    Ожидание прерываемое: пока задание стоит в очереди, клиент может его
    удалить, и продолжать ждать слот после этого незачем.
    """
    deadline = time.monotonic() + slot_timeout
    try:
        while True:
            if cancel is not None and cancel.is_set():
                raise JobCancelled("задание отменено до начала обработки")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise concurrency.CapacityExceeded(
                    f"нет свободных слотов обработки в течение {slot_timeout} с"
                )
            try:
                with concurrency.document_slot(timeout=min(1.0, remaining)):
                    return _process_in_slot(path, name, profile, render)
            except concurrency.CapacityExceeded:
                continue  # слот не освободился за шаг — проверяем отмену и ждём дальше
    finally:
        path.unlink(missing_ok=True)


def _process_in_slot(path: Path, name: str, profile: str, render: str) -> dict[str, Any]:
    doc = process_pdf(str(path), profile=profile, render=render)

    # Если распознавание не дало ни одной страницы, отдавать 200 с пустым
    # текстом нельзя: клиент, не разобравший `warnings`, запишет пустоту
    # как результат проверки документа. Такой отказ относится к сервису,
    # а не к входному файлу, поэтому наружу он уходит как 503.
    ocr_pages = [p for p in doc.pages if p.source == "ocr"]
    failed_pages = {w.page for w in doc.warnings if w.type == "page_failed"}
    if ocr_pages and len(failed_pages) >= len(ocr_pages):
        details = sorted({w.detail for w in doc.warnings if w.type == "page_failed"})
        raise RecognitionUnavailable("; ".join(details)[:300])

    return _document_payload(doc, name, render)


def _document_payload(doc, name: str, render: str) -> dict[str, Any]:
    return {
        "file": name,
        "profile": doc.profile,
        "render": render,
        "elapsed_s": doc.elapsed_s,
        "mean_conf": round(doc.mean_conf, 4),
        "text": render_document(doc, render),
        "pages": [
            {
                "index": p.index,
                "source": p.source,
                "engine": p.engine,
                "rotation": p.rotation,
                "skew": p.skew,
                "dpi": p.dpi,
                "mean_conf": round(p.mean_conf, 4),
            }
            for p in doc.pages
        ],
        "fields": doc.fields,
        "warnings": [asdict(w) for w in doc.warnings],
    }


def _validate_request(profile: str, render: str) -> None:
    if profile not in PIPELINE_PROFILES:
        raise _client_error(400, f"неизвестный профиль: {profile}; доступны {_profile_names()}")
    if render not in PROFILES:
        raise _client_error(400, f"неизвестный render-профиль: {render}; доступны {sorted(PROFILES)}")

    # Профиль существует, но его модели не на месте: это отказ сервиса, а не
    # ошибка клиента. 400 здесь врал бы — запрос корректен.
    problem = _profile_problem(PIPELINE_PROFILES[profile])
    if problem:
        raise HTTPException(
            status_code=503,
            detail=f"профиль {profile} недоступен: {problem}",
            headers={"Retry-After": "30"},
        )


# --------------------------------------------------------------------------
# Диагностика
# --------------------------------------------------------------------------


def _tessdata_dir(profile) -> str:
    """Каталог моделей, которым пользуется движок этого профиля.

    Важно спрашивать именно его, а не `TESSDATA_PREFIX`: движок запускается с
    `--tessdata-dir`, и это разные пути. Если проверять не тот каталог,
    readiness расходится с реальностью — проверено: при сломанном
    `TESSDATA_BEST` готовность отвечала 503, а распознавание продолжало
    работать, потому что Tesseract молча откатывался на пакетные модели
    Debian, то есть на другие веса, чем объявляет профиль.
    """
    from ocr.engines.tesseract import TESSDATA_BEST, TESSDATA_FAST

    return TESSDATA_BEST if profile.tessdata == "best" else TESSDATA_FAST


def _tesseract_state(tessdata_dir: str | None = None) -> tuple[str, list[str]]:
    binary = shutil.which("tesseract")
    if not binary:
        return "", []
    version_lines = subprocess.run(
        [binary, "--version"], capture_output=True, text=True, check=False
    ).stdout.splitlines()

    cmd = [binary, "--list-langs"]
    if tessdata_dir:
        cmd += ["--tessdata-dir", tessdata_dir]
    langs = [
        ln.strip()
        for ln in subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.splitlines()[1:]
        if ln.strip()
    ]
    return (version_lines[0] if version_lines else ""), langs


def _profile_problem(profile) -> str | None:
    """Причина, по которой профиль не отработает, либо None.

    Проверяется каталог моделей самого профиля, а не общий
    `TESSDATA_PREFIX`: движок запускается с `--tessdata-dir`, и это разные
    пути. При несуществующем каталоге Tesseract не сообщает об ошибке, а
    молча берёт модели из вкомпилированного пути — сервис работал бы на
    весах, которых профиль не объявлял.
    """
    tessdata_dir = _tessdata_dir(profile)
    required = [lang for lang in profile.lang.split("+") if lang]
    missing = [lang for lang in required if not Path(tessdata_dir, f"{lang}.traineddata").is_file()]
    if missing:
        return f"в каталоге {tessdata_dir} нет моделей {missing}"

    if "paddle" in getattr(profile, "engines", ()) :
        try:
            import paddleocr  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            return f"PaddleOCR недоступен: {type(exc).__name__}"
    return None


def _profiles_state() -> tuple[dict[str, str], list[str]]:
    """Готовые профили и причины отказа по остальным.

    Рекламировать имя, по которому сервис ответит 503, нельзя: клиент считает
    список из `/readyz` перечнем работающих режимов. Поэтому наружу уходят
    только проверенные профили, а забракованные — с причиной.
    """
    unavailable: dict[str, str] = {}
    ready: list[str] = []
    for name, profile in sorted(PIPELINE_PROFILES.items()):
        problem = _profile_problem(profile)
        if problem:
            unavailable[name] = problem
        else:
            ready.append(name)
    return unavailable, ready


def _readiness() -> tuple[bool, dict[str, Any]]:
    """Проверяет ровно то, без чего сервис не отработает."""
    problems: list[str] = []

    default_profile = PIPELINE_PROFILES[DEFAULT_PROFILE]
    tessdata_dir = _tessdata_dir(default_profile)
    version, langs = _tesseract_state(tessdata_dir)

    if not version:
        problems.append("бинарь tesseract не найден в PATH")

    required = [lang for lang in default_profile.lang.split("+") if lang]

    default_problem = _profile_problem(default_profile)
    if default_problem:
        problems.append(f"профиль по умолчанию нерабочий: {default_problem}")

    missing_langs = [lang for lang in required if lang not in langs]
    if missing_langs:
        problems.append(f"Tesseract не видит языки профиля по умолчанию: {missing_langs}")

    backends = available_backends()
    if DEFAULT_BACKEND not in backends:
        problems.append(f"PDF-бэкенд по умолчанию {DEFAULT_BACKEND} недоступен; есть {backends}")

    # Пул, исчерпавший попытки восстановления, — нерабочий runtime: каждая
    # многостраничная обработка будет падать. Рекламировать такой контейнер
    # готовым нельзя, иначе балансировщик продолжит слать на него нагрузку.
    if concurrency.pool_is_broken():
        problems.append("пул страничных воркеров сломан и не восстановился")

    unavailable, ready_profiles = _profiles_state()

    return not problems, {
        "tesseract": version,
        "langs": langs,
        "required_langs": required,
        "tessdata_dir": tessdata_dir,
        "pdf_backends": backends,
        "pdf_backend_default": DEFAULT_BACKEND,
        "profiles": ready_profiles,
        "profiles_unavailable": unavailable,
        "problems": problems,
    }


@app.get("/healthz")
def healthz() -> dict[str, str]:
    """Liveness: процесс жив и отвечает. Никаких внешних проверок."""
    return {"status": "alive"}


@app.get("/readyz")
def readyz() -> JSONResponse:
    """Readiness: fail-closed.

    Отдаёт 503, если отсутствует хоть что-то, без чего профиль по умолчанию не
    отработает. Прежняя версия возвращала `status: ok` даже без Tesseract и
    языковых моделей — балансировщик считал такой контейнер исправным.

    `profiles` — только те профили, что реально отработают. Профиль, который
    объявлен, но ответит 503, не должен попадать в этот список: клиент читает
    его как перечень доступных режимов.
    """
    ready, detail = _readiness()
    payload: dict[str, Any] = {
        "status": "ready" if ready else "not_ready",
        **detail,
        "render_profiles": sorted(PROFILES),
        "limits": {
            "max_bytes": MAX_BYTES,
            "max_pages": MAX_PAGES,
            "max_queued_jobs": MAX_QUEUED_JOBS,
            "job_ttl_s": JOB_TTL_SECONDS,
            "page_workers": concurrency.page_worker_budget(),
            "max_documents": concurrency.max_documents(),
        },
    }
    return JSONResponse(payload, status_code=200 if ready else 503)


# --------------------------------------------------------------------------
# Распознавание
# --------------------------------------------------------------------------


@app.post("/ocr")
def ocr(
    file: UploadFile = File(...),
    profile: str = Query(DEFAULT_PROFILE),
    render: str = Query(DEFAULT_RENDER),
) -> JSONResponse:
    """Синхронное распознавание. Для больших документов используйте /jobs."""
    _validate_request(profile, render)
    path = _save_upload(file)
    try:
        payload = _run(path, file.filename or path.name, profile, render)
    except HTTPException:
        raise
    except concurrency.CapacityExceeded as exc:
        raise HTTPException(
            status_code=503,
            detail="сервис занят, повторите позже",
            headers={"Retry-After": "30"},
        ) from exc
    except RecognitionUnavailable as exc:
        log.error("распознавание недоступно: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="распознавание недоступно, обратитесь к администратору",
        ) from exc
    except InvalidPdf as exc:
        # `str(exc)` содержит путь временного файла и вывод pdfinfo: наружу
        # это утечка деталей файловой системы. Клиенту — фиксированная
        # формулировка, полная причина только в журнал.
        log.warning("некорректный PDF: %s", exc)
        raise _client_error(422, "файл не читается как PDF или повреждён") from exc
    except ValueError as exc:
        raise _client_error(422, str(exc)) from exc
    except Exception as exc:
        raise _server_error(exc, "ocr") from exc
    return JSONResponse(payload)


# --------------------------------------------------------------------------
# Фоновые задания
# --------------------------------------------------------------------------


def _reap_jobs(now: float | None = None) -> int:
    """Удаляет просроченные задания вместе с их временными файлами."""
    now = now if now is not None else time.monotonic()
    removed = 0
    with _jobs_lock:
        for job_id, job in list(_jobs.items()):
            expires = job.get("_expires_at")
            if expires is not None and expires <= now:
                upload = job.get("_upload")
                if upload:
                    Path(upload).unlink(missing_ok=True)
                del _jobs[job_id]
                removed += 1
    return removed


def _public_job(job: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in job.items() if not k.startswith("_")}


@app.post("/jobs", status_code=202)
def create_job(
    file: UploadFile = File(...),
    profile: str = Query(DEFAULT_PROFILE),
    render: str = Query(DEFAULT_RENDER),
) -> JSONResponse:
    _validate_request(profile, render)
    _reap_jobs()

    # Слот резервируется тем же действием, что и проверка вместимости.
    # Проверять под замком, отпускать его на приём файла и вставлять запись
    # потом — это гонка: параллельные запросы проходят проверку одновременно,
    # пока ни один ещё не занял место. Воспроизводилось барьером перед
    # приёмом файла: при лимите 2 все шесть запросов получали 202.
    job_id = uuid.uuid4().hex
    with _jobs_lock:
        if len(_jobs) >= MAX_QUEUED_JOBS:
            raise HTTPException(
                status_code=429,
                detail=f"очередь заданий заполнена ({MAX_QUEUED_JOBS}); заберите готовые результаты",
                headers={"Retry-After": "30"},
            )
        cancel = threading.Event()
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "accepted",
            "file": file.filename or "",
            "_upload": None,
            "_cancel": cancel,
            "_expires_at": None,
        }

    # Дальше любая ошибка обязана освободить зарезервированное место,
    # иначе очередь деградирует до нуля свободных слотов.
    try:
        path = _save_upload(file)
    except BaseException:
        with _jobs_lock:
            _jobs.pop(job_id, None)
        raise

    name = file.filename or path.name
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:  # резервацию удалили, пока принимался файл
            path.unlink(missing_ok=True)
            raise _client_error(409, "задание отменено во время приёма файла")
        job.update({"status": "running", "file": name, "_upload": str(path)})

    def task() -> None:
        try:
            # Принятое задание ждёт слот весь свой TTL: очередь для того и
            # нужна, чтобы пережить занятость, а не падать от неё.
            result = _run(path, name, profile, render, slot_timeout=JOB_TTL_SECONDS, cancel=cancel)
            record: dict[str, Any] = {"status": "done", "result": result}
        except JobCancelled:
            log.info("job %s отменено клиентом до начала обработки", job_id)
            return
        except concurrency.CapacityExceeded:
            record = {"status": "failed", "error": "сервис занят, задание не запущено"}
        except RecognitionUnavailable as exc:
            log.error("job %s: распознавание недоступно: %s", job_id, exc)
            record = {"status": "failed", "error": "распознавание недоступно"}
        except InvalidPdf as exc:
            log.warning("job %s: некорректный PDF: %s", job_id, exc)
            record = {"status": "failed", "error": "файл не читается как PDF или повреждён"}
        except ValueError as exc:
            record = {"status": "failed", "error": str(exc)}
        except Exception as exc:  # noqa: BLE001 — подробности уходят в журнал
            error_id = uuid.uuid4().hex[:12]
            log.exception("[%s] job %s: %s: %s", error_id, job_id, type(exc).__name__, exc)
            record = {"status": "failed", "error": f"внутренняя ошибка, идентификатор {error_id}"}

        with _jobs_lock:
            job = _jobs.get(job_id)
            if job is None:  # задание успели удалить
                return
            job.update(record)
            job["_upload"] = None
            job["_expires_at"] = time.monotonic() + JOB_TTL_SECONDS

    try:
        _executor.submit(task)
    except BaseException:
        # Пул может быть остановлен или переполнен: без этой ветки слот и
        # временный файл остались бы занятыми навсегда, а задание — вечно
        # «running».
        with _jobs_lock:
            _jobs.pop(job_id, None)
        path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=503,
            detail="сервис не принимает задания",
            headers={"Retry-After": "30"},
        )

    return JSONResponse(
        {"job_id": job_id, "status": "running", "ttl_s": JOB_TTL_SECONDS},
        status_code=202,
    )


@app.get("/jobs/{job_id}")
def get_job(job_id: str, consume: bool = Query(False)) -> dict[str, Any]:
    """Состояние задания.

    `consume=1` удаляет результат сразу после выдачи. Иначе он живёт до
    истечения TTL, после чего удаляется вместе с временными файлами.
    """
    _reap_jobs()
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise _client_error(404, "задание не найдено или устарело")
        payload = _public_job(job)
        if consume and job["status"] in ("done", "failed"):
            del _jobs[job_id]
    return payload


@app.delete("/jobs/{job_id}", status_code=204)
def delete_job(job_id: str) -> Response:
    """Удаляет задание; работающее — отменяет, не вырывая у него файл.

    Прежняя версия удаляла входной файл сразу. Если задание в этот момент
    обрабатывалось, воркер терял файл под собой и падал внутренней ошибкой.
    Владелец временного файла — сама задача: она удаляет его в `finally`,
    поэтому здесь достаточно снять запись и поднять флаг отмены.
    """
    with _jobs_lock:
        job = _jobs.pop(job_id, None)
    if job is None:
        raise _client_error(404, "задание не найдено")

    cancel = job.get("_cancel")
    if cancel is not None:
        cancel.set()

    # Файл удаляется здесь только у завершённых заданий: у них владельца уже
    # нет. У принятого или работающего задания его уберёт сама задача.
    if job.get("status") in ("done", "failed"):
        upload = job.get("_upload")
        if upload:
            Path(upload).unlink(missing_ok=True)
    # Именно Response, а не JSONResponse: у 204 не должно быть тела, а
    # JSONResponse(content=None) сериализует "null", расходится с
    # Content-Length и роняет соединение с RuntimeError в uvicorn.
    return Response(status_code=204)


@app.get("/jobs")
def list_jobs() -> dict[str, Any]:
    _reap_jobs()
    with _jobs_lock:
        return {
            "jobs": [_public_job(j) for j in _jobs.values()],
            "queued": len(_jobs),
            "running": sum(1 for j in _jobs.values() if j["status"] == "running"),
            "capacity": MAX_QUEUED_JOBS,
        }
