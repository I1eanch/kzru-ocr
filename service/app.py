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

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response

from ocr import concurrency
from ocr.pdf_backend import DEFAULT_BACKEND, available_backends
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

UPLOAD_CHUNK = 1024 * 1024

_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=max(1, concurrency.max_documents()))


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


def _run(path: Path, name: str, profile: str, render: str) -> dict[str, Any]:
    """Обрабатывает документ, занимая слот общего бюджета."""
    try:
        with concurrency.document_slot(timeout=SLOT_TIMEOUT_SECONDS):
            doc = process_pdf(str(path), profile=profile, render=render)
        return _document_payload(doc, name, render)
    finally:
        path.unlink(missing_ok=True)


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


def _readiness() -> tuple[bool, dict[str, Any]]:
    """Проверяет ровно то, без чего профиль по умолчанию не отработает."""
    problems: list[str] = []

    default_profile = PIPELINE_PROFILES["balanced"]
    tessdata_dir = _tessdata_dir(default_profile)
    version, langs = _tesseract_state(tessdata_dir)

    if not version:
        problems.append("бинарь tesseract не найден в PATH")

    required = [lang for lang in default_profile.lang.split("+") if lang]

    # Файлы проверяются отдельно от `--list-langs`: при несуществующем
    # каталоге Tesseract не сообщает об ошибке, а берёт модели из
    # вкомпилированного пути. Тогда сервис работал бы на весах, которых
    # профиль не объявлял.
    missing_files = [lang for lang in required if not Path(tessdata_dir, f"{lang}.traineddata").is_file()]
    if missing_files:
        problems.append(f"в каталоге {tessdata_dir} нет файлов моделей: {missing_files}")

    missing_langs = [lang for lang in required if lang not in langs]
    if missing_langs:
        problems.append(f"Tesseract не видит языки профиля по умолчанию: {missing_langs}")

    backends = available_backends()
    if DEFAULT_BACKEND not in backends:
        problems.append(f"PDF-бэкенд по умолчанию {DEFAULT_BACKEND} недоступен; есть {backends}")

    return not problems, {
        "tesseract": version,
        "langs": langs,
        "required_langs": required,
        "tessdata_dir": tessdata_dir,
        "pdf_backends": backends,
        "pdf_backend_default": DEFAULT_BACKEND,
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
    """
    ready, detail = _readiness()
    payload: dict[str, Any] = {
        "status": "ready" if ready else "not_ready",
        **detail,
        "profiles": _profile_names(),
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
    profile: str = Query("balanced"),
    render: str = Query("default"),
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


def _active_jobs() -> int:
    with _jobs_lock:
        return sum(1 for j in _jobs.values() if j["status"] == "running")


def _public_job(job: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in job.items() if not k.startswith("_")}


@app.post("/jobs", status_code=202)
def create_job(
    file: UploadFile = File(...),
    profile: str = Query("balanced"),
    render: str = Query("default"),
) -> JSONResponse:
    _validate_request(profile, render)
    _reap_jobs()

    with _jobs_lock:
        if len(_jobs) >= MAX_QUEUED_JOBS:
            raise HTTPException(
                status_code=429,
                detail=f"очередь заданий заполнена ({MAX_QUEUED_JOBS}); заберите готовые результаты",
                headers={"Retry-After": "30"},
            )

    path = _save_upload(file)
    job_id = uuid.uuid4().hex
    name = file.filename or path.name

    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "running",
            "file": name,
            "_upload": str(path),
            "_expires_at": None,
        }

    def task() -> None:
        try:
            result = _run(path, name, profile, render)
            record: dict[str, Any] = {"status": "done", "result": result}
        except concurrency.CapacityExceeded:
            record = {"status": "failed", "error": "сервис занят, задание не запущено"}
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

    _executor.submit(task)
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
    with _jobs_lock:
        job = _jobs.pop(job_id, None)
    if job is None:
        raise _client_error(404, "задание не найдено")
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
