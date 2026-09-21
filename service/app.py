"""Локальный HTTP-сервис для вызова из PHP (CodeIgniter 4).

Почему сервис, а не CLI на каждый файл: загрузка моделей стоит секунды, а при
запуске процесса на документ это время платится каждый раз. Здесь модели
живут в памяти воркеров.

Контракт ответа включает `warnings` — это сознательная часть продукта, а не
отладочный вывод. Сервис проверки госдокументов должен знать, где OCR не
уверен: непрошедшая контрольная сумма БИН, расхождение суммы цифрами и
прописью, низкая уверенность страницы. Такие места возвращаются наружу, чтобы
интерфейс мог показать «проверьте вручную».
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from ocr.pipeline import MAX_BYTES, MAX_PAGES, PIPELINE_PROFILES, process_pdf
from ocr.render import PROFILES, render_document

app = FastAPI(title="kzru-ocr", version="0.1.0")

_jobs: dict[str, dict[str, Any]] = {}
_executor = ThreadPoolExecutor(max_workers=2)

ProfileName = Literal["fast", "balanced", "accurate"]


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


def _save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "upload.pdf").suffix or ".pdf"
    tmp = Path(tempfile.mkstemp(suffix=suffix)[1])
    with tmp.open("wb") as fh:
        shutil.copyfileobj(upload.file, fh, length=1024 * 1024)
    if tmp.stat().st_size > MAX_BYTES:
        tmp.unlink(missing_ok=True)
        raise HTTPException(status_code=413, detail=f"файл больше {MAX_BYTES} байт")
    return tmp


def _run(path: Path, name: str, profile: str, render: str) -> dict[str, Any]:
    try:
        doc = process_pdf(str(path), profile=profile, render=render)
        return _document_payload(doc, name, render)
    finally:
        path.unlink(missing_ok=True)


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    tesseract = shutil.which("tesseract")
    version = ""
    langs: list[str] = []
    if tesseract:
        version = subprocess.run(
            [tesseract, "--version"], capture_output=True, text=True, check=False
        ).stdout.splitlines()[:1]
        version = version[0] if version else ""
        langs = [
            ln.strip()
            for ln in subprocess.run(
                [tesseract, "--list-langs"], capture_output=True, text=True, check=False
            ).stdout.splitlines()[1:]
            if ln.strip()
        ]

    try:
        import paddleocr  # noqa: F401

        paddle = True
    except ImportError:
        paddle = False

    return {
        "status": "ok",
        "tesseract": version,
        "langs": langs,
        "paddle": paddle,
        "profiles": sorted(PIPELINE_PROFILES),
        "render_profiles": sorted(PROFILES),
        "limits": {"max_bytes": MAX_BYTES, "max_pages": MAX_PAGES},
    }


@app.post("/ocr")
def ocr(
    file: UploadFile = File(...),
    profile: ProfileName = Query("balanced"),
    render: str = Query("default"),
) -> JSONResponse:
    """Синхронное распознавание. Для больших документов используйте /jobs."""
    if render not in PROFILES:
        raise HTTPException(status_code=400, detail=f"неизвестный render-профиль: {render}")
    path = _save_upload(file)
    try:
        payload = _run(path, file.filename or path.name, profile, render)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
    return JSONResponse(payload)


@app.post("/jobs", status_code=202)
def create_job(
    file: UploadFile = File(...),
    profile: ProfileName = Query("balanced"),
    render: str = Query("default"),
) -> dict[str, str]:
    if render not in PROFILES:
        raise HTTPException(status_code=400, detail=f"неизвестный render-профиль: {render}")
    path = _save_upload(file)
    job_id = uuid.uuid4().hex
    name = file.filename or path.name
    _jobs[job_id] = {"status": "running", "file": name}

    def task() -> None:
        try:
            _jobs[job_id] = {"status": "done", "file": name, "result": _run(path, name, profile, render)}
        except Exception as exc:
            _jobs[job_id] = {"status": "failed", "file": name, "error": f"{type(exc).__name__}: {exc}"}

    _executor.submit(task)
    return {"job_id": job_id, "status": "running"}


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job не найден")
    return job
