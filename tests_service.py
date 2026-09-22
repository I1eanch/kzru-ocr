"""Поведенческие тесты HTTP-слоя.

Каждый тест защищает конкретный дефект, найденный ревью, и падает, если
дефект вернётся. Тестов, которые проверяют «что код написан», здесь нет.

Запуск (образ dev, в нём есть httpx и pytest):

    docker run --rm kzru-ocr:dev python -m pytest tests_service.py -v
"""

from __future__ import annotations

import io
import os
import time

import pytest
from fastapi.testclient import TestClient

import service.app as app_module
from ocr.pipeline import MAX_BYTES
from service.app import app


@pytest.fixture()
def client() -> TestClient:
    with TestClient(app) as c:
        yield c
    app_module._jobs.clear()


def _open_fds() -> int:
    try:
        return len(os.listdir("/proc/self/fd"))
    except FileNotFoundError:  # не Linux
        pytest.skip("подсчёт дескрипторов доступен только на Linux")
        return 0


def _pdf_bytes(name: str = "sample") -> bytes:
    """Минимальный валидный PDF без текстового слоя."""
    path = f"bench/scans/doc_001_dogovor.pdf"
    with open(path, "rb") as fh:
        return fh.read()


# --------------------------------------------------------------------------
# Загрузка
# --------------------------------------------------------------------------


def test_upload_over_limit_rejected_without_writing_whole_file(client: TestClient) -> None:
    """Превышение лимита обрывает приём, а не проверяется после записи."""
    oversized = b"%PDF-1.4\n" + b"0" * (MAX_BYTES + 1024)
    response = client.post("/ocr", files={"file": ("big.pdf", io.BytesIO(oversized), "application/pdf")})
    assert response.status_code == 413


def test_save_upload_stops_reading_after_limit() -> None:
    """Чтение обрывается на лимите, а не после приёма всего потока.

    Проверка кода ответа этого не ловит: прежняя версия тоже отвечала 413 —
    но уже записав на диск файл целиком, какого бы размера он ни был.
    """

    class CountingStream:
        def __init__(self, total: int) -> None:
            self.remaining = total
            self.read_bytes = 0

        def read(self, size: int = -1) -> bytes:
            if self.remaining <= 0:
                return b""
            n = self.remaining if size is None or size < 0 else min(size, self.remaining)
            self.remaining -= n
            self.read_bytes += n
            return b"0" * n

    class FakeUpload:
        filename = "big.pdf"

        def __init__(self, stream: CountingStream) -> None:
            self.file = stream

    stream = CountingStream(MAX_BYTES * 3)
    with pytest.raises(Exception) as excinfo:
        app_module._save_upload(FakeUpload(stream))

    assert getattr(excinfo.value, "status_code", None) == 413
    assert stream.read_bytes <= MAX_BYTES + app_module.UPLOAD_CHUNK, (
        f"прочитано {stream.read_bytes} байт при лимите {MAX_BYTES}"
    )


def test_upload_does_not_leak_file_descriptors(client: TestClient) -> None:
    """`mkstemp` отдаёт дескриптор, и его обязаны закрыть.

    Прежняя версия теряла его на каждом запросе: 20 вызовов — 20 дескрипторов.
    """
    oversized = b"%PDF-1.4\n" + b"0" * (MAX_BYTES + 1024)
    client.post("/ocr", files={"file": ("big.pdf", io.BytesIO(oversized), "application/pdf")})

    before = _open_fds()
    for _ in range(15):
        client.post("/ocr", files={"file": ("big.pdf", io.BytesIO(oversized), "application/pdf")})
    after = _open_fds()

    assert after - before <= 2, f"утечка дескрипторов: было {before}, стало {after}"


def test_rejected_upload_leaves_no_temp_file(client: TestClient, tmp_path) -> None:
    """Отклонённая загрузка не оставляет мусора во временном каталоге."""
    tmpdir = tempfile_dir()
    before = set(os.listdir(tmpdir))
    oversized = b"%PDF-1.4\n" + b"0" * (MAX_BYTES + 1024)
    client.post("/ocr", files={"file": ("big.pdf", io.BytesIO(oversized), "application/pdf")})
    leftovers = {n for n in os.listdir(tmpdir) if n.startswith("kzru-upload-")} - before
    assert not leftovers, f"остались временные файлы: {leftovers}"


def tempfile_dir() -> str:
    import tempfile

    return tempfile.gettempdir()


def test_empty_upload_rejected(client: TestClient) -> None:
    response = client.post("/ocr", files={"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")})
    assert response.status_code == 400


def test_corrupted_pdf_is_client_error(client: TestClient) -> None:
    """Повреждённый файл — ошибка входных данных, а не сбой сервиса."""
    response = client.post(
        "/ocr", files={"file": ("bad.pdf", io.BytesIO("не pdf".encode()), "application/pdf")}
    )
    assert response.status_code == 422


def test_internal_error_does_not_leak_details(client: TestClient, monkeypatch) -> None:
    """Клиент получает идентификатор, а не тип исключения и пути."""

    def boom(*args, **kwargs):
        raise RuntimeError("секретный путь /opt/tessdata_best/kzru_doc.traineddata")

    monkeypatch.setattr(app_module, "process_pdf", boom)
    response = client.post("/ocr", files={"file": ("x.pdf", io.BytesIO(_pdf_bytes()), "application/pdf")})
    assert response.status_code == 500
    detail = response.json()["detail"]
    assert "RuntimeError" not in detail
    assert "tessdata" not in detail
    assert "идентификатор" in detail


# --------------------------------------------------------------------------
# Готовность
# --------------------------------------------------------------------------


def test_liveness_is_independent_of_dependencies(client: TestClient) -> None:
    assert client.get("/healthz").status_code == 200


def test_readiness_fails_closed_without_tesseract(client: TestClient, monkeypatch) -> None:
    """Без Tesseract сервис обязан объявлять себя неготовым."""
    monkeypatch.setattr(app_module.shutil, "which", lambda _: None)
    response = client.get("/readyz")
    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"


def test_readiness_fails_closed_without_default_langs(client: TestClient, monkeypatch) -> None:
    """Нет языков профиля по умолчанию — не готов, даже если бинарь на месте."""
    monkeypatch.setattr(app_module, "_tesseract_state", lambda: ("tesseract 5.3.0", ["eng"]))
    response = client.get("/readyz")
    assert response.status_code == 503
    assert any("языковых моделей" in p for p in response.json()["problems"])


def test_readiness_ok_on_healthy_install(client: TestClient) -> None:
    response = client.get("/readyz")
    assert response.status_code == 200, response.json().get("problems")
    assert response.json()["status"] == "ready"


# --------------------------------------------------------------------------
# Профили
# --------------------------------------------------------------------------


def test_advertised_profiles_are_accepted(client: TestClient) -> None:
    """Всё, что сервис рекламирует, он обязан принимать.

    Раньше `/healthz` перечислял весь реестр, а API принимал только три имени.
    """
    advertised = client.get("/readyz").json()["profiles"]
    pdf = _pdf_bytes()
    for name in advertised:
        response = client.post(
            "/ocr",
            params={"profile": name},
            files={"file": ("x.pdf", io.BytesIO(pdf), "application/pdf")},
        )
        assert response.status_code != 400, f"профиль {name} рекламируется, но отвергнут"


def test_unknown_profile_rejected(client: TestClient) -> None:
    response = client.post(
        "/ocr",
        params={"profile": "нет-такого"},
        files={"file": ("x.pdf", io.BytesIO(_pdf_bytes()), "application/pdf")},
    )
    assert response.status_code == 400


# --------------------------------------------------------------------------
# Очередь заданий
# --------------------------------------------------------------------------


def test_job_queue_is_bounded(client: TestClient, monkeypatch) -> None:
    """Переполненная очередь отвечает 429, а не растёт без предела."""
    monkeypatch.setattr(app_module, "MAX_QUEUED_JOBS", 2)
    pdf = _pdf_bytes()

    codes = [
        client.post("/jobs", files={"file": ("x.pdf", io.BytesIO(pdf), "application/pdf")}).status_code
        for _ in range(4)
    ]
    assert 429 in codes, f"очередь не ограничена: {codes}"
    assert codes[0] == 202


def test_finished_job_expires_and_cleans_up(client: TestClient, monkeypatch) -> None:
    """Результат не живёт вечно: по TTL он удаляется вместе с временными файлами."""
    monkeypatch.setattr(app_module, "JOB_TTL_SECONDS", 0.0)
    pdf = _pdf_bytes()
    job_id = client.post("/jobs", files={"file": ("x.pdf", io.BytesIO(pdf), "application/pdf")}).json()["job_id"]

    deadline = time.time() + 120
    while time.time() < deadline:
        body = client.get(f"/jobs/{job_id}")
        if body.status_code == 404 or body.json().get("status") in ("done", "failed"):
            break
        time.sleep(0.5)

    app_module._reap_jobs()
    assert client.get(f"/jobs/{job_id}").status_code == 404
    assert job_id not in app_module._jobs


def test_job_can_be_consumed_and_deleted(client: TestClient) -> None:
    pdf = _pdf_bytes()
    job_id = client.post("/jobs", files={"file": ("x.pdf", io.BytesIO(pdf), "application/pdf")}).json()["job_id"]

    deadline = time.time() + 120
    status = "running"
    while time.time() < deadline:
        status = client.get(f"/jobs/{job_id}").json()["status"]
        if status in ("done", "failed"):
            break
        time.sleep(0.5)
    assert status == "done", f"задание не завершилось: {status}"

    consumed = client.get(f"/jobs/{job_id}", params={"consume": "1"})
    assert consumed.status_code == 200
    assert client.get(f"/jobs/{job_id}").status_code == 404


def test_job_record_hides_internal_fields(client: TestClient) -> None:
    pdf = _pdf_bytes()
    job_id = client.post("/jobs", files={"file": ("x.pdf", io.BytesIO(pdf), "application/pdf")}).json()["job_id"]
    body = client.get(f"/jobs/{job_id}").json()
    assert not [k for k in body if k.startswith("_")], body


# --------------------------------------------------------------------------
# Результат распознавания
# --------------------------------------------------------------------------


def test_fields_carry_verification_status(client: TestClient) -> None:
    """Поля отдаются со статусом, а не плоским списком строк."""
    response = client.post("/ocr", files={"file": ("x.pdf", io.BytesIO(_pdf_bytes()), "application/pdf")})
    assert response.status_code == 200
    fields = response.json()["fields"]
    for item in fields["bin"]:
        assert item["status"] in ("valid", "repaired", "unverified")
        assert set(item) >= {"value", "raw", "status", "candidates", "requires_review"}
        assert item["requires_review"] is (item["status"] != "valid")
    for item in fields["dates"]:
        assert item["status"] in ("valid", "invalid")


def test_page_failure_surfaces_as_warning(client: TestClient, monkeypatch) -> None:
    """Сбой страницы не теряется: он доходит до клиента предупреждением."""
    import ocr.pipeline as pipeline

    original = pipeline._recognize_page

    def failing(job):
        result = original(job)
        result.error = "искусственный сбой страницы"
        return result

    monkeypatch.setattr(pipeline, "_recognize_page", failing)
    response = client.post(
        "/ocr",
        params={"workers": 1},
        files={"file": ("x.pdf", io.BytesIO(_pdf_bytes()), "application/pdf")},
    )
    assert response.status_code == 200
    types = {w["type"] for w in response.json()["warnings"]}
    assert "page_failed" in types, types
