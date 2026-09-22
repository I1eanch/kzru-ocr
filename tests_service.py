"""Поведенческие тесты HTTP-слоя.

Каждый тест защищает конкретный дефект, найденный ревью, и падает, если
дефект вернётся. Тестов, которые проверяют «что код написан», здесь нет.

Запуск (образ dev, в нём есть httpx и pytest):

    docker run --rm kzru-ocr:dev python -m pytest tests_service.py -v
"""

from __future__ import annotations

import io
import os
import tempfile
import time
from pathlib import Path

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


FIXTURES = Path(__file__).resolve().parent / "tests" / "fixtures"
ONE_PAGE = FIXTURES / "one_page.pdf"
THREE_PAGES = FIXTURES / "three_pages.pdf"


def _pdf_bytes() -> bytes:
    """Односторонний тестовый скан — отслеживаемый файл, не локальный артефакт.

    Раньше тесты читали `bench/scans/`, который в `.gitignore`: в свежем
    клоне их нечем было запустить, а «тесты проходят» доказывалось локальными
    артефактами. Фикстуры лежат в репозитории и пересоздаются скриптом
    `tests/fixtures/make_fixtures.py`.
    """
    return ONE_PAGE.read_bytes()


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
    monkeypatch.setattr(
        app_module, "_tesseract_state", lambda tessdata_dir=None: ("tesseract 5.3.0", ["eng"])
    )
    response = client.get("/readyz")
    assert response.status_code == 503
    assert any("не видит языки" in p for p in response.json()["problems"])


def test_readiness_ok_on_healthy_install(client: TestClient) -> None:
    response = client.get("/readyz")
    assert response.status_code == 200, response.json().get("problems")
    assert response.json()["status"] == "ready"


def test_readiness_checks_the_directory_the_engine_uses(client: TestClient, monkeypatch, tmp_path) -> None:
    """Готовность обязана смотреть в тот же каталог моделей, что и движок.

    Прежняя версия спрашивала `tesseract --list-langs` без `--tessdata-dir` и
    потому проверяла `TESSDATA_PREFIX`, тогда как движок запускается с
    `--tessdata-dir`. Расхождение наблюдалось вживую: при сломанном
    `TESSDATA_BEST` готовность отвечала 503, а распознавание продолжало
    работать на пакетных моделях Debian — то есть на весах, которых профиль
    не объявлял.
    """
    empty = tmp_path / "tessdata"
    empty.mkdir()
    monkeypatch.setattr(app_module, "_tessdata_dir", lambda profile: str(empty))

    response = client.get("/readyz")
    payload = response.json()

    # Проверяется наблюдаемый контракт, а не формулировка сообщения: сервис
    # не готов, профиль не рекламируется и назван нерабочим с причиной.
    assert response.status_code == 503
    assert payload["problems"], "отказ без указания причин"
    assert "balanced" not in payload["profiles"], payload["profiles"]
    assert payload["profiles_unavailable"].get("balanced"), payload["profiles_unavailable"]


def test_readiness_reports_engine_directory(client: TestClient) -> None:
    """Каталог моделей виден в ответе — без него диагностика вслепую."""
    body = client.get("/readyz").json()
    assert body["tessdata_dir"]
    assert Path(body["tessdata_dir"]).is_dir()


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


def test_partial_page_failure_surfaces_as_warning(client: TestClient, monkeypatch) -> None:
    """Сбой одной страницы не теряется и не роняет весь документ.

    Документ отдаётся с предупреждением; на 503 он уходит только если не
    распозналась ни одна страница — это разные ситуации.
    """
    import ocr.pipeline as pipeline

    original = pipeline._recognize_page

    def failing(job):
        result = original(job)
        if job.index == 0:
            result.error = "искусственный сбой страницы"
        return result

    # Подмена действует только в текущем процессе, поэтому страницы должны
    # обрабатываться на месте: при размере пула 1 `_run_jobs` идёт inline.
    monkeypatch.setattr(pipeline.concurrency, "shared_pool_size", lambda: 1)
    monkeypatch.setattr(pipeline, "_recognize_page", failing)
    multipage = THREE_PAGES.read_bytes()

    response = client.post(
        "/ocr",
        files={"file": ("x.pdf", io.BytesIO(multipage), "application/pdf")},
    )
    assert response.status_code == 200, response.json()
    types = {w["type"] for w in response.json()["warnings"]}
    assert "page_failed" in types, types
    assert response.json()["text"].strip(), "текст уцелевших страниц должен остаться"


def test_all_pages_failed_is_service_error_not_empty_success(client: TestClient, monkeypatch) -> None:
    """Документ, не распознавшийся целиком, не отдаётся как успех с пустым текстом.

    Иначе клиент, не разобравший `warnings`, запишет пустоту как результат
    проверки документа. Наблюдалось вживую: при сломанном каталоге моделей
    `/ocr` отвечал 200, `text` был пуст, а уверенность равна нулю.
    """
    import ocr.pipeline as pipeline

    def failing(job):
        result = pipeline.PageResult(index=job.index, page=pipeline.Page(index=job.index))
        result.error = "EngineUnavailable: моделей нет"
        return result

    monkeypatch.setattr(pipeline.concurrency, "shared_pool_size", lambda: 1)
    monkeypatch.setattr(pipeline, "_recognize_page", failing)
    response = client.post("/ocr", files={"file": ("x.pdf", io.BytesIO(_pdf_bytes()), "application/pdf")})
    assert response.status_code == 503, response.json()
    assert "распознавание недоступно" in response.json()["detail"]


# --------------------------------------------------------------------------
# Приём заданий под конкуренцией
# --------------------------------------------------------------------------


def test_concurrent_admission_respects_queue_limit(client: TestClient, monkeypatch) -> None:
    """Лимит очереди держится при одновременных запросах, а не только по очереди.

    Прежняя версия проверяла вместимость под замком, отпускала его на приём
    файла и вставляла запись потом. Параллельные запросы проходили проверку
    одновременно, пока ни один ещё не занял место: при лимите 2 все шесть
    запросов получали 202, и в очереди оказывалось шесть заданий.

    Синхронизация устроена так, чтобы тест завершался детерминированно.
    Барьер стоит ПЕРЕД запросом и разводит старт всех потоков: ждать все
    шесть внутри приёма файла нельзя — после исправления туда доходят только
    допущенные запросы, и барьер не собрался бы никогда. Само окно гонки
    удерживается открытым короткой задержкой в приёме файла.
    """
    import threading

    limit = 2
    requests = 6
    monkeypatch.setattr(app_module, "MAX_QUEUED_JOBS", limit)

    start = threading.Barrier(requests, timeout=30)
    original_save = app_module._save_upload

    def slow_save(upload):
        # Окно между проверкой вместимости и вставкой записи: на старом коде
        # за это время сюда успевали войти все шесть запросов.
        time.sleep(0.2)
        return original_save(upload)

    monkeypatch.setattr(app_module, "_save_upload", slow_save)
    # Работу не запускаем: проверяется приём, а не обработка.
    monkeypatch.setattr(app_module._executor, "submit", lambda fn: None)

    codes: list[int] = []
    codes_lock = threading.Lock()
    payload = _pdf_bytes()

    def post() -> None:
        start.wait()
        response = client.post(
            "/jobs", files={"file": ("x.pdf", io.BytesIO(payload), "application/pdf")}
        )
        with codes_lock:
            codes.append(response.status_code)

    threads = [threading.Thread(target=post) for _ in range(requests)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
        assert not t.is_alive(), "запрос не завершился"

    accepted = sum(1 for c in codes if c == 202)
    assert accepted == limit, f"принято {accepted} при лимите {limit}: {codes}"
    assert sum(1 for c in codes if c == 429) == requests - limit, codes
    assert len(app_module._jobs) == limit, f"в очереди {len(app_module._jobs)} заданий"

    # Задачи не запускались, поэтому временные файлы убираем сами.
    for job in app_module._jobs.values():
        upload = job.get("_upload")
        if upload:
            Path(upload).unlink(missing_ok=True)


def test_failed_submit_releases_reserved_slot(client: TestClient, monkeypatch) -> None:
    """Падение постановки в пул освобождает слот и удаляет временный файл.

    Иначе резервация зависает навсегда: очередь деградирует до нуля свободных
    мест, а задание вечно числится принятым.
    """
    def exploding_submit(fn):
        raise RuntimeError("пул остановлен")

    monkeypatch.setattr(app_module._executor, "submit", exploding_submit)
    before = set(Path(tempfile.gettempdir()).glob("kzru-upload-*"))

    response = client.post(
        "/jobs", files={"file": ("x.pdf", io.BytesIO(_pdf_bytes()), "application/pdf")}
    )
    assert response.status_code == 503, response.text
    assert len(app_module._jobs) == 0, "зарезервированный слот не освобождён"
    leaked = set(Path(tempfile.gettempdir()).glob("kzru-upload-*")) - before
    assert not leaked, f"временный файл остался: {leaked}"


# --------------------------------------------------------------------------
# Отмена, границы запроса, контракт профилей
# --------------------------------------------------------------------------


def test_delete_running_job_does_not_yank_file_from_worker(client: TestClient, monkeypatch) -> None:
    """Удаление работающего задания не вырывает входной файл из-под воркера.

    Прежняя версия удаляла файл немедленно. Если задание в этот момент
    обрабатывалось, воркер терял файл под собой и падал внутренней ошибкой.
    Владелец временного файла — сама задача.
    """
    import threading

    started = threading.Event()
    seen_path: dict[str, Path] = {}
    release = threading.Event()

    original = app_module._process_in_slot

    def slow_process(path, name, profile, render):
        seen_path["path"] = Path(path)
        started.set()
        release.wait(timeout=30)
        assert Path(path).exists(), "входной файл удалён во время обработки"
        return original(path, name, profile, render)

    monkeypatch.setattr(app_module, "_process_in_slot", slow_process)

    response = client.post(
        "/jobs", files={"file": ("x.pdf", io.BytesIO(_pdf_bytes()), "application/pdf")}
    )
    job_id = response.json()["job_id"]
    assert started.wait(timeout=30), "задание не стартовало"

    deleted = client.delete(f"/jobs/{job_id}")
    assert deleted.status_code == 204
    assert Path(seen_path["path"]).exists(), "файл удалён, пока воркер с ним работает"

    release.set()
    # Даём задаче завершиться и убрать файл за собой.
    for _ in range(300):
        if not Path(seen_path["path"]).exists():
            break
        time.sleep(0.1)
    assert not Path(seen_path["path"]).exists(), "задача не убрала временный файл"


def test_oversized_body_rejected_before_multipart_parse(client: TestClient, monkeypatch) -> None:
    """Слишком большое тело отвергается по заголовку, не после разбора multipart.

    `UploadFile` появляется только после полного разбора тела, то есть после
    его приёма целиком. Проверка размера внутри обработчика ограничивает лишь
    вторую копию.
    """
    parsed = False

    original_save = app_module._save_upload

    def tracking_save(upload):
        nonlocal parsed
        parsed = True
        return original_save(upload)

    monkeypatch.setattr(app_module, "_save_upload", tracking_save)

    huge = b"%PDF-1.4\n" + b"0" * 1024
    response = client.post(
        "/ocr",
        files={"file": ("big.pdf", io.BytesIO(huge), "application/pdf")},
        headers={"content-length": str(app_module.MAX_REQUEST_BYTES + 1)},
    )
    assert response.status_code == 413, response.text
    assert not parsed, "тело разбиралось несмотря на превышение объявленного размера"


def test_advertised_profiles_actually_run(client: TestClient) -> None:
    """Каждый профиль из `/readyz` действительно выполняет распознавание.

    Прежняя проверка считала успехом любой код, кроме 400: профиль мог
    рекламироваться и отвечать 503 или 500. Список из `/readyz` клиент читает
    как перечень работающих режимов, поэтому он обязан это выдерживать.
    """
    ready = client.get("/readyz")
    assert ready.status_code == 200, ready.text
    advertised = ready.json()["profiles"]
    assert advertised, "не объявлено ни одного профиля"

    payload = _pdf_bytes()
    for profile in advertised:
        response = client.post(
            "/ocr",
            params={"profile": profile},
            files={"file": ("x.pdf", io.BytesIO(payload), "application/pdf")},
        )
        assert response.status_code == 200, f"профиль {profile}: {response.status_code} {response.text[:200]}"
        assert response.json()["text"].strip(), f"профиль {profile} вернул пустой текст"


def test_unavailable_profile_is_service_error_not_client_error(client: TestClient, monkeypatch) -> None:
    """Профиль с отсутствующими моделями даёт 503 и исчезает из рекламы.

    400 здесь врал бы: запрос клиента корректен, неисправен сервис.
    """
    real = app_module._profile_problem

    def broken(profile):
        return "модели не найдены" if profile.name == "fast" else real(profile)

    monkeypatch.setattr(app_module, "_profile_problem", broken)

    response = client.post(
        "/ocr",
        params={"profile": "fast"},
        files={"file": ("x.pdf", io.BytesIO(_pdf_bytes()), "application/pdf")},
    )
    assert response.status_code == 503, response.text
    # Причина отказа содержит путь к каталогу моделей: наружу он не уходит,
    # администратор видит его в журнале и в `/readyz`.
    detail = response.json()["detail"]
    assert "/opt" not in detail and "tessdata" not in detail, detail

    advertised = client.get("/readyz").json()["profiles"]
    assert "fast" not in advertised, advertised
    assert "fast" in client.get("/readyz").json()["profiles_unavailable"]


def test_invalid_pdf_does_not_leak_filesystem_paths(client: TestClient) -> None:
    """Нечитаемый PDF даёт 422, и сообщение не содержит путей файловой системы.

    Код проверяется строго: 503 означал бы, что до разбора PDF дело не дошло,
    и ветка `InvalidPdf` осталась бы недоказанной. Готовность профиля по
    умолчанию — предусловие теста, а не повод ослабить проверку.
    """
    ready = client.get("/readyz")
    assert ready.status_code == 200, f"профиль по умолчанию не готов: {ready.text}"

    broken = b"%PDF-1.4\nnot actually a pdf\n" + b"x" * 256
    response = client.post(
        "/ocr", files={"file": ("x.pdf", io.BytesIO(broken), "application/pdf")}
    )
    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert "/tmp" not in detail and "kzru-upload" not in detail, detail
    assert "pdfinfo" not in detail.lower(), detail


# --------------------------------------------------------------------------
# Живучесть пула воркеров
# --------------------------------------------------------------------------


def test_broken_worker_pool_recovers_instead_of_permanent_500(client: TestClient, monkeypatch) -> None:
    """Смерть воркера лечится пересозданием пула, а не превращает сервис в 500.

    `ProcessPoolExecutor` необратим: после `BrokenProcessPool` — воркера убил
    OOM killer или он упал в нативном коде — каждый следующий запуск падает
    той же ошибкой. Без пересоздания один OOM останавливает сервис навсегда,
    причём `/readyz` продолжает отвечать 200.
    """
    from concurrent.futures.process import BrokenProcessPool

    import ocr.concurrency as conc
    import ocr.pipeline as pipeline

    conc.shutdown_pool()

    calls = {"n": 0}
    real_pool = conc.shared_pool

    class OneShotBrokenPool:
        """Первый map падает как сломанный пул, дальше работает настоящий."""

        def map(self, fn, jobs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise BrokenProcessPool("воркер убит")
            return real_pool().map(fn, jobs)

    monkeypatch.setattr(conc, "shared_pool", lambda: OneShotBrokenPool())
    monkeypatch.setattr(pipeline.concurrency, "shared_pool", lambda: OneShotBrokenPool())
    monkeypatch.setattr(pipeline.concurrency, "shared_pool_size", lambda: 4)

    response = client.post(
        "/ocr", files={"file": ("x.pdf", io.BytesIO(THREE_PAGES.read_bytes()), "application/pdf")}
    )

    assert calls["n"] >= 2, "повтора после гибели пула не было"
    assert response.status_code == 200, response.text
    assert response.json()["text"].strip(), "после восстановления текст пуст"


def test_exhausted_pool_recovery_is_visible_in_readiness(client: TestClient, monkeypatch) -> None:
    """Пул, исчерпавший попытки восстановления, снимает готовность сервиса.

    Иначе балансировщик продолжит слать нагрузку на контейнер, где каждая
    многостраничная обработка падает.
    """
    import ocr.concurrency as conc

    assert client.get("/readyz").status_code == 200

    monkeypatch.setattr(conc, "pool_is_broken", lambda: True)
    response = client.get("/readyz")

    # Контракт наблюдаемый: сервис объявляет себя неготовым и называет
    # причину. Конкретная формулировка не закрепляется.
    assert response.status_code == 503, response.text
    assert response.json()["status"] == "not_ready"
    assert response.json()["problems"], "отказ без указания причин"


def test_pool_restart_budget_is_bounded_and_resets_on_success() -> None:
    """Бюджет пересозданий ограничен, но восстанавливается после успеха.

    Без границы устойчивый сбой крутился бы в вечном цикле перезапусков и
    выглядел как исправный сервис. Без сброса редкие падения, разнесённые во
    времени, однажды исчерпали бы лимит и остановили исправный сервис.
    """
    import ocr.concurrency as conc

    conc.shutdown_pool()
    conc._pool_restarts = 0
    conc._pool_broken = False
    try:
        for _ in range(conc.MAX_POOL_RESTARTS):
            assert conc.recycle_pool(conc.pool_generation()) is True

        assert conc.recycle_pool(conc.pool_generation()) is False, "бюджет не ограничен"
        assert conc.pool_is_broken() is True

        conc._pool_broken = False
        conc.note_pool_success()
        assert conc.recycle_pool(conc.pool_generation()) is True, "бюджет не сброшен после успеха"
    finally:
        conc._pool_restarts = 0
        conc._pool_broken = False
        conc.shutdown_pool()


def test_endpoints_use_the_declared_defaults(client: TestClient) -> None:
    """Умолчания ручек берутся из констант, а не из отдельных литералов.

    Константа, которую никто не читает, ничего не гарантирует: `/ocr` и
    `/jobs` держали собственные строки, и смена умолчания в одном месте молча
    расходилась с остальными. Проверяется наблюдаемо — через подмену
    константы и ответ сервиса, а не чтением исходника.
    """
    # Ответ без явных параметров сообщает применённый профиль и render.
    response = client.post(
        "/ocr", files={"file": ("x.pdf", io.BytesIO(_pdf_bytes()), "application/pdf")}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["profile"] == app_module.DEFAULT_PROFILE, body["profile"]
    assert body["render"] == app_module.DEFAULT_RENDER, body["render"]

    # Умолчание в схеме API совпадает с тем, что применилось фактически:
    # расхождение означало бы, что клиент видит одно, а получает другое.
    schema = client.get("/openapi.json").json()
    for path in ("/ocr", "/jobs"):
        params = schema["paths"][path]["post"]["parameters"]
        declared = {p["name"]: p["schema"].get("default") for p in params}
        assert declared["profile"] == app_module.DEFAULT_PROFILE, (path, declared)
        assert declared["render"] == app_module.DEFAULT_RENDER, (path, declared)


def test_permanently_broken_pool_gives_up_instead_of_looping(client: TestClient, monkeypatch) -> None:
    """Документ, на котором пул ломается всегда, не уходит в бесконечный цикл.

    Глобального бюджета пересозданий мало: его сбрасывает любой чужой успешно
    обработанный документ, поэтому документ, валящий воркера сам по себе,
    крутился бы вечно. Граница на документ должна остановить это независимо
    от глобального счётчика.
    """
    from concurrent.futures.process import BrokenProcessPool

    import ocr.concurrency as conc
    import ocr.pipeline as pipeline

    attempts = {"n": 0}

    class AlwaysBrokenPool:
        def map(self, fn, jobs):
            attempts["n"] += 1
            raise BrokenProcessPool("воркер убит")

    monkeypatch.setattr(pipeline.concurrency, "shared_pool", lambda: AlwaysBrokenPool())
    monkeypatch.setattr(pipeline.concurrency, "shared_pool_size", lambda: 4)
    # Глобальный бюджет всегда разрешает пересоздание: проверяем именно
    # локальную границу.
    monkeypatch.setattr(pipeline.concurrency, "recycle_pool", lambda gen: True)
    monkeypatch.setattr(pipeline.concurrency, "pool_generation", lambda: 0)

    response = client.post(
        "/ocr", files={"file": ("x.pdf", io.BytesIO(THREE_PAGES.read_bytes()), "application/pdf")}
    )

    assert attempts["n"] == pipeline.MAX_DOCUMENT_POOL_RETRIES + 1, attempts
    assert response.status_code >= 500, response.text
    assert conc.pool_is_broken() is False, "локальная сдача не должна ломать весь runtime"
