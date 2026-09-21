"""Оркестратор: PDF → Document.

Три профиля закрывают неизвестный бюджет времени: пока заказчик не назвал
допустимые секунды на документ, попадание в него обеспечивается выбором
профиля, а не переписыванием пайплайна.

    fast      один движок, tessdata_fast, без OSD и без бинаризации Sauvola
    balanced  один движок, tessdata_best, OSD + deskew + Sauvola
    accurate  два движка с арбитражем + доуточняющий проход по цифрам

Параллелизм — по страницам через процессы: и Tesseract, и PaddleOCR внутри
себя масштабируются плохо, поэтому в воркере OMP_NUM_THREADS=1, а выигрыш
берётся числом процессов.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import raster
from .domain.fields import extract_fields, validate_text
from .ensemble import arbitrate
from .layout import xy_cut
from .model import Document, Page
from .preprocess import Prepared, prepare
from .render import PROFILES, RenderProfile, render_document, render_page

MAX_PAGES = 30
MAX_BYTES = 15 * 1024 * 1024
LOW_CONF_THRESHOLD = 0.70


@dataclass(slots=True)
class PipelineProfile:
    name: str
    engines: tuple[str, ...] = ("tesseract",)
    tessdata: str = "best"
    psm: int = 6
    orientation: bool = True
    deskew: bool = True
    denoise: bool = True
    digit_pass: bool = True
    base_dpi: int = raster.BASE_DPI
    adaptive_dpi: bool = True
    paddle_limit: int = 2000
    """`text_det_limit_side_len` детектора PP-OCR. Дефолтные 960 ужимают A4."""


PIPELINE_PROFILES: dict[str, PipelineProfile] = {
    "fast": PipelineProfile(
        name="fast",
        tessdata="fast",
        orientation=False,
        deskew=False,
        denoise=False,
        digit_pass=False,
        base_dpi=250,
    ),
    "balanced": PipelineProfile(name="balanced"),
    "accurate": PipelineProfile(name="accurate", engines=("tesseract", "paddle")),
}

_ENGINES: dict[tuple, object] = {}


def _engine(kind: str, profile: PipelineProfile):
    """Кэш движка в пределах процесса: загрузка модели стоит секунды."""
    key = (kind, profile.tessdata, profile.psm, profile.paddle_limit)
    cached = _ENGINES.get(key)
    if cached is not None:
        return cached

    if kind == "tesseract":
        from .engines.tesseract import TesseractEngine

        engine = TesseractEngine(tessdata=profile.tessdata, psm=profile.psm)
    elif kind == "paddle":
        from .engines.paddle import PaddleEngine

        engine = PaddleEngine(limit_side_len=profile.paddle_limit)
    else:
        raise ValueError(f"неизвестный движок: {kind}")

    _ENGINES[key] = engine
    return engine


@dataclass(slots=True)
class PageJob:
    index: int
    image: np.ndarray
    dpi: int
    profile: PipelineProfile
    """Сам профиль, а не имя: при старте воркеров через spawn глобальный
    реестр PIPELINE_PROFILES в дочернем процессе пуст, и кастомные профили
    bake-off по имени не нашлись бы."""


@dataclass(slots=True)
class PageResult:
    index: int
    page: Page
    disagreements: list[str] = field(default_factory=list)
    error: str = ""


def _recognize_page(job: PageJob) -> PageResult:
    # OpenCV держит собственный пул потоков, который OMP_NUM_THREADS не
    # контролирует. Без этого каждый воркер поднимает пул на все ядра, и N
    # воркеров дают N×ядер потоков: машина уходит в переподписку, а время
    # растёт вместо падения.
    cv2.setNumThreads(1)

    profile = job.profile
    try:
        primary_kind = profile.engines[0]
        engine = _engine(primary_kind, profile)

        rotation = 0
        if profile.orientation and primary_kind == "tesseract":
            rotation = engine.detect_orientation(job.image)

        prepared: Prepared = prepare(
            job.image, rotation=rotation, denoise=profile.denoise, do_deskew=profile.deskew
        )
        lines = engine.recognize(prepared)
        disagreements: list[str] = []

        if len(profile.engines) > 1:
            secondary_lines: list = []
            for kind in profile.engines[1:]:
                try:
                    secondary_lines = _engine(kind, profile).recognize(prepared)
                except Exception as exc:  # движок недоступен — деградируем, но сообщаем
                    disagreements.append(f"движок {kind} недоступен: {exc}")
                    continue
            if secondary_lines:
                lines, extra = arbitrate(lines, secondary_lines)
                disagreements.extend(extra)

        if profile.digit_pass and primary_kind == "tesseract":
            _refine_digits(engine, prepared, lines)

        page = Page(
            index=job.index,
            width=prepared.gray.shape[1],
            height=prepared.gray.shape[0],
            blocks=xy_cut(lines),
            source="ocr",
            rotation=rotation,
            skew=prepared.skew,
            dpi=job.dpi,
            engine="+".join(profile.engines),
        )
        return PageResult(index=job.index, page=page, disagreements=disagreements)
    except Exception as exc:
        empty = Page(index=job.index, source="ocr", dpi=job.dpi)
        return PageResult(index=job.index, page=empty, error=f"{type(exc).__name__}: {exc}")


def _refine_digits(engine, prepared: Prepared, lines: list) -> None:
    """Перечитывает числовые слова кропом с ограничением алфавита.

    Заменяет текст только если результат — те же по количеству цифры, но
    прочитанные увереннее; иначе оставляет исходный вариант. Цель — снять
    путаницу `0`/`O` и `1`/`l`, а не переписать строку заново.
    """
    import re

    digit_re = re.compile(r"^[\d\s.,]{4,}$")
    pad = 4
    h, w = prepared.binary.shape

    for line in lines:
        for word in line.words:
            if not digit_re.match(word.text) or word.conf > 0.92:
                continue
            y0 = max(0, word.y0 - pad)
            y1 = min(h, word.y1 + pad)
            x0 = max(0, word.x0 - pad)
            x1 = min(w, word.x1 + pad)
            crop = prepared.binary[y0:y1, x0:x1]
            refined = engine.read_digits(crop)
            if not refined:
                continue
            digits_before = sum(ch.isdigit() for ch in word.text)
            digits_after = sum(ch.isdigit() for ch in refined)
            if digits_after == digits_before and refined != word.text:
                word.text = refined


def process_pdf(
    path: str,
    profile: str = "balanced",
    render: str | RenderProfile = "default",
    workers: int | None = None,
    max_pages: int = MAX_PAGES,
    use_text_layer: bool = True,
) -> Document:
    if profile not in PIPELINE_PROFILES:
        raise ValueError(f"неизвестный профиль: {profile}")

    size = os.path.getsize(path)
    if size > MAX_BYTES:
        raise ValueError(f"файл больше лимита: {size} > {MAX_BYTES} байт")

    started = time.perf_counter()
    pipeline_profile = PIPELINE_PROFILES[profile]
    doc = Document(profile=profile)

    pdf = raster.open_pdf(path)
    try:
        if pdf.page_count > max_pages:
            raise ValueError(f"страниц больше лимита: {pdf.page_count} > {max_pages}")

        jobs: list[PageJob] = []
        for page in pdf:
            layer = raster.assess_text_layer(page) if use_text_layer else None
            if layer is not None and layer.usable:
                doc.pages.append(
                    Page(
                        index=layer.index,
                        width=int(page.rect.width),
                        height=int(page.rect.height),
                        source="text_layer",
                        raw_text=layer.text,
                        engine="text_layer",
                    )
                )
                continue

            if layer is not None:
                doc.warn("text_layer_rejected", layer.index, layer.reason)

            rp = raster.rasterize_page(page, base_dpi=pipeline_profile.base_dpi, adaptive=pipeline_profile.adaptive_dpi)
            jobs.append(PageJob(index=rp.index, image=rp.image, dpi=rp.dpi, profile=pipeline_profile))
    finally:
        pdf.close()

    results = _run_jobs(jobs, workers)

    for res in results:
        doc.pages.append(res.page)
        if res.error:
            doc.warn("page_failed", res.index, res.error)
        for msg in res.disagreements:
            doc.warn("engine_disagreement", res.index, msg)

    doc.pages.sort(key=lambda p: p.index)

    render_profile = PROFILES[render] if isinstance(render, str) else render
    for page in doc.pages:
        if page.source == "ocr" and page.blocks and page.mean_conf < LOW_CONF_THRESHOLD:
            doc.warn("low_confidence_page", page.index, f"средняя уверенность {page.mean_conf:.2f}")
        for wtype, detail in validate_text(render_page(page, render_profile)):
            doc.warn(wtype, page.index, detail)

    doc.fields = extract_fields(render_document(doc, render_profile))
    doc.elapsed_s = round(time.perf_counter() - started, 3)
    return doc


def available_cpus() -> int:
    """Сколько ядер реально доступно процессу.

    `os.cpu_count()` возвращает число ядер машины и игнорирует и affinity, и
    cgroup-квоту. В контейнере с `--cpus=2` на 8-ядерном хосте это приводит к
    запуску 7 процессов на 2 ядра: они конкурируют, и обработка становится
    медленнее последовательной. Проверено замером: 8 страниц при 7 воркерах на
    одном доступном ядре — 46 с против 14.6 с в один процесс.
    """
    limits: list[int] = []

    try:
        limits.append(len(os.sched_getaffinity(0)))
    except AttributeError:  # не Linux
        pass

    # cgroup v2
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            limits.append(max(1, int(float(quota) / float(period))))
    except (OSError, ValueError):
        pass

    # cgroup v1
    try:
        quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if quota > 0 and period > 0:
            limits.append(max(1, quota // period))
    except (OSError, ValueError):
        pass

    if not limits:
        limits.append(os.cpu_count() or 1)
    return max(1, min(limits))


def _run_jobs(jobs: list[PageJob], workers: int | None) -> list[PageResult]:
    if not jobs:
        return []
    if workers is None:
        workers = max(1, min(len(jobs), available_cpus()))
    if workers == 1:
        # Один воркер — пусть Tesseract использует все ядра сам: замерено
        # wall 1.64 с против 1.99 с в однопоточном режиме на странице A4.
        return [_recognize_page(job) for job in jobs]

    # Tesseract ИГНОРИРУЕТ OMP_NUM_THREADS и берёт около 3.3 потоков на
    # страницу (wall 1.64 с при 5.43 с CPU). Ограничивает его только
    # OMP_THREAD_LIMIT. Без этого N воркеров требуют 3.3*N ядер, и на
    # 4-ядерной машине параллелизм замедляет работу вместо ускорения.
    os.environ["OMP_THREAD_LIMIT"] = "1"
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    # Контекст spawn, а не fork: родительский процесс уже работал с OpenCV
    # (растеризация, оценка масштаба), и fork копирует его пул потоков в
    # неконсистентном состоянии — воркеры зависают вместо работы.
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        return list(pool.map(_recognize_page, jobs))
