"""Оркестратор: PDF → Document.

Три профиля закрывают неизвестный бюджет времени: пока заказчик не назвал
допустимые секунды на документ, попадание в него обеспечивается выбором
профиля, а не переписыванием пайплайна.

    fast      tessdata_fast, 250 DPI, без OSD/deskew/бинаризации
    balanced  tessdata_fast, 300 DPI, OSD + deskew + Sauvola + проход по цифрам
    accurate  то же при 400 DPI и psm 4 — минимальная ошибка в цифрах

PaddleOCR доступен как `--engine paddle`, но в дефолтные профили не входит:
bake-off показал ошибку на цифрах в 6-11 раз выше при сопоставимом CER.

Параллелизм — по страницам через процессы, воркеры запускаются через spawn и
переводятся в однопоточный режим (`OMP_THREAD_LIMIT=1`): Tesseract иначе
берёт около 3.3 потоков на страницу и несколько воркеров дерутся за ядра.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import concurrency, raster
from .concurrency import available_cpus  # noqa: F401  — публичное имя сохранено
from .domain.fields import extract_fields, validate_text
from .ensemble import arbitrate
from .layout import xy_cut
from .model import Document, Page
from .preprocess import Prepared, prepare
from .render import PROFILES, RenderProfile, render_document, render_page

log = logging.getLogger("kzru.pipeline")

MAX_PAGES = 30
MAX_BYTES = 15 * 1024 * 1024
LOW_CONF_THRESHOLD = 0.70


@dataclass(slots=True)
class PipelineProfile:
    name: str
    engines: tuple[str, ...] = ("tesseract",)
    tessdata: str = "best"
    lang: str = "rus+kaz"
    """Языки Tesseract. Дообученная модель подключается сюда: `kzru_doc`."""
    psm: int = 6
    orientation: bool = True
    deskew: bool = True
    denoise: bool = True
    digit_pass: bool = True
    base_dpi: int = raster.BASE_DPI
    adaptive_dpi: bool = True
    paddle_limit: int = 2000
    """`text_det_limit_side_len` детектора PP-OCR. Дефолтные 960 ужимают A4."""


# Состав профилей выбран по bake-off на 8 документах (13 страниц), а не по
# ожиданиям. Актуальные результаты — ПОСЛЕ исправления удаления вертикальных
# линеек, которое перевернуло исходный порядок конфигураций (render=cells):
#
#   tess-fast-psm4-300dpi     CER 0.0190  digit 0.0195  ← лучший CER, дефолт
#   tess-fast-psm6-400dpi     CER 0.0206  digit 0.0107  ← лучшие цифры
#   ft-kzru_doc-psm4-300dpi   CER 0.0220                ← дообученная, хуже
#
# Выводы, каждый против исходного ожидания:
#
# 1. Дообученная `kzru_doc` до исправления линеек выигрывала по CER и цифрам,
#    а после — проигрывает stock (0.0220 против 0.0190). Плюс она ТЕРЯЕТ один
#    БИН из шестнадцати: читает 985435346248 как 085435346248, контрольная
#    сумма даёт несколько кандидатов, и номер по правилу владельца не
#    исправляется, а помечается. `bin` recall падает с 1.0000 до 0.9375, и
#    профиль с 400 DPI эту ошибку не выправляет. ТЗ требует безошибочные БИН,
#    суммы и даты, поэтому дефолт — stock. Дообученная доступна профилями
#    `ft-*` для перепроверки на реальных сканах.
# 2. Комбинация `kzru_doc+rus` заметно хуже одиночной `kzru_doc`: у
#    дообученной модели свой unicharset, и подмешивание stock-русского
#    только добавляет разнобоя.
# 3. PaddleOCR даёт сопоставимый общий CER, но ошибается на цифрах в 6-11 раз
#    чаще (digit 0.0788-0.1373 против 0.0125-0.0165), а ансамбль с ним
#    ухудшает CER относительно каждого движка по отдельности. Для задачи, где
#    ошибка в цифре недопустима, Paddle в дефолтный путь не входит.
#
# Все числа получены на синтетических псевдосканах. На реальных документах
# заказчика расклад может оказаться другим — это и есть причина, по которой
# альтернативы оставлены доступными, а не удалены.
PIPELINE_PROFILES: dict[str, PipelineProfile] = {
    "fast": PipelineProfile(
        name="fast",
        lang="rus+kaz",
        tessdata="fast",
        psm=6,
        orientation=False,
        deskew=False,
        denoise=False,
        digit_pass=False,
        base_dpi=250,
    ),
    "balanced": PipelineProfile(
        name="balanced",
        lang="rus+kaz",
        tessdata="fast",
        psm=6,
        base_dpi=300,
    ),
    # Точный профиль оптимизирован по цифрам, а не по общему CER: критерий
    # приёмки требует безошибочных БИН, сумм и дат.
    "accurate": PipelineProfile(
        name="accurate",
        lang="rus+kaz",
        tessdata="fast",
        psm=6,
        base_dpi=400,
    ),
    # Дообученная модель: лучший CER, но на текущем наборе теряет один БИН.
    # Проверяется на реальных сканах перед тем, как становиться дефолтом.
    "ft-balanced": PipelineProfile(
        name="ft-balanced",
        lang="kzru_doc",
        psm=6,
        base_dpi=300,
    ),
    "ft-accurate": PipelineProfile(
        name="ft-accurate",
        lang="kzru_doc",
        psm=4,
        base_dpi=400,
    ),
}

_ENGINES: dict[tuple, object] = {}


def _engine(kind: str, profile: PipelineProfile):
    """Кэш движка в пределах процесса: загрузка модели стоит секунды."""
    key = (kind, profile.tessdata, profile.lang, profile.psm, profile.paddle_limit)
    cached = _ENGINES.get(key)
    if cached is not None:
        return cached

    if kind == "tesseract":
        from .engines.tesseract import TesseractEngine

        engine = TesseractEngine(lang=profile.lang, tessdata=profile.tessdata, psm=profile.psm)
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
    pdf_backend: str | None = None,
) -> Document:
    if profile not in PIPELINE_PROFILES:
        raise ValueError(f"неизвестный профиль: {profile}")

    size = os.path.getsize(path)
    if size > MAX_BYTES:
        raise ValueError(f"файл больше лимита: {size} > {MAX_BYTES} байт")

    started = time.perf_counter()
    pipeline_profile = PIPELINE_PROFILES[profile]
    doc = Document(profile=profile)

    pdf = raster.open_pdf(path, backend=pdf_backend)
    try:
        if pdf.page_count > max_pages:
            raise ValueError(f"страниц больше лимита: {pdf.page_count} > {max_pages}")

        jobs: list[PageJob] = []
        for index in range(pdf.page_count):
            layer = raster.assess_text_layer(pdf, index) if use_text_layer else None
            if layer is not None and layer.usable:
                info = pdf.page_info(index)
                doc.pages.append(
                    Page(
                        index=index,
                        width=int(info.width_pt),
                        height=int(info.height_pt),
                        source="text_layer",
                        raw_text=layer.text,
                        engine="text_layer",
                    )
                )
                continue

            if layer is not None:
                doc.warn("text_layer_rejected", index, layer.reason)

            rp = raster.rasterize_page(
                pdf, index, base_dpi=pipeline_profile.base_dpi, adaptive=pipeline_profile.adaptive_dpi
            )
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


# Сколько раз один документ вправе пережить гибель пула. Граница локальна и
# не зависит от глобального бюджета: другой поток, успешно обработавший свой
# документ, сбрасывает общий счётчик, и без этой границы документ, который
# ломает воркера сам по себе (например, своим размером), крутился бы в
# бесконечном цикле «упал — пересоздали — упал».
MAX_DOCUMENT_POOL_RETRIES = 2


def _map_with_recovery(jobs: list[PageJob]) -> list[PageResult]:
    """Разложить страницы по общему пулу, пережив смерть воркера.

    Воркера может убить OOM killer или уронить нативная библиотека. После
    этого `ProcessPoolExecutor` необратимо переходит в состояние
    `BrokenProcessPool`, и каждый следующий запуск падает той же ошибкой —
    сервис отвечает 500 на все документы, пока его не перезапустят руками.

    Поэтому пул пересоздаётся и работа повторяется. Повтор ограничен дважды.
    Глобальный бюджет в `ocr.concurrency` защищает процесс: когда он исчерпан,
    runtime помечается нерабочим и это видно в `/readyz`. Локальный счётчик
    защищает от документа, который валит воркера сам: глобальный бюджет
    сбрасывается чужими успешными документами, и одной этой защиты мало.
    """
    attempts = 0
    while True:
        generation = concurrency.pool_generation()
        try:
            results = list(concurrency.shared_pool().map(_recognize_page, jobs))
        except BrokenProcessPool:
            attempts += 1
            if attempts > MAX_DOCUMENT_POOL_RETRIES:
                log.error("пул ломается на этом документе %s раз подряд, сдаюсь", attempts)
                raise
            log.warning(
                "пул страничных воркеров сломан, пересоздаю (поколение %s, попытка %s)",
                generation, attempts,
            )
            if not concurrency.recycle_pool(generation):
                raise
            continue
        # Бюджет пересозданий сбрасывается только после успеха: иначе редкие
        # падения, разнесённые во времени, однажды исчерпали бы лимит и
        # остановили исправный сервис.
        concurrency.note_pool_success()
        return results


def _run_jobs(jobs: list[PageJob], workers: int | None) -> list[PageResult]:
    """Распознаёт страницы, уважая общий бюджет процесса.

    `workers=None` — штатный путь: страницы уходят в общий пул из
    `ocr.concurrency`, единый на всё приложение. Параллельные запросы делят
    один бюджет ядер вместо того, чтобы каждый поднимать собственный пул, и
    не платят заново за старт воркеров на каждом документе.

    `workers=1` — последовательно в текущем процессе: одиночному Tesseract
    выгоднее занять все ядра самому (замерено: 1.64 с против 1.99 с на
    странице A4 в однопоточном режиме).

    Явное `workers > 1` — только для замеров масштабирования
    (`bench/measure_parallel.py`): поднимает отдельный пул нужного размера и
    сознательно игнорирует общий бюджет. Сервис этот путь не использует.
    """
    if not jobs:
        return []

    if workers is None:
        if len(jobs) == 1 or concurrency.shared_pool_size() == 1:
            return [_recognize_page(job) for job in jobs]
        return _map_with_recovery(jobs)

    if workers == 1:
        return [_recognize_page(job) for job in jobs]

    import multiprocessing as mp

    os.environ["OMP_THREAD_LIMIT"] = "1"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        return list(pool.map(_recognize_page, jobs))
