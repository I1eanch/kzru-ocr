"""Глобальный бюджет параллелизма процесса.

Два разных дефекта решаются одной конструкцией.

Первый: без общего бюджета каждый запрос создавал собственный
`ProcessPoolExecutor` на `available_cpus()` воркеров. Два параллельных `/ocr`
плюс два фоновых задания давали четырёхкратную переподписку хоста — при том
что внутри воркера Tesseract и так занимает ядро целиком.

Второй: пул создавался и уничтожался на каждый документ. Старт воркера через
`spawn` — это новый интерпретатор с импортом numpy и OpenCV, десятки
миллисекунд на воркер, которые платились заново на каждом документе.

Поэтому пул один на процесс, создаётся лениво и переиспользуется, а его
размер и есть бюджет страничных воркеров. Дополнительно число одновременно
обрабатываемых документов ограничено семафором: страницы растеризуются в
родительском процессе до отправки в пул, и без этого предела память растёт
пропорционально числу параллельных запросов.

Настраивается переменными окружения:

    KZRU_PAGE_WORKERS   размер общего пула; по умолчанию доступные ядра
    KZRU_MAX_DOCUMENTS  одновременно обрабатываемых документов; по умолчанию 2
"""

from __future__ import annotations

import atexit
import math
import os
import threading
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path

DEFAULT_MAX_DOCUMENTS = 2

# Максимум пересозданий пула подряд. Смысл границы: одиночная смерть воркера
# (OOM killer, падение нативной библиотеки) обязана лечиться сама, а
# устойчиво воспроизводящийся сбой — становиться видимым отказом, а не
# бесконечным циклом перезапусков, который выглядит как исправный сервис.
MAX_POOL_RESTARTS = 3

_pool: ProcessPoolExecutor | None = None
_pool_size = 0
_pool_lock = threading.RLock()
_pool_generation = 0
_pool_restarts = 0
_pool_broken = False

_doc_semaphore: threading.BoundedSemaphore | None = None
_doc_lock = threading.Lock()


class CapacityExceeded(RuntimeError):
    """Свободных слотов обработки нет и ожидание превысило лимит."""


def available_cpus() -> int:
    """Сколько ядер реально доступно процессу.

    `os.cpu_count()` возвращает число ядер машины и игнорирует и affinity, и
    cgroup-квоту. В контейнере с `--cpus=2` на 8-ядерном хосте это приводит к
    запуску лишних процессов, которые конкурируют между собой.

    `nproc` для этой цели не годится: coreutils уважает `OMP_NUM_THREADS`,
    который в образе выставлен в 1, и вернёт 1 на любой машине.
    """
    limits: list[int] = []

    try:
        limits.append(len(os.sched_getaffinity(0)))
    except AttributeError:  # не Linux
        pass

    try:  # cgroup v2
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            limits.append(max(1, math.floor(float(quota) / float(period))))
    except (OSError, ValueError):
        pass

    try:  # cgroup v1
        quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if quota > 0 and period > 0:
            limits.append(max(1, quota // period))
    except (OSError, ValueError):
        pass

    if not limits:
        limits.append(os.cpu_count() or 1)
    return max(1, min(limits))


def page_worker_budget() -> int:
    """Размер общего пула страничных воркеров."""
    raw = os.environ.get("KZRU_PAGE_WORKERS")
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return available_cpus()


def max_documents() -> int:
    raw = os.environ.get("KZRU_MAX_DOCUMENTS")
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return DEFAULT_MAX_DOCUMENTS


def shared_pool() -> ProcessPoolExecutor:
    """Общий пул процессов на всё приложение.

    Контекст spawn, а не fork: родитель к этому моменту уже работал с OpenCV,
    и fork копирует его пул потоков в неконсистентном состоянии — воркеры
    зависают вместо работы.
    """
    global _pool, _pool_size

    with _pool_lock:
        if _pool is None:
            import multiprocessing as mp

            _pool_size = page_worker_budget()
            # Воркеры однопоточные: Tesseract игнорирует OMP_NUM_THREADS и
            # берёт около 3.3 потоков на страницу, ограничивает его только
            # OMP_THREAD_LIMIT.
            os.environ["OMP_THREAD_LIMIT"] = "1"
            os.environ.setdefault("OMP_NUM_THREADS", "1")
            _pool = ProcessPoolExecutor(
                max_workers=_pool_size,
                mp_context=mp.get_context("spawn"),
            )
        return _pool


def shared_pool_size() -> int:
    """Размер бюджета воркеров.

    Намеренно не создаёт пул: на inline-пути (бюджет 1) пул не нужен вовсе,
    а создание меняло бы OMP-переменные окружения процесса как побочный
    эффект простого запроса размера.
    """
    with _pool_lock:
        if _pool is not None:
            return _pool_size
    return page_worker_budget()


def pool_generation() -> int:
    """Номер поколения пула; растёт при каждом пересоздании."""
    with _pool_lock:
        return _pool_generation


def pool_is_broken() -> bool:
    """Пул сломан и исчерпал попытки восстановления.

    Читается проверкой готовности: рекламировать работающий runtime, когда
    каждый запуск страницы падает, недопустимо.
    """
    with _pool_lock:
        return _pool_broken


def recycle_pool(seen_generation: int) -> bool:
    """Пересоздать пул после гибели воркера.

    `ProcessPoolExecutor` необратим: после `BrokenProcessPool` (воркера убил
    OOM killer или он упал в нативном коде) любой последующий `submit` падает
    той же ошибкой навсегда. Без пересоздания один OOM превращает сервис в
    вечные 500 при живой проверке готовности.

    `seen_generation` — поколение, на котором вызывающий получил ошибку. Если
    пул уже пересоздан кем-то другим, работа не дублируется, и параллельные
    страницы одного документа не устраивают гонку пересозданий.

    Возвращает False, когда попытки исчерпаны: тогда runtime считается
    нерабочим до вмешательства оператора, и это видно в `/readyz`.
    """
    global _pool, _pool_size, _pool_generation, _pool_restarts, _pool_broken

    with _pool_lock:
        if seen_generation != _pool_generation:
            return not _pool_broken  # уже пересоздан другим вызывающим
        if _pool_restarts >= MAX_POOL_RESTARTS:
            _pool_broken = True
            return False
        old = _pool
        _pool = None
        _pool_size = 0
        _pool_generation += 1
        _pool_restarts += 1
    if old is not None:
        # Вне замка: shutdown сломанного пула может блокировать.
        old.shutdown(wait=False, cancel_futures=True)
    return True



def discard_pool(seen_generation: int, refund: int = 0) -> None:
    """Выбросить сломанный пул, не расходуя бюджет восстановления.

    Вызывается, когда обработку прекращает сам документ: пул ломается именно
    на нём, локальная граница повторов исчерпана, и дальше пробовать
    бессмысленно. Два свойства обязательны.

    Первое: сломанный экземпляр нельзя оставлять установленным. Иначе
    следующий документ получит его же, немедленно упадёт и потратит ещё одно
    пересоздание из общего бюджета — отравленный файл утаскивал бы за собой
    исправные.

    Второе: пересоздания, потраченные на этот документ, возвращаются
    (`refund`). Общий бюджет существует, чтобы отличить единичный OOM от
    неисправного окружения; расход на заведомо проблемный файл к этому
    отношения не имеет, и без возврата пара таких файлов пометила бы
    работоспособный runtime нерабочим.

    Состояние `_pool_broken` не меняется: сдача на одном документе — не
    приговор процессу.
    """
    global _pool, _pool_size, _pool_generation, _pool_restarts

    with _pool_lock:
        if seen_generation != _pool_generation:
            return  # пул уже сменили; чужой экземпляр трогать нельзя
        old = _pool
        _pool = None
        _pool_size = 0
        _pool_generation += 1
        if refund:
            _pool_restarts = max(0, _pool_restarts - refund)
    if old is not None:
        old.shutdown(wait=False, cancel_futures=True)


def note_pool_success() -> None:
    """Документ обработан целиком: бюджет пересозданий восстановлен.

    Иначе редкие падения, разнесённые на недели, однажды исчерпали бы лимит
    и остановили исправный сервис.
    """
    global _pool_restarts
    with _pool_lock:
        _pool_restarts = 0


def shutdown_pool() -> None:
    global _pool, _pool_size
    with _pool_lock:
        if _pool is not None:
            _pool.shutdown(wait=False, cancel_futures=True)
            _pool = None
            _pool_size = 0


def _semaphore() -> threading.BoundedSemaphore:
    global _doc_semaphore
    with _doc_lock:
        if _doc_semaphore is None:
            _doc_semaphore = threading.BoundedSemaphore(max_documents())
        return _doc_semaphore


@contextmanager
def document_slot(timeout: float | None = None):
    """Слот на обработку одного документа.

    `timeout` в секундах; при его истечении поднимается `CapacityExceeded`,
    которую HTTP-слой отображает в 503 с `Retry-After`. Ждать бесконечно
    нельзя: очередь запросов будет расти, пока не кончится память.
    """
    sem = _semaphore()
    acquired = sem.acquire(timeout=timeout) if timeout is not None else sem.acquire()
    if not acquired:
        raise CapacityExceeded(f"нет свободных слотов обработки в течение {timeout} с")
    try:
        yield
    finally:
        sem.release()


atexit.register(shutdown_pool)
