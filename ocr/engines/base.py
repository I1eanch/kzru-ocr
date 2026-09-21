"""Контракт движка распознавания.

Движок получает подготовленную страницу и отдаёт плоский список строк с
координатами и уверенностью. Порядок чтения, блоки и текст — не его дело:
этим занимаются layout и render.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..model import Line
from ..preprocess import Prepared


@runtime_checkable
class Engine(Protocol):
    name: str

    def recognize(self, prepared: Prepared) -> list[Line]:
        ...


class EngineUnavailable(RuntimeError):
    """Движок не установлен в этом окружении."""
