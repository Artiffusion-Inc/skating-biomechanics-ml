"""Базовый интерфейс процессора.

Abstract processor interface for extensibility.
"""

from abc import ABC, abstractmethod
from typing import Any


class Processor(ABC):
    """Базовый интерфейс процессора данных.

    Abstract processor interface for type-safe data processing pipeline.
    """

    @abstractmethod
    def process(self, **kwargs: Any) -> Any:
        """Обработать данные.

        Args:
            **kwargs: Аргументы для обработки.

        Returns:
            Результат обработки.
        """
        pass
