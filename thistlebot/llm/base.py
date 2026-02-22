from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable


class BaseLLMClient(ABC):
    @abstractmethod
    def list_models(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def chat(self, messages: list[dict[str, str]], model: str, stream: bool = False) -> str | Iterable[str]:
        raise NotImplementedError
