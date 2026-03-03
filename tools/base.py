from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolResult:
    success: bool
    output: Any
    error: str = ""


class BaseTool(ABC):
    name: str
    description: str

    @abstractmethod
    def run(self, **kwargs) -> ToolResult:
        pass

    def to_anthropic_schema(self) -> dict:
        """Return tool definition compatible with the Anthropic tool_use API."""
        raise NotImplementedError
