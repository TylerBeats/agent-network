from tools.base import BaseTool, ToolResult


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def run(self, name: str, **kwargs) -> ToolResult:
        tool = self.get(name)
        if tool is None:
            return ToolResult(success=False, output=None, error=f"Tool '{name}' not found")
        return tool.run(**kwargs)

    def schemas(self) -> list[dict]:
        return [t.to_anthropic_schema() for t in self._tools.values()]

    @property
    def names(self) -> list[str]:
        return list(self._tools.keys())
