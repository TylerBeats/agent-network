import os
from config.settings import DEFAULT_MODEL
from core.message import Message
class BaseAgent:
    def __init__(self, name: str, role: str, client):
        self.name = name
        self.role = role
        self.client = client
        self.memory: list[dict] = []
        self._inbox: list[Message] = []
        self._last_input_tokens:  int = 0
        self._last_output_tokens: int = 0
        self._total_input_tokens:  int = 0
        self._total_output_tokens: int = 0
    def receive(self, message: Message):
        self._inbox.append(message)
    def process(self) -> list[Message]:
        responses = []
        while self._inbox:
            msg = self._inbox.pop(0)
            response = self._handle(msg)
            if response is not None:
                responses.append(response)
        return responses
    def _handle(self, message: Message) -> Message | None:
        raise NotImplementedError
    def _call_llm(self, prompt: str) -> str:
        self.memory.append({"role": "user", "content": prompt})
        if os.getenv("LOCAL_LLM", "false").lower() == "true":
            response = self.client.chat.completions.create(
                model=os.getenv("LOCAL_LLM_MODEL", DEFAULT_MODEL),
                max_tokens=8192,
                timeout=600,
                messages=[
                    {"role": "system", "content": f"You are {self.name}, a {self.role}."},
                    *self.memory,
                ],
            )
            text = response.choices[0].message.content
            try:
                self._last_input_tokens  = response.usage.prompt_tokens
                self._last_output_tokens = response.usage.completion_tokens
                self._total_input_tokens  += response.usage.prompt_tokens
                self._total_output_tokens += response.usage.completion_tokens
            except (AttributeError, TypeError):
                pass
        else:
            response = self.client.messages.create(
                model=DEFAULT_MODEL,
                max_tokens=8192,
                system=f"You are {self.name}, a {self.role}.",
                messages=self.memory,
            )
            text = response.content[0].text
            try:
                self._last_input_tokens  = response.usage.input_tokens
                self._last_output_tokens = response.usage.output_tokens
                self._total_input_tokens  += int(response.usage.input_tokens)
                self._total_output_tokens += int(response.usage.output_tokens)
            except (AttributeError, TypeError, ValueError):
                pass
        self.memory.append({"role": "assistant", "content": text})
        return text
