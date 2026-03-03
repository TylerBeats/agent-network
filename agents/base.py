from anthropic import Anthropic

from config.settings import DEFAULT_MODEL
from core.message import Message, MessageType


class BaseAgent:
    def __init__(self, name: str, role: str, client: Anthropic):
        self.name = name
        self.role = role
        self.client = client
        self.memory: list[dict] = []  # conversation history sent to the LLM
        self._inbox: list[Message] = []
        self._last_input_tokens:  int = 0
        self._last_output_tokens: int = 0

    def receive(self, message: Message):
        self._inbox.append(message)

    def process(self) -> list[Message]:
        """Drain the inbox and return any outgoing messages."""
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
        response = self.client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=4096,
            system=f"You are {self.name}, a {self.role}.",
            messages=self.memory,
        )
        self._last_input_tokens  = response.usage.input_tokens
        self._last_output_tokens = response.usage.output_tokens
        text = response.content[0].text
        self.memory.append({"role": "assistant", "content": text})
        return text
