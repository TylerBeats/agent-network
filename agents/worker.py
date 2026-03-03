from anthropic import Anthropic

from agents.base import BaseAgent
from core.message import Message, MessageType


class WorkerAgent(BaseAgent):
    """A general-purpose agent that handles TASK messages and returns results."""

    def __init__(self, name: str, role: str, client: Anthropic):
        super().__init__(name=name, role=role, client=client)

    def _handle(self, message: Message) -> Message | None:
        if message.type != MessageType.TASK:
            return None

        result = self._call_llm(message.content)
        recipient = message.metadata.get("original_sender", message.sender)
        return Message(
            sender=self.name,
            recipient=recipient,
            type=MessageType.RESULT,
            content=result,
        )
