import json
import logging

from anthropic import Anthropic

from agents.base import BaseAgent
from core.message import Message, MessageType

logger = logging.getLogger(__name__)


class CoordinatorAgent(BaseAgent):
    """Routes incoming tasks to the appropriate trading network agent."""

    def __init__(self, client: Anthropic, worker_names: list[str]):
        super().__init__(
            name="coordinator",
            role=(
                "coordinator for an autonomous trading network comprising four specialist agents: "
                "strategy_builder (generates batches of 10 candidate trading strategies as JSON), "
                "backtester (evaluates all 10 strategies against historical data, applies hard "
                "elimination filters, and selects the single best performer), "
                "trader (executes the winning strategy faithfully in live or paper markets with "
                "strict risk controls), and "
                "trading_coach (monitors live performance weekly and monthly, adjusts risk "
                "exposure, and feeds structured insights back to the strategy_builder). "
                "You route each incoming task to the correct specialist and provide a clear, "
                "detailed instruction tailored to that agent's responsibilities."
            ),
            client=client,
        )
        self.worker_names = worker_names

    def _handle(self, message: Message) -> Message | None:
        if message.type not in (MessageType.TASK, MessageType.BROADCAST):
            return None

        prompt = (
            f"Available specialist agents: {', '.join(self.worker_names)}.\n\n"
            f"Task: {message.content}\n\n"
            "Which agent should handle this? Reply with JSON only: "
            '{"agent": "<name>", "instruction": "<detailed instruction>"}'
        )

        response_text = self._call_llm(prompt)

        try:
            data = json.loads(response_text)
            logger.info("Coordinator routing to %s", data["agent"])
            return Message(
                sender=self.name,
                recipient=data["agent"],
                type=MessageType.TASK,
                content=data["instruction"],
                metadata={"original_sender": message.sender},
            )
        except (json.JSONDecodeError, KeyError) as exc:
            logger.error("Coordinator failed to parse routing response: %s", exc)
            return Message(
                sender=self.name,
                recipient=message.sender,
                type=MessageType.ERROR,
                content=f"Routing failed: {response_text}",
            )
