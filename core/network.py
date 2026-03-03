import logging

from agents.base import BaseAgent
from core.bus import MessageBus
from core.message import Message

logger = logging.getLogger(__name__)


class AgentNetwork:
    def __init__(self):
        self.bus = MessageBus()
        self._agents: dict[str, BaseAgent] = {}

    def register(self, agent: BaseAgent):
        self._agents[agent.name] = agent
        self.bus.register_agent(agent.name)
        logger.info("Registered agent: %s (%s)", agent.name, agent.role)

    def step(self):
        """Deliver pending messages to agents, then collect and forward their responses."""
        for name, agent in self._agents.items():
            for msg in self.bus.receive(name):
                agent.receive(msg)
            for outgoing in agent.process():
                self.bus.send(outgoing)

    def run(self, initial_message: Message, max_steps: int = 10) -> list[Message]:
        """Run the network until quiet or max_steps is reached."""
        self.bus.send(initial_message)
        for i in range(max_steps):
            if not self.bus.has_pending():
                logger.info("Network quiet after %d step(s)", i)
                break
            self.step()
        return self.bus.log
