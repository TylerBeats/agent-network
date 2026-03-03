from collections import defaultdict

from core.message import Message, MessageType


class MessageBus:
    def __init__(self):
        self._queues: dict[str, list[Message]] = defaultdict(list)
        self._agents: set[str] = set()
        self._log: list[Message] = []

    def register_agent(self, name: str):
        self._agents.add(name)

    def send(self, message: Message):
        self._log.append(message)
        if message.type == MessageType.BROADCAST:
            for agent in self._agents:
                self._queues[agent].append(message)
        else:
            self._queues[message.recipient].append(message)

    def receive(self, agent_name: str) -> list[Message]:
        messages = list(self._queues[agent_name])
        self._queues[agent_name].clear()
        return messages

    def has_pending(self) -> bool:
        return any(self._queues[name] for name in self._agents)

    @property
    def log(self) -> list[Message]:
        return list(self._log)
