from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import uuid4


class MessageType(Enum):
    TASK = "task"
    RESULT = "result"
    ERROR = "error"
    BROADCAST = "broadcast"


@dataclass
class Message:
    sender: str
    recipient: str  # agent name or "broadcast"
    type: MessageType
    content: str
    metadata: dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
