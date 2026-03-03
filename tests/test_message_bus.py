from core.bus import MessageBus
from core.message import Message, MessageType


def _msg(sender="a", recipient="b", type=MessageType.TASK, content="hello"):
    return Message(sender=sender, recipient=recipient, type=type, content=content)


def test_send_and_receive():
    bus = MessageBus()
    bus.register_agent("b")
    bus.send(_msg())
    received = bus.receive("b")
    assert len(received) == 1
    assert received[0].content == "hello"


def test_receive_clears_queue():
    bus = MessageBus()
    bus.register_agent("b")
    bus.send(_msg())
    bus.receive("b")
    assert bus.receive("b") == []


def test_broadcast_reaches_all():
    bus = MessageBus()
    bus.register_agent("a")
    bus.register_agent("b")
    bus.send(_msg(sender="x", recipient="broadcast", type=MessageType.BROADCAST, content="hi"))
    assert len(bus.receive("a")) == 1
    assert len(bus.receive("b")) == 1


def test_has_pending():
    bus = MessageBus()
    bus.register_agent("b")
    assert not bus.has_pending()
    bus.send(_msg())
    assert bus.has_pending()
    bus.receive("b")
    assert not bus.has_pending()


def test_log_accumulates():
    bus = MessageBus()
    bus.register_agent("b")
    bus.send(_msg(content="first"))
    bus.send(_msg(content="second"))
    assert len(bus.log) == 2
