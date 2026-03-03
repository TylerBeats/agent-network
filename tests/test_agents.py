from unittest.mock import MagicMock

from agents.backtester import BacktesterAgent
from agents.coordinator import CoordinatorAgent
from agents.strategy_builder import StrategyBuilderAgent
from agents.trader import TraderAgent
from agents.trading_coach import TradingCoachAgent
from agents.worker import WorkerAgent
from core.message import Message, MessageType


def _mock_client(text: str) -> MagicMock:
    client = MagicMock()
    client.messages.create.return_value.content = [MagicMock(text=text)]
    return client


# --- WorkerAgent (base behaviour) ---

def test_worker_handles_task():
    worker = WorkerAgent("trader", "execution agent", _mock_client("Done."))
    msg = Message(sender="coordinator", recipient="trader", type=MessageType.TASK, content="Execute strategy")
    worker.receive(msg)
    responses = worker.process()
    assert len(responses) == 1
    assert responses[0].type == MessageType.RESULT
    assert responses[0].sender == "trader"
    assert responses[0].recipient == "coordinator"


def test_worker_ignores_non_task():
    worker = WorkerAgent("trader", "execution agent", _mock_client(""))
    msg = Message(sender="x", recipient="trader", type=MessageType.RESULT, content="Not a task")
    worker.receive(msg)
    assert worker.process() == []


def test_worker_replies_to_original_sender():
    worker = WorkerAgent("backtester", "backtester", _mock_client("Result."))
    msg = Message(
        sender="coordinator",
        recipient="backtester",
        type=MessageType.TASK,
        content="Evaluate strategies",
        metadata={"original_sender": "user"},
    )
    worker.receive(msg)
    responses = worker.process()
    assert responses[0].recipient == "user"


# --- Specialized trading agents ---

def test_strategy_builder_agent_name():
    agent = StrategyBuilderAgent(_mock_client("10 strategies"))
    assert agent.name == "strategy_builder"


def test_backtester_agent_name():
    agent = BacktesterAgent(_mock_client("winner selected"))
    assert agent.name == "backtester"


def test_trader_agent_name():
    agent = TraderAgent(_mock_client("trade executed"))
    assert agent.name == "trader"


def test_trading_coach_agent_name():
    agent = TradingCoachAgent(_mock_client("feedback report"))
    assert agent.name == "trading_coach"


def test_strategy_builder_handles_task():
    agent = StrategyBuilderAgent(_mock_client("10 strategies generated"))
    msg = Message(
        sender="coordinator",
        recipient="strategy_builder",
        type=MessageType.TASK,
        content="Generate 10 momentum strategies",
        metadata={"original_sender": "user"},
    )
    agent.receive(msg)
    responses = agent.process()
    assert len(responses) == 1
    assert responses[0].type == MessageType.RESULT
    assert responses[0].recipient == "user"


# --- CoordinatorAgent ---

def test_coordinator_routes_task():
    routing = '{"agent": "strategy_builder", "instruction": "Generate 10 candidate strategies"}'
    coordinator = CoordinatorAgent(
        _mock_client(routing),
        worker_names=["strategy_builder", "backtester", "trader", "trading_coach"],
    )
    msg = Message(sender="user", recipient="coordinator", type=MessageType.TASK, content="Start a new strategy cycle")
    coordinator.receive(msg)
    responses = coordinator.process()
    assert len(responses) == 1
    assert responses[0].recipient == "strategy_builder"
    assert responses[0].type == MessageType.TASK


def test_coordinator_returns_error_on_bad_json():
    coordinator = CoordinatorAgent(
        _mock_client("not json"),
        worker_names=["strategy_builder", "backtester", "trader", "trading_coach"],
    )
    msg = Message(sender="user", recipient="coordinator", type=MessageType.TASK, content="Do something")
    coordinator.receive(msg)
    responses = coordinator.process()
    assert responses[0].type == MessageType.ERROR
