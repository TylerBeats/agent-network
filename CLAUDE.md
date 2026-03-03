# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Fill in ANTHROPIC_API_KEY in .env
```

## Commands

```bash
# Run the network with the default task
python main.py

# Run all tests
pytest

# Run a single test file
pytest tests/test_message_bus.py

# Run a single test
pytest tests/test_agents.py::test_worker_handles_task
```

## Architecture

The network is a **synchronous, step-driven message-passing system**. Agents never call each other directly — all communication goes through the `MessageBus`.

### Core flow

```
user → MessageBus → CoordinatorAgent
                  → WorkerAgent → MessageBus → (back to original sender)
```

`AgentNetwork.run()` calls `step()` in a loop. Each `step()` drains each agent's queue from the bus, feeds messages to agents, then re-sends any outgoing responses back onto the bus. The loop stops when the bus has no pending messages or `max_steps` is reached.

### Key files

| File | Responsibility |
|---|---|
| `core/message.py` | `Message` dataclass + `MessageType` enum |
| `core/bus.py` | `MessageBus` — routes messages by recipient name, supports broadcast |
| `core/network.py` | `AgentNetwork` — owns the bus and agents, drives the step loop |
| `agents/base.py` | `BaseAgent` — inbox/process/_handle interface, LLM call with memory |
| `agents/coordinator.py` | Routes tasks to workers via LLM-generated JSON `{"agent":…,"instruction":…}` |
| `agents/worker.py` | Handles TASK messages, replies RESULT to `metadata["original_sender"]` |
| `tools/base.py` | `BaseTool` ABC + `ToolResult` dataclass |
| `tools/registry.py` | `ToolRegistry` — register, run, and expose Anthropic schemas for tools |
| `config/settings.py` | Reads `.env` via `python-dotenv` |

### Extending the network

**Add a new agent type** — subclass `BaseAgent` and implement `_handle(message) -> Message | None`. Register it with `network.register()`.

**Add a tool** — subclass `BaseTool`, implement `run()` and `to_anthropic_schema()`, then register it with a `ToolRegistry` instance. Wire the registry into an agent's `_handle` method to enable tool-use calls.

**Change the model** — set `DEFAULT_MODEL` in `.env`. The value is used by every `BaseAgent._call_llm()` call.
