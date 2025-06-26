from pathlib import Path

from livekit.agents.flow.hybrid_flow import FlowRunner
from livekit.agents.flow.schema import Node


class DummySession:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def say(self, text: str) -> None:
        self.messages.append(text)


async def test_flow_runner_basic():
    path = Path("livekit-agents/livekit/agents/flow/retell.json")

    def evaluator(node: Node, user_input: str | None) -> str | None:
        if node.id == "start-node-1735257992613":
            return "node-1735258172746"
        if node.id == "node-1735258172746":
            return "node-1736564330428"
        return None

    runner = FlowRunner(str(path), evaluator)
    # manually step through a few nodes to verify parsing and evaluator
    start_node = runner.flow.nodes[runner.flow.start_node_id]
    assert start_node.instruction.text.startswith("Hello this is Retell")
    assert start_node.instruction.type == "static_text"

    next_id = evaluator(start_node, None)
    next_node = runner.flow.nodes[next_id]
    assert next_node.instruction.text.startswith("Ask user what's the reason")
    assert next_node.instruction.type == "prompt"
