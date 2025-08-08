import asyncio
import json
from pathlib import Path

import pytest

from livekit.agents.flow import FlowRunner, load_flow
from livekit.agents.flow.io import save_flow
from livekit.agents.llm import (
    LLM,
    ChatChunk,
    ChatContext,
    ChoiceDelta,
    LLMStream,
)
from livekit.agents.voice import AgentSession
from tests.fake_llm import FakeLLM
from tests.fake_stt import FakeSTT
from tests.fake_tts import FakeTTS
from tests.fake_vad import FakeVAD


def _make_min_session() -> AgentSession:
    stt = FakeSTT(fake_user_speeches=[])
    session = AgentSession(
        vad=FakeVAD(fake_user_speeches=[]),
        stt=stt,
        llm=FakeLLM(),
        tts=FakeTTS(),
        min_interruption_duration=0.05,
        min_endpointing_delay=0.05,
        max_endpointing_delay=0.5,
    )
    return session


class _OneShotLLMStream(LLMStream):
    def __init__(self, llm: "_OneShotLLM", *, chat_ctx: ChatContext, content: str):
        super().__init__(llm, chat_ctx=chat_ctx, tools=[], conn_options=None)  # type: ignore[arg-type]
        self._content = content

    async def _run(self) -> None:
        # Emit the entire content in one chunk
        self._event_ch.send_nowait(
            ChatChunk(
                id="oneshot",
                delta=ChoiceDelta(role="assistant", content=self._content, tool_calls=[]),
            )
        )


class _OneShotLLM(LLM):
    """LLM that always returns the provided JSON string in a single chunk."""

    def __init__(self, content: str):  # content should be a JSON string
        super().__init__()
        self._content = content

    def chat(self, *, chat_ctx: ChatContext, **_: object) -> LLMStream:  # type: ignore[override]
        return _OneShotLLMStream(self, chat_ctx=chat_ctx, content=self._content)


@pytest.mark.asyncio
async def test_function_node_error_edge_transition(tmp_path: Path):
    flow_dict = {
        "conversation_flow_id": "fn_error_flow",
        "version": 1,
        "global_prompt": "Test",
        "nodes": [
            {
                "id": "fn",
                "name": "Function",
                "type": "function",
                "tool_id": "will_fail",
                "wait_for_result": True,
                "instruction": {"type": "static_text", "text": "Trying..."},
                "edges": [
                    {
                        "id": "err",
                        "condition": "failed",
                        "transition_condition": {
                            "type": "equation",
                            "equations": [
                                {
                                    "left_operand": "success",
                                    "operator": "==",
                                    "right_operand": "false",
                                }
                            ],
                        },
                        "destination_node_id": "error_handler",
                    }
                ],
            },
            {
                "id": "error_handler",
                "name": "Error",
                "type": "conversation",
                "instruction": {"type": "static_text", "text": "We hit an error."},
                "edges": [],
            },
        ],
        "start_node_id": "fn",
        "tools": [
            {
                "tool_id": "will_fail",
                "name": "will_fail",
                "type": "local",
                "description": "desc",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
    }

    flow_path = tmp_path / "flow.json"
    flow_path.write_text(json.dumps(flow_dict))
    runner = FlowRunner(flow_path=str(flow_path), edge_evaluator_llm=FakeLLM())

    def _failing_handler(_: dict) -> dict:
        raise RuntimeError("boom")

    runner.register_function("will_fail", _failing_handler)

    async with _make_min_session() as sess:
        await runner.start(sess)
        await asyncio.sleep(0.1)
        assert runner.context.current_node_id == "error_handler"
        await runner.cleanup()


@pytest.mark.asyncio
async def test_loop_protection_routes_to_end(tmp_path: Path):
    # Two nodes that skip back and forth; end node present. Limit transitions to force end routing.
    flow_dict = {
        "conversation_flow_id": "loop_flow",
        "version": 1,
        "global_prompt": "Test",
        "nodes": [
            {
                "id": "a",
                "name": "A",
                "type": "conversation",
                "instruction": {"type": "static_text", "text": "A"},
                "skip_response_edge": {
                    "id": "a_to_b",
                    "condition": "skip",
                    "transition_condition": {"type": "prompt", "prompt": "always"},
                    "destination_node_id": "b",
                },
            },
            {
                "id": "b",
                "name": "B",
                "type": "conversation",
                "instruction": {"type": "static_text", "text": "B"},
                "skip_response_edge": {
                    "id": "b_to_a",
                    "condition": "skip",
                    "transition_condition": {"type": "prompt", "prompt": "always"},
                    "destination_node_id": "a",
                },
            },
            {"id": "end", "name": "End", "type": "end", "edges": []},
        ],
        "start_node_id": "a",
        "tools": [],
    }

    flow_path = tmp_path / "flow.json"
    flow_path.write_text(json.dumps(flow_dict))
    runner = FlowRunner(
        flow_path=str(flow_path), edge_evaluator_llm=FakeLLM(), max_total_transitions=2
    )

    async with _make_min_session() as sess:
        await runner.start(sess)
        # Allow transitions and loop protection to kick in
        await asyncio.sleep(0.2)
        # Should have routed to end node due to loop protection
        assert runner.context.current_node_id == "end"
        await runner.cleanup()


@pytest.mark.asyncio
async def test_prompt_min_confidence_prevents_transition():
    # Build a simple prompt edge and ensure low confidence blocks it.
    from livekit.agents.flow.base import FlowContext
    from livekit.agents.flow.fields import Edge, TransitionCondition
    from livekit.agents.flow.transition_evaluator import (
        FlowContextVariableProvider,
        TransitionEvaluator,
    )

    edges = [
        Edge(
            id="e1",
            condition="ask",
            transition_condition=TransitionCondition(type="prompt", prompt="go"),
            destination_node_id="next",
        )
    ]

    provider = FlowContextVariableProvider(FlowContext())
    # Return low-confidence selection
    llm = _OneShotLLM(content=json.dumps({"option": 1, "confidence": 0.1}))
    evaluator = TransitionEvaluator(provider, edge_llm=llm, min_prompt_confidence=0.9)
    result = await evaluator.evaluate_transitions(edges=edges, user_text="hi")
    assert result is None


def test_flow_round_trip(tmp_path: Path):
    # Load example flow, save, then load again
    example = Path(__file__).parent.parent / "examples" / "flows" / "minimal_test_flow.json"
    flow = load_flow(str(example))
    out = tmp_path / "round.json"
    save_flow(flow, str(out))
    loaded = load_flow(str(out))
    assert loaded.conversation_flow_id == flow.conversation_flow_id
    assert len(loaded.nodes) == len(flow.nodes)


@pytest.mark.asyncio
async def test_conversation_skip_response_edge_immediate_transition(tmp_path: Path):
    flow_dict = {
        "conversation_flow_id": "skip_edge_flow",
        "version": 1,
        "global_prompt": "Test",
        "nodes": [
            {
                "id": "start",
                "name": "Start",
                "type": "conversation",
                "instruction": {"type": "static_text", "text": "Hello"},
                "skip_response_edge": {
                    "id": "to_next",
                    "condition": "skip",
                    "transition_condition": {"type": "prompt", "prompt": "always"},
                    "destination_node_id": "next",
                },
            },
            {
                "id": "next",
                "name": "Next",
                "type": "conversation",
                "instruction": {"type": "static_text", "text": "Next"},
                "edges": [],
            },
        ],
        "start_node_id": "start",
        "tools": [],
    }

    flow_path = tmp_path / "flow.json"
    flow_path.write_text(json.dumps(flow_dict))
    runner = FlowRunner(flow_path=str(flow_path), edge_evaluator_llm=FakeLLM())

    async with _make_min_session() as sess:
        await runner.start(sess)
        await asyncio.sleep(0.1)
        assert runner.context.current_node_id == "next"
        await runner.cleanup()
