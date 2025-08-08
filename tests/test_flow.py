import asyncio
import json
from pathlib import Path

import pytest

from livekit.agents.flow import FlowRunner, FlowSpec, load_flow
from livekit.agents.voice import AgentSession
from tests.fake_llm import FakeLLM
from tests.fake_stt import FakeSTT
from tests.fake_tts import FakeTTS
from tests.fake_vad import FakeVAD


def _write_flow(tmp_path: Path, flow_dict: dict) -> str:
    path = tmp_path / "test_flow.json"
    path.write_text(json.dumps(flow_dict, indent=2))
    return str(path)


def _make_min_session() -> AgentSession:
    # Minimal session suitable for flow agent entry/transition tests
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


@pytest.mark.asyncio
async def test_flow_spec_parse_and_validate(tmp_path: Path):
    flow_dict = {
        "conversation_flow_id": "schema_ok",
        "version": 1,
        "global_prompt": "Test",
        "nodes": [
            {
                "id": "start",
                "name": "Start",
                "type": "conversation",
                "instruction": {"type": "static_text", "text": "Hello"},
                "edges": [
                    {
                        "id": "to_end",
                        "condition": "done",
                        "transition_condition": {
                            "type": "equation",
                            "equations": [
                                {"left_operand": "x", "operator": "==", "right_operand": "x"}
                            ],
                        },
                        "destination_node_id": "end",
                    }
                ],
            },
            {
                "id": "end",
                "name": "End",
                "type": "conversation",
                "instruction": {"type": "static_text", "text": "Bye"},
                "edges": [],
            },
        ],
        "start_node_id": "start",
        "tools": [],
    }

    flow = FlowSpec.from_dict(flow_dict)
    assert flow.start_node_id == "start"
    assert not flow.validate_flow_structure()

    # Also exercise load_flow path
    flow_path = _write_flow(tmp_path, flow_dict)
    loaded = load_flow(flow_path)
    assert isinstance(loaded, FlowSpec)
    assert loaded.nodes["start"].name == "Start"


@pytest.mark.asyncio
async def test_function_node_equation_transition(tmp_path: Path):
    # Flow: function node evaluates, then transitions to next via equation on result
    flow_dict = {
        "conversation_flow_id": "fn_equation_flow",
        "version": 1,
        "global_prompt": "Test",
        "nodes": [
            {
                "id": "fn",
                "name": "Function",
                "type": "function",
                "tool_id": "mytool",
                "instruction": {"type": "static_text", "text": "Processing..."},
                "wait_for_result": True,
                "edges": [
                    {
                        "id": "ok",
                        "condition": "ok",
                        "transition_condition": {
                            "type": "equation",
                            "equations": [
                                {
                                    "left_operand": "result_email",
                                    "operator": "==",
                                    "right_operand": "customer_email",
                                }
                            ],
                        },
                        "destination_node_id": "done",
                    }
                ],
            },
            {
                "id": "done",
                "name": "Done",
                "type": "conversation",
                "instruction": {"type": "static_text", "text": "All set"},
                "edges": [],
            },
        ],
        "start_node_id": "fn",
        "tools": [
            {
                "tool_id": "mytool",
                "name": "mytool",
                "type": "local",
                "description": "desc",
                "parameters": {
                    "type": "object",
                    "properties": {"customer_email": {"type": "string"}},
                    "required": ["customer_email"],
                },
            }
        ],
    }

    flow_path = _write_flow(tmp_path, flow_dict)
    runner = FlowRunner(flow_path=flow_path, edge_evaluator_llm=FakeLLM())

    # Seed variable used by function args and equation
    runner.context.set_variable("customer_email", "user@example.com")

    # Register local handler that mirrors the email back as result_email
    def _handler(args: dict):
        return {"result_email": args.get("customer_email")}

    runner.register_function("mytool", _handler)

    async with _make_min_session() as sess:
        await runner.start(sess)
        # Allow function execution and transition to occur
        await asyncio.sleep(0.1)

        # Expect we transitioned to "done"
        assert runner.context.current_node_id == "done"
        assert runner.context.execution_path[:2] == ["fn", "done"]

        # Ensure cleanup completes to avoid leaked tasks
        await runner.cleanup()


@pytest.mark.asyncio
async def test_gather_node_prefilled_transition(tmp_path: Path):
    # Flow: gather node with required email -> auto-transition to next when prefilled
    flow_dict = {
        "conversation_flow_id": "gather_prefilled",
        "version": 1,
        "global_prompt": "Test",
        "nodes": [
            {
                "id": "gather",
                "name": "Gather",
                "type": "gather_input",
                "gather_input_instruction": "I need your email.",
                "gather_input_variables": [
                    {
                        "name": "customer_email",
                        "type": "email",
                        "description": "Email",
                        "required": True,
                    }
                ],
                "edges": [
                    {
                        "id": "to_next",
                        "condition": "collected",
                        "transition_condition": {"type": "prompt", "prompt": "after gather"},
                        "destination_node_id": "next",
                    }
                ],
            },
            {
                "id": "next",
                "name": "Next",
                "type": "conversation",
                "instruction": {"type": "static_text", "text": "Proceed"},
                "edges": [],
            },
        ],
        "start_node_id": "gather",
        "tools": [],
    }

    flow_path = _write_flow(tmp_path, flow_dict)
    runner = FlowRunner(flow_path=flow_path, edge_evaluator_llm=FakeLLM())

    # Prefill gathered data so the node completes immediately on enter
    runner.context.set_variable("gather_data_gather", {"customer_email": "user@example.com"})

    async with _make_min_session() as sess:
        await runner.start(sess)
        # Allow on_enter to process and transition
        await asyncio.sleep(0.1)

        assert runner.context.current_node_id == "next"
        assert runner.context.execution_path[:2] == ["gather", "next"]

        # Cleanup
        await runner.cleanup()
