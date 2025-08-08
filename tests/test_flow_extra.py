import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from livekit.agents.flow import FlowRunner
from livekit.agents.llm import (
    LLM,
    ChatChunk,
    ChatContext,
    ChoiceDelta,
    LLMStream,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.voice import AgentSession
from tests.fake_io import FakeAudioInput
from tests.fake_llm import FakeLLM
from tests.fake_stt import FakeSTT, FakeUserSpeech
from tests.fake_tts import FakeTTS
from tests.fake_vad import FakeVAD


def _make_session_with_speeches(user_speeches: list[FakeUserSpeech]) -> AgentSession:
    stt = FakeSTT(fake_user_speeches=user_speeches)
    session = AgentSession(
        vad=FakeVAD(
            fake_user_speeches=user_speeches,
            min_silence_duration=0.05,
            min_speech_duration=0.02,
        ),
        stt=stt,
        llm=FakeLLM(),
        tts=FakeTTS(),
        min_interruption_duration=0.05,
        min_endpointing_delay=0.05,
        max_endpointing_delay=0.5,
    )
    session.input.audio = FakeAudioInput()
    return session


class _OneShotLLMStream(LLMStream):
    def __init__(self, llm: "_OneShotLLM", *, chat_ctx: ChatContext, content: str):
        super().__init__(llm, chat_ctx=chat_ctx, tools=[], conn_options=DEFAULT_API_CONNECT_OPTIONS)
        self._content = content

    async def _run(self) -> None:
        self._event_ch.send_nowait(
            ChatChunk(
                id="oneshot",
                delta=ChoiceDelta(role="assistant", content=self._content, tool_calls=[]),
            )
        )


class _OneShotLLM(LLM):
    def __init__(self, content: str):
        super().__init__()
        self._content = content

    def chat(self, *, chat_ctx: ChatContext, **_: object) -> LLMStream:  # type: ignore[override]
        return _OneShotLLMStream(self, chat_ctx=chat_ctx, content=self._content)


async def _wait_for_node(runner: FlowRunner, node_id: str, *, timeout: float = 2.0) -> None:
    """Poll until the runner reaches the desired node or timeout.

    Uses short sleeps to avoid timing flakiness from scheduler variance.
    """
    start = asyncio.get_event_loop().time()
    while True:
        if runner.context.current_node_id == node_id:
            return
        if asyncio.get_event_loop().time() - start > timeout:
            pytest.fail(
                f"Timed out waiting for node '{node_id}', "
                f"current='{runner.context.current_node_id}'"
            )
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_conversation_prompt_transition(tmp_path: Path):
    flow = {
        "conversation_flow_id": "prompt_conv",
        "version": 1,
        "global_prompt": "Test",
        "nodes": [
            {
                "id": "start",
                "name": "Start",
                "type": "conversation",
                "instruction": {"type": "static_text", "text": "Talk"},
                "edges": [
                    {
                        "id": "to_next",
                        "condition": "go next",
                        "transition_condition": {"type": "prompt", "prompt": "user wants next"},
                        "destination_node_id": "next",
                    }
                ],
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
        "start_speaker": "user",
        "tools": [],
    }
    flow_path = tmp_path / "flow.json"
    flow_path.write_text(json.dumps(flow))

    # LLM returns high-confidence selection of option 1
    edge_llm = _OneShotLLM(content=json.dumps({"option": 1, "confidence": 0.99}))
    runner = FlowRunner(flow_path=str(flow_path), edge_evaluator_llm=edge_llm)

    speech = FakeUserSpeech(
        start_time=0.1, end_time=0.2, transcript="please go to next", stt_delay=0.05
    )

    async with _make_session_with_speeches([speech]) as sess:
        await runner.start(sess)
        # kick off audio
        assert isinstance(sess.input.audio, FakeAudioInput)
        sess.input.audio.push(0.1)
        await sess.stt.fake_user_speeches_done  # type: ignore[attr-defined]
        await asyncio.sleep(0.05)
        assert runner.context.current_node_id == "next"
        await runner.cleanup()


@pytest.mark.asyncio
async def test_global_node_trigger(tmp_path: Path):
    flow = {
        "conversation_flow_id": "global_trigger",
        "version": 1,
        "global_prompt": "Test",
        "nodes": [
            {
                "id": "start",
                "name": "Start",
                "type": "conversation",
                "instruction": {"type": "static_text", "text": "Talk"},
                "edges": [],
            },
            {
                "id": "helpdesk",
                "name": "Help",
                "type": "conversation",
                "instruction": {"type": "static_text", "text": "Help"},
                "global_node_setting": {"condition": "user asks for help"},
                "edges": [],
            },
        ],
        "start_node_id": "start",
        "start_speaker": "user",
        "tools": [],
    }
    flow_path = tmp_path / "flow.json"
    flow_path.write_text(json.dumps(flow))

    edge_llm = _OneShotLLM(content=json.dumps({"option": 1, "confidence": 0.99}))
    runner = FlowRunner(flow_path=str(flow_path), edge_evaluator_llm=edge_llm)

    speech = FakeUserSpeech(start_time=0.1, end_time=0.2, transcript="help me", stt_delay=0.05)

    async with _make_session_with_speeches([speech]) as sess:
        await runner.start(sess)
        assert isinstance(sess.input.audio, FakeAudioInput)
        sess.input.audio.push(0.1)
        await sess.stt.fake_user_speeches_done  # type: ignore[attr-defined]
        await asyncio.sleep(0.05)
        assert runner.context.current_node_id == "helpdesk"
        await runner.cleanup()


@pytest.mark.asyncio
async def test_function_async_wait_for_result_false_transition(tmp_path: Path):
    flow = {
        "conversation_flow_id": "fn_async",
        "version": 1,
        "global_prompt": "Test",
        "nodes": [
            {
                "id": "fn",
                "name": "Function",
                "type": "function",
                "tool_id": "mytool",
                "wait_for_result": False,
                "instruction": {"type": "static_text", "text": "Working"},
                "edges": [
                    {
                        "id": "by_tool",
                        "condition": "tool match",
                        "transition_condition": {
                            "type": "equation",
                            "equations": [
                                {
                                    "left_operand": "tool_id",
                                    "operator": "==",
                                    "right_operand": "expected_tool_id",
                                }
                            ],
                        },
                        "destination_node_id": "next",
                    }
                ],
            },
            {
                "id": "next",
                "name": "Next",
                "type": "conversation",
                "instruction": {"type": "static_text", "text": "Done"},
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
                "parameters": {"type": "object", "properties": {}},
            }
        ],
    }

    flow_path = tmp_path / "flow.json"
    flow_path.write_text(json.dumps(flow))
    runner = FlowRunner(
        flow_path=str(flow_path),
        edge_evaluator_llm=FakeLLM(),
        initial_context={"expected_tool_id": "mytool"},
    )

    # Register a slow async handler to ensure it's truly fire-and-forget
    async def _slow(_: dict) -> dict:
        await asyncio.sleep(0.2)
        return {"ok": True}

    runner.register_function("mytool", _slow)

    async with AgentSession(
        vad=FakeVAD(fake_user_speeches=[]), stt=FakeSTT(), llm=FakeLLM(), tts=FakeTTS()
    ) as sess:  # type: ignore[arg-type]
        await runner.start(sess)
        # Fire-and-forget: poll for transition to appear instead of fixed sleep
        await _wait_for_node(runner, "next", timeout=2.0)
        assert runner.context.current_node_id == "next"
    # allow FlowRunner.close hook to run and cleanup to complete
    await asyncio.sleep(0.1)
    await runner.cleanup()
    await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_start_speaker_user_suppresses_autosay(tmp_path: Path):
    flow = {
        "conversation_flow_id": "start_user",
        "version": 1,
        "global_prompt": "Test",
        "nodes": [
            {
                "id": "start",
                "name": "Start",
                "type": "conversation",
                "instruction": {"type": "static_text", "text": "Hello"},
                "edges": [],
            }
        ],
        "start_node_id": "start",
        "start_speaker": "user",
        "tools": [],
    }
    flow_path = tmp_path / "flow.json"
    flow_path.write_text(json.dumps(flow))

    runner = FlowRunner(flow_path=str(flow_path), edge_evaluator_llm=FakeLLM())
    events: list[Any] = []

    async with AgentSession(
        vad=FakeVAD(fake_user_speeches=[]), stt=FakeSTT(), llm=FakeLLM(), tts=FakeTTS()
    ) as sess:  # type: ignore[arg-type]
        sess.on("conversation_item_added", lambda e: events.append(e))
        await runner.start(sess)
        await asyncio.sleep(0.05)
        # No assistant message should be emitted automatically
        assert all(getattr(e.item, "role", None) != "assistant" for e in events)
        await runner.cleanup()


@pytest.mark.asyncio
async def test_toolexecutor_http_response_variables(tmp_path: Path):
    flow = {
        "conversation_flow_id": "http_tool",
        "version": 1,
        "global_prompt": "Test",
        "nodes": [
            {
                "id": "fn",
                "name": "Function",
                "type": "function",
                "tool_id": "verify",
                "wait_for_result": True,
                "instruction": {"type": "static_text", "text": "Verifying"},
                "edges": [
                    {
                        "id": "ok",
                        "condition": "ok",
                        "transition_condition": {
                            "type": "equation",
                            "equations": [
                                {
                                    "left_operand": "customer_email_returned",
                                    "operator": "==",
                                    "right_operand": "expected_customer_email",
                                }
                            ],
                        },
                        "destination_node_id": "next",
                    }
                ],
            },
            {
                "id": "next",
                "name": "Next",
                "type": "conversation",
                "instruction": {"type": "static_text", "text": "Done"},
                "edges": [],
            },
        ],
        "start_node_id": "fn",
        "tools": [
            {
                "tool_id": "verify",
                "name": "verify",
                "type": "custom",
                "description": "desc",
                "url": "https://example.com/api",
                "http_method": "POST",
                "parameters": {"type": "object", "properties": {}},
                "response_variables": {"customer_email_returned": "$.json.email"},
            }
        ],
    }

    class _Resp:
        def __init__(self, status: int, payload: dict[str, Any]):
            self.status = status
            self._payload = payload
            self.headers = {}
            self.history = []
            self.request_info = None

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def raise_for_status(self):
            return None

        async def json(self):
            return self._payload

        async def text(self):
            return json.dumps(self._payload)

    class _FakeSession:
        def request(self, method: str, url: str, **kwargs: Any):  # noqa: D401
            return _Resp(200, {"json": {"email": "user@example.com"}})

    flow_path = tmp_path / "flow.json"
    flow_path.write_text(json.dumps(flow))
    runner = FlowRunner(
        flow_path=str(flow_path),
        edge_evaluator_llm=FakeLLM(),
        initial_context={"expected_customer_email": "user@example.com"},
    )

    async def _fake_get_http_session():
        return _FakeSession()

    # Monkeypatch the http session creator
    runner.get_http_session = _fake_get_http_session  # type: ignore[assignment]

    async with AgentSession(
        vad=FakeVAD(fake_user_speeches=[]), stt=FakeSTT(), llm=FakeLLM(), tts=FakeTTS()
    ) as sess:  # type: ignore[arg-type]
        await runner.start(sess)
        await _wait_for_node(runner, "next", timeout=2.0)
        assert runner.context.current_node_id == "next"
        await runner.cleanup()


@pytest.mark.asyncio
async def test_concurrent_agent_cache_singleton(tmp_path: Path):
    flow = {
        "conversation_flow_id": "cache",
        "version": 1,
        "global_prompt": "Test",
        "nodes": [{"id": "start", "name": "Start", "type": "conversation", "edges": []}],
        "start_node_id": "start",
        "tools": [],
    }
    flow_path = tmp_path / "flow.json"
    flow_path.write_text(json.dumps(flow))
    runner = FlowRunner(flow_path=str(flow_path), edge_evaluator_llm=FakeLLM())

    # Create many concurrent fetches for the same node
    agents = await asyncio.gather(*[runner.get_or_create_agent("start") for _ in range(10)])
    # All should be the same object instance
    assert len({id(a) for a in agents}) == 1
