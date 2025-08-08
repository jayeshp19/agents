import asyncio
import json

import aiohttp
import pytest

from livekit.agents.flow.agents.function_agent import ToolExecutor


class _Resp:
    def __init__(self, status: int, payload: dict):
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


class _SessionFailOnce:
    def __init__(self):
        self.calls = 0

    def request(self, method: str, url: str, **kwargs):  # noqa: D401
        self.calls += 1
        if self.calls == 1:
            return _Resp(500, {"error": "fail"})
        return _Resp(200, {"ok": True})


class _SessionAlwaysFail:
    def request(self, method: str, url: str, **kwargs):  # noqa: D401
        return _Resp(500, {"error": "fail"})


class _SessionSlow:
    def __init__(self, delay_s: float):
        self.delay_s = delay_s

    def request(self, method: str, url: str, **kwargs):
        delay = self.delay_s

        class _Ctx(_Resp):
            async def __aenter__(self_nonlocal):  # type: ignore[override]
                await asyncio.sleep(delay)
                return self_nonlocal

        return _Ctx(200, {"ok": True})


@pytest.mark.asyncio
async def test_http_retry_succeeds_after_failure(monkeypatch):
    tool = {
        "tool_id": "t",
        "name": "t",
        "type": "custom",
        "description": "d",
        "url": "https://example.com",
        "http_method": "GET",
        "parameters": {},
        "max_retries": 2,
        "retry_backoff_ms": 1,
    }

    session = _SessionFailOnce()
    execu = ToolExecutor(tool, session)
    out = await execu({})
    assert out == {"ok": True}


@pytest.mark.asyncio
async def test_http_circuit_breaker_opens(monkeypatch):
    tool = {
        "tool_id": "t",
        "name": "t",
        "type": "custom",
        "description": "d",
        "url": "https://example.com",
        "http_method": "GET",
        "parameters": {},
        "max_retries": 0,
        "retry_backoff_ms": 1,
        "cb_max_failures": 1,
        "cb_reset_ms": 30000,
    }

    session = _SessionAlwaysFail()
    execu = ToolExecutor(tool, session)
    with pytest.raises(aiohttp.ClientResponseError):
        await execu({})

    # Second call should be blocked by circuit breaker
    with pytest.raises(RuntimeError) as ei:
        await execu({})
    assert "circuit_open" in str(ei.value)


@pytest.mark.asyncio
async def test_http_max_concurrency_limits_parallelism():
    tool = {
        "tool_id": "slow",
        "name": "slow",
        "type": "custom",
        "description": "d",
        "url": "https://example.com",
        "http_method": "GET",
        "parameters": {},
        "max_concurrency": 1,
    }
    session = _SessionSlow(0.1)
    execu = ToolExecutor(tool, session)

    async def call():
        return await execu({})

    start = asyncio.get_event_loop().time()
    await asyncio.gather(call(), call())
    elapsed = asyncio.get_event_loop().time() - start
    # With concurrency limit 1 and ~0.1s per call, elapsed should be >= 0.19s
    assert elapsed >= 0.18
