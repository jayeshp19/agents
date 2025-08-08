import json

import aiohttp
import pytest

from livekit.agents.flow.agents.tool_http import (
    parse_tool_response,
    prepare_request,
    send_with_retries,
)


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


class _ReqCtx:
    def __init__(self, resp: _Resp):
        self._resp = resp

    def __await__(self):
        async def _coro():
            return self._resp

        return _coro().__await__()

    async def __aenter__(self):
        return await self._resp.__aenter__()

    async def __aexit__(self, exc_type, exc, tb):
        return await self._resp.__aexit__(exc_type, exc, tb)


class _SessionFailOnce:
    def __init__(self):
        self.calls = 0

    def request(self, method: str, url: str, **kwargs):  # noqa: D401
        self.calls += 1
        if self.calls == 1:
            return _ReqCtx(_Resp(500, {"error": "fail"}))
        return _ReqCtx(_Resp(200, {"ok": True}))


class _SessionAlwaysFail:
    def request(self, method: str, url: str, **kwargs):  # noqa: D401
        return _ReqCtx(_Resp(500, {"error": "fail"}))


def _extract(data, mapping):
    # simple extractor used for testing parse_tool_response
    return {k: data.get(k) for k in mapping}


def test_prepare_request_builds_url_and_params():
    tool = {
        "url": "https://example.com/api",
        "http_method": "GET",
        "query_parameters": {"x": "1"},
    }
    args = {"a": "b"}
    url, method, kwargs, ptype = prepare_request(tool, args)
    assert url == "https://example.com/api?x=1"
    assert method == "GET"
    assert kwargs["params"] == args
    assert ptype == "json"


def test_parse_tool_response_extracts_variables():
    tool = {"response_variables": {"foo": "$.foo"}}
    data = {"foo": 42}
    out = parse_tool_response(data, tool, _extract)
    assert out == {"foo": 42}


@pytest.mark.asyncio
async def test_send_with_retries_succeeds_after_failure():
    tool = {"cb_max_failures": 0}
    session = _SessionFailOnce()
    data = await send_with_retries(
        session,
        "GET",
        "https://example.com",
        {},
        parameter_type="json",
        max_retries=1,
        retry_backoff_ms=1,
        tool=tool,
    )
    assert data == {"ok": True}


@pytest.mark.asyncio
async def test_send_with_retries_circuit_breaker_opens():
    tool = {"cb_max_failures": 1, "cb_reset_ms": 30000}
    session = _SessionAlwaysFail()
    with pytest.raises(aiohttp.ClientResponseError):
        await send_with_retries(
            session,
            "GET",
            "https://example.com",
            {},
            parameter_type="json",
            max_retries=0,
            retry_backoff_ms=1,
            tool=tool,
        )
    with pytest.raises(RuntimeError):
        await send_with_retries(
            session,
            "GET",
            "https://example.com",
            {},
            parameter_type="json",
            max_retries=0,
            retry_backoff_ms=1,
            tool=tool,
        )
