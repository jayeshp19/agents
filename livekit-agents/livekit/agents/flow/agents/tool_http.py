import asyncio
import logging
import random
from typing import Any, Callable
from urllib.parse import parse_qsl, urlencode, urlparse

import aiohttp

logger = logging.getLogger(__name__)


def _merge_query(url: str, extra: dict[str, str]) -> str:
    if not extra:
        return url
    parsed = urlparse(url)
    merged = dict(parse_qsl(parsed.query))
    merged.update(extra)
    query = urlencode(merged)
    return parsed._replace(query=query).geturl()


def prepare_request(
    tool: dict[str, Any], args: dict[str, Any]
) -> tuple[str, str, dict[str, Any], str]:
    """Construct the components required for an HTTP tool invocation.

    Args:
        tool: Mapping describing the tool including URL, method, headers and
            other request metadata.
        args: Arguments supplied by the caller that will be encoded into the
            request body or query string.

    Returns:
        A tuple ``(url, method, req_kwargs, parameter_type)`` where ``req_kwargs``
        are keyword arguments for :meth:`aiohttp.ClientSession.request` and
        ``parameter_type`` identifies how ``args`` were encoded (``json``,
        ``form`` or ``multipart``).
    """
    url = tool["url"]
    method = tool.get("http_method", "POST").upper()
    headers = tool.get("headers", {})
    timeout_ms = tool.get("timeout_ms", 30000)
    parameter_type = tool.get("parameter_type", "json")

    qp = tool.get("query_parameters") or tool.get("query_params")
    if qp:
        url = _merge_query(url, qp)

    timeout = aiohttp.ClientTimeout(total=timeout_ms / 1000)
    req_kwargs: dict[str, Any] = {"headers": headers, "timeout": timeout}

    if method == "GET":
        req_kwargs["params"] = args
    else:
        match parameter_type:
            case "json":
                req_kwargs["json"] = args
            case "form":
                req_kwargs["data"] = args
            case "multipart":
                form_data = aiohttp.FormData()
                for key, value in args.items():
                    form_data.add_field(key, value)
                req_kwargs["data"] = form_data
            case _:
                raise ValueError(f"Unsupported parameter_type '{parameter_type}'")

    return url, method, req_kwargs, parameter_type


async def send_with_retries(
    session: aiohttp.ClientSession,
    method: str,
    url: str,
    req_kwargs: dict[str, Any],
    *,
    parameter_type: str,
    max_retries: int,
    retry_backoff_ms: int,
    tool: dict[str, Any],
) -> Any:
    """Send an HTTP request with retries and circuit breaker tracking.

    The request is retried on network errors, timeouts, or HTTP 5xx/429
    responses using exponential backoff with jitter. Circuit breaker counters
    are stored on ``tool`` using the ``cb_*`` fields.

    Args:
        session: ``aiohttp`` session used to perform the request.
        method: HTTP method name.
        url: Destination URL.
        req_kwargs: Keyword arguments forwarded to
            :meth:`aiohttp.ClientSession.request`.
        parameter_type: Encoding type for ``args`` (``json``, ``form`` or
            ``multipart``) used only for logging.
        max_retries: Maximum number of retry attempts before failing.
        retry_backoff_ms: Initial backoff duration in milliseconds for retries.
        tool: Tool definition dict where circuit breaker state is maintained.

    Returns:
        Parsed JSON payload from the successful HTTP response.

    Raises:
        RuntimeError: If the circuit breaker is open.
        aiohttp.ClientError: If the request fails after all retries.
        asyncio.TimeoutError: If the request times out.
    """
    cb_max_failures = int(tool.get("cb_max_failures", 0))
    cb_reset_ms = int(tool.get("cb_reset_ms", 30000))

    if cb_max_failures > 0:
        now = asyncio.get_event_loop().time()
        last_reset = float(tool.get("_cb_last_reset", 0.0))
        recent_failures = int(tool.get("_cb_recent_failures", 0))
        if now - last_reset > (cb_reset_ms / 1000.0):
            tool["_cb_recent_failures"] = 0
            tool["_cb_last_reset"] = now
            recent_failures = 0
        if recent_failures >= cb_max_failures:
            raise RuntimeError("circuit_open: too many recent failures")

    attempt = 0
    while True:
        try:
            logger.debug(
                f"Making {method} request to {url} with {parameter_type} params (attempt {attempt + 1})"
            )
            async with session.request(method, url, **req_kwargs) as response:
                if response.status >= 500 or response.status == 429:
                    text = await response.text()
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status,
                        message=f"Retryable HTTP {response.status}: {text[:200]}",
                        headers=response.headers,
                    )
                response.raise_for_status()
                data: Any = await response.json()
            if cb_max_failures > 0:
                tool["_cb_recent_failures"] = 0
            return data
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            attempt += 1
            if cb_max_failures > 0:
                tool["_cb_recent_failures"] = int(tool.get("_cb_recent_failures", 0)) + 1
                tool["_cb_last_reset"] = asyncio.get_event_loop().time()
            if attempt > max_retries:
                logger.error(f"HTTP error calling {url}, no retries left: {e}")
                raise
            base = (retry_backoff_ms * (2 ** (attempt - 1))) / 1000.0
            jitter = random.uniform(0.5, 1.5)
            backoff = base * jitter
            logger.warning(
                f"HTTP error calling {url}: {e} — retrying in {backoff:.2f}s ({attempt}/{max_retries})"
            )
            await asyncio.sleep(backoff)
        except Exception as e:
            logger.error(f"Unexpected error calling {url}: {e}")
            raise


def parse_tool_response(
    data: Any,
    tool: dict[str, Any],
    extract_json_paths: Callable[[Any, dict[str, str]], dict[str, Any]],
) -> dict[str, Any]:
    """Interpret the HTTP tool response and extract configured variables.

    Args:
        data: JSON-decoded payload returned from the tool request.
        tool: Tool definition that may define ``response_variables`` for
            extracting specific values.
        extract_json_paths: Callback that receives ``data`` and a mapping of
            variable names to JSON paths and returns the extracted values.

    Returns:
        Either the mapping of extracted variables or the original ``data`` if
        no variables were configured or extraction yielded nothing.
    """
    if "response_variables" in tool:
        extracted = extract_json_paths(data, tool["response_variables"])
        return extracted if extracted else data
    return data
