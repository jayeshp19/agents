import asyncio
import inspect
import logging
from typing import Any, Optional, cast

import aiohttp
import jsonschema
from jsonpath_ng import parse as jsonpath_parse

from ..base import BaseFlowAgent
from ..transition_evaluator import FlowContextVariableProvider, TransitionEvaluator
from .tool_http import parse_tool_response, prepare_request, send_with_retries

logger = logging.getLogger(__name__)


class _ToolConcurrency:
    _locks: dict[str, asyncio.Semaphore] = {}

    @classmethod
    def get(cls, tool_id: str, max_concurrency: int) -> asyncio.Semaphore:
        if not tool_id:
            # anonymous limiter key
            tool_id = "__anon__"
        lock = cls._locks.get(tool_id)
        if lock is None:
            lock = asyncio.Semaphore(max_concurrency)
            cls._locks[tool_id] = lock
        return lock


class ToolExecutor:
    """Handles HTTP and local execution for function tools."""

    def __init__(self, tool_def: dict[str, Any], http_session: aiohttp.ClientSession | None):
        self.tool = tool_def
        self.session = http_session
        self._max_retries: int = int(tool_def.get("max_retries", 1))
        self._retry_backoff_ms: int = int(tool_def.get("retry_backoff_ms", 250))
        self._compiled_jsonpaths: dict[str, Any] = {}
        try:
            rv = self.tool.get("response_variables")
            if isinstance(rv, dict):
                for var_name, path in rv.items():
                    try:
                        self._compiled_jsonpaths[var_name] = jsonpath_parse(path)
                    except Exception as e:
                        logger.error(f"Invalid JSONPath for '{var_name}': '{path}' - {e}")
        except Exception:
            # Don't fail tool init on compilation issues; they will be logged during extraction
            pass

        self._semaphore = None
        try:
            max_conc = int(self.tool.get("max_concurrency", 0))
            if max_conc > 0:
                self._semaphore = _ToolConcurrency.get(self.tool.get("tool_id", ""), max_conc)
        except Exception:
            pass

    async def __call__(self, args: dict[str, Any]) -> Any:
        # Validate arguments against schema if present
        if "parameters" in self.tool:
            self._validate_args(self.tool["parameters"], args)

        async def _run() -> Any:
            if self.tool.get("type") == "custom" and "url" in self.tool:
                return await self._execute_http(args)
            return await self._execute_local(args)

        if self._semaphore is None:
            return await _run()
        async with self._semaphore:
            return await _run()

    def _validate_args(self, schema: dict[str, Any], args: dict[str, Any]) -> None:
        if schema:
            try:
                jsonschema.validate(instance=args, schema=schema)
            except jsonschema.ValidationError as e:
                logger.error(f"Argument validation failed: {e}")
                raise ValueError(f"Invalid arguments: {e.message}") from e

    async def _execute_http(self, args: dict[str, Any]) -> dict[str, Any]:
        if self.session is None:
            raise RuntimeError("HTTP session not initialized for ToolExecutor")
        url, method, req_kwargs, parameter_type = prepare_request(self.tool, args)

        data = await send_with_retries(
            self.session,
            method,
            url,
            req_kwargs,
            parameter_type=parameter_type,
            max_retries=self._max_retries,
            retry_backoff_ms=self._retry_backoff_ms,
            tool=self.tool,
        )

        return cast(
            dict[str, Any],
            parse_tool_response(data, self.tool, self._extract_json_paths),
        )

    async def _execute_local(self, args: dict[str, Any]) -> Any:
        handler = self.tool.get("handler")

        if not handler or not callable(handler):
            raise ValueError(f"No valid handler for function {self.tool.get('tool_id', 'unknown')}")

        if inspect.iscoroutinefunction(handler):
            return await handler(args)
        return handler(args)

    def _extract_json_paths(self, data: Any, mapping: dict[str, str]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for var_name, path in mapping.items():
            try:
                parsed = self._compiled_jsonpaths.get(var_name)
                if parsed is None:
                    parsed = jsonpath_parse(path)
                    self._compiled_jsonpaths[var_name] = parsed
                matches = parsed.find(data)
                result[var_name] = matches[0].value if matches else None
                logger.debug(f"JSONPath extraction succeeded for {var_name}: {result[var_name]}")
            except Exception as e:
                if "parse" in str(e).lower() or "syntax" in str(e).lower():
                    logger.error(f"Invalid JSONPath syntax for {var_name}: '{path}' - {e}")
                    continue
                else:
                    logger.warning(
                        f"JSONPath extraction failed for {var_name} with path '{path}': {e}"
                    )
                    result[var_name] = None
        return result


class FunctionNodeAgent(BaseFlowAgent):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._function_result: Optional[Any] = None
        self._execution_error: Optional[Exception] = None
        self._executor: Optional[ToolExecutor] = None
        self._async_tasks: list[asyncio.Task[Any]] = []

        provider = FlowContextVariableProvider(self.flow_context)
        self._transition_evaluator = TransitionEvaluator(provider)

    async def _on_enter_node(self) -> None:
        if not hasattr(self.node, "tool_id"):
            await self._handle_function_error(
                ValueError("Function node missing tool_id configuration")
            )
            return

        if getattr(self.node, "speak_during_execution", False):
            message = (
                self.node.instruction.text
                if self.node.instruction
                else "Let me process that for you..."
            )
            self.session.say(message)

        try:
            if not getattr(self.node, "tool_id", None):
                raise ValueError("Function node missing tool_id")
            tool_id = cast(str, self.node.tool_id)
            function_def = self.flow_runner.get_function_definition(tool_id)
            if not function_def:
                raise ValueError(f"Function {tool_id} not registered")
            handler = self.flow_runner.get_function_handler(tool_id)
            if handler and "handler" not in function_def:
                function_def = {**function_def, "handler": handler}

            args = self._prepare_function_arguments(function_def)

            wait_for_result = getattr(self.node, "wait_for_result", True)

            if not wait_for_result:
                task = asyncio.create_task(self._execute_and_store_async(function_def, args))
                self._async_tasks.append(task)
                self._async_tasks = [t for t in self._async_tasks if not t.done()]
                self._function_result = None
                await self._evaluate_and_transition()
            else:
                self._function_result = await self._execute_function(function_def, args)
                await self._store_result_and_transition()

        except Exception as e:
            logger.error(f"Error executing function in node {self.node.id}: {e}")
            self._execution_error = e
            await self._handle_function_error(e)

    def _prepare_function_arguments(self, function_def: dict[str, Any]) -> dict[str, Any]:
        args = {}

        gathered_data = self._collect_gathered_data()

        params = function_def.get("parameters", {})
        properties = params.get("properties", {})
        required = params.get("required", [])

        for param_name, param_def in properties.items():
            # Try to find value from multiple sources
            value = self._resolve_parameter_value(param_name, gathered_data)

            if value is not None:
                args[param_name] = value
            elif "default" in param_def:
                args[param_name] = param_def["default"]
            elif param_name in required:
                logger.warning(
                    f"Required parameter {param_name} not found for function {self.node.tool_id}"
                )

        return args

    def _collect_gathered_data(self) -> dict[str, Any]:
        gathered_data = {}
        for key, value in self.flow_context.variables.items():
            if key.startswith("gathered_data_") and isinstance(value, dict):
                gathered_data.update(value)
        return gathered_data

    def _resolve_parameter_value(
        self, param_name: str, gathered_data: dict[str, Any]
    ) -> Optional[Any]:
        if param_name in gathered_data:
            return gathered_data[param_name]

        if param_name in self.flow_context.variables:
            return self.flow_context.variables[param_name]

        for key, val in gathered_data.items():
            if key.lower() == param_name.lower():
                return val

        for key, val in self.flow_context.variables.items():
            if key.lower() == param_name.lower():
                return val

        return None

    async def _execute_function(self, function_def: dict[str, Any], args: dict[str, Any]) -> Any:
        if not self._executor:
            session: aiohttp.ClientSession | None = None
            try:
                if function_def.get("type") == "custom" and function_def.get("url"):
                    session = await self.flow_runner.get_http_session()
            except Exception:
                session = None
            self._executor = ToolExecutor(function_def, session)
        deadline_ms = function_def.get("deadline_ms")
        if isinstance(deadline_ms, (int, float)) and deadline_ms > 0:
            return await asyncio.wait_for(self._executor(args), timeout=deadline_ms / 1000.0)
        return await self._executor(args)

    async def _execute_and_store_async(
        self, function_def: dict[str, Any], args: dict[str, Any]
    ) -> None:
        try:
            result = await self._execute_function(function_def, args)
            self.flow_context.set_variable(
                f"async_result_{self.node.tool_id}",
                {
                    "node_id": self.node.id,
                    "result": result,
                    "timestamp": asyncio.get_event_loop().time(),
                },
            )
        except Exception as e:
            logger.error(f"Async function {self.node.tool_id} failed: {e}")
            self.flow_context.set_variable(
                f"async_error_{self.node.tool_id}",
                {
                    "node_id": self.node.id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )

    async def _store_result_and_transition(self) -> None:
        self.flow_context.function_results[cast(str, self.node.tool_id)] = {
            "node_id": self.node.id,
            "result": self._function_result,
            "timestamp": asyncio.get_event_loop().time(),
        }

        self.flow_context.set_variable(
            f"tool_result_{cast(str, self.node.tool_id)}", self._function_result
        )

        await self._evaluate_and_transition()

    async def _evaluate_and_transition(self) -> None:
        if not self.node.edges:
            transition = await self._evaluate_transition()
            if transition and transition.destination_node_id:
                await self._transition_to_node(transition.destination_node_id)
            else:
                logger.warning(f"Function node {self.node.id} completed but no transition found")
                await self.session.say(
                    "I've completed the operation, but I'm not sure what to do next."
                )
            return

        context = {
            "result": self._function_result,
            "tool_id": self.node.tool_id,
            "node_id": self.node.id,
        }

        if isinstance(self._function_result, dict):
            context.update(self._function_result)

        transition = await self._transition_evaluator.evaluate_transitions(
            edges=self.node.edges,
            user_text=None,
            context=context,
        )

        if transition and transition.destination_node_id:
            await self._transition_to_node(transition.destination_node_id)
        else:
            logger.warning(f"Function node {self.node.id} completed but no transition matched")
            await self.session.say(
                "I've completed the operation, but I'm not sure what to do next."
            )

    async def _handle_function_error(self, error: Exception) -> None:
        self.flow_context.set_variable(
            f"function_error_{self.node.id}",
            {
                "tool_id": getattr(self.node, "tool_id", "unknown"),
                "error_type": type(error).__name__,
                "error_message": str(error),
                "timestamp": asyncio.get_event_loop().time(),
            },
        )

        error_message = "encountered an issue."
        await self.session.say(error_message)

        if self.node.edges:
            error_context = {
                "error": str(error),
                "error_type": type(error).__name__,
                "status": "error",
                "success": False,
                "tool_id": getattr(self.node, "tool_id", "unknown"),
            }

            transition = await self._transition_evaluator.evaluate_transitions(
                edges=self.node.edges, user_text=None, context=error_context
            )

            if transition and transition.destination_node_id:
                await self._transition_to_node(transition.destination_node_id)

    async def _on_exit_node(self) -> None:
        await self._cancel_tasks(self._async_tasks, timeout=2.0, phase="function node exit")

        summary = {
            "node_id": self.node.id,
            "tool_id": getattr(self.node, "tool_id", "unknown"),
            "had_error": self._execution_error is not None,
            "result_type": type(self._function_result).__name__ if self._function_result else None,
            "was_async": getattr(self.node, "wait_for_result", True) is False,
            "async_tasks_count": len(self._async_tasks),
        }

        self.flow_context.set_variable(f"function_summary_{self.node.id}", summary)
