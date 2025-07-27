import asyncio
import inspect
import logging
from typing import Any, Optional
from urllib.parse import parse_qsl, urlencode, urlparse

import aiohttp
import jsonschema  # type: ignore[import-untyped]
from jsonpath_ng import parse as jsonpath_parse  # type: ignore[import-untyped]

from ..base import BaseFlowAgent
from ..transition_evaluator import FlowContextVariableProvider, TransitionEvaluator

logger = logging.getLogger(__name__)


class ToolExecutor:
    """Handles HTTP and local execution for function tools."""

    def __init__(self, tool_def: dict[str, Any], http_session: aiohttp.ClientSession):
        self.tool = tool_def
        self.session = http_session

    async def __call__(self, args: dict[str, Any]) -> dict[str, Any]:
        # Validate arguments against schema if present
        if "parameters" in self.tool:
            self._validate_args(self.tool["parameters"], args)

        if self.tool.get("type") == "custom" and "url" in self.tool:
            return await self._execute_http(args)
        return await self._execute_local(args)

    def _validate_args(self, schema: dict[str, Any], args: dict[str, Any]) -> None:
        """Validate arguments against JSON Schema."""
        if schema:  # empty dict means no constraints
            try:
                jsonschema.validate(instance=args, schema=schema)
            except jsonschema.ValidationError as e:
                logger.error(f"Argument validation failed: {e}")
                raise ValueError(f"Invalid arguments: {e.message}") from e

    async def _execute_http(self, args: dict[str, Any]) -> dict[str, Any]:
        """Execute HTTP function call with flexible parameter handling."""
        url = self.tool["url"]
        method = self.tool.get("http_method", "POST").upper()
        headers = self.tool.get("headers", {})
        timeout_ms = self.tool.get("timeout_ms", 30000)
        parameter_type = self.tool.get("parameter_type", "json")

        # Merge query params if specified
        if "query_params" in self.tool:
            url = self._merge_query(url, self.tool["query_params"])

        timeout = aiohttp.ClientTimeout(total=timeout_ms / 1000)
        req_kwargs: dict[str, Any] = {"headers": headers, "timeout": timeout}

        # Handle different parameter types
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

        try:
            logger.debug(f"Making {method} request to {url} with {parameter_type} params")

            async with self.session.request(method, url, **req_kwargs) as response:
                response.raise_for_status()
                data = await response.json()

            # Extract variables using JSONPath if specified
            if "response_variables" in self.tool:
                extracted = self._extract_json_paths(data, self.tool["response_variables"])
                return extracted if extracted else data

            return data

        except aiohttp.ClientError as e:
            logger.error(f"HTTP error calling {url}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error calling {url}: {e}")
            raise

    async def _execute_local(self, args: dict[str, Any]) -> Any:
        """Execute local function handler."""
        handler = self.tool.get("handler")

        if not handler or not callable(handler):
            raise ValueError(f"No valid handler for function {self.tool.get('tool_id', 'unknown')}")

        # Support both sync and async handlers
        if inspect.iscoroutinefunction(handler):
            return await handler(args)
        return handler(args)

    def _merge_query(self, url: str, extra: dict[str, str]) -> str:
        """Merge extra query parameters into URL."""
        if not extra:
            return url
        parsed = urlparse(url)
        merged = dict(parse_qsl(parsed.query))
        merged.update(extra)
        query = urlencode(merged)
        return parsed._replace(query=query).geturl()

    def _extract_json_paths(self, data: Any, mapping: dict[str, str]) -> dict[str, Any]:
        """Extract values from response using JSONPath expressions."""
        result: dict[str, Any] = {}
        for var_name, path in mapping.items():
            try:
                matches = jsonpath_parse(path).find(data)
                result[var_name] = matches[0].value if matches else None
            except Exception as e:
                logger.warning(f"JSONPath extraction failed for {var_name} with path {path}: {e}")
                result[var_name] = None
        return result


class FunctionNodeAgent(BaseFlowAgent):
    """Enhanced function node agent with improved error handling and flexibility."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._function_result: Optional[Any] = None
        self._execution_error: Optional[Exception] = None
        self._executor: Optional[ToolExecutor] = None

        # Create transition evaluator for function result evaluation
        provider = FlowContextVariableProvider(self.flow_context)
        self._transition_evaluator = TransitionEvaluator(provider)

    async def _on_enter_node(self) -> None:
        # Validate node configuration
        if not hasattr(self.node, "tool_id"):
            await self._handle_function_error(
                ValueError("Function node missing tool_id configuration")
            )
            return

        # Speak during execution if configured
        if getattr(self.node, "speak_during_execution", False):
            message = (
                self.node.instruction.text
                if self.node.instruction
                else "Let me process that for you..."
            )
            self.session.say(message)

        try:
            logger.info(f"Executing function {self.node.tool_id} for node {self.node.id}")

            # Get function definition
            function_def = self.flow_runner.get_function_definition(self.node.tool_id)
            if not function_def:
                raise ValueError(f"Function {self.node.tool_id} not registered")

            # Prepare arguments
            args = self._prepare_function_arguments(function_def)

            # Check if we should wait for result
            wait_for_result = getattr(self.node, "wait_for_result", True)

            if not wait_for_result:
                # Fire-and-forget mode - execute async and continue
                asyncio.create_task(self._execute_and_store_async(function_def, args))
                self._function_result = None
                await self._evaluate_and_transition()
            else:
                # Normal execution - wait for result
                self._function_result = await self._execute_function(function_def, args)
                await self._store_result_and_transition()

        except Exception as e:
            logger.error(f"Error executing function in node {self.node.id}: {e}")
            self._execution_error = e
            await self._handle_function_error(e)

    def _prepare_function_arguments(self, function_def: dict[str, Any]) -> dict[str, Any]:
        """Prepare function arguments with improved variable resolution."""
        args = {}

        # Gather all available data sources
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

        logger.info(f"Function {self.node.tool_id} prepared args: {list(args.keys())}")
        return args

    def _collect_gathered_data(self) -> dict[str, Any]:
        """Collect all gathered data from flow context."""
        gathered_data = {}
        for key, value in self.flow_context.variables.items():
            if key.startswith("gathered_data_") and isinstance(value, dict):
                gathered_data.update(value)
        return gathered_data

    def _resolve_parameter_value(
        self, param_name: str, gathered_data: dict[str, Any]
    ) -> Optional[Any]:
        """Resolve parameter value from various sources."""
        # Direct match in gathered data
        if param_name in gathered_data:
            return gathered_data[param_name]

        if param_name in self.flow_context.variables:
            return self.flow_context.variables[param_name]

        # Case-insensitive search in gathered data
        for key, val in gathered_data.items():
            if key.lower() == param_name.lower():
                return val

        # Case-insensitive search in variables
        for key, val in self.flow_context.variables.items():
            if key.lower() == param_name.lower():
                return val

        return None

    async def _execute_function(self, function_def: dict[str, Any], args: dict[str, Any]) -> Any:
        """Execute function using appropriate executor."""
        # Get or create executor
        if not self._executor:
            session = await self.flow_runner.get_http_session()
            self._executor = ToolExecutor(function_def, session)

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
            logger.info(f"Async function {self.node.tool_id} completed")
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
        self.flow_context.function_results[self.node.tool_id] = {
            "node_id": self.node.id,
            "result": self._function_result,
            "timestamp": asyncio.get_event_loop().time(),
        }

        self.flow_context.set_variable(f"tool_result_{self.node.tool_id}", self._function_result)

        logger.info(f"Function {self.node.tool_id} completed successfully")
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
            logger.info(f"Transitioning from {self.node.id} to {transition.destination_node_id}")
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
                logger.info(f"Taking error edge: {transition.edge_id}")
                await self._transition_to_node(transition.destination_node_id)
                return

        logger.warning(f"No error handling edge for function node {self.node.id}")

    async def _on_exit_node(self) -> None:
        summary = {
            "node_id": self.node.id,
            "tool_id": getattr(self.node, "tool_id", "unknown"),
            "had_error": self._execution_error is not None,
            "result_type": type(self._function_result).__name__ if self._function_result else None,
            "was_async": getattr(self.node, "wait_for_result", True) is False,
        }

        self.flow_context.set_variable(f"function_summary_{self.node.id}", summary)
        logger.debug(f"Exiting function node {self.node.id} with summary: {summary}")
