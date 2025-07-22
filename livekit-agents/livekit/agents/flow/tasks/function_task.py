import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING, Any, Optional

import aiohttp

from livekit.agents import AgentTask

from ..schema import Node
from ..types.types import TransitionResult

if TYPE_CHECKING:
    from ..core.runner import FlowRunner

logger = logging.getLogger(__name__)


class FunctionTask(AgentTask[Optional[TransitionResult]]):
    """Handles execution of custom functions in flow nodes.

    This task executes HTTP-based custom functions and handles
    parameter extraction from the flow context.
    """

    def __init__(self, node: Node, flow_runner: "FlowRunner") -> None:
        super().__init__(instructions="")
        self.node = node
        self.flow_runner = flow_runner
        self._completed = False

    def complete(self, result: Optional[TransitionResult]) -> None:
        """Override to track completion state"""
        if not self._completed:
            self._completed = True
            super().complete(result)

    async def on_enter(self) -> None:
        """Called when entering the function node."""
        try:
            self.flow_runner.context.execution_path.append(self.node.id)
            self.flow_runner.context.save_checkpoint(self.node.id)

            if self.node.tool_id:
                # Execute custom HTTP function (simplified - no fallback to Python functions)
                result = await self._execute_custom_function()
                if result is not None:
                    self.flow_runner.context.function_results[self.node.tool_id] = result
                    logger.info(f"Custom function {self.node.tool_id} executed successfully")
                else:
                    logger.warning(
                        f"Custom function execution failed for tool_id: {self.node.tool_id}"
                    )

            # Evaluate where to go next after function execution
            evaluation = await self.flow_runner._evaluate_transition_conditions(self.node)
            transition_result = TransitionResult.from_evaluation(evaluation)
            self.complete(transition_result)

        except Exception as e:
            logger.error(f"Error in FunctionTask.on_enter for node {self.node.id}: {e}")
            self.flow_runner._handle_task_error(e, self.node.id)
            # Complete task with None result on error
            self.complete(None)
            return

    async def _execute_custom_function(self) -> Optional[dict[str, Any]]:
        """Execute HTTP-based custom function with all logic inline"""
        if not self.node.tool_id:
            return None

        tool_spec = self.flow_runner.flow.tools.get(self.node.tool_id)
        if not tool_spec or tool_spec.type != "custom" or not tool_spec.url:
            logger.error(f"Invalid custom function configuration for {self.node.tool_id}")
            return None

        try:
            start_time = time.time()

            parameters = await self._get_parameters_from_context(tool_spec)

            if self._has_missing_required_parameters(tool_spec, parameters):
                return {"error": "Missing required parameters"}

            result = await self._execute_http_request(tool_spec, parameters)

            execution_time = (time.time() - start_time) * 1000
            result["execution_time_ms"] = execution_time

            # Optionally speak during execution
            if self.node.speak_during_execution:
                if result.get("success", False):
                    await self.session.say(f"I've executed {tool_spec.name} and got the results.")
                else:
                    await self.session.say(
                        "I encountered an issue processing your request. Let me try to help you differently."
                    )

            return result

        except Exception as e:
            logger.error(f"Custom function execution failed for {self.node.tool_id}: {e}")
            return {"error": f"Function execution failed: {str(e)}", "tool_name": tool_spec.name}

    async def _get_parameters_from_context(self, tool_spec) -> dict[str, Any]:
        """Extract parameters directly from flow context"""
        parameter_schema = tool_spec.parameters

        if not parameter_schema:
            return {}

        # Handle both JSON Schema and flat parameter formats
        if isinstance(parameter_schema, dict) and "properties" in parameter_schema:
            properties = parameter_schema.get("properties", {})
        else:
            properties = {name: {"type": "string"} for name in parameter_schema.keys()}

        extracted_params = {}
        context_vars = self.flow_runner.context.variables

        for param_name in properties.keys():
            if param_name in context_vars:
                extracted_params[param_name] = context_vars[param_name]
            elif f"input_{param_name}" in context_vars:
                extracted_params[param_name] = context_vars[f"input_{param_name}"]
            elif f"{tool_spec.tool_id}_{param_name}" in context_vars:
                extracted_params[param_name] = context_vars[f"{tool_spec.tool_id}_{param_name}"]

        return extracted_params

    def _has_missing_required_parameters(self, tool_spec, parameters: dict[str, Any]) -> bool:
        """Check if required parameters are missing"""
        parameter_schema = tool_spec.parameters

        if not parameter_schema:
            return False

        required_params = parameter_schema.get("required", [])
        missing = [param for param in required_params if param not in parameters]

        if missing:
            logger.error(f"Missing required parameters for {tool_spec.name}: {missing}")
            return True
        return False

    async def _execute_http_request(self, tool_spec, parameters: dict[str, Any]) -> dict[str, Any]:
        """Execute HTTP request to the custom function endpoint"""
        session = await self.flow_runner.get_http_session()

        try:
            method = tool_spec.http_method.upper()
            url = tool_spec.url
            headers = tool_spec.headers.copy()
            timeout = aiohttp.ClientTimeout(total=tool_spec.timeout_ms / 1000.0)
            params = tool_spec.query_parameters.copy()

            data = None
            if method in ["POST", "PUT", "PATCH"]:
                headers["Content-Type"] = headers.get("Content-Type", "application/json")
                data = json.dumps(parameters)
            elif method == "GET":
                params.update(parameters)

            logger.info(f"Executing {method} request to {url} with parameters: {parameters}")

            async with session.request(
                method=method, url=url, headers=headers, params=params, data=data, timeout=timeout
            ) as response:
                status_code = response.status
                response_text = await response.text()

                try:
                    result_data = await response.json()
                except (json.JSONDecodeError, aiohttp.ContentTypeError):
                    result_data = {"response": response_text}

                if 200 <= status_code < 300:
                    logger.info(
                        f"Function {tool_spec.name} executed successfully (status: {status_code})"
                    )
                    return {
                        "success": True,
                        "result": result_data,
                        "status_code": status_code,
                        "tool_name": tool_spec.name,
                    }
                else:
                    error_msg = f"HTTP {status_code}: {response_text[:500]}"
                    logger.warning(f"Function {tool_spec.name} returned error: {error_msg}")
                    return {
                        "success": False,
                        "error": error_msg,
                        "status_code": status_code,
                        "result": result_data,
                        "tool_name": tool_spec.name,
                    }

        except asyncio.TimeoutError:
            error_msg = f"Request timeout after {tool_spec.timeout_ms}ms"
            logger.error(f"Function {tool_spec.name} timed out: {error_msg}")
            return {"success": False, "error": error_msg, "tool_name": tool_spec.name}
        except Exception as e:
            error_msg = f"HTTP request failed: {str(e)}"
            logger.error(f"Function {tool_spec.name} failed: {error_msg}")
            return {"success": False, "error": error_msg, "tool_name": tool_spec.name}
