import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Optional

import aiohttp

from livekit.agents import AgentSession, llm

from ..schema import Edge, FlowSpec, Node, load_flow
from ..tasks import ConversationTask, FunctionTask, GatherInputTask
from ..types.types import EdgeEvaluation, FlowContext, TransitionDecision
from ..utils.utils import clean_json_response

logger = logging.getLogger(__name__)


class FlowRunner:
    def __init__(
        self,
        path: str,
        edge_evaluator_llm: llm.LLM,
        initial_context: Optional[dict[str, Any]] = None,
        max_iterations: int = 100,
    ) -> None:
        self.flow_path = Path(path)
        self._load_and_validate_flow()

        self.context = FlowContext()
        if initial_context:
            self.context.variables.update(initial_context)

        self.max_iterations = max_iterations

        self._registered_functions: dict[str, Callable] = {}

        self._metrics = {
            "executions": 0,
            "errors": 0,
            "transitions": 0,
            "function_calls": 0,
            "llm_calls": 0,
            "llm_tokens_consumed": 0,
            "average_confidence": 0.0,
            "execution_time": 0.0,
            "nodes_visited": 0,
        }

        self._transition_times: list[float] = []
        self._confidence_scores: list[float] = []

        self.edge_llm = edge_evaluator_llm

        self._http_session: Optional[aiohttp.ClientSession] = None

    async def get_http_session(self) -> aiohttp.ClientSession:
        if self._http_session is None or self._http_session.closed:
            try:
                timeout = aiohttp.ClientTimeout(total=300)  # 5 minute total timeout
                self._http_session = aiohttp.ClientSession(timeout=timeout)
            except Exception as e:
                logger.error(f"Failed to create HTTP session: {e}")
                # Ensure we don't leak a partially created session
                if self._http_session and not self._http_session.closed:
                    try:
                        await self._http_session.close()
                    except Exception:
                        pass
                    self._http_session = None
                raise
        return self._http_session

    async def cleanup(self) -> None:
        try:
            if self._http_session and not self._http_session.closed:
                await self._http_session.close()
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

    def __del__(self) -> None:
        try:
            if hasattr(self, "_http_session") and self._http_session:
                if not self._http_session.closed:
                    logger.warning("HTTP session not properly closed in FlowRunner")
        except Exception:
            pass

    def _collect_metrics(self) -> dict[str, Any]:
        avg_confidence = (
            sum(self._confidence_scores) / len(self._confidence_scores)
            if self._confidence_scores
            else 0.0
        )

        avg_transition_time = (
            sum(self._transition_times) / len(self._transition_times)
            if self._transition_times
            else 0.0
        )

        return {
            **self._metrics,
            "average_confidence": avg_confidence,
            "average_transition_time": avg_transition_time,
            "total_nodes_in_flow": len(self.flow.nodes),
            "execution_path_length": len(self.context.execution_path),
            "error_rate": self._metrics["errors"] / max(self._metrics["transitions"], 1),
        }

    def _load_and_validate_flow(self):
        try:
            self.flow: FlowSpec = load_flow(str(self.flow_path))
            self._validate_flow_structure()
            logger.info(f"Loaded and validated flow with {len(self.flow.nodes)} nodes")
        except Exception as e:
            logger.error(f"Failed to load flow from {self.flow_path}: {e}")
            raise

    def _validate_flow_structure(self) -> None:
        if self.flow.start_node_id not in self.flow.nodes:
            raise ValueError(f"Start node '{self.flow.start_node_id}' not found")

        for node_id, node in self.flow.nodes.items():
            for edge in node.edges:
                if edge.destination_node_id and edge.destination_node_id not in self.flow.nodes:
                    logger.warning(
                        f"Edge in node '{node_id}' points to non-existent node "
                        f"'{edge.destination_node_id}'"
                    )
            if node.skip_response_edge and node.skip_response_edge.destination_node_id:
                if node.skip_response_edge.destination_node_id not in self.flow.nodes:
                    logger.warning(
                        f"Skip response edge in node '{node_id}' points to non-existent node "
                        f"'{node.skip_response_edge.destination_node_id}'"
                    )

        for node_id, node in self.flow.nodes.items():
            if node.tool_id and node.tool_id not in self.flow.tools:
                logger.warning(f"Node '{node_id}' references undefined tool '{node.tool_id}'")

    def register_function(self, tool_id: str, function: Callable) -> None:
        self._registered_functions[tool_id] = function
        logger.info(f"Registered function for tool: {tool_id}")

    def get_registered_function(self, tool_id: str) -> Optional[Callable]:
        return self._registered_functions.get(tool_id)

    def _handle_task_error(self, error: Exception, node_id: str) -> None:
        self._metrics["errors"] += 1
        logger.error(f"Task error in node {node_id}: {error}")

    async def _evaluate_prompt_conditions(
        self, edges: list[Edge], user_text: str
    ) -> EdgeEvaluation:
        if not edges or not user_text:
            logger.debug(f"No edges ({len(edges) if edges else 0}) or user input ('{user_text}')")
            return EdgeEvaluation(None, None, 0.0, "No edges or user input")

        prompt_edges = [e for e in edges if e.destination_node_id and e.transition_condition]
        logger.debug(f"Found {len(prompt_edges)} prompt edges to evaluate")

        if not prompt_edges:
            logger.debug("No prompt edges found")
            return EdgeEvaluation(None, None, 0.0, "No prompt edges found")

        # Log the edges being evaluated
        for i, edge in enumerate(prompt_edges):
            logger.debug(
                f"  Edge {i + 1}: condition='{edge.transition_condition.prompt if edge.transition_condition else 'None'}' -> {edge.destination_node_id}"
            )

        try:
            start_time = time.time()
            logger.debug(f"Evaluating prompt conditions for user input: '{user_text}'")

            # Track LLM call
            self._metrics["llm_calls"] += 1

            evaluation = await self._structured_edge_evaluation(prompt_edges, user_text)

            # Track timing and confidence
            evaluation_time = time.time() - start_time
            self._transition_times.append(evaluation_time)
            self._confidence_scores.append(evaluation.confidence)

            logger.debug(
                f"Evaluation result: destination={evaluation.destination_node_id}, confidence={evaluation.confidence:.2f}, time={evaluation_time:.2f}s, reasoning='{evaluation.reasoning}'"
            )
            return evaluation

        except Exception as e:
            logger.error(f"Error evaluating prompt conditions: {e}")
            self._metrics["errors"] += 1
            raise

    async def _structured_edge_evaluation(
        self, edges: list[Edge], user_text: str
    ) -> EdgeEvaluation:
        edge_options = []
        for i, edge in enumerate(edges):
            option = {
                "option_number": i + 1,
                "destination": edge.destination_node_id,
                "condition": edge.transition_condition.prompt if edge.transition_condition else "",
            }
            edge_options.append(option)

        logger.debug(f"Edge options for evaluation: {json.dumps(edge_options, indent=2)}")

        evaluation_prompt = f"""
You are evaluating conversation flow transitions. Analyze the user input and
determine which transition condition best matches.

User Input: "{user_text}"

Available Options:
{json.dumps(edge_options, indent=2)}

Respond with a JSON object containing:
{{
    "selected_option": <number or null if no match>,
    "confidence": <float between 0.0 and 1.0>,
    "reasoning": "<explanation of decision>",
    "user_intent": "<brief description of what user wants>"
}}

If no option matches well, set selected_option to null and explain why.
"""

        logger.debug(f"Sending evaluation prompt to LLM: {evaluation_prompt}")

        try:
            edge_ctx = llm.ChatContext()
            edge_ctx.add_message(role="user", content=evaluation_prompt)

            response_parts = []
            async with self.edge_llm.chat(chat_ctx=edge_ctx) as stream:
                async for chunk in stream:
                    if chunk.delta and chunk.delta.content:
                        response_parts.append(chunk.delta.content)

            response_text = "".join(response_parts).strip()
            logger.debug(f"Raw LLM response: '{response_text}'")

            response_text = clean_json_response(response_text)
            logger.debug(f"Cleaned LLM response: '{response_text}'")

            try:
                response_data = json.loads(response_text)
                logger.debug(f"Parsed JSON response: {response_data}")
                decision = TransitionDecision.from_dict(response_data)
                logger.debug(
                    f"Decision object: selected_option={decision.selected_option}, confidence={decision.confidence}, reasoning='{decision.reasoning}'"
                )
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON response: {response_text}")
                raise ValueError(f"Failed to parse JSON response: {response_text}") from e

            if decision.selected_option and 1 <= decision.selected_option <= len(edges):
                selected_edge = edges[decision.selected_option - 1]

                logger.info(
                    f"Edge evaluation SUCCESS: {decision.reasoning} -> {selected_edge.destination_node_id} "
                    f"(confidence: {decision.confidence:.2f})"
                )
                return EdgeEvaluation(
                    selected_edge.id,
                    selected_edge.destination_node_id,
                    decision.confidence,
                    decision.reasoning,
                )
            else:
                logger.info(
                    f"Edge evaluation FAILED: No matching edge selected. Reasoning: {decision.reasoning}"
                )
                return EdgeEvaluation(None, None, decision.confidence, decision.reasoning)

        except Exception as e:
            logger.error(f"Structured evaluation failed: {e}")
            raise

    async def _check_global_nodes(self, user_text: str, current_node: Node) -> EdgeEvaluation:
        if not user_text:
            return EdgeEvaluation(None, None, 0.0, "No user input")

        global_nodes = []
        for node_id, node in self.flow.nodes.items():
            if node.global_node_setting and node.global_node_setting.condition:
                # Skip global nodes that point to the current node to prevent self-loops
                if node_id != current_node.id:
                    global_nodes.append((node_id, node.global_node_setting.condition))
                else:
                    logger.debug(
                        f"Skipping global node {node_id} (current node) to prevent self-loop"
                    )

        if not global_nodes:
            logger.debug("No global nodes defined (or all point to current node)")
            return EdgeEvaluation(
                None, None, 0.0, "No global nodes defined (or all point to current node)"
            )

        try:
            options = []
            for i, (node_id, condition) in enumerate(global_nodes):
                options.append(
                    {"option_number": i + 1, "destination": node_id, "condition": condition}
                )

            evaluation_prompt = f"""
You are checking if user input triggers any global conversation flow transitions
(like "talk to human", "end call", etc.).

User Input: "{user_text}"

Global Conditions:
{json.dumps(options, indent=2)}

Respond with JSON:
{{
    "selected_option": <number or null>,
    "confidence": <float 0.0-1.0>,
    "reasoning": "<explanation>",
    "user_intent": "<user's intent>"
}}

Global conditions should only trigger on clear, explicit user requests.
"""

            edge_ctx = llm.ChatContext()
            edge_ctx.add_message(role="user", content=evaluation_prompt)

            response_parts = []
            async with self.edge_llm.chat(chat_ctx=edge_ctx) as stream:
                async for chunk in stream:
                    if chunk.delta and chunk.delta.content:
                        response_parts.append(chunk.delta.content)

            response_text = "".join(response_parts).strip()
            response_text = clean_json_response(response_text)

            try:
                response_data = json.loads(response_text)
                decision = TransitionDecision.from_dict(response_data)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSON response: {response_text}") from e

            if decision.selected_option and 1 <= decision.selected_option <= len(global_nodes):
                node_id, condition = global_nodes[decision.selected_option - 1]
                logger.info(
                    f"Global node triggered: '{condition}' -> {node_id} "
                    f"(confidence: {decision.confidence:.2f})"
                )
                return EdgeEvaluation(
                    f"global_{node_id}",
                    node_id,
                    decision.confidence,
                    f"Global condition: {decision.reasoning}",
                )
            else:
                return EdgeEvaluation(None, None, decision.confidence, decision.reasoning)

        except Exception as e:
            logger.error(f"Error evaluating global nodes: {e}")
            return EdgeEvaluation(None, None, 0.0, f"Global evaluation failed: {e}")

    async def _evaluate_transition_conditions(
        self, node: Node, user_text: Optional[str] = None
    ) -> EdgeEvaluation:
        self._metrics["transitions"] += 1
        logger.debug("=== EVALUATING TRANSITION CONDITIONS ===")
        logger.debug(f"Node: {node.id} (type: {node.type})")
        logger.debug(f"User text: '{user_text}'")
        logger.debug(f"Node has {len(node.edges)} edges")
        logger.debug(f"Node has skip_response_edge: {node.skip_response_edge is not None}")

        global_eval = None
        if user_text:
            logger.debug("Checking global nodes first...")
            global_eval = await self._check_global_nodes(user_text, node)
            logger.debug(
                f"Global evaluation result: destination={global_eval.destination_node_id}, confidence={global_eval.confidence:.2f}"
            )
            if global_eval.destination_node_id and global_eval.confidence >= 0.7:
                logger.info(f"Global transition: {global_eval.reasoning}")
                return global_eval

        if node.skip_response_edge and node.skip_response_edge.destination_node_id:
            logger.info(f"Using skip response edge: {node.skip_response_edge.destination_node_id}")
            return EdgeEvaluation(
                node.skip_response_edge.id,
                node.skip_response_edge.destination_node_id,
                1.0,
                "Skip response edge (immediate transition)",
            )

        if not node.edges:
            logger.info("No edges found, ending flow")
            return EdgeEvaluation(None, None, 0.0, "No edges available")

        if user_text:
            logger.debug("Evaluating prompt conditions...")
            prompt_eval = await self._evaluate_prompt_conditions(node.edges, user_text)
            logger.debug(
                f"Prompt evaluation result: destination={prompt_eval.destination_node_id}, confidence={prompt_eval.confidence:.2f}"
            )
            if prompt_eval.destination_node_id:
                return prompt_eval

        if (
            user_text
            and global_eval
            and global_eval.destination_node_id
            and global_eval.confidence >= 0.5
        ):
            logger.info(f"Using lower confidence global transition: {global_eval.reasoning}")
            return global_eval

        logger.debug("No transition conditions matched")
        return EdgeEvaluation(None, None, 0.0, "No transition conditions matched")

    async def run(self, session: AgentSession) -> None:
        self._metrics["executions"] += 1
        self._session = session  # Store session for tasks that need it
        node_id: Optional[str] = self.flow.start_node_id
        iterations = 0
        start_time = time.time()
        node_iteration_count: dict[str, int] = {}  # Track iterations per node

        logger.info(f"Starting flow execution from node: {node_id}")

        try:
            while node_id and iterations < self.max_iterations:
                iterations += 1

                # Track per-node iterations for self-loops
                node_iteration_count[node_id] = node_iteration_count.get(node_id, 0) + 1

                node = self.flow.nodes.get(node_id)
                if not node:
                    raise RuntimeError(f"Node '{node_id}' not found in flow")

                logger.info(
                    f"Entering node: {node_id} (type: {node.type}) "
                    f"[global iteration {iterations}, node iteration {node_iteration_count[node_id]}]"
                )

                try:
                    transition_result = None

                    if node.type == "conversation":
                        # ConversationTask handles its own transitions and returns TransitionResult
                        task = ConversationTask(node, self)
                        transition_result = await task  # Use AgentTask await pattern
                        logger.info(
                            f"ConversationTask completed - transition_result: {transition_result}"
                        )

                    elif node.type == "function":
                        self._metrics["function_calls"] += 1
                        task = FunctionTask(node, self)
                        # FunctionTask handles its own transitions and returns TransitionResult
                        transition_result = await task
                        logger.info(
                            f"FunctionTask completed - transition_result: {transition_result}"
                        )

                    elif node.type == "end":
                        logger.info("Flow execution completed at end node")
                        if node.instruction:
                            if node.instruction.type == "prompt":
                                await session.generate_reply(instructions=node.instruction.text)
                            elif node.instruction.type == "static_text":
                                await session.say(node.instruction.text)
                        break

                    elif node.type == "transfer_call":
                        logger.info(f"Transfer call node reached: {node_id}")
                        await self._handle_transfer_call(node, session)
                        break

                    elif node.type == "gather_input":
                        self._metrics["function_calls"] += 1
                        task = GatherInputTask(node, self)
                        # GatherInputTask handles its own transitions and returns TransitionResult
                        transition_result = await task
                        logger.info(
                            f"GatherInputTask completed - transition_result: {transition_result}"
                        )
                    else:
                        logger.warning(f"Unhandled node type: {node.type}")
                        transition_result = None

                    # Check if we got a valid transition result
                    if (
                        transition_result is not None
                        and transition_result.destination_node_id
                        and transition_result.destination_node_id != node_id
                    ):
                        logger.info(
                            f"Transitioning to {transition_result.destination_node_id} "
                            f"(edge: {transition_result.edge_id}, "
                            f"confidence: {transition_result.confidence:.2f}, "
                            f"reason: {transition_result.reasoning})"
                        )
                        # Store transition context for potential use by next node
                        self.context.set_variable(
                            "last_transition",
                            {
                                "edge_id": transition_result.edge_id,
                                "user_text": transition_result.user_text,
                                "confidence": transition_result.confidence,
                                "reasoning": transition_result.reasoning,
                            },
                        )
                        node_id = transition_result.destination_node_id
                    elif (
                        transition_result is not None
                        and transition_result.destination_node_id == node_id
                    ):
                        logger.info(
                            f"Self-loop detected at node {node_id}, "
                            f"continuing with new task instance "
                            f"(iteration {node_iteration_count[node_id]})"
                        )
                        # Continue the loop with the same node_id,
                        # which will create a new task instance
                        continue
                    elif transition_result is None:
                        logger.error(f"Task failed in node {node_id}, ending flow")
                        await session.say(
                            "I'm sorry, I encountered an issue and need to end our conversation."
                        )
                        break
                    else:
                        logger.info("No valid transition found, ending flow")
                        break

                except Exception as e:
                    logger.error(f"Error executing node {node_id}: {e}")
                    await self._handle_node_error(e, node_id, session)
                    break

            if iterations >= self.max_iterations:
                logger.error(f"Flow execution exceeded maximum iterations ({self.max_iterations})")
                await session.say(
                    "I'm sorry, but I've encountered an issue. "
                    "Let me transfer you to a human agent."
                )
                raise RuntimeError("Flow execution exceeded maximum iterations")

            execution_time = time.time() - start_time
            logger.info(
                f"Flow execution finished in {execution_time:.2f}s after {iterations} iterations"
            )
            logger.info(f"Execution path: {' -> '.join(self.context.execution_path)}")

        except Exception as e:
            self._metrics["errors"] += 1
            logger.error(f"Flow execution failed: {e}")
            # Cleanup on error
            try:
                await self.cleanup()
            except Exception as cleanup_error:
                logger.warning(f"Error during cleanup after flow failure: {cleanup_error}")
            raise

    async def _handle_transfer_call(self, node: Node, session: AgentSession) -> None:
        """Handle call transfer logic"""
        logger.info(f"Processing transfer call for node {node.id}")

        # TODO: Implement transfer call logic

        if node.instruction:
            if node.instruction.type == "prompt":
                await session.generate_reply(instructions=node.instruction.text)
            elif node.instruction.type == "static_text":
                await session.say(node.instruction.text)

        if node.transfer_destination:
            logger.info(
                f"Would transfer to: {node.transfer_destination.type} - "
                f"{node.transfer_destination.number}"
            )

        if node.transfer_option:
            logger.info(f"Transfer type: {node.transfer_option.type}")

    async def _handle_node_error(
        self, error: Exception, node_id: str, session: AgentSession
    ) -> None:
        logger.error(f"Node error in {node_id}: {error}")

        try:
            await session.say(
                "I'm sorry, I encountered an issue. Let me try to help you in a different way."
            )
        except Exception as say_error:
            logger.error(f"Failed to send error message: {say_error}")

    @property
    def execution_history(self) -> list[str]:
        return self.context.execution_path.copy()

    @property
    def metrics(self) -> dict[str, Any]:
        return self._collect_metrics()

    def get_context_summary(self) -> dict[str, Any]:
        return {
            "execution_path": self.context.execution_path,
            "variables": self.context.variables,
            "function_results": self.context.function_results,
            "conversation_length": len(self.context.conversation_history),
            "execution_time": time.time() - self.context.start_time,
            "checkpoints": len(self.context.checkpoints),
            "metrics": self._collect_metrics(),
        }
