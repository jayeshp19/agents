import asyncio
import logging
from pathlib import Path
from typing import Any, Callable, Optional

import aiohttp

from livekit.agents import Agent, AgentSession, llm

from .agents import ConversationNodeAgent, FunctionNodeAgent, GatherInputNode
from .base import FlowContext, FlowTransition
from .schema import FlowSpec, Node, load_flow
from .transition_evaluator import FlowContextVariableProvider, TransitionEvaluator

logger = logging.getLogger(__name__)


NODE_AGENT_CLASSES: dict[str, type[Agent]] = {
    "conversation": ConversationNodeAgent,
    "gather_input": GatherInputNode,
    "function": FunctionNodeAgent,
    "end": ConversationNodeAgent,
}


class FlowRunner:
    def __init__(
        self,
        flow_path: str,
        edge_evaluator_llm: llm.LLM,
        initial_context: Optional[dict[str, Any]] = None,
        *,
        max_total_transitions: int = 200,
        max_node_revisits: int = 20,
        min_prompt_confidence: float = 0.0,
        min_prompt_eval_interval_ms: int = 0,
    ):
        self.flow_path = Path(flow_path)
        self.edge_llm = edge_evaluator_llm

        self._load_and_validate_flow()

        self.context = FlowContext()
        if initial_context:
            self.context.variables.update(initial_context)

        self._agent_cache: dict[str, Agent] = {}

        self._registered_functions: dict[str, dict[str, Any]] = {}
        self._function_handlers: dict[str, Callable[..., Any]] = {}

        self._http_session: Optional[aiohttp.ClientSession] = None
        self._cache_lock = asyncio.Lock()

        self._session: Optional[AgentSession] = None

        # Loop protection thresholds
        self._max_total_transitions = max(1, max_total_transitions)
        self._max_node_revisits = max(1, max_node_revisits)

        # Create transition evaluator
        provider = FlowContextVariableProvider(self.context)
        self._transition_evaluator = TransitionEvaluator(
            provider,
            edge_llm=self.edge_llm,
            min_prompt_confidence=min_prompt_confidence,
            min_prompt_eval_interval_ms=min_prompt_eval_interval_ms,
        )

        logger.info(f"FlowRunner initialized with flow: {self.flow.conversation_flow_id}")

    def _load_and_validate_flow(self) -> None:
        try:
            self.flow: FlowSpec = load_flow(str(self.flow_path))
            # Prefer FlowSpec's comprehensive validation
            structure_errors = self.flow.validate_flow_structure()
            if structure_errors:
                raise ValueError("Invalid flow structure: " + "; ".join(structure_errors))
            logger.info(f"Loaded flow with {len(self.flow.nodes)} nodes")
        except Exception as e:
            logger.error(f"Failed to load flow from {self.flow_path}: {e}")
            raise

    async def start(self, session: AgentSession) -> None:
        self._session = session

        initial_agent = await self.get_or_create_agent(self.flow.start_node_id)

        if not initial_agent:
            raise RuntimeError(f"Failed to create initial agent for node {self.flow.start_node_id}")

        start_node = self.flow.nodes.get(self.flow.start_node_id)
        start_name = start_node.name if start_node else self.flow.start_node_id
        logger.info(f"FLOW START: {start_name}")
        logger.info(
            f"FLOW PATH: {' -> '.join([node.name for node in self.flow.nodes.values()][:5])}{'...' if len(self.flow.nodes) > 5 else ''}"
        )
        await session.start(agent=initial_agent)

    # ---- Loop protection helpers ----
    def can_transition_to(self, node_id: str) -> bool:
        """Return False if transition exceeds configured limits."""
        # Total hops limit
        if self.context.total_transitions >= self._max_total_transitions:
            return False
        # Per-node revisit limit
        if self.context.transition_counts.get(node_id, 0) >= self._max_node_revisits:
            return False
        return True

    def find_end_node_id(self) -> Optional[str]:
        try:
            for nid, node in self.flow.nodes.items():
                if str(node.type) == "end":
                    return nid
        except Exception:
            pass
        return None

    async def get_or_create_agent(self, node_id: str) -> Optional[Agent]:
        async with self._cache_lock:
            if node_id in self._agent_cache:
                cached_agent = self._agent_cache[node_id]
                return cached_agent

            node = self.flow.nodes.get(node_id)
            if not node:
                logger.error(f"Node {node_id} not found in flow")
                return None

            try:
                agent = self._create_agent_for_node(node)
                self._agent_cache[node_id] = agent
                return agent

            except Exception as e:
                logger.error(f"Failed to create agent for node {node_id} ({node.name}): {e}")
                return None

    def _create_agent_for_node(self, node: Node) -> Agent:
        node_type = str(node.type)
        agent_cls = NODE_AGENT_CLASSES.get(node_type)
        if agent_cls is None:
            raise ValueError(f"Unknown node type: {node.type}")

        kwargs: dict[str, Any] = {
            "node": node,
            "flow_runner": self,
            "flow_context": self.context,
        }
        if agent_cls is GatherInputNode:
            kwargs["extraction_llm"] = self.edge_llm

        return agent_cls(**kwargs)

    def register_function(self, tool_id: str, handler: Callable[..., Any]) -> None:
        self._function_handlers[tool_id] = handler
        logger.debug(f"Registered function handler for {tool_id}")

    def get_function_definition(self, tool_id: str) -> Optional[dict[str, Any]]:
        if tool_id in self._registered_functions:
            return self._registered_functions[tool_id]

        if self.flow.tools and tool_id in self.flow.tools:
            tool = self.flow.tools[tool_id]
            return {
                "name": tool.name,
                "description": tool.description,
                "tool_id": tool.tool_id,
                "type": tool.type,
                "parameters": tool.parameters,
                "url": tool.url,
                "http_method": tool.http_method,
                "headers": tool.headers,
                "timeout_ms": tool.timeout_ms,
                "response_variables": tool.response_variables,
            }

        return None

    def get_function_handler(self, tool_id: str) -> Optional[Callable]:
        return self._function_handlers.get(tool_id)

    async def get_http_session(self) -> aiohttp.ClientSession:
        async with self._cache_lock:  # Reuse cache lock for HTTP session
            if self._http_session is None or self._http_session.closed:
                timeout = aiohttp.ClientTimeout(total=300)
                connector = aiohttp.TCPConnector(
                    limit=100,  # Total connection pool size
                    limit_per_host=30,  # Per host limit
                    ttl_dns_cache=300,  # DNS cache TTL
                    use_dns_cache=True,
                )
                self._http_session = aiohttp.ClientSession(timeout=timeout, connector=connector)
                logger.debug("Created new HTTP session with connection limits")

        return self._http_session

    async def cleanup(self) -> None:
        """Cleanup resources with timeout and error handling."""
        cleanup_tasks = []

        # HTTP session cleanup
        if self._http_session and not self._http_session.closed:
            cleanup_tasks.append(self._cleanup_http_session())

        # Execute cleanup tasks with timeout
        if cleanup_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*cleanup_tasks, return_exceptions=True), timeout=10.0
                )
            except asyncio.TimeoutError:
                logger.warning("HTTP session cleanup timed out")
            except Exception as e:
                logger.error(f"Error during HTTP session cleanup: {e}")

        self._agent_cache.clear()
        logger.info("FlowRunner cleanup completed")

    async def _cleanup_http_session(self) -> None:
        try:
            if self._http_session is not None:
                await self._http_session.close()
            # Wait a bit for connections to close properly
            await asyncio.sleep(0.1)
            logger.debug("HTTP session closed successfully")
        except Exception as e:
            logger.warning(f"Error closing HTTP session: {e}")

    async def evaluate_transition(
        self, node: Node, user_text: Optional[str] = None
    ) -> Optional[FlowTransition]:
        logger.debug(f"Evaluating transition for node {node.id}")

        # Build list of edges to evaluate, including global node settings
        all_edges = []

        # Add current node's edges
        if node.edges:
            all_edges.extend(node.edges)

        # Add global node settings as edges (if user text provided)
        if user_text:
            for global_node_id, global_node in self.flow.nodes.items():
                if (
                    global_node.global_node_setting
                    and global_node.global_node_setting.condition
                    and global_node_id != node.id
                ):
                    # Create a synthetic edge for global node transition
                    from .fields import Edge, TransitionCondition

                    global_edge = Edge(
                        id=f"global_{global_node_id}",
                        condition=global_node.global_node_setting.condition,
                        transition_condition=TransitionCondition(
                            type="prompt", prompt=global_node.global_node_setting.condition
                        ),
                        destination_node_id=global_node_id,
                    )
                    all_edges.append(global_edge)

        if not all_edges:
            return None

        # Use the centralized transition evaluator
        return await self._transition_evaluator.evaluate_transitions(
            edges=all_edges,
            user_text=user_text,
            context={},  # Additional context can be passed here if needed
        )
