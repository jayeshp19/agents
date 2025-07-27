import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from livekit.agents import Agent, llm

if TYPE_CHECKING:
    from .runner import FlowRunner
    from .schema import Node

logger = logging.getLogger(__name__)


@dataclass
class FlowTransition:
    destination_node_id: str
    edge_id: Optional[str] = None
    confidence: float = 1.0
    reasoning: str = ""
    user_text: Optional[str] = None


class FlowContext:
    def __init__(self):
        self.variables: dict[str, Any] = {}
        self.conversation_history: list[llm.ChatMessage] = []
        self.current_node_id: Optional[str] = None
        self.execution_path: list[str] = []
        self.function_results: dict[str, Any] = {}

    def add_message(self, message: llm.ChatMessage) -> None:
        self.conversation_history.append(message)

    def set_variable(self, key: str, value: Any) -> None:
        self.variables[key] = value

    def get_variable(self, key: str, default: Any = None) -> Any:
        return self.variables.get(key, default)

    def record_transition(self, from_node: str, to_node: str) -> None:
        if from_node not in self.execution_path:
            self.execution_path.append(from_node)
        self.execution_path.append(to_node)
        self.current_node_id = to_node


class BaseFlowAgent(Agent, ABC):
    def __init__(
        self, node: "Node", flow_runner: "FlowRunner", flow_context: FlowContext, **kwargs
    ):
        instructions = ""
        if node.instruction:
            if node.instruction.type == "prompt":
                instructions = node.instruction.text
            elif node.instruction.type == "static_text":
                instructions = node.instruction.text

        super().__init__(instructions=instructions, **kwargs)

        self.node = node
        self.flow_runner = flow_runner
        self.flow_context = flow_context
        self._transition_pending = False

        logger.debug(f"Initialized {self.__class__.__name__} for node {node.id} ({node.name})")

    async def on_enter(self) -> None:
        logger.info(f"ENTER: {self.node.name}")

        if self.flow_context.current_node_id:
            self.flow_context.record_transition(self.flow_context.current_node_id, self.node.id)
        else:
            self.flow_context.current_node_id = self.node.id
            self.flow_context.execution_path.append(self.node.id)

        # Handle node instruction if static text
        if self.node.instruction and self.node.instruction.type == "static_text":
            await self.session.say(self.node.instruction.text)

        # Execute node-specific entry logic
        await self._on_enter_node()

    async def on_exit(self) -> None:
        logger.debug(f"EXIT: {self.node.name}")
        await self._on_exit_node()

    @abstractmethod
    async def _on_enter_node(self) -> None:
        pass

    async def _on_exit_node(self) -> None:
        pass

    async def _transition_to_node(self, node_id: str) -> None:
        if self._transition_pending:
            logger.warning(f"Transition already pending from node {self.node.id}")
            return

        self._transition_pending = True

        try:
            logger.info(f"Transitioning from {self.node.id} ({self.node.name}) to {node_id}")

            next_agent = await self.flow_runner.get_or_create_agent(node_id)

            if next_agent:
                # Update the agent
                self.session.update_agent(next_agent)
                logger.info(f"Updated agent to {next_agent.__class__.__name__} for node {node_id}")

                # The new agent's on_enter will be called by the session's activity management
                # We don't need to wait here as the session handles the lifecycle
            else:
                logger.error(f"Failed to create agent for node {node_id}")
                await self._handle_transition_error(node_id)
                # Reset transition pending on error
                self._transition_pending = False

        except Exception as e:
            logger.error(f"Error during transition from {self.node.id} to {node_id}: {e}")
            await self._handle_transition_error(node_id, e)
            # Reset transition pending on error
            self._transition_pending = False

    async def _handle_transition_error(
        self, target_node_id: str, error: Optional[Exception] = None
    ) -> None:
        error_msg = f"Encountered an issue transitioning to {target_node_id}"
        if error:
            logger.error(f"Transition error details: {error}", exc_info=True)

        await self.session.say(error_msg)

    async def _evaluate_transition(
        self, user_text: Optional[str] = None
    ) -> Optional[FlowTransition]:
        return await self.flow_runner.evaluate_transition(self.node, user_text)

    def _get_node_context(self) -> dict[str, Any]:
        return {
            "node_id": self.node.id,
            "node_type": self.node.type,
            "flow_variables": self.flow_context.variables,
            "execution_path": self.flow_context.execution_path,
        }
