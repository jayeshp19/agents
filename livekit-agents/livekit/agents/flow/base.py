import asyncio
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
    def __init__(self) -> None:
        self.variables: dict[str, Any] = {}
        self.conversation_history: list[llm.ChatMessage] = []
        self.current_node_id: Optional[str] = None
        self.execution_path: list[str] = []
        self.function_results: dict[str, Any] = {}
        # Transition tracking for loop protection
        self.transition_counts: dict[str, int] = {}
        self.total_transitions: int = 0

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
        self.total_transitions += 1
        self.transition_counts[to_node] = self.transition_counts.get(to_node, 0) + 1


class BaseFlowAgent(Agent, ABC):
    def __init__(
        self,
        node: "Node",
        flow_runner: "FlowRunner",
        flow_context: FlowContext,
        **kwargs: Any,
    ) -> None:
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
        self._transition_lock = asyncio.Lock()

    async def on_enter(self) -> None:
        if self.flow_context.current_node_id:
            self.flow_context.record_transition(self.flow_context.current_node_id, self.node.id)
        else:
            self.flow_context.current_node_id = self.node.id
            self.flow_context.execution_path.append(self.node.id)

        # Handle node instruction if static text
        # Respect flow start speaker: if this is the start node and start speaker is USER, suppress auto-say
        suppress_autosay = False
        try:
            flow = self.flow_runner.flow
            if (
                flow
                and flow.start_node_id == self.node.id
                and str(getattr(flow, "start_speaker", "")).lower() == "user"
            ):
                suppress_autosay = True
        except Exception:
            pass

        if (
            self.node.instruction
            and self.node.instruction.type == "static_text"
            and not suppress_autosay
        ):
            await self.session.say(self.node.instruction.text)

        # Execute node-specific entry logic
        await self._on_enter_node()

    async def on_exit(self) -> None:
        await self._on_exit_node()

    @abstractmethod
    async def _on_enter_node(self) -> None:
        pass

    async def _on_exit_node(self) -> None:
        pass

    async def _try_interrupt(self, timeout: float = 0.2) -> None:
        """Best-effort session interrupt with a tiny timeout."""
        try:
            fut = self.session.interrupt()
            await asyncio.wait_for(fut, timeout=timeout)
        except Exception:
            pass

    async def _cancel_tasks(
        self, tasks: list[asyncio.Task[Any]], *, timeout: float = 2.0, phase: str = ""
    ) -> None:
        """Cancel a list of tasks and await their completion with timeout.

        Silent on expected cancellation exceptions; logs timeouts at debug level.
        """
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True), timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.debug(
                    f"Some tasks failed to cancel within {timeout:.1f}s during {phase or 'cleanup'}"
                )
            except Exception as e:
                logger.debug(f"Expected exception during task cancellation: {e}")
                # Ignore miscellaneous cancellation-related exceptions
                pass

    async def _transition_to_node(self, node_id: str) -> None:
        """Safely transition to a new node with race condition protection."""
        async with self._transition_lock:
            if self._transition_pending:
                logger.warning(f"Transition already pending from node {self.node.id}")
                return

            self._transition_pending = True

        try:
            # Loop protection: check with flow runner if we can transition
            if hasattr(
                self.flow_runner, "can_transition_to"
            ) and not self.flow_runner.can_transition_to(node_id):
                logger.error(
                    f"Transition to {node_id} blocked by loop protection (total={self.flow_context.total_transitions}, count={self.flow_context.transition_counts.get(node_id, 0)})"
                )
                # Attempt graceful handling: try to route to an end node if available
                try:
                    end_id = getattr(self.flow_runner, "find_end_node_id", lambda: None)()
                    if end_id and end_id != self.node.id and end_id != node_id:
                        next_agent = await self.flow_runner.get_or_create_agent(end_id)
                        if next_agent:
                            self.session.update_agent(next_agent)
                        return
                except Exception:
                    pass
                await self.session.say(
                    "I'm encountering a loop in the conversation. Let's pause here."
                )
                return

            next_agent = await self.flow_runner.get_or_create_agent(node_id)

            if next_agent:
                # Update the agent
                self.session.update_agent(next_agent)

                # The new agent's on_enter will be called by the session's activity management
                # We don't need to wait here as the session handles the lifecycle
            else:
                logger.error(f"Failed to create agent for node {node_id}")
                await self._handle_transition_error(node_id)

        except Exception as e:
            logger.error(f"Error during transition from {self.node.id} to {node_id}: {e}")
            await self._handle_transition_error(node_id, e)
        finally:
            async with self._transition_lock:
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
