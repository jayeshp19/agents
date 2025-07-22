import asyncio
import logging
from typing import TYPE_CHECKING, Optional

from livekit.agents import AgentTask, llm

from ..schema import Node
from ..types.types import TransitionResult

if TYPE_CHECKING:
    from ..core.runner import FlowRunner

logger = logging.getLogger(__name__)


class ConversationTask(AgentTask[Optional[TransitionResult]]):
    """Handles conversation flow in a single node.

    This task manages user interactions within a conversation node,
    evaluates transition conditions, and determines the next node.
    """

    def __init__(self, node: Node, flow_runner: "FlowRunner") -> None:
        # Use empty instructions as we'll handle conversation through the FlowAgent
        super().__init__(instructions="")
        self.node = node
        self.flow_runner = flow_runner
        self.turn_count = 0
        self.last_user_input: Optional[str] = None
        self.should_transition = False
        self._completed = False
        self._result_future: asyncio.Future[Optional[TransitionResult]] = asyncio.Future()

    def complete(self, result: Optional[TransitionResult]) -> None:
        """Complete the task with the given result."""
        if not self._completed:
            self._completed = True

            # Unregister from user turn events
            session = self.flow_runner._session
            self._unregister_user_turn_handler(session)

            # Complete our own future
            if not self._result_future.done():
                self._result_future.set_result(result)
                logger.info(f"ConversationTask completed with result: {result}")

    async def __await_impl(self) -> Optional[TransitionResult]:
        """Override AgentTask.__await_impl to avoid session switching deadlock."""
        logger.info("ConversationTask starting (no session switching)")

        # Skip the complex session switching and directly call on_enter
        try:
            await self.on_enter()
            # Wait for completion without session switching
            return await asyncio.shield(self._result_future)
        except Exception as e:
            logger.error(f"ConversationTask execution failed: {e}")
            if not self._result_future.done():
                self._result_future.set_exception(e)
            raise

    def __await__(self):
        """Override to use our custom await implementation."""
        return self.__await_impl().__await__()

    async def on_enter(self) -> None:
        """Called when entering the conversation node."""
        try:
            self.flow_runner.context.execution_path.append(self.node.id)
            self.flow_runner.context.save_checkpoint(self.node.id)

            # Get session from FlowRunner instead of self.session (which needs activity context)
            session = self.flow_runner._session

            if self.node.instruction:
                if self.node.instruction.type == "prompt":
                    await session.generate_reply(instructions=self.node.instruction.text)
                elif self.node.instruction.type == "static_text":
                    await session.say(self.node.instruction.text)

            if self.node.skip_response_edge:
                self.should_transition = True
                transition_result = TransitionResult(
                    destination_node_id=self.node.skip_response_edge.destination_node_id,
                    edge_id=self.node.skip_response_edge.id,
                    user_text=None,
                    confidence=1.0,
                    reasoning="Skip response edge",
                )
                self.complete(transition_result)
                return

            # Register for user turn completed events through the parent FlowAgent
            self._register_user_turn_handler(session)

        except Exception as e:
            logger.error(f"Error in ConversationTask.on_enter for node {self.node.id}: {e}")
            self.flow_runner._handle_task_error(e, self.node.id)
            self.complete(None)
            return

    def _register_user_turn_handler(self, session) -> None:
        """Register to handle user turn completion events."""
        # Get the parent FlowAgent and register this task for event delegation
        if hasattr(session, "current_agent") and hasattr(
            session.current_agent, "set_active_conversation_task"
        ):
            session.current_agent.set_active_conversation_task(self)
            logger.debug(f"Registered ConversationTask {self.node.id} with FlowAgent")

    def _unregister_user_turn_handler(self, session) -> None:
        """Unregister from user turn completion events."""
        if hasattr(session, "current_agent") and hasattr(
            session.current_agent, "clear_active_conversation_task"
        ):
            session.current_agent.clear_active_conversation_task()
            logger.debug(f"Unregistered ConversationTask {self.node.id} from FlowAgent")

    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        """Called when user completes a turn in the conversation."""
        if self._completed:
            return

        try:
            self.turn_count += 1
            text_content = new_message.text_content or ""
            self.last_user_input = text_content

            logger.debug("=== USER TURN COMPLETED ===")
            logger.debug(f"Turn count: {self.turn_count}")
            logger.debug(f"Node: {self.node.id}")
            logger.debug(f"User input: '{text_content}'")
            logger.debug(f"Should transition: {self.should_transition}")

            self.flow_runner.context.add_message(new_message)

            logger.debug(f"User turn {self.turn_count} in node {self.node.id}: {text_content}")

            if self.should_transition:
                logger.debug("Already marked for transition, skipping evaluation")
                return

            logger.debug("Evaluating transition conditions...")
            evaluation = await self.flow_runner._evaluate_transition_conditions(
                self.node, text_content
            )

            logger.debug(
                f"Evaluation result: destination={evaluation.destination_node_id}, confidence={evaluation.confidence:.2f}"
            )

            if evaluation.destination_node_id:
                logger.info(
                    f"Transitioning from {self.node.id} to {evaluation.destination_node_id} "
                    f"(edge: {evaluation.edge_id}, confidence: {evaluation.confidence:.2f})"
                )
                self.should_transition = True
                transition_result = TransitionResult.from_evaluation(evaluation, text_content)
                logger.debug(f"Completing task with transition result: {transition_result}")
                self.complete(transition_result)
            else:
                logger.debug(
                    f"Continuing conversation in node {self.node.id} - no valid transition found"
                )

        except Exception as e:
            logger.error(f"Error in user turn completion for node {self.node.id}: {e}")
            self.flow_runner._handle_task_error(e, self.node.id)
            self.complete(None)
