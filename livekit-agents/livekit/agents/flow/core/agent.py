import asyncio
import logging
from typing import Any, Callable, Optional

from livekit.agents import Agent, llm
from livekit.agents.voice.agent import _set_activity_task_info

from .runner import FlowRunner

logger = logging.getLogger(__name__)


class FlowAgent(Agent):
    def __init__(
        self,
        path: str,
        edge_evaluator_llm: llm.LLM,
        initial_context: Optional[dict[str, Any]] = None,
        max_iterations: int = 100,
        **kwargs,
    ) -> None:
        super().__init__(instructions="", **kwargs)
        self.runner = FlowRunner(path, edge_evaluator_llm, initial_context, max_iterations)
        self._active_conversation_task = None

    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        """Delegate user turn completion to the active conversation task."""
        if self._active_conversation_task:
            await self._active_conversation_task.on_user_turn_completed(turn_ctx, new_message)

    def set_active_conversation_task(self, task) -> None:
        """Set the currently active conversation task for event delegation."""
        self._active_conversation_task = task

    def clear_active_conversation_task(self) -> None:
        """Clear the active conversation task."""
        self._active_conversation_task = None

    async def on_enter(self) -> None:
        try:
            logger.info(f"Starting flow agent with flow: {self.runner.flow.conversation_flow_id}")
            current_task = asyncio.current_task()
            if current_task is not None:
                _set_activity_task_info(current_task, inline_task=True)
            await self.runner.run(self.session)

        except Exception as e:
            logger.error(f"Flow agent execution failed: {e}")

            try:
                await self.session.say(
                    "I apologize, but I'm experiencing technical difficulties. "
                    "Please contact our support team for assistance."
                )
            except Exception as fallback_error:
                logger.error(f"Even fallback message failed: {fallback_error}")
            raise
        finally:
            # Cleanup resources
            try:
                await self.runner.cleanup()
            except Exception as cleanup_error:
                logger.warning(f"Cleanup error: {cleanup_error}")

    def register_tool_function(self, tool_id: str, function: Callable) -> None:
        self.runner.register_function(tool_id, function)

    @property
    def execution_history(self) -> list[str]:
        return self.runner.execution_history

    @property
    def flow_metrics(self) -> dict[str, Any]:
        return self.runner.metrics

    def get_flow_context(self) -> dict[str, Any]:
        return self.runner.get_context_summary()
