import logging

from livekit.agents import llm
from livekit.agents.llm import StopResponse

from ..base import BaseFlowAgent, FlowTransition

logger = logging.getLogger(__name__)


class ConversationNodeAgent(BaseFlowAgent):
    async def _on_enter_node(self) -> None:
        if self.node.skip_response_edge and self.node.skip_response_edge.destination_node_id:
            logger.info(f"SKIP: {self.node.name} -> [Auto-transition]")

            transition = FlowTransition(
                destination_node_id=self.node.skip_response_edge.destination_node_id,
                edge_id=self.node.skip_response_edge.id,
                confidence=1.0,
                reasoning="Skip response edge - immediate transition",
            )

            self.flow_context.set_variable(
                f"transition_from_{self.node.id}",
                {
                    "type": "skip_response",
                    "to": transition.destination_node_id,
                    "reasoning": transition.reasoning,
                },
            )

            await self._transition_to_node(transition.destination_node_id)
            return

        if self.node.instruction and self.node.instruction.type == "prompt":
            logger.debug(f"READY: {self.node.name}")

    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        if self._transition_pending:
            logger.debug(f"Transition already pending for node {self.node.id} ({self.node.name}), ignoring turn")
            return

        try:
            user_text = new_message.text_content or ""

            logger.info(f"USER [{self.node.name}]: '{user_text}'")

            self.flow_context.add_message(new_message)

            transition = await self._evaluate_transition(user_text)

            if transition and transition.destination_node_id:
                dest_node = self.flow_runner.flow.nodes.get(transition.destination_node_id)
                dest_name = dest_node.name if dest_node else transition.destination_node_id
                logger.info(f"FLOW: {self.node.name} -> {dest_name} ({transition.reasoning})")

                self.session.interrupt()

                self.flow_context.set_variable(
                    f"transition_from_{self.node.id}",
                    {
                        "type": "user_triggered",
                        "user_text": user_text,
                        "to": transition.destination_node_id,
                        "confidence": transition.confidence,
                        "reasoning": transition.reasoning,
                    },
                )

                await self._transition_to_node(transition.destination_node_id)
                
                # Raise StopResponse to prevent LLM from generating a response
                raise StopResponse()
            else:
                logger.debug(f"CONTINUE: {self.node.name}")

        except StopResponse:
            # Re-raise StopResponse to prevent agent from generating a response
            raise
        except Exception as e:
            logger.error(f"Error handling user turn in conversation node {self.node.id} ({self.node.name}): {e}")
            await self.session.say("error occurred while handling user turn")

    async def _on_exit_node(self) -> None:
        conversation_summary = {
            "node_id": self.node.id,
            "message_count": len(self.flow_context.conversation_history),
            "had_skip_edge": bool(self.node.skip_response_edge),
            "edge_count": len(self.node.edges) if self.node.edges else 0,
        }

        self.flow_context.set_variable(f"conversation_summary_{self.node.id}", conversation_summary)

        logger.debug(
            f"Exiting conversation node {self.node.id} ({self.node.name}) with summary: {conversation_summary}"
        )
