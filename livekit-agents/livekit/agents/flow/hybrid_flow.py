import asyncio
from typing import Callable, Optional

from livekit.agents import Agent, AgentSession, AgentTask, RunContext
from livekit.agents.llm.chat_context import ChatMessage
from livekit.agents.voice.agent import _set_activity_task_info

from .schema import FlowSpec, Node, load_flow


class ConversationTask(AgentTask[str]):
    def __init__(self, node: Node) -> None:
        super().__init__(instructions="")
        self.node = node

    async def on_enter(self) -> None:
        if self.node.instruction:
            if self.node.instruction.type == "prompt":
                self.session.generate_reply(instructions=self.node.instruction.text)
            else:
                self.session.say(self.node.instruction.text)

    async def on_user_turn_completed(
        self, ctx: RunContext, new_message: ChatMessage
    ) -> None:
        self.complete(new_message.text_content)


class FlowRunner:
    def __init__(
        self,
        path: str,
        edge_evaluator: Callable[[Node, str | None], Optional[str]],
    ) -> None:
        self.flow: FlowSpec = load_flow(path)
        self.edge_evaluator = edge_evaluator

    async def run(self, session: AgentSession) -> None:
        node_id: Optional[str] = self.flow.start_node_id
        last_result: str | None = None
        while node_id:
            node = self.flow.nodes[node_id]
            if node.type == "conversation" and node.instruction is not None:
                result = await ConversationTask(node)
            else:
                if node.type == "end" and node.instruction is not None:
                    if node.instruction.type == "prompt":
                        session.generate_reply(instructions=node.instruction.text)
                    else:
                        session.say(node.instruction.text)
                break
            last_result = result
            node_id = self.edge_evaluator(node, last_result)



class FlowAgent(Agent):
    def __init__(
        self, path: str, edge_evaluator: Callable[[Node, str | None], Optional[str]]
    ) -> None:
        super().__init__(instructions="")
        self.runner = FlowRunner(path, edge_evaluator)

    async def on_enter(self) -> None:
        _set_activity_task_info(asyncio.current_task(), inline_task=True)
        await self.runner.run(self.session)
