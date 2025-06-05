import asyncio
import json
import logging
from typing import Literal, TypedDict

from dotenv import load_dotenv

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli, llm, utils
from livekit.agents.llm.chat_context import ChatContext, ChatMessage
from livekit.agents.llm.tool_context import ToolContext, function_tool
from livekit.agents.llm.utils import prepare_function_arguments
from livekit.agents.voice.events import RunContext
from livekit.agents.voice.speech_handle import SpeechHandle
from livekit.plugins import openai, silero

logger = logging.getLogger("tool-checker-direct")
logger.setLevel(logging.INFO)

load_dotenv()


FunctionName = Literal["lookup_weather", "switch_agent", "none"]


class ToolDecision(TypedDict, total=False):
    function_call_needed: bool
    function_name: FunctionName
    arguments: dict[str, object]


class TimeAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="I can tell the current time.")

    async def on_enter(self) -> None:
        self.session.generate_reply(tool_choice="none")


class MyAgent(Agent):
    def __init__(self, other_agents: dict[str, Agent]):
        super().__init__(instructions="You are a helpful assistant.")
        self._tool_llm = openai.LLM(model="gpt-4o-mini")
        self._tool_llm_prompt = llm.ChatMessage(
            role="system",
            content=[
                "Decide if a function call is needed for the user's last message.",
                (
                    "If so, respond in JSON: {\"function_call_needed\": true, "
                    "\"function_name\": \"<name>\", \"arguments\": {...}}."
                ),
                "Otherwise respond with {\"function_call_needed\": false}.",
            ],
        )
        self._tools = ToolContext([self.lookup_weather, self.switch_agent])
        self._other_agents = other_agents
        self._last_tool_task: asyncio.Task[None] | None = None

    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        if self._last_tool_task and not self._last_tool_task.done():
            self._last_tool_task.cancel()

        self._last_tool_task = asyncio.create_task(self._tool_handler(turn_ctx, new_message))

    async def _tool_handler(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        ctx = (
            turn_ctx.copy(
                exclude_instructions=True,
                exclude_function_call=True,
            ).truncate(max_items=3)
        )
        ctx.items.insert(0, self._tool_llm_prompt)
        ctx.items.append(new_message)

        result = ""
        async for chunk in self._tool_llm.chat(
            chat_ctx=ctx,
            tools=list(self._tools.function_tools.values()),
            response_format=ToolDecision,
        ).to_str_iterable():
            result += chunk
        logger.info("Tool decision: %s", result)
        try:
            verdict = json.loads(result)
        except json.JSONDecodeError:
            return

        if verdict.get("function_call_needed"):
            await self._execute_tool(verdict.get("function_name"), verdict.get("arguments", {}))

    async def _execute_tool(self, fn_name: str, args: dict[str, object]) -> None:
        tool = self._tools.function_tools.get(fn_name)
        if tool is None:
            logger.warning("Unknown tool %s", fn_name)
            return

        fn_call = llm.FunctionCall(
            call_id=utils.shortuuid("fnc_"),
            arguments=json.dumps(args),
            name=fn_name,
        )
        speech_handle = SpeechHandle.create()
        run_ctx = RunContext(
            session=self.session,
            speech_handle=speech_handle,
            function_call=fn_call,
        )
        call_args, call_kwargs = prepare_function_arguments(
            fnc=tool,
            json_arguments=json.dumps(args),
            call_ctx=run_ctx,
        )
        output = await tool(*call_args, **call_kwargs)

        self._chat_ctx.items.append(fn_call)
        out_msg = llm.FunctionCallOutput(
            name=fn_name,
            call_id=fn_call.call_id,
            output=str(output) if output is not None else "",
            is_error=False,
        )
        self._chat_ctx.items.append(out_msg)

        if isinstance(output, Agent):
            self.session.update_agent(output)
        elif output:
            await self.session.say(str(output))

    @function_tool()
    async def lookup_weather(self, location: str, context: RunContext) -> str:
        logger.info("Looking up weather for %s", location)
        return f"It is always sunny in {location}."

    @function_tool()
    async def switch_agent(self, agent: Literal["time"], context: RunContext) -> Agent:
        return self._other_agents[agent]


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()

    time_agent = TimeAgent()
    agent = MyAgent({"time": time_agent})

    session = AgentSession(
        stt=openai.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=openai.TTS(),
        vad=silero.VAD.load(),
    )
    await session.start(agent=agent, room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
