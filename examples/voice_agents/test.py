import logging

from dotenv import load_dotenv

from livekit.agents import (
    AgentSession,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
)
from livekit.agents.flow import FlowAgent
from livekit.plugins import openai
from livekit.plugins.silero import VAD

logger = logging.getLogger("flow-agent")
load_dotenv()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = VAD.load()


async def entrypoint(ctx: JobContext):
    conversation_llm = openai.LLM(model="gpt-4o")
    edge_llm = openai.LLM(model="gpt-4o-mini")

    agent = FlowAgent(
        path="livekit-agents/livekit/agents/flow/retell.json",
        edge_evaluator_llm=edge_llm,
    )
    agent._llm = conversation_llm

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        llm=conversation_llm,
        stt=openai.STT(),
        tts=openai.TTS(),
    )

    await session.start(
        agent=agent,
        room=ctx.room,
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
