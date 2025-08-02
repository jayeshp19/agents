import logging

from dotenv import load_dotenv

from livekit.agents import (
    AgentSession,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
)
from livekit.agents.flow import FlowRunner
from livekit.plugins import openai, silero

logger = logging.getLogger("flow-agent-v2")
logger.setLevel(logging.DEBUG)

logging.getLogger("livekit.agents.flow").setLevel(logging.DEBUG)

load_dotenv()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    flow_path = "/Users/jayesh/Developer/personal/agents/examples/flows/tech_support_example.json"

    conversation_llm = openai.LLM(model="gpt-4o")
    edge_llm = openai.LLM(model="gpt-4o-mini")

    flow_runner = FlowRunner(flow_path=flow_path, edge_evaluator_llm=edge_llm)

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=openai.STT(),
        llm=conversation_llm,
        tts=openai.TTS(),
    )

    @session.on("conversation_item_added")
    def conversation_item_added(msg):
        logger.info(f"Conversation item added, {msg.item.role}: {msg.item.content}")

    await flow_runner.start(session)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
