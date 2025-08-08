import logging
import os
from pathlib import Path

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

logger = logging.getLogger("flow-console")
logger.setLevel(logging.INFO)

load_dotenv()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    # Resolve flow path (default to advanced example)
    default_flow = Path(__file__).parents[1] / "flows" / "advanced_test_flow.json"
    flow_path = os.environ.get("FLOW_PATH", str(default_flow))

    # LLMs: edge evaluator can be a smaller model
    conversation_llm = openai.LLM(model="gpt-4o")
    edge_llm = openai.LLM(model="gpt-4o-mini")

    flow_runner = FlowRunner(flow_path=flow_path, edge_evaluator_llm=edge_llm)

    # Register local function handlers
    def _triage_handler(args: dict):
        desc = (args.get("issue_description") or "").lower()
        priority = (args.get("priority") or "").lower()
        # naive severity heuristic
        if any(k in desc for k in ["crash", "data loss", "security", "breach"]):
            severity = "critical"
        elif any(k in desc for k in ["error", "sync", "fail", "timeout"]):
            severity = "major"
        elif desc.strip():
            severity = "minor"
        else:
            severity = "none"
        # pretend emails from example.com are not verified to exercise path
        email = (args.get("customer_email") or "").lower()
        verified = not email.endswith("@example.com") and bool(email)
        return {
            "issue_severity": severity,
            "customer_verified": verified,
            "priority": priority,
            "allowed_severities": ["none", "minor", "major", "critical"],
        }

    flow_runner.register_function("triage-tool", _triage_handler)

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=openai.STT(),
        llm=conversation_llm,
        tts=openai.TTS(),
    )

    await flow_runner.start(session)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
