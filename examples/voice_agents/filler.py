import asyncio

from livekit.agents import AgentSession, JobContext, WorkerOptions, cli
from livekit.agents.llm.tool_context import function_tool
from livekit.agents.voice.agent import Agent
from livekit.agents.voice.events import UserStateChangedEvent
from livekit.plugins import deepgram, google, groq, silero


class MyAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are a helpful assistant",
        )

    @function_tool
    async def get_weather(self, city: str) -> str:
        await asyncio.sleep(2)
        return f"The weather in {city} is sunny."


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    # — your normal Session
    session = AgentSession(
        stt=deepgram.STT(),
        tts=google.TTS(),
        llm=google.LLM(),
        vad=silero.VAD.load(),
    )

    # — second “filler” LLM
    filler_llm = groq.LLM(temperature=0.7)

    # keep a handle so we can cancel if the user stops speaking
    filler_speech = None
    # keep track of all previous filler outputs to avoid repeats
    seen_fillers: list[str] = []

    async def on_user_state(ev: UserStateChangedEvent):
        nonlocal filler_speech, seen_fillers

        if ev.new_state == "speaking":
            # build a prompt from the conversation so far, instructing not to repeat past fillers
            filler_chat_ctx = session.history.copy()
            system_prompt = (
                "You're a filler-word assistant. As the user speaks, generate natural "
                "filler utterances (uh-huh, mm-hm, okay…) based on the conversation so far."
            )
            if seen_fillers:
                system_prompt += (
                    " Do not repeat any of these fillers: " + ", ".join(seen_fillers) + "."
                )
            filler_chat_ctx.add_message(role="system", content=system_prompt)

            print(filler_chat_ctx.items)
            # stream LLM → TTS in parallel with STT
            out = ""
            async for chunk in filler_llm.chat(chat_ctx=filler_chat_ctx).to_str_iterable():
                out += chunk
            print(out)

            filler_speech = session.say(out, add_to_chat_ctx=False, allow_interruptions=False)

    # subscribe to the state‐change event
    session.on("user_state_changed", lambda ev: asyncio.create_task(on_user_state(ev)))

    # now start your real agent (or skip it if you only want filler)
    await session.start(MyAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
