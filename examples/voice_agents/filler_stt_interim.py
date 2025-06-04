import asyncio
import time
from collections.abc import AsyncIterable

from google.cloud import texttospeech

from livekit import rtc
from livekit.agents import AgentSession, JobContext, WorkerOptions, cli, stt
from livekit.agents.llm.tool_context import function_tool
from livekit.agents.voice.agent import Agent, ModelSettings
from livekit.plugins import google, groq, silero
from livekit.plugins.google import TTS as GoogleTTS


class FillerAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are a helpful assistant",
        )
        self.first_interim_received = False
        self.current_turn_id = 0
        self.filler_llm = groq.LLM(temperature=0.7)
        self.seen_fillers = []

    @function_tool
    async def get_weather(self, city: str) -> str:
        await asyncio.sleep(2)
        return f"The weather in {city} is sunny."

    async def stt_node(self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings):
        default_stream = Agent.default.stt_node(self, audio, model_settings)

        async for event in default_stream:
            await self._process_stt_event(event)
            yield event

    async def play_filler_audio(self, text: str):
        audio_out = self.session.output.audio
        if audio_out is None:
            return

        tts = GoogleTTS(audio_encoding=texttospeech.AudioEncoding.LINEAR16)
        stream = tts.synthesize(text)
        async for ev in stream:
            await audio_out.capture_frame(ev.frame)
        await tts.aclose()

    async def _process_stt_event(self, event: stt.SpeechEvent):
        if event.type == stt.SpeechEventType.INTERIM_TRANSCRIPT:
            transcript_text = event.alternatives[0].text.strip()

            if not self.first_interim_received and transcript_text:
                self.first_interim_received = True
                print(f"First interim transcript detected: '{transcript_text}'")
                print(f"Confidence: {event.alternatives[0].confidence:.2f}")

                asyncio.create_task(self._generate_and_play_filler())
            else:
                print(f"Interim update: '{transcript_text}'")

        elif event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
            transcript_text = event.alternatives[0].text.strip()
            print(f"Final transcript: '{transcript_text}'")
            self._reset_turn_state()

        elif event.type == stt.SpeechEventType.START_OF_SPEECH:
            print("Speech started")

        elif event.type == stt.SpeechEventType.END_OF_SPEECH:
            print("Speech ended")

    def _reset_turn_state(self):
        """Reset state for the next user turn"""
        self.first_interim_received = False
        self.current_turn_id += 1
        print(f"Turn {self.current_turn_id} complete - ready for next turn")

    async def _generate_and_play_filler(self):
        """Generate filler text using LLM and play it as audio"""
        try:
            start_time = time.time()
            filler_chat_ctx = self.session.history.copy()

            system_prompt = (
                "You're a filler-word assistant. As the user speaks, generate natural "
                "filler utterances (uh-huh, mm-hm, okay, I see, right, yeah) based on the conversation so far. "
                "Keep it very short (1-3 words max). Be natural and conversational."
            )

            if self.seen_fillers:
                recent_fillers = self.seen_fillers[-5:]
                system_prompt += f" Don't repeat these recent fillers: {', '.join(recent_fillers)}."

            filler_chat_ctx.add_message(role="system", content=system_prompt)
            print("Generating filler response...")

            filler_text = ""
            async for chunk in self.filler_llm.chat(chat_ctx=filler_chat_ctx).to_str_iterable():
                filler_text += chunk

            filler_text = filler_text.strip().strip('"').strip("'")
            generation_time = time.time() - start_time
            print(f"Generated filler: '{filler_text}' (took {generation_time:.2f}s)")

            self.seen_fillers.append(filler_text)
            if len(self.seen_fillers) > 20:
                self.seen_fillers.pop(0)

            await self.play_filler_audio(filler_text)

            total_time = time.time() - start_time
            print(f"Filler complete: '{filler_text}' (total: {total_time:.2f}s)")

        except Exception as e:
            print(f"Error generating filler: {e}")


async def entrypoint(ctx: JobContext):
    """Main entrypoint for the filler agent"""
    print("Starting STT Interim Filler Agent...")
    await ctx.connect()

    session = AgentSession(
        stt=google.STT(),
        tts=google.TTS(),
        llm=google.LLM(),
        vad=silero.VAD.load(),
    )

    await session.start(FillerAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
