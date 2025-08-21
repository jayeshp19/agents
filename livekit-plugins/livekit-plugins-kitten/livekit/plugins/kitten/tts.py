from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from typing import Final

from livekit.agents import APIConnectOptions, tts, utils
from livekit.agents.job import get_job_context
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from .model import HG_MODEL, resolve_model_name
from .runner import _KittenRunner

SAMPLE_RATE: Final[int] = 24000
NUM_CHANNELS: Final[int] = 1


@dataclass
class _TTSOptions:
    model_name: str
    voice: str
    speed: float
    emit_chunks: bool
    frame_size_ms: int


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        model_name: str = HG_MODEL,
        voice: str = "expr-voice-5-m",
        speed: float = 1.0,
        emit_chunks: bool = False,
        frame_size_ms: int = 200,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )

        self._opts = _TTSOptions(
            model_name=resolve_model_name(model_name),
            voice=voice,
            speed=speed,
            emit_chunks=emit_chunks,
            frame_size_ms=frame_size_ms,
        )

    def update_options(
        self,
        *,
        model_name: NotGivenOr[str] = NOT_GIVEN,
        voice: NotGivenOr[str] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        emit_chunks: NotGivenOr[bool] = NOT_GIVEN,
        frame_size_ms: NotGivenOr[int] = NOT_GIVEN,
    ) -> None:
        if is_given(model_name):
            self._opts.model_name = resolve_model_name(model_name)
        if is_given(voice):
            self._opts.voice = voice
        if is_given(speed):
            self._opts.speed = speed
        if is_given(emit_chunks):
            self._opts.emit_chunks = emit_chunks
        if is_given(frame_size_ms):
            self._opts.frame_size_ms = int(frame_size_ms)

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class ChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
            frame_size_ms=self._opts.frame_size_ms,
        )

        executor = get_job_context().inference_executor

        async def _synthesize(text: str) -> bytes:
            payload = {
                "text": text,
                "voice": self._opts.voice,
                "speed": self._opts.speed,
            }
            data = json.dumps(payload).encode()
            result = await executor.do_inference(_KittenRunner.INFERENCE_METHOD, data)
            assert result is not None
            return result

        if not self._opts.emit_chunks:
            data = await _synthesize(self._input_text)
            output_emitter.push(data)
            output_emitter.flush()
        else:
            parts = [
                p.strip()
                for p in re.split(r"([.!?]+)\s+", self._input_text)
                if p and not p.isspace()
            ]
            chunks: list[str] = []
            buf = ""
            for p in parts:
                if re.fullmatch(r"[.!?]+", p):
                    buf += p
                    chunks.append(buf)
                    buf = ""
                else:
                    if buf:
                        chunks.append(buf)
                        buf = ""
                    buf = p
            if buf:
                chunks.append(buf)

            if not chunks:
                chunks = [self._input_text]

            for chunk in chunks:
                data = await _synthesize(chunk)
                output_emitter.push(data)
                output_emitter.flush()
