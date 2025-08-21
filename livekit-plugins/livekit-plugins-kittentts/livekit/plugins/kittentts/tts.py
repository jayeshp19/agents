from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, replace
from typing import Final

import numpy as np

from kittentts.onnx_model import KittenTTS_1_Onnx as _KittenOnnx
from livekit.agents import tts, utils
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

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
        model_name: str = "KittenML/kitten-tts-nano-0.1",
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
            model_name=model_name,
            voice=voice,
            speed=speed,
            emit_chunks=emit_chunks,
            frame_size_ms=frame_size_ms,
        )

        self._model: _KittenOnnx | None = None
        self._model_lock = threading.Lock()

    def _ensure_model(self) -> _KittenOnnx:
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    import json
                    import os

                    from huggingface_hub import errors, hf_hub_download  # type: ignore

                    repo_id = self._opts.model_name
                    revision = os.getenv("KITTENTTS_REVISION")
                    try:
                        cfg_path = hf_hub_download(
                            repo_id=repo_id,
                            filename="config.json",
                            revision=revision,
                            local_files_only=True,
                        )
                        with open(cfg_path) as f:
                            cfg = json.load(f)

                        if cfg.get("type") != "ONNX1":
                            raise RuntimeError(
                                "Unsupported KittenTTS model type. Only ONNX1 is supported."
                            )

                        model_file = str(cfg.get("model_file"))
                        voices_file = str(cfg.get("voices"))

                        model_path = hf_hub_download(
                            repo_id=repo_id,
                            filename=model_file,
                            revision=revision,
                            local_files_only=True,
                        )
                        voices_path = hf_hub_download(
                            repo_id=repo_id,
                            filename=voices_file,
                            revision=revision,
                            local_files_only=True,
                        )
                    except (errors.LocalEntryNotFoundError, OSError):
                        raise RuntimeError(
                            "KittenTTS assets not found locally. Pre-download them first via "
                            "`python myagent.py download-files` (ensure `from livekit.plugins import kittentts` is imported)."
                        ) from None

                    self._model = _KittenOnnx(model_path=model_path, voices_path=voices_path)

        return self._model

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
            self._opts.model_name = model_name
            self._model = None
        if is_given(voice):
            self._opts.voice = voice
        if is_given(speed):
            self._opts.speed = speed
        if is_given(emit_chunks):
            self._opts.emit_chunks = emit_chunks
        if is_given(frame_size_ms):
            self._opts.frame_size_ms = int(frame_size_ms)

    def synthesize(self, text: str, *, conn_options=DEFAULT_API_CONNECT_OPTIONS) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class ChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: TTS, input_text: str, conn_options) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts = tts
        self._opts = replace(tts._opts)

    @staticmethod
    def _to_pcm16_bytes(audio: np.ndarray) -> bytes:
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        audio = np.clip(audio, -1.0, 1.0)
        pcm16 = (audio * 32767.0).astype(np.int16)
        return pcm16.tobytes()

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
            frame_size_ms=self._opts.frame_size_ms,
        )

        # helper to run blocking synthesis in a thread
        def _synthesize(text: str) -> bytes:
            model = self._tts._ensure_model()
            audio = model.generate(text, voice=self._opts.voice, speed=self._opts.speed)
            return self._to_pcm16_bytes(audio)

        if not self._opts.emit_chunks:
            data: bytes = await asyncio.to_thread(_synthesize, self._input_text)
            output_emitter.push(data)
            output_emitter.flush()
        else:
            import re

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
                data: bytes = await asyncio.to_thread(_synthesize, chunk)
                output_emitter.push(data)
                output_emitter.flush()
