"""KittenTTS plugin for LiveKit Agents"""

from .tts import TTS, ChunkedStream
from .version import __version__

__all__ = ["TTS", "ChunkedStream", "KittenTTSPlugin", "__version__"]

import os

from livekit.agents import Plugin

from .model import (
    DEFAULT_MODEL,
    HG_MODEL,
    MODEL_REVISIONS,
    ONNX_FILENAME,
    VOICES_FILENAME,
)


class KittenTTSPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__)

    def download_files(self) -> None:
        """Pre-download KittenTTS model assets from Hugging Face.

        Env:
        - `KITTENTTS_REPO_ID` (default: ``HG_MODEL``)
        - `KITTENTTS_REVISION` (optional)
        """
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer  # type: ignore[import-untyped]

        repo_id = os.getenv("KITTENTTS_REPO_ID", HG_MODEL)
        revision = os.getenv(
            "KITTENTTS_REVISION", MODEL_REVISIONS[DEFAULT_MODEL]
        )

        AutoTokenizer.from_pretrained(repo_id, revision=revision)
        hf_hub_download(repo_id=repo_id, filename=ONNX_FILENAME, revision=revision)
        hf_hub_download(repo_id=repo_id, filename=VOICES_FILENAME, revision=revision)


Plugin.register_plugin(KittenTTSPlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
