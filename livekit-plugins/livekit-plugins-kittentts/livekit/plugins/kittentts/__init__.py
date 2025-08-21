"""KittenTTS plugin for LiveKit Agents"""

from .tts import TTS, ChunkedStream
from .version import __version__

__all__ = ["TTS", "ChunkedStream", "KittenTTSPlugin", "__version__"]

from livekit.agents import Plugin
import json
import os

from typing import Any


class KittenTTSPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__)

    def download_files(self) -> None:
        """Pre-download KittenTTS model assets from Hugging Face.

        Env:
        - `KITTENTTS_REPO_ID` (default: `KittenML/kitten-tts-nano-0.1`)
        - `KITTENTTS_REVISION` (optional)
        """
        from huggingface_hub import hf_hub_download  # type: ignore

        repo_id = os.getenv("KITTENTTS_REPO_ID", "KittenML/kitten-tts-nano-0.1")
        revision = os.getenv("KITTENTTS_REVISION")
        # fetch config.json to learn filenames
        config_path = hf_hub_download(
            repo_id=repo_id, filename="config.json", revision=revision
        )
        with open(config_path, "r") as f:
            cfg: dict[str, Any] = json.load(f)

        if cfg.get("type") != "ONNX1":
            # Only ONNX1 models supported by this plugin currently
            return

        model_file = cfg.get("model_file")
        voices_file = cfg.get("voices")
        if model_file:
            hf_hub_download(repo_id=repo_id, filename=str(model_file), revision=revision)
        if voices_file:
            hf_hub_download(repo_id=repo_id, filename=str(voices_file), revision=revision)


Plugin.register_plugin(KittenTTSPlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
