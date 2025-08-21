from __future__ import annotations

from typing import Literal

# Available KittenTTS models
KittenTTSModel = Literal["nano-0.1"]

# Default model shipped with this plugin
DEFAULT_MODEL: KittenTTSModel = "nano-0.1"

# Mapping of model identifiers to their Hugging Face revision.
# The "nano-0.1" model currently lives on the "main" branch.
MODEL_REVISIONS: dict[KittenTTSModel, str] = {
    "nano-0.1": "main",
}

# Hugging Face repository hosting the default KittenTTS ONNX assets
HG_MODEL = "KittenML/kitten-tts-nano-0.1"

# Filenames within the repository
ONNX_FILENAME = "kitten_tts_nano_v0_1.onnx"
VOICES_FILENAME = "voices.npz"
