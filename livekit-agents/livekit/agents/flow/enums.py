from enum import Enum


class NodeType(str, Enum):
    CONVERSATION = "conversation"
    FUNCTION = "function"
    END = "end"
    GATHER_INPUT = "gather_input"

    def __str__(self) -> str:
        return self.value


class ToolType(str, Enum):
    CUSTOM = "custom"
    LOCAL = "local"

    def __str__(self) -> str:
        return self.value


class InstructionType(str, Enum):
    STATIC_TEXT = "static_text"
    PROMPT = "prompt"

    def __str__(self) -> str:
        return self.value


class SpeakerType(str, Enum):
    AGENT = "agent"
    USER = "user"

    def __str__(self) -> str:
        return self.value


ALL_NODE_TYPES = {e.value for e in NodeType}
ALL_TOOL_TYPES = {e.value for e in ToolType}
ALL_INSTRUCTION_TYPES = {e.value for e in InstructionType}
ALL_SPEAKER_TYPES = {e.value for e in SpeakerType}

__all__ = [
    "NodeType",
    "ToolType",
    "InstructionType",
    "SpeakerType",
    "ALL_NODE_TYPES",
    "ALL_TOOL_TYPES",
    "ALL_INSTRUCTION_TYPES",
    "ALL_SPEAKER_TYPES",
]
