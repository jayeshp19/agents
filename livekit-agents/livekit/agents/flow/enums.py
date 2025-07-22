"""
Enumeration definitions for conversation flow schema.

This module contains all enum types used throughout the flow system,
providing type safety and preventing typos in string constants.
"""

from enum import Enum


class NodeType(str, Enum):
    """Supported node types in conversation flows.

    - CONVERSATION: Standard conversational interaction nodes
    - FUNCTION: Tool/function execution nodes
    - END: Flow termination nodes
    - GATHER_INPUT: Input collection nodes with validation
    """

    CONVERSATION = "conversation"
    FUNCTION = "function"
    END = "end"
    GATHER_INPUT = "gather_input"

    def __str__(self):
        return self.value


class ToolType(str, Enum):
    """Supported tool types for function nodes.

    - CUSTOM: Custom HTTP API tools
    - LOCAL: Locally registered Python functions
    - BUILT_IN: System built-in functions
    - CHECK_AVAILABILITY_CAL: Calendar availability checking
    - BOOK_APPOINTMENT_CAL: Calendar appointment booking
    """

    CUSTOM = "custom"
    LOCAL = "local"
    BUILT_IN = "built-in"
    CHECK_AVAILABILITY_CAL = "check_availability_cal"
    BOOK_APPOINTMENT_CAL = "book_appointment_cal"

    def __str__(self):
        return self.value


class InstructionType(str, Enum):
    """Supported instruction types for nodes.

    - STATIC_TEXT: Fixed text responses
    - PROMPT: Dynamic LLM-generated responses
    """

    STATIC_TEXT = "static_text"
    PROMPT = "prompt"

    def __str__(self):
        return self.value


class TransitionConditionType(str, Enum):
    """Supported transition condition evaluation types.

    - PROMPT: LLM-based condition evaluation
    """

    PROMPT = "prompt"

    def __str__(self):
        return self.value


class SpeakerType(str, Enum):
    """Supported speaker types in conversations.

    - AGENT: AI agent responses
    - USER: Human user inputs
    """

    AGENT = "agent"
    USER = "user"

    def __str__(self):
        return self.value


# Convenience sets for validation
ALL_NODE_TYPES = {e.value for e in NodeType}
ALL_TOOL_TYPES = {e.value for e in ToolType}
ALL_INSTRUCTION_TYPES = {e.value for e in InstructionType}
ALL_TRANSITION_CONDITION_TYPES = {e.value for e in TransitionConditionType}
ALL_SPEAKER_TYPES = {e.value for e in SpeakerType}

# Export all enum types
__all__ = [
    "NodeType",
    "ToolType",
    "InstructionType",
    "TransitionConditionType",
    "SpeakerType",
    "ALL_NODE_TYPES",
    "ALL_TOOL_TYPES",
    "ALL_INSTRUCTION_TYPES",
    "ALL_TRANSITION_CONDITION_TYPES",
    "ALL_SPEAKER_TYPES",
]
