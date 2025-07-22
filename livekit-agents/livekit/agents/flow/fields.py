"""
Field-level dataclass definitions for conversation flow schema.

This module contains the smaller dataclasses that represent individual
fields and components within larger flow structures like nodes and tools.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .enums import InstructionType, TransitionConditionType


@dataclass
class Instruction:
    """Represents an instruction for how a node should behave.

    Attributes:
        type: The type of instruction (static_text or prompt)
        text: The actual instruction content
    """

    type: InstructionType | str
    text: str

    def __post_init__(self):
        if isinstance(self.type, str):
            try:
                self.type = InstructionType(self.type)
            except ValueError:
                # Allow custom instruction types
                pass
        # Allow empty text - it might be valid in some cases

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Instruction:
        """Create an Instruction from a dictionary representation."""
        if not isinstance(d, dict):
            raise ValueError("Instruction must be a dictionary")

        instruction_type = d.get("type", "prompt")
        text = d.get("text", "")

        if not text:
            raise ValueError("Instruction text is required")

        return Instruction(type=instruction_type, text=text)


@dataclass
class TransitionCondition:
    """Represents a condition that determines when to transition between nodes.

    Attributes:
        type: The type of condition evaluation (prompt, regex, exact_match)
        prompt: The condition prompt or pattern
    """

    type: TransitionConditionType | str
    prompt: str

    def __post_init__(self):
        if isinstance(self.type, str):
            try:
                self.type = TransitionConditionType(self.type)
            except ValueError:
                # Allow custom transition condition types
                pass
        if not self.prompt.strip():
            raise ValueError("Transition condition prompt cannot be empty")

    @staticmethod
    def from_dict(d: dict[str, Any]) -> TransitionCondition:
        """Create a TransitionCondition from a dictionary representation."""
        if not isinstance(d, dict):
            raise ValueError("TransitionCondition must be a dictionary")

        condition_type = d.get("type", "prompt")
        prompt = d.get("prompt", "")

        if not prompt:
            raise ValueError("Transition condition prompt is required")

        return TransitionCondition(type=condition_type, prompt=prompt)


@dataclass
class Edge:
    """Represents a connection between nodes in the conversation flow.

    Attributes:
        id: Unique identifier for this edge
        condition: Human-readable description of the transition condition
        transition_condition: The actual condition logic
        destination_node_id: ID of the target node (None for terminal edges)
    """

    id: str
    condition: str
    transition_condition: TransitionCondition
    destination_node_id: str | None = None

    def __post_init__(self):
        if not self.id.strip():
            raise ValueError("Edge ID cannot be empty")
        if not self.condition.strip():
            raise ValueError("Edge condition cannot be empty")

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Edge:
        """Create an Edge from a dictionary representation."""
        if not isinstance(d, dict):
            raise ValueError("Edge must be a dictionary")

        edge_id = d.get("id", "")
        condition = d.get("condition", "")

        if not edge_id:
            raise ValueError("Edge ID is required")
        if not condition:
            raise ValueError("Edge condition is required")

        tc_data = d.get("transition_condition", {})
        if not tc_data:
            raise ValueError("Edge transition_condition is required")

        return Edge(
            id=edge_id,
            condition=condition,
            transition_condition=TransitionCondition.from_dict(tc_data),
            destination_node_id=d.get("destination_node_id"),
        )


@dataclass
class GlobalNodeSetting:
    """Represents a global node setting that can trigger from anywhere in the flow.

    Attributes:
        condition: The condition that triggers this global transition
    """

    condition: str

    def __post_init__(self):
        if not self.condition.strip():
            raise ValueError("Global node setting condition cannot be empty")

    @staticmethod
    def from_dict(d: dict[str, Any]) -> GlobalNodeSetting:
        """Create a GlobalNodeSetting from a dictionary representation."""
        if not isinstance(d, dict):
            raise ValueError("GlobalNodeSetting must be a dictionary")

        condition = d.get("condition", "")
        if not condition:
            raise ValueError("Global node setting condition is required")

        return GlobalNodeSetting(condition=condition)


@dataclass
class GatherInputVariable:
    """Represents a variable to be collected from user input with validation.

    Attributes:
        name: Variable name for storage and reference
        type: Data type (string, email, phone, date, number, etc.)
        description: Human-readable description for the variable
        required: Whether this variable is required
        max_attempts: Maximum number of collection attempts before failure
        regex_pattern: Optional regex pattern for validation
        regex_error_message: Error message when regex validation fails
    """

    name: str
    type: str
    description: str
    required: bool = False
    max_attempts: int = 3
    regex_pattern: str | None = None
    regex_error_message: str | None = None

    def __post_init__(self):
        if not self.name.strip():
            raise ValueError("Gather input variable name cannot be empty")
        if not self.description.strip():
            raise ValueError("Gather input variable description cannot be empty")
        if self.max_attempts < 1:
            raise ValueError("Max attempts must be at least 1")

    @staticmethod
    def from_dict(d: dict[str, Any]) -> GatherInputVariable:
        """Create a GatherInputVariable from a dictionary representation."""
        if not isinstance(d, dict):
            raise ValueError("GatherInputVariable must be a dictionary")

        name = d.get("name", "")
        if not name:
            raise ValueError("Gather input variable name is required")

        return GatherInputVariable(
            name=name,
            type=d.get("type", "string"),
            description=d.get("description", ""),
            required=d.get("required", False),
            max_attempts=d.get("max_attempts", 3),
            regex_pattern=d.get("regex_pattern"),
            regex_error_message=d.get("regex_error_message"),
        )


@dataclass
class TransferDestination:
    """Represents a call transfer destination.

    Attributes:
        type: Type of transfer destination
        number: Phone number or destination identifier
        sip_uri: SIP URI for SIP-based transfers
        description: Human-readable description of the destination
    """

    type: str
    number: str | None = None
    sip_uri: str | None = None
    description: str | None = None

    @staticmethod
    def from_dict(d: dict[str, Any]) -> TransferDestination:
        """Create a TransferDestination from a dictionary representation."""
        if not isinstance(d, dict):
            raise ValueError("TransferDestination must be a dictionary")
        return TransferDestination(
            type=d.get("type", ""),
            number=d.get("number"),
            sip_uri=d.get("sip_uri"),
            description=d.get("description"),
        )


@dataclass
class TransferOption:
    """Represents options for call transfer behavior.

    Attributes:
        type: Type of transfer option
        option: Additional configuration options
        show_transferee_as_caller: Whether to show the transferee as the caller
    """

    type: str
    option: dict[str, Any] | None = None
    show_transferee_as_caller: bool | None = None

    @staticmethod
    def from_dict(d: dict[str, Any]) -> TransferOption:
        """Create a TransferOption from a dictionary representation."""
        if not isinstance(d, dict):
            raise ValueError("TransferOption must be a dictionary")
        return TransferOption(
            type=d.get("type", ""),
            option=d.get("option"),
            show_transferee_as_caller=d.get("show_transferee_as_caller"),
        )


# Export all field types
__all__ = [
    "Instruction",
    "TransitionCondition",
    "Edge",
    "GlobalNodeSetting",
    "GatherInputVariable",
    "TransferDestination",
    "TransferOption",
]
