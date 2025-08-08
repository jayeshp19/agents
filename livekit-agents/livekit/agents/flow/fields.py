from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Literal

from .utils import parse_dataclass


@dataclass
class Instruction:
    type: Literal["static_text", "prompt"] = "prompt"
    text: str = ""

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("Instruction text cannot be empty")

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Instruction:
        return parse_dataclass(Instruction, d)


@dataclass
class Equation:
    left_operand: str = ""
    operator: str = ""
    right_operand: str = ""

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Equation:
        return parse_dataclass(Equation, d)


@dataclass
class TransitionCondition:
    type: Literal["prompt", "equation"] = "prompt"
    prompt: str | None = None
    equations: list[Equation] | None = None
    operator: str | None = None

    def __post_init__(self) -> None:
        if self.prompt is None and not self.equations:
            raise ValueError("Transition condition must have either prompt or equations")
        if self.prompt is not None and not self.prompt.strip():
            raise ValueError("Transition condition prompt cannot be empty")

    @staticmethod
    def from_dict(d: dict[str, Any]) -> TransitionCondition:
        return parse_dataclass(TransitionCondition, d)


@dataclass
class Edge:
    id: str = ""
    condition: str = ""
    transition_condition: TransitionCondition | None = None
    destination_node_id: str | None = None

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("Edge ID cannot be empty")
        if not self.condition.strip():
            raise ValueError("Edge condition cannot be empty")
        if not self.transition_condition:
            raise ValueError("Edge transition_condition cannot be empty")

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Edge:
        return parse_dataclass(Edge, d)


@dataclass
class GlobalNodeSetting:
    condition: str = ""

    def __post_init__(self) -> None:
        if not self.condition.strip():
            raise ValueError("Global node setting condition cannot be empty")

    @staticmethod
    def from_dict(d: dict[str, Any]) -> GlobalNodeSetting:
        return parse_dataclass(GlobalNodeSetting, d)


@dataclass
class GatherInputVariable:
    name: str = ""
    type: Literal[
        "string",
        "email",
        "phone",
        "number",
        "integer",
        "float",
        "date",
        "url",
        "boolean",
        "custom",
    ] = "string"
    description: str = ""
    required: bool = False
    max_attempts: int = 3
    regex_pattern: str | None = None
    regex_error_message: str | None = None

    _DEFAULT_PATTERNS = {
        "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
        "phone": r"^(\+?1[-\s]?)?\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{4}$",
        "number": r"^-?\d*\.?\d+$",
        "integer": r"^-?\d+$",
        "float": r"^-?\d*\.\d+$",
        "date": r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$|^\d{4}[/-]\d{1,2}[/-]\d{1,2}$",
        "url": r"^https?://[^\s/$.?#].[^\s]*$",
        "boolean": r"^(true|false|yes|no|1|0)$",
    }

    _DEFAULT_ERROR_MESSAGES = {
        "email": "Please provide a valid email address",
        "phone": "Please provide a valid phone number",
        "number": "Please provide a valid number",
        "integer": "Please provide a valid integer",
        "float": "Please provide a valid decimal number",
        "date": "Please provide a valid date (MM/DD/YYYY or YYYY-MM-DD)",
        "url": "Please provide a valid URL",
        "boolean": "Please provide yes/no, true/false, or 1/0",
    }

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Gather input variable name cannot be empty")
        if not self.description.strip():
            raise ValueError("Gather input variable description cannot be empty")
        if self.max_attempts < 1:
            raise ValueError("Max attempts must be at least 1")

        # Validate custom regex patterns to prevent ReDoS attacks
        if self.regex_pattern:
            self._validate_regex_safety(self.regex_pattern)

    def get_validation_pattern(self) -> str | None:
        if self.regex_pattern:
            return self.regex_pattern
        return self._DEFAULT_PATTERNS.get(self.type)

    def get_validation_error_message(self) -> str:
        if self.regex_error_message:
            return self.regex_error_message
        return self._DEFAULT_ERROR_MESSAGES.get(self.type, "Invalid input format")

    def _validate_regex_safety(self, pattern: str) -> None:
        """Validate regex pattern for safety against ReDoS attacks."""
        # Check for common ReDoS patterns
        dangerous_patterns = [
            r"\(\?\=.*\)\+",  # Positive lookahead with repetition
            r"\(\?\!.*\)\+",  # Negative lookahead with repetition
            r"\(\.\*\)\*",  # Nested quantifiers like (.*)*
            r"\(\.\+\)\+",  # Nested quantifiers like (.+)+
            r"\(\.\*\)\{\d*,\d*\}",  # Grouped .* with quantifiers
            r"\(\.\+\)\{\d*,\d*\}",  # Grouped .+ with quantifiers
            r"\(\*.*\*\)",  # Grouped multiple asterisks
            r"\(\+.*\+\)",  # Grouped multiple plus signs
            r"\([^)]*\*[^)]*\)\+",  # Group with * followed by +
            r"\([^)]*\+[^)]*\)\*",  # Group with + followed by *
            r"\(\.\*\?\)\+",  # Non-greedy .* with repetition
        ]

        for dangerous in dangerous_patterns:
            if re.search(dangerous, pattern):
                raise ValueError(f"Potentially dangerous regex pattern detected: {pattern}")

        # Test regex compilation and basic performance
        try:
            compiled = re.compile(pattern)
            # Test with a known problematic string
            test_string = "a" * 100
            start_time = time.time()
            compiled.match(test_string)
            elapsed = time.time() - start_time

            # If regex takes too long on simple input, it's likely dangerous
            if elapsed > 0.1:  # 100ms threshold
                raise ValueError(f"Regex pattern appears to have performance issues: {pattern}")

        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}") from e

    def validate_value(self, value: Any) -> tuple[bool, str]:
        if value is None or (isinstance(value, str) and not value.strip()):
            if self.required:
                return False, f"{self.name} is required"
            return True, ""

        str_value = str(value).strip()

        pattern = self.get_validation_pattern()
        if pattern:
            try:
                # Apply timeout to regex matching to prevent ReDoS
                start_time = time.time()
                match = re.match(pattern, str_value, re.IGNORECASE)
                elapsed = time.time() - start_time

                # If matching takes too long, abort
                if elapsed > 1.0:  # 1 second timeout
                    return False, "Validation timeout - pattern too complex"

                if not match:
                    return False, self.get_validation_error_message()
            except re.error as e:
                return False, f"Invalid regex pattern: {e}"

        return True, ""

    @staticmethod
    def from_dict(d: dict[str, Any]) -> GatherInputVariable:
        return parse_dataclass(GatherInputVariable, d)


@dataclass
class TransferDestination:
    """Transfer destination configuration."""

    number: str = ""
    sip_uri: str | None = None

    def __post_init__(self) -> None:
        if not self.number.strip():
            raise ValueError("Transfer destination number cannot be empty")

    @staticmethod
    def from_dict(d: dict[str, Any]) -> TransferDestination:
        return parse_dataclass(TransferDestination, d)


@dataclass
class TransferOption:
    """Transfer option configuration."""

    timeout: int = 30

    @staticmethod
    def from_dict(d: dict[str, Any]) -> TransferOption:
        return parse_dataclass(TransferOption, d)


__all__ = [
    "Instruction",
    "Equation",
    "TransitionCondition",
    "Edge",
    "GlobalNodeSetting",
    "GatherInputVariable",
    "TransferDestination",
    "TransferOption",
]
