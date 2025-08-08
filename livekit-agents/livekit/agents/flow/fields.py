from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class Instruction:
    type: Literal["static_text", "prompt"]
    text: str

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Instruction:
        if not isinstance(d, dict):
            raise ValueError("Instruction must be a dictionary")

        instruction_type = d.get("type", "prompt")
        text = d.get("text", "")

        if not text:
            raise ValueError("Instruction text is required")

        return Instruction(type=instruction_type, text=text)


@dataclass
class Equation:
    left_operand: str
    operator: str
    right_operand: str

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Equation:
        if not isinstance(d, dict):
            raise ValueError("Equation must be a dictionary")

        return Equation(
            left_operand=d.get("left_operand", ""),
            operator=d.get("operator", ""),
            right_operand=d.get("right_operand", ""),
        )


@dataclass
class TransitionCondition:
    type: Literal["prompt", "equation"]
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
        if not isinstance(d, dict):
            raise ValueError("TransitionCondition must be a dictionary")

        condition_type = d.get("type", "")
        prompt = d.get("prompt", None)  # Use None as default, not empty string
        equations = d.get("equations", None)
        operator = d.get("operator", None)

        if not prompt and not equations:
            raise ValueError("Transition condition prompt or equation is required")

        if equations:
            equations = [Equation.from_dict(e) for e in equations]

        return TransitionCondition(
            type=condition_type,
            prompt=prompt,
            equations=equations,
            operator=operator,
        )


@dataclass
class Edge:
    id: str
    condition: str
    transition_condition: TransitionCondition
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
        if not isinstance(d, dict):
            raise ValueError("Edge must be a dictionary")

        edge_id = d.get("id", "")
        condition = d.get("condition", "")

        if not edge_id:
            raise ValueError("Edge ID is required")
        if not condition:
            raise ValueError("Edge condition is required")

        tc_data = d.get("transition_condition", None)
        if not tc_data:
            raise ValueError("Edge transition_condition is required")

        if not isinstance(tc_data, dict):
            raise ValueError("Edge transition_condition must be a dictionary")

        return Edge(
            id=edge_id,
            condition=condition,
            transition_condition=TransitionCondition.from_dict(tc_data),
            destination_node_id=d.get("destination_node_id"),
        )


@dataclass
class GlobalNodeSetting:
    condition: str

    def __post_init__(self) -> None:
        if not self.condition.strip():
            raise ValueError("Global node setting condition cannot be empty")

    @staticmethod
    def from_dict(d: dict[str, Any]) -> GlobalNodeSetting:
        if not isinstance(d, dict):
            raise ValueError("GlobalNodeSetting must be a dictionary")

        condition = d.get("condition", "")
        if not condition:
            raise ValueError("Global node setting condition is required")

        return GlobalNodeSetting(condition=condition)


@dataclass
class GatherInputVariable:
    name: str
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
    ]
    description: str
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
    """Transfer destination configuration."""

    number: str
    sip_uri: str | None = None

    @staticmethod
    def from_dict(d: dict[str, Any]) -> TransferDestination:
        if not isinstance(d, dict):
            raise ValueError("TransferDestination must be a dictionary")

        return TransferDestination(number=d.get("number", ""), sip_uri=d.get("sip_uri"))


@dataclass
class TransferOption:
    """Transfer option configuration."""

    timeout: int = 30

    @staticmethod
    def from_dict(d: dict[str, Any]) -> TransferOption:
        if not isinstance(d, dict):
            raise ValueError("TransferOption must be a dictionary")

        return TransferOption(timeout=d.get("timeout", 30))


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
