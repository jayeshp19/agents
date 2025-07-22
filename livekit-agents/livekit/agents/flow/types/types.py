import time
from dataclasses import dataclass, field
from typing import Any, Optional

from livekit.agents import llm


@dataclass
class EdgeEvaluation:
    edge_id: Optional[str]
    destination_node_id: Optional[str]
    confidence: float
    reasoning: str
    fallback_used: bool = False


@dataclass
class TransitionResult:
    destination_node_id: Optional[str]
    edge_id: Optional[str]
    user_text: Optional[str]
    confidence: float
    reasoning: str

    @staticmethod
    def from_evaluation(
        evaluation: EdgeEvaluation, user_text: Optional[str] = None
    ) -> "TransitionResult":
        return TransitionResult(
            destination_node_id=evaluation.destination_node_id,
            edge_id=evaluation.edge_id,
            user_text=user_text,
            confidence=evaluation.confidence,
            reasoning=evaluation.reasoning,
        )


@dataclass
class FlowError:
    error_type: str
    message: str
    node_id: str
    timestamp: float
    retry_count: int = 0
    is_recoverable: bool = True

    @classmethod
    def from_exception(
        cls, error: Exception, node_id: str, error_type: str = "system_error"
    ) -> "FlowError":
        return cls(
            error_type=error_type,
            message=str(error),
            node_id=node_id,
            timestamp=time.time(),
            is_recoverable=error_type in ["llm_failure", "validation_error"],
        )


@dataclass
class FlowContext:
    conversation_history: list[llm.ChatMessage] = field(default_factory=list)
    variables: dict[str, Any] = field(default_factory=dict)
    execution_path: list[str] = field(default_factory=list)
    function_results: dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    checkpoints: list[dict[str, Any]] = field(default_factory=list)
    errors: list[FlowError] = field(default_factory=list)  # Track errors for analysis

    max_history_length: int = 50

    def add_message(self, message: llm.ChatMessage) -> None:
        self.conversation_history.append(message)
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = (
                self.conversation_history[:1]
                + self.conversation_history[-(self.max_history_length - 1) :]
            )

    def set_variable(self, key: str, value: Any) -> None:
        self.variables[key] = value

    def get_variable(self, key: str, default: Any = None) -> Any:
        return self.variables.get(key, default)

    def save_checkpoint(self, node_id: str) -> None:
        checkpoint = {
            "node_id": node_id,
            "timestamp": time.time(),
            "variables": self.variables.copy(),
            "execution_path": self.execution_path.copy(),
        }
        self.checkpoints.append(checkpoint)
        if len(self.checkpoints) > 10:
            self.checkpoints.pop(0)

    def add_error(self, error: FlowError) -> None:
        self.errors.append(error)
        if len(self.errors) > 20:
            self.errors.pop(0)


@dataclass
class TransitionDecision:
    selected_option: Optional[int]
    confidence: float
    reasoning: str
    user_intent: str

    @classmethod
    def from_dict(cls, data: dict) -> "TransitionDecision":
        return cls(
            selected_option=data.get("selected_option"),
            confidence=data.get("confidence", 0.0),
            reasoning=data.get("reasoning", ""),
            user_intent=data.get("user_intent", ""),
        )
