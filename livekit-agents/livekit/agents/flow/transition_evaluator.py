import json
import logging
import time
from typing import Any, Optional, Protocol

from jsonpath_ng import parse as jsonpath_parse

from livekit.agents import llm

from .base import FlowTransition
from .fields import Edge, Equation, TransitionCondition
from .utils.utils import clean_json_response, stream_chat_to_text

logger = logging.getLogger(__name__)


class VariableProvider(Protocol):
    def get_variable(self, name: str) -> Optional[Any]: ...


class TransitionEvaluator:
    def __init__(
        self,
        variable_provider: VariableProvider,
        edge_llm: Optional[llm.LLM[Any]] = None,
        *,
        min_prompt_confidence: float = 0.0,
        min_prompt_eval_interval_ms: int = 0,
    ):
        self.variable_provider = variable_provider
        self.edge_llm = edge_llm
        # Minimum confidence required for prompt-based transitions (0..1)
        self.min_prompt_confidence = max(0.0, min(1.0, min_prompt_confidence))
        # Minimum interval between prompt evaluations
        self._min_prompt_eval_interval = max(0, int(min_prompt_eval_interval_ms)) / 1000.0
        self._last_prompt_eval_ts: float = 0.0

    async def evaluate_transitions(
        self,
        edges: list[Edge],
        user_text: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> Optional[FlowTransition]:
        if context is None:
            context = {}

        for edge in edges:
            if (
                edge.transition_condition
                and edge.transition_condition.type == "equation"
                and edge.destination_node_id
            ):
                if self.evaluate_equation_condition(edge.transition_condition, context):
                    logger.info(
                        f"Equation transition matched: {edge.condition} -> {edge.destination_node_id}"
                    )
                    assert edge.destination_node_id is not None
                    return FlowTransition(
                        destination_node_id=edge.destination_node_id,
                        edge_id=edge.id,
                        confidence=1.0,
                        reasoning="Equation condition matched",
                        user_text=user_text,
                    )

        if user_text and self.edge_llm is not None:
            prompt_edges = [
                edge
                for edge in edges
                if edge.transition_condition
                and edge.transition_condition.type == "prompt"
                and edge.destination_node_id
            ]

            if prompt_edges:
                transition = await self._evaluate_prompt_transitions(prompt_edges, user_text)
                if transition:
                    return transition
        elif user_text:
            prompt_edges = [
                edge
                for edge in edges
                if edge.transition_condition and edge.transition_condition.type == "prompt"
            ]
            if prompt_edges:
                edge_ids = ", ".join(e.id for e in prompt_edges)
                logger.warning(
                    "Prompt transitions present (%s) but no edge_evaluator_llm configured; "
                    "these edges cannot be evaluated via prompt conditions.",
                    edge_ids,
                )

        return None

    async def _evaluate_prompt_transitions(
        self, edges: list[Edge], user_text: str
    ) -> Optional[FlowTransition]:
        assert self.edge_llm is not None
        now = time.time()
        if (
            self._min_prompt_eval_interval > 0
            and (now - self._last_prompt_eval_ts) < self._min_prompt_eval_interval
        ):
            return None
        all_options = []
        option_map = {}

        for i, edge in enumerate(edges, 1):
            all_options.append(
                {
                    "option": i,
                    "prompt": edge.transition_condition.prompt,
                }
            )
            option_map[i] = edge

        prompt = f'''User said: "{user_text}"

Available transitions:
{json.dumps(all_options, indent=2)}

Analyze the user's input and determine which transition condition best matches their intent.
Return the option number if there's a clear match, or 0 if none match well. Also return a confidence score (0..1).

Respond with JSON only:
{{"option": <number>, "confidence": <float between 0 and 1>}}'''

        try:
            response_text = await stream_chat_to_text(self.edge_llm, prompt)

            response_text = clean_json_response(response_text)
            result = json.loads(response_text)
            selected = result.get("option", 0)
            confidence = float(result.get("confidence", 0.9))
            self._last_prompt_eval_ts = time.time()
            if selected and selected in option_map and confidence >= self.min_prompt_confidence:
                edge = option_map[selected]
                logger.info(
                    f"Prompt transition selected: {edge.transition_condition.prompt} (option {selected})"
                )

                assert edge.destination_node_id is not None
                return FlowTransition(
                    destination_node_id=edge.destination_node_id,
                    edge_id=edge.id,
                    confidence=confidence,
                    reasoning=f"User input matched: {edge.transition_condition.prompt}",
                    user_text=user_text,
                )

        except Exception as e:
            logger.error(f"Error evaluating prompt transitions: {e}")

        return None

    def evaluate_equation_condition(
        self, condition: TransitionCondition, context: dict[str, Any]
    ) -> bool:
        if condition.type != "equation" or not condition.equations:
            return False

        eval_context = self._build_evaluation_context(context)

        op_group = (condition.operator or "AND").upper()

        if op_group in {"OR", "||"}:
            return any(
                self._evaluate_single_equation(eq, eval_context) for eq in condition.equations
            )
        else:
            return all(
                self._evaluate_single_equation(eq, eval_context) for eq in condition.equations
            )

    def _evaluate_single_equation(self, equation: Equation, context: dict[str, Any]) -> bool:
        try:
            left_value = self._resolve_value(equation.left_operand, context)
            right_value = self._resolve_value(equation.right_operand, context)
            operator = equation.operator

            if left_value is None:
                return operator == "==" and right_value in ["null", "None", None]

            return self._compare_values(left_value, right_value, operator)

        except Exception as e:
            logger.warning(
                f"Error evaluating equation {equation.left_operand} {equation.operator} "
                f"{equation.right_operand}: {e}",
                exc_info=True,
            )
            return False

    def _resolve_value(self, operand: str, context: dict[str, Any]) -> Any:
        if not isinstance(operand, str):
            return operand

        looks_like_jsonpath = any(ch in operand for ch in ("$", "@", ".", "[", "]", "*"))

        if len(operand) > 256:
            looks_like_jsonpath = False

        if looks_like_jsonpath:
            try:
                matches = jsonpath_parse(operand).find(context)
                if matches:
                    return matches[0].value
            except Exception:
                pass

        if operand in context:
            return context[operand]

        val = self.variable_provider.get_variable(operand)
        if val is not None:
            return val

        return operand

    def _coerce_value(self, value: str) -> Any:
        """Conservatively coerce simple scalar types; otherwise return string.

        Only supports: int, float, booleans (true/false/yes/no/1/0), and null/none.
        Lists/dicts are returned as-is (string) to avoid unintended evaluations.
        """
        if value is None:
            return None
        if isinstance(value, (int, float, bool)):
            return value
        if not isinstance(value, str):
            return value

        lower = value.strip().lower()
        if lower in {"none", "null"}:
            return None
        if lower in {"true", "yes", "1"}:
            return True
        if lower in {"false", "no", "0"}:
            return False
        # numeric
        try:
            if "." in lower:
                return float(lower)
            return int(lower)
        except ValueError:
            return value

    def _to_bool(self, value: Any) -> bool:
        """
        Convert a value to boolean consistently.
        """
        BOOL_TRUE = {"true", "1", "yes"}

        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.lower() in BOOL_TRUE
        return bool(value)

    def _compare_values(self, left: Any, right: str, operator: str) -> bool:
        """
        Compare two values based on the operator.
        Handles type conversion and various comparison operators.
        """
        # Coerce right value to proper type
        right_value = self._coerce_value(right)

        # Boolean comparison - only if explicitly boolean or boolean strings
        if isinstance(left, bool) or isinstance(right_value, bool):
            # One side is explicitly boolean
            left_bool = self._to_bool(left)
            right_bool = self._to_bool(right_value)

            if operator == "==":
                return left_bool == right_bool
            elif operator == "!=":
                return left_bool != right_bool

        # Check for boolean string values (but not numeric strings like "25")
        if isinstance(left, str) and isinstance(right_value, str):
            # Both are strings - check if they're boolean-like
            left_lower = left.lower()
            right_lower = right.lower() if isinstance(right, str) else str(right_value).lower()

            bool_strings = {"true", "false", "yes", "no"}
            if left_lower in bool_strings or right_lower in bool_strings:
                left_bool = self._to_bool(left)
                right_bool = self._to_bool(right_value)

                if operator == "==":
                    return left_bool == right_bool
                elif operator == "!=":
                    return left_bool != right_bool

        # String operations
        if operator in ["contains", "startswith", "endswith"]:
            # Handle list contains for complex types
            if operator == "contains" and isinstance(left, (list, tuple)):
                # For list of dicts/complex objects
                if any(isinstance(item, dict) for item in left):
                    return any(right.lower() in str(item).lower() for item in left)
                # For simple lists
                return right_value in left or right in [str(item) for item in left]

            # String operations
            left_str = str(left)
            if operator == "contains":
                return right.lower() in left_str.lower()
            elif operator == "startswith":
                return left_str.lower().startswith(right.lower())
            elif operator == "endswith":
                return left_str.lower().endswith(right.lower())

        # Case-insensitive equality/inequality for strings
        if operator in ["==~", "!=~"] and isinstance(left, str):
            left_lower = left.lower()
            right_lower = (
                str(right_value).lower()
                if not isinstance(right_value, str)
                else right_value.lower()
            )
            if operator == "==~":
                return left_lower == right_lower
            else:
                return left_lower != right_lower

        # Regex matches
        if operator in ["matches", "matches_i"] and isinstance(left, str):
            try:
                flags = 0
                pattern = str(right_value)
                if operator == "matches_i":
                    import re

                    flags = re.IGNORECASE
                import re

                return re.search(pattern, left, flags) is not None
            except Exception:
                return False

        # Membership tests
        if operator == "in":
            # List/tuple/set membership
            if isinstance(right_value, (list, tuple, set)):
                return left in right_value or str(left) in {str(v) for v in right_value}
            # Comma-separated string list
            if isinstance(right_value, str) and "," in right_value:
                tokens = [t.strip() for t in right_value.split(",") if t.strip()]
                return str(left) in tokens
            # Fallback: substring containment if both strings
            if isinstance(left, str) and isinstance(right_value, str):
                return left in right_value
            return False

        # Numeric comparisons
        if operator in [">", "<", ">=", "<=", "==", "!="]:
            # Try numeric comparison first
            if isinstance(left, (int, float)) and isinstance(right_value, (int, float)):
                if operator == ">":
                    return left > right_value
                elif operator == "<":
                    return left < right_value
                elif operator == ">=":
                    return left >= right_value
                elif operator == "<=":
                    return left <= right_value
                elif operator == "==":
                    return left == right_value
                elif operator == "!=":
                    return left != right_value

            # Try converting strings to numbers
            try:
                left_num = float(left) if not isinstance(left, (int, float)) else left
                right_num = (
                    float(right_value) if not isinstance(right_value, (int, float)) else right_value
                )

                if operator == ">":
                    return left_num > right_num
                elif operator == "<":
                    return left_num < right_num
                elif operator == ">=":
                    return left_num >= right_num
                elif operator == "<=":
                    return left_num <= right_num
                elif operator == "==":
                    return left_num == right_num
                elif operator == "!=":
                    return left_num != right_num
            except (ValueError, TypeError):
                # Fall back to string comparison for == and !=
                if operator == "==":
                    return str(left) == str(right_value)
                elif operator == "!=":
                    return str(left) != str(right_value)

        return False

    def _build_evaluation_context(self, context: dict[str, Any]) -> dict[str, Any]:
        eval_context = {}

        if hasattr(self.variable_provider, "flow_context") and hasattr(
            self.variable_provider.flow_context, "variables"
        ):
            for key, value in self.variable_provider.flow_context.variables.items():
                eval_context[key] = value

        eval_context.update(context)

        return eval_context


class FlowContextVariableProvider:
    def __init__(self, flow_context: Any):
        self.flow_context = flow_context

    def get_variable(self, name: str) -> Optional[Any]:
        return self.flow_context.variables.get(name)
