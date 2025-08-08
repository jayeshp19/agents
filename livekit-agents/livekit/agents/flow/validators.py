from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

from .enums import NodeType

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    from .flow_spec import Node

logger = logging.getLogger(__name__)


def _validate_function_node(node: Node) -> None:
    if not node.tool_id:
        raise ValueError("Function nodes must have a tool_id")
    if not node.edges:
        raise ValueError(
            f"Function node '{node.id}' must have at least one edge to handle results. "
            "Consider adding edges for success/failure conditions."
        )
    for edge in node.edges:
        if not edge.destination_node_id:
            raise ValueError(
                f"Function node '{node.id}' has edge '{edge.id}' without destination_node_id"
            )


def _validate_gather_node(node: Node) -> None:
    if not node.gather_input_variables:
        raise ValueError("Gather input nodes must have variables")
    if len(node.edges) != 1:
        raise ValueError("Gather input nodes must have exactly one exit edge")
    if not node.edges[0].destination_node_id:
        raise ValueError(f"Gather input node '{node.id}' edge has no destination_node_id")


def _validate_conversation_node(node: Node) -> None:
    for edge in node.edges:
        if not edge.destination_node_id:
            raise ValueError(
                f"Conversation node '{node.id}' has edge '{edge.id}' without destination_node_id"
            )


def _validate_end_node(node: Node) -> None:
    if node.edges:
        logger.warning(
            f"End node '{node.id}' has {len(node.edges)} edges. End nodes should not have outgoing edges."
        )


_VALIDATORS: dict[NodeType, Callable[[Node], None]] = {
    NodeType.FUNCTION: _validate_function_node,
    NodeType.GATHER_INPUT: _validate_gather_node,
    NodeType.CONVERSATION: _validate_conversation_node,
    NodeType.END: _validate_end_node,
}


def validate_node(node: Node) -> None:
    validator = _VALIDATORS.get(node.type)  # type: ignore[dict-item]
    if validator:
        validator(node)
