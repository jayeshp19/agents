from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from .enums import NodeType, SpeakerType, ToolType
from .fields import (
    Edge,
    GatherInputVariable,
    GlobalNodeSetting,
    Instruction,
    TransferDestination,
    TransferOption,
)
from .utils import parse_dataclass
from .validators import validate_node

logger = logging.getLogger(__name__)


@dataclass
class Node:
    id: str
    name: str
    type: NodeType | str
    instruction: Instruction | None = None
    tool_id: str | None = None
    tool_type: str | None = None
    speak_during_execution: bool | None = None
    wait_for_result: bool | None = None
    edges: list[Edge] = field(default_factory=list)
    skip_response_edge: Edge | None = None
    global_node_setting: GlobalNodeSetting | None = None
    # Transfer call specific fields
    transfer_destination: TransferDestination | None = None
    transfer_option: TransferOption | None = None
    edge: Edge | None = None  # Single edge for transfer failures
    # Gather input specific fields
    gather_input_variables: list[GatherInputVariable] = field(default_factory=list)
    gather_input_instruction: str | None = None
    # Additional fields for completeness
    display_position: dict[str, Any] | None = None
    finetune_conversation_examples: list[Any] = field(default_factory=list)
    start_speaker: SpeakerType | str | None = None

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("Node ID cannot be empty")
        if not self.name.strip():
            raise ValueError("Node name cannot be empty")

        # Convert string types to enums if possible
        if isinstance(self.type, str):
            try:
                self.type = NodeType(self.type)
            except ValueError:
                # Allow custom node types
                pass

        if isinstance(self.start_speaker, str):
            try:
                self.start_speaker = SpeakerType(self.start_speaker)
            except ValueError:
                # Allow custom speaker types
                pass

        validate_node(self)

    def validate_edges(self, available_node_ids: set[str]) -> list[str]:
        """Validate that all edge destinations point to valid nodes.

        Args:
            available_node_ids: Set of all valid node IDs in the flow

        Returns:
            List of validation error messages
        """
        errors = []

        for edge in self.edges:
            if edge.destination_node_id and edge.destination_node_id not in available_node_ids:
                errors.append(
                    f"Edge {edge.id} points to non-existent node: {edge.destination_node_id}"
                )

        if self.skip_response_edge and self.skip_response_edge.destination_node_id:
            if self.skip_response_edge.destination_node_id not in available_node_ids:
                errors.append(
                    f"Skip response edge points to non-existent node: {self.skip_response_edge.destination_node_id}"
                )

        return errors

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Node:
        return parse_dataclass(Node, d)


@dataclass
class ToolSpec:
    name: str
    description: str
    tool_id: str
    type: ToolType | str
    parameters: dict[str, Any]
    url: str | None = None
    # HTTP Configuration for custom functions
    http_method: Literal["GET", "POST", "PUT", "DELETE", "PATCH"] | None = None
    headers: dict[str, str] = field(default_factory=dict)
    query_parameters: dict[str, str] = field(default_factory=dict)
    timeout_ms: int = 120000  # 120 seconds default
    response_variables: dict[str, str] = field(default_factory=dict)
    # Calendar-specific fields
    event_type_id: int | None = None
    cal_api_key: str | None = None
    timezone: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Tool name cannot be empty")
        if not self.tool_id.strip():
            raise ValueError("Tool ID cannot be empty")
        if not self.description.strip():
            raise ValueError("Tool description cannot be empty")

        if isinstance(self.type, str):
            try:
                self.type = ToolType(self.type)
            except ValueError:
                pass

        if self.timeout_ms <= 0:
            raise ValueError("Timeout must be positive")

        if self.type == ToolType.CUSTOM:
            if not self.url:
                raise ValueError("Custom tools must have a URL")
            if self.http_method is None:
                self.http_method = "POST"

        if self.http_method is not None:
            valid_methods = ["GET", "POST", "PUT", "DELETE", "PATCH"]
            if self.http_method not in valid_methods:
                raise ValueError(f"Invalid HTTP method: {self.http_method}")

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ToolSpec:
        return parse_dataclass(ToolSpec, d)


@dataclass
class FlowSpec:
    conversation_flow_id: str
    version: int
    global_prompt: str
    nodes: dict[str, Node]
    start_node_id: str
    start_speaker: SpeakerType | str | None = None
    tools: dict[str, ToolSpec] = field(default_factory=dict)
    model_choice: dict[str, Any] | None = None
    begin_tag_display_position: dict[str, Any] | None = None
    is_published: bool | None = None

    def __post_init__(self) -> None:
        if not self.conversation_flow_id.strip():
            raise ValueError("Flow ID cannot be empty")
        if not self.global_prompt.strip():
            raise ValueError("Global prompt cannot be empty")
        if not self.start_node_id.strip():
            raise ValueError("Start node ID cannot be empty")

        # Check that we have at least one node
        if not self.nodes:
            raise ValueError("Flow must have at least one node")

        if self.start_node_id not in self.nodes:
            raise ValueError(f"Start node '{self.start_node_id}' not found")

        # Convert string types to enums if possible
        if isinstance(self.start_speaker, str):
            try:
                self.start_speaker = SpeakerType(self.start_speaker)
            except ValueError:
                # Allow custom speaker types
                pass

        # Validate that function nodes reference existing tools
        for node in self.nodes.values():
            if node.type == NodeType.FUNCTION and node.tool_id:
                if node.tool_id not in self.tools:
                    raise ValueError(f"Tool '{node.tool_id}' not found in tools")

        # Validate that edges reference existing nodes
        for node in self.nodes.values():
            for edge in node.edges:
                if edge.destination_node_id and edge.destination_node_id not in self.nodes:
                    raise ValueError(
                        f"Edge destination '{edge.destination_node_id}' not found in nodes"
                    )

            # Check skip_response_edge
            if (
                node.skip_response_edge
                and node.skip_response_edge.destination_node_id
                and node.skip_response_edge.destination_node_id not in self.nodes
            ):
                raise ValueError(
                    f"Skip response edge destination '{node.skip_response_edge.destination_node_id}' not found in nodes"
                )

    def validate_flow_structure(self) -> list[str]:
        """Validate the entire flow structure and return any errors.

        This performs comprehensive validation including:
        - Edge connectivity (all edges point to valid nodes)
        - Reachability analysis (detecting unreachable nodes)
        - Global node accessibility

        Returns:
            List of validation error messages (empty if no errors)
        """
        errors = []
        available_node_ids = set(self.nodes.keys())

        for _, node in self.nodes.items():
            node_errors = node.validate_edges(available_node_ids)
            errors.extend(node_errors)

        # Check for unreachable nodes, including nodes accessible via global settings
        reachable_nodes: set[str] = set()
        self._find_reachable_nodes(self.start_node_id, reachable_nodes)

        # Also include nodes that can be reached via global node settings
        for node_id, node in self.nodes.items():
            if node.global_node_setting:
                self._find_reachable_nodes(node_id, reachable_nodes)

        unreachable = available_node_ids - reachable_nodes
        if unreachable:
            # Only warn about truly unreachable nodes, not global entry points
            truly_unreachable = []
            for node_id in unreachable:
                node = self.nodes[node_id]
                # Don't warn about global nodes or end nodes
                if not node.global_node_setting and node.type != NodeType.END:
                    truly_unreachable.append(node_id)

            if truly_unreachable:
                errors.extend([f"Unreachable node: {node_id}" for node_id in truly_unreachable])

        return errors

    def _find_reachable_nodes(self, node_id: str, visited: set[str]) -> None:
        """Recursively find all reachable nodes from a starting point.

        Args:
            node_id: Starting node ID
            visited: Set to track visited nodes (modified in place)
        """
        if node_id in visited or node_id not in self.nodes:
            return

        visited.add(node_id)
        node = self.nodes[node_id]

        for edge in node.edges:
            if edge.destination_node_id:
                self._find_reachable_nodes(edge.destination_node_id, visited)

        if node.skip_response_edge and node.skip_response_edge.destination_node_id:
            self._find_reachable_nodes(node.skip_response_edge.destination_node_id, visited)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> FlowSpec:
        return parse_dataclass(FlowSpec, data)


# Export main types
__all__ = [
    "Node",
    "ToolSpec",
    "FlowSpec",
]
