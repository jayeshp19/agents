
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

        # Node-specific validation
        if self.type == "function":
            # Function nodes MUST have edges to handle success/failure
            if not self.edges or len(self.edges) == 0:
                raise ValueError(
                    f"Function node '{self.id}' must have at least one edge to handle results. "
                    "Consider adding edges for success/failure conditions."
                )
            # Validate that all edges have destination nodes
            for edge in self.edges:
                if not edge.destination_node_id:
                    raise ValueError(
                        f"Function node '{self.id}' has edge '{edge.id}' without destination_node_id"
                    )

        elif self.type == "gather_input":
            # Gather nodes MUST have exactly one edge (automatic transition)
            if not self.edges or len(self.edges) == 0:
                raise ValueError(
                    f"Gather input node '{self.id}' must have exactly one edge. "
                    "Gather nodes transition automatically when all required data is collected."
                )
            elif len(self.edges) > 1:
                logger.warning(
                    f"Gather input node '{self.id}' has {len(self.edges)} edges. "
                    "Only the first edge will be used for automatic transition."
                )
            # Validate destination exists
            if not self.edges[0].destination_node_id:
                raise ValueError(f"Gather input node '{self.id}' edge has no destination_node_id")

        elif self.type == "conversation":
            # Conversation nodes can have 0 edges (rely on global transitions)
            # But if they have edges, validate destinations
            for edge in self.edges:
                if not edge.destination_node_id:
                    raise ValueError(
                        f"Conversation node '{self.id}' has edge '{edge.id}' without destination_node_id"
                    )

        elif self.type == "end":
            # End nodes should not have edges
            if self.edges and len(self.edges) > 0:
                logger.warning(
                    f"End node '{self.id}' has {len(self.edges)} edges. "
                    "End nodes should not have outgoing edges."
                )
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

        # Validate node type specific requirements
        if self.type == NodeType.FUNCTION and not self.tool_id:
            raise ValueError("Function nodes must have a tool_id")
        if self.type == NodeType.GATHER_INPUT and not self.gather_input_variables:
            raise ValueError("Gather input nodes must have variables")
        if self.type == NodeType.GATHER_INPUT and len(self.edges) > 1:
            raise ValueError("Gather input nodes can only have one exit edge")
        if self.type == NodeType.GATHER_INPUT and len(self.edges) == 0:
            raise ValueError("Gather input nodes must have exactly one exit edge")

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
        """Create a Node from a dictionary representation.

        Args:
            d: Dictionary containing node data

        Returns:
            Node instance

        Raises:
            ValueError: If required fields are missing or invalid
        """
        if not isinstance(d, dict):
            raise ValueError("Node must be a dictionary")

        node_id = d.get("id", "")
        if not node_id:
            raise ValueError("Node ID is required")

        instruction = None
        if "instruction" in d and d["instruction"] is not None:
            instruction = Instruction.from_dict(d["instruction"])

        skip = None
        if "skip_response_edge" in d and d["skip_response_edge"] is not None:
            skip = Edge.from_dict(d["skip_response_edge"])

        global_setting = None
        if "global_node_setting" in d and d["global_node_setting"] is not None:
            global_setting = GlobalNodeSetting.from_dict(d["global_node_setting"])

        transfer_dest = None
        if "transfer_destination" in d and d["transfer_destination"] is not None:
            transfer_dest = TransferDestination.from_dict(d["transfer_destination"])

        transfer_opt = None
        if "transfer_option" in d and d["transfer_option"] is not None:
            transfer_opt = TransferOption.from_dict(d["transfer_option"])

        single_edge = None
        if "edge" in d and d["edge"] is not None:
            single_edge = Edge.from_dict(d["edge"])

        edges = []
        if "edges" in d:
            edges = [Edge.from_dict(e) for e in d["edges"] if e is not None]

        gather_vars = []
        if "gather_input_variables" in d:
            gather_vars = [
                GatherInputVariable.from_dict(var)
                for var in d["gather_input_variables"]
                if var is not None
            ]

        return Node(
            id=node_id,
            name=d.get("name", ""),
            type=d.get("type", "conversation"),
            instruction=instruction,
            tool_id=d.get("tool_id"),
            tool_type=d.get("tool_type"),
            speak_during_execution=d.get("speak_during_execution"),
            wait_for_result=d.get("wait_for_result"),
            edges=edges,
            skip_response_edge=skip,
            global_node_setting=global_setting,
            transfer_destination=transfer_dest,
            transfer_option=transfer_opt,
            edge=single_edge,
            gather_input_variables=gather_vars,
            gather_input_instruction=d.get("gather_input_instruction"),
            display_position=d.get("display_position"),
            finetune_conversation_examples=d.get("finetune_conversation_examples", []),
            start_speaker=d.get("start_speaker"),
        )


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
        """Create a ToolSpec from a dictionary representation.

        Args:
            d: Dictionary containing tool data

        Returns:
            ToolSpec instance

        Raises:
            ValueError: If required fields are missing or invalid
        """
        if not isinstance(d, dict):
            raise ValueError("ToolSpec must be a dictionary")

        name = d.get("name", "")
        tool_id = d.get("tool_id", "")

        if not name:
            raise ValueError("Tool name is required")
        if not tool_id:
            raise ValueError("Tool ID is required")

        return ToolSpec(
            name=name,
            description=d.get("description", ""),
            tool_id=tool_id,
            type=d.get("type", "custom"),
            parameters=d.get("parameters", {}),
            url=d.get("url"),
            http_method=d.get("http_method", "POST"),
            headers=d.get("headers", {}),
            query_parameters=d.get("query_parameters", {}),
            timeout_ms=d.get("timeout_ms", 120000),
            response_variables=d.get("response_variables", {}),
            event_type_id=d.get("event_type_id"),
            cal_api_key=d.get("cal_api_key"),
            timezone=d.get("timezone"),
        )


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
        """Create a FlowSpec from a dictionary representation.

        Args:
            data: Dictionary containing complete flow data

        Returns:
            FlowSpec instance

        Raises:
            ValueError: If required fields are missing or data is invalid
        """
        if not isinstance(data, dict):
            raise ValueError("FlowSpec data must be a dictionary")

        # Validate required fields
        required_fields = ["conversation_flow_id", "nodes", "start_node_id"]
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise ValueError(f"Flow is missing required fields: {', '.join(missing_fields)}")

        nodes_data = data.get("nodes", [])
        tools_data = data.get("tools", [])

        # Handle nodes - can be array or dictionary
        nodes = {}
        if isinstance(nodes_data, list):
            # Array format: [{"id": "node1", ...}, {"id": "node2", ...}]
            for node_data in nodes_data:
                try:
                    node = Node.from_dict(node_data)
                    nodes[node.id] = node
                except Exception as e:
                    raise ValueError(
                        f"Error parsing node {node_data.get('id', 'unknown')}: {e}"
                    ) from e
        elif isinstance(nodes_data, dict):
            # Dictionary format: {"node1": {"id": "node1", ...}, "node2": {...}}
            for node_id, node_data in nodes_data.items():
                try:
                    node = Node.from_dict(node_data)
                    nodes[node.id] = node
                except Exception as e:
                    raise ValueError(f"Error parsing node {node_id}: {e}") from e
        else:
            raise ValueError("Nodes must be a list or dictionary")

        # Handle tools - can be array or dictionary
        tools = {}
        if isinstance(tools_data, list):
            # Array format: [{"tool_id": "tool1", ...}, {"tool_id": "tool2", ...}]
            for tool_data in tools_data:
                try:
                    tool = ToolSpec.from_dict(tool_data)
                    tools[tool.tool_id] = tool
                except Exception as e:
                    raise ValueError(
                        f"Error parsing tool {tool_data.get('tool_id', 'unknown')}: {e}"
                    ) from e
        elif isinstance(tools_data, dict):
            # Dictionary format: {"tool1": {"tool_id": "tool1", ...}, "tool2": {...}}
            for tool_id, tool_data in tools_data.items():
                try:
                    tool = ToolSpec.from_dict(tool_data)
                    tools[tool.tool_id] = tool
                except Exception as e:
                    raise ValueError(f"Error parsing tool {tool_id}: {e}") from e
        else:
            raise ValueError("Tools must be a list or dictionary")

        return FlowSpec(
            conversation_flow_id=data["conversation_flow_id"],
            version=data.get("version", 0),
            global_prompt=data.get("global_prompt", ""),
            nodes=nodes,
            start_node_id=data["start_node_id"],
            start_speaker=data.get("start_speaker"),
            tools=tools,
            model_choice=data.get("model_choice"),
            begin_tag_display_position=data.get("begin_tag_display_position"),
            is_published=data.get("is_published"),
        )


# Export main types
__all__ = [
    "Node",
    "ToolSpec",
    "FlowSpec",
]
