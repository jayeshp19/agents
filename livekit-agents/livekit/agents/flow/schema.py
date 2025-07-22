"""
Conversation flow schema definitions.

This module provides backward compatibility by re-exporting all schema
components from their new modular locations. For new code, prefer
importing directly from the specific modules.

Modular structure:
- enums: NodeType, ToolType, InstructionType, etc.
- fields: Instruction, Edge, GatherInputVariable, etc.
- flow_spec: Node, ToolSpec, FlowSpec
- builder: FlowBuilder
- io: load_flow, save_flow, etc.
"""

from __future__ import annotations

# Re-export builder
from .builder import FlowBuilder

# Re-export all enum types
from .enums import (
    ALL_INSTRUCTION_TYPES,
    ALL_NODE_TYPES,
    ALL_SPEAKER_TYPES,
    ALL_TOOL_TYPES,
    ALL_TRANSITION_CONDITION_TYPES,
    InstructionType,
    NodeType,
    SpeakerType,
    ToolType,
    TransitionConditionType,
)

# Re-export all field types
from .fields import (
    Edge,
    GatherInputVariable,
    GlobalNodeSetting,
    Instruction,
    TransferDestination,
    TransferOption,
    TransitionCondition,
)

# Re-export main flow types
from .flow_spec import FlowSpec, Node, ToolSpec

# Re-export I/O functions
from .io import flow_to_dict, load_flow, load_flow_async, save_flow, save_flow_async

# Backward compatibility exports
__all__ = [
    # Enums
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
    # Fields
    "Instruction",
    "TransitionCondition",
    "Edge",
    "GlobalNodeSetting",
    "GatherInputVariable",
    "TransferDestination",
    "TransferOption",
    # Main types
    "Node",
    "ToolSpec",
    "FlowSpec",
    # Builder
    "FlowBuilder",
    # I/O
    "load_flow",
    "save_flow",
    "flow_to_dict",
    "load_flow_async",
    "save_flow_async",
]
