"""
Input/output utilities for conversation flows.

This module provides functions for loading and saving conversation flows
from various sources including files, URLs, and in-memory data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

from .flow_spec import FlowSpec


def load_flow(source: Union[str, Path, dict[str, Any]]) -> FlowSpec:
    """Load a conversation flow from various sources.

    Args:
        source: Can be:
            - File path (str or Path) to a JSON file
            - Dictionary containing flow data

    Returns:
        FlowSpec instance

    Raises:
        FileNotFoundError: If file path doesn't exist
        ValueError: If data is invalid
        json.JSONDecodeError: If JSON is malformed

    Example:
        ```python
        # Load from file
        flow = load_flow("my_flow.json")

        # Load from dictionary
        flow_data = {"conversation_flow_id": "test", ...}
        flow = load_flow(flow_data)
        ```
    """
    if isinstance(source, dict):
        # Load from dictionary
        return FlowSpec.from_dict(source)

    # Load from file path
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Flow file not found: {path}")

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in {path}: {e.msg}", e.doc, e.pos) from e

    return FlowSpec.from_dict(data)


def save_flow(flow: FlowSpec, target: Union[str, Path]) -> None:
    """Save a conversation flow to a JSON file.

    Args:
        flow: FlowSpec instance to save
        target: File path where to save the flow

    Raises:
        OSError: If file cannot be written

    Example:
        ```python
        save_flow(my_flow, "output/saved_flow.json")
        ```
    """
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert FlowSpec to dictionary representation
    data = flow_to_dict(flow)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def flow_to_dict(flow: FlowSpec, use_arrays: bool = True) -> dict[str, Any]:
    """Convert a FlowSpec to its dictionary representation.

    Args:
        flow: FlowSpec instance to convert
        use_arrays: If True, nodes and tools are returned as arrays.
                   If False, they are returned as dictionaries keyed by ID.

    Returns:
        Dictionary representation suitable for JSON serialization
    """
    # Convert nodes dict to list
    nodes_list = []
    for node in flow.nodes.values():
        node_data: dict[str, Any] = {
            "id": node.id,
            "name": node.name,
            "type": str(node.type),
        }

        if node.instruction:
            node_data["instruction"] = {
                "type": str(node.instruction.type),
                "text": node.instruction.text,
            }

        if node.tool_id:
            node_data["tool_id"] = node.tool_id
        if node.tool_type:
            node_data["tool_type"] = node.tool_type
        if node.speak_during_execution is not None:
            node_data["speak_during_execution"] = node.speak_during_execution
        if node.wait_for_result is not None:
            node_data["wait_for_result"] = node.wait_for_result

        if node.edges:
            node_data["edges"] = [
                {
                    "id": edge.id,
                    "condition": edge.condition,
                    "transition_condition": {
                        "type": str(edge.transition_condition.type),
                        "prompt": edge.transition_condition.prompt,
                    },
                    "destination_node_id": edge.destination_node_id,
                }
                for edge in node.edges
            ]

        if node.skip_response_edge:
            edge = node.skip_response_edge
            node_data["skip_response_edge"] = {
                "id": edge.id,
                "condition": edge.condition,
                "transition_condition": {
                    "type": str(edge.transition_condition.type),
                    "prompt": edge.transition_condition.prompt,
                },
                "destination_node_id": edge.destination_node_id,
            }

        if node.global_node_setting:
            node_data["global_node_setting"] = {"condition": node.global_node_setting.condition}

        if node.gather_input_variables:
            node_data["gather_input_variables"] = [
                {
                    "name": var.name,
                    "type": var.type,
                    "description": var.description,
                    "required": var.required,
                    "max_attempts": var.max_attempts,
                    "regex_pattern": var.regex_pattern,
                    "regex_error_message": var.regex_error_message,
                }
                for var in node.gather_input_variables
            ]

        if node.gather_input_instruction:
            node_data["gather_input_instruction"] = node.gather_input_instruction

        if node.display_position:
            node_data["display_position"] = node.display_position

        if node.finetune_conversation_examples:
            node_data["finetune_conversation_examples"] = node.finetune_conversation_examples

        if node.start_speaker:
            node_data["start_speaker"] = str(node.start_speaker)

        nodes_list.append(node_data)

    # Convert tools dict to list
    tools_list = []
    for tool in flow.tools.values():
        tool_data: dict[str, Any] = {
            "name": tool.name,
            "description": tool.description,
            "tool_id": tool.tool_id,
            "type": str(tool.type),
            "parameters": tool.parameters,
        }

        if tool.url:
            tool_data["url"] = tool.url
        if tool.http_method is not None:
            tool_data["http_method"] = tool.http_method
        if tool.headers:
            tool_data["headers"] = tool.headers
        if tool.query_parameters:
            tool_data["query_parameters"] = tool.query_parameters
        if tool.timeout_ms != 120000:
            tool_data["timeout_ms"] = tool.timeout_ms
        if tool.event_type_id:
            tool_data["event_type_id"] = tool.event_type_id
        if tool.cal_api_key:
            tool_data["cal_api_key"] = tool.cal_api_key
        if tool.timezone:
            tool_data["timezone"] = tool.timezone

        tools_list.append(tool_data)

    # Build final dictionary
    if use_arrays:
        # Use arrays for nodes and tools
        result = {
            "conversation_flow_id": flow.conversation_flow_id,
            "version": flow.version,
            "global_prompt": flow.global_prompt,
            "nodes": nodes_list,
            "start_node_id": flow.start_node_id,
            "tools": tools_list,
        }
    else:
        # Use dictionaries keyed by ID for nodes and tools
        nodes_dict = {node_data["id"]: node_data for node_data in nodes_list}
        tools_dict = {tool_data["tool_id"]: tool_data for tool_data in tools_list}

        result = {
            "conversation_flow_id": flow.conversation_flow_id,
            "version": flow.version,
            "global_prompt": flow.global_prompt,
            "nodes": nodes_dict,
            "start_node_id": flow.start_node_id,
            "tools": tools_dict,
        }

    if flow.start_speaker:
        result["start_speaker"] = str(flow.start_speaker)
    if flow.model_choice:
        result["model_choice"] = flow.model_choice
    if flow.begin_tag_display_position:
        result["begin_tag_display_position"] = flow.begin_tag_display_position
    if flow.is_published is not None:
        result["is_published"] = flow.is_published

    return result


# Async versions using aiofiles if available
try:
    import aiofiles
    import aiofiles.os

    _HAS_AIOFILES = True
except ImportError:
    _HAS_AIOFILES = False


async def load_flow_async(source: Union[str, Path, dict[str, Any]]) -> FlowSpec:
    """Async version of load_flow.

    Args:
        source: Can be:
            - File path (str or Path) to a JSON file
            - Dictionary containing flow data

    Returns:
        FlowSpec instance

    Raises:
        ImportError: If aiofiles is not installed
        FileNotFoundError: If file path doesn't exist
        ValueError: If data is invalid
        json.JSONDecodeError: If JSON is malformed

    Note:
        Requires aiofiles package for file operations.
    """
    if not _HAS_AIOFILES:
        raise ImportError("aiofiles package required for async flow loading")

    if isinstance(source, dict):
        # Load from dictionary (no async needed)
        return FlowSpec.from_dict(source)

    # Load from file path
    path = Path(source)
    if not await aiofiles.os.path.exists(path):
        raise FileNotFoundError(f"Flow file not found: {path}")

    try:
        async with aiofiles.open(path, encoding="utf-8") as f:
            content = await f.read()
            data = json.loads(content)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in {path}: {e.msg}", e.doc, e.pos) from e

    return FlowSpec.from_dict(data)


async def save_flow_async(flow: FlowSpec, target: Union[str, Path]) -> None:
    """Async version of save_flow.

    Args:
        flow: FlowSpec instance to save
        target: File path where to save the flow

    Raises:
        ImportError: If aiofiles is not installed
        OSError: If file cannot be written
    """
    if not _HAS_AIOFILES:
        raise ImportError("aiofiles package required for async flow saving")

    path = Path(target)
    await aiofiles.os.makedirs(path.parent, exist_ok=True)

    # Convert FlowSpec to dictionary representation
    data = flow_to_dict(flow)
    content = json.dumps(data, indent=2, ensure_ascii=False)

    async with aiofiles.open(path, "w", encoding="utf-8") as f:
        await f.write(content)


# Export all functions
__all__ = [
    "load_flow",
    "save_flow",
    "flow_to_dict",
    "load_flow_async",
    "save_flow_async",
]
