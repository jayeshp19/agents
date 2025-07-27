from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .flow_spec import FlowSpec


def load_flow(source: str | Path | dict[str, Any]) -> FlowSpec:
    if isinstance(source, dict):
        return FlowSpec.from_dict(source)

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Flow file not found: {path}")

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in {path}: {e.msg}", e.doc, e.pos) from e

    return FlowSpec.from_dict(data)


def save_flow(flow: FlowSpec, target: str | Path) -> None:
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = flow_to_dict(flow)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def flow_to_dict(flow: FlowSpec, use_arrays: bool = True) -> dict[str, Any]:
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
                        "type": edge.transition_condition.type,
                        "prompt": edge.transition_condition.prompt,
                        "equations": [
                            {
                                "left_operand": eq.left_operand,
                                "operator": eq.operator,
                                "right_operand": eq.right_operand,
                            }
                            for eq in (edge.transition_condition.equations or [])
                        ]
                        if edge.transition_condition.equations
                        else None,
                        "operator": edge.transition_condition.operator,
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
                    "type": edge.transition_condition.type,
                    "prompt": edge.transition_condition.prompt,
                    "equations": [
                        {
                            "left_operand": eq.left_operand,
                            "operator": eq.operator,
                            "right_operand": eq.right_operand,
                        }
                        for eq in (edge.transition_condition.equations or [])
                    ]
                    if edge.transition_condition.equations
                    else None,
                    "operator": edge.transition_condition.operator,
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

    if use_arrays:
        result = {
            "conversation_flow_id": flow.conversation_flow_id,
            "version": flow.version,
            "global_prompt": flow.global_prompt,
            "nodes": nodes_list,
            "start_node_id": flow.start_node_id,
            "tools": tools_list,
        }
    else:
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


__all__ = ["load_flow", "save_flow", "flow_to_dict"]
