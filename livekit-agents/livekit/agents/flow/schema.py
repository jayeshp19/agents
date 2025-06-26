from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Instruction:
    type: str
    text: str

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Instruction:
        return Instruction(type=d.get("type", ""), text=d.get("text", ""))


@dataclass
class TransitionCondition:
    type: str
    prompt: str

    @staticmethod
    def from_dict(d: dict[str, Any]) -> TransitionCondition:
        return TransitionCondition(type=d.get("type", ""), prompt=d.get("prompt", ""))


@dataclass
class Edge:
    id: str
    condition: str
    transition_condition: TransitionCondition
    destination_node_id: str | None = None

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Edge:
        tc = d.get("transition_condition", {})
        return Edge(
            id=d.get("id", ""),
            condition=d.get("condition", ""),
            transition_condition=TransitionCondition.from_dict(tc),
            destination_node_id=d.get("destination_node_id"),
        )


@dataclass
class GlobalNodeSetting:
    condition: str

    @staticmethod
    def from_dict(d: dict[str, Any]) -> GlobalNodeSetting:
        return GlobalNodeSetting(condition=d.get("condition", ""))


@dataclass
class Node:
    id: str
    name: str
    type: str
    instruction: Instruction | None = None
    tool_id: str | None = None
    tool_type: str | None = None
    speak_during_execution: bool | None = None
    wait_for_result: bool | None = None
    edges: list[Edge] = field(default_factory=list)
    skip_response_edge: Edge | None = None
    global_node_setting: GlobalNodeSetting | None = None

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Node:
        instruction = d.get("instruction")
        if instruction is not None:
            instruction = Instruction.from_dict(instruction)
        skip = d.get("skip_response_edge")
        if skip is not None:
            skip = Edge.from_dict(skip)
        global_setting = d.get("global_node_setting")
        if global_setting is not None:
            global_setting = GlobalNodeSetting.from_dict(global_setting)
        edges = [Edge.from_dict(e) for e in d.get("edges", [])]
        return Node(
            id=d["id"],
            name=d.get("name", ""),
            type=d.get("type", ""),
            instruction=instruction,
            tool_id=d.get("tool_id"),
            tool_type=d.get("tool_type"),
            speak_during_execution=d.get("speak_during_execution"),
            wait_for_result=d.get("wait_for_result"),
            edges=edges,
            skip_response_edge=skip,
            global_node_setting=global_setting,
        )


@dataclass
class ToolSpec:
    name: str
    description: str
    tool_id: str
    type: str
    parameters: dict[str, Any]
    url: str | None = None

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ToolSpec:
        return ToolSpec(
            name=d.get("name", ""),
            description=d.get("description", ""),
            tool_id=d.get("tool_id", ""),
            type=d.get("type", ""),
            parameters=d.get("parameters", {}),
            url=d.get("url"),
        )


@dataclass
class FlowSpec:
    conversation_flow_id: str
    version: int
    global_prompt: str
    nodes: dict[str, Node]
    start_node_id: str
    start_speaker: str | None = None
    tools: dict[str, ToolSpec] = field(default_factory=dict)
    model_choice: dict[str, Any] | None = None

    @staticmethod
    def from_dict(data: dict[str, Any]) -> FlowSpec:
        nodes = {n["id"]: Node.from_dict(n) for n in data.get("nodes", [])}
        tools = {t["tool_id"]: ToolSpec.from_dict(t) for t in data.get("tools", [])}
        return FlowSpec(
            conversation_flow_id=data.get("conversation_flow_id", ""),
            version=data.get("version", 0),
            global_prompt=data.get("global_prompt", ""),
            nodes=nodes,
            start_node_id=data.get("start_node_id", ""),
            start_speaker=data.get("start_speaker"),
            tools=tools,
            model_choice=data.get("model_choice"),
        )


def load_flow(path: str) -> FlowSpec:
    import json

    with open(path) as f:
        data = json.load(f)
    return FlowSpec.from_dict(data)
