import pytest

from livekit.agents.flow.base import FlowContext
from livekit.agents.flow.fields import Edge, TransitionCondition
from livekit.agents.flow.transition_evaluator import (
    FlowContextVariableProvider,
    TransitionEvaluator,
)


@pytest.mark.asyncio
async def test_case_insensitive_equals():
    ctx = FlowContext()
    provider = FlowContextVariableProvider(ctx)
    evaluator = TransitionEvaluator(provider)

    edges = [
        Edge(
            id="e1",
            condition="eqi",
            transition_condition=TransitionCondition(
                type="equation",
                equations=[
                    {  # type: ignore[arg-type]
                        "left_operand": "val",
                        "operator": "==~",
                        "right_operand": "Hello",
                    }
                ],
            ),
            destination_node_id="next",
        )
    ]

    result = await evaluator.evaluate_transitions(
        edges=edges, user_text=None, context={"val": "hello"}
    )
    assert result is not None and result.destination_node_id == "next"


@pytest.mark.asyncio
async def test_in_operator_list_membership():
    ctx = FlowContext()
    provider = FlowContextVariableProvider(ctx)
    evaluator = TransitionEvaluator(provider)

    edges = [
        Edge(
            id="e1",
            condition="inlist",
            transition_condition=TransitionCondition(
                type="equation",
                equations=[
                    {  # type: ignore[arg-type]
                        "left_operand": "color",
                        "operator": "in",
                        "right_operand": "allowed_colors",
                    }
                ],
            ),
            destination_node_id="ok",
        )
    ]

    ctx.variables["allowed_colors"] = ["red", "green", "blue"]
    result = await evaluator.evaluate_transitions(
        edges=edges, user_text=None, context={"color": "green"}
    )
    assert result is not None and result.destination_node_id == "ok"


@pytest.mark.asyncio
async def test_regex_matches_and_case_insensitive_neq():
    ctx = FlowContext()
    provider = FlowContextVariableProvider(ctx)
    evaluator = TransitionEvaluator(provider)

    edges = [
        Edge(
            id="e1",
            condition="re",
            transition_condition=TransitionCondition(
                type="equation",
                equations=[
                    {  # type: ignore[arg-type]
                        "left_operand": "text",
                        "operator": "matches",
                        "right_operand": "^hello\\s+world$",
                    }
                ],
            ),
            destination_node_id="ok",
        )
    ]

    result = await evaluator.evaluate_transitions(
        edges=edges, user_text=None, context={"text": "hello world"}
    )
    assert result is not None and result.destination_node_id == "ok"

    # Case-insensitive inequality
    edges2 = [
        Edge(
            id="e2",
            condition="neqi",
            transition_condition=TransitionCondition(
                type="equation",
                equations=[
                    {  # type: ignore[arg-type]
                        "left_operand": "kind",
                        "operator": "!=~",
                        "right_operand": "Admin",
                    }
                ],
            ),
            destination_node_id="ok2",
        )
    ]
    result2 = await evaluator.evaluate_transitions(
        edges=edges2, user_text=None, context={"kind": "user"}
    )
    assert result2 is not None and result2.destination_node_id == "ok2"
