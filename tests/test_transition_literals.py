import pytest

from livekit.agents.flow.base import FlowContext
from livekit.agents.flow.fields import Edge, TransitionCondition
from livekit.agents.flow.transition_evaluator import (
    FlowContextVariableProvider,
    TransitionEvaluator,
)


@pytest.mark.asyncio
async def test_equation_literal_right_operand_matches():
    ctx = FlowContext()
    provider = FlowContextVariableProvider(ctx)
    evaluator = TransitionEvaluator(provider)

    edges = [
        Edge(
            id="e1",
            condition="eq",
            transition_condition=TransitionCondition(
                type="equation",
                equations=[
                    {  # type: ignore[arg-type]
                        "left_operand": "status",
                        "operator": "==",
                        "right_operand": "ok",
                    }
                ],
            ),
            destination_node_id="next",
        )
    ]

    result = await evaluator.evaluate_transitions(
        edges=edges, user_text=None, context={"status": "ok"}
    )
    assert result is not None and result.destination_node_id == "next"


@pytest.mark.asyncio
async def test_equation_boolean_literal_coercion():
    ctx = FlowContext()
    provider = FlowContextVariableProvider(ctx)
    evaluator = TransitionEvaluator(provider)

    edges = [
        Edge(
            id="e1",
            condition="bool",
            transition_condition=TransitionCondition(
                type="equation",
                equations=[
                    {  # type: ignore[arg-type]
                        "left_operand": "success",
                        "operator": "==",
                        "right_operand": "false",
                    }
                ],
            ),
            destination_node_id="next",
        )
    ]

    result = await evaluator.evaluate_transitions(
        edges=edges, user_text=None, context={"success": False}
    )
    assert result is not None and result.destination_node_id == "next"
