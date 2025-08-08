from pathlib import Path

from livekit.agents.flow import load_flow
from livekit.agents.flow.io import save_flow


def _example_path() -> Path:
    return Path(__file__).parent.parent / "examples" / "flows" / "advanced_test_flow.json"


def test_advanced_flow_round_trip(tmp_path: Path):
    flow = load_flow(str(_example_path()))
    assert flow.conversation_flow_id == "advanced_test_flow"
    assert flow.start_node_id == "start-conversation"

    out = tmp_path / "round.json"
    save_flow(flow, str(out))
    loaded = load_flow(str(out))

    assert loaded.conversation_flow_id == flow.conversation_flow_id
    assert len(loaded.nodes) == len(flow.nodes)


def test_advanced_flow_tools_and_nodes():
    flow = load_flow(str(_example_path()))

    # Nodes present
    assert "triage-issue" in flow.nodes
    assert "provide-solution" in flow.nodes
    assert "create-ticket" in flow.nodes

    # Tools present and typed
    assert "triage-tool" in flow.tools
    assert "check-verification-tool" in flow.tools
    assert "apply-fix-tool" in flow.tools
    assert "create-ticket-tool" in flow.tools

    # Tool type should resolve to enum; compare string value
    assert str(flow.tools["triage-tool"].type) == "local"

    # No structural validation errors
    assert not flow.validate_flow_structure()

