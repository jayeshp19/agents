Flow Runner for LiveKit Agents
===============================

Table of Contents
-----------------
1. Motivation & Problem Statement  
2. Goals & Non-Goals  
3. High-level Architecture  
4. Data Model (schema)  
5. Developer Ergonomics (toolkit)  
6. Runtime Life-cycle (FlowRunner)  
7. Concurrency & Error Handling Strategies  
8. Extensibility & Backwards-compatibility  
9. Implementation Road-map & Milestones  
10. Open Questions

---

1. Motivation & Problem Statement
--------------------------------
LiveKit-Agents today is **code-first**: developers express conversation logic as imperative
Python async functions. For multi-step customer journeys (intake forms, guided support)
this becomes verbose and hard to visualize.

A **Flow Runner** layer lets developers (or non-devs using a GUI) describe the
conversation as a declarative **node graph** (similar to Retell or Pipecat-Flows)
while still leveraging LiveKit-Agents' best-in-class realtime media pipeline.

2. Goals & Non-Goals
--------------------
### Goals
* Declarative description of conversation flows (JSON / YAML / Python-dict).
* Zero changes required in `AgentSession`, `AgentActivity`, or media pipeline.
* Easy reuse of existing `@function_tool` code.
* Hot-swap flows at runtime; dynamic node creation supported.
* Feature parity with Retell/Pipecat core concepts (context strategies, edge & node
  functions, pre/post actions).

### Non-Goals
* Shipping a hosted visual editor (can follow later).
* Persisting flow state to a DB; runtime only.
* Replacing LiveKit's own turn-detection / interruption logic.

3. High-level Architecture
--------------------------
```mermaid
flowchart TD
    subgraph LiveKit_Runtime
        SES(AgentSession) --> ACT(AgentActivity) --> Pipeline
    end

    subgraph FlowPkg["livekit/agents/flow"]
        Runner[FlowRunner]\n(orchestrator)
        Schema[Node & FlowConfig]
        Toolkit[flow_tool / load_flow]
        Runner --> SES
        Runner --« uses »-- Schema
        Runner --« uses »-- Toolkit
    end

    Worker((entrypoint.py)) --> Toolkit --> Runner
```

4. Data Model (schema.py)
-------------------------
```python
class ContextStrategy(str, Enum):
    APPEND = "append"
    RESET = "reset"
    RESET_WITH_SUMMARY = "reset_with_summary"

class FunctionRef(BaseModel):
    name: str                       # matches a @flow_tool
    transition_to: str | None = None
    description: str | None = None
    parameters: dict = {}

class Node(BaseModel):
    id: str | None = None           # auto-filled if missing
    role_messages: list[dict] = []
    task_messages: list[dict]
    functions: list[FunctionRef] = []
    context_strategy: ContextStrategy = ContextStrategy.APPEND
    respond_immediately: bool = True

class FlowConfig(BaseModel):
    initial_node: str
    nodes: dict[str, Node]
```
* Pydantic v2 ensures validation and helpful error messages.

5. Developer Ergonomics (toolkit.py)
------------------------------------
```python
from livekit.agents.llm import function_tool

def flow_tool(*args, **kwargs):
    def wrap(fn):
        tool = function_tool(*args, **kwargs)(fn)
        ToolRegistry.register(tool)
        return tool
    return wrap

def load_flow(src):
    if isinstance(src, dict):
        return FlowConfig(**src)
    return FlowConfig(**yaml.safe_load(open(src)))
```

6. Runtime Life-cycle (runner.py)
---------------------------------
### Key attributes
```python
class FlowRunner:
    _sess: AgentSession
    _flow: FlowConfig
    _state: dict        # shared user data
    _current: Node | None
    _lock: asyncio.Lock # serialises transitions
    _pending_calls: int
```

### `start()`
```python
async def start(self):
    await self._enter_node(self._flow.initial_node)
```

### `_enter_node()`
1. Build `Agent` with updated prompts.  
2. Swap tool list via `Agent.update_tools()`.  
3. Apply context strategy (`RESET`, `SUMMARY`, etc.).  
4. If `respond_immediately`, call `session.generate_reply()`.

### Tool Dispatch
* For each `FunctionRef` Runner creates a **wrapper coroutine** that:
    1. Executes the real function (if registered).
    2. Returns `(result, next_node)` or uses `transition_to` field.
    3. Calls `await self._enter_node(next_node)` under lock.
    4. Decrements `_pending_calls` so Runner knows when it is safe to
       transition again.

### Turn Completion Hook
Runner attaches to current Agent's
`on_user_turn_completed` to be notified when user finished speaking.  If
`_pending_calls == 0` and node defines `default_transition`, Runner moves on.

7. Concurrency & Error Handling
------------------------------
* `self._lock` ensures only one transition executes at a time.
* Wrapper catches exceptions; converts to `ToolError` visible to LLM but keeps
  Runner alive.
* Depth-counter (e.g. 10) avoids infinite recursion of edge functions.
* When realtime-LLM is active, Runner waits for `speech_handle.wait_done()`
  before altering tool set (provider API limitation).

8. Extensibility & Back-compat
------------------------------
* Opt-in extra: `pip install "livekit-agents[flow]"`.
* Existing imperative apps unaffected.
* Schema version field allows future additions (`actions`, `guards`, etc.).
* Compatible with Pipecat-Flows: same YAML minus voice/audio fields.

9. Implementation Road-map
-------------------------
| Milestone | Deliverable | ETA |
|-----------|-------------|-----|
| M1 | `schema.py`, `toolkit.py`, unit tests | day 1 |
| M2 | `runner.py` basic (static flows, append strategy) | day 2 |
| M3 | Context strategies, error handling, dynamic nodes | day 3 |
| M4 | Docs & sample YAML, CLI flag `--flow` | day 4 |
| M5 | CI matrix, load-test, public beta | day 5 |

10. Open Questions
------------------
1. Should FlowRunner own its own `Agent` subclass (to expose extra hooks) or
   reuse vanilla `Agent`?  
2. Do we support multi-agent co-existence (two FlowRunners in one room) in v1? 
3. Do we minify tool schemas automatically or leave to developer?

---
*Last updated: $(date)* 