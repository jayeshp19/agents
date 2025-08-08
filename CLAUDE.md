# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is the **LiveKit Agents** repository, a powerful framework for building real-time voice AI agents. The repository contains two main frameworks:

1. **LiveKit Agents** - A comprehensive voice agent framework with WebRTC integration
2. **Pipecat** - A framework for building conversational voice agents with streaming pipelines

## Development Setup and Commands

### Package Management
This repository uses **UV** as the primary package manager and build tool.

```bash
# Install all dependencies including dev dependencies and optional extras
uv sync --all-extras --dev

# Run commands within the UV environment
uv run <command>
```

### Code Quality Commands
```bash
# Linting and formatting (always run these before committing)
uv run ruff format .              # Auto-format code
uv run ruff check --fix           # Auto-fix linting issues
uv run ruff check --select I --fix  # Fix import ordering
uv run ruff check --output-format=github .  # Check with GitHub format

# Type checking
uv run mypy --install-types --non-interactive
uv run mypy -p livekit.agents     # Check specific modules
```

### Testing Commands
```bash
# Run tests
uv run pytest                     # All tests
uv run pytest tests/test_agent_session.py  # Specific test file
uv run pytest -s --color=yes --tb=short --log-cli-level=DEBUG  # Detailed output

# Integration tests (requires Docker)
cd tests && make test            # Full integration test suite
cd tests && make up              # Start test infrastructure
cd tests && make down            # Stop test infrastructure
```

### Agent Development Commands
```bash
# Test agent locally in terminal (no external dependencies)
python myagent.py console

# Development mode with hot reloading (requires LiveKit server)
python myagent.py dev

# Production mode
python myagent.py start
```

### Required Environment Variables
For running agents, you'll typically need:
- `LIVEKIT_URL` - LiveKit server URL  
- `LIVEKIT_API_KEY` - API key
- `LIVEKIT_API_SECRET` - API secret
- Service-specific API keys (e.g., `OPENAI_API_KEY`, `DEEPGRAM_API_KEY`)

## Architecture Overview

### Core Components

#### Main Abstractions
- **`Agent`** - Base class defining agent behavior with configurable components
- **`AgentSession`** - Runtime that manages complete agent lifecycle and orchestrates components
- **`AgentTask`** - Specialized agent for task-based interactions with completion semantics
- **`Worker`** - Main process coordinating job scheduling and launching agents
- **`JobContext`** - Context for individual agent sessions with room connection

#### Component Interfaces
- **`STT`** (Speech-to-Text) - Speech recognition with streaming capabilities
- **`TTS`** (Text-to-Speech) - Voice synthesis with streaming support
- **`LLM`** (Large Language Model) - Text generation and tool calling
- **`VAD`** (Voice Activity Detection) - Speech boundary detection for turn management
- **`RealtimeModel`** - Real-time multimodal models (e.g., OpenAI Realtime API)

### Architecture Patterns

#### Plugin-Based Architecture
The framework uses a sophisticated plugin system with 30+ provider integrations:
```
livekit-plugins/
├── livekit-plugins-openai/     # OpenAI integration (GPT, TTS, STT)
├── livekit-plugins-anthropic/  # Claude integration  
├── livekit-plugins-deepgram/   # Deepgram STT/TTS
├── livekit-plugins-elevenlabs/ # ElevenLabs TTS
└── ... (30+ more providers)
```

Components implement common interfaces while wrapping provider-specific APIs.

#### Voice Processing Pipeline
```python
# Typical pipeline flow
audio_input → VAD → STT → LLM → TTS → audio_output

# Processing nodes are composable and streaming
stt_node() → llm_node() → transcription_node() → tts_node()
```

#### Multi-Agent Support
```python
# Seamless agent handoffs
session.update_agent(new_agent)

# Task-based agents with completion semantics
class SpecializedTask(AgentTask[str]):
    async def on_enter(self):
        # Task-specific logic
        pass
    
    def complete(self, result: str):
        # Return control to previous agent
        super().complete(result)
```

#### Flow System
Declarative conversation flows using JSON schema in `/flow/schema.py`:
- **`FlowSpec`** - Top-level flow definition with nodes and edges
- **`Node`** - Individual conversation states with instructions
- **`Edge`** - Conditional transitions based on LLM decisions
- **`ToolSpec`** - External tool integrations

### Process Management (IPC)
The `/ipc/` directory implements sophisticated process isolation:
- **`JobProcExecutor`** - Process-based execution for stability
- **`ProcPool`** - Manages pools of warm processes for fast startup
- **`InferenceExecutor`** - Separate processes for ML inference

### WebRTC Integration
- Agents join LiveKit rooms as participants for real-time communication
- Bidirectional audio/video streaming with low latency
- Built-in interruption handling and turn detection
- Quality adaptation based on network conditions

## Key Files and Directories

### Core Framework
- `livekit-agents/livekit/agents/voice/agent.py` - Main Agent classes
- `livekit-agents/livekit/agents/voice/agent_session.py` - Session management
- `livekit-agents/livekit/agents/job.py` - Job execution context
- `livekit-agents/livekit/agents/worker.py` - Main worker process
- `livekit-agents/livekit/agents/flow/schema.py` - Flow system implementation

### Component Abstractions
- `livekit-agents/livekit/agents/stt/stt.py` - Speech-to-text interface
- `livekit-agents/livekit/agents/tts/tts.py` - Text-to-speech interface
- `livekit-agents/livekit/agents/llm/llm.py` - LLM interface
- `livekit-agents/livekit/agents/vad.py` - Voice activity detection

### Examples
- `examples/voice_agents/basic_agent.py` - Simple voice agent template
- `examples/voice_agents/multi_agent.py` - Multi-agent handoff patterns
- `examples/voice_agents/restaurant_agent.py` - Complex real-world example

## Development Workflow

1. **Setup**: `uv sync --all-extras --dev`
2. **Code Changes**: Implement features following existing patterns
3. **Format**: `uv run ruff format .`
4. **Lint**: `uv run ruff check --fix`
5. **Type Check**: `uv run mypy --install-types --non-interactive`  
6. **Test**: `uv run pytest`
7. **Local Testing**: `python myagent.py console`
8. **Integration Testing**: `python myagent.py dev`

## Common Patterns

### Creating Agents
```python
# Simple agent with basic components
agent = Agent(
    instructions="You are a helpful assistant",
    tools=[my_tool_function],
)

# Agent with custom components
session = AgentSession(
    vad=silero.VAD.load(),
    stt=deepgram.STT(model="nova-3"),
    llm=openai.LLM(model="gpt-4o-mini"),
    tts=elevenlabs.TTS(),
)
```

### Tool Integration
```python
@function_tool  
async def lookup_weather(context: RunContext, location: str):
    """Look up weather information."""
    return {"weather": "sunny", "temperature": 70}
```

### Agent Handoffs
```python
# In an agent method
new_agent = SpecializedAgent(user_data)
return new_agent, "Transferring to specialist..."
```

## Testing Strategy

- **Unit Tests**: Component-level testing with mocks
- **Integration Tests**: Full pipeline testing with Docker infrastructure  
- **Manual Testing**: Console mode for interactive development
- **Live Testing**: Dev mode with real WebRTC connections

## Important Notes

- Always use streaming interfaces for real-time performance
- Handle interruptions gracefully in TTS components
- Manage conversation context carefully in multi-agent scenarios
- Use process isolation for production deployments
- Monitor resource usage, especially with ML models
- Test with various audio quality conditions
- Consider latency impact of component choices


# docs: https://docs.livekit.io/llms.txt