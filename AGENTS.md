# AGENTS.md

This file provides guidance when working with code in this repository.

## Development Commands

### Python Environment Management
This project uses `uv` for dependency management:
```bash
# Install dependencies
uv sync --all-extras --dev

# Install specific plugin dependencies  
uv sync --extra openai --extra deepgram --extra elevenlabs

# Run commands with uv
uv run python examples/voice_agents/basic_agent.py dev
uv run pytest tests/
```

### Testing
```bash
# Run all tests
uv run pytest

# Run specific test files
uv run pytest tests/test_agent_session.py -v

# Run tests with specific plugin
PLUGIN=openai uv run pytest tests/test_tts.py

# Run tests in Docker environment (from tests/ directory)
make test
make up    # Start test infrastructure
make down  # Stop test infrastructure
```

### Code Quality
```bash
# Lint code with ruff
uv run ruff check
uv run ruff check --fix

# Format code
uv run ruff format

# Type checking
uv run mypy livekit-agents/
```

### Development Modes
```bash
# Development mode (hot reload, single process)
python myagent.py dev

# Console mode (interactive testing)
python myagent.py console

# Production mode
python myagent.py start
```

## High-Level Architecture

LiveKit Agents is a **plugin-based, multi-process framework** for building realtime voice AI applications. The architecture follows a **worker-job-session model** with clear separation between infrastructure and AI components.

### Core Components

**Worker** (`livekit-agents/livekit/agents/worker.py`)
- Central orchestrator managing agent job lifecycle
- Connects to LiveKit server, handles job assignment and load balancing
- Manages process/thread pools for job execution
- Provides health checks, metrics, and graceful shutdown

**JobContext** (`livekit-agents/livekit/agents/job.py`)
- Runtime environment for each agent job
- Manages room connections, participant lifecycle, shutdown callbacks
- Process/thread-safe execution context

**Agent** (`livekit-agents/livekit/agents/voice/agent.py`)
- Core abstraction defining agent behavior with customizable pipeline:
  - `stt_node()` - Speech-to-text processing
  - `llm_node()` - Language model processing  
  - `tts_node()` - Text-to-speech synthesis
- Supports instructions, tools, chat context, lifecycle hooks

**AgentSession** (`livekit-agents/livekit/agents/voice/agent_session.py`)
- Runtime orchestrator connecting audio/video streams with AI components
- Manages turn detection, interruptions, endpointing, multi-step tool calls
- Handles real-time voice interactions with configurable VAD

### Plugin Architecture

- **Registry-based discovery** with automatic plugin registration (`livekit-agents/livekit/agents/plugin.py`)
- Hot-reloadable in development mode
- Each plugin in `livekit-plugins/livekit-plugins-{provider}/` provides STT, TTS, LLM, or VAD implementations
- Plugins follow standardized interfaces with fallback adapters

### Voice Processing Pipeline

```
Audio Input → VAD/STT → LLM → TTS → Audio Output
     ↓           ↓       ↓     ↓        ↓  
  Turn Detection → Transcription → Tool Execution → Synthesis
```

**Key Abstractions:**
- **STT**: Streaming/non-streaming transcription with interim/final results
- **TTS**: Streaming/non-streaming synthesis with audio frame output
- **LLM**: Chat completion with tool calling, streaming responses, context management
- **VAD**: Real-time speech detection for turn management
- **RealtimeModel**: Server-side voice processing (OpenAI Realtime API, etc.)

### Development Pattern

Typical agent development follows this pattern:

1. **Define Agent** with instructions and function tools:
```python
@function_tool
async def my_tool(context: RunContext, param: str):
    return {"result": "value"}

agent = Agent(
    instructions="You are a helpful assistant.",
    tools=[my_tool]
)
```

2. **Create AgentSession** with STT/LLM/TTS components:
```python
session = AgentSession(
    vad=silero.VAD.load(),
    stt=deepgram.STT(model="nova-3"),
    llm=openai.LLM(model="gpt-4o-mini"),
    tts=elevenlabs.TTS(),
)
```

3. **Set up Worker** with job entrypoint:
```python
async def entrypoint(ctx: JobContext):
    await ctx.connect()
    await session.start(agent=agent, room=ctx.room)

cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
```

## Project Structure

```
/
├── livekit-agents/          # Core framework
│   ├── livekit/agents/      # Main package
│   │   ├── voice/          # Voice interaction components
│   │   ├── llm/            # LLM abstractions and utilities
│   │   ├── cli/            # Command-line interface
│   │   └── ipc/            # Inter-process communication
│   └── tests/              # Core framework tests
├── livekit-plugins/         # Plugin ecosystem
│   ├── livekit-plugins-openai/
│   ├── livekit-plugins-deepgram/
│   └── ...                 # Other provider plugins
├── examples/               # Example implementations
│   ├── voice_agents/       # Voice agent examples
│   ├── avatar_agents/      # Avatar integration examples
│   └── other/              # Other use cases
├── tests/                  # Integration tests
└── pyproject.toml          # Root workspace configuration
```

## Environment Variables

Required for most examples:
```bash
LIVEKIT_URL=wss://your-livekit-server.com
LIVEKIT_API_KEY=your-api-key
LIVEKIT_API_SECRET=your-api-secret

# Provider-specific keys
OPENAI_API_KEY=your-openai-key
DEEPGRAM_API_KEY=your-deepgram-key
ELEVENLABS_API_KEY=your-elevenlabs-key
```

## Key Design Patterns

- **Composition over inheritance**: Agents compose STT/LLM/TTS rather than inheriting
- **Event-driven architecture**: Components emit events for observability
- **Context propagation**: Thread/process-safe context using `contextvars`
- **Streaming pipeline**: Async generators with backpressure handling
- **Fallback adapters**: Graceful degradation when components fail

## Testing Considerations

- Tests use Docker Compose for external service dependencies
- Fake implementations available for unit testing (`tests/fake_*.py`)
- Plugin-specific tests require provider API keys
- Use `PLUGIN` environment variable to test specific providers

## Production Deployment

- Multi-process architecture with configurable process pools
- OpenTelemetry integration for distributed tracing
- Prometheus metrics endpoint for monitoring
- Health checks and graceful shutdown handling
- Connection pooling for external API calls
