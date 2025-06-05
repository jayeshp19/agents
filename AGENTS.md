# LiveKit Agents Repository Overview

## Repository Purpose

This repository contains the **LiveKit Agents framework** - a comprehensive Python framework for building realtime voice AI agents that can see, hear, and speak. It enables the creation of server-side agentic applications with full integration to the LiveKit WebRTC media server ecosystem.

## Architecture Overview

### Core Components

1. **Agent** - The main LLM-based application with defined instructions
2. **AgentSession** - Container that manages interactions between agents and end users
3. **Worker** - Main process that coordinates job scheduling and launches agents for user sessions
4. **JobContext** - Execution context for agent jobs
5. **entrypoint** - Starting point for interactive sessions (similar to web server request handler)

### Key Modules

```
livekit-agents/livekit/agents/
├── voice/           # Voice agent implementations and session management
├── llm/             # Large Language Model integrations and chat context
├── stt/             # Speech-to-Text providers
├── tts/             # Text-to-Speech providers
├── vad/             # Voice Activity Detection
├── cli/             # Command line interface
├── worker.py        # Worker process management
└── job.py          # Job execution and context management
```

## Plugin Architecture

The framework uses a modular plugin system where different AI service providers are implemented as separate plugins:

### Available Plugins
- **LLM Providers**: OpenAI, Anthropic, Google (Gemini), AWS Bedrock, Groq
- **STT Providers**: Deepgram, OpenAI Whisper, Google Cloud STT, AssemblyAI, Azure, AWS Transcribe
- **TTS Providers**: OpenAI, ElevenLabs, Google Cloud TTS, Cartesia, AWS Polly, Azure
- **VAD Providers**: Silero (most common), Turn Detector
- **Avatar Providers**: Tavus, BitHuman, Bey
- **MCP Support**: Native Model Context Protocol integration

Each plugin follows the pattern:
```python
from livekit.agents import Plugin

class ProviderPlugin(Plugin):
    def __init__(self):
        super().__init__(__name__, __version__, __package__, logger)

Plugin.register_plugin(ProviderPlugin())
```

## Async/Await Architecture

The framework is built on Python's asyncio and follows strict async patterns:

### Key Async Patterns

1. **Session Management**: All agent sessions are async
```python
async def entrypoint(ctx: JobContext):
    await ctx.connect()
    session = AgentSession(...)
    await session.start(agent=agent, room=ctx.room)
```

2. **Agent Lifecycle**: Agent methods are async
```python
class MyAgent(Agent):
    async def on_enter(self):
        # Called when agent enters session
        await self.session.generate_reply()
    
    async def on_exit(self):
        # Called when agent exits session
        pass
```

3. **Function Tools**: Agent tools are async
```python
@function_tool
async def lookup_weather(context: RunContext, location: str):
    """Tool for weather lookup"""
    # Async operations here
    return {"weather": "sunny"}
```

4. **Stream Processing**: All audio/text processing is async
```python
async for chunk in llm_stream:
    # Process streaming responses
    await handle_chunk(chunk)
```

## Dependencies and Setup

### Package Management
The project uses **uv** (ultra-fast Python package manager) instead of pip:

```bash
# Install dependencies
uv sync

# Install with specific plugins
uv add "livekit-agents[openai,silero,deepgram,cartesia,turn-detector]~=1.0"
```

### Core Dependencies (from pyproject.toml)
- **livekit**: WebRTC media server SDK
- **asyncio**: Async runtime
- **aiohttp**: Async HTTP client/server
- **python-dotenv**: Environment variable management

### Development Dependencies
- **mypy**: Type checking (strict mode enabled)
- **ruff**: Linting and formatting
- **pytest**: Testing with async support
- **pytest-asyncio**: Async test support

### Environment Variables
```bash
# Required for most agents
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=your_api_key
LIVEKIT_API_SECRET=your_api_secret

# Provider-specific keys
OPENAI_API_KEY=your_openai_key
DEEPGRAM_API_KEY=your_deepgram_key
ELEVENLABS_API_KEY=your_elevenlabs_key
```

## Worker Architecture

### Job Execution Models
- **Process-based**: Default on Linux (forkserver) and other platforms (spawn)
- **Thread-based**: Alternative execution model
- **Multiprocessing contexts**: Configurable (`spawn`, `forkserver`)

### Worker Configuration
```python
WorkerOptions(
    entrypoint_fnc=entrypoint,
    prewarm_fnc=prewarm,  # Optional process initialization
    load_threshold=0.75,  # CPU load threshold
    job_executor_type=JobExecutorType.PROCESS,
    num_idle_processes=cpu_count(),  # Warm process pool
)
```

### Memory Management
- **job_memory_warn_mb**: Warning threshold (default: 500MB)
- **job_memory_limit_mb**: Hard limit (default: disabled)
- **shutdown_process_timeout**: Graceful shutdown time (default: 60s)

## Real-time Features

### Voice Activity Detection (VAD)
```python
vad = silero.VAD.load()  # Pre-trained model
session = AgentSession(vad=vad, ...)
```

### Turn Detection
- **Semantic turn detection**: Transformer-based model to detect conversation turns
- **Multilingual support**: Available for multiple languages
- **Configurable timing**: `min_endpointing_delay`, `max_endpointing_delay`

### Interruption Handling
```python
# Configure interruption behavior
session = AgentSession(
    allow_interruptions=True,
    min_interruption_duration=0.5,
    min_interruption_words=0,
)

# Programmatic interruption
await session.interrupt()
```

## Best Practices

### 1. Resource Management
- Always use `async with` for contexts
- Properly close sessions with `await session.aclose()`
- Use prewarm functions for expensive initialization

### 2. Error Handling
```python
@utils.log_exceptions(logger=logger)
async def my_function():
    try:
        # Agent logic
        pass
    except (llm.LLMError, stt.STTError, tts.TTSError) as e:
        if e.recoverable:
            # Handle recoverable errors
            pass
        else:
            # Log and close session
            logger.error("Unrecoverable error", exc_info=e)
```

### 3. Performance Optimization
- Use warm process pools in production
- Configure appropriate load thresholds
- Monitor memory usage with job limits
- Use background audio for better UX

### 4. Development vs Production
```python
# Development: More verbose logging, no load limits
WorkerOptions(load_threshold=math.inf, num_idle_processes=0)

# Production: Resource limits, process pools
WorkerOptions(load_threshold=0.75, num_idle_processes=cpu_count())
```

## Testing and Development

### Running Tests
```bash
# Run all tests
uv run pytest

# Run with async support
uv run pytest --asyncio-mode=auto
```

### Linting and Type Checking
```bash
# Format code
uv run ruff format

# Lint code
uv run ruff check

# Type checking
uv run mypy
```

### Development Server
```bash
# Run with development mode
python main.py --dev

# Production mode
python main.py
```

## Common Patterns

### Basic Voice Agent
```python
async def entrypoint(ctx: JobContext):
    await ctx.connect()
    
    session = AgentSession(
        vad=silero.VAD.load(),
        stt=deepgram.STT(model="nova-3"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=elevenlabs.TTS(),
    )
    
    agent = Agent(instructions="Your instructions here")
    await session.start(agent=agent, room=ctx.room)
```

### Multi-Agent Handoff
```python
class AgentA(Agent):
    @function_tool
    async def handoff_to_b(self, context: RunContext):
        return AgentB(), "Switching to specialist agent"

class AgentB(Agent):
    async def on_enter(self):
        await self.session.generate_reply()
```

### Realtime API Usage
```python
# Use OpenAI Realtime API instead of separate STT/LLM/TTS
session = AgentSession(
    llm=openai.realtime.RealtimeModel(voice="alloy"),
    vad=silero.VAD.load(),
)
```

## File Structure for New Agents

```
your_agent/
├── main.py              # Entry point with entrypoint function
├── requirements.txt     # Dependencies
├── .env                # Environment variables
├── agents/
│   ├── __init__.py
│   └── my_agent.py     # Agent implementation
└── tools/
    ├── __init__.py
    └── custom_tools.py  # Custom function tools
```

## Deployment Considerations

### Docker
- Use multi-stage builds for optimization
- Include all required models and dependencies
- Set appropriate resource limits

### Scaling
- Workers automatically distribute across available CPUs
- Use load balancing for multiple worker instances
- Monitor memory usage per job

### Monitoring
- Built-in metrics collection for LLM, STT, TTS usage
- Health check endpoints on configurable ports
- Structured logging with context fields

This documentation provides the foundational knowledge needed to understand and work with the LiveKit Agents framework effectively. go through code in order to get full context.