# Pipecat & Pipecat-Flows: Complete Technical Guide

## Table of Contents

1. [Platform Overview](#platform-overview)
2. [Core Architecture](#core-architecture) 
3. [Pipeline Framework](#pipeline-framework)
4. [Pipecat-Flows State Machine](#pipecat-flows-state-machine)
5. [Transport Integrations](#transport-integrations)
6. [Service Integrations](#service-integrations)
7. [Advanced Flow Patterns](#advanced-flow-patterns)
8. [Performance Optimization](#performance-optimization)

## Platform Overview

Pipecat is an open-source Python framework for building voice and multimodal conversational AI applications using a pipeline-based architecture. Pipecat-Flows adds a declarative state machine layer for conversation flow orchestration.

### Key Features

- **Pipeline Architecture**: Modular, composable processing pipeline
- **Multi-Transport Support**: Daily, LiveKit, local audio, telephony, and more
- **Flexible Service Integration**: LLM, STT, TTS, VAD from multiple providers
- **Real-time Processing**: Low-latency audio/video processing
- **Flow Orchestration**: Optional declarative conversation flows
- **Visual Editor**: Web-based flow designer (Pipecat-Flows)

### Architecture Layers

```
Application Layer
├── Conversation Flows (Pipecat-Flows)
│   ├── FlowManager        # State machine orchestrator
│   ├── Adapters          # LLM-specific formatting
│   └── Actions           # Side-effect handlers

Pipeline Layer (Pipecat Core)
├── Transports           # Audio/video I/O
├── Processors           # Frame processing logic
├── Services             # External service integrations
└── Pipeline             # Execution orchestration

Foundation Layer
├── Frames              # Data containers
├── Tasks               # Async execution
└── Utils               # Helper utilities
```

## Core Architecture

### Frame-Based Processing Model

Pipecat uses a frame-based processing model where all data flows through the pipeline as discrete frames:

```python
# Core frame types
class Frame:
    """Base frame type"""
    pass

class AudioFrame(Frame):
    """Audio data frame"""
    def __init__(self, data: bytes, sample_rate: int, num_channels: int):
        self.data = data
        self.sample_rate = sample_rate
        self.num_channels = num_channels

class TextFrame(Frame):
    """Text data frame"""
    def __init__(self, text: str):
        self.text = text

class UserMessageFrame(Frame):
    """User input message"""
    def __init__(self, content: str, timestamp: float):
        self.content = content
        self.timestamp = timestamp

class BotMessageFrame(Frame):
    """Bot response message"""
    def __init__(self, content: str, timestamp: float):
        self.content = content
        self.timestamp = timestamp

class LLMMessagesFrame(Frame):
    """LLM conversation context"""
    def __init__(self, messages: list[dict]):
        self.messages = messages

class FunctionCallFrame(Frame):
    """Function call request"""
    def __init__(self, name: str, arguments: dict, call_id: str):
        self.name = name
        self.arguments = arguments
        self.call_id = call_id
```

### Processor Base Classes

```python
class FrameProcessor:
    """Base processor class"""
    
    def __init__(self, name: str = None):
        self._name = name or self.__class__.__name__
        self._sink: FrameProcessor | None = None
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process a single frame"""
        await self.push_frame(frame, direction)
        
    async def push_frame(self, frame: Frame, direction: FrameDirection):
        """Push frame to next processor"""
        if self._sink:
            await self._sink.process_frame(frame, direction)

class AudioProcessor(FrameProcessor):
    """Base class for audio processing"""
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, AudioFrame):
            await self.process_audio(frame, direction)
        else:
            await self.push_frame(frame, direction)
            
    async def process_audio(self, frame: AudioFrame, direction: FrameDirection):
        """Override to implement audio processing"""
        await self.push_frame(frame, direction)
```

## Pipeline Framework

### Pipeline Construction

```python
from pipecat import Pipeline
from pipecat.transports.daily import DailyTransport
from pipecat.services.openai import OpenAILLMService
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.services.deepgram import DeepgramSTTService

async def create_pipeline():
    # Initialize transport
    transport = DailyTransport(
        room_url="https://example.daily.co/room",
        token="your_token"
    )
    
    # Initialize services
    llm = OpenAILLMService(
        api_key="your_openai_key",
        model="gpt-4"
    )
    
    tts = ElevenLabsTTSService(
        api_key="your_elevenlabs_key",
        voice_id="rachel"
    )
    
    stt = DeepgramSTTService(
        api_key="your_deepgram_key",
        model="nova-2"
    )
    
    # Create pipeline
    pipeline = Pipeline([
        transport.input(),              # Audio input from transport
        stt,                           # Speech-to-text
        llm,                           # Language model processing
        tts,                           # Text-to-speech
        transport.output()             # Audio output to transport
    ])
    
    return pipeline, transport
```

## Pipecat-Flows State Machine

### FlowManager Architecture

```python
from pipecat_flows import FlowManager, FlowConfig, NodeConfig

class FlowManager(FrameProcessor):
    def __init__(
        self,
        *,
        task: PipelineTask,
        llm: LLMService,
        context_aggregator: ContextAggregator,
        flow_config: FlowConfig | None = None
    ):
        super().__init__()
        self.task = task
        self.llm = llm
        self.context_aggregator = context_aggregator
        
        # Flow configuration
        if flow_config:
            self.nodes = flow_config["nodes"]
            self.initial_node = flow_config["initial_node"]
            self.mode = "static"
        else:
            self.nodes = {}
            self.mode = "dynamic"
            
        # State management
        self.state = {}
        self.current_node = None
        self.current_functions = set()

    async def set_node(self, node_name: str, node_config: NodeConfig = None):
        """Set current node and configure functions"""
        
        if node_config:
            self.nodes[node_name] = node_config
        elif node_name not in self.nodes:
            raise ValueError(f"Node '{node_name}' not found")
            
        self.current_node = node_name
        node = self.nodes[node_name]
        
        # Set up functions for this node
        self.current_functions = set()
        if "functions" in node:
            for func in node["functions"]:
                self.current_functions.add(func)
```

### Function Definition Types

#### 1. FlowsFunctionSchema (Recommended)

```python
from pipecat_flows.types import FlowsFunctionSchema, FlowArgs, FlowResult

select_size_function = FlowsFunctionSchema(
    name="select_pizza_size",
    description="Select the size of pizza the customer wants",
    properties={
        "size": {
            "type": "string",
            "enum": ["small", "medium", "large"],
            "description": "Pizza size selection"
        }
    },
    required=["size"],
    handler=handle_size_selection,
    transition_to="toppings_node"
)

async def handle_size_selection(args: FlowArgs) -> FlowResult:
    size = args["size"]
    return {
        "status": "success",
        "data": {"selected_size": size}
    }
```

#### 2. Direct Functions with Auto-Schema

```python
async def book_appointment(
    flow_manager: FlowManager,
    service_type: str,
    date: str,
    time: str,
    customer_name: str
) -> tuple[FlowResult, NodeConfig]:
    """Book an appointment and move to confirmation."""
    
    try:
        appointment = await create_appointment_external_api(
            service=service_type,
            datetime=f"{date} {time}",
            customer=customer_name
        )
        
        result = {
            "status": "success",
            "data": {"appointment_id": appointment.id}
        }
        
        next_node = {
            "name": "appointment_confirmation",
            "task_messages": [
                {
                    "role": "system",
                    "content": f"Confirm appointment details for {customer_name}"
                }
            ]
        }
        
        return result, next_node
        
    except Exception as e:
        result = {"status": "error", "message": str(e)}
        return result, None
```

## Transport Integrations

### Daily Transport

```python
from pipecat.transports.daily import DailyTransport

class DailyTransport(FrameProcessor):
    """Daily.co video calling platform integration"""
    
    def __init__(
        self,
        room_url: str,
        token: str,
        bot_name: str = "Assistant",
        camera_enabled: bool = False
    ):
        super().__init__()
        self.room_url = room_url
        self.token = token
        self.bot_name = bot_name
        self.camera_enabled = camera_enabled
        
    async def run(self):
        """Connect to Daily room and start processing"""
        
        self.client = daily.CallClient()
        
        # Set up event handlers
        self.client.on("joined", self._on_joined)
        self.client.on("participant-joined", self._on_participant_joined)
        self.client.on("audio-data", self._on_audio_data)
        
        # Join room
        await self.client.join(
            url=self.room_url,
            token=self.token,
            client_settings={
                "username": self.bot_name,
                "camera": self.camera_enabled
            }
        )
        
    async def _on_audio_data(self, data):
        """Handle incoming audio data"""
        
        audio_frame = AudioFrame(
            data=data["audio_data"],
            sample_rate=data["sample_rate"], 
            num_channels=data["num_channels"]
        )
        
        await self.push_frame(audio_frame, FrameDirection.DOWNSTREAM)
```

### LiveKit Transport

```python
from pipecat.transports.livekit import LiveKitTransport
import livekit

class LiveKitTransport(FrameProcessor):
    """LiveKit real-time communication platform integration"""
    
    def __init__(
        self,
        url: str,
        token: str,
        room_name: str,
        participant_name: str = "AI Assistant"
    ):
        super().__init__()
        self.url = url
        self.token = token
        self.room_name = room_name
        self.participant_name = participant_name
        
    async def run(self):
        """Connect to LiveKit room"""
        
        self.room = livekit.Room()
        
        # Set up event handlers
        self.room.on("participant_connected", self._on_participant_connected)
        self.room.on("track_subscribed", self._on_track_subscribed)
        
        # Connect to room
        await self.room.connect(self.url, self.token)
        
    async def _on_track_subscribed(self, track, publication, participant):
        """Handle track subscription"""
        
        if track.kind == livekit.TrackKind.KIND_AUDIO:
            async for audio_frame in track:
                pipecat_frame = AudioFrame(
                    data=audio_frame.data,
                    sample_rate=audio_frame.sample_rate,
                    num_channels=audio_frame.num_channels
                )
                await self.push_frame(pipecat_frame, FrameDirection.DOWNSTREAM)
```

## Service Integrations

### LLM Services

```python
from pipecat.services.openai import OpenAILLMService

class OpenAILLMService(LLMProcessor):
    """OpenAI language model integration"""
    
    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gpt-4",
        base_url: str = None,
        temperature: float = 0.7
    ):
        super().__init__()
        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model
        self.temperature = temperature
        
    async def process_llm_messages(
        self, 
        frame: LLMMessagesFrame, 
        direction: FrameDirection
    ):
        """Process LLM messages and generate response"""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=frame.messages,
                temperature=self.temperature,
                functions=frame.functions if hasattr(frame, 'functions') else None,
                function_call="auto" if hasattr(frame, 'functions') else None,
                stream=True
            )
            
            async for chunk in response:
                if chunk.choices[0].delta.content:
                    # Generate text frame
                    text_frame = TextFrame(chunk.choices[0].delta.content)
                    await self.push_frame(text_frame, direction)
                    
                elif chunk.choices[0].delta.function_call:
                    # Generate function call frame
                    func_call = chunk.choices[0].delta.function_call
                    if func_call.name and func_call.arguments:
                        call_frame = FunctionCallFrame(
                            name=func_call.name,
                            arguments=json.loads(func_call.arguments),
                            call_id=chunk.id
                        )
                        await self.push_frame(call_frame, direction)
                        
        except Exception as e:
            logger.exception(f"LLM processing error: {e}")
            error_frame = TextFrame("I apologize, but I'm having trouble processing that request.")
            await self.push_frame(error_frame, direction)
```

### TTS Services

```python
from pipecat.services.elevenlabs import ElevenLabsTTSService

class ElevenLabsTTSService(FrameProcessor):
    """ElevenLabs text-to-speech integration"""
    
    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        model: str = "eleven_turbo_v2",
        stability: float = 0.7,
        similarity_boost: float = 0.8
    ):
        super().__init__()
        self.client = elevenlabs.AsyncElevenLabs(api_key=api_key)
        self.voice_id = voice_id
        self.model = model
        self.voice_settings = elevenlabs.VoiceSettings(
            stability=stability,
            similarity_boost=similarity_boost
        )
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, TextFrame):
            await self._synthesize_speech(frame, direction)
        else:
            await self.push_frame(frame, direction)
            
    async def _synthesize_speech(self, frame: TextFrame, direction: FrameDirection):
        """Convert text to speech"""
        
        try:
            audio_generator = await self.client.generate(
                text=frame.text,
                voice=self.voice_id,
                model=self.model,
                voice_settings=self.voice_settings,
                stream=True
            )
            
            async for audio_chunk in audio_generator:
                audio_frame = AudioFrame(
                    data=audio_chunk,
                    sample_rate=22050,  # ElevenLabs default
                    num_channels=1
                )
                await self.push_frame(audio_frame, direction)
                
        except Exception as e:
            logger.exception(f"TTS error: {e}")
```

### STT Services

```python
from pipecat.services.deepgram import DeepgramSTTService

class DeepgramSTTService(FrameProcessor):
    """Deepgram speech-to-text integration"""
    
    def __init__(
        self,
        *,
        api_key: str,
        model: str = "nova-2",
        language: str = "en-US",
        interim_results: bool = True
    ):
        super().__init__()
        self.client = deepgram.DeepgramClient(api_key=api_key)
        self.model = model
        self.language = language
        self.interim_results = interim_results
        
        # WebSocket connection
        self.websocket = None
        
    async def start(self):
        """Initialize STT connection"""
        
        options = deepgram.LiveOptions(
            model=self.model,
            language=self.language,
            interim_results=self.interim_results,
            smart_format=True,
            endpointing=True
        )
        
        self.websocket = await self.client.listen.asynclive.v("1").start(
            options,
            on_message=self._on_message,
            on_error=self._on_error
        )
        
    async def _on_message(self, result):
        """Handle STT results"""
        
        if result.is_final:
            transcript = result.channel.alternatives[0].transcript
            if transcript.strip():
                # Generate user message frame
                message_frame = UserMessageFrame(
                    content=transcript,
                    timestamp=time.time()
                )
                await self.push_frame(message_frame, FrameDirection.DOWNSTREAM)
                
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, AudioFrame):
            # Send audio to Deepgram
            if self.websocket:
                await self.websocket.send(frame.data)
        else:
            await self.push_frame(frame, direction)
```

## Advanced Flow Patterns

### Multi-Agent Orchestration

```python
class MultiAgentFlowManager(FlowManager):
    """Manage multiple specialized agents"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.agents = {
            "customer_service": CustomerServiceAgent(),
            "technical_support": TechnicalSupportAgent(),
            "sales": SalesAgent(),
            "billing": BillingAgent()
        }
        self.current_agent = "customer_service"
        
    async def route_to_agent(self, agent_name: str, context: dict = None):
        """Route conversation to specialized agent"""
        
        if agent_name not in self.agents:
            raise ValueError(f"Unknown agent: {agent_name}")
            
        # Prepare handoff context
        handoff_context = {
            "previous_agent": self.current_agent,
            "conversation_summary": await self._summarize_conversation(),
            "customer_context": self.state.get("customer", {}),
            "additional_context": context or {}
        }
        
        # Switch to new agent
        self.current_agent = agent_name
        agent = self.agents[agent_name]
        
        # Configure agent's initial node
        initial_node = await agent.get_initial_node(handoff_context)
        await self.set_node(f"{agent_name}_entry", initial_node)
```

## Performance Optimization

```python
class OptimizedFlowManager(FlowManager):
    """Performance-optimized flow manager"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Caching
        self.response_cache = TTLCache(maxsize=1000, ttl=300)  # 5 minute TTL
        self.function_cache = TTLCache(maxsize=500, ttl=60)    # 1 minute TTL
        
        # Rate limiting
        self.rate_limiter = AsyncRateLimiter(max_calls=100, time_window=60)
        
        # Metrics
        self.metrics = MetricsCollector()
        
    async def _generate_llm_response(self, node: NodeConfig):
        """Optimized LLM response generation"""
        
        # Check cache first
        cache_key = self._build_cache_key(node)
        cached_response = self.response_cache.get(cache_key)
        
        if cached_response:
            self.metrics.record("cache_hit")
            await self._send_cached_response(cached_response)
            return
            
        # Rate limiting
        async with self.rate_limiter:
            # Generate new response
            start_time = time.time()
            
            try:
                await super()._generate_llm_response(node)
                
                # Record metrics
                response_time = time.time() - start_time
                self.metrics.record("llm_response_time", response_time)
                self.metrics.record("llm_success")
                
            except Exception as e:
                self.metrics.record("llm_error")
                raise
```

This comprehensive guide covers the core architecture, pipeline framework, flow orchestration, transport integrations, and performance optimization patterns for Pipecat and Pipecat-Flows, providing developers with the technical depth needed to build sophisticated voice applications.
