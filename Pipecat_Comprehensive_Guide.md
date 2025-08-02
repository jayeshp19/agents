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

class VideoFrame(Frame):
    """Video data frame"""
    def __init__(self, image: PIL.Image, timestamp: float):
        self.image = image
        self.timestamp = timestamp

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

class FunctionResultFrame(Frame):
    """Function call result"""
    def __init__(self, result: any, call_id: str):
        self.result = result
        self.call_id = call_id
```

### Processor Base Classes

All processing logic inherits from base processor classes:

```python
class FrameProcessor:
    """Base processor class"""
    
    def __init__(self, name: str = None):
        self._name = name or self.__class__.__name__
        self._sink: FrameProcessor | None = None
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process a single frame"""
        # Override in subclasses
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

class LLMProcessor(FrameProcessor):
    """Base class for LLM processing"""
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, LLMMessagesFrame):
            await self.process_llm_messages(frame, direction)
        else:
            await self.push_frame(frame, direction)
            
    async def process_llm_messages(self, frame: LLMMessagesFrame, direction: FrameDirection):
        """Override to implement LLM processing"""
        await self.push_frame(frame, direction)
```

## Pipeline Framework

### Pipeline Construction

Pipelines are constructed by linking processors together:

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

# Run pipeline
async def main():
    pipeline, transport = await create_pipeline()
    
    # Set up transport event handlers
    @transport.event_handler("on_first_participant_joined")
    async def on_participant_joined(transport, participant):
        # Start pipeline when participant joins
        task = PipelineTask(pipeline)
        await task.run()
    
    # Join the session
    await transport.run()
```

### Advanced Pipeline Patterns

#### Parallel Processing

```python
class ParallelProcessor(FrameProcessor):
    """Process frames through multiple parallel paths"""
    
    def __init__(self, processors: list[FrameProcessor]):
        super().__init__()
        self.processors = processors
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # Send frame to all processors in parallel
        tasks = []
        for processor in self.processors:
            task = asyncio.create_task(
                processor.process_frame(frame, direction)
            )
            tasks.append(task)
        
        # Wait for all to complete
        await asyncio.gather(*tasks)

# Usage
sentiment_analyzer = SentimentAnalysisProcessor()
keyword_extractor = KeywordExtractionProcessor()
context_manager = ContextManagementProcessor()

parallel_processor = ParallelProcessor([
    sentiment_analyzer,
    keyword_extractor,
    context_manager
])
```

#### Conditional Processing

```python
class ConditionalProcessor(FrameProcessor):
    """Route frames based on conditions"""
    
    def __init__(self, condition_fn, true_processor, false_processor):
        super().__init__()
        self.condition_fn = condition_fn
        self.true_processor = true_processor
        self.false_processor = false_processor
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if self.condition_fn(frame):
            await self.true_processor.process_frame(frame, direction)
        else:
            await self.false_processor.process_frame(frame, direction)

# Usage - route based on message length
def is_long_message(frame):
    if isinstance(frame, UserMessageFrame):
        return len(frame.content.split()) > 50
    return False

detailed_processor = DetailedResponseProcessor()
quick_processor = QuickResponseProcessor()

conditional = ConditionalProcessor(
    condition_fn=is_long_message,
    true_processor=detailed_processor,
    false_processor=quick_processor
)
```

#### Frame Filtering and Transformation

```python
class FilterProcessor(FrameProcessor):
    """Filter frames based on criteria"""
    
    def __init__(self, filter_fn):
        super().__init__()
        self.filter_fn = filter_fn
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if self.filter_fn(frame):
            await self.push_frame(frame, direction)
        # Else discard frame

class TransformProcessor(FrameProcessor):
    """Transform frames"""
    
    def __init__(self, transform_fn):
        super().__init__()
        self.transform_fn = transform_fn
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        transformed_frame = self.transform_fn(frame)
        await self.push_frame(transformed_frame, direction)

# Usage
# Filter out silence frames
silence_filter = FilterProcessor(
    lambda frame: not (isinstance(frame, AudioFrame) and is_silence(frame.data))
)

# Transform text to uppercase
uppercase_transform = TransformProcessor(
    lambda frame: TextFrame(frame.text.upper()) 
    if isinstance(frame, TextFrame) else frame
)
```

## Pipecat-Flows State Machine

### FlowManager Architecture

The FlowManager orchestrates conversation flows using a state machine approach:

```python
from pipecat_flows import FlowManager, FlowConfig, NodeConfig
from pipecat_flows.types import FlowArgs, FlowResult
from pipecat_flows.adapters import create_adapter

class FlowManager(FrameProcessor):
    def __init__(
        self,
        *,
        task: PipelineTask,
        llm: LLMService,
        context_aggregator: ContextAggregator,
        flow_config: FlowConfig | None = None,
        context_strategy: ContextStrategyConfig = None
    ):
        super().__init__()
        self.task = task
        self.llm = llm
        self.context_aggregator = context_aggregator
        
        # Create LLM-specific adapter
        self.adapter = create_adapter(llm)
        
        # Flow configuration
        if flow_config:
            self.nodes = flow_config["nodes"]
            self.initial_node = flow_config["initial_node"]
            self.mode = "static"
        else:
            self.nodes = {}
            self.mode = "dynamic"
            
        # State management
        self.state = {}                    # Shared state across nodes
        self.current_node = None
        self.current_functions = set()
        
        # Context strategy
        self.context_strategy = context_strategy or ContextStrategyConfig()
        
        # Action manager for side effects
        self.action_manager = ActionManager(task, flow_manager=self)

    async def initialize(self):
        """Initialize flow manager"""
        if self.mode == "static" and self.initial_node:
            await self.set_node(self.initial_node)

    async def set_node(self, node_name: str, node_config: NodeConfig = None):
        """Set current node and configure functions"""
        
        if node_config:
            # Dynamic node configuration
            self.nodes[node_name] = node_config
        elif node_name not in self.nodes:
            raise ValueError(f"Node '{node_name}' not found")
            
        self.current_node = node_name
        node = self.nodes[node_name]
        
        # Update context strategy for this node
        if "context_strategy" in node:
            self.context_strategy = node["context_strategy"]
            
        # Set up functions for this node
        self.current_functions = set()
        if "functions" in node:
            for func in node["functions"]:
                self.current_functions.add(func)
                
        # Execute pre-actions
        if "pre_actions" in node:
            for action in node["pre_actions"]:
                await self.action_manager.execute_action(action)
                
        # Respond immediately if configured
        if node.get("respond_immediately", False):
            await self._generate_llm_response(node)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames through flow logic"""
        
        if isinstance(frame, UserMessageFrame):
            await self._handle_user_message(frame)
        elif isinstance(frame, FunctionCallFrame):
            await self._handle_function_call(frame)
        else:
            await self.push_frame(frame, direction)

    async def _handle_user_message(self, frame: UserMessageFrame):
        """Handle user input and generate response"""
        
        # Add to context
        await self.context_aggregator.process_frame(frame, FrameDirection.DOWNSTREAM)
        
        # Generate LLM response if we have a current node
        if self.current_node:
            node = self.nodes[self.current_node]
            await self._generate_llm_response(node)

    async def _generate_llm_response(self, node: NodeConfig):
        """Generate LLM response for current node"""
        
        # Build context
        context = await self._build_context(node)
        
        # Get available functions
        functions = [f for f in self.current_functions]
        
        # Create LLM messages frame
        llm_frame = LLMMessagesFrame(
            messages=context,
            functions=functions,
            context_strategy=self.context_strategy
        )
        
        # Send to LLM
        await self.push_frame(llm_frame, FrameDirection.DOWNSTREAM)

    async def _build_context(self, node: NodeConfig) -> list[dict]:
        """Build conversation context for LLM"""
        
        context = []
        
        # Add role messages (persona/system)
        if "role_messages" in node:
            context.extend(node["role_messages"])
            
        # Get conversation history from context aggregator
        history = await self.context_aggregator.get_context()
        context.extend(history)
        
        # Add task messages (current objective)
        if "task_messages" in node:
            context.extend(node["task_messages"])
            
        return context

    async def _handle_function_call(self, frame: FunctionCallFrame):
        """Handle function call execution"""
        
        function_name = frame.name
        
        # Find function in current node
        function = None
        for func in self.current_functions:
            if func.name == function_name:
                function = func
                break
                
        if not function:
            # Function not available in current node
            error_result = FunctionResultFrame(
                result={"error": f"Function {function_name} not available"},
                call_id=frame.call_id
            )
            await self.push_frame(error_result, FrameDirection.DOWNSTREAM)
            return
            
        try:
            # Execute function
            if hasattr(function, 'handler'):
                # FlowsFunctionSchema with handler
                result = await function.handler(FlowArgs(frame.arguments))
                
                # Check for node transition
                if hasattr(function, 'transition_to') and function.transition_to:
                    await self.set_node(function.transition_to)
                    
            else:
                # Direct function call
                if asyncio.iscoroutinefunction(function):
                    result = await function(self, **frame.arguments)
                else:
                    result = function(self, **frame.arguments)
                    
                # Handle tuple result (result, next_node)
                if isinstance(result, tuple) and len(result) == 2:
                    function_result, next_node = result
                    
                    if isinstance(next_node, dict):
                        # Dynamic node configuration
                        node_name = next_node.get("name", f"dynamic_{int(time.time())}")
                        await self.set_node(node_name, next_node)
                    elif isinstance(next_node, str):
                        # Static node reference
                        await self.set_node(next_node)
                        
                    result = function_result
                    
            # Create result frame
            result_frame = FunctionResultFrame(
                result=result,
                call_id=frame.call_id
            )
            
            await self.push_frame(result_frame, FrameDirection.DOWNSTREAM)
            
            # Execute post-actions if configured
            if self.current_node:
                node = self.nodes[self.current_node]
                if "post_actions" in node:
                    for action in node["post_actions"]:
                        await self.action_manager.execute_action(action)
                        
        except Exception as e:
            # Handle function execution error
            error_result = FunctionResultFrame(
                result={"error": str(e)},
                call_id=frame.call_id
            )
            await self.push_frame(error_result, FrameDirection.DOWNSTREAM)
```

### Function Definition Patterns

#### 1. FlowsFunctionSchema (Recommended)

```python
from pipecat_flows.types import FlowsFunctionSchema, FlowArgs, FlowResult

# Edge function (transitions to specific node)
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
    """Handle pizza size selection"""
    size = args["size"]
    
    # Store in state
    flow_manager.state["pizza_size"] = size
    
    # Return success result
    return {
        "status": "success",
        "data": {"selected_size": size},
        "message": f"Great! You've selected a {size} pizza."
    }

# Non-edge function (stays in same node)
add_topping_function = FlowsFunctionSchema(
    name="add_pizza_topping",
    description="Add a topping to the current pizza",
    properties={
        "topping": {
            "type": "string",
            "description": "Name of the topping to add"
        }
    },
    required=["topping"],
    handler=handle_add_topping
    # No transition_to - stays in current node
)

async def handle_add_topping(args: FlowArgs) -> FlowResult:
    """Handle adding pizza topping"""
    topping = args["topping"]
    
    # Get current toppings from state
    toppings = flow_manager.state.get("pizza_toppings", [])
    toppings.append(topping)
    flow_manager.state["pizza_toppings"] = toppings
    
    return {
        "status": "success",
        "data": {"current_toppings": toppings},
        "message": f"Added {topping}. Current toppings: {', '.join(toppings)}"
    }
```

#### 2. Direct Functions with Auto-Schema

```python
async def book_appointment(
    flow_manager: FlowManager,
    service_type: str,
    date: str,
    time: str,
    customer_name: str,
    customer_phone: str = ""
) -> tuple[FlowResult, NodeConfig]:
    """Book an appointment and move to confirmation.
    
    Args:
        service_type: Type of service (consultation, repair, maintenance)
        date: Appointment date in YYYY-MM-DD format
        time: Appointment time in HH:MM format
        customer_name: Customer's full name
        customer_phone: Customer's phone number (optional)
        
    Returns:
        Tuple of (result, next_node_config)
    """
    
    try:
        # Validate inputs
        if not all([service_type, date, time, customer_name]):
            return {
                "status": "error",
                "message": "Missing required information for booking"
            }, None
            
        # Create appointment
        appointment = await create_appointment_external_api(
            service=service_type,
            datetime=f"{date} {time}",
            customer=customer_name,
            phone=customer_phone
        )
        
        # Store appointment info in state
        flow_manager.state["appointment"] = {
            "id": appointment.id,
            "service": service_type,
            "datetime": f"{date} {time}",
            "customer": customer_name,
            "confirmation": appointment.confirmation_number
        }
        
        # Result
        result = {
            "status": "success",
            "data": {
                "appointment_id": appointment.id,
                "confirmation_number": appointment.confirmation_number
            },
            "message": f"Appointment booked! Your confirmation number is {appointment.confirmation_number}"
        }
        
        # Next node configuration
        next_node = {
            "name": "appointment_confirmation",
            "task_messages": [
                {
                    "role": "system",
                    "content": f"Confirm the appointment details: {service_type} on {date} at {time} for {customer_name}. Provide next steps and ask if they need anything else."
                }
            ],
            "functions": [send_confirmation_email, reschedule_appointment],
            "context_strategy": {
                "strategy": "append"
            }
        }
        
        return result, next_node
        
    except Exception as e:
        # Handle booking error
        result = {
            "status": "error",
            "message": f"Sorry, I couldn't book that appointment: {str(e)}. Let me try a different approach."
        }
        
        # Return to booking node with error context
        error_node = {
            "name": "booking_retry",
            "task_messages": [
                {
                    "role": "system",
                    "content": "There was an issue with the booking. Help the customer try again with different details or offer alternative solutions."
                }
            ],
            "functions": [book_appointment, check_alternative_times]
        }
        
        return result, error_node

async def check_alternative_times(
    flow_manager: FlowManager,
    service_type: str,
    preferred_date: str
) -> tuple[FlowResult, NodeConfig]:
    """Check for alternative appointment times"""
    
    # Query available times
    alternatives = await get_available_times(service_type, preferred_date)
    
    if alternatives:
        flow_manager.state["alternative_times"] = alternatives
        
        result = {
            "status": "success",
            "data": {"alternatives": alternatives},
            "message": f"I found these alternative times: {', '.join(alternatives)}"
        }
        
        # Move to selection node
        next_node = {
            "name": "select_alternative",
            "task_messages": [
                {
                    "role": "system",
                    "content": "Present the alternative appointment times and help the customer choose one."
                }
            ],
            "functions": [confirm_alternative_time]
        }
        
    else:
        result = {
            "status": "error",
            "message": "No alternative times available for that date."
        }
        next_node = None  # Stay in current node
        
    return result, next_node
```

#### 3. Advanced Function Patterns

```python
# Function with conditional transitions
async def process_payment(
    flow_manager: FlowManager,
    payment_method: str,
    amount: float
) -> tuple[FlowResult, NodeConfig]:
    """Process payment with conditional next steps"""
    
    try:
        payment_result = await charge_payment(payment_method, amount)
        
        if payment_result.success:
            # Payment succeeded
            flow_manager.state["payment"] = {
                "transaction_id": payment_result.transaction_id,
                "amount": amount,
                "method": payment_method,
                "status": "completed"
            }
            
            result = {
                "status": "success",
                "data": {"transaction_id": payment_result.transaction_id},
                "message": "Payment processed successfully!"
            }
            
            # Determine next node based on payment amount
            if amount > 1000:
                # High-value transaction - extra confirmation
                next_node = {
                    "name": "premium_confirmation",
                    "task_messages": [
                        {
                            "role": "system",
                            "content": "This was a high-value transaction. Provide premium confirmation and next steps."
                        }
                    ],
                    "functions": [send_premium_receipt, schedule_followup]
                }
            else:
                # Standard transaction
                next_node = {
                    "name": "standard_confirmation",
                    "task_messages": [
                        {
                            "role": "system",
                            "content": "Confirm the transaction and provide standard next steps."
                        }
                    ],
                    "functions": [send_receipt]
                }
                
        else:
            # Payment failed
            result = {
                "status": "error",
                "message": f"Payment failed: {payment_result.error_message}"
            }
            
            # Go to payment retry flow
            next_node = {
                "name": "payment_retry",
                "task_messages": [
                    {
                        "role": "system",
                        "content": "Payment failed. Help customer try again or offer alternative payment methods."
                    }
                ],
                "functions": [process_payment, try_different_method]
            }
            
        return result, next_node
        
    except Exception as e:
        # Payment processing error
        result = {
            "status": "error",
            "message": "Payment system temporarily unavailable. Please try again later."
        }
        
        # Go to error handling node
        error_node = {
            "name": "payment_error",
            "task_messages": [
                {
                    "role": "system",
                    "content": "There was a technical issue with payment processing. Apologize and offer alternatives."
                }
            ],
            "functions": [retry_payment, contact_support]
        }
        
        return result, error_node

# Function with external API integration
async def lookup_customer_info(
    flow_manager: FlowManager,
    email: str,
    phone: str = ""
) -> tuple[FlowResult, NodeConfig]:
    """Look up customer information from CRM"""
    
    try:
        # Try email first
        customer = await crm_api.get_customer_by_email(email)
        
        if not customer and phone:
            # Try phone as backup
            customer = await crm_api.get_customer_by_phone(phone)
            
        if customer:
            # Customer found
            flow_manager.state["customer"] = {
                "id": customer.id,
                "name": customer.name,
                "email": customer.email,
                "phone": customer.phone,
                "tier": customer.tier,
                "history": customer.interaction_history
            }
            
            result = {
                "status": "success",
                "data": {"customer_name": customer.name, "tier": customer.tier},
                "message": f"Welcome back, {customer.name}!"
            }
            
            # Route based on customer tier
            if customer.tier == "premium":
                next_node = {
                    "name": "premium_service",
                    "task_messages": [
                        {
                            "role": "system",
                            "content": f"This is a premium customer ({customer.name}). Provide premium level service and prioritize their requests."
                        }
                    ],
                    "functions": [premium_support_functions()]
                }
            else:
                next_node = {
                    "name": "standard_service",
                    "task_messages": [
                        {
                            "role": "system",
                            "content": f"Provide standard customer service for {customer.name}."
                        }
                    ],
                    "functions": [standard_support_functions()]
                }
                
        else:
            # New customer
            result = {
                "status": "info",
                "message": "I don't see an existing account. Let me help you create one."
            }
            
            next_node = {
                "name": "new_customer_registration",
                "task_messages": [
                    {
                        "role": "system",
                        "content": "Help the new customer register an account. Collect necessary information."
                    }
                ],
                "functions": [create_customer_account, skip_registration]
            }
            
        return result, next_node
        
    except Exception as e:
        # CRM lookup error
        logger.exception(f"Customer lookup failed: {e}")
        
        result = {
            "status": "error",
            "message": "I'm having trouble accessing customer information right now."
        }
        
        # Continue without customer info
        next_node = {
            "name": "no_customer_info",
            "task_messages": [
                {
                    "role": "system",
                    "content": "Customer lookup failed. Provide service without customer history context."
                }
            ],
            "functions": [manual_customer_entry, continue_without_info]
        }
        
        return result, next_node
```

### Context Management Strategies

```python
from enum import Enum

class ContextStrategy(Enum):
    APPEND = "append"                    # Keep adding messages
    RESET = "reset"                     # Clear and start fresh  
    RESET_WITH_SUMMARY = "reset_with_summary"  # Summarize then reset

class ContextStrategyConfig:
    def __init__(
        self,
        strategy: ContextStrategy = ContextStrategy.APPEND,
        summary_prompt: str = None,
        max_context_length: int = 4000,
        preserve_system_messages: bool = True
    ):
        self.strategy = strategy
        self.summary_prompt = summary_prompt
        self.max_context_length = max_context_length
        self.preserve_system_messages = preserve_system_messages

class ContextAggregator(FrameProcessor):
    """Manages conversation context with configurable strategies"""
    
    def __init__(self, llm: LLMService, strategy: ContextStrategyConfig = None):
        super().__init__()
        self.llm = llm
        self.strategy = strategy or ContextStrategyConfig()
        self.messages = []
        self.system_messages = []
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, UserMessageFrame):
            await self._add_user_message(frame)
        elif isinstance(frame, BotMessageFrame):
            await self._add_bot_message(frame)
        elif isinstance(frame, LLMMessagesFrame):
            await self._apply_context_strategy(frame)
            
        await self.push_frame(frame, direction)
        
    async def _add_user_message(self, frame: UserMessageFrame):
        """Add user message to context"""
        message = {
            "role": "user",
            "content": frame.content,
            "timestamp": frame.timestamp
        }
        self.messages.append(message)
        
    async def _add_bot_message(self, frame: BotMessageFrame):
        """Add bot message to context"""
        message = {
            "role": "assistant", 
            "content": frame.content,
            "timestamp": frame.timestamp
        }
        self.messages.append(message)
        
    async def _apply_context_strategy(self, frame: LLMMessagesFrame):
        """Apply context management strategy"""
        
        # Check if context is too long
        total_length = self._estimate_token_count(self.messages)
        
        if total_length > self.strategy.max_context_length:
            if self.strategy.strategy == ContextStrategy.RESET:
                await self._reset_context()
            elif self.strategy.strategy == ContextStrategy.RESET_WITH_SUMMARY:
                await self._reset_with_summary()
                
        # Update frame with managed context
        frame.messages = self._build_final_context(frame.messages)
        
    async def _reset_context(self):
        """Reset context keeping only system messages"""
        if self.strategy.preserve_system_messages:
            self.messages = [msg for msg in self.messages if msg.get("role") == "system"]
        else:
            self.messages = []
            
    async def _reset_with_summary(self):
        """Reset context with summary of conversation"""
        
        if not self.strategy.summary_prompt:
            # Default summary prompt
            summary_prompt = "Summarize the key points and current state of this conversation:"
        else:
            summary_prompt = self.strategy.summary_prompt
            
        # Create summary
        summary_messages = [
            {"role": "system", "content": summary_prompt},
            *self.messages
        ]
        
        summary_response = await self.llm.chat(summary_messages)
        summary_text = summary_response.choices[0].message.content
        
        # Reset with summary
        summary_message = {
            "role": "system",
            "content": f"Previous conversation summary: {summary_text}",
            "timestamp": time.time()
        }
        
        if self.strategy.preserve_system_messages:
            system_msgs = [msg for msg in self.messages if msg.get("role") == "system"]
            self.messages = system_msgs + [summary_message]
        else:
            self.messages = [summary_message]
            
    def _build_final_context(self, additional_messages: list) -> list:
        """Build final context for LLM"""
        
        # Combine system messages + conversation history + additional messages
        final_context = []
        
        # Add preserved system messages
        if self.strategy.preserve_system_messages:
            final_context.extend(self.system_messages)
            
        # Add conversation history
        final_context.extend(self.messages)
        
        # Add additional messages from frame
        final_context.extend(additional_messages)
        
        return final_context
        
    def _estimate_token_count(self, messages: list) -> int:
        """Estimate token count for messages"""
        # Simple estimation - 4 characters per token
        total_chars = sum(len(str(msg.get("content", ""))) for msg in messages)
        return total_chars // 4
        
    async def get_context(self) -> list:
        """Get current conversation context"""
        return self.messages.copy()
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
        camera_enabled: bool = False,
        microphone_enabled: bool = True,
        speaker_enabled: bool = True
    ):
        super().__init__()
        self.room_url = room_url
        self.token = token
        self.bot_name = bot_name
        self.camera_enabled = camera_enabled
        self.microphone_enabled = microphone_enabled
        self.speaker_enabled = speaker_enabled
        
        # Daily client
        self.client = None
        self.participant_id = None
        
    async def run(self):
        """Connect to Daily room and start processing"""
        
        # Initialize Daily client
        self.client = daily.CallClient()
        
        # Set up event handlers
        self.client.on("joined", self._on_joined)
        self.client.on("participant-joined", self._on_participant_joined)
        self.client.on("participant-left", self._on_participant_left)
        self.client.on("app-message", self._on_app_message)
        self.client.on("audio-data", self._on_audio_data)
        
        # Join room
        await self.client.join(
            url=self.room_url,
            token=self.token,
            client_settings={
                "username": self.bot_name,
                "camera": self.camera_enabled,
                "microphone": self.microphone_enabled,
                "speaker": self.speaker_enabled
            }
        )
        
    async def _on_joined(self, data):
        """Handle successful room join"""
        self.participant_id = data["local"]["user_id"]
        print(f"Bot joined room as {self.participant_id}")
        
    async def _on_participant_joined(self, data):
        """Handle participant joining"""
        participant = data["participant"]
        print(f"Participant joined: {participant['user_id']}")
        
        # Emit event for pipeline
        await self.push_frame(
            ParticipantJoinedFrame(
                participant_id=participant["user_id"],
                participant_name=participant.get("username", "Guest")
            ),
            FrameDirection.DOWNSTREAM
        )
        
    async def _on_audio_data(self, data):
        """Handle incoming audio data"""
        
        # Convert Daily audio format to AudioFrame
        audio_frame = AudioFrame(
            data=data["audio_data"],
            sample_rate=data["sample_rate"], 
            num_channels=data["num_channels"]
        )
        
        # Send to pipeline
        await self.push_frame(audio_frame, FrameDirection.DOWNSTREAM)
        
    def input(self) -> 'DailyInputProcessor':
        """Get input processor for pipeline"""
        return DailyInputProcessor(self)
        
    def output(self) -> 'DailyOutputProcessor':
        """Get output processor for pipeline"""
        return DailyOutputProcessor(self)

class DailyInputProcessor(FrameProcessor):
    """Processes input from Daily transport"""
    
    def __init__(self, transport: DailyTransport):
        super().__init__()
        self.transport = transport
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # Daily input processor mainly receives frames from transport
        # and passes them downstream
        await self.push_frame(frame, direction)

class DailyOutputProcessor(FrameProcessor):
    """Processes output to Daily transport"""
    
    def __init__(self, transport: DailyTransport):
        super().__init__()
        self.transport = transport
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, AudioFrame):
            # Send audio to Daily
            await self.transport.client.send_audio(frame.data)
        elif isinstance(frame, TextFrame):
            # Send text as app message
            await self.transport.client.send_app_message({
                "type": "text",
                "content": frame.text
            })
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
        
        # LiveKit room and participant
        self.room = None
        self.local_participant = None
        
    async def run(self):
        """Connect to LiveKit room"""
        
        # Create room instance
        self.room = livekit.Room()
        
        # Set up event handlers
        self.room.on("participant_connected", self._on_participant_connected)
        self.room.on("participant_disconnected", self._on_participant_disconnected)
        self.room.on("track_published", self._on_track_published)
        self.room.on("track_subscribed", self._on_track_subscribed)
        
        # Connect to room
        await self.room.connect(self.url, self.token)
        
        # Set up local participant
        self.local_participant = self.room.local_participant
        await self.local_participant.set_name(self.participant_name)
        
    async def _on_participant_connected(self, participant):
        """Handle participant connection"""
        print(f"Participant connected: {participant.name}")
        
        # Emit event
        await self.push_frame(
            ParticipantJoinedFrame(
                participant_id=participant.sid,
                participant_name=participant.name
            ),
            FrameDirection.DOWNSTREAM
        )
        
    async def _on_track_subscribed(self, track, publication, participant):
        """Handle track subscription"""
        
        if track.kind == livekit.TrackKind.KIND_AUDIO:
            # Set up audio processing
            async for audio_frame in track:
                pipecat_frame = AudioFrame(
                    data=audio_frame.data,
                    sample_rate=audio_frame.sample_rate,
                    num_channels=audio_frame.num_channels
                )
                await self.push_frame(pipecat_frame, FrameDirection.DOWNSTREAM)
```

### Local Audio Transport

```python
from pipecat.transports.local import LocalTransport
import pyaudio
import numpy as np

class LocalTransport(FrameProcessor):
    """Local audio input/output using PyAudio"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 1024,
        input_device: int = None,
        output_device: int = None
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.input_device = input_device
        self.output_device = output_device
        
        # PyAudio instance
        self.audio = pyaudio.PyAudio()
        self.input_stream = None
        self.output_stream = None
        
    async def run(self):
        """Start local audio processing"""
        
        # Set up input stream
        self.input_stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.input_device,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._input_callback
        )
        
        # Set up output stream
        self.output_stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            output=True,
            output_device_index=self.output_device,
            frames_per_buffer=self.chunk_size
        )
        
        # Start streams
        self.input_stream.start_stream()
        self.output_stream.start_stream()
        
    def _input_callback(self, in_data, frame_count, time_info, status):
        """Handle audio input"""
        
        # Convert to AudioFrame
        audio_frame = AudioFrame(
            data=in_data,
            sample_rate=self.sample_rate,
            num_channels=self.channels
        )
        
        # Send to pipeline (in background task)
        asyncio.create_task(
            self.push_frame(audio_frame, FrameDirection.DOWNSTREAM)
        )
        
        return (None, pyaudio.paContinue)
        
    def input(self) -> 'LocalInputProcessor':
        return LocalInputProcessor(self)
        
    def output(self) -> 'LocalOutputProcessor':
        return LocalOutputProcessor(self)

class LocalOutputProcessor(FrameProcessor):
    """Process output audio to local speakers"""
    
    def __init__(self, transport: LocalTransport):
        super().__init__()
        self.transport = transport
        self.audio_queue = asyncio.Queue()
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, AudioFrame):
            # Add to output queue
            await self.audio_queue.put(frame.data)
            
            # Play audio
            self.transport.output_stream.write(frame.data)
```

This comprehensive documentation covers the core architecture, pipeline framework, flow orchestration, and transport integrations of Pipecat and Pipecat-Flows, providing developers with detailed technical understanding for building sophisticated voice applications. 