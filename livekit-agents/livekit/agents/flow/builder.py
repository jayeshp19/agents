"""
Flow builder utilities for programmatic flow construction.

This module provides the FlowBuilder class which offers a fluent API
for creating conversation flows programmatically, with comprehensive
validation and error checking. Ideal for testing, dynamic flow generation,
and programmatic flow construction.

Key Features:
    - Fluent API with method chaining
    - Comprehensive validation at each step
    - Context manager support for automatic validation
    - Rich error messages with actionable feedback
    - Support for all node types (conversation, function, gather_input, end)
    - Tool management and validation
    - Flow introspection capabilities

Example Usage:
    Basic flow creation:
    ```python
    flow = (FlowBuilder("customer_service", "You are a helpful customer service agent")
           .add_conversation_node("start", "Welcome", "Hello! How can I help?")
           .add_gather_input_node("contact", "Get Contact", [
               GatherInputVariable("email", "email", "Your email", required=True)
           ])
           .add_edge("start", "contact", "user needs to provide contact info")
           .set_start_node("start")
           .build())
    ```

    Using context manager for automatic validation:
    ```python
    with FlowBuilder("support_flow") as builder:
        builder.add_conversation_node("welcome", "Welcome", "Hi there!")
        builder.add_end_node("goodbye", "End", "Thanks for calling!")
        builder.add_edge("welcome", "goodbye", "conversation is complete")
        builder.set_start_node("welcome")
        # Flow is automatically validated on context exit
        flow = builder.build()
    ```

    Complex flow with function calls:
    ```python
    from .flow_spec import ToolSpec

    tool = ToolSpec(
        tool_id="check_status",
        name="check_order_status",
        description="Check order status",
        type="custom",
        parameters={"type": "object", "properties": {"order_id": {"type": "string"}}},
        url="https://api.example.com/orders",
        http_method="GET"
    )

    flow = (FlowBuilder("order_support")
           .add_tool(tool)
           .add_conversation_node("start", "Start", "What's your order ID?")
           .add_function_node("check", "Check Status", "check_status",
                            speak_during_execution=True)
           .add_conversation_node("result", "Show Result", "Here's your order status")
           .add_edge("start", "check", "user provided order ID")
           .add_edge("check", "result", "status retrieved successfully")
           .set_start_node("start")
           .build())
    ```
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .enums import NodeType
from .fields import Edge, GatherInputVariable, Instruction, TransitionCondition

if TYPE_CHECKING:
    from .flow_spec import FlowSpec, Node, ToolSpec

logger = logging.getLogger(__name__)


class FlowBuilder:
    """Builder pattern for creating conversation flows programmatically.

    The FlowBuilder provides a fluent API for constructing conversation flows
    step by step, with comprehensive validation and error checking at each step.
    It supports all node types, tool management, and flow validation.

    Features:
        - Fluent API with method chaining for readable flow construction
        - Comprehensive validation with detailed error messages
        - Context manager support for automatic validation
        - Support for all node types: conversation, function, gather_input, end
        - Tool specification management and validation
        - Flow introspection and debugging capabilities
        - Rich error handling with actionable feedback

    Thread Safety:
        FlowBuilder instances are not thread-safe. Each thread should use
        its own builder instance.

    Example:
        Basic usage with method chaining:
        ```python
        flow = (FlowBuilder("my_flow", "You are a helpful assistant")
               .add_conversation_node("start", "Welcome", "Hello! How can I help?")
               .add_gather_input_node("gather", "Get Info", [
                   GatherInputVariable("name", "string", "Your name", required=True),
                   GatherInputVariable("email", "email", "Your email", required=True)
               ])
               .add_function_node("process", "Process Data", "process_tool")
               .add_end_node("end", "Complete", "Thank you!")
               .add_edge("start", "gather", "user wants to provide info")
               .add_edge("gather", "process", "information collected")
               .add_edge("process", "end", "processing complete")
               .set_start_node("start")
               .build())
        ```

        Using context manager for automatic validation:
        ```python
        with FlowBuilder("customer_service") as builder:
            builder.add_conversation_node("welcome", "Welcome", "Hi there!")
            builder.set_start_node("welcome")
            # Validation happens automatically on exit
            flow = builder.build()
        ```

    Attributes:
        flow_id (str): Unique identifier for the flow
        global_prompt (str): Global instructions for the flow
        nodes (dict[str, Node]): Dictionary of nodes by ID
        tools (dict[str, ToolSpec]): Dictionary of tools by ID
        start_node_id (str): ID of the starting node
    """

    def __init__(self, flow_id: str, global_prompt: str = ""):
        """Initialize a new flow builder.

        Args:
            flow_id: Unique identifier for the flow. Must be non-empty and
                    should follow naming conventions (alphanumeric, underscores).
            global_prompt: Global instructions that apply to the entire flow.
                         If empty, defaults to a helpful assistant prompt.

        Raises:
            ValueError: If flow_id is empty or contains only whitespace.

        Example:
            ```python
            # Basic initialization
            builder = FlowBuilder("customer_support")

            # With custom global prompt
            builder = FlowBuilder(
                "technical_support",
                "You are a technical support specialist. Be precise and helpful."
            )
            ```
        """
        if not flow_id.strip():
            raise ValueError("Flow ID cannot be empty or contain only whitespace")

        self.flow_id = flow_id.strip()
        self.global_prompt = global_prompt.strip() or "You are a helpful assistant."
        self.nodes: dict[str, Node] = {}
        self.tools: dict[str, ToolSpec] = {}
        self.start_node_id: str = ""
        self._context_validation_enabled = False

    def add_conversation_node(
        self, node_id: str, name: str, instruction_text: str, instruction_type: str = "prompt"
    ) -> FlowBuilder:
        """Add a conversation node to the flow.

        Conversation nodes handle back-and-forth dialogue with users. They can
        use either static text (predetermined responses) or prompt-based
        instructions (dynamic AI-generated responses).

        Args:
            node_id: Unique identifier for the node. Must be non-empty and unique
                    within the flow. Recommended to use descriptive names like
                    "welcome", "gather_info", "confirm_order".
            name: Human-readable name for the node. Used for debugging and
                 flow visualization. Should be descriptive.
            instruction_text: The instruction or response text for the node.
                            For "prompt" type: instructions to the AI agent.
                            For "static_text" type: exact text to speak/display.
            instruction_type: Type of instruction, either "prompt" (default)
                            for AI-generated responses or "static_text" for
                            predetermined responses.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If node_id is empty, already exists, name is empty,
                       instruction_text is empty, or instruction_type is invalid.

        Example:
            ```python
            # AI-generated response based on prompt
            builder.add_conversation_node(
                "welcome",
                "Welcome Node",
                "Greet the user warmly and ask how you can help them today."
            )

            # Static predetermined response
            builder.add_conversation_node(
                "disclaimer",
                "Legal Disclaimer",
                "This call may be recorded for quality assurance.",
                instruction_type="static_text"
            )
            ```
        """
        node_id = self._validate_node_id(node_id)
        self._validate_node_name(name)
        if not instruction_text.strip():
            raise ValueError("Instruction text cannot be empty or contain only whitespace")
        if instruction_type not in ("prompt", "static_text"):
            raise ValueError(
                f"Invalid instruction type '{instruction_type}'. Must be 'prompt' or 'static_text'"
            )

        # Import here to avoid circular imports
        from .flow_spec import Node

        instruction = Instruction(type=instruction_type, text=instruction_text.strip())
        node = Node(
            id=node_id, name=name.strip(), type=NodeType.CONVERSATION, instruction=instruction
        )
        self.nodes[node_id] = node

        logger.debug(
            f"Added conversation node '{node_id}' with instruction type '{instruction_type}'"
        )
        return self

    def add_gather_input_node(
        self, node_id: str, name: str, variables: list[GatherInputVariable], instruction: str = ""
    ) -> FlowBuilder:
        r"""Add a gather input node to the flow.

        Gather input nodes collect structured information from users using
        a conversational interface. They validate input against specified
        patterns and types, ensuring data quality.

        Args:
            node_id: Unique identifier for the node. Should be descriptive
                    like "collect_contact", "get_order_details".
            name: Human-readable name for the node.
            variables: List of variables to collect from the user. Each variable
                      specifies name, type, description, and validation rules.
                      Must contain at least one variable.
            instruction: Optional instruction text explaining what information
                        is being collected and why. If empty, a default message
                        will be used.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If node_id is invalid, name is empty, or variables list
                       is empty.

        Example:
            ```python
            from .fields import GatherInputVariable

            contact_vars = [
                GatherInputVariable(
                    name="full_name",
                    type="string",
                    description="Your full name",
                    required=True,
                    regex_pattern=r"^[A-Za-z\s]{2,50}$",
                    regex_error_message="Name must be 2-50 characters, letters only"
                ),
                GatherInputVariable(
                    name="email",
                    type="email",
                    description="Your email address",
                    required=True
                ),
                GatherInputVariable(
                    name="phone",
                    type="phone",
                    description="Your phone number",
                    required=False
                )
            ]

            builder.add_gather_input_node(
                "contact_info",
                "Collect Contact Information",
                contact_vars,
                "I need to collect some contact information to help you better."
            )
            ```
        """
        node_id = self._validate_node_id(node_id)
        self._validate_node_name(name)
        if not variables:
            raise ValueError("Gather input nodes must have at least one variable")

        # Validate variable names are unique
        var_names = [var.name for var in variables]
        if len(var_names) != len(set(var_names)):
            raise ValueError("Variable names must be unique within a gather input node")

        # Import here to avoid circular imports
        from .flow_spec import Node

        node = Node(
            id=node_id,
            name=name.strip(),
            type=NodeType.GATHER_INPUT,
            gather_input_variables=variables,
            gather_input_instruction=instruction.strip() or None,
        )
        self.nodes[node_id] = node

        logger.debug(f"Added gather input node '{node_id}' with {len(variables)} variables")
        return self

    def add_function_node(
        self,
        node_id: str,
        name: str,
        tool_id: str,
        speak_during_execution: bool = False,
        wait_for_result: bool = True,
    ) -> FlowBuilder:
        """Add a function node to the flow.

        Function nodes execute external tools or API calls during the conversation.
        They can integrate with external systems for data retrieval, processing,
        or actions.

        Args:
            node_id: Unique identifier for the node. Should indicate the function
                    purpose like "check_order", "process_payment", "send_email".
            name: Human-readable name for the node.
            tool_id: ID of the tool to execute. The tool must be added to the flow
                    using add_tool() before or after creating this node.
            speak_during_execution: Whether the agent should speak while the
                                  function is executing. Useful for long-running
                                  operations to keep user engaged.
            wait_for_result: Whether to wait for the function to complete before
                           proceeding. Set to False for fire-and-forget operations.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If node_id is invalid, name is empty, or tool_id is empty.

        Note:
            The referenced tool_id does not need to exist when the node is created,
            but must exist when the flow is built. This allows flexible ordering
            of node and tool creation.

        Example:
            ```python
            # Function that speaks during execution (e.g., "Let me check that...")
            builder.add_function_node(
                "lookup_order",
                "Look Up Order Status",
                "order_lookup_tool",
                speak_during_execution=True,
                wait_for_result=True
            )

            # Background operation that doesn't block
            builder.add_function_node(
                "log_interaction",
                "Log Customer Interaction",
                "logging_tool",
                speak_during_execution=False,
                wait_for_result=False
            )
            ```
        """
        node_id = self._validate_node_id(node_id)
        self._validate_node_name(name)
        if not tool_id.strip():
            raise ValueError("Tool ID cannot be empty or contain only whitespace")

        # Import here to avoid circular imports
        from .flow_spec import Node

        node = Node(
            id=node_id,
            name=name.strip(),
            type=NodeType.FUNCTION,
            tool_id=tool_id.strip(),
            speak_during_execution=speak_during_execution,
            wait_for_result=wait_for_result,
        )
        self.nodes[node_id] = node

        logger.debug(f"Added function node '{node_id}' using tool '{tool_id}'")
        return self

    def add_end_node(self, node_id: str, name: str, farewell_text: str = "") -> FlowBuilder:
        """Add an end node to the flow.

        End nodes terminate the conversation flow. They can optionally include
        a farewell message to provide closure to the user.

        Args:
            node_id: Unique identifier for the node. Common names include
                    "end", "goodbye", "complete", "terminate".
            name: Human-readable name for the node.
            farewell_text: Optional farewell message to speak before ending.
                          If empty, the conversation ends without additional text.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If node_id is invalid or name is empty.

        Example:
            ```python
            # End with farewell message
            builder.add_end_node(
                "goodbye",
                "End Conversation",
                "Thank you for calling! Have a great day!"
            )

            # Silent end
            builder.add_end_node("terminate", "Silent End")
            ```
        """
        node_id = self._validate_node_id(node_id)
        self._validate_node_name(name)

        # Import here to avoid circular imports
        from .flow_spec import Node

        instruction = None
        if farewell_text.strip():
            instruction = Instruction(type="static_text", text=farewell_text.strip())

        node = Node(
            id=node_id,
            name=name.strip(),
            type=NodeType.END,
            instruction=instruction,
        )
        self.nodes[node_id] = node

        logger.debug(f"Added end node '{node_id}' with farewell: {bool(farewell_text)}")
        return self

    def add_edge(
        self, from_node_id: str, to_node_id: str, condition: str, edge_id: str | None = None
    ) -> FlowBuilder:
        """Add an edge between nodes.

        Edges define the flow transitions between nodes based on conditions.
        Conditions are evaluated by the AI to determine when to move from
        one node to another.

        Args:
            from_node_id: Source node ID. Must exist in the flow.
            to_node_id: Destination node ID. Does not need to exist yet,
                       allowing flexible construction order.
            condition: Human-readable condition description that tells the AI
                      when to follow this edge. Should be clear and specific.
            edge_id: Optional custom edge ID. If not provided, a default ID
                    will be generated from the node IDs.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If source node doesn't exist, condition is empty,
                       or edge_id conflicts with existing edge.

        Example:
            ```python
            # Simple condition
            builder.add_edge("start", "gather_info", "user wants to provide information")

            # More specific condition
            builder.add_edge(
                "payment",
                "success",
                "payment was processed successfully and confirmation received"
            )

            # Custom edge ID for tracking
            builder.add_edge(
                "verification",
                "approved",
                "user identity verified successfully",
                edge_id="verification_success"
            )
            ```
        """
        if from_node_id not in self.nodes:
            available_nodes = list(self.nodes.keys())
            raise ValueError(
                f"Source node '{from_node_id}' not found. Available nodes: {available_nodes}"
            )
        if not condition.strip():
            raise ValueError("Edge condition cannot be empty or contain only whitespace")

        actual_edge_id = edge_id or f"{from_node_id}_to_{to_node_id}"

        # Check for duplicate edge IDs within the source node
        existing_edge_ids = [edge.id for edge in self.nodes[from_node_id].edges]
        if actual_edge_id in existing_edge_ids:
            raise ValueError(f"Edge ID '{actual_edge_id}' already exists in node '{from_node_id}'")

        edge = Edge(
            id=actual_edge_id,
            condition=condition.strip(),
            transition_condition=TransitionCondition(type="prompt", prompt=condition.strip()),
            destination_node_id=to_node_id,
        )
        self.nodes[from_node_id].edges.append(edge)

        logger.debug(f"Added edge '{actual_edge_id}' from '{from_node_id}' to '{to_node_id}'")
        return self

    def add_tool(self, tool: ToolSpec) -> FlowBuilder:
        """Add a tool specification to the flow.

        Tools define external functions that can be called by function nodes.
        They specify the interface, parameters, and execution details for
        integrating with external systems.

        Args:
            tool: Tool specification containing all necessary details for
                 execution including parameters, URL, HTTP method, etc.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If tool ID already exists in the flow.

        Example:
            ```python
            from .flow_spec import ToolSpec

            order_tool = ToolSpec(
                tool_id="check_order_status",
                name="check_order",
                description="Check the status of a customer order",
                type="custom",
                parameters={
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "Customer order ID"
                        }
                    },
                    "required": ["order_id"]
                },
                url="https://api.company.com/orders",
                http_method="GET",
                headers={"Authorization": "Bearer token"},
                timeout_ms=10000
            )

            builder.add_tool(order_tool)
            ```
        """
        if tool.tool_id in self.tools:
            raise ValueError(
                f"Tool with ID '{tool.tool_id}' already exists. "
                f"Existing tools: {list(self.tools.keys())}"
            )

        self.tools[tool.tool_id] = tool

        logger.debug(f"Added tool '{tool.tool_id}' of type '{tool.type}'")
        return self

    def set_start_node(self, node_id: str) -> FlowBuilder:
        """Set the starting node for the flow.

        The start node is where conversation begins. Every flow must have
        exactly one start node defined.

        Args:
            node_id: ID of the node to start from. Must exist in the flow.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If node doesn't exist in the flow.

        Example:
            ```python
            builder.set_start_node("welcome")
            ```
        """
        if node_id not in self.nodes:
            available_nodes = list(self.nodes.keys())
            raise ValueError(
                f"Start node '{node_id}' not found. Available nodes: {available_nodes}"
            )

        self.start_node_id = node_id

        logger.debug(f"Set start node to '{node_id}'")
        return self

    def get_node_info(self) -> dict[str, dict]:
        """Get information about all nodes in the flow.

        Returns:
            Dictionary mapping node IDs to node information including
            type, name, edge count, and other relevant details.

        Example:
            ```python
            info = builder.get_node_info()
            for node_id, details in info.items():
                print(f"{node_id}: {details['type']} - {details['edges']} edges")
            ```
        """
        return {
            node_id: {
                "type": node.type,
                "name": node.name,
                "edges": len(node.edges),
                "has_instruction": node.instruction is not None,
                "tool_id": getattr(node, "tool_id", None),
                "variable_count": len(getattr(node, "gather_input_variables", [])),
            }
            for node_id, node in self.nodes.items()
        }

    def get_unreachable_nodes(self) -> list[str]:
        """Find nodes that cannot be reached from the start node.

        Returns:
            List of node IDs that are unreachable from the start node.
            An empty list indicates all nodes are reachable.

        Example:
            ```python
            unreachable = builder.get_unreachable_nodes()
            if unreachable:
                print(f"Warning: Unreachable nodes: {unreachable}")
            ```
        """
        if not self.start_node_id or self.start_node_id not in self.nodes:
            return list(self.nodes.keys())

        visited = set()
        to_visit = [self.start_node_id]

        while to_visit:
            current = to_visit.pop()
            if current in visited:
                continue
            visited.add(current)

            if current in self.nodes:
                for edge in self.nodes[current].edges:
                    if edge.destination_node_id and edge.destination_node_id not in visited:
                        to_visit.append(edge.destination_node_id)

        return [node_id for node_id in self.nodes.keys() if node_id not in visited]

    def get_missing_tools(self) -> list[str]:
        """Find tools referenced by function nodes but not defined.

        Returns:
            List of tool IDs that are referenced but not defined in the flow.

        Example:
            ```python
            missing = builder.get_missing_tools()
            if missing:
                print(f"Error: Missing tool definitions: {missing}")
            ```
        """
        referenced_tools = set()
        for node in self.nodes.values():
            if node.type == NodeType.FUNCTION and node.tool_id:
                referenced_tools.add(node.tool_id)

        defined_tools = set(self.tools.keys())
        return list(referenced_tools - defined_tools)

    def validate(self) -> list[str]:
        """Validate the current flow state.

        Performs comprehensive validation of the flow structure, including:
        - Start node configuration
        - Node connectivity and reachability
        - Tool references and definitions
        - Edge destination validation
        - Structural integrity checks

        Returns:
            List of validation errors. Empty list indicates the flow is valid.

        Example:
            ```python
            errors = builder.validate()
            if errors:
                for error in errors:
                    print(f"Validation error: {error}")
            else:
                print("Flow is valid!")
            ```
        """
        errors = []

        # Basic structure validation
        if not self.start_node_id:
            errors.append("Start node must be set using set_start_node()")
        elif self.start_node_id not in self.nodes:
            errors.append(f"Start node '{self.start_node_id}' not found in flow")

        if not self.nodes:
            errors.append("Flow must have at least one node")

        # Edge destination validation
        all_node_ids = set(self.nodes.keys())
        for node_id, node in self.nodes.items():
            for edge in node.edges:
                if edge.destination_node_id and edge.destination_node_id not in all_node_ids:
                    errors.append(
                        f"Edge '{edge.id}' in node '{node_id}' points to non-existent node: "
                        f"'{edge.destination_node_id}'"
                    )

        # Tool validation
        missing_tools = self.get_missing_tools()
        if missing_tools:
            errors.append(
                f"Function nodes reference undefined tools: {missing_tools}. "
                f"Add them using add_tool()"
            )

        # Reachability validation (warning, not error)
        unreachable = self.get_unreachable_nodes()
        if unreachable:
            errors.append(
                f"Warning: Unreachable nodes detected: {unreachable}. "
                f"These nodes cannot be reached from start node '{self.start_node_id}'"
            )

        # Flow structure validation
        if self.start_node_id and self.start_node_id in self.nodes:
            start_node = self.nodes[self.start_node_id]
            if start_node.type == NodeType.END:
                errors.append("Start node cannot be an END node")

        return errors

    def build(self) -> FlowSpec:
        """Build the final FlowSpec.

        Validates the flow and creates the final FlowSpec instance. This is
        the final step in flow construction and produces a complete, validated
        conversation flow.

        Returns:
            Complete FlowSpec instance ready for execution.

        Raises:
            ValueError: If the flow validation fails. The error message will
                       contain all validation issues found.

        Example:
            ```python
            try:
                flow = builder.build()
                print(f"Flow '{flow.conversation_flow_id}' built successfully!")
            except ValueError as e:
                print(f"Flow validation failed: {e}")
            ```
        """
        errors = self.validate()
        if errors:
            error_msg = f"Flow validation failed with {len(errors)} error(s):\n"
            error_msg += "\n".join(f"  - {error}" for error in errors)
            raise ValueError(error_msg)

        # Import here to avoid circular imports
        from .flow_spec import FlowSpec

        flow = FlowSpec(
            conversation_flow_id=self.flow_id,
            version=1,
            global_prompt=self.global_prompt,
            nodes=self.nodes,
            start_node_id=self.start_node_id,
            tools=self.tools,
        )

        logger.info(
            f"Built flow '{self.flow_id}' with {len(self.nodes)} nodes and {len(self.tools)} tools"
        )
        return flow

    def _validate_node_id(self, node_id: str) -> str:
        """Validate node ID format and uniqueness, returning the stripped ID."""
        stripped_id = node_id.strip()
        if not stripped_id:
            raise ValueError("Node ID cannot be empty or contain only whitespace")
        if stripped_id in self.nodes:
            raise ValueError(
                f"Node with ID '{stripped_id}' already exists. "
                f"Existing nodes: {list(self.nodes.keys())}"
            )
        return stripped_id

    def _validate_node_name(self, name: str) -> None:
        """Validate node name format."""
        if not name.strip():
            raise ValueError("Node name cannot be empty or contain only whitespace")

    def __enter__(self) -> FlowBuilder:
        """Context manager entry.

        Enables automatic flow validation when used as a context manager.
        The flow will be validated when exiting the context.

        Returns:
            Self for use within the context.

        Example:
            ```python
            with FlowBuilder("my_flow") as builder:
                builder.add_conversation_node("start", "Start", "Hello!")
                builder.set_start_node("start")
                # Validation happens automatically here
            ```
        """
        self._context_validation_enabled = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit.

        Performs automatic validation if no exceptions occurred during
        context execution. If validation fails, logs warnings but does
        not raise exceptions.

        Args:
            exc_type: Exception type if any occurred
            exc_val: Exception value if any occurred
            exc_tb: Exception traceback if any occurred
        """
        self._context_validation_enabled = False

        # Only validate if no exceptions occurred in the context
        if exc_type is None:
            try:
                errors = self.validate()
                if errors:
                    logger.warning(
                        "Flow validation issues found in context manager:\n"
                        + "\n".join(f"  - {error}" for error in errors)
                    )
                else:
                    logger.debug(f"Flow '{self.flow_id}' passed context manager validation")
            except Exception as e:
                logger.error(f"Error during context manager validation: {e}")


# Export the builder
__all__ = ["FlowBuilder"]
