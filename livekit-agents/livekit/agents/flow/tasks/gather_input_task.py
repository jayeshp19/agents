import json
import logging
import re
from typing import TYPE_CHECKING, Any, Optional

from livekit.agents import AgentTask, llm

from ..schema import GatherInputVariable, Node
from ..types.types import TransitionResult
from ..utils.utils import clean_json_response

if TYPE_CHECKING:
    from ..core.runner import FlowRunner

logger = logging.getLogger(__name__)


class GatherInputTask(AgentTask[Optional[TransitionResult]]):
    """Handles gathering and validating user input in flow nodes.

    This task collects required and optional information from users
    through natural conversation, validates the input, and stores
    it in the flow context.
    """

    def __init__(self, node: Node, flow_runner: "FlowRunner") -> None:
        instructions = self._build_gather_instructions(node)
        super().__init__(instructions=instructions)

        self.node = node
        self.flow_runner = flow_runner
        self.gathered_data: dict[str, Any] = {}
        self.missing_required_fields: set[str] = set()
        self.validation_errors: dict[str, str] = {}
        self.turn_count = 0
        self.should_transition = False
        self._completed = False

        for var in self.node.gather_input_variables:
            if var.required:
                self.missing_required_fields.add(var.name)

    def complete(self, result: Optional[TransitionResult]) -> None:
        """Override to track completion state"""
        if not self._completed:
            self._completed = True
            super().complete(result)

    def _build_gather_instructions(self, node: Node) -> str:
        """Build instructions for the data collection assistant."""
        if not node.gather_input_variables:
            return "You are a helpful assistant."

        required_fields = []
        optional_fields = []

        for var in node.gather_input_variables:
            field_info = f"- {var.name} ({var.type}): {var.description}"
            if var.required:
                required_fields.append(field_info)
            else:
                optional_fields.append(field_info)

        instructions = f"""You are an intelligent data collection assistant. Your job is to naturally gather the following information from users through conversation.

REQUIRED INFORMATION:
{chr(10).join(required_fields) if required_fields else "None"}

OPTIONAL INFORMATION:
{chr(10).join(optional_fields) if optional_fields else "None"}

CORE PRINCIPLES:
1. Be conversational and natural - don't sound like a form
2. Allow users to provide multiple pieces of information at once
3. Confirm information when you receive it
4. Ask follow-up questions for missing or unclear data
5. Guide the conversation toward collecting all required fields
6. Be patient and helpful with validation errors

CONVERSATION FLOW:
- Start by explaining what information you need
- Encourage users to provide information in any order
- Acknowledge each piece of information you receive
- Gently prompt for missing required information
- Confirm all collected data before proceeding

{node.gather_input_instruction or "Please collect the required information through natural conversation."}

Remember: You have intelligent extraction capabilities that will identify relevant information from user responses automatically."""

        return instructions

    async def on_enter(self) -> None:
        """Called when entering the gather input node."""
        try:
            self.flow_runner.context.execution_path.append(self.node.id)
            self.flow_runner.context.save_checkpoint(self.node.id)

            logger.info(f"Starting gather input task for node {self.node.id}")
            logger.debug(
                f"Variables to gather: {[var.name for var in self.node.gather_input_variables]}"
            )

            if not self.node.gather_input_variables:
                logger.warning(f"No gather input variables defined for node {self.node.id}")
                evaluation = await self.flow_runner._evaluate_transition_conditions(self.node)
                transition_result = TransitionResult.from_evaluation(evaluation)
                self.complete(transition_result)
                return

            await self._initiate_gathering_conversation()

        except Exception as e:
            logger.error(f"Error in GatherInputTask.on_enter for node {self.node.id}: {e}")
            self.flow_runner._handle_task_error(e, self.node.id)
            # Complete task with None result on error
            self.complete(None)
            return

    async def _initiate_gathering_conversation(self) -> None:
        """Start the gathering conversation"""
        required_items = [
            var.description for var in self.node.gather_input_variables if var.required
        ]
        optional_items = [
            var.description for var in self.node.gather_input_variables if not var.required
        ]

        if required_items and optional_items:
            intro_message = f"I need to collect some information from you. I'll need your {', '.join(required_items)}, and optionally your {', '.join(optional_items)}. You can provide any or all of this information now."
        elif required_items:
            intro_message = f"I need to collect your {', '.join(required_items)}. You can provide this information in any order you prefer."
        elif optional_items:
            intro_message = (
                f"I can collect your {', '.join(optional_items)} if you'd like to provide it."
            )
        else:
            intro_message = "I'm ready to collect some information from you."

        await self.session.say(intro_message)

    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        """Called when user completes a turn in the conversation."""
        try:
            self.turn_count += 1
            text_content = new_message.text_content or ""

            logger.debug(f"GatherInputTask - User input: '{text_content}'")
            self.flow_runner.context.add_message(new_message)

            # Extract data first, then generate response to avoid race conditions
            await self._extract_and_update_data(text_content)
            await self._generate_conversational_response(text_content)

        except Exception as e:
            logger.error(f"Error in user turn completion for gather input node {self.node.id}: {e}")
            self.flow_runner._handle_task_error(e, self.node.id)
            # Don't complete here - let the task continue or timeout

    async def _extract_and_update_data(self, user_input: str) -> None:
        """Extract and validate data from user input."""
        extracted_data = await self._extract_information_with_llm(user_input)

        for var_name, value in extracted_data.items():
            if value is not None:
                self.gathered_data[var_name] = value
                if var_name in self.missing_required_fields:
                    self.missing_required_fields.remove(var_name)
                # Clear any previous validation errors for this field
                if var_name in self.validation_errors:
                    del self.validation_errors[var_name]
                logger.info(f"Successfully gathered '{var_name}': {value}")

        # Log progress
        total_fields = len(self.node.gather_input_variables)
        collected_fields = len(self.gathered_data)
        required_remaining = len(self.missing_required_fields)

        logger.info(
            f"Progress: {collected_fields}/{total_fields} fields collected, {required_remaining} required fields remaining"
        )

    async def _generate_conversational_response(self, user_input: str) -> None:
        """Generate a conversational response based on current gathering state."""
        if not self.missing_required_fields:
            await self._complete_gathering()
            return

        collected_fields = list(self.gathered_data.keys())
        missing_required = [
            var.name
            for var in self.node.gather_input_variables
            if var.name in self.missing_required_fields
        ]

        # Build validation error context
        validation_error_info = ""
        if self.validation_errors:
            error_details = []
            for field_name, error_msg in self.validation_errors.items():
                var_config = next(
                    (v for v in self.node.gather_input_variables if v.name == field_name), None
                )
                if var_config:
                    error_details.append(f"  - {var_config.description}: {error_msg}")
                else:
                    error_details.append(f"  - {field_name}: {error_msg}")
            validation_error_info = "\n- Recent validation errors:\n" + "\n".join(error_details)

        status_context = f"""
CURRENT STATUS:
- Collected fields: {collected_fields}
- Missing required fields: {missing_required}{validation_error_info}
- User just said: "{user_input}"

Generate a natural response that:
1. Acknowledges any information just provided
2. If there were validation errors, politely explain what went wrong and ask for the correct format
3. Asks for missing required information if needed
4. Maintains conversational flow
5. Doesn't repeat information already collected
6. Be specific about format requirements when validation fails
"""

        await self.session.generate_reply(instructions=status_context)

    async def _extract_information_with_llm(self, user_input: str) -> dict[str, Any]:
        """Use LLM to extract structured information from user input."""
        extraction_schema = {}
        for var in self.node.gather_input_variables:
            extraction_schema[var.name] = {
                "type": var.type,
                "description": var.description,
                "required": var.required,
                "already_collected": var.name in self.gathered_data,
            }

        extraction_prompt = f"""
TASK: Extract and validate information from user input.

FIELDS TO EXTRACT:
{json.dumps(extraction_schema, indent=2)}

USER INPUT: "{user_input}"

EXTRACTION RULES:
- Only extract information explicitly mentioned or clearly implied
- Validate data types (email must have @, phone needs 10+ digits, etc.)
- Don't extract fields already collected
- Return null for invalid or missing data

VALIDATION CRITERIA:
- email: Must contain @ and valid domain format
- phone: Must have at least 10 digits (any format)
- number: Must be convertible to int/float
- date: Must be recognizable date format (MM/DD/YYYY, YYYY-MM-DD, etc.)
- string: Any non-empty text

OUTPUT: JSON object with extracted values:
{{
    "field_name": "validated_value_or_null"
}}

Only include fields that were mentioned in the user input."""

        try:
            extraction_ctx = llm.ChatContext()
            extraction_ctx.add_message(role="user", content=extraction_prompt)

            response_parts = []
            async with self.flow_runner.edge_llm.chat(chat_ctx=extraction_ctx) as stream:
                async for chunk in stream:
                    if chunk.delta and chunk.delta.content:
                        response_parts.append(chunk.delta.content)

            response_text = "".join(response_parts).strip()
            response_text = clean_json_response(response_text)

            try:
                extracted_data = json.loads(response_text)
                logger.debug(f"LLM extracted data: {extracted_data}")

                validated_data = {}
                for var_name, value in extracted_data.items():
                    if value is not None and var_name in [
                        v.name for v in self.node.gather_input_variables
                    ]:
                        if var_name in self.gathered_data:
                            continue

                        var_config = next(
                            (v for v in self.node.gather_input_variables if v.name == var_name),
                            None,
                        )
                        if not var_config:
                            logger.warning(
                                f"Variable {var_name} not found in gather_input_variables"
                            )
                            continue

                        validation_result = await self._validate_extracted_value(var_config, value)
                        if validation_result["valid"]:
                            validated_data[var_name] = validation_result["value"]
                            # Clear any previous validation errors for this field
                            if var_name in self.validation_errors:
                                del self.validation_errors[var_name]
                        else:
                            # Store validation error for user feedback
                            self.validation_errors[var_name] = validation_result["error"]
                            logger.debug(
                                f"Validation failed for {var_name}: {validation_result['error']}"
                            )

                return validated_data

            except json.JSONDecodeError:
                logger.warning(f"Failed to parse extraction response: {response_text}")
                return {}

        except Exception as e:
            logger.error(f"Error in LLM extraction: {e}")
            return {}

    async def _validate_extracted_value(
        self, variable: GatherInputVariable, value: Any
    ) -> dict[str, Any]:
        """Validate an extracted value according to variable configuration."""
        if not value:
            return {"valid": False, "error": "Empty value", "value": None}

        value_str = str(value).strip()

        # Priority 1: Check regex pattern if provided
        if variable.regex_pattern:
            return await self._validate_with_regex(variable, value_str)

        # Priority 2: Fall back to built-in type validation
        return await self._validate_built_in_type(variable, value_str)

    async def _validate_with_regex(
        self, variable: GatherInputVariable, value_str: str
    ) -> dict[str, Any]:
        """Validate value using custom regex pattern"""
        # Ensure regex_pattern is not None (should be guaranteed by caller)
        if not variable.regex_pattern:
            return {"valid": False, "error": "No regex pattern provided", "value": None}

        try:
            if re.match(variable.regex_pattern, value_str):
                logger.debug(f"Regex validation passed for {variable.name}: '{value_str}'")
                return {"valid": True, "error": None, "value": value_str}
            else:
                error_msg = (
                    variable.regex_error_message
                    or f"Value doesn't match the required format for {variable.description}"
                )
                logger.debug(
                    f"Regex validation failed for {variable.name}: '{value_str}' doesn't match pattern '{variable.regex_pattern}'"
                )
                return {"valid": False, "error": error_msg, "value": None}

        except re.error as e:
            logger.error(
                f"Invalid regex pattern for variable '{variable.name}': {variable.regex_pattern}. Error: {e}"
            )
            error_msg = (
                f"Configuration error: invalid validation pattern for {variable.description}"
            )
            return {"valid": False, "error": error_msg, "value": None}

        except Exception as e:
            logger.error(f"Unexpected error during regex validation for {variable.name}: {e}")
            error_msg = f"Validation error for {variable.description}"
            return {"valid": False, "error": error_msg, "value": None}

    async def _validate_built_in_type(
        self, variable: GatherInputVariable, value_str: str
    ) -> dict[str, Any]:
        """Validate value using built-in type validation rules."""
        if variable.type == "email":
            email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
            if re.match(email_pattern, value_str):
                return {"valid": True, "error": None, "value": value_str}
            return {"valid": False, "error": "Invalid email format", "value": None}

        elif variable.type == "phone":
            digits = re.sub(r"[^\d]", "", value_str)
            if len(digits) >= 10:
                return {"valid": True, "error": None, "value": value_str}
            return {"valid": False, "error": "Invalid phone number", "value": None}

        elif variable.type == "number":
            try:
                if "." in value_str:
                    parsed_value = float(value_str)
                else:
                    parsed_value = int(value_str)
                return {"valid": True, "error": None, "value": parsed_value}
            except ValueError:
                return {"valid": False, "error": "Invalid number", "value": None}

        elif variable.type == "date":
            date_patterns = [
                r"^\d{1,2}/\d{1,2}/\d{4}$",
                r"^\d{4}-\d{1,2}-\d{1,2}$",
            ]
            if any(re.match(pattern, value_str) for pattern in date_patterns):
                return {"valid": True, "error": None, "value": value_str}
            return {"valid": False, "error": "Invalid date format", "value": None}

        else:
            return {"valid": True, "error": None, "value": value_str}

    async def _complete_gathering(self) -> None:
        """Complete the gathering process and store results in context."""
        logger.info(f"Completed gathering input for node {self.node.id}")
        logger.debug(f"Gathered data: {self.gathered_data}")

        self.flow_runner.context.set_variable("gathered_input", self.gathered_data)

        # Store each variable directly for easy retrieval by function nodes
        for var_name, value in self.gathered_data.items():
            # Store without prefix (best practice)
            self.flow_runner.context.set_variable(var_name, value)
            # Optional: also store legacy prefixed key for backward-compatibility
            self.flow_runner.context.set_variable(f"input_{var_name}", value)

        gathered_info = []
        for var_name, value in self.gathered_data.items():
            var_config = next(
                (v for v in self.node.gather_input_variables if v.name == var_name), None
            )
            if var_config:
                gathered_info.append(f"{var_config.description}: {value}")
            else:
                gathered_info.append(f"{var_name}: {value}")

        confirmation_message = "Perfect! I have all the information I need:\n" + "\n".join(
            gathered_info
        )
        await self.session.say(confirmation_message)

        evaluation = await self.flow_runner._evaluate_transition_conditions(
            self.node, "gathering_completed"
        )
        transition_result = TransitionResult.from_evaluation(evaluation)

        self.should_transition = True
        self.complete(transition_result)
