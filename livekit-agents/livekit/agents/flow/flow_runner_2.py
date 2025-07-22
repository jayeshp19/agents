# import asyncio
# import json
# import logging
# import re
# import time
# from dataclasses import dataclass, field
# from pathlib import Path
# from typing import Any, Callable, Optional

# import aiohttp

# from livekit.agents import (
#     Agent,
#     AgentSession,
#     AgentTask,
#     llm,
# )
# from livekit.agents.voice.agent import _set_activity_task_info

# from .schema import Edge, FlowSpec, GatherInputVariable, Node, load_flow

# logger = logging.getLogger(__name__)


# def clean_json_response(response_text: str) -> str:
#     response_text = re.sub(r"```json\s*", "", response_text)
#     response_text = re.sub(r"```\s*$", "", response_text)
#     response_text = re.sub(r"^```\s*", "", response_text)

#     response_text = response_text.strip()

#     return response_text


# @dataclass
# class EdgeEvaluation:
#     edge_id: Optional[str]
#     destination_node_id: Optional[str]
#     confidence: float
#     reasoning: str
#     fallback_used: bool = False


# @dataclass
# class TransitionResult:
#     destination_node_id: Optional[str]
#     edge_id: Optional[str]
#     user_text: Optional[str]
#     confidence: float
#     reasoning: str

#     @staticmethod
#     def from_evaluation(
#         evaluation: EdgeEvaluation, user_text: Optional[str] = None
#     ) -> "TransitionResult":
#         return TransitionResult(
#             destination_node_id=evaluation.destination_node_id,
#             edge_id=evaluation.edge_id,
#             user_text=user_text,
#             confidence=evaluation.confidence,
#             reasoning=evaluation.reasoning,
#         )


# @dataclass
# class FlowError:
#     error_type: str
#     message: str
#     node_id: str
#     timestamp: float
#     retry_count: int = 0
#     is_recoverable: bool = True

#     @classmethod
#     def from_exception(
#         cls, error: Exception, node_id: str, error_type: str = "system_error"
#     ) -> "FlowError":
#         return cls(
#             error_type=error_type,
#             message=str(error),
#             node_id=node_id,
#             timestamp=time.time(),
#             is_recoverable=error_type in ["llm_failure", "validation_error"],
#         )


# @dataclass
# class FlowContext:
#     conversation_history: list[llm.ChatMessage] = field(default_factory=list)
#     variables: dict[str, Any] = field(default_factory=dict)
#     execution_path: list[str] = field(default_factory=list)
#     function_results: dict[str, Any] = field(default_factory=dict)
#     start_time: float = field(default_factory=time.time)
#     checkpoints: list[dict[str, Any]] = field(default_factory=list)
#     errors: list[FlowError] = field(default_factory=list)  # Track errors for analysis

#     max_history_length: int = 50

#     def add_message(self, message: llm.ChatMessage) -> None:
#         self.conversation_history.append(message)
#         if len(self.conversation_history) > self.max_history_length:
#             self.conversation_history = (
#                 self.conversation_history[:1]
#                 + self.conversation_history[-(self.max_history_length - 1) :]
#             )

#     def set_variable(self, key: str, value: Any) -> None:
#         self.variables[key] = value

#     def get_variable(self, key: str, default: Any = None) -> Any:
#         return self.variables.get(key, default)

#     def save_checkpoint(self, node_id: str) -> None:
#         checkpoint = {
#             "node_id": node_id,
#             "timestamp": time.time(),
#             "variables": self.variables.copy(),
#             "execution_path": self.execution_path.copy(),
#         }
#         self.checkpoints.append(checkpoint)
#         if len(self.checkpoints) > 10:
#             self.checkpoints.pop(0)

#     def add_error(self, error: FlowError) -> None:
#         self.errors.append(error)
#         if len(self.errors) > 20:
#             self.errors.pop(0)


# @dataclass
# class TransitionDecision:
#     selected_option: Optional[int]
#     confidence: float
#     reasoning: str
#     user_intent: str

#     @classmethod
#     def from_dict(cls, data: dict) -> "TransitionDecision":
#         return cls(
#             selected_option=data.get("selected_option"),
#             confidence=data.get("confidence", 0.0),
#             reasoning=data.get("reasoning", ""),
#             user_intent=data.get("user_intent", ""),
#         )


# class ConversationTask(AgentTask[Optional[TransitionResult]]):
#     def __init__(self, node: Node, flow_runner: "FlowRunner") -> None:
#         super().__init__(instructions="")
#         self.node = node
#         self.flow_runner = flow_runner
#         self.turn_count = 0
#         self.last_user_input: Optional[str] = None
#         self.should_transition = False
#         self._completed = False

#     def complete(self, result: Optional[TransitionResult]) -> None:
#         if not self._completed:
#             self._completed = True
#             super().complete(result)

#     async def on_enter(self) -> None:
#         try:
#             self.flow_runner.context.execution_path.append(self.node.id)
#             self.flow_runner.context.save_checkpoint(self.node.id)

#             if self.node.instruction:
#                 if self.node.instruction.type == "prompt":
#                     await self.session.generate_reply(instructions=self.node.instruction.text)
#                 elif self.node.instruction.type == "static_text":
#                     await self.session.say(self.node.instruction.text)

#             if self.node.skip_response_edge:
#                 self.should_transition = True
#                 transition_result = TransitionResult(
#                     destination_node_id=self.node.skip_response_edge.destination_node_id,
#                     edge_id=self.node.skip_response_edge.id,
#                     user_text=None,
#                     confidence=1.0,
#                     reasoning="Skip response edge",
#                 )
#                 self.complete(transition_result)
#                 return

#         except Exception as e:
#             logger.error(f"Error in ConversationTask.on_enter for node {self.node.id}: {e}")
#             self.flow_runner._handle_task_error(e, self.node.id)
#             self.complete(None)
#             return

#     async def on_user_turn_completed(
#         self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
#     ) -> None:
#         try:
#             self.turn_count += 1
#             text_content = new_message.text_content or ""
#             self.last_user_input = text_content

#             logger.debug("=== USER TURN COMPLETED ===")
#             logger.debug(f"Turn count: {self.turn_count}")
#             logger.debug(f"Node: {self.node.id}")
#             logger.debug(f"User input: '{text_content}'")
#             logger.debug(f"Should transition: {self.should_transition}")

#             self.flow_runner.context.add_message(new_message)

#             logger.debug(f"User turn {self.turn_count} in node {self.node.id}: {text_content}")

#             if self.should_transition:
#                 logger.debug("Already marked for transition, skipping evaluation")
#                 return

#             logger.debug("Evaluating transition conditions...")
#             evaluation = await self.flow_runner._evaluate_transition_conditions(
#                 self.node, text_content
#             )

#             logger.debug(
#                 f"Evaluation result: destination={evaluation.destination_node_id}, confidence={evaluation.confidence:.2f}"
#             )

#             if evaluation.destination_node_id:
#                 logger.info(
#                     f"Transitioning from {self.node.id} to {evaluation.destination_node_id} "
#                     f"(edge: {evaluation.edge_id}, confidence: {evaluation.confidence:.2f})"
#                 )
#                 self.should_transition = True
#                 transition_result = TransitionResult.from_evaluation(evaluation, text_content)
#                 logger.debug(f"Completing task with transition result: {transition_result}")
#                 self.complete(transition_result)
#             else:
#                 logger.debug(
#                     f"Continuing conversation in node {self.node.id} - no valid transition found"
#                 )

#         except Exception as e:
#             logger.error(f"Error in user turn completion for node {self.node.id}: {e}")
#             self.flow_runner._handle_task_error(e, self.node.id)


# class FunctionTask(AgentTask[Optional[TransitionResult]]):
#     def __init__(self, node: Node, flow_runner: "FlowRunner") -> None:
#         super().__init__(instructions="")
#         self.node = node
#         self.flow_runner = flow_runner
#         self._completed = False

#     def complete(self, result: Optional[TransitionResult]) -> None:
#         """Override to track completion state"""
#         if not self._completed:
#             self._completed = True
#             super().complete(result)

#     async def on_enter(self) -> None:
#         try:
#             self.flow_runner.context.execution_path.append(self.node.id)
#             self.flow_runner.context.save_checkpoint(self.node.id)

#             if self.node.tool_id:
#                 # Execute custom HTTP function (simplified - no fallback to Python functions)
#                 result = await self._execute_custom_function()
#                 if result is not None:
#                     self.flow_runner.context.function_results[self.node.tool_id] = result
#                     logger.info(f"Custom function {self.node.tool_id} executed successfully")
#                 else:
#                     logger.warning(
#                         f"Custom function execution failed for tool_id: {self.node.tool_id}"
#                     )

#             # Evaluate where to go next after function execution
#             evaluation = await self.flow_runner._evaluate_transition_conditions(self.node)
#             transition_result = TransitionResult.from_evaluation(evaluation)
#             self.complete(transition_result)

#         except Exception as e:
#             logger.error(f"Error in FunctionTask.on_enter for node {self.node.id}: {e}")
#             self.flow_runner._handle_task_error(e, self.node.id)
#             # Complete task with None result on error
#             self.complete(None)
#             return

#     async def _execute_custom_function(self) -> Optional[dict[str, Any]]:
#         """Execute HTTP-based custom function with all logic inline"""
#         if not self.node.tool_id:
#             return None

#         tool_spec = self.flow_runner.flow.tools.get(self.node.tool_id)
#         if not tool_spec or tool_spec.type != "custom" or not tool_spec.url:
#             logger.error(f"Invalid custom function configuration for {self.node.tool_id}")
#             return None

#         try:
#             start_time = time.time()

#             parameters = await self._get_parameters_from_context(tool_spec)

#             if self._has_missing_required_parameters(tool_spec, parameters):
#                 return {"error": "Missing required parameters"}

#             result = await self._execute_http_request(tool_spec, parameters)

#             execution_time = (time.time() - start_time) * 1000
#             result["execution_time_ms"] = execution_time

#             # Optionally speak during execution
#             if self.node.speak_during_execution:
#                 if result.get("success", False):
#                     await self.session.say(f"I've executed {tool_spec.name} and got the results.")
#                 else:
#                     await self.session.say(
#                         "I encountered an issue processing your request. Let me try to help you differently."
#                     )

#             return result

#         except Exception as e:
#             logger.error(f"Custom function execution failed for {self.node.tool_id}: {e}")
#             return {"error": f"Function execution failed: {str(e)}", "tool_name": tool_spec.name}

#     async def _get_parameters_from_context(self, tool_spec) -> dict[str, Any]:
#         """Extract parameters directly from flow context"""
#         parameter_schema = tool_spec.parameters

#         if not parameter_schema:
#             return {}

#         # Handle both JSON Schema and flat parameter formats
#         if isinstance(parameter_schema, dict) and "properties" in parameter_schema:
#             properties = parameter_schema.get("properties", {})
#         else:
#             properties = {name: {"type": "string"} for name in parameter_schema.keys()}

#         extracted_params = {}
#         context_vars = self.flow_runner.context.variables

#         for param_name in properties.keys():
#             if param_name in context_vars:
#                 extracted_params[param_name] = context_vars[param_name]
#             elif f"input_{param_name}" in context_vars:
#                 extracted_params[param_name] = context_vars[f"input_{param_name}"]
#             elif f"{tool_spec.tool_id}_{param_name}" in context_vars:
#                 extracted_params[param_name] = context_vars[f"{tool_spec.tool_id}_{param_name}"]

#         return extracted_params

#     def _has_missing_required_parameters(self, tool_spec, parameters: dict[str, Any]) -> bool:
#         parameter_schema = tool_spec.parameters

#         if not parameter_schema:
#             return False

#         required_params = parameter_schema.get("required", [])
#         missing = [param for param in required_params if param not in parameters]

#         if missing:
#             logger.error(f"Missing required parameters for {tool_spec.name}: {missing}")
#             return True
#         return False

#     async def _execute_http_request(self, tool_spec, parameters: dict[str, Any]) -> dict[str, Any]:
#         import json

#         session = await self.flow_runner.get_http_session()

#         try:
#             method = tool_spec.http_method.upper()
#             url = tool_spec.url
#             headers = tool_spec.headers.copy()
#             timeout = aiohttp.ClientTimeout(total=tool_spec.timeout_ms / 1000.0)
#             params = tool_spec.query_parameters.copy()

#             data = None
#             if method in ["POST", "PUT", "PATCH"]:
#                 headers["Content-Type"] = headers.get("Content-Type", "application/json")
#                 data = json.dumps(parameters)
#             elif method == "GET":
#                 params.update(parameters)

#             logger.info(f"Executing {method} request to {url} with parameters: {parameters}")

#             async with session.request(
#                 method=method, url=url, headers=headers, params=params, data=data, timeout=timeout
#             ) as response:
#                 status_code = response.status
#                 response_text = await response.text()

#                 try:
#                     result_data = await response.json()
#                 except (json.JSONDecodeError, aiohttp.ContentTypeError):
#                     result_data = {"response": response_text}

#                 if 200 <= status_code < 300:
#                     logger.info(
#                         f"Function {tool_spec.name} executed successfully (status: {status_code})"
#                     )
#                     return {
#                         "success": True,
#                         "result": result_data,
#                         "status_code": status_code,
#                         "tool_name": tool_spec.name,
#                     }
#                 else:
#                     error_msg = f"HTTP {status_code}: {response_text[:500]}"
#                     logger.warning(f"Function {tool_spec.name} returned error: {error_msg}")
#                     return {
#                         "success": False,
#                         "error": error_msg,
#                         "status_code": status_code,
#                         "result": result_data,
#                         "tool_name": tool_spec.name,
#                     }

#         except asyncio.TimeoutError:
#             error_msg = f"Request timeout after {tool_spec.timeout_ms}ms"
#             logger.error(f"Function {tool_spec.name} timed out: {error_msg}")
#             return {"success": False, "error": error_msg, "tool_name": tool_spec.name}
#         except Exception as e:
#             error_msg = f"HTTP request failed: {str(e)}"
#             logger.error(f"Function {tool_spec.name} failed: {error_msg}")
#             return {"success": False, "error": error_msg, "tool_name": tool_spec.name}


# class GatherInputTask(AgentTask[Optional[TransitionResult]]):
#     def __init__(self, node: Node, flow_runner: "FlowRunner") -> None:
#         instructions = self._build_gather_instructions(node)
#         super().__init__(instructions=instructions)

#         self.node = node
#         self.flow_runner = flow_runner
#         self.gathered_data: dict[str, Any] = {}
#         self.missing_required_fields: set[str] = set()
#         self.validation_errors: dict[str, str] = {}
#         self.turn_count = 0
#         self.should_transition = False
#         self._completed = False

#         for var in self.node.gather_input_variables:
#             if var.required:
#                 self.missing_required_fields.add(var.name)

#     def complete(self, result: Optional[TransitionResult]) -> None:
#         """Override to track completion state"""
#         if not self._completed:
#             self._completed = True
#             super().complete(result)

#     def _build_gather_instructions(self, node: Node) -> str:
#         if not node.gather_input_variables:
#             return "You are a helpful assistant."

#         required_fields = []
#         optional_fields = []

#         for var in node.gather_input_variables:
#             field_info = f"- {var.name} ({var.type}): {var.description}"
#             if var.required:
#                 required_fields.append(field_info)
#             else:
#                 optional_fields.append(field_info)

#         instructions = f"""You are an intelligent data collection assistant. Your job is to naturally gather the following information from users through conversation.

# REQUIRED INFORMATION:
# {chr(10).join(required_fields) if required_fields else "None"}

# OPTIONAL INFORMATION:
# {chr(10).join(optional_fields) if optional_fields else "None"}

# CORE PRINCIPLES:
# 1. Be conversational and natural - don't sound like a form
# 2. Allow users to provide multiple pieces of information at once
# 3. Confirm information when you receive it
# 4. Ask follow-up questions for missing or unclear data
# 5. Guide the conversation toward collecting all required fields
# 6. Be patient and helpful with validation errors

# CONVERSATION FLOW:
# - Start by explaining what information you need
# - Encourage users to provide information in any order
# - Acknowledge each piece of information you receive
# - Gently prompt for missing required information
# - Confirm all collected data before proceeding

# {node.gather_input_instruction or "Please collect the required information through natural conversation."}

# Remember: You have intelligent extraction capabilities that will identify relevant information from user responses automatically."""

#         return instructions

#     async def on_enter(self) -> None:
#         try:
#             self.flow_runner.context.execution_path.append(self.node.id)
#             self.flow_runner.context.save_checkpoint(self.node.id)

#             logger.info(f"Starting gather input task for node {self.node.id}")
#             logger.debug(
#                 f"Variables to gather: {[var.name for var in self.node.gather_input_variables]}"
#             )

#             if not self.node.gather_input_variables:
#                 logger.warning(f"No gather input variables defined for node {self.node.id}")
#                 evaluation = await self.flow_runner._evaluate_transition_conditions(self.node)
#                 transition_result = TransitionResult.from_evaluation(evaluation)
#                 self.complete(transition_result)
#                 return

#             await self._initiate_gathering_conversation()

#         except Exception as e:
#             logger.error(f"Error in GatherInputTask.on_enter for node {self.node.id}: {e}")
#             self.flow_runner._handle_task_error(e, self.node.id)
#             # Complete task with None result on error
#             self.complete(None)
#             return

#     async def _initiate_gathering_conversation(self) -> None:
#         """Start the gathering conversation"""
#         required_items = [
#             var.description for var in self.node.gather_input_variables if var.required
#         ]
#         optional_items = [
#             var.description for var in self.node.gather_input_variables if not var.required
#         ]

#         if required_items and optional_items:
#             intro_message = f"I need to collect some information from you. I'll need your {', '.join(required_items)}, and optionally your {', '.join(optional_items)}. You can provide any or all of this information now."
#         elif required_items:
#             intro_message = f"I need to collect your {', '.join(required_items)}. You can provide this information in any order you prefer."
#         elif optional_items:
#             intro_message = (
#                 f"I can collect your {', '.join(optional_items)} if you'd like to provide it."
#             )
#         else:
#             intro_message = "I'm ready to collect some information from you."

#         await self.session.say(intro_message)

#     async def on_user_turn_completed(
#         self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
#     ) -> None:
#         try:
#             self.turn_count += 1
#             text_content = new_message.text_content or ""

#             logger.debug(f"GatherInputTask - User input: '{text_content}'")
#             self.flow_runner.context.add_message(new_message)

#             # Extract data first, then generate response to avoid race conditions
#             await self._extract_and_update_data(text_content)
#             await self._generate_conversational_response(text_content)

#         except Exception as e:
#             logger.error(f"Error in user turn completion for gather input node {self.node.id}: {e}")
#             self.flow_runner._handle_task_error(e, self.node.id)
#             # Don't complete here - let the task continue or timeout

#     async def _extract_and_update_data(self, user_input: str) -> None:
#         extracted_data = await self._extract_information_with_llm(user_input)

#         for var_name, value in extracted_data.items():
#             if value is not None:
#                 self.gathered_data[var_name] = value
#                 if var_name in self.missing_required_fields:
#                     self.missing_required_fields.remove(var_name)
#                 # Clear any previous validation errors for this field
#                 if var_name in self.validation_errors:
#                     del self.validation_errors[var_name]
#                 logger.info(f"Successfully gathered '{var_name}': {value}")

#         # Log progress
#         total_fields = len(self.node.gather_input_variables)
#         collected_fields = len(self.gathered_data)
#         required_remaining = len(self.missing_required_fields)

#         logger.info(
#             f"Progress: {collected_fields}/{total_fields} fields collected, {required_remaining} required fields remaining"
#         )

#     async def _generate_conversational_response(self, user_input: str) -> None:
#         if not self.missing_required_fields:
#             await self._complete_gathering()
#             return

#         collected_fields = list(self.gathered_data.keys())
#         missing_required = [
#             var.name
#             for var in self.node.gather_input_variables
#             if var.name in self.missing_required_fields
#         ]

#         # Build validation error context
#         validation_error_info = ""
#         if self.validation_errors:
#             error_details = []
#             for field_name, error_msg in self.validation_errors.items():
#                 var_config = next(
#                     (v for v in self.node.gather_input_variables if v.name == field_name), None
#                 )
#                 if var_config:
#                     error_details.append(f"  - {var_config.description}: {error_msg}")
#                 else:
#                     error_details.append(f"  - {field_name}: {error_msg}")
#             validation_error_info = "\n- Recent validation errors:\n" + "\n".join(error_details)

#         status_context = f"""
# CURRENT STATUS:
# - Collected fields: {collected_fields}
# - Missing required fields: {missing_required}{validation_error_info}
# - User just said: "{user_input}"

# Generate a natural response that:
# 1. Acknowledges any information just provided
# 2. If there were validation errors, politely explain what went wrong and ask for the correct format
# 3. Asks for missing required information if needed
# 4. Maintains conversational flow
# 5. Doesn't repeat information already collected
# 6. Be specific about format requirements when validation fails
# """

#         await self.session.generate_reply(instructions=status_context)

#     async def _extract_information_with_llm(self, user_input: str) -> dict[str, Any]:
#         extraction_schema = {}
#         for var in self.node.gather_input_variables:
#             extraction_schema[var.name] = {
#                 "type": var.type,
#                 "description": var.description,
#                 "required": var.required,
#                 "already_collected": var.name in self.gathered_data,
#             }

#         extraction_prompt = f"""
# TASK: Extract and validate information from user input.

# FIELDS TO EXTRACT:
# {json.dumps(extraction_schema, indent=2)}

# USER INPUT: "{user_input}"

# EXTRACTION RULES:
# - Only extract information explicitly mentioned or clearly implied
# - Validate data types (email must have @, phone needs 10+ digits, etc.)
# - Don't extract fields already collected
# - Return null for invalid or missing data

# VALIDATION CRITERIA:
# - email: Must contain @ and valid domain format
# - phone: Must have at least 10 digits (any format)
# - number: Must be convertible to int/float
# - date: Must be recognizable date format (MM/DD/YYYY, YYYY-MM-DD, etc.)
# - string: Any non-empty text

# OUTPUT: JSON object with extracted values:
# {{
#     "field_name": "validated_value_or_null"
# }}

# Only include fields that were mentioned in the user input."""

#         try:
#             extraction_ctx = llm.ChatContext()
#             extraction_ctx.add_message(role="user", content=extraction_prompt)

#             response_parts = []
#             async with self.flow_runner.edge_llm.chat(chat_ctx=extraction_ctx) as stream:
#                 async for chunk in stream:
#                     if chunk.delta and chunk.delta.content:
#                         response_parts.append(chunk.delta.content)

#             response_text = "".join(response_parts).strip()
#             response_text = clean_json_response(response_text)

#             try:
#                 extracted_data = json.loads(response_text)
#                 logger.debug(f"LLM extracted data: {extracted_data}")

#                 validated_data = {}
#                 for var_name, value in extracted_data.items():
#                     if value is not None and var_name in [
#                         v.name for v in self.node.gather_input_variables
#                     ]:
#                         if var_name in self.gathered_data:
#                             continue

#                         var_config = next(
#                             (v for v in self.node.gather_input_variables if v.name == var_name),
#                             None,
#                         )
#                         if not var_config:
#                             logger.warning(
#                                 f"Variable {var_name} not found in gather_input_variables"
#                             )
#                             continue

#                         validation_result = await self._validate_extracted_value(var_config, value)
#                         if validation_result["valid"]:
#                             validated_data[var_name] = validation_result["value"]
#                             # Clear any previous validation errors for this field
#                             if var_name in self.validation_errors:
#                                 del self.validation_errors[var_name]
#                         else:
#                             # Store validation error for user feedback
#                             self.validation_errors[var_name] = validation_result["error"]
#                             logger.debug(
#                                 f"Validation failed for {var_name}: {validation_result['error']}"
#                             )

#                 return validated_data

#             except json.JSONDecodeError:
#                 logger.warning(f"Failed to parse extraction response: {response_text}")
#                 return {}

#         except Exception as e:
#             logger.error(f"Error in LLM extraction: {e}")
#             return {}

#     async def _validate_extracted_value(
#         self, variable: GatherInputVariable, value: Any
#     ) -> dict[str, Any]:
#         if not value:
#             return {"valid": False, "error": "Empty value", "value": None}

#         value_str = str(value).strip()

#         # Priority 1: Check regex pattern if provided
#         if variable.regex_pattern:
#             return await self._validate_with_regex(variable, value_str)

#         # Priority 2: Fall back to built-in type validation
#         return await self._validate_built_in_type(variable, value_str)

#     async def _validate_with_regex(
#         self, variable: GatherInputVariable, value_str: str
#     ) -> dict[str, Any]:
#         """Validate value using custom regex pattern"""
#         import re

#         # Ensure regex_pattern is not None (should be guaranteed by caller)
#         if not variable.regex_pattern:
#             return {"valid": False, "error": "No regex pattern provided", "value": None}

#         try:
#             if re.match(variable.regex_pattern, value_str):
#                 logger.debug(f"Regex validation passed for {variable.name}: '{value_str}'")
#                 return {"valid": True, "error": None, "value": value_str}
#             else:
#                 error_msg = (
#                     variable.regex_error_message
#                     or f"Value doesn't match the required format for {variable.description}"
#                 )
#                 logger.debug(
#                     f"Regex validation failed for {variable.name}: '{value_str}' doesn't match pattern '{variable.regex_pattern}'"
#                 )
#                 return {"valid": False, "error": error_msg, "value": None}

#         except re.error as e:
#             logger.error(
#                 f"Invalid regex pattern for variable '{variable.name}': {variable.regex_pattern}. Error: {e}"
#             )
#             error_msg = (
#                 f"Configuration error: invalid validation pattern for {variable.description}"
#             )
#             return {"valid": False, "error": error_msg, "value": None}

#         except Exception as e:
#             logger.error(f"Unexpected error during regex validation for {variable.name}: {e}")
#             error_msg = f"Validation error for {variable.description}"
#             return {"valid": False, "error": error_msg, "value": None}

#     async def _validate_built_in_type(
#         self, variable: GatherInputVariable, value_str: str
#     ) -> dict[str, Any]:
#         import re

#         if variable.type == "email":
#             email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
#             if re.match(email_pattern, value_str):
#                 return {"valid": True, "error": None, "value": value_str}
#             return {"valid": False, "error": "Invalid email format", "value": None}

#         elif variable.type == "phone":
#             digits = re.sub(r"[^\d]", "", value_str)
#             if len(digits) >= 10:
#                 return {"valid": True, "error": None, "value": value_str}
#             return {"valid": False, "error": "Invalid phone number", "value": None}

#         elif variable.type == "number":
#             try:
#                 if "." in value_str:
#                     parsed_value = float(value_str)
#                 else:
#                     parsed_value = int(value_str)
#                 return {"valid": True, "error": None, "value": parsed_value}
#             except ValueError:
#                 return {"valid": False, "error": "Invalid number", "value": None}

#         elif variable.type == "date":
#             date_patterns = [
#                 r"^\d{1,2}/\d{1,2}/\d{4}$",
#                 r"^\d{4}-\d{1,2}-\d{1,2}$",
#             ]
#             if any(re.match(pattern, value_str) for pattern in date_patterns):
#                 return {"valid": True, "error": None, "value": value_str}
#             return {"valid": False, "error": "Invalid date format", "value": None}

#         else:
#             return {"valid": True, "error": None, "value": value_str}

#     async def _complete_gathering(self) -> None:
#         logger.info(f"Completed gathering input for node {self.node.id}")
#         logger.debug(f"Gathered data: {self.gathered_data}")

#         self.flow_runner.context.set_variable("gathered_input", self.gathered_data)

#         # Store each variable directly for easy retrieval by function nodes
#         for var_name, value in self.gathered_data.items():
#             # Store without prefix (best practice)
#             self.flow_runner.context.set_variable(var_name, value)
#             # Optional: also store legacy prefixed key for backward-compatibility
#             self.flow_runner.context.set_variable(f"input_{var_name}", value)

#         gathered_info = []
#         for var_name, value in self.gathered_data.items():
#             var_config = next(
#                 (v for v in self.node.gather_input_variables if v.name == var_name), None
#             )
#             if var_config:
#                 gathered_info.append(f"{var_config.description}: {value}")
#             else:
#                 gathered_info.append(f"{var_name}: {value}")

#         confirmation_message = "Perfect! I have all the information I need:\n" + "\n".join(
#             gathered_info
#         )
#         await self.session.say(confirmation_message)

#         evaluation = await self.flow_runner._evaluate_transition_conditions(
#             self.node, "gathering_completed"
#         )
#         transition_result = TransitionResult.from_evaluation(evaluation)

#         self.should_transition = True
#         self.complete(transition_result)


# class FlowRunner:
#     def __init__(
#         self,
#         path: str,
#         edge_evaluator_llm: llm.LLM,
#         initial_context: Optional[dict[str, Any]] = None,
#         max_iterations: int = 100,
#     ) -> None:
#         self.flow_path = Path(path)
#         self._load_and_validate_flow()

#         self.context = FlowContext()
#         if initial_context:
#             self.context.variables.update(initial_context)

#         self.max_iterations = max_iterations

#         self._registered_functions: dict[str, Callable] = {}

#         self._metrics = {
#             "executions": 0,
#             "errors": 0,
#             "transitions": 0,
#             "function_calls": 0,
#             "llm_calls": 0,
#             "llm_tokens_consumed": 0,
#             "average_confidence": 0.0,
#             "execution_time": 0.0,
#             "nodes_visited": 0,
#         }

#         self._transition_times: list[float] = []
#         self._confidence_scores: list[float] = []

#         self.edge_llm = edge_evaluator_llm

#         self._http_session: Optional[aiohttp.ClientSession] = None

#     async def get_http_session(self) -> aiohttp.ClientSession:
#         if self._http_session is None or self._http_session.closed:
#             try:
#                 timeout = aiohttp.ClientTimeout(total=300)  # 5 minute total timeout
#                 self._http_session = aiohttp.ClientSession(timeout=timeout)
#             except Exception as e:
#                 logger.error(f"Failed to create HTTP session: {e}")
#                 # Ensure we don't leak a partially created session
#                 if self._http_session and not self._http_session.closed:
#                     try:
#                         await self._http_session.close()
#                     except Exception:
#                         pass
#                     self._http_session = None
#                 raise
#         return self._http_session

#     async def cleanup(self) -> None:
#         try:
#             if self._http_session and not self._http_session.closed:
#                 await self._http_session.close()
#         except Exception as e:
#             logger.warning(f"Error during cleanup: {e}")

#     def __del__(self) -> None:
#         try:
#             if hasattr(self, "_http_session") and self._http_session:
#                 if not self._http_session.closed:
#                     logger.warning("HTTP session not properly closed in FlowRunner")
#         except Exception:
#             pass

#     def _collect_metrics(self) -> dict[str, Any]:
#         avg_confidence = (
#             sum(self._confidence_scores) / len(self._confidence_scores)
#             if self._confidence_scores
#             else 0.0
#         )

#         avg_transition_time = (
#             sum(self._transition_times) / len(self._transition_times)
#             if self._transition_times
#             else 0.0
#         )

#         return {
#             **self._metrics,
#             "average_confidence": avg_confidence,
#             "average_transition_time": avg_transition_time,
#             "total_nodes_in_flow": len(self.flow.nodes),
#             "execution_path_length": len(self.context.execution_path),
#             "error_rate": self._metrics["errors"] / max(self._metrics["transitions"], 1),
#         }

#     def _load_and_validate_flow(self):
#         try:
#             self.flow: FlowSpec = load_flow(str(self.flow_path))
#             self._validate_flow_structure()
#             logger.info(f"Loaded and validated flow with {len(self.flow.nodes)} nodes")
#         except Exception as e:
#             logger.error(f"Failed to load flow from {self.flow_path}: {e}")
#             raise

#     def _validate_flow_structure(self) -> None:
#         if self.flow.start_node_id not in self.flow.nodes:
#             raise ValueError(f"Start node '{self.flow.start_node_id}' not found")

#         for node_id, node in self.flow.nodes.items():
#             for edge in node.edges:
#                 if edge.destination_node_id and edge.destination_node_id not in self.flow.nodes:
#                     logger.warning(
#                         f"Edge in node '{node_id}' points to non-existent node "
#                         f"'{edge.destination_node_id}'"
#                     )
#             if node.skip_response_edge and node.skip_response_edge.destination_node_id:
#                 if node.skip_response_edge.destination_node_id not in self.flow.nodes:
#                     logger.warning(
#                         f"Skip response edge in node '{node_id}' points to non-existent node "
#                         f"'{node.skip_response_edge.destination_node_id}'"
#                     )

#         for node_id, node in self.flow.nodes.items():
#             if node.tool_id and node.tool_id not in self.flow.tools:
#                 logger.warning(f"Node '{node_id}' references undefined tool '{node.tool_id}'")

#     def register_function(self, tool_id: str, function: Callable) -> None:
#         self._registered_functions[tool_id] = function
#         logger.info(f"Registered function for tool: {tool_id}")

#     def get_registered_function(self, tool_id: str) -> Optional[Callable]:
#         return self._registered_functions.get(tool_id)

#     def _handle_task_error(self, error: Exception, node_id: str) -> None:
#         self._metrics["errors"] += 1
#         logger.error(f"Task error in node {node_id}: {error}")

#     async def _evaluate_prompt_conditions(
#         self, edges: list[Edge], user_text: str
#     ) -> EdgeEvaluation:
#         if not edges or not user_text:
#             logger.debug(f"No edges ({len(edges) if edges else 0}) or user input ('{user_text}')")
#             return EdgeEvaluation(None, None, 0.0, "No edges or user input")

#         prompt_edges = [e for e in edges if e.destination_node_id and e.transition_condition]
#         logger.debug(f"Found {len(prompt_edges)} prompt edges to evaluate")

#         if not prompt_edges:
#             logger.debug("No prompt edges found")
#             return EdgeEvaluation(None, None, 0.0, "No prompt edges found")

#         # Log the edges being evaluated
#         for i, edge in enumerate(prompt_edges):
#             logger.debug(
#                 f"  Edge {i + 1}: condition='{edge.transition_condition.prompt if edge.transition_condition else 'None'}' -> {edge.destination_node_id}"
#             )

#         try:
#             start_time = time.time()
#             logger.debug(f"Evaluating prompt conditions for user input: '{user_text}'")

#             # Track LLM call
#             self._metrics["llm_calls"] += 1

#             evaluation = await self._structured_edge_evaluation(prompt_edges, user_text)

#             # Track timing and confidence
#             evaluation_time = time.time() - start_time
#             self._transition_times.append(evaluation_time)
#             self._confidence_scores.append(evaluation.confidence)

#             logger.debug(
#                 f"Evaluation result: destination={evaluation.destination_node_id}, confidence={evaluation.confidence:.2f}, time={evaluation_time:.2f}s, reasoning='{evaluation.reasoning}'"
#             )
#             return evaluation

#         except Exception as e:
#             logger.error(f"Error evaluating prompt conditions: {e}")
#             self._metrics["errors"] += 1
#             raise

#     async def _structured_edge_evaluation(
#         self, edges: list[Edge], user_text: str
#     ) -> EdgeEvaluation:
#         edge_options = []
#         for i, edge in enumerate(edges):
#             option = {
#                 "option_number": i + 1,
#                 "destination": edge.destination_node_id,
#                 "condition": edge.transition_condition.prompt if edge.transition_condition else "",
#             }
#             edge_options.append(option)

#         logger.debug(f"Edge options for evaluation: {json.dumps(edge_options, indent=2)}")

#         evaluation_prompt = f"""
# You are evaluating conversation flow transitions. Analyze the user input and
# determine which transition condition best matches.

# User Input: "{user_text}"

# Available Options:
# {json.dumps(edge_options, indent=2)}

# Respond with a JSON object containing:
# {{
#     "selected_option": <number or null if no match>,
#     "confidence": <float between 0.0 and 1.0>,
#     "reasoning": "<explanation of decision>",
#     "user_intent": "<brief description of what user wants>"
# }}

# If no option matches well, set selected_option to null and explain why.
# """

#         logger.debug(f"Sending evaluation prompt to LLM: {evaluation_prompt}")

#         try:
#             edge_ctx = llm.ChatContext()
#             edge_ctx.add_message(role="user", content=evaluation_prompt)

#             response_parts = []
#             async with self.edge_llm.chat(chat_ctx=edge_ctx) as stream:
#                 async for chunk in stream:
#                     if chunk.delta and chunk.delta.content:
#                         response_parts.append(chunk.delta.content)

#             response_text = "".join(response_parts).strip()
#             logger.debug(f"Raw LLM response: '{response_text}'")

#             response_text = clean_json_response(response_text)
#             logger.debug(f"Cleaned LLM response: '{response_text}'")

#             try:
#                 response_data = json.loads(response_text)
#                 logger.debug(f"Parsed JSON response: {response_data}")
#                 decision = TransitionDecision.from_dict(response_data)
#                 logger.debug(
#                     f"Decision object: selected_option={decision.selected_option}, confidence={decision.confidence}, reasoning='{decision.reasoning}'"
#                 )
#             except json.JSONDecodeError as e:
#                 logger.warning(f"Failed to parse JSON response: {response_text}")
#                 raise ValueError(f"Failed to parse JSON response: {response_text}") from e

#             if decision.selected_option and 1 <= decision.selected_option <= len(edges):
#                 selected_edge = edges[decision.selected_option - 1]

#                 logger.info(
#                     f"Edge evaluation SUCCESS: {decision.reasoning} -> {selected_edge.destination_node_id} "
#                     f"(confidence: {decision.confidence:.2f})"
#                 )
#                 return EdgeEvaluation(
#                     selected_edge.id,
#                     selected_edge.destination_node_id,
#                     decision.confidence,
#                     decision.reasoning,
#                 )
#             else:
#                 logger.info(
#                     f"Edge evaluation FAILED: No matching edge selected. Reasoning: {decision.reasoning}"
#                 )
#                 return EdgeEvaluation(None, None, decision.confidence, decision.reasoning)

#         except Exception as e:
#             logger.error(f"Structured evaluation failed: {e}")
#             raise

#     async def _check_global_nodes(self, user_text: str) -> EdgeEvaluation:
#         if not user_text:
#             return EdgeEvaluation(None, None, 0.0, "No user input")

#         global_nodes = []
#         for node_id, node in self.flow.nodes.items():
#             if node.global_node_setting and node.global_node_setting.condition:
#                 global_nodes.append((node_id, node.global_node_setting.condition))

#         if not global_nodes:
#             return EdgeEvaluation(None, None, 0.0, "No global nodes defined")

#         try:
#             options = []
#             for i, (node_id, condition) in enumerate(global_nodes):
#                 options.append(
#                     {"option_number": i + 1, "destination": node_id, "condition": condition}
#                 )

#             evaluation_prompt = f"""
# You are checking if user input triggers any global conversation flow transitions
# (like "talk to human", "end call", etc.).

# User Input: "{user_text}"

# Global Conditions:
# {json.dumps(options, indent=2)}

# Respond with JSON:
# {{
#     "selected_option": <number or null>,
#     "confidence": <float 0.0-1.0>,
#     "reasoning": "<explanation>",
#     "user_intent": "<user's intent>"
# }}

# Global conditions should only trigger on clear, explicit user requests.
# """

#             edge_ctx = llm.ChatContext()
#             edge_ctx.add_message(role="user", content=evaluation_prompt)

#             response_parts = []
#             async with self.edge_llm.chat(chat_ctx=edge_ctx) as stream:
#                 async for chunk in stream:
#                     if chunk.delta and chunk.delta.content:
#                         response_parts.append(chunk.delta.content)

#             response_text = "".join(response_parts).strip()
#             response_text = clean_json_response(response_text)

#             try:
#                 response_data = json.loads(response_text)
#                 decision = TransitionDecision.from_dict(response_data)
#             except json.JSONDecodeError as e:
#                 raise ValueError(f"Failed to parse JSON response: {response_text}") from e

#             if decision.selected_option and 1 <= decision.selected_option <= len(global_nodes):
#                 node_id, condition = global_nodes[decision.selected_option - 1]
#                 logger.info(
#                     f"Global node triggered: '{condition}' -> {node_id} "
#                     f"(confidence: {decision.confidence:.2f})"
#                 )
#                 return EdgeEvaluation(
#                     f"global_{node_id}",
#                     node_id,
#                     decision.confidence,
#                     f"Global condition: {decision.reasoning}",
#                 )
#             else:
#                 return EdgeEvaluation(None, None, decision.confidence, decision.reasoning)

#         except Exception as e:
#             logger.error(f"Error evaluating global nodes: {e}")
#             return EdgeEvaluation(None, None, 0.0, f"Global evaluation failed: {e}")

#     async def _evaluate_transition_conditions(
#         self, node: Node, user_text: Optional[str] = None
#     ) -> EdgeEvaluation:
#         self._metrics["transitions"] += 1
#         logger.debug("=== EVALUATING TRANSITION CONDITIONS ===")
#         logger.debug(f"Node: {node.id} (type: {node.type})")
#         logger.debug(f"User text: '{user_text}'")
#         logger.debug(f"Node has {len(node.edges)} edges")
#         logger.debug(f"Node has skip_response_edge: {node.skip_response_edge is not None}")

#         global_eval = None
#         if user_text:
#             logger.debug("Checking global nodes first...")
#             global_eval = await self._check_global_nodes(user_text)
#             logger.debug(
#                 f"Global evaluation result: destination={global_eval.destination_node_id}, confidence={global_eval.confidence:.2f}"
#             )
#             if global_eval.destination_node_id and global_eval.confidence >= 0.7:
#                 logger.info(f"Global transition: {global_eval.reasoning}")
#                 return global_eval

#         if node.skip_response_edge and node.skip_response_edge.destination_node_id:
#             logger.info(f"Using skip response edge: {node.skip_response_edge.destination_node_id}")
#             return EdgeEvaluation(
#                 node.skip_response_edge.id,
#                 node.skip_response_edge.destination_node_id,
#                 1.0,
#                 "Skip response edge (immediate transition)",
#             )

#         if not node.edges:
#             logger.info("No edges found, ending flow")
#             return EdgeEvaluation(None, None, 0.0, "No edges available")

#         if user_text:
#             logger.debug("Evaluating prompt conditions...")
#             prompt_eval = await self._evaluate_prompt_conditions(node.edges, user_text)
#             logger.debug(
#                 f"Prompt evaluation result: destination={prompt_eval.destination_node_id}, confidence={prompt_eval.confidence:.2f}"
#             )
#             if prompt_eval.destination_node_id:
#                 return prompt_eval

#         if (
#             user_text
#             and global_eval
#             and global_eval.destination_node_id
#             and global_eval.confidence >= 0.5
#         ):
#             logger.info(f"Using lower confidence global transition: {global_eval.reasoning}")
#             return global_eval

#         logger.debug("No transition conditions matched")
#         return EdgeEvaluation(None, None, 0.0, "No transition conditions matched")

#     async def run(self, session: AgentSession) -> None:
#         self._metrics["executions"] += 1
#         node_id: Optional[str] = self.flow.start_node_id
#         iterations = 0
#         start_time = time.time()
#         node_iteration_count: dict[str, int] = {}  # Track iterations per node

#         logger.info(f"Starting flow execution from node: {node_id}")

#         try:
#             while node_id and iterations < self.max_iterations:
#                 iterations += 1

#                 # Track per-node iterations for self-loops
#                 node_iteration_count[node_id] = node_iteration_count.get(node_id, 0) + 1

#                 node = self.flow.nodes.get(node_id)
#                 if not node:
#                     raise RuntimeError(f"Node '{node_id}' not found in flow")

#                 logger.info(
#                     f"Entering node: {node_id} (type: {node.type}) "
#                     f"[global iteration {iterations}, node iteration {node_iteration_count[node_id]}]"
#                 )

#                 try:
#                     transition_result = None

#                     if node.type == "conversation":
#                         # ConversationTask handles its own transitions and returns TransitionResult
#                         task = ConversationTask(node, self)
#                         transition_result = await task

#                     elif node.type == "function":
#                         self._metrics["function_calls"] += 1
#                         task = FunctionTask(node, self)
#                         # FunctionTask handles its own transitions and returns TransitionResult
#                         transition_result = await task

#                     elif node.type == "end":
#                         logger.info("Flow execution completed at end node")
#                         if node.instruction:
#                             if node.instruction.type == "prompt":
#                                 await session.generate_reply(instructions=node.instruction.text)
#                             elif node.instruction.type == "static_text":
#                                 await session.say(node.instruction.text)
#                         break

#                     elif node.type == "transfer_call":
#                         logger.info(f"Transfer call node reached: {node_id}")
#                         await self._handle_transfer_call(node, session)
#                         break

#                     elif node.type == "gather_input":
#                         self._metrics["function_calls"] += 1
#                         task = GatherInputTask(node, self)
#                         # GatherInputTask handles its own transitions and returns TransitionResult
#                         transition_result = await task

#                     else:
#                         logger.warning(f"Unhandled node type: {node.type}")
#                         transition_result = None

#                     # Check if we got a valid transition result
#                     if (
#                         transition_result is not None
#                         and transition_result.destination_node_id
#                         and transition_result.destination_node_id != node_id
#                     ):
#                         logger.info(
#                             f"Transitioning to {transition_result.destination_node_id} "
#                             f"(edge: {transition_result.edge_id}, "
#                             f"confidence: {transition_result.confidence:.2f}, "
#                             f"reason: {transition_result.reasoning})"
#                         )
#                         # Store transition context for potential use by next node
#                         self.context.set_variable(
#                             "last_transition",
#                             {
#                                 "edge_id": transition_result.edge_id,
#                                 "user_text": transition_result.user_text,
#                                 "confidence": transition_result.confidence,
#                                 "reasoning": transition_result.reasoning,
#                             },
#                         )
#                         node_id = transition_result.destination_node_id
#                     elif (
#                         transition_result is not None
#                         and transition_result.destination_node_id == node_id
#                     ):
#                         logger.info(
#                             f"Self-loop detected at node {node_id}, "
#                             f"continuing with new task instance "
#                             f"(iteration {node_iteration_count[node_id]})"
#                         )
#                         # Continue the loop with the same node_id,
#                         # which will create a new task instance
#                         continue
#                     elif transition_result is None:
#                         logger.error(f"Task failed in node {node_id}, ending flow")
#                         await session.say(
#                             "I'm sorry, I encountered an issue and need to end our conversation."
#                         )
#                         break
#                     else:
#                         logger.info("No valid transition found, ending flow")
#                         break

#                 except Exception as e:
#                     logger.error(f"Error executing node {node_id}: {e}")
#                     await self._handle_node_error(e, node_id, session)
#                     break

#             if iterations >= self.max_iterations:
#                 logger.error(f"Flow execution exceeded maximum iterations ({self.max_iterations})")
#                 await session.say(
#                     "I'm sorry, but I've encountered an issue. "
#                     "Let me transfer you to a human agent."
#                 )
#                 raise RuntimeError("Flow execution exceeded maximum iterations")

#             execution_time = time.time() - start_time
#             logger.info(
#                 f"Flow execution finished in {execution_time:.2f}s after {iterations} iterations"
#             )
#             logger.info(f"Execution path: {' -> '.join(self.context.execution_path)}")

#         except Exception as e:
#             self._metrics["errors"] += 1
#             logger.error(f"Flow execution failed: {e}")
#             # Cleanup on error
#             try:
#                 await self.cleanup()
#             except Exception as cleanup_error:
#                 logger.warning(f"Error during cleanup after flow failure: {cleanup_error}")
#             raise

#     async def _handle_transfer_call(self, node: Node, session: AgentSession) -> None:
#         """Handle call transfer logic"""
#         logger.info(f"Processing transfer call for node {node.id}")

#         # TODO: Implement transfer call logic

#         if node.instruction:
#             if node.instruction.type == "prompt":
#                 await session.generate_reply(instructions=node.instruction.text)
#             elif node.instruction.type == "static_text":
#                 await session.say(node.instruction.text)

#         if node.transfer_destination:
#             logger.info(
#                 f"Would transfer to: {node.transfer_destination.type} - "
#                 f"{node.transfer_destination.number}"
#             )

#         if node.transfer_option:
#             logger.info(f"Transfer type: {node.transfer_option.type}")

#     async def _handle_node_error(
#         self, error: Exception, node_id: str, session: AgentSession
#     ) -> None:
#         logger.error(f"Node error in {node_id}: {error}")

#         try:
#             await session.say(
#                 "I'm sorry, I encountered an issue. Let me try to help you in a different way."
#             )
#         except Exception as say_error:
#             logger.error(f"Failed to send error message: {say_error}")

#     @property
#     def execution_history(self) -> list[str]:
#         return self.context.execution_path.copy()

#     @property
#     def metrics(self) -> dict[str, Any]:
#         return self._collect_metrics()

#     def get_context_summary(self) -> dict[str, Any]:
#         return {
#             "execution_path": self.context.execution_path,
#             "variables": self.context.variables,
#             "function_results": self.context.function_results,
#             "conversation_length": len(self.context.conversation_history),
#             "execution_time": time.time() - self.context.start_time,
#             "checkpoints": len(self.context.checkpoints),
#             "metrics": self._collect_metrics(),
#         }


# class FlowAgent(Agent):
#     def __init__(
#         self,
#         path: str,
#         edge_evaluator_llm: llm.LLM,
#         initial_context: Optional[dict[str, Any]] = None,
#         max_iterations: int = 100,
#         **kwargs,
#     ) -> None:
#         super().__init__(instructions="", **kwargs)
#         self.runner = FlowRunner(path, edge_evaluator_llm, initial_context, max_iterations)

#     async def on_enter(self) -> None:
#         try:
#             logger.info(f"Starting flow agent with flow: {self.runner.flow.conversation_flow_id}")
#             current_task = asyncio.current_task()
#             if current_task is not None:
#                 _set_activity_task_info(current_task, inline_task=True)
#             await self.runner.run(self.session)

#         except Exception as e:
#             logger.error(f"Flow agent execution failed: {e}")

#             try:
#                 await self.session.say(
#                     "I apologize, but I'm experiencing technical difficulties. "
#                     "Please contact our support team for assistance."
#                 )
#             except Exception as fallback_error:
#                 logger.error(f"Even fallback message failed: {fallback_error}")
#             raise
#         finally:
#             # Cleanup resources
#             try:
#                 await self.runner.cleanup()
#             except Exception as cleanup_error:
#                 logger.warning(f"Cleanup error: {cleanup_error}")

#     def register_tool_function(self, tool_id: str, function: Callable) -> None:
#         self.runner.register_function(tool_id, function)

#     @property
#     def execution_history(self) -> list[str]:
#         return self.runner.execution_history

#     @property
#     def flow_metrics(self) -> dict[str, Any]:
#         return self.runner.metrics

#     def get_flow_context(self) -> dict[str, Any]:
#         return self.runner.get_context_summary()
