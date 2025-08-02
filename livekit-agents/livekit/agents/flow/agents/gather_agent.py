import asyncio
import json
import logging
from typing import Any, Optional

from livekit.agents import llm
from livekit.agents.llm import StopResponse

from ..base import BaseFlowAgent
from ..fields import GatherInputVariable

logger = logging.getLogger(__name__)


class GatherInputNode(BaseFlowAgent):
    def __init__(self, extraction_llm: Optional[llm.LLM] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extraction_llm = extraction_llm

        self.collected_data: dict[str, Any] = {}
        self.flagged_fields: set[str] = set()
        self.field_definitions: list[GatherInputVariable] = []

        self._extraction_tasks: list[asyncio.Task] = []
        self._conversation_history_buffer: list[str] = []
        self._completing = False

        self._setup_fields()

    @property
    def instructions(self) -> str:
        required_fields = [f for f in self.field_definitions if f.required]
        optional_fields = [f for f in self.field_definitions if not f.required]
        return self._build_conversation_instructions(required_fields, optional_fields)

    def _setup_fields(self) -> None:
        if not hasattr(self.node, "gather_input_variables"):
            logger.warning(f"Node {self.node.id} missing gather_input_variables")
            self.field_definitions = []
            return

        self.field_definitions = []
        for var_config in self.node.gather_input_variables:
            if isinstance(var_config, GatherInputVariable):
                self.field_definitions.append(var_config)
            elif isinstance(var_config, dict):
                self.field_definitions.append(GatherInputVariable.from_dict(var_config))

        logger.info(
            f"Parallel gather agent {self.node.id} configured with {len(self.field_definitions)} fields"
        )

    async def _on_enter_node(self) -> None:
        self.collected_data.clear()
        self.flagged_fields.clear()
        self._conversation_history_buffer.clear()

        prefilled_data = self.flow_context.get_variable(f"gather_data_{self.node.id}", {})
        if prefilled_data:
            self.collected_data.update(prefilled_data)
            self.flagged_fields.update(prefilled_data.keys())

        if hasattr(self.node, "gather_input_instruction") and self.node.gather_input_instruction:
            await self.session.say(self.node.gather_input_instruction)

        if self._all_required_data_collected():
            await self._complete_gathering()

    def _build_conversation_instructions(
        self, required_fields: list[GatherInputVariable], optional_fields: list[GatherInputVariable]
    ) -> str:
        base_context = ""
        if hasattr(self.node, "gather_input_instruction") and self.node.gather_input_instruction:
            base_context = f"Context: {self.node.gather_input_instruction}\n\n"

        instructions = [
            base_context,
            "You are gathering specific information from the user. Your role is to:",
            "1. Stay focused on collecting the required information",
            "2. Be conversational but purposeful - guide the conversation toward gathering the needed data",
            "3. A separate system extracts data automatically, so you don't need to parse responses",
            "4. When the user seems confused, gently clarify what information you need",
        ]

        if required_fields:
            instructions.append("\nRequired information to collect:")
            for field in required_fields:
                instructions.append(f"- {field.description}")

        if optional_fields:
            instructions.append("\nOptional information (collect if provided):")
            for field in optional_fields:
                instructions.append(f"- {field.description}")

        instructions.extend(
            [
                "\nConversation approach:",
                "- Start by acknowledging any information the user provides",
                "- If they give partial information, acknowledge it and ask for the missing parts",
                "- If they seem unsure, provide examples of the format you need",
                "- When asking for remaining required fields, casually mention optional fields too",
                "- Don't push for optional fields if the user doesn't provide them",
                "- Stay patient and helpful throughout the process",
                "- Once you have all required information, the system will automatically proceed",
            ]
        )

        return "\n".join(instructions)

    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        # If we're completing, don't process more input
        if hasattr(self, "_completing") and self._completing:
            raise StopResponse()

        # If transition is pending, don't process more input
        if self._transition_pending:
            raise StopResponse()

        user_text = new_message.text_content or ""
        logger.debug(f"Parallel gather agent received: '{user_text}'")

        # Update conversation history buffer
        self._conversation_history_buffer.append(f"User: {user_text}")
        if len(self._conversation_history_buffer) > 10:
            self._conversation_history_buffer.pop(0)

        extraction_task = asyncio.create_task(self._extract_data_parallel(user_text))
        self._extraction_tasks.append(extraction_task)

    async def _extract_data_parallel(self, user_text: str) -> None:
        try:
            pending_fields = [
                f for f in self.field_definitions if f.name not in self.flagged_fields
            ]

            logger.debug(
                f"Pending fields: {[f.name for f in pending_fields]}, Flagged: {list(self.flagged_fields)}"
            )

            if not pending_fields:
                return

            extraction_prompt = self._build_extraction_prompt(pending_fields, user_text)
            logger.debug(f"Extraction prompt for '{user_text}': {extraction_prompt[:200]}...")

            if not self.extraction_llm:
                logger.error("No extraction LLM available")
                return

            response_text = ""
            extraction_ctx = llm.ChatContext()
            extraction_ctx.add_message(role="user", content=extraction_prompt)
            async with self.extraction_llm.chat(chat_ctx=extraction_ctx) as stream:
                async for chunk in stream:
                    if chunk.delta and chunk.delta.content:
                        response_text += chunk.delta.content

            extracted_data = self._parse_extraction_response(response_text)
            logger.debug(f"Extracted data: {extracted_data}")

            newly_flagged = await self._validate_and_flag_data(extracted_data)
            logger.debug(f"Newly flagged fields: {newly_flagged}")

            if newly_flagged and self._all_required_data_collected():
                # Only schedule completion if not already completing or transition pending
                if (
                    not (hasattr(self, "_completing") and self._completing)
                    and not self._transition_pending
                ):
                    asyncio.create_task(self._schedule_completion())
                    logger.info(f"All required fields collected: {list(self.flagged_fields)}")

        except Exception as e:
            logger.error(f"Error in parallel data extraction: {e}")

    def _build_extraction_prompt(
        self, pending_fields: list[GatherInputVariable], user_text: str
    ) -> str:
        fields_info = []
        for field_def in pending_fields:
            field_info = {
                "name": field_def.name,
                "type": field_def.type,
                "description": field_def.description,
                "required": field_def.required,
            }

            pattern = field_def.get_validation_pattern()
            if pattern:
                field_info["validation_pattern"] = pattern

            fields_info.append(field_info)

        return f'''
You are a data extraction specialist. Analyze the user's message for specific information.

FIELDS TO EXTRACT:
{json.dumps(fields_info, indent=2)}

USER MESSAGE: "{user_text}"

CONVERSATION CONTEXT: {" ".join(self._conversation_history_buffer[-3:])}

TASK:
1. Extract any fields from above that can be clearly identified from the current message
2. Only extract data you are confident about
3. Consider the conversation context for better accuracy
4. Return JSON with extracted fields or empty object if nothing found

RESPONSE FORMAT:
{{"field_name": "extracted_value", "confidence": 0.9}}

'''

    def _parse_extraction_response(self, response_text: str) -> dict[str, Any]:
        try:
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            else:
                json_text = response_text.strip()

            extracted = json.loads(json_text)

            if not isinstance(extracted, dict):
                logger.debug(f"Extracted data is not a dictionary: {type(extracted)}")
                return {}

            cleaned_data = {}
            for key, value in extracted.items():
                if key != "confidence" and not key.endswith("_confidence"):
                    cleaned_data[key] = value

            return cleaned_data

        except (json.JSONDecodeError, Exception) as e:
            logger.debug(f"Failed to parse extraction response: {e}")
            return {}

    async def _validate_and_flag_data(self, extracted_data: dict[str, Any]) -> set[str]:
        newly_flagged = set()

        for field_name, value in extracted_data.items():
            field_def = next((f for f in self.field_definitions if f.name == field_name), None)
            if not field_def:
                continue

            if field_name in self.flagged_fields:
                continue

            is_valid, error_message = field_def.validate_value(value)

            if is_valid:
                self.collected_data[field_name] = value
                self.flagged_fields.add(field_name)
                newly_flagged.add(field_name)

                logger.info(f"Flagged field '{field_name}' with value: {value}")
            else:
                logger.debug(f"Validation failed for field '{field_name}': {error_message}")

        return newly_flagged

    def _all_required_data_collected(self) -> bool:
        if not self.field_definitions:
            return False
        required_fields = {f.name for f in self.field_definitions if f.required}
        return required_fields.issubset(self.flagged_fields)

    async def _schedule_completion(self) -> None:
        self.session.interrupt()
        await asyncio.sleep(0.1)

        await self._complete_gathering()

    async def _complete_gathering(self) -> None:
        if hasattr(self, "_completing") and self._completing:
            return
        self._completing = True

        logger.info(
            f"Parallel gather agent {self.node.id} completed with flagged fields: {list(self.flagged_fields)}"
        )

        for task in self._extraction_tasks:
            if not task.done():
                task.cancel()

        self.flow_context.set_variable(f"gathered_data_{self.node.id}", self.collected_data)

        for field_name, value in self.collected_data.items():
            self.flow_context.set_variable(field_name, value)

        await self._acknowledge_completion()

        if self.node.edges and len(self.node.edges) > 0:
            destination_node_id = self.node.edges[0].destination_node_id
            if destination_node_id:
                logger.info(
                    f"Gather node {self.node.id} transitioning directly to {destination_node_id}"
                )
                await self._transition_to_node(destination_node_id)
            else:
                logger.error(f"Gather node {self.node.id} edge has no destination")
        else:
            logger.error(
                f"Gather node {self.node.id} has no exit edge - this should not happen due to schema validation"
            )

    async def _acknowledge_completion(self) -> None:
        if not self.collected_data:
            return

        self.session.interrupt()

        await self.session.say(
            "Perfect! I have everything I need. Let me help you with the next step."
        )

    async def _on_exit_node(self) -> None:
        for task in self._extraction_tasks:
            if not task.done():
                task.cancel()

        summary = {
            "node_id": self.node.id,
            "flagged_fields": list(self.flagged_fields),
            "collected_data": dict(self.collected_data),
            "extraction_tasks_count": len(self._extraction_tasks),
            "completion_rate": len(self.flagged_fields) / len(self.field_definitions)
            if self.field_definitions
            else 0,
        }

        self.flow_context.set_variable(f"gather_summary_{self.node.id}", summary)
        logger.info(f"Parallel gather agent {self.node.id} exit summary: {summary}")
