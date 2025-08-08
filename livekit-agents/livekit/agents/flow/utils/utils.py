import re
from typing import Any

from livekit.agents import llm


def clean_json_response(response_text: str) -> str:
    response_text = re.sub(r"```json\s*", "", response_text)
    response_text = re.sub(r"```\s*$", "", response_text)
    response_text = re.sub(r"^```\s*", "", response_text)

    response_text = response_text.strip()

    return response_text


async def stream_chat_to_text(model: llm.LLM[Any], prompt: str) -> str:
    """helper function to stream chat to text"""
    ctx = llm.ChatContext()
    ctx.add_message(role="user", content=prompt)

    aggregated = ""
    async with model.chat(chat_ctx=ctx) as stream:
        async for chunk in stream:
            if chunk.delta and chunk.delta.content:
                aggregated += chunk.delta.content

    return aggregated
