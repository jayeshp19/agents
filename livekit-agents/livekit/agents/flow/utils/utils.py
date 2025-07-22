import re


def clean_json_response(response_text: str) -> str:
    """Clean JSON response text by removing code block markers and extra whitespace."""
    response_text = re.sub(r"```json\s*", "", response_text)
    response_text = re.sub(r"```\s*$", "", response_text)
    response_text = re.sub(r"^```\s*", "", response_text)

    response_text = response_text.strip()

    return response_text
