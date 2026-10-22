from service.drama_by_creativity.api_client import LLM, ModelResponseError
import os
from drama_local.json_response import parse_json_response


class JSONResponseError(ValueError):
    def __init__(self, error, raw_response):
        super().__init__(str(error))
        self.raw_response = raw_response if isinstance(raw_response, str) else None


def query_json(
    messages,
    model_name="your-model-name",
    temperature=0.2,
    max_new_tokens=8192,
):
    client = LLM(
        model=os.getenv("DRAMA_LLM_MODEL") or model_name,
        system_prompt="",
        temperature=temperature,
        top_p=0.95,
        max_new_tokens=max_new_tokens,
        template=None,
    )
    try:
        text = client.request_messages(messages).response
    except ModelResponseError as error:
        raise JSONResponseError(error, client.last_output) from error
    try:
        return (parse_json_response(text), text)
    except Exception as error:
        raise JSONResponseError(error, text) from error
