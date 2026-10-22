import asyncio
import os
from drama_local.api_settings import chat_completions_url, validate_api_settings
from types import SimpleNamespace
import requests


class ModelAPIError(RuntimeError):
    def __init__(self, status_code):
        super().__init__(f"Model API HTTP {status_code}")
        self.retryable = status_code in {408, 409, 425, 429} or status_code >= 500


class ModelResponseError(ValueError):
    pass


class LLM:
    def __init__(
        self,
        *,
        model,
        system_prompt,
        temperature,
        top_p,
        max_new_tokens,
        template,
        base_url=None,
    ):
        self.model = os.getenv("DRAMA_LLM_MODEL") or model
        self.base_url = base_url or os.getenv("DRAMA_LLM_BASE_URL", "")
        self.url = chat_completions_url(self.base_url)
        self.system_prompt = system_prompt
        self.template = template
        self.temperature = temperature
        self.top_p = top_p
        limit = os.getenv("DRAMA_LLM_MAX_TOKENS")
        self.max_tokens = int(limit) if limit is not None else max_new_tokens
        if self.max_tokens is not None and self.max_tokens < 0:
            raise ValueError("DRAMA_LLM_MAX_TOKENS must be >= 0")
        self.timeout = float(os.getenv("DRAMA_LLM_TIMEOUT", "0"))
        if self.timeout < 0:
            raise ValueError("DRAMA_LLM_TIMEOUT must be >= 0")

    async def request(self, params):
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": self.template.render(**params)})
        return await asyncio.to_thread(self.request_messages, messages)

    def request_messages(self, messages):
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": False,
        }
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens
        self.last_output = None
        return self._post(payload)

    def _post(self, payload):
        validate_api_settings(self.base_url, self.model)
        headers = {"Content-Type": "application/json"}
        api_key = os.getenv("DRAMA_LLM_API_KEY")
        if api_key:
            headers["Authorization"] = "Bearer " + api_key
        with requests.Session() as session:
            response = session.post(
                self.url,
                json=payload,
                headers=headers,
                timeout=(5, self.timeout or None),
            )
        if not response.ok:
            raise ModelAPIError(response.status_code)
        data = response.json()
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelResponseError(
                "Model API response missing choices[0].message.content"
            ) from exc
        self.last_output = content
        if choice.get("finish_reason") == "length":
            raise ModelResponseError(
                "Model API output truncated; increase DRAMA_LLM_MAX_TOKENS or shorten the input"
            )
        if not isinstance(content, str) or not content.strip():
            raise ModelResponseError("Model API returned empty content")
        return SimpleNamespace(response=content, usage=data.get("usage", {}))
