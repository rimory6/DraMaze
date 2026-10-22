from urllib.parse import urlparse


def validate_api_settings(base_url, model):
    if not base_url or not base_url.strip():
        raise ValueError("Set DRAMA_LLM_BASE_URL to your model API base URL")
    parsed = urlparse(base_url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("DRAMA_LLM_BASE_URL must be an HTTP or HTTPS API base URL")
    if parsed.hostname.endswith(".example"):
        raise ValueError(
            "Replace the example DRAMA_LLM_BASE_URL with your actual model API base URL"
        )
    if not model or not model.strip() or model.strip() == "your-model-name":
        raise ValueError(
            "Set DRAMA_LLM_MODEL to the model name or ID provided by your API service"
        )


def chat_completions_url(base_url):
    base_url = (base_url or "").strip().rstrip("/")
    if not base_url:
        return ""
    return (
        base_url
        if base_url.endswith("/chat/completions")
        else base_url + "/chat/completions"
    )
