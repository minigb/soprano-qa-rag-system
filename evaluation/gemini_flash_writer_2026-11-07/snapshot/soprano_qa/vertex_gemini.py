"""Optional Vertex AI answer writer, authenticated with the user's Google ADC.

Only the writer is remote. Retrieval and English query translation retain their
existing implementations. There is no answer review, repair or model fallback.
"""
from __future__ import annotations

from functools import lru_cache
import os
import time


class VertexGenerationError(RuntimeError):
    """The provider did not return a complete text answer."""

    def __init__(self, message, *, metadata=None, partial_text=""):
        super().__init__(message)
        self.metadata = metadata or {}
        self.partial_text = partial_text


class VertexText(str):
    """Final public text and request diagnostics, never private thought parts."""


def options(settings):
    if settings.get("pipeline") != "single_pass":
        raise ValueError("Vertex Gemini requires the single_pass pipeline")
    config = settings.get("vertex_gemini") or {}
    model = str(config.get("model", "")).strip()
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global").strip()
    if not model or not project or not location:
        raise ValueError("Configure vertex_gemini.model and source your Vertex env.sh")
    level = str(config.get("thinking_level", "LOW")).upper()
    if level not in {"LOW", "MEDIUM", "HIGH"}:
        raise ValueError("thinking_level must be LOW, MEDIUM or HIGH")
    timeout_ms = int(config.get("timeout_ms", 180000))
    max_tokens = int(config.get("max_output_tokens", settings.get("max_tokens", 768)))
    if timeout_ms <= 0 or not 0 < max_tokens <= 32768:
        raise ValueError("Use a positive timeout and 1..32768 output tokens")
    return model, project, location, level, timeout_ms, max_tokens


@lru_cache(maxsize=4)
def _client(project, location, timeout_ms, credentials_path, cloudsdk_config):
    import google.auth
    from google import genai
    from google.genai import types

    # Explicit ADC credentials keep this on Vertex even if a Developer API
    # key also happens to be present in the calling shell.
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    return genai.Client(
        vertexai=True, project=project, location=location, credentials=credentials,
        http_options=types.HttpOptions(
            api_version="v1", timeout=timeout_ms,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )


def get_client(settings):
    _, project, location, _, timeout_ms, _ = options(settings)
    return _client(project, location, timeout_ms,
                   os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", ""),
                   os.environ.get("CLOUDSDK_CONFIG", ""))


def validate_requirements(settings):
    """Check configuration/SDK/ADC without generating a paid probe answer."""
    get_client(settings)


def build_request(messages, settings):
    """Map roles without changing prompt or evidence text."""
    model, _, _, level, _, max_tokens = options(settings)
    if not messages or messages[0].get("role") != "system":
        raise ValueError("Expected an initial system instruction")
    if not isinstance(messages[0].get("content"), str):
        raise ValueError("System instruction must be text")
    contents = []
    for message in messages[1:]:
        role = {"user": "user", "assistant": "model"}.get(message.get("role"))
        if role is None or not isinstance(message.get("content"), str):
            raise ValueError("Expected text user/assistant messages after the system instruction")
        contents.append({"role": role, "parts": [{"text": message["content"]}]})
    if not contents:
        raise ValueError("Expected at least one user message")
    config = {
        "system_instruction": messages[0]["content"],
        "temperature": float(settings.get("temperature", 0.0)),
        "top_p": float(settings.get("top_p", 0.8)),
        "max_output_tokens": max_tokens,
        "thinking_config": {"thinking_level": level, "include_thoughts": False},
    }
    if "seed" in settings:
        config["seed"] = int(settings["seed"])
    return {"model": model, "contents": contents, "config": config}


def parse_response(response, *, requested_model, elapsed_seconds):
    candidate = response.candidates[0] if response.candidates else None
    reason = getattr(getattr(candidate, "finish_reason", None), "name", None)
    parts = getattr(getattr(candidate, "content", None), "parts", None) or []
    text = "".join(part.text or "" for part in parts if not getattr(part, "thought", False))
    feedback = getattr(response, "prompt_feedback", None)
    blocked = getattr(getattr(feedback, "block_reason", None), "name", None)
    usage = getattr(response, "usage_metadata", None)
    metadata = {
        "provider": "vertex_gemini", "requested_model": requested_model,
        "model_version": getattr(response, "model_version", None),
        "finish_reason": reason, "prompt_block_reason": blocked,
        "elapsed_seconds": elapsed_seconds,
        "usage": usage.model_dump(mode="json", exclude_none=True) if usage else {},
    }
    if reason != "STOP" or not text.strip():
        raise VertexGenerationError(
            "Vertex did not return a complete answer (finish=%s, prompt_block=%s)" % (reason, blocked),
            metadata=metadata, partial_text=text,
        )
    result = VertexText(text.strip())
    result.generation_metadata = metadata
    return result


def generate(messages, settings):
    request = build_request(messages, settings)
    client = get_client(settings)
    started = time.perf_counter()
    response = client.models.generate_content(**request)
    return parse_response(response, requested_model=request["model"],
                          elapsed_seconds=time.perf_counter() - started)
