#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local GGUF inference using llama-cpp-python."""

from __future__ import annotations

from functools import lru_cache
import json
import os
import re
import threading
from typing import Any, Dict, List


_INFERENCE_LOCK = threading.Lock()


class GenerationBackendUnavailable(RuntimeError):
    """Raised before retrieval when the configured answer model cannot load."""


class StructuredGenerationError(RuntimeError):
    """Raised when the sole constrained model call violates its contract."""


@lru_cache(maxsize=1)
def load_llama(model_path: str, n_ctx: int, n_gpu_layers: int, chat_format: str = "chatml") -> Any:
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise RuntimeError(
            "llama-cpp-python is not installed. Run: "
            "conda run -n soprano-qa python -m pip install -r requirements.txt"
        ) from exc

    return Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        chat_format=chat_format,
        verbose=False,
    )


def validate_generation_requirements(
    model_path: str,
    llm_settings: Dict[str, Any],
) -> None:
    """Eagerly validate the sole configured generation backend and model.

    A generation request must not begin corpus work and then silently switch
    to extraction or internal model knowledge. Loading here also catches a
    broken native llama.cpp installation and an unreadable/incompatible GGUF,
    not merely a missing Python module or path.
    """

    configured_path = str(model_path or "").strip()
    if not configured_path or not os.path.isfile(configured_path):
        display_path = configured_path or "(model_path is not configured)"
        raise GenerationBackendUnavailable(
            "Required answer-generation checkpoint not found: %s. Run "
            "`conda run -n soprano-qa python scripts/download_model.py` "
            "before starting a generation run." % display_path
        )
    try:
        load_llama(
            model_path=configured_path,
            n_ctx=int(llm_settings.get("n_ctx", 8192)),
            n_gpu_layers=int(llm_settings.get("n_gpu_layers", -1)),
            chat_format=str(llm_settings.get("chat_format", "chatml")),
        )
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise GenerationBackendUnavailable(
            "Required answer-generation backend/model could not be loaded: "
            "%s. Activate the `soprano-qa` Conda environment, install "
            "requirements.txt, and verify the configured GGUF checkpoint."
            % configured_path
        ) from exc


def strip_thinking(text: str) -> str:
    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    unmatched = re.search(r"<think>", text, flags=re.IGNORECASE)
    if unmatched:
        text = text[: unmatched.start()]
    return re.sub(r"</?think>", "", text, flags=re.IGNORECASE).strip()


def generate(
    model_path: str,
    messages: List[Dict[str, str]],
    llm_settings: Dict[str, Any],
) -> str:
    # Consumers may serve requests concurrently, while llama.cpp completions
    # are not reentrant. Keep lazy model loading and inference within one
    # process-wide critical section.
    with _INFERENCE_LOCK:
        llm = load_llama(
            model_path=model_path,
            n_ctx=int(llm_settings.get("n_ctx", 8192)),
            n_gpu_layers=int(llm_settings.get("n_gpu_layers", -1)),
            chat_format=str(llm_settings.get("chat_format", "chatml")),
        )
        response = llm.create_chat_completion(
            messages=messages,
            temperature=float(llm_settings.get("temperature", 0.0)),
            top_p=float(llm_settings.get("top_p", 0.8)),
            max_tokens=int(llm_settings.get("max_tokens", 768)),
        )
    return strip_thinking(response["choices"][0]["message"]["content"])


def generate_json(
    model_path: str,
    messages: List[Dict[str, str]],
    llm_settings: Dict[str, Any],
    *,
    schema: Dict[str, Any],
    max_tokens: int = 384,
) -> Dict[str, Any]:
    """Make one grammar-constrained JSON call to the required local model.

    This deliberately has no retry, repair, alternate model, or parser
    fallback.  A selector contract failure is a backend failure and must stop
    the request instead of silently changing how evidence is chosen.
    """

    with _INFERENCE_LOCK:
        llm = load_llama(
            model_path=model_path,
            n_ctx=int(llm_settings.get("n_ctx", 8192)),
            n_gpu_layers=int(llm_settings.get("n_gpu_layers", -1)),
            chat_format=str(llm_settings.get("chat_format", "chatml")),
        )
        response = llm.create_chat_completion(
            messages=messages,
            temperature=0.0,
            top_p=float(llm_settings.get("top_p", 0.8)),
            max_tokens=min(
                int(llm_settings.get("max_tokens", 768)),
                max_tokens,
            ),
            response_format={
                "type": "json_object",
                "schema": schema,
            },
        )
    raw = strip_thinking(response["choices"][0]["message"]["content"])
    try:
        parsed = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise StructuredGenerationError(
            "Constrained local-model output was not valid JSON"
        ) from exc
    if not isinstance(parsed, dict):
        raise StructuredGenerationError(
            "Constrained local-model output must be a JSON object"
        )
    return parsed
