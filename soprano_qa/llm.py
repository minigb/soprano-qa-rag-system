#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local GGUF inference using llama-cpp-python."""

from __future__ import annotations

from functools import lru_cache
import re
import threading
from typing import Any, Dict, List


_INFERENCE_LOCK = threading.Lock()


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
