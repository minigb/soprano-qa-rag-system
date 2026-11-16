#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The answer model: local GGUF inference by default, or Vertex Gemini.

``llm.provider`` in the settings, or the SOPRANO_QA_LLM_PROVIDER environment
variable, selects the model; everything else in the pipeline is the same.

- ``local`` (default): the configured GGUF checkpoint (Qwen3-14B) through
  llama-cpp-python.
- ``vertex_gemini``: Gemini on Vertex AI (soprano_qa/vertex_gemini.py), for
  hosts without a GPU such as the soprano-lab web demo. It needs
  requirements-vertex.txt and Google application-default credentials.
"""

from __future__ import annotations

from functools import lru_cache
import os
import re
import threading
from typing import Any, Dict, List


PROVIDERS = ("local", "vertex_gemini")

_INFERENCE_LOCK = threading.Lock()


class GenerationBackendUnavailable(RuntimeError):
    """Raised before retrieval when the configured answer model cannot load."""


class GenerationFailed(RuntimeError):
    """The answer model was reachable but returned no usable completion.

    A remote provider raises this for errors, timeouts and incomplete answers,
    so callers can handle every provider's failures the same way.
    """


def provider(llm_settings: Dict[str, Any]) -> str:
    """Return the configured answer model provider."""

    name = str(llm_settings.get("provider", "local"))
    if name not in PROVIDERS:
        raise GenerationBackendUnavailable(
            "Unknown llm.provider %r; expected one of %s" % (name, ", ".join(PROVIDERS))
        )
    return name


@lru_cache(maxsize=1)
def load_llama(
    model_path: str,
    n_ctx: int,
    n_gpu_layers: int,
    chat_format: str | None = None,
) -> Any:
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise RuntimeError(
            "llama-cpp-python is not installed. Run: "
            "conda run -n soprano-qa python -m pip install -r requirements.txt"
        ) from exc

    native_controlled = chat_format in {"native_non_thinking", "native_controlled"}
    model = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        chat_format=None if native_controlled else chat_format,
        verbose=False,
    )
    if native_controlled:
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter

        template = model.metadata.get("tokenizer.chat_template", "")
        if "enable_thinking" not in template:
            model.close()
            raise ValueError("The checkpoint template has no enable_thinking switch")
        # The embedded template, tokenizer and stop tokens remain authoritative.
        # This is a trusted template parameter, never inferred from user text.
        eos, bos = model.token_eos(), model.token_bos()
        model.chat_handler = Jinja2ChatFormatter(
            template="{%- set enable_thinking = false -%}" + template,
            eos_token=model._model.token_get_text(eos) if eos != -1 else "",
            bos_token=model._model.token_get_text(bos) if bos != -1 else "",
            stop_token_ids=[eos] if eos != -1 else [],
        ).to_chat_handler()
    return model


def _configured_chat_format(llm_settings: Dict[str, Any]) -> str | None:
    """Use an explicit override only when one is actually configured.

    The production configuration omits this field so llama.cpp reads the
    official chat template embedded in the Qwen GGUF metadata.
    """

    if "chat_format" not in llm_settings:
        return None
    return str(llm_settings["chat_format"]).strip() or None


def _load_local(model_path: str, llm_settings: Dict[str, Any]) -> Any:
    return load_llama(
        model_path=model_path,
        n_ctx=int(llm_settings.get("n_ctx", 8192)),
        n_gpu_layers=int(llm_settings.get("n_gpu_layers", -1)),
        chat_format=_configured_chat_format(llm_settings),
    )


def validate_generation_requirements(
    model_path: str,
    llm_settings: Dict[str, Any],
) -> None:
    """Eagerly validate the configured answer model before corpus work.

    For the local model, loading catches a broken native llama.cpp
    installation and an unreadable or incompatible GGUF, not merely a missing
    Python module or path. For Vertex Gemini, it checks the configuration,
    SDK and credentials without a paid request.
    """

    if provider(llm_settings) == "vertex_gemini":
        from soprano_qa.vertex_gemini import validate_requirements
        try:
            validate_requirements(llm_settings)
        except Exception as exc:
            raise GenerationBackendUnavailable(
                "Vertex Gemini configuration/ADC unavailable; install requirements-vertex.txt "
                "and source your Vertex env.sh. Error type: " + type(exc).__name__
            ) from exc
        return
    configured_path = str(model_path or "").strip()
    if not configured_path or not os.path.isfile(configured_path):
        display_path = configured_path or "(model_path is not configured)"
        raise GenerationBackendUnavailable(
            "Required answer-generation checkpoint not found: %s. Run "
            "`conda run -n soprano-qa python scripts/download_model.py` "
            "before starting a generation run." % display_path
        )
    try:
        _load_local(configured_path, llm_settings)
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
    """Return one completion from the configured answer model."""

    if provider(llm_settings) == "vertex_gemini":
        from soprano_qa.vertex_gemini import generate as vertex_generate
        return vertex_generate(messages, llm_settings)

    completion_options: Dict[str, Any] = {}
    if "seed" in llm_settings:
        completion_options["seed"] = int(llm_settings["seed"])
    for name in ("top_k", "min_p"):
        if name in llm_settings:
            completion_options[name] = llm_settings[name]
    if _configured_chat_format(llm_settings) in {"native_non_thinking", "native_controlled"}:
        # The template switch disables reasoning; soft /think commands are
        # not used. Never rewrite untrusted user or evidence content.
        messages = [{**m, "content": re.sub(r"(?m)^/(?:no_)?think\s*$", "", m["content"])}
                    if m["role"] == "system" else m for m in messages]
    # llama.cpp completions are not reentrant.
    with _INFERENCE_LOCK:
        llm = _load_local(model_path, llm_settings)
        response = llm.create_chat_completion(
            messages=messages,
            temperature=float(llm_settings.get("temperature", 0.0)),
            top_p=float(llm_settings.get("top_p", 0.8)),
            max_tokens=int(llm_settings.get("max_tokens", 768)),
            **completion_options,
        )
    return strip_thinking(response["choices"][0]["message"]["content"])
