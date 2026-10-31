#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local GGUF inference using llama-cpp-python."""

from __future__ import annotations

from functools import lru_cache
import os
import re
import threading
from typing import Any, Dict, List

from soprano_qa.security import (
    SecurityViolation, screen_generated_answer, semantic_guard,
    select_question_evidence, serialize_untrusted,
)
from soprano_qa.structured_answer import (
    NO_GROUNDED_ANSWER_SENTINEL,
    STRUCTURED_ANSWER_POLICY,
    answer_schema,
    render_structured_answer,
)


_INFERENCE_LOCK = threading.Lock()


class GenerationBackendUnavailable(RuntimeError):
    """Raised before retrieval when the configured answer model cannot load."""


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

    return Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        chat_format=chat_format,
        verbose=False,
    )


def _configured_chat_format(llm_settings: Dict[str, Any]) -> str | None:
    """Use an explicit override only when one is actually configured.

    The production configuration omits this field so llama.cpp reads the
    official chat template embedded in the Qwen GGUF metadata.
    """

    if "chat_format" not in llm_settings:
        return None
    return str(llm_settings["chat_format"]).strip() or None


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
            chat_format=_configured_chat_format(llm_settings),
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
    *,
    security_context: Dict[str, Any] | None = None,
) -> str:
    # Public answer paths supply the frozen question/evidence bundle. Guard
    # calls omit this argument, so they cannot recursively invoke themselves.
    # Run outside the inference lock: each completion acquires it separately.
    completion_options: Dict[str, Any] = {}
    if security_context is not None:
        if not messages or messages[0].get("role") != "system":
            raise SecurityViolation()
        semantic_guard(
            model_path, llm_settings, security_context,
            stage="admission", generate_fn=generate,
        )
        selected_context = select_question_evidence(
            model_path, llm_settings, security_context, generate_fn=generate,
        )
        if selected_context is not security_context:
            security_context = selected_context
            messages = [messages[0], {"role": "user", "content": serialize_untrusted(security_context)}]
        # Citation IDs are selected by the model in constrained data, then
        # rendered by application code and verified against the cited text.
        # This prevents a missing inline label from invalidating useful prose.
        # Keep one leading policy message. A trailing system turn after the
        # evidence separates the output format from the grounding contract
        # and can encourage the model to summarize the entire payload.
        messages = [
            {**messages[0], "content": STRUCTURED_ANSWER_POLICY + "\n" + messages[0]["content"]},
            *messages[1:],
        ]
        completion_options["response_format"] = {
            "type": "json_object", "schema": answer_schema(security_context),
        }
    # A denied output may receive one new composition under the same policy.
    # Admission denials, malformed verdicts, unsafe text and backend failures
    # never enter this path. A replacement still needs every normal check.
    for attempt in range(2 if security_context is not None else 1):
        # llama.cpp completions are not reentrant.
        with _INFERENCE_LOCK:
            llm = load_llama(
                model_path=model_path,
                n_ctx=int(llm_settings.get("n_ctx", 8192)),
                n_gpu_layers=int(llm_settings.get("n_gpu_layers", -1)),
                chat_format=_configured_chat_format(llm_settings),
            )
            response = llm.create_chat_completion(
                messages=messages,
                temperature=float(llm_settings.get("temperature", 0.0)),
                top_p=float(llm_settings.get("top_p", 0.8)),
                max_tokens=int(llm_settings.get("max_tokens", 768)),
                **completion_options,
            )
        answer = strip_thinking(response["choices"][0]["message"]["content"])
        if security_context is None:
            return answer
        answer = render_structured_answer(answer, security_context)
        if answer == NO_GROUNDED_ANSWER_SENTINEL:
            # An explicit abstention has no prose to assess. The public answer
            # finalizer turns it into the normal fixed unavailable response.
            return answer
        screen_generated_answer(answer, security_context)
        try:
            semantic_guard(
                model_path, llm_settings, {**security_context, "draft": str(answer),
                                          "composition": list(getattr(answer, "parts", ()))},
                stage="output", generate_fn=generate,
            )
        except SecurityViolation as exc:
            if exc.reason != "security_output_rejected" or attempt:
                raise
            messages = [
                {**messages[0], "content": messages[0]["content"] + "\n"
                 "The previous composition did not pass review. Compose a new answer "
                 "under the same rules. Answer only the actual question. For general "
                 "music concepts, use accurate background without unrelated source "
                 "summaries. For source-based advice retain the original actor, "
                 "location, conditions, alternatives and certainty. If a requested "
                 "piece fact is unknown, state that limit; do not invent it. All "
                 "security, language and structured-output requirements still apply."},
                *messages[1:],
            ]
            continue
        return answer
    raise AssertionError("Composition loop ended without a verdict")
