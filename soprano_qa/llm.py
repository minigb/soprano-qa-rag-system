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

from soprano_qa.question_intent import explicit_general_intent
from soprano_qa.meter import basic_meter_explanation

from soprano_qa.security import (
    SecurityViolation, screen_generated_answer, semantic_guard,
    select_question_evidence, serialize_untrusted,
    SECURITY_SYSTEM_POLICY,
)
from soprano_qa.structured_answer import (
    NO_GROUNDED_ANSWER_SENTINEL,
    STRUCTURED_ANSWER_POLICY,
    SOURCE_PRESERVATION_POLICY,
    SOURCE_SELECTION_POLICY,
    GENERAL_MUSIC_SYSTEM_PROMPT,
    GENERAL_PRACTICE_SYSTEM_PROMPT,
    GENERAL_PRACTICE_SYSTEM_PROMPT_KO,
    GENERAL_MUSIC_FORMAT_POLICY,
    answer_schema,
    render_structured_answer,
)
from soprano_qa.source_translation import translate_source_parts
from soprano_qa.source_comparison import (
    sources_disagree, render_comparison, COMPARISON_POLICY, comparison_language_policy,
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

    A generation request requires this backend before corpus work. Authorized
    general explanations use the same model, not a backend fallback. Loading catches a
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
    response_schema: Dict[str, Any] | None = None,
) -> str:
    # Public answer paths supply the frozen question/evidence bundle. Guard
    # calls omit this argument, so they cannot recursively invoke themselves.
    # Run outside the inference lock: each completion acquires it separately.
    completion_options: Dict[str, Any] = {}
    if response_schema is not None:
        if security_context is not None:
            raise ValueError("A guarded answer must use its own application schema")
        completion_options["response_format"] = {"type": "json_object", "schema": response_schema}
    separate_translation = False
    if security_context is not None:
        if not messages or messages[0].get("role") != "system":
            raise SecurityViolation()
        semantic_guard(
            model_path, llm_settings, security_context,
            stage="admission", generate_fn=generate,
        )
        # Short questions inherit the selected work's context. A question-only
        # intent classifier cannot establish that all reviewed answers are
        # irrelevant: it misrouted score and performance questions as abstract
        # theory. Judge relevance using the actual passages before discarding
        # them, and permit background in either composition path.
        preserve_sources = llm_settings.get("source_preservation") == "verbatim_or_translation"
        calculated = basic_meter_explanation(security_context) if preserve_sources else None
        if calculated is not None:
            # Numeric beat units are derived from ordinary meter rules, rather
            # than guessed by the model. Both security boundaries still run.
            context = {**security_context, "Retrieved evidence": []}
            answer = render_structured_answer(json.dumps({"sentences": [
                {"kind": "background", "evidence": [], "text": calculated}
            ]}, ensure_ascii=False), context)
            screen_generated_answer(answer, context)
            semantic_guard(model_path, llm_settings,
                           {**context, "draft": str(answer), "composition": list(answer.parts)},
                           stage="output", generate_fn=generate)
            answer.answer_basis = "calculated_music_knowledge"
            answer.generation_mode = "llm_guarded_calculation"
            return answer
        general_intent = explicit_general_intent(security_context) if preserve_sources else None
        if general_intent in {"practice", "universal"}:
            # Explicit global practice/universal claims are not established by
            # local performance advice. Named works and local referents have
            # already kept their evidence path. All guards remain mandatory.
            selected_context = {**security_context, "Retrieved evidence": []}
        else:
            selected_context = select_question_evidence(
                model_path, llm_settings, security_context, generate_fn=generate,
                assess_single=general_intent == "definition",
            )
        if selected_context is not security_context:
            security_context = selected_context
            messages = [messages[0], {"role": "user", "content": serialize_untrusted(security_context)}]
        # Conditional alternatives remain intact. Only differing accounts of
        # the same fact need an explicit, bounded source comparison.
        synthesize_sources = preserve_sources and sources_disagree(
            model_path, llm_settings, security_context, generate_fn=generate,
        )
        if synthesize_sources:
            preserve_sources = False
            security_context = {**security_context, "source_synthesis": True}
            security_context.pop("source_preservation", None)
            security_context.pop("separate_source_translation", None)
        if preserve_sources:
            security_context = {**security_context, "source_preservation": True}
            separate_translation = (security_context.get("answer_language") == "en"
                                    and llm_settings.get("evidence_translation") in {"segmented", "reviewed_catalog"})
            if separate_translation:
                security_context["separate_source_translation"] = True
        messages = [messages[0], {"role": "user", "content": serialize_untrusted(security_context)}]
        general_only = not any(r.get("citation_label") for r in security_context.get("Retrieved evidence", []))
        if general_only:
            general_policy = GENERAL_PRACTICE_SYSTEM_PROMPT if general_intent == "practice" else GENERAL_MUSIC_SYSTEM_PROMPT
            if general_intent == "practice" and security_context.get("answer_language", "ko") == "ko":
                general_policy = GENERAL_PRACTICE_SYSTEM_PROMPT_KO
            messages = [{"role": "system", "content": general_policy + "\n" + SECURITY_SYSTEM_POLICY},
                        *messages[1:]]
        elif synthesize_sources:
            # Compare the actual source facts, independent of a question's
            # actor/premise. Admission and final review still receive the full
            # original question; only this extraction view omits it.
            comparison_view = {"answer_language": security_context.get("answer_language", "ko"),
                               "Retrieved evidence": [{key: record[key] for key in
                                   ("citation_label", "answer", "evidence_role", "measure_range", "query_scope_relation")
                                   if key in record} for record in security_context["Retrieved evidence"]]}
            messages = [{"role": "system", "content": SECURITY_SYSTEM_POLICY + "\n"
                         + comparison_language_policy(security_context.get("answer_language", "ko")) + "\n/think"},
                        {"role": "user", "content": serialize_untrusted(comparison_view)}]
        # Citation IDs are selected by the model in constrained data, then
        # rendered by application code and verified against the cited text.
        # This prevents a missing inline label from invalidating useful prose.
        # Keep one leading policy message. A trailing system turn after the
        # evidence separates the output format from the grounding contract
        # and can encourage the model to summarize the entire payload.
        messages = [
            {**messages[0], "content": (GENERAL_MUSIC_FORMAT_POLICY if general_only else
                                       COMPARISON_POLICY if synthesize_sources else
                                       SOURCE_SELECTION_POLICY if separate_translation else
                                       SOURCE_PRESERVATION_POLICY if preserve_sources else STRUCTURED_ANSWER_POLICY)
             + "\n" + messages[0]["content"]},
            *messages[1:],
        ]
        if not synthesize_sources:
            completion_options["response_format"] = {
                "type": "json_object", "schema": answer_schema(security_context),
            }
        # A forced JSON first token suppresses Qwen's comparison reasoning.
        # Only this narrow route uses an unconstrained final completion, whose
        # JSON, excerpts, citations and meaning are still fully validated below.
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
                max_tokens=(2560 if security_context is not None and synthesize_sources
                            else int(llm_settings.get("max_tokens", 768))),
                **completion_options,
            )
        answer = strip_thinking(response["choices"][0]["message"]["content"])
        if security_context is None:
            return answer
        answer = (render_comparison(answer, security_context) if synthesize_sources
                  else render_structured_answer(answer, security_context))
        if answer == NO_GROUNDED_ANSWER_SENTINEL:
            # An explicit abstention has no prose to assess. The public answer
            # finalizer turns it into the normal fixed unavailable response.
            return answer
        if separate_translation:
            answer = translate_source_parts(answer, model_path=model_path,
                                            llm_settings=llm_settings, generate_fn=generate)
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
            if general_intent == "definition" and not general_only:
                # A definition's selected passage can be merely related. A
                # rejected source-based draft gets the SAME one retry as a
                # general explanation, still under full output review. Local
                # questions, unsafe text and failed guards never use this path.
                security_context = {**security_context, "Retrieved evidence": []}
                security_context.pop("separate_source_translation", None)
                separate_translation = False
                synthesize_sources = False
                general_only = True
                messages = [
                    {"role": "system", "content": GENERAL_MUSIC_FORMAT_POLICY + "\n"
                     + GENERAL_MUSIC_SYSTEM_PROMPT + "\n" + SECURITY_SYSTEM_POLICY},
                    {"role": "user", "content": serialize_untrusted(security_context)},
                ]
                completion_options["response_format"] = {
                    "type": "json_object", "schema": answer_schema(security_context),
                }
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
