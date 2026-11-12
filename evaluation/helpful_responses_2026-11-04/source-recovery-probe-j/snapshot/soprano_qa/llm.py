#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local GGUF inference using llama-cpp-python."""

from __future__ import annotations

from functools import lru_cache
import json
import os
import re
import threading
from typing import Any, Callable, Dict, List

from soprano_qa.question_intent import explicit_general_intent, concise_pitch_question
from soprano_qa.meter import basic_meter_explanation

from soprano_qa.security import (
    SecurityViolation, screen_generated_answer, semantic_guard,
    select_question_evidence, serialize_untrusted,
    SECURITY_SYSTEM_POLICY,
)
from soprano_qa.structured_answer import (
    NO_GROUNDED_ANSWER_SENTINEL,
    STRUCTURED_ANSWER_POLICY,
    SEMANTIC_ANSWER_POLICY,
    concise_synthesis_policy, synthesis_evidence_view, source_comparison_policy,
    SOURCE_PRESERVATION_POLICY,
    SOURCE_SELECTION_POLICY,
    GENERAL_MUSIC_SYSTEM_PROMPT,
    GENERAL_PRACTICE_SYSTEM_PROMPT,
    GENERAL_PRACTICE_SYSTEM_PROMPT_KO,
    concise_concept_system_prompt,
    GENERAL_MUSIC_FORMAT_POLICY,
    answer_schema,
    render_structured_answer,
)
from soprano_qa.source_translation import translate_source_parts
from soprano_qa.source_comparison import opening_dynamics_comparison


_INFERENCE_LOCK = threading.Lock()


class GenerationBackendUnavailable(RuntimeError):
    """Raised before retrieval when the configured answer model cannot load."""


class GenerationFinalizationRejected(RuntimeError):
    """A reviewed composition failed the application's final content checks."""


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
        modes = (False, True) if chat_format == "native_controlled" else (False,)
        model._soprano_native_chat_handlers = {
            enabled: Jinja2ChatFormatter(
                template="{%- set enable_thinking = " + str(enabled).lower() + " -%}" + template,
                eos_token=model._model.token_get_text(eos) if eos != -1 else "",
                bos_token=model._model.token_get_text(bos) if bos != -1 else "",
                stop_token_ids=[eos] if eos != -1 else [],
            ).to_chat_handler() for enabled in modes
        }
        model.chat_handler = model._soprano_native_chat_handlers[False]
    return model


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
    finalizer: Callable[[str], str] | None = None,
    warning_finalizer: Callable[[str], str] | None = None,
    general_answer_fallback: bool = False,
) -> str:
    # Public answer paths supply the frozen question/evidence bundle. Guard
    # calls omit this argument, so they cannot recursively invoke themselves.
    # Run outside the inference lock: each completion acquires it separately.
    completion_options: Dict[str, Any] = {}
    if finalizer is not None and security_context is None:
        raise ValueError("Finalization requires the guarded answer path")
    if "seed" in llm_settings:
        completion_options["seed"] = int(llm_settings["seed"])
    for name in ("top_k", "min_p"):
        if name in llm_settings:
            completion_options[name] = llm_settings[name]
    if response_schema is not None:
        if security_context is not None:
            raise ValueError("A guarded answer must use its own application schema")
        completion_options["response_format"] = {"type": "json_object", "schema": response_schema}
    separate_translation = False
    reason_general = False
    reason_synthesis = False
    native_controlled = _configured_chat_format(llm_settings) == "native_controlled"
    supports_reasoning = _configured_chat_format(llm_settings) != "native_non_thinking"
    if not supports_reasoning and llm_settings.get("synthesis_reasoning") is True:
        raise ValueError("Synthesis reasoning conflicts with native_non_thinking")
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
        source_mode = llm_settings.get("source_preservation")
        preserve_sources = source_mode == "verbatim_or_translation"
        synthesize_sources = source_mode == "semantic"
        source_aware = preserve_sources or synthesize_sources
        calculated = basic_meter_explanation(security_context) if source_aware else None
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
        general_intent = explicit_general_intent(security_context) if source_aware else None
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
        compared = opening_dynamics_comparison(security_context) if preserve_sources else None
        if compared is not None:
            context = {k: v for k, v in security_context.items()
                       if k not in ("source_preservation", "separate_source_translation")}
            context["source_synthesis"] = True
            screen_generated_answer(compared, context)
            semantic_guard(model_path, llm_settings,
                           {**context, "draft": str(compared), "composition": list(compared.parts)},
                           stage="output", generate_fn=generate)
            compared.generation_mode = "llm_guarded_source_comparison"
            return compared
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
            concise_general = general_intent != "practice" and concise_pitch_question(security_context)
            reason_general = concise_general and supports_reasoning
            if concise_general:
                general_policy = concise_concept_system_prompt(security_context.get("answer_language", "ko"))
                view = {"question": security_context["question"],
                        "answer_language": security_context.get("answer_language", "ko")}
                messages[1] = {"role": "user", "content": serialize_untrusted(view)}
            if general_answer_fallback:
                from soprano_qa.response_policy import GENERAL_FALLBACK_INSTRUCTIONS
                # With no trusted passage available, allow bounded reasoning
                # to keep conditional choices distinct. The same 14B writes
                # the answer; all reasoning is removed before output review.
                reason_general = supports_reasoning
                general_policy = GENERAL_FALLBACK_INSTRUCTIONS + (
                    "\n한국어 합니다체로 자연스럽게 답합니다.\n/no_think\n"
                    if security_context.get('answer_language', 'ko') == 'ko'
                    else "\nWrite your answer in natural English.\n/no_think\n")
            messages = [{"role": "system", "content": general_policy + "\n" + SECURITY_SYSTEM_POLICY},
                        *messages[1:]]
        # Citation IDs are selected by the model in constrained data, then
        # rendered by application code and verified against the cited text.
        # This prevents a missing inline label from invalidating useful prose.
        # Keep one leading policy message. A trailing system turn after the
        # evidence separates the output format from the grounding contract
        # and can encourage the model to summarize the entire payload.
        messages = [
            {**messages[0], "content": (GENERAL_MUSIC_FORMAT_POLICY if general_only else
                                       SOURCE_SELECTION_POLICY if separate_translation else
                                       SOURCE_PRESERVATION_POLICY if preserve_sources else
                                       SEMANTIC_ANSWER_POLICY if synthesize_sources else STRUCTURED_ANSWER_POLICY)
             + "\n" + messages[0]["content"]},
            *messages[1:],
        ]
        if general_only and general_answer_fallback:
            from soprano_qa.response_policy import GENERAL_FALLBACK_FORMAT
            messages[0]['content'] = GENERAL_FALLBACK_FORMAT + '\n' + general_policy + '\n' + SECURITY_SYSTEM_POLICY
        if synthesize_sources and not general_only and llm_settings.get("synthesis_prompt") == "concise":
            view = synthesis_evidence_view(security_context)
            comparison = bool(view.get("source_disagreements"))
            if comparison:
                security_context = {**security_context, "source_synthesis": True}
            messages = [
                {"role": "system", "content": (source_comparison_policy if comparison else concise_synthesis_policy)(security_context.get("answer_language", "ko"))
                 + "\n" + SECURITY_SYSTEM_POLICY},
                {"role": "user", "content": serialize_untrusted(view)},
            ]
        reason_synthesis = (synthesize_sources and not general_only
                            and llm_settings.get("synthesis_reasoning") is True)
        if reason_general or reason_synthesis:
            # A JSON grammar forces the first token and suppresses Qwen's
            # reasoning. The existing pitch-operation path uses bounded
            # reasoning before drafting its explanation. The final
            # JSON still passes the same strict parser and all output checks.
            if not native_controlled:
                messages[0]["content"] += "\n/think"
            completion_options["top_k"] = int(llm_settings.get("reasoning_top_k", 20))
            completion_options["min_p"] = float(llm_settings.get("reasoning_min_p", 0.0))
        else:
            completion_options["response_format"] = {
                "type": "json_object", "schema": answer_schema(security_context),
            }
    # A denied output or finalization failure shares ONE new composition.
    # Admission denials, malformed verdicts, unsafe text and backend failures
    # never enter this path. A replacement still needs every normal check.
    for attempt in range(2 if security_context is not None else 1):
        if native_controlled:
            # Qwen3.5 uses its template switch, not soft /think commands. Do
            # not rewrite untrusted user/evidence content to control the mode.
            messages = [{**m, "content": re.sub(r"(?m)^/(?:no_)?think\s*$", "", m["content"])}
                        if m["role"] == "system" else m for m in messages]
        # llama.cpp completions are not reentrant.
        with _INFERENCE_LOCK:
            llm = load_llama(
                model_path=model_path,
                n_ctx=int(llm_settings.get("n_ctx", 8192)),
                n_gpu_layers=int(llm_settings.get("n_gpu_layers", -1)),
                chat_format=_configured_chat_format(llm_settings),
            )
            if native_controlled:
                # Set on every call while holding the inference lock: guard
                # calls must not inherit a writer's reasoning prefill.
                llm.chat_handler = llm._soprano_native_chat_handlers[bool(reason_general or reason_synthesis)]
            response = llm.create_chat_completion(
                messages=messages,
                temperature=float(llm_settings.get("reasoning_temperature", 0.6) if reason_general or reason_synthesis
                                  else llm_settings.get("temperature", 0.0)),
                top_p=float(llm_settings.get("reasoning_top_p", 0.95) if reason_general or reason_synthesis
                            else llm_settings.get("top_p", 0.8)),
                max_tokens=(int(llm_settings.get("synthesis_reasoning_max_tokens", 4096))
                            if reason_synthesis else 1536 if reason_general
                            else int(llm_settings.get("max_tokens", 768))),
                **completion_options,
            )
        raw_answer = response["choices"][0]["message"]["content"]
        if native_controlled and (reason_general or reason_synthesis):
            # The native template already opens <think>, so the completion
            # may contain only the closing tag. Never expose that private
            # prefix, including when the generation budget ends mid-thought.
            raw_answer = raw_answer.split("</think>", 1)[1] if "</think>" in raw_answer else ""
        answer = strip_thinking(raw_answer)
        if security_context is None:
            return answer
        if reason_synthesis or reason_general:
            # Reasoning completions may wrap their final JSON in one code fence.
            # Accept only a whole-response wrapper; the exact JSON schema and
            # all content checks still run on its unchanged inner text.
            wrapped = re.fullmatch(r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n```", answer, re.DOTALL)
            if wrapped:
                answer = wrapped.group(1)
        answer = render_structured_answer(answer, security_context)
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
            if finalizer is not None:
                answer.validated_text = finalizer(answer)
        except (SecurityViolation, GenerationFinalizationRejected) as exc:
            if attempt and warning_finalizer is not None and (
                isinstance(exc, GenerationFinalizationRejected)
                or isinstance(exc, SecurityViolation) and exc.reason == "security_output_rejected"
            ):
                # Quality review can request a rewrite, but cannot erase the
                # last usable answer. Release it only after a distinct safety
                # check, then apply presentation sanitization and a visible warning.
                from soprano_qa.security import instruction_guard
                instruction_guard(model_path, llm_settings,
                    {**security_context, "draft": str(answer)}, generate_fn=generate)
                answer.validated_text = warning_finalizer(answer)
                answer.generation_mode = "llm_with_warning"
                answer.generation_validation_warning = (
                    "output_quality_review" if isinstance(exc, SecurityViolation)
                    else "answer_validation_warning"
                )
                return answer
            if attempt or (isinstance(exc, SecurityViolation)
                           and exc.reason != "security_output_rejected"):
                raise
            if isinstance(exc, SecurityViolation) and general_intent == "definition" and not general_only:
                # A definition's selected passage can be merely related. A
                # rejected source-based draft gets the SAME one retry as a
                # general explanation, still under full output review. Local
                # questions, unsafe text and failed guards never use this path.
                security_context = {**security_context, "Retrieved evidence": []}
                security_context.pop("separate_source_translation", None)
                separate_translation = False
                general_only = True
                reason_synthesis = False
                concise_general = concise_pitch_question(security_context)
                reason_general = concise_general and supports_reasoning
                general_policy = (concise_concept_system_prompt(security_context.get("answer_language", "ko"))
                                  if concise_general else GENERAL_MUSIC_SYSTEM_PROMPT)
                view = ({"question": security_context["question"],
                         "answer_language": security_context.get("answer_language", "ko")}
                        if concise_general else security_context)
                messages = [
                    {"role": "system", "content": GENERAL_MUSIC_FORMAT_POLICY + "\n"
                     + general_policy + "\n" + SECURITY_SYSTEM_POLICY
                     + ("\n/think" if reason_general and not native_controlled else "")},
                    {"role": "user", "content": serialize_untrusted(view)},
                ]
                if reason_general:
                    completion_options.pop("response_format", None)
                else:
                    completion_options["response_format"] = {
                        "type": "json_object", "schema": answer_schema(security_context),
                    }
            if (warning_finalizer is not None and synthesize_sources and not general_only
                    and isinstance(exc, SecurityViolation)):
                # Repeating the full initial prompt can reproduce the same
                # unhelpful summary. Keep the same evidence and trust boundary,
                # but give the single revision a direct writing task.
                revision_policy = """질문에 바로 답하는 성악 도우미의 답변을 새로 작성합니다.
자료의 핵심 권고를 자신의 문장으로 연결합니다. 조건, 부정, 선택지, 주체,
적용 범위, 불확실성을 그대로 유지합니다. 서로 다른 자료가 충돌하면 구분합니다.
질문이 방법을 묻는데 자료가 방향만 말한다면, 그 방향을 실제로 연습할 수 있는
구체적인 일반 연습 방법도 제안합니다. 그 방법을 전문가의 말이나 이 곡의
확정된 사실로 쓰지 않습니다. 모르는 악보 표기, 마디 사실, 작곡 의도를 꾸미지 않습니다.
자료의 문장을 옮겨 적거나 같은 뜻을 반복하는 것으로 답변을 끝내지 않습니다.
짧고 자연스럽게 설명하되 질문에 답하는 핵심 조건과 세부 내용은 남깁니다.
자료가 이미 구체적인 답을 주면 그 답을 설명하는 것으로 충분합니다.
질문에 필요하지 않은 일반론이나 원문과 다른 지침을 덧붙이지 않습니다.
자료에 근거한 설명은 kind=evidence와 실제 라벨을 사용합니다.
일반 지식이나 연습 제안은 별도의 kind=background, evidence=[] 항목으로 씁니다.
출력은 {"sentences":[{"kind":"evidence","evidence":["E1"],"text":"자료에 근거한 답변"},
{"kind":"background","evidence":[],"text":"필요할 때 보충하는 일반적인 설명"}]} 형식입니다.
필요한 항목만 사용하고 내부 라벨이나 형식 설명을 본문에 쓰지 않습니다.
"""
                revision_policy += ("한국어 합니다체로 답합니다.\n" if security_context.get('answer_language', 'ko') == 'ko'
                                    else "Write all answer text in natural English.\n")
                messages[0] = {'role': 'system', 'content': revision_policy + SECURITY_SYSTEM_POLICY}
            messages = [
                {**messages[0], "content": messages[0]["content"] + "\n"
                 "The previous composition did not pass review. Compose a new answer "
                 "under the same rules. Answer only the actual question. For general "
                 "music concepts, use accurate background without unrelated source "
                 "summaries. For source-based advice retain the original actor, "
                 "location, conditions, alternatives and certainty. If a requested "
                 "piece fact is unknown, state that limit; do not invent it. All "
                 "security, language and structured-output requirements still apply. "
                 "Distinguish a source's performance guideline from a universal music-theory law. "
                 "When a general rule is debatable, explicitly attribute the source's guidance "
                 "rather than assert it as a universal definition. Accurate general context may "
                 "clarify that distinction without rewriting what the source actually says."},
                *messages[1:],
            ]
            if isinstance(exc, GenerationFinalizationRejected):
                # Never interpolate the exception: it can contain untrusted
                # generated tokens. Only trusted, fixed corrective guidance
                # enters the system message. Evidence and guard policy stay.
                detail = (
                    "The draft's location or frequency claim exceeded its cited scope. "
                    "Limit advice to the actual musical situation and retain its conditions; "
                    "do not turn a local example into a claim about the whole piece."
                    if "overgeneralizes local evidence" in str(exc) else
                    "The draft failed citation or presentation checks. Keep source claims in "
                    "evidence items with their actual supporting labels. Score_lyrics may only "
                    "identify supplied sung words, never establish rhythm or performance advice. "
                    "Do not print internal labels or metadata in prose. Use 마디 for score measures."
                )
                messages[0]["content"] += "\n" + detail
            if warning_finalizer is not None:
                messages[0]["content"] += """
Write the revised answer as a short, connected explanation in your own words.
Use about 2-4 sentences, adding more only to preserve a necessary condition or
alternative. Merge overlapping recommendations and explain each point once.
Do not concatenate or lightly edit entire source paragraphs. Avoid restating
the same advice as both a recommendation and its explanation. Preserve the
sources' uncertainty and contrast between situations while making this concise.
Keep the detail that directly answers the question, such as a duration's endpoint
or a condition for using a technique, even when shortening the explanation.
한국어에서는 같은 조언을 반복하지 말고, 질문에 바로 답하는 짧은 합니다체 설명으로 연결합니다.
"""
                if isinstance(exc, SecurityViolation):
                    messages[0]["content"] += """
For a 'how' question, merely repeating that one should use a technique is not
enough. Add a concrete, optional practice step from general music knowledge in
a background item when the source gives only a broad recommendation. Clearly
introduce that step as a general practice suggestion, not as the expert's advice
or as a fact about the piece. Do not invent an exact score detail, an expert
quotation, or a specific composer's intention.
방법을 묻는 질문인데 자료에는 구체적인 실행 방법 없이 방향만 제시되어 있다면,
자료의 권고를 한 문장으로 설명한 뒤 일반적으로 해 볼 수 있는 연습 방법을
kind=background, evidence=[]인 별도 항목에 한두 문장으로 제안합니다.
방향만 되풀이하고 끝내지 말고, 무엇을 해 보면 되는지 설명합니다.
"""
            continue
        return answer
    raise AssertionError("Composition loop ended without a verdict")
