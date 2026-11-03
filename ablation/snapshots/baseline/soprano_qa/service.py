"""Reusable service facade for the measure-aware RAG and local-LLM pipeline."""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional

from soprano_qa.answer import (
    GeneratedAnswerRejected,
    NO_CORPUS_EVIDENCE_MESSAGE,
    build_extractive_answer,
    build_messages,
    build_evidence_notices,
    build_messages,
    ensure_corpus,
    finalize_grounded_generated_answer,
    finalize_user_visible_answer,
    has_primary_grounding,
    has_selected_range_grounding,
    is_grounded_insufficiency_answer,
    is_underspecified_ranged_query,
    no_corpus_evidence_message,
    retrieve_generation_evidence,
)
from soprano_qa.llm import (
    generate as generate_llm,
    validate_generation_requirements,
)
from soprano_qa.dense import (
    SearchIndex,
    build_retrieval_index,
    retrieval_abstained_for_ambiguity,
    retrieval_abstained_for_missing_causal_authority,
    retrieval_diagnostics,
    validate_retrieval_requirements,
)
from soprano_qa.retrieval import (
    SearchResult,
    format_measure_range,
    load_corpus,
    scope_evidence_role,
    validate_question_measure_contract,
)
from soprano_qa.score_context import (
    format_score_context,
    load_measure_lyrics,
    selected_measure_lyrics,
)
from soprano_qa.settings import load_settings
from soprano_qa.translation import (
    translate_question_to_korean,
    validate_language,
)


SETTINGS = load_settings(use_legacy_dataset_env=False)

_index_lock = threading.Lock()
_index: Optional[SearchIndex] = None
_corpus_signature: Optional[
    tuple[tuple[int, int], tuple[int, int], Optional[tuple[int, int]]]
] = None


def _file_identity(path: str) -> tuple[int, int]:
    stat = Path(path).stat()
    return stat.st_mtime_ns, stat.st_size


def _optional_file_identity(path: str) -> Optional[tuple[int, int]]:
    try:
        return _file_identity(path)
    except FileNotFoundError:
        return None


def _current_corpus_signature() -> tuple[
    tuple[int, int],
    tuple[int, int],
    Optional[tuple[int, int]],
]:
    return (
        _file_identity(SETTINGS["corpus_path"]),
        _file_identity(SETTINGS["stats_path"]),
        _optional_file_identity(SETTINGS["embedding_model_path"]),
    )


def _get_index() -> SearchIndex:
    """Validate every request and reload the index after a corpus change."""
    global _corpus_signature, _index

    validate_retrieval_requirements(SETTINGS)
    with _index_lock:
        # Never let a warm index bypass source validation. In particular,
        # ensure_corpus rebuilds from the expert-review documents and raises
        # when any knowledge unit is no longer release-ready.
        ensure_corpus(SETTINGS, rebuild=False)
        signature = _current_corpus_signature()
        if _index is None or signature != _corpus_signature:
            _index = build_retrieval_index(
                load_corpus(SETTINGS["corpus_path"]),
                SETTINGS,
            )
            _corpus_signature = signature
        return _index


def model_status() -> dict:
    model_path = Path(SETTINGS["model_path"])
    return {
        "path": str(model_path),
        "checkpoint_exists": model_path.is_file(),
        "backend": "llama-cpp-python",
    }


def corpus_stats() -> dict:
    index = _get_index()
    with open(SETTINGS["stats_path"], encoding="utf-8") as file:
        stats = json.load(file)
    return {
        **stats,
        "pipeline": "soprano_qa",
        "model": model_status(),
        "retrieval": retrieval_diagnostics(index),
    }


def _result_to_evidence(
    result: SearchResult,
    *,
    measure_ranges: list[list[int]],
) -> dict:
    record = result.record
    record_measure_ranges = record.get("measure_range") or []
    if measure_ranges:
        in_requested_scope = result.scope_match == "overlaps_query_range"
    else:
        in_requested_scope = result.scope_match in {
            "general_evidence",
            "local_example",
            "unspecified_scope",
        }

    inherited_attributions = [
        item["attribution"]
        for item in record.get("inherited_licenses", [])
        if item.get("attribution")
    ]
    evidence = {
        "id": record["id"],
        "piece_id": record.get("piece"),
        "kind": (
            "expert"
            if record.get("evidence_type") == "expert_annotation"
            else "web"
        ),
        "evidence_type": record.get("evidence_type"),
        "usage_class": record.get("usage_class"),
        "topic": record.get("topic", ""),
        "question": record.get("question", ""),
        "text": record["answer"],
        "measure_ranges": record_measure_ranges,
        "is_local": bool(record_measure_ranges),
        "measure_scope": record.get("measure_scope"),
        "measure_status": record.get("measure_status"),
        "scope_match": result.scope_match,
        "generation_role": scope_evidence_role(result.scope_match),
        "selected_range_claim_authority": (
            result.scope_match == "overlaps_query_range"
            if measure_ranges
            else None
        ),
        "in_requested_scope": in_requested_scope,
        "score": round(result.score, 6),
        "text_score": round(result.text_score, 6),
        "dense_score": round(result.dense_score, 6),
        "dense_content_score": round(result.dense_content_score, 6),
        "fusion_score": round(result.fusion_score, 6),
        "retrieval_mode": result.retrieval_mode,
        "measure_score": round(result.measure_score, 6),
        "alias_score": round(result.alias_score, 6),
        "concept_coverage": round(result.concept_coverage, 6),
        "content_concept_coverage": round(
            result.content_concept_coverage,
            6,
        ),
        "answer_relation_score": round(
            result.answer_relation_score,
            6,
        ),
        "semantic_match_type": result.semantic_match_type,
        "source_ids": record.get("source_ids", []),
        "web_source_ids": record.get("web_source_ids", []),
        "claim_ids": record.get("claim_ids", []),
        "sources": record.get("sources", []),
        "generated_text_license": record.get("generated_text_license"),
        "attributions": inherited_attributions,
    }
    return evidence


def _retrieval_unavailable_reason(
    index: SearchIndex,
    *,
    question: str,
    piece_id: str,
    measure_ranges: list[list[int]],
    results: list[SearchResult],
) -> str:
    if retrieval_abstained_for_ambiguity(index):
        return "ambiguous_dense_grounding"
    if retrieval_abstained_for_missing_causal_authority(index):
        return "missing_scoped_causal_authority"
    if (
        not results
        and is_underspecified_ranged_query(
            question,
            piece_id,
            measure_ranges,
        )
    ):
        return "underspecified_ranged_question"
    return "no_corpus_evidence"


def ask(
    *,
    piece_id: str,
    question: str,
    measure_range: Optional[tuple[int, int]],
    generate: bool,
    top_k: int = 6,
    language: str = "ko",
) -> dict:
    """Compose one grounded answer from the retrieved reviewed texts.

    Generation eagerly validates the sole local GGUF backend before corpus
    work. Every grounded case uses the same RAG+LLM composer; confidence never
    switches the renderer to concatenated KU prose. If a deterministic scope
    check questions the sole draft, the sanitized draft is still returned for
    semantic red-flag review instead of being replaced by extraction or
    internal model knowledge.

    ``language="en"`` accepts an English question and, when generating,
    answers in English. The Korean-calibrated retriever never sees the
    English text: one extra local-model call translates the question to
    Korean, and that translation (returned as ``retrieval_query``) drives
    retrieval and Korean-oriented gating. Measure mentions are still parsed
    from the original question, and the generation prompt carries the
    original English question. ``language="ko"`` (the default) makes no
    extra model call and is unchanged.
    """

    validate_language(language)
    # English questions need the local model for translation even when no
    # answer is generated.
    if generate or language != "ko":
        validate_generation_requirements(
            SETTINGS["model_path"],
            SETTINGS["llm"],
        )

    selected_measure_ranges = (
        [[measure_range[0], measure_range[1]]]
        if measure_range is not None
        else []
    )
    question_measure_ranges = validate_question_measure_contract(
        question,
        selected_measure_ranges,
    )
    # A broad selection is only the allowed envelope. An explicit locator in
    # the question is the effective retrieval and grounding scope.
    measure_ranges = question_measure_ranges or selected_measure_ranges
    measures = format_measure_range(measure_ranges) if measure_ranges else ""

    index = _get_index()
    # Korean-oriented retrieval and gating always see Korean text.
    retrieval_query = (
        translate_question_to_korean(
            question,
            model_path=SETTINGS["model_path"],
            llm_settings=SETTINGS["llm"],
            generate_fn=generate_llm,
        )
        if language == "en"
        else question
    )
    # The extractive path returns Korean evidence text in either language.
    answer_language = language if generate else "ko"
    unavailable_reason: str | None = None
    generation_validation_warning: str | None = None
    score_context = None
    if measure_ranges and (SETTINGS.get("score_context") or {}).get("lyrics", False):
        score_context = selected_measure_lyrics(
            load_measure_lyrics(SETTINGS["dataset_root"]),
            piece_id,
            measure_ranges,
        )
    score_block = format_score_context(score_context)

    if generate:
        results, messages = retrieve_generation_evidence(
            index,
            query=retrieval_query,
            piece=piece_id,
            measure_ranges=measure_ranges,
            measures=measures,
            topic=None,
            top_k=top_k,
            web_coaching_supplement=int(
                (SETTINGS.get("web_coaching") or {}).get("generation_supplement", 0)
            ),
            web_coaching_supplement_gate=str(
                (SETTINGS.get("web_coaching") or {}).get(
                    "generation_supplement_gate", "always"
                )
            ),
            web_coaching_supplement_margin=float(
                (SETTINGS.get("web_coaching") or {}).get(
                    "generation_supplement_margin", 0.0
                )
            ),
            score_context=score_block,
        )
        if language != "ko":
            # Same frozen evidence bundle; only the system prompt and the
            # displayed question change for the answer language.
            messages = build_messages(
                piece_id,
                measures,
                question,
                results,
                score_context=score_block,
                language=language,
            )
        grounded = bool(results) and not (
            measure_ranges and not has_primary_grounding(results)
        )
        ungrounded_reason = (
            None
            if grounded
            else _retrieval_unavailable_reason(
                index,
                question=retrieval_query,
                piece_id=piece_id,
                measure_ranges=measure_ranges,
                results=results,
            )
        )
        # Score context may answer only where nothing grounds the selected
        # bars; it never overrides an ambiguity, causal-authority, or
        # underspecified-question abstention.
        use_score_context = bool(score_context) and (
            ungrounded_reason == "no_corpus_evidence"
        )
        if not grounded and not use_score_context:
            answer = no_corpus_evidence_message(language)
            generation_mode = "unavailable"
            answer_basis = "no_corpus_evidence"
            unavailable_reason = ungrounded_reason
        else:
            # Without grounding evidence for the selected bars, only the
            # labelled score context may inform the answer; non-grounding
            # records stay out of the prompt.
            generation_results = results if grounded else []
            if not grounded:
                messages = build_messages(
                    piece_id,
                    measures,
                    question,
                    [],
                    score_context=score_block,
                    language=language,
                )
            raw_answer = generate_llm(
                SETTINGS["model_path"],
                messages,
                SETTINGS["llm"],
            )
            try:
                answer = finalize_grounded_generated_answer(
                    raw_answer,
                    generation_results,
                    messages,
                    measure_ranges,
                    language=language,
                )
            except GeneratedAnswerRejected as exc:
                # Preserve the sole grounded model draft for review instead
                # of substituting an extractive or internal-knowledge answer.
                # Presentation sanitization remains mandatory; semantic or
                # scope concerns are recorded for the red-flag evaluation.
                if (
                    not raw_answer.strip()
                    or is_grounded_insufficiency_answer(raw_answer)
                ):
                    if grounded:
                        raise
                    answer = no_corpus_evidence_message(language)
                else:
                    answer = finalize_user_visible_answer(
                        raw_answer,
                        generation_results,
                        language=language,
                    )
                    if answer == no_corpus_evidence_message(language) and grounded:
                        raise
                    generation_validation_warning = str(exc)
            if grounded:
                generation_mode = "llm"
                answer_basis = "retrieved_evidence"
            elif answer == no_corpus_evidence_message(language):
                generation_mode = "unavailable"
                answer_basis = "no_corpus_evidence"
                unavailable_reason = "score_context_insufficient"
            else:
                generation_mode = "llm"
                answer_basis = "score_context"
    else:
        results = index.search(
            query=retrieval_query,
            piece=piece_id,
            measure_ranges=measure_ranges,
            topic=None,
            top_k=top_k,
        )
        answer = build_extractive_answer(results, max_items=4)
        if not results and language != "ko":
            answer = no_corpus_evidence_message(language)
            answer_language = language
        generation_mode = (
            "retrieval_only" if results else "unavailable"
        )
        if results and has_primary_grounding(results):
            answer_basis = "retrieved_evidence"
        elif results:
            answer_basis = "retrieved_secondary_context"
        else:
            answer_basis = "no_corpus_evidence"
            unavailable_reason = _retrieval_unavailable_reason(
                index,
                question=retrieval_query,
                piece_id=piece_id,
                measure_ranges=measure_ranges,
                results=results,
            )

    answer = finalize_user_visible_answer(
        answer,
        results,
        language=answer_language,
    )
    if (
        answer == no_corpus_evidence_message(answer_language)
        and answer_basis not in {
            "no_corpus_evidence",
            "retrieved_secondary_context",
        }
    ):
        generation_mode = "unavailable"
        answer_basis = "no_corpus_evidence"
        unavailable_reason = "forbidden_internal_metadata"

    status = model_status()
    return {
        "piece_id": piece_id,
        "scope": "range" if measure_range else "whole_piece",
        "measure_range": list(measure_range) if measure_range else None,
        "language": language,
        "answer_language": answer_language,
        "evidence_language": "ko",
        "retrieval_query": retrieval_query,
        "answer": answer,
        "generation_mode": generation_mode,
        "answer_basis": answer_basis,
        "unavailable_reason": unavailable_reason,
        "generation_validation_warning": generation_validation_warning,
        "score_context": score_context,
        "has_primary_grounding": has_primary_grounding(results),
        "has_selected_range_grounding": (
            has_selected_range_grounding(results)
            if selected_measure_ranges
            else None
        ),
        "has_confirmed_local_examples": any(
            result.scope_match == "local_example"
            for result in results
        ),
        "pipeline": "soprano_qa",
        "model": status,
        "retrieval": retrieval_diagnostics(index),
        "evidence": [
            _result_to_evidence(
                result,
                measure_ranges=measure_ranges,
            )
            for result in results
        ],
        "evidence_notices": build_evidence_notices(results),
    }
