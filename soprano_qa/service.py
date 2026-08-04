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
    build_evidence_notices,
    ensure_corpus,
    finalize_grounded_generated_answer,
    finalize_user_visible_answer,
    has_primary_grounding,
    has_selected_range_grounding,
    is_grounded_insufficiency_answer,
    is_underspecified_ranged_query,
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
from soprano_qa.settings import load_settings


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
) -> dict:
    """Compose one grounded answer from the retrieved reviewed texts.

    Generation eagerly validates the sole local GGUF backend before corpus
    work. Every grounded case uses the same RAG+LLM composer; confidence never
    switches the renderer to concatenated KU prose. If a deterministic scope
    check questions the sole draft, the sanitized draft is still returned for
    semantic red-flag review instead of being replaced by extraction or
    internal model knowledge.
    """

    if generate:
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
    unavailable_reason: str | None = None
    generation_validation_warning: str | None = None

    if generate:
        results, messages = retrieve_generation_evidence(
            index,
            query=question,
            piece=piece_id,
            measure_ranges=measure_ranges,
            measures=measures,
            topic=None,
            top_k=top_k,
        )
        if not results or (
            measure_ranges and not has_primary_grounding(results)
        ):
            answer = NO_CORPUS_EVIDENCE_MESSAGE
            generation_mode = "unavailable"
            answer_basis = "no_corpus_evidence"
            unavailable_reason = _retrieval_unavailable_reason(
                index,
                question=question,
                piece_id=piece_id,
                measure_ranges=measure_ranges,
                results=results,
            )
        else:
            raw_answer = generate_llm(
                SETTINGS["model_path"],
                messages,
                SETTINGS["llm"],
            )
            try:
                answer = finalize_grounded_generated_answer(
                    raw_answer,
                    results,
                    messages,
                    measure_ranges,
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
                    raise
                answer = finalize_user_visible_answer(raw_answer, results)
                if answer == NO_CORPUS_EVIDENCE_MESSAGE:
                    raise
                generation_validation_warning = str(exc)
            generation_mode = "llm"
            answer_basis = "retrieved_evidence"
    else:
        results = index.search(
            query=question,
            piece=piece_id,
            measure_ranges=measure_ranges,
            topic=None,
            top_k=top_k,
        )
        answer = build_extractive_answer(results, max_items=4)
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
                question=question,
                piece_id=piece_id,
                measure_ranges=measure_ranges,
                results=results,
            )

    answer = finalize_user_visible_answer(answer, results)
    if (
        answer == NO_CORPUS_EVIDENCE_MESSAGE
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
        "answer": answer,
        "generation_mode": generation_mode,
        "answer_basis": answer_basis,
        "unavailable_reason": unavailable_reason,
        "generation_validation_warning": generation_validation_warning,
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
