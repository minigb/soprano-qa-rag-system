"""Reusable service facade for the measure-aware RAG and local-LLM pipeline."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from soprano_qa.answer import (
    INTERNAL_GENERATION_UNAVAILABLE_MESSAGE,
    build_extractive_answer,
    build_internal_knowledge_messages,
    build_evidence_notices,
    ensure_corpus,
    finalize_answer_citations,
    finalize_internal_knowledge_answer,
    generate_with_context_retry,
    is_grounded_insufficiency_answer,
)
from soprano_qa.llm import generate as generate_llm
from soprano_qa.retrieval import (
    BM25Index,
    SearchResult,
    format_measure_range,
    load_corpus,
    validate_question_measure_contract,
)
from soprano_qa.settings import load_settings


SETTINGS = load_settings(use_legacy_dataset_env=False)
CORPUS_REFRESH_SECONDS = float(
    os.environ.get("SOPRANO_QA_CORPUS_REFRESH_SECONDS", "5")
)

_index_lock = threading.Lock()
_generation_lock = threading.Lock()
_index: Optional[BM25Index] = None
_corpus_signature: Optional[tuple[tuple[int, int], tuple[int, int]]] = None
_last_corpus_check = 0.0


def _file_identity(path: str) -> tuple[int, int]:
    stat = Path(path).stat()
    return stat.st_mtime_ns, stat.st_size


def _current_corpus_signature() -> tuple[tuple[int, int], tuple[int, int]]:
    return (
        _file_identity(SETTINGS["corpus_path"]),
        _file_identity(SETTINGS["stats_path"]),
    )


def _get_index() -> BM25Index:
    """Validate the corpus periodically and reload the index after a rebuild."""
    global _corpus_signature, _index, _last_corpus_check

    now = time.monotonic()
    if (
        _index is not None
        and now - _last_corpus_check < CORPUS_REFRESH_SECONDS
    ):
        return _index

    with _index_lock:
        now = time.monotonic()
        if (
            _index is not None
            and now - _last_corpus_check < CORPUS_REFRESH_SECONDS
        ):
            return _index

        ensure_corpus(SETTINGS, rebuild=False)
        signature = _current_corpus_signature()
        if _index is None or signature != _corpus_signature:
            _index = BM25Index(load_corpus(SETTINGS["corpus_path"]))
            _corpus_signature = signature
        _last_corpus_check = now
        return _index


def model_status() -> dict:
    model_path = Path(SETTINGS["model_path"])
    return {
        "path": str(model_path),
        "checkpoint_exists": model_path.is_file(),
        "llama_cpp_available": (
            importlib.util.find_spec("llama_cpp") is not None
        ),
    }


def corpus_stats() -> dict:
    _get_index()
    with open(SETTINGS["stats_path"], encoding="utf-8") as file:
        stats = json.load(file)
    return {
        **stats,
        "pipeline": "soprano_qa",
        "model": model_status(),
    }


def _result_to_evidence(
    result: SearchResult,
    *,
    has_measure_range: bool,
) -> dict:
    record = result.record
    measure_ranges = record.get("measure_range") or []
    if has_measure_range:
        in_requested_scope = result.scope_match == "overlaps_query_range"
    else:
        in_requested_scope = result.scope_match == "general_evidence"

    inherited_attributions = [
        item["attribution"]
        for item in record.get("inherited_licenses", [])
        if item.get("attribution")
    ]
    return {
        "id": record["id"],
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
        "measure_ranges": measure_ranges,
        "is_local": bool(measure_ranges),
        "measure_scope": record.get("measure_scope"),
        "scope_match": result.scope_match,
        "in_requested_scope": in_requested_scope,
        "score": round(result.score, 6),
        "text_score": round(result.text_score, 6),
        "measure_score": round(result.measure_score, 6),
        "source_ids": record.get("source_ids", []),
        "web_source_ids": record.get("web_source_ids", []),
        "claim_ids": record.get("claim_ids", []),
        "sources": record.get("sources", []),
        "generated_text_license": record.get("generated_text_license"),
        "attributions": inherited_attributions,
    }


def _generation_unavailable_answer() -> str:
    return INTERNAL_GENERATION_UNAVAILABLE_MESSAGE


def ask(
    *,
    piece_id: str,
    question: str,
    measure_range: Optional[tuple[int, int]],
    generate: bool,
    top_k: int = 6,
) -> dict:
    measure_ranges = (
        [[measure_range[0], measure_range[1]]]
        if measure_range is not None
        else []
    )
    validate_question_measure_contract(question, measure_ranges)

    index = _get_index()
    measures = format_measure_range(measure_ranges) if measure_ranges else ""
    generation_mode = "extractive"
    answer_basis = "retrieval_extractive"
    generation_fallback_reason = None
    context_limited = False
    used_internal_knowledge = False
    results: list[SearchResult]

    status = model_status()
    if generate and status["checkpoint_exists"] and status["llama_cpp_available"]:
        def run_generation(messages: list[dict[str, str]]) -> str:
            return generate_llm(
                SETTINGS["model_path"],
                messages,
                SETTINGS["llm"],
            )

        try:
            # Serialize the complete retry sequence so concurrent HTTP
            # requests cannot interleave calls into one cached llama model.
            with _generation_lock:
                raw_answer, results, _, context_limited = (
                    generate_with_context_retry(
                        index,
                        query=question,
                        piece=piece_id,
                        measure_ranges=measure_ranges,
                        measures=measures,
                        topic=None,
                        top_k=top_k,
                        generator=run_generation,
                    )
                )
                if (
                    is_grounded_insufficiency_answer(raw_answer)
                    or (
                        not context_limited
                        and (
                            not results
                            or not raw_answer.strip()
                        )
                    )
                ):
                    raw_answer = run_generation(
                        build_internal_knowledge_messages(
                            piece_id,
                            measures,
                            question,
                        )
                    )
                    results = []
                    used_internal_knowledge = True
            if used_internal_knowledge:
                answer = finalize_internal_knowledge_answer(raw_answer)
                if answer:
                    generation_mode = "llm"
                    answer_basis = "internal_knowledge"
                else:
                    answer = _generation_unavailable_answer()
                    generation_mode = "unavailable"
                    answer_basis = "generation_unavailable"
                    generation_fallback_reason = (
                        "local model returned no usable answer"
                    )
            elif context_limited and not raw_answer:
                if results:
                    answer = build_extractive_answer(results)
                    generation_fallback_reason = (
                        "model context limit exceeded"
                    )
                else:
                    answer = (
                        "검색 근거가 모델 컨텍스트 한도를 초과하여 안전하게 "
                        "답변하지 못했습니다."
                    )
                    generation_mode = "unavailable"
                    answer_basis = "generation_unavailable"
            elif not results:
                # Defensive fallback: ordinary no-hit generation is handled
                # by the internal-knowledge branch above.
                answer = _generation_unavailable_answer()
                generation_mode = "unavailable"
                answer_basis = "generation_unavailable"
            else:
                answer = finalize_answer_citations(raw_answer, results)
                generation_mode = "llm"
                answer_basis = "retrieved_evidence"
        except Exception as exc:  # keep consumers usable without CUDA
            print(
                "[qa] local generation failed; using extractive retrieval: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            generation_fallback_reason = "local generation failed"
            results = index.search(
                query=question,
                piece=piece_id,
                measure_ranges=measure_ranges,
                topic=None,
                top_k=top_k,
            )
            if results:
                answer = build_extractive_answer(results)
            else:
                answer = _generation_unavailable_answer()
                generation_mode = "unavailable"
                answer_basis = "generation_unavailable"
    else:
        if generate:
            generation_fallback_reason = (
                "model checkpoint not found"
                if not status["checkpoint_exists"]
                else "llama-cpp-python is unavailable"
            )
        results = index.search(
            query=question,
            piece=piece_id,
            measure_ranges=measure_ranges,
            topic=None,
            top_k=top_k,
        )
        if results:
            answer = build_extractive_answer(results)
        elif generate:
            answer = _generation_unavailable_answer()
            generation_mode = "unavailable"
            answer_basis = "generation_unavailable"
        else:
            answer = build_extractive_answer(results)
            answer_basis = "no_corpus_evidence"

    return {
        "piece_id": piece_id,
        "scope": "range" if measure_range else "whole_piece",
        "measure_range": list(measure_range) if measure_range else None,
        "answer": answer,
        "generation_mode": generation_mode,
        "answer_basis": answer_basis,
        "generation_fallback_reason": generation_fallback_reason,
        "context_limited": context_limited,
        "pipeline": "soprano_qa",
        "model": status,
        "evidence": [
            _result_to_evidence(
                result,
                has_measure_range=bool(measure_ranges),
            )
            for result in results
        ],
        "evidence_notices": build_evidence_notices(results),
    }
