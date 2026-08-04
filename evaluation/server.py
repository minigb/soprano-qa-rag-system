#!/usr/bin/env python3
"""Serve the local, read-only RAG expected-vs-generated answer viewer."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Final, Mapping
from urllib.parse import unquote, urlparse, urlsplit


PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from soprano_qa.answer import (  # noqa: E402
    has_forbidden_user_visible_artifact,
    is_grounded_insufficiency_answer,
)


EVALUATION_ROOT: Final = Path(__file__).resolve().parent
DEFAULT_HOST: Final = "127.0.0.1"
DEFAULT_PORT: Final = 8766
DEFAULT_ORIGINAL_RESULTS_FILE: Final = EVALUATION_ROOT / "qualitative.json"
DEFAULT_SYNTHESIZED_RESULTS_FILE: Final = (
    EVALUATION_ROOT / "synthesized_question_results.json"
)
DEFAULT_QUALITY_REVIEW_FILE: Final = (
    EVALUATION_ROOT / "semantic_quality_review.json"
)
LOCAL_HOSTS: Final = frozenset({"127.0.0.1", "localhost", "::1"})

STATIC_ROUTES: Final = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/review-state.js": ("review-state.js", "text/javascript; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}
RED_FLAGS_ROUTES: Final = frozenset({"/red-flags", "/red-flags.html"})
RESULTS_ROUTE: Final = "/api/results"
QUALITATIVE_ARTIFACT_TYPE: Final = (
    "five_piece_qualitative_rag_llm_evaluation"
)
QUALITATIVE_SCHEMA_VERSION: Final = "1.2"
SYNTHESIZED_ARTIFACT_TYPE: Final = (
    "soprano_qa_synthesized_hybrid_rag_llm_evaluation"
)
SYNTHESIZED_SCHEMA_VERSION: Final = "1.2"
COMPARISON_ARTIFACT_TYPE: Final = (
    "soprano_qa_original_synthesized_answer_comparison"
)
COMPARISON_SCHEMA_VERSION: Final = "1.0"
QUALITY_REVIEW_ARTIFACT_TYPE: Final = (
    "soprano_qa_semantic_quality_review"
)
QUALITY_REVIEW_SCHEMA_VERSION: Final = "1.0"
QUALITY_REVIEW_METHOD: Final = "direct_codex_review"
QUALITY_REVIEW_NOT_AVAILABLE: Final = "not_available"
QUALITY_REVIEW_COMPLETE: Final = "complete"
QUALITY_REVIEW_BASE_FIELDS: Final = frozenset(
    {"case_id", "semantic_hash", "red_flag"}
)
QUALITY_REVIEW_FLAG_FIELDS: Final = frozenset(
    {"severity", "reason_codes", "rationale"}
)
QUALITY_REVIEW_SEVERITIES: Final = frozenset(
    {"critical", "high", "medium", "low"}
)
ORIGINAL_QUESTION_ORIGIN: Final = "original"
SYNTHESIZED_QUESTION_ORIGIN: Final = "synthesized"
ORIGINAL_QUESTION_KIND: Final = "human_original"
SOURCE_PARAPHRASE_KIND: Final = "source_question_paraphrase"
DERIVED_QUESTION_KIND: Final = "knowledge_unit_derived"
SOURCE_VARIANT_PROVENANCE: Final = "synthesized_variant"
DERIVED_QUESTION_PROVENANCE: Final = "knowledge_unit_derived"
REPORTED_REGRESSION_PROVENANCE: Final = "reported_regression"
HUMAN_SOURCE_SCENARIO_KIND: Final = "human_source"
DERIVED_SCENARIO_KIND: Final = "knowledge_unit_derived"
CURRENT_ARTIFACT_SCHEMAS: Final = {
    QUALITATIVE_ARTIFACT_TYPE: QUALITATIVE_SCHEMA_VERSION,
    SYNTHESIZED_ARTIFACT_TYPE: SYNTHESIZED_SCHEMA_VERSION,
}
CURRENT_ANSWER_PATH: Final = ("llm", "retrieved_evidence")
REMOVED_RUNTIME_FIELDS: Final = frozenset(
    {
        "allow_internal_knowledge",
        "context_limited",
        "fallback_reason",
        "fallback_used",
        "generation_fallback_reason",
        "lexical_fallback",
    }
)
PIPELINE_RESULT_FIELDS: Final = frozenset(
    {
        "piece_id",
        "scope",
        "measure_range",
        "answer",
        "generation_mode",
        "answer_basis",
        "unavailable_reason",
        "has_primary_grounding",
        "has_selected_range_grounding",
        "has_confirmed_local_examples",
        "pipeline",
        "model",
        "retrieval",
        "evidence",
        "evidence_notices",
        "generation_validation_warning",
    }
)
SYNTHESIZED_RESULT_FIELDS: Final = PIPELINE_RESULT_FIELDS | frozenset(
    {
        "diagnostics",
        "expert_evidence_ids",
        "linked_knowledge_unit_ids_retrieved",
        "retrieval_eligible_knowledge_unit_ids_retrieved",
        "expected_knowledge_unit_ids_retrieved",
        "expected_expert_evidence_retrieved",
        "all_linked_expert_units_retrieved",
        "all_retrieval_eligible_expert_units_retrieved",
        "retrieval_eligible_target_available",
        "all_expected_expert_units_retrieved",
        "retrieval_validation",
    }
)


class EvaluationArtifactError(ValueError):
    """Raised when the viewer is given a stale or incompatible artifact."""


class EvaluationViewerServer(ThreadingHTTPServer):
    """HTTP server carrying the evaluation result snapshot to expose."""

    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        evaluation_root: Path = EVALUATION_ROOT,
        original_results_file: Path,
        synthesized_results_file: Path,
        quality_review_file: Path,
        original_only: bool = False,
    ) -> None:
        self.evaluation_root = evaluation_root.resolve()
        self.original_results_file = original_results_file.resolve()
        self.synthesized_results_file = synthesized_results_file.resolve()
        self.quality_review_file = quality_review_file.resolve()
        self.original_only = original_only
        super().__init__(server_address, EvaluationViewerRequestHandler)


def _artifact_error(label: str, message: str) -> EvaluationArtifactError:
    return EvaluationArtifactError(f"{label}: {message}")


def _reject_removed_runtime_fields(value: Any, *, label: str) -> None:
    if isinstance(value, Mapping):
        removed = REMOVED_RUNTIME_FIELDS & set(value)
        if removed:
            raise _artifact_error(
                label,
                "contains removed runtime fields: "
                + ", ".join(sorted(removed)),
            )
        for key, item in value.items():
            _reject_removed_runtime_fields(
                item,
                label=f"{label}.{key}",
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_removed_runtime_fields(
                item,
                label=f"{label}[{index}]",
            )


def _validate_pipeline_result(
    output: Any,
    *,
    label: str,
    allowed_fields: frozenset[str],
) -> None:
    if not isinstance(output, Mapping):
        raise _artifact_error(label, "must be an object")
    unknown = set(output) - allowed_fields
    if unknown:
        raise _artifact_error(
            label,
            "contains fields outside the current response contract: "
            + ", ".join(sorted(unknown)),
        )
    _reject_removed_runtime_fields(output, label=label)
    if output.get("pipeline") != "soprano_qa":
        raise _artifact_error(label, "pipeline must equal 'soprano_qa'")

    mode = output.get("generation_mode")
    basis = output.get("answer_basis")
    if (mode, basis) != CURRENT_ANSWER_PATH:
        raise _artifact_error(
            label,
            "benchmark answers must use llm/retrieved_evidence",
        )
    answer = output.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise _artifact_error(label, "answer must be a non-empty string")
    if is_grounded_insufficiency_answer(answer):
        raise _artifact_error(
            label,
            "answer must not be a grounded-insufficiency refusal",
        )
    if has_forbidden_user_visible_artifact(answer):
        raise _artifact_error(
            label,
            "answer contains an internal user-visible artifact",
        )
    if output.get("unavailable_reason") is not None:
        raise _artifact_error(
            label,
            "generated benchmark answers cannot have unavailable_reason",
        )

    evidence = output.get("evidence")
    if not isinstance(evidence, list) or any(
        not isinstance(item, Mapping) for item in evidence
    ):
        raise _artifact_error(label, "evidence must be a list of objects")
    if not evidence:
        raise _artifact_error(
            label,
            "retrieved-evidence LLM answers require evidence",
        )

    model = output.get("model")
    if not isinstance(model, Mapping):
        raise _artifact_error(label, "model status must be an object")
    if model.get("backend") != "llama-cpp-python":
        raise _artifact_error(
            label,
            "model backend must equal 'llama-cpp-python'",
        )
    if model.get("checkpoint_exists") is not True:
        raise _artifact_error(label, "generation checkpoint is unavailable")
    if not isinstance(model.get("path"), str) or not model["path"].strip():
        raise _artifact_error(label, "generation model path is missing")

    retrieval = output.get("retrieval")
    if not isinstance(retrieval, Mapping):
        raise _artifact_error(label, "retrieval diagnostics must be an object")
    if retrieval.get("configured_mode") != "hybrid":
        raise _artifact_error(label, "configured retrieval mode is not hybrid")
    if retrieval.get("active_mode") != "hybrid":
        raise _artifact_error(label, "active retrieval mode is not hybrid")
    if retrieval.get("dense_available") is not True:
        raise _artifact_error(label, "dense retrieval model is unavailable")
    if retrieval.get("last_dense_error") not in {None, ""}:
        raise _artifact_error(label, "dense retrieval reported an error")
    if retrieval.get("query_route") not in {"hybrid", "lexical", "none"}:
        raise _artifact_error(label, "query route is invalid")
    if not isinstance(retrieval.get("dense_attempted"), bool):
        raise _artifact_error(label, "dense_attempted must be boolean")
    if not isinstance(retrieval.get("dense_contributed"), bool):
        raise _artifact_error(label, "dense_contributed must be boolean")


def _validate_case_outputs(
    cases: Any,
    *,
    result_field: str,
    allowed_fields: frozenset[str],
    label: str,
) -> None:
    if not isinstance(cases, list):
        raise _artifact_error(label, "inference_runs must be a list")
    for index, case in enumerate(cases):
        case_label = f"{label}[{index}]"
        if not isinstance(case, Mapping):
            raise _artifact_error(case_label, "case must be an object")
        status = case.get("status")
        if status not in {"pending", "running", "error", "completed"}:
            raise _artifact_error(case_label, "case status is invalid")
        result = case.get(result_field)
        if status == "completed":
            _validate_pipeline_result(
                result,
                label=f"{case_label}.{result_field}",
                allowed_fields=allowed_fields,
            )
        elif result is not None:
            raise _artifact_error(
                case_label,
                f"{status} case cannot contain {result_field}",
            )


def _validate_current_artifact(payload: Any) -> str:
    if not isinstance(payload, Mapping):
        raise EvaluationArtifactError("Evaluation artifact must be an object")
    artifact_type = payload.get("artifact_type")
    if artifact_type not in CURRENT_ARTIFACT_SCHEMAS:
        raise EvaluationArtifactError(
            "Unsupported evaluation artifact type; regenerate results with "
            "the current five-piece evaluator"
        )
    expected_schema = CURRENT_ARTIFACT_SCHEMAS[artifact_type]
    if payload.get("schema_version") != expected_schema:
        raise EvaluationArtifactError(
            f"{artifact_type} requires schema {expected_schema}; regenerate "
            "this stale result before opening it"
        )
    _reject_removed_runtime_fields(payload, label="artifact")
    run = payload.get("run")
    if not isinstance(run, Mapping) or run.get("generate") is not True:
        raise EvaluationArtifactError(
            "Current viewer artifacts require mandatory answer generation"
        )

    collection_name = (
        "questions"
        if artifact_type == QUALITATIVE_ARTIFACT_TYPE
        else "results"
    )
    collection = payload.get(collection_name)
    if not isinstance(collection, list):
        raise EvaluationArtifactError(
            f"Artifact {collection_name} must be a list"
        )
    result_field = (
        "pipeline_result"
        if artifact_type == QUALITATIVE_ARTIFACT_TYPE
        else "generated_answer"
    )
    allowed_fields = (
        PIPELINE_RESULT_FIELDS
        if artifact_type == QUALITATIVE_ARTIFACT_TYPE
        else SYNTHESIZED_RESULT_FIELDS
    )
    for index, item in enumerate(collection):
        if not isinstance(item, Mapping):
            raise _artifact_error(
                f"{collection_name}[{index}]",
                "question must be an object",
            )
        _validate_case_outputs(
            item.get("inference_runs"),
            result_field=result_field,
            allowed_fields=allowed_fields,
            label=f"{collection_name}[{index}].inference_runs",
        )
    return str(artifact_type)


def _expert_evidence_ids(output: Mapping[str, Any]) -> list[str]:
    return [
        str(item.get("id"))
        for item in output.get("evidence") or []
        if isinstance(item, Mapping)
        and (
            item.get("kind") == "expert"
            or item.get("evidence_type") == "expert_annotation"
        )
    ]


def _qualitative_generated_answer(
    case: Mapping[str, Any],
) -> dict[str, Any] | None:
    pipeline_result = case.get("pipeline_result")
    if not isinstance(pipeline_result, Mapping):
        return None
    generated = {
        field: deepcopy(pipeline_result[field])
        for field in PIPELINE_RESULT_FIELDS
        if field in pipeline_result
    }
    diagnostics = case.get("retrieval_diagnostics")
    generated["diagnostics"] = (
        deepcopy(dict(diagnostics))
        if isinstance(diagnostics, Mapping)
        else {}
    )
    expert_ids = _expert_evidence_ids(generated)
    generated["expert_evidence_ids"] = expert_ids
    expected = case.get("expected_retrieval")
    expected_ids = (
        list(expected.get("retrieval_eligible_knowledge_unit_ids") or [])
        if isinstance(expected, Mapping)
        else []
    )
    generated["expected_expert_evidence_retrieved"] = bool(
        set(expected_ids) & set(expert_ids)
    )
    return generated


def _qualitative_retrieval_probe(
    generated: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if generated is None:
        return None
    return {
        "piece_id": generated.get("piece_id"),
        "scope": generated.get("scope"),
        "measure_range": deepcopy(generated.get("measure_range")),
        "pipeline": generated.get("pipeline"),
        "answer_basis": "same_call_retrieval",
        "retrieval": deepcopy(generated.get("retrieval")),
        "evidence": deepcopy(generated.get("evidence") or []),
        "evidence_notices": deepcopy(
            generated.get("evidence_notices") or []
        ),
        "diagnostics": deepcopy(generated.get("diagnostics") or {}),
    }


def _qualitative_inference_run(
    question: Mapping[str, Any],
    case: Mapping[str, Any],
) -> dict[str, Any]:
    run = deepcopy(dict(case))
    expected = case.get("expected_retrieval")
    expected = dict(expected) if isinstance(expected, Mapping) else {}
    raw_input = case.get("inference_input")
    inference_input = (
        deepcopy(dict(raw_input))
        if isinstance(raw_input, Mapping)
        else {}
    )
    inference_input.setdefault("source_id", question.get("source_id"))
    inference_input.setdefault(
        "expected_knowledge_unit_ids",
        deepcopy(expected.get("linked_knowledge_unit_ids") or []),
    )
    inference_input.setdefault(
        "expected_retrieval_eligible_knowledge_unit_ids",
        deepcopy(
            expected.get("retrieval_eligible_knowledge_unit_ids") or []
        ),
    )
    inference_input.setdefault(
        "range_applicable_knowledge_unit_ids",
        deepcopy(expected.get("range_applicable_knowledge_unit_ids") or []),
    )
    generated = _qualitative_generated_answer(case)
    run["inference_input"] = inference_input
    run["generated_answer"] = generated
    run["retrieval_probe"] = _qualitative_retrieval_probe(generated)
    return run


def normalize_results_payload(payload: Any) -> Any:
    """Validate a current artifact and project canonical results for viewing.

    The source file remains unchanged. Incompatible or stale artifacts are
    rejected so removed runtime fields can never be exposed by this server.
    """

    artifact_type = _validate_current_artifact(payload)
    if artifact_type == SYNTHESIZED_ARTIFACT_TYPE:
        return payload

    normalized = deepcopy(dict(payload))
    results = []
    for raw_question in payload["questions"]:
        if not isinstance(raw_question, Mapping):
            continue
        question = deepcopy(dict(raw_question))
        reference_material = question.get("reference_material")
        question["authoritative_reference"] = (
            deepcopy(dict(reference_material))
            if isinstance(reference_material, Mapping)
            else {}
        )
        question["expected_retrieval_eligible_knowledge_unit_ids"] = (
            deepcopy(
                question.get("retrieval_eligible_knowledge_unit_ids") or []
            )
        )
        question["synthesized_question_variants"] = []
        question["inference_runs"] = [
            _qualitative_inference_run(question, case)
            for case in question.get("inference_runs") or []
            if isinstance(case, Mapping)
        ]
        results.append(question)
    normalized["results"] = results
    normalized["viewer_projection"] = {
        "source_artifact_type": QUALITATIVE_ARTIFACT_TYPE,
        "source_collection": "questions",
        "projection": "canonical_qualitative_to_results",
    }
    return normalized


def _require_comparison_source(
    payload: Any,
    *,
    expected_artifact_type: str,
    label: str,
) -> dict[str, Any]:
    artifact_type = _validate_current_artifact(payload)
    if artifact_type != expected_artifact_type:
        raise _artifact_error(
            label,
            f"must be {expected_artifact_type}, not {artifact_type}",
        )
    if not isinstance(payload, Mapping):
        raise _artifact_error(label, "must be an object")
    run = payload.get("run")
    expected_status = (
        "completed"
        if expected_artifact_type == QUALITATIVE_ARTIFACT_TYPE
        else "complete"
    )
    if not isinstance(run, Mapping) or run.get("status") != expected_status:
        raise _artifact_error(
            label,
            f"run.status must be {expected_status!r} for a complete comparison",
        )
    return deepcopy(dict(payload))


def _question_map(
    results: Any,
    *,
    label: str,
) -> dict[str, Mapping[str, Any]]:
    if not isinstance(results, list):
        raise _artifact_error(label, "must be a list")
    mapped: dict[str, Mapping[str, Any]] = {}
    for index, result in enumerate(results):
        item_label = f"{label}[{index}]"
        if not isinstance(result, Mapping):
            raise _artifact_error(item_label, "must be an object")
        source_id = result.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise _artifact_error(item_label, "source_id must be non-empty")
        if source_id in mapped:
            raise _artifact_error(label, f"repeats source_id {source_id!r}")
        mapped[source_id] = result
    return mapped


def _scenario_signature(
    result: Mapping[str, Any],
    run: Mapping[str, Any],
    *,
    label: str,
) -> tuple[str, str, str, tuple[int, int] | None]:
    source_id = result.get("source_id")
    piece_id = result.get("piece_id")
    inference_input = run.get("inference_input")
    if not isinstance(source_id, str) or not source_id:
        raise _artifact_error(label, "source_id must be non-empty")
    if not isinstance(piece_id, str) or not piece_id:
        raise _artifact_error(label, "piece_id must be non-empty")
    if not isinstance(inference_input, Mapping):
        raise _artifact_error(label, "inference_input must be an object")
    case_kind = inference_input.get("case_kind")
    if not isinstance(case_kind, str) or not case_kind:
        raise _artifact_error(label, "case_kind must be non-empty")
    measure_range = inference_input.get("measure_range")
    if measure_range is None:
        normalized_range = None
    elif (
        isinstance(measure_range, list)
        and len(measure_range) == 2
        and all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in measure_range
        )
        and 1 <= measure_range[0] <= measure_range[1]
    ):
        normalized_range = (measure_range[0], measure_range[1])
    else:
        raise _artifact_error(
            label,
            "measure_range must be null or a positive inclusive pair",
        )
    return piece_id, source_id, case_kind, normalized_range


def _tag_comparison_run(
    run: Mapping[str, Any],
    *,
    question_origin: str,
    question_kind: str,
) -> dict[str, Any]:
    tagged = deepcopy(dict(run))
    tagged["viewer_question_origin"] = question_origin
    tagged["viewer_question_kind"] = question_kind
    return tagged


def _source_case_count(payload: Mapping[str, Any]) -> int:
    collection = (
        payload.get("questions")
        if payload.get("artifact_type") == QUALITATIVE_ARTIFACT_TYPE
        else payload.get("results")
    )
    if not isinstance(collection, list):
        return 0
    return sum(
        len(item.get("inference_runs") or [])
        for item in collection
        if isinstance(item, Mapping)
    )


def semantic_review_case_hash(
    result: Mapping[str, Any],
    run: Mapping[str, Any],
) -> str:
    """Fingerprint every field that can change a semantic assessment.

    The review artifact deliberately hashes the exact, unsanitized question,
    authority, and generated answer. A review therefore cannot silently carry
    over after inference output or judgment context changes.
    """

    inference_input = run.get("inference_input")
    generated_answer = run.get("generated_answer")
    inference_input = (
        inference_input if isinstance(inference_input, Mapping) else {}
    )
    generated_answer = (
        generated_answer if isinstance(generated_answer, Mapping) else {}
    )
    judgment_input = {
        "piece_id": result.get("piece_id"),
        "source_evaluation_id": result.get("source_evaluation_id"),
        "question_origin": run.get("viewer_question_origin"),
        "question_kind": run.get("viewer_question_kind"),
        "question_provenance": inference_input.get("question_provenance"),
        "case_id": run.get("case_id"),
        "question": inference_input.get("question"),
        "case_kind": inference_input.get("case_kind"),
        "measure_range": inference_input.get("measure_range"),
        "case_reference_authority": inference_input.get(
            "case_reference_authority"
        ),
        "generated_answer": generated_answer.get("answer"),
        "answer_evidence": generated_answer.get("evidence"),
        "generation_validation_warning": generated_answer.get(
            "generation_validation_warning"
        ),
        "generation_mode": generated_answer.get("generation_mode"),
        "answer_basis": generated_answer.get("answer_basis"),
        "scope": generated_answer.get("scope"),
    }
    canonical = json.dumps(
        judgment_input,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _comparison_runs(
    comparison: Mapping[str, Any],
) -> list[tuple[Mapping[str, Any], dict[str, Any]]]:
    indexed: list[tuple[Mapping[str, Any], dict[str, Any]]] = []
    for result in comparison.get("results") or []:
        if not isinstance(result, Mapping):
            continue
        for run in result.get("inference_runs") or []:
            if isinstance(run, dict):
                indexed.append((result, run))
    return indexed


def _validate_quality_assessment(
    assessment: Any,
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(assessment, Mapping):
        raise _artifact_error(label, "must be an object")
    fields = set(assessment)
    allowed = QUALITY_REVIEW_BASE_FIELDS | QUALITY_REVIEW_FLAG_FIELDS
    unknown = fields - allowed
    if unknown:
        raise _artifact_error(
            label,
            "contains unsupported fields: " + ", ".join(sorted(unknown)),
        )
    missing = QUALITY_REVIEW_BASE_FIELDS - fields
    if missing:
        raise _artifact_error(
            label,
            "is missing required fields: " + ", ".join(sorted(missing)),
        )
    case_id = assessment.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        raise _artifact_error(label, "case_id must be a non-empty string")
    semantic_hash = assessment.get("semantic_hash")
    if (
        not isinstance(semantic_hash, str)
        or len(semantic_hash) != 64
        or any(character not in "0123456789abcdef" for character in semantic_hash)
    ):
        raise _artifact_error(label, "semantic_hash must be lowercase SHA-256")
    red_flag = assessment.get("red_flag")
    if not isinstance(red_flag, bool):
        raise _artifact_error(label, "red_flag must be boolean")
    present_flag_fields = fields & QUALITY_REVIEW_FLAG_FIELDS
    if not red_flag:
        if present_flag_fields:
            raise _artifact_error(
                label,
                "a passing assessment cannot contain red-flag metadata",
            )
        return deepcopy(dict(assessment))

    missing_flag_fields = QUALITY_REVIEW_FLAG_FIELDS - present_flag_fields
    if missing_flag_fields:
        raise _artifact_error(
            label,
            "a red flag is missing required metadata: "
            + ", ".join(sorted(missing_flag_fields)),
        )
    severity = assessment.get("severity")
    if severity not in QUALITY_REVIEW_SEVERITIES:
        raise _artifact_error(
            label,
            "severity must be critical, high, medium, or low",
        )
    reason_codes = assessment.get("reason_codes")
    if (
        not isinstance(reason_codes, list)
        or not reason_codes
        or any(not isinstance(code, str) or not code for code in reason_codes)
        or len(set(reason_codes)) != len(reason_codes)
    ):
        raise _artifact_error(
            label,
            "reason_codes must be a non-empty list of unique strings",
        )
    rationale = assessment.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise _artifact_error(label, "rationale must be a non-empty string")
    return deepcopy(dict(assessment))


def attach_quality_review(
    comparison: dict[str, Any],
    review_payload: Any | None,
) -> dict[str, Any]:
    """Attach a complete, current review or an explicit unavailable state."""

    indexed_runs = _comparison_runs(comparison)
    if review_payload is None:
        comparison["quality_review"] = {
            "review_status": QUALITY_REVIEW_NOT_AVAILABLE,
        }
        for _, run in indexed_runs:
            run["quality_review"] = {
                "review_status": QUALITY_REVIEW_NOT_AVAILABLE,
            }
        return comparison

    if not isinstance(review_payload, Mapping):
        raise _artifact_error("quality review", "must be an object")
    if review_payload.get("artifact_type") != QUALITY_REVIEW_ARTIFACT_TYPE:
        raise _artifact_error(
            "quality review",
            f"artifact_type must be {QUALITY_REVIEW_ARTIFACT_TYPE!r}",
        )
    if review_payload.get("schema_version") != QUALITY_REVIEW_SCHEMA_VERSION:
        raise _artifact_error(
            "quality review",
            f"schema_version must be {QUALITY_REVIEW_SCHEMA_VERSION!r}",
        )
    if review_payload.get("review_status") != QUALITY_REVIEW_COMPLETE:
        raise _artifact_error(
            "quality review",
            "review_status must be 'complete' when the artifact exists",
        )
    if review_payload.get("review_method") != QUALITY_REVIEW_METHOD:
        raise _artifact_error(
            "quality review",
            f"review_method must be {QUALITY_REVIEW_METHOD!r}",
        )
    assessments = review_payload.get("assessments")
    if not isinstance(assessments, list):
        raise _artifact_error("quality review.assessments", "must be a list")

    by_case_id: dict[str, dict[str, Any]] = {}
    for index, raw_assessment in enumerate(assessments):
        assessment = _validate_quality_assessment(
            raw_assessment,
            label=f"quality review.assessments[{index}]",
        )
        case_id = assessment["case_id"]
        if case_id in by_case_id:
            raise _artifact_error(
                "quality review.assessments",
                f"repeats case_id {case_id!r}",
            )
        by_case_id[case_id] = assessment

    run_case_ids = {str(run.get("case_id")) for _, run in indexed_runs}
    review_case_ids = set(by_case_id)
    if review_case_ids != run_case_ids:
        missing = sorted(run_case_ids - review_case_ids)
        unexpected = sorted(review_case_ids - run_case_ids)
        raise _artifact_error(
            "quality review.assessments",
            "must cover every current inference run exactly; "
            f"missing={missing}, unexpected={unexpected}",
        )

    red_flag_count = 0
    for result, run in indexed_runs:
        case_id = str(run.get("case_id"))
        assessment = by_case_id[case_id]
        current_hash = semantic_review_case_hash(result, run)
        if assessment["semantic_hash"] != current_hash:
            raise _artifact_error(
                f"quality review assessment {case_id!r}",
                "semantic_hash does not match current judgment input",
            )
        red_flag_count += int(assessment["red_flag"])
        run["quality_review"] = {
            "review_status": QUALITY_REVIEW_COMPLETE,
            "review_method": QUALITY_REVIEW_METHOD,
            **deepcopy(assessment),
        }

    comparison["quality_review"] = {
        "artifact_type": QUALITY_REVIEW_ARTIFACT_TYPE,
        "schema_version": QUALITY_REVIEW_SCHEMA_VERSION,
        "review_status": QUALITY_REVIEW_COMPLETE,
        "review_method": QUALITY_REVIEW_METHOD,
        "assessment_count": len(assessments),
        "red_flag_count": red_flag_count,
    }
    return comparison


def build_original_only_payload(original_payload: Any) -> dict[str, Any]:
    """Build a comparison-compatible projection of every original run."""

    original_source = _require_comparison_source(
        original_payload,
        expected_artifact_type=QUALITATIVE_ARTIFACT_TYPE,
        label="original artifact",
    )
    original = normalize_results_payload(original_source)
    original_results = original.get("results")
    if not isinstance(original_results, list) or not original_results:
        raise _artifact_error(
            "original results",
            "must contain the complete evaluation question set",
        )
    original_by_source = _question_map(
        original_results,
        label="original results",
    )
    original_targets = original_source.get("run", {}).get("target_pieces")
    if (
        not isinstance(original_targets, list)
        or not original_targets
        or any(
            not isinstance(piece_id, str) or not piece_id
            for piece_id in original_targets
        )
        or len(set(original_targets)) != len(original_targets)
    ):
        raise EvaluationArtifactError(
            "Original artifact must declare a non-empty target_pieces list"
        )
    result_pieces = {
        result.get("piece_id") for result in original_by_source.values()
    }
    if result_pieces != set(original_targets):
        raise EvaluationArtifactError(
            "Every declared target piece must have original results"
        )

    scenarios: dict[
        tuple[str, str, str, tuple[int, int] | None],
        dict[str, Any],
    ] = {}
    comparison_results: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    per_piece: dict[str, dict[str, int]] = {}
    for result_index, original_result in enumerate(original_results):
        source_id = original_result.get("source_id")
        runs = original_result.get("inference_runs")
        if not isinstance(runs, list) or not runs:
            raise _artifact_error(
                f"original results[{result_index}]",
                "must contain at least one inference run",
            )
        for run_index, run in enumerate(runs):
            run_label = (
                f"original results[{result_index}].inference_runs[{run_index}]"
            )
            if not isinstance(run, Mapping):
                raise _artifact_error(run_label, "must be an object")
            if run.get("status") != "completed":
                raise _artifact_error(
                    run_label,
                    "must be completed for the all-results preview",
                )
            case_id = run.get("case_id")
            if (
                not isinstance(case_id, str)
                or not case_id
                or case_id in seen_case_ids
            ):
                raise _artifact_error(
                    run_label,
                    "case_id must be non-empty and unique",
                )
            seen_case_ids.add(case_id)
            signature = _scenario_signature(
                original_result,
                run,
                label=run_label,
            )
            if signature in scenarios:
                raise _artifact_error(
                    run_label,
                    "duplicates an original scenario signature",
                )

            tagged_run = _tag_comparison_run(
                run,
                question_origin=ORIGINAL_QUESTION_ORIGIN,
                question_kind=ORIGINAL_QUESTION_KIND,
            )
            scenario = deepcopy(dict(original_result))
            scenario["source_evaluation_id"] = source_id
            scenario["source_id"] = case_id
            scenario["viewer_scenario_id"] = case_id
            scenario["synthesized_question_variants"] = []
            scenario["viewer_scenario_kind"] = HUMAN_SOURCE_SCENARIO_KIND
            scenario["viewer_synthesized_only"] = False
            scenario["inference_runs"] = [tagged_run]
            scenario["viewer_comparison"] = {
                "original_case_count": 1,
                "synthesized_case_count": 0,
            }
            scenarios[signature] = scenario
            comparison_results.append(scenario)

            piece_id = str(scenario.get("piece_id"))
            counts = per_piece.setdefault(
                piece_id,
                {
                    "scenarios": 0,
                    "original": 0,
                    "synthesized": 0,
                    "synthesized_paraphrase": 0,
                    "knowledge_unit_derived": 0,
                    "total": 0,
                },
            )
            counts["scenarios"] += 1
            counts["original"] += 1
            counts["total"] += 1

    original_case_count = len(seen_case_ids)
    comparison = {
        "artifact_type": COMPARISON_ARTIFACT_TYPE,
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "run": {
            "status": "complete",
            "generate": True,
            "target_pieces": deepcopy(original_targets),
        },
        "source_artifacts": {
            ORIGINAL_QUESTION_ORIGIN: {
                "artifact_type": original_source.get("artifact_type"),
                "schema_version": original_source.get("schema_version"),
                "status": original_source.get("run", {}).get("status"),
                "case_count": _source_case_count(original_source),
            },
        },
        "results": comparison_results,
        "summary": {
            "question_group_count": len(original_by_source),
            "human_source_question_group_count": len(original_by_source),
            "paired_human_question_group_count": 0,
            "original_only_question_group_count": len(original_by_source),
            "knowledge_unit_derived_question_group_count": 0,
            "scenario_count": len(comparison_results),
            "paired_scenario_count": 0,
            "original_only_scenario_count": len(comparison_results),
            "synthesized_only_scenario_count": 0,
            "original_case_count": original_case_count,
            "synthesized_paraphrase_case_count": 0,
            "knowledge_unit_derived_case_count": 0,
            "synthesized_case_count": 0,
            "total_case_count": original_case_count,
            "per_piece": per_piece,
        },
        "viewer_projection": {
            "projection": "all_original_cases_by_scenario",
            "original_only": True,
            "scenario_join": [
                "piece_id",
                "source_evaluation_id",
                "case_kind",
                "measure_range",
            ],
        },
    }
    return attach_quality_review(comparison, None)


def build_comparison_payload(
    original_payload: Any,
    synthesized_payload: Any,
    quality_review_payload: Any | None = None,
) -> dict[str, Any]:
    """Build the all-case original/synthesized viewer projection."""

    original_source = _require_comparison_source(
        original_payload,
        expected_artifact_type=QUALITATIVE_ARTIFACT_TYPE,
        label="original artifact",
    )
    synthesized_source = _require_comparison_source(
        synthesized_payload,
        expected_artifact_type=SYNTHESIZED_ARTIFACT_TYPE,
        label="synthesized artifact",
    )
    original = normalize_results_payload(original_source)
    synthesized = normalize_results_payload(synthesized_source)
    original_results = original.get("results")
    synthesized_results = synthesized.get("results")
    if not isinstance(original_results, list) or not original_results:
        raise _artifact_error(
            "original results",
            "must contain the complete evaluation question set",
        )
    if not isinstance(synthesized_results, list) or not synthesized_results:
        raise _artifact_error(
            "synthesized results",
            "must contain the complete evaluation question set",
        )
    original_by_source = _question_map(
        original_results,
        label="original results",
    )
    synthesized_source_results: list[Mapping[str, Any]] = []
    derived_results: list[Mapping[str, Any]] = []
    for index, result in enumerate(synthesized_results):
        label = f"synthesized results[{index}]"
        if not isinstance(result, Mapping):
            raise _artifact_error(label, "must be an object")
        provenance = result.get("question_provenance")
        if provenance == SOURCE_VARIANT_PROVENANCE:
            if result.get("synthesized_question_kind") != SOURCE_PARAPHRASE_KIND:
                raise _artifact_error(
                    label,
                    "source variants must declare source_question_paraphrase",
                )
            synthesized_source_results.append(result)
        elif provenance == DERIVED_QUESTION_PROVENANCE:
            if result.get("synthesized_question_kind") != DERIVED_QUESTION_KIND:
                raise _artifact_error(
                    label,
                    "derived questions must declare knowledge_unit_derived",
                )
            derived_results.append(result)
        else:
            raise _artifact_error(
                label,
                "question_provenance must be synthesized_variant or "
                "knowledge_unit_derived",
            )
    synthesized_by_source = _question_map(
        synthesized_source_results,
        label="synthesized source-question results",
    )
    derived_by_source = _question_map(
        derived_results,
        label="synthesized derived-question results",
    )
    original_targets = original_source.get("run", {}).get("target_pieces")
    synthesized_targets = synthesized_source.get("run", {}).get(
        "target_pieces"
    )
    if (
        not isinstance(original_targets, list)
        or not original_targets
        or any(
            not isinstance(piece_id, str) or not piece_id
            for piece_id in original_targets
        )
        or len(set(original_targets)) != len(original_targets)
        or original_targets != synthesized_targets
    ):
        raise EvaluationArtifactError(
            "Original and synthesized artifacts must declare the same "
            "non-empty target_pieces list"
        )
    result_pieces = {
        result.get("piece_id")
        for result in original_by_source.values()
    }
    if result_pieces != set(original_targets):
        raise EvaluationArtifactError(
            "Every declared target piece must have comparison results"
        )
    unexpected_synthesized_sources = sorted(
        set(synthesized_by_source) - set(original_by_source)
    )
    if unexpected_synthesized_sources:
        raise EvaluationArtifactError(
            "Synthesized source-question results have no matching original: "
            f"{unexpected_synthesized_sources}"
        )
    missing_synthesized_sources = sorted(
        set(original_by_source) - set(synthesized_by_source)
    )
    invalid_original_only_sources = [
        source_id
        for source_id in missing_synthesized_sources
        if original_by_source[source_id].get("review_status")
        != "excluded_unanswerable"
    ]
    if invalid_original_only_sources:
        raise EvaluationArtifactError(
            "Original question groups without synthesized paraphrases must be "
            "excluded_unanswerable: "
            f"{invalid_original_only_sources}"
        )
    if set(derived_by_source) & set(original_by_source):
        raise EvaluationArtifactError(
            "Knowledge-unit-derived question IDs must not reuse human source IDs"
        )
    synthesized_pieces = {
        result.get("piece_id") for result in synthesized_results
    }
    if not synthesized_pieces <= set(original_targets):
        raise EvaluationArtifactError(
            "Synthesized results contain a piece outside target_pieces"
        )

    comparison_results: list[dict[str, Any]] = []
    scenarios: dict[
        tuple[str, str, str, tuple[int, int] | None],
        dict[str, Any],
    ] = {}
    seen_original_case_ids: set[str] = set()
    for result_index, original_result in enumerate(original_results or []):
        source_id = original_result.get("source_id")
        synthesized_result = synthesized_by_source.get(str(source_id))
        if synthesized_result is not None:
            for field in (
                "piece_id",
                "annotator",
                "inventory_schema_version",
                "original_question",
                "paraphrased_question",
                "review_status",
            ):
                if original_result.get(field) != synthesized_result.get(field):
                    raise _artifact_error(
                        f"source {source_id}",
                        f"{field} differs between original and synthesized artifacts",
                    )
        runs = original_result.get("inference_runs")
        if not isinstance(runs, list) or not runs:
            raise _artifact_error(
                f"original results[{result_index}]",
                "must contain at least one inference run",
            )
        for run_index, run in enumerate(runs):
            run_label = (
                f"original results[{result_index}].inference_runs[{run_index}]"
            )
            if not isinstance(run, Mapping):
                raise _artifact_error(run_label, "must be an object")
            if run.get("status") != "completed":
                raise _artifact_error(
                    run_label,
                    "must be completed for the all-results comparison",
                )
            case_id = run.get("case_id")
            if (
                not isinstance(case_id, str)
                or not case_id
                or case_id in seen_original_case_ids
            ):
                raise _artifact_error(
                    run_label,
                    "case_id must be non-empty and unique",
                )
            seen_original_case_ids.add(case_id)
            signature = _scenario_signature(
                original_result,
                run,
                label=run_label,
            )
            if signature in scenarios:
                raise _artifact_error(
                    run_label,
                    "duplicates an original scenario signature",
                )
            scenario = deepcopy(dict(original_result))
            # Browser selection and review state are keyed by result.source_id.
            # Use the unique canonical case ID there so source IDs shared by
            # both artifacts cannot collapse distinct scenarios.
            scenario["source_evaluation_id"] = source_id
            scenario["source_id"] = case_id
            scenario["viewer_scenario_id"] = case_id
            scenario["synthesized_question_variants"] = deepcopy(
                synthesized_result.get("synthesized_question_variants") or []
                if synthesized_result is not None
                else []
            )
            scenario["viewer_scenario_kind"] = HUMAN_SOURCE_SCENARIO_KIND
            scenario["viewer_synthesized_only"] = False
            scenario["inference_runs"] = [
                _tag_comparison_run(
                    run,
                    question_origin=ORIGINAL_QUESTION_ORIGIN,
                    question_kind=ORIGINAL_QUESTION_KIND,
                )
            ]
            scenario["viewer_comparison"] = {
                "original_case_count": 1,
                "synthesized_case_count": 0,
            }
            scenarios[signature] = scenario
            comparison_results.append(scenario)

    seen_synthesized_case_ids: set[str] = set()
    for result_index, synthesized_result in enumerate(
        synthesized_source_results
    ):
        runs = synthesized_result.get("inference_runs")
        if not isinstance(runs, list) or not runs:
            raise _artifact_error(
                f"synthesized results[{result_index}]",
                "must contain at least one inference run",
            )
        for run_index, run in enumerate(runs):
            run_label = (
                "synthesized results"
                f"[{result_index}].inference_runs[{run_index}]"
            )
            if not isinstance(run, Mapping):
                raise _artifact_error(run_label, "must be an object")
            if run.get("status") != "completed":
                raise _artifact_error(
                    run_label,
                    "must be completed for the all-results comparison",
                )
            case_id = run.get("case_id")
            if (
                not isinstance(case_id, str)
                or not case_id
                or case_id in seen_original_case_ids
                or case_id in seen_synthesized_case_ids
            ):
                raise _artifact_error(
                    run_label,
                    "case_id must be non-empty and globally unique",
                )
            seen_synthesized_case_ids.add(case_id)
            signature = _scenario_signature(
                synthesized_result,
                run,
                label=run_label,
            )
            scenario = scenarios.get(signature)
            if scenario is None:
                raise _artifact_error(
                    run_label,
                    "has no matching original inference scenario",
                )
            original_authority = (
                scenario["inference_runs"][0]
                .get("inference_input", {})
                .get("case_reference_authority")
            )
            synthesized_authority = (
                run.get("inference_input", {})
                .get("case_reference_authority")
            )
            if original_authority != synthesized_authority:
                raise _artifact_error(
                    run_label,
                    "case reference authority differs from the original scenario",
                )
            scenario["inference_runs"].append(
                _tag_comparison_run(
                    run,
                    question_origin=SYNTHESIZED_QUESTION_ORIGIN,
                    question_kind=SOURCE_PARAPHRASE_KIND,
                )
            )

    derived_case_count = 0
    for result_index, derived_result in enumerate(derived_results):
        result_label = f"derived results[{result_index}]"
        if derived_result.get("original_question") is not None:
            raise _artifact_error(
                result_label,
                "knowledge-unit-derived questions cannot claim a human original",
            )
        if derived_result.get("annotator") is not None:
            raise _artifact_error(
                result_label,
                "knowledge-unit-derived questions cannot claim a human annotator",
            )
        if derived_result.get("synthesized_question_variants") not in ([], None):
            raise _artifact_error(
                result_label,
                "knowledge-unit-derived questions cannot contain paraphrase variants",
            )
        derived_source_id = derived_result.get("source_id")
        runs = derived_result.get("inference_runs")
        if not isinstance(runs, list) or not runs:
            raise _artifact_error(
                result_label,
                "must contain at least one inference run",
            )
        seen_derived_ranges: set[tuple[int, int] | None] = set()
        for run_index, run in enumerate(runs):
            run_label = f"{result_label}.inference_runs[{run_index}]"
            if not isinstance(run, Mapping):
                raise _artifact_error(run_label, "must be an object")
            if run.get("status") != "completed":
                raise _artifact_error(
                    run_label,
                    "must be completed for the all-results comparison",
                )
            case_id = run.get("case_id")
            if (
                not isinstance(case_id, str)
                or not case_id
                or case_id in seen_original_case_ids
                or case_id in seen_synthesized_case_ids
            ):
                raise _artifact_error(
                    run_label,
                    "case_id must be non-empty and globally unique",
                )
            inference_input = run.get("inference_input")
            if not isinstance(inference_input, Mapping):
                raise _artifact_error(run_label, "inference_input must be an object")
            if (
                inference_input.get("question_provenance")
                != DERIVED_QUESTION_PROVENANCE
                or inference_input.get("synthesized_question_kind")
                != DERIVED_QUESTION_KIND
                or inference_input.get("synthesized_variant_id") is not None
                or inference_input.get("variant_index") is not None
            ):
                raise _artifact_error(
                    run_label,
                    "must be a non-variant knowledge-unit-derived question",
                )
            signature = _scenario_signature(
                derived_result,
                run,
                label=run_label,
            )
            normalized_range = signature[-1]
            if normalized_range in seen_derived_ranges:
                raise _artifact_error(
                    run_label,
                    "duplicates a derived question range context",
                )
            seen_derived_ranges.add(normalized_range)
            seen_synthesized_case_ids.add(case_id)
            derived_case_count += 1

            scenario = deepcopy(dict(derived_result))
            scenario["source_evaluation_id"] = derived_source_id
            scenario["source_id"] = case_id
            scenario["viewer_scenario_id"] = case_id
            scenario["viewer_scenario_kind"] = DERIVED_SCENARIO_KIND
            scenario["viewer_synthesized_only"] = True
            scenario["synthesized_question_variants"] = []
            scenario["inference_runs"] = [
                _tag_comparison_run(
                    run,
                    question_origin=SYNTHESIZED_QUESTION_ORIGIN,
                    question_kind=DERIVED_QUESTION_KIND,
                )
            ]
            scenario["viewer_comparison"] = {
                "original_case_count": 0,
                "synthesized_case_count": 1,
            }
            comparison_results.append(scenario)

    paired_scenarios = 0
    original_only_scenarios = 0
    synthesized_only_scenarios = 0
    per_piece: dict[str, dict[str, int]] = {}
    for scenario in comparison_results:
        original_runs = [
            run
            for run in scenario["inference_runs"]
            if run.get("viewer_question_origin") == ORIGINAL_QUESTION_ORIGIN
        ]
        synthesized_runs = sorted(
            (
                run
                for run in scenario["inference_runs"]
                if run.get("viewer_question_origin")
                == SYNTHESIZED_QUESTION_ORIGIN
            ),
            key=lambda run: (
                run.get("inference_input", {}).get("variant_index", 0),
                str(run.get("case_id") or ""),
            ),
        )
        scenario_kind = scenario.get("viewer_scenario_kind")
        if scenario_kind == DERIVED_SCENARIO_KIND:
            if len(original_runs) != 0 or len(synthesized_runs) != 1:
                raise EvaluationArtifactError(
                    f"Derived scenario {scenario['viewer_scenario_id']} must "
                    "contain exactly one synthesized-only run"
                )
            if (
                synthesized_runs[0].get("viewer_question_kind")
                != DERIVED_QUESTION_KIND
            ):
                raise EvaluationArtifactError(
                    f"Derived scenario {scenario['viewer_scenario_id']} is "
                    "missing its KU-derived marker"
                )
            synthesized_only_scenarios += 1
        elif len(original_runs) != 1:
            raise EvaluationArtifactError(
                f"Scenario {scenario['viewer_scenario_id']} does not contain "
                "exactly one original run"
            )
        elif synthesized_runs:
            variant_indexes = [
                run.get("inference_input", {}).get("variant_index")
                for run in synthesized_runs
            ]
            if variant_indexes != [1, 2, 3]:
                raise EvaluationArtifactError(
                    f"Scenario {scenario['viewer_scenario_id']} must contain "
                    "synthesized variants 1, 2, and 3 exactly once"
                )
            paired_scenarios += 1
        else:
            original_provenance = (
                original_runs[0]
                .get("inference_input", {})
                .get("question_provenance")
            )
            if (
                scenario.get("source_evaluation_id") in synthesized_by_source
                and original_provenance != REPORTED_REGRESSION_PROVENANCE
            ):
                raise EvaluationArtifactError(
                    f"Scenario {scenario['viewer_scenario_id']} is missing its "
                    "three synthesized paraphrases"
                )
            original_only_scenarios += 1
        scenario["inference_runs"] = [*original_runs, *synthesized_runs]
        scenario["viewer_comparison"] = {
            "original_case_count": len(original_runs),
            "synthesized_case_count": len(synthesized_runs),
        }
        piece_id = str(scenario.get("piece_id"))
        counts = per_piece.setdefault(
            piece_id,
            {
                "scenarios": 0,
                "original": 0,
                "synthesized": 0,
                "synthesized_paraphrase": 0,
                "knowledge_unit_derived": 0,
                "total": 0,
            },
        )
        counts["scenarios"] += 1
        counts["original"] += len(original_runs)
        counts["synthesized"] += len(synthesized_runs)
        if scenario_kind == DERIVED_SCENARIO_KIND:
            counts["knowledge_unit_derived"] += len(synthesized_runs)
        else:
            counts["synthesized_paraphrase"] += len(synthesized_runs)
        counts["total"] += len(original_runs) + len(synthesized_runs)

    original_case_count = len(seen_original_case_ids)
    synthesized_case_count = len(seen_synthesized_case_ids)
    comparison = {
        "artifact_type": COMPARISON_ARTIFACT_TYPE,
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "run": {
            "status": "complete",
            "generate": True,
            "target_pieces": deepcopy(
                original_source.get("run", {}).get("target_pieces") or []
            ),
        },
        "source_artifacts": {
            ORIGINAL_QUESTION_ORIGIN: {
                "artifact_type": original_source.get("artifact_type"),
                "schema_version": original_source.get("schema_version"),
                "status": original_source.get("run", {}).get("status"),
                "case_count": _source_case_count(original_source),
            },
            SYNTHESIZED_QUESTION_ORIGIN: {
                "artifact_type": synthesized_source.get("artifact_type"),
                "schema_version": synthesized_source.get("schema_version"),
                "status": synthesized_source.get("run", {}).get("status"),
                "case_count": _source_case_count(synthesized_source),
            },
        },
        "results": comparison_results,
        "summary": {
            "question_group_count": (
                len(original_by_source) + len(derived_by_source)
            ),
            "human_source_question_group_count": len(original_by_source),
            "paired_human_question_group_count": len(synthesized_by_source),
            "original_only_question_group_count": len(
                missing_synthesized_sources
            ),
            "knowledge_unit_derived_question_group_count": len(
                derived_by_source
            ),
            "scenario_count": len(comparison_results),
            "paired_scenario_count": paired_scenarios,
            "original_only_scenario_count": original_only_scenarios,
            "synthesized_only_scenario_count": synthesized_only_scenarios,
            "original_case_count": original_case_count,
            "synthesized_paraphrase_case_count": (
                synthesized_case_count - derived_case_count
            ),
            "knowledge_unit_derived_case_count": derived_case_count,
            "synthesized_case_count": synthesized_case_count,
            "total_case_count": original_case_count + synthesized_case_count,
            "per_piece": per_piece,
        },
        "viewer_projection": {
            "projection": (
                "all_original_paraphrase_and_ku_derived_cases_by_scenario"
            ),
            "scenario_join": [
                "piece_id",
                "source_evaluation_id",
                "case_kind",
                "measure_range",
            ],
        },
    }
    return attach_quality_review(comparison, quality_review_payload)


class EvaluationViewerRequestHandler(BaseHTTPRequestHandler):
    """Serve only the evaluation viewer and its immutable result snapshot."""

    server_version = "SopranoQAEvaluationViewer/1.0"

    def log_message(self, format_string: str, *args: object) -> None:
        sys.stderr.write(
            f"{self.address_string()} - {format_string % args}\n"
        )

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self'; "
            "connect-src 'self'; "
            "img-src 'self' data:; "
            "object-src 'none'; "
            "base-uri 'none'; "
            "frame-ancestors 'none'",
        )
        super().end_headers()

    def _send_bytes(
        self,
        body: bytes,
        *,
        content_type: str,
        status: HTTPStatus = HTTPStatus.OK,
        head_only: bool = False,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        if head_only:
            return
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_json_error(
        self,
        message: str,
        *,
        code: str,
        status: HTTPStatus,
        head_only: bool = False,
        headers: dict[str, str] | None = None,
    ) -> None:
        body = json.dumps(
            {"error": message, "code": code},
            ensure_ascii=False,
        ).encode("utf-8")
        self._send_bytes(
            body,
            content_type="application/json; charset=utf-8",
            status=status,
            head_only=head_only,
            headers=headers,
        )

    def _allow_local_request(self, *, head_only: bool) -> bool:
        host = self.headers.get("Host", "")
        try:
            parsed_host = urlsplit(f"//{host}")
            hostname = parsed_host.hostname
            host_port = parsed_host.port
        except ValueError:
            hostname = None
            host_port = None
        if hostname not in LOCAL_HOSTS:
            self._send_json_error(
                "This viewer accepts only loopback requests.",
                code="non_loopback_request",
                status=HTTPStatus.FORBIDDEN,
                head_only=head_only,
            )
            return False

        origin = self.headers.get("Origin")
        if not origin:
            return True
        try:
            parsed_origin = urlparse(origin)
            origin_port = parsed_origin.port
        except ValueError:
            parsed_origin = None
            origin_port = None
        if (
            parsed_origin is None
            or parsed_origin.scheme != "http"
            or parsed_origin.hostname not in LOCAL_HOSTS
            or parsed_origin.hostname != hostname
            or origin_port != host_port
        ):
            self._send_json_error(
                "Cross-origin requests are not allowed.",
                code="cross_origin_request",
                status=HTTPStatus.FORBIDDEN,
                head_only=head_only,
            )
            return False
        return True

    def _send_file(
        self,
        path: Path,
        *,
        content_type: str,
        head_only: bool,
        missing_code: str = "not_found",
    ) -> None:
        try:
            body = path.read_bytes()
        except FileNotFoundError:
            self._send_json_error(
                "Not found.",
                code=missing_code,
                status=HTTPStatus.NOT_FOUND,
                head_only=head_only,
            )
            return
        except OSError as error:
            self._send_json_error(
                f"Could not read viewer data: {error}",
                code="file_read_failed",
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
                head_only=head_only,
            )
            return
        self._send_bytes(
            body,
            content_type=content_type,
            head_only=head_only,
        )

    def _send_results(self, *, head_only: bool) -> None:
        try:
            original_raw = self.server.original_results_file.read_bytes()
            synthesized_raw = (
                None
                if self.server.original_only
                else self.server.synthesized_results_file.read_bytes()
            )
        except FileNotFoundError:
            self._send_json_error(
                (
                    "The original result file is required."
                    if self.server.original_only
                    else "Both original and synthesized result files are required."
                ),
                code="results_not_found",
                status=HTTPStatus.NOT_FOUND,
                head_only=head_only,
            )
            return
        except OSError as error:
            self._send_json_error(
                f"Could not read viewer data: {error}",
                code="file_read_failed",
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
                head_only=head_only,
            )
            return

        try:
            original_payload = json.loads(original_raw)
            if self.server.original_only:
                comparison = build_original_only_payload(original_payload)
            else:
                assert synthesized_raw is not None
                synthesized_payload = json.loads(synthesized_raw)
                try:
                    quality_review_raw = (
                        self.server.quality_review_file.read_bytes()
                    )
                except FileNotFoundError:
                    quality_review_payload = None
                except OSError as error:
                    raise _artifact_error(
                        "quality review",
                        f"could not be read: {error}",
                    ) from error
                else:
                    quality_review_payload = json.loads(quality_review_raw)
                comparison = build_comparison_payload(
                    original_payload,
                    synthesized_payload,
                    quality_review_payload,
                )
            body = (
                json.dumps(comparison, ensure_ascii=False, indent=2) + "\n"
            ).encode("utf-8")
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            TypeError,
            EvaluationArtifactError,
        ) as error:
            self._send_json_error(
                f"Could not parse viewer data: {error}",
                code="results_invalid",
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
                head_only=head_only,
            )
            return

        self._send_bytes(
            body,
            content_type="application/json; charset=utf-8",
            head_only=head_only,
        )

    def _send_viewer_page(
        self,
        *,
        red_flags: bool,
        head_only: bool,
    ) -> None:
        index_path = self.server.evaluation_root / "index.html"
        try:
            html = index_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            self._send_json_error(
                "Not found.",
                code="not_found",
                status=HTTPStatus.NOT_FOUND,
                head_only=head_only,
            )
            return
        except (OSError, UnicodeDecodeError) as error:
            self._send_json_error(
                f"Could not read viewer page: {error}",
                code="file_read_failed",
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
                head_only=head_only,
            )
            return
        if red_flags:
            marker = 'data-viewer-mode="comparison"'
            if marker not in html:
                self._send_json_error(
                    "Viewer page is missing its page-mode marker.",
                    code="viewer_invalid",
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                    head_only=head_only,
                )
                return
            html = html.replace(
                marker,
                'data-viewer-mode="red-flags"',
                1,
            )
        self._send_bytes(
            html.encode("utf-8"),
            content_type="text/html; charset=utf-8",
            head_only=head_only,
        )

    def _dispatch_read(self, *, head_only: bool) -> None:
        if not self._allow_local_request(head_only=head_only):
            return

        path = unquote(urlsplit(self.path).path)
        evaluation_root = self.server.evaluation_root
        if path == RESULTS_ROUTE:
            self._send_results(head_only=head_only)
            return
        if path in RED_FLAGS_ROUTES:
            self._send_viewer_page(red_flags=True, head_only=head_only)
            return

        static_route = STATIC_ROUTES.get(path)
        if static_route is not None:
            filename, content_type = static_route
            self._send_file(
                evaluation_root / filename,
                content_type=content_type,
                head_only=head_only,
            )
            return

        self._send_json_error(
            "Not found.",
            code="not_found",
            status=HTTPStatus.NOT_FOUND,
            head_only=head_only,
        )

    def _method_not_allowed(self) -> None:
        if not self._allow_local_request(head_only=False):
            return
        self._send_json_error(
            "Only GET and HEAD are supported.",
            code="method_not_allowed",
            status=HTTPStatus.METHOD_NOT_ALLOWED,
            headers={"Allow": "GET, HEAD"},
        )

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch_read(head_only=False)

    def do_HEAD(self) -> None:  # noqa: N802
        self._dispatch_read(head_only=True)

    def do_POST(self) -> None:  # noqa: N802
        self._method_not_allowed()

    def do_PUT(self) -> None:  # noqa: N802
        self._method_not_allowed()

    def do_PATCH(self) -> None:  # noqa: N802
        self._method_not_allowed()

    def do_DELETE(self) -> None:  # noqa: N802
        self._method_not_allowed()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._method_not_allowed()

    def do_TRACE(self) -> None:  # noqa: N802
        self._method_not_allowed()


def create_server(
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    evaluation_root: Path = EVALUATION_ROOT,
    original_results_file: Path = DEFAULT_ORIGINAL_RESULTS_FILE,
    synthesized_results_file: Path = DEFAULT_SYNTHESIZED_RESULTS_FILE,
    quality_review_file: Path = DEFAULT_QUALITY_REVIEW_FILE,
    original_only: bool = False,
) -> EvaluationViewerServer:
    """Create a viewer server; callers remain responsible for serving it."""

    return EvaluationViewerServer(
        (host, port),
        evaluation_root=evaluation_root,
        original_results_file=original_results_file,
        synthesized_results_file=synthesized_results_file,
        quality_review_file=quality_review_file,
        original_only=original_only,
    )


def _parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--original-only",
        action="store_true",
        help=(
            "Preview only the complete original evaluation artifact without "
            "reading synthesized results or a semantic quality review."
        ),
    )
    parser.add_argument(
        "--original-results-file",
        type=Path,
        default=DEFAULT_ORIGINAL_RESULTS_FILE,
        help=(
            "Complete current canonical five-piece evaluation JSON."
        ),
    )
    parser.add_argument(
        "--synthesized-results-file",
        type=Path,
        default=DEFAULT_SYNTHESIZED_RESULTS_FILE,
        help="Complete current synthesized-question evaluation JSON.",
    )
    parser.add_argument(
        "--quality-review-file",
        type=Path,
        default=DEFAULT_QUALITY_REVIEW_FILE,
        help=(
            "Optional complete direct semantic review JSON. If absent, the "
            "viewer reports semantic review as not available."
        ),
    )
    arguments = parser.parse_args(argv)
    required_files = [
        ("--original-results-file", arguments.original_results_file),
    ]
    if not arguments.original_only:
        required_files.append(
            (
                "--synthesized-results-file",
                arguments.synthesized_results_file,
            )
        )
    for option, path in required_files:
        if not path.is_file():
            parser.error(f"{option} must point to an existing result artifact")
    return arguments


def main() -> None:
    arguments = _parse_arguments()

    server = create_server(
        port=arguments.port,
        original_results_file=arguments.original_results_file,
        synthesized_results_file=arguments.synthesized_results_file,
        quality_review_file=arguments.quality_review_file,
        original_only=arguments.original_only,
    )
    host, port = server.server_address[:2]
    print(
        f"Serving RAG answer comparison at http://{host}:{port}/",
        flush=True,
    )
    print(
        f"Original results: {server.original_results_file}",
        flush=True,
    )
    if arguments.original_only:
        print("Mode: original-only preview", flush=True)
    else:
        print(
            f"Synthesized results: {server.synthesized_results_file}",
            flush=True,
        )
        print(
            "Semantic quality review: "
            + (
                str(server.quality_review_file)
                if server.quality_review_file.is_file()
                else "not available"
            ),
            flush=True,
        )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
