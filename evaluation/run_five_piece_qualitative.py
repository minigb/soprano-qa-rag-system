#!/usr/bin/env python3
"""Run the lightweight five-piece qualitative RAG+LLM evaluation.

This runner is deliberately separate from the strict schema-1.3 automatic-
pass protocol in ``run_question_evaluation.py``.  It checkpoints the real
service response plus deterministic retrieval diagnostics and authoritative
reference material for later human inspection or a future LLM-as-a-judge
pass.  It does not issue an automatic quality verdict.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from soprano_qa.settings import load_settings  # noqa: E402


ARTIFACT_TYPE = "five_piece_qualitative_rag_llm_evaluation"
SCHEMA_VERSION = "1.0"
PIECE_IDS = (
    "die-forelle",
    "in-flowery-clouds",
    "la-capinera",
    "nella-fantasia",
    "una-voce-poco-fa",
)
ELIGIBLE_MEASURE_STATUSES = {"specific", "whole_piece", "unspecified"}
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation" / "five_piece_qualitative.json"
EXCLUDED_INFERENCE_CASES = {
    ("kim-la-capinera-01", (78, 81)): (
        "The linked unit transfers an opening pp idea to the reprise, but "
        "its lyric anchor does not support this question at measures 78-81."
    ),
}


class QualitativeEvaluationInputError(ValueError):
    """Raised when an evaluation input or resume artifact is inconsistent."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_file_records(paths: Iterable[Path]) -> list[dict[str, str]]:
    records = []
    for path in sorted({path.resolve() for path in paths}):
        if not path.is_file():
            raise FileNotFoundError(f"Evaluation input not found: {path}")
        records.append({"path": str(path), "sha256": file_sha256(path)})
    return records


def embedding_model_record(path: Path) -> dict[str, Any]:
    """Describe an optional GGUF checkpoint without requiring its presence."""

    resolved = path.resolve()
    if resolved.is_file():
        return {
            "path": str(resolved),
            "checkpoint_exists": True,
            "kind": "file",
            "backend": "llama-cpp-embedding",
            "sha256": file_sha256(resolved),
        }
    return {
        "path": str(resolved),
        "checkpoint_exists": resolved.exists(),
        "kind": "directory" if resolved.is_dir() else "missing",
        "backend": "llama-cpp-embedding",
        "sha256": None,
    }


def ranges_overlap(left: Sequence[int], right: Sequence[int]) -> bool:
    return left[0] <= right[1] and right[0] <= left[1]


def validate_measure_range(value: Any, *, label: str) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(not isinstance(item, int) for item in value)
        or value[0] < 1
        or value[1] < value[0]
    ):
        raise QualitativeEvaluationInputError(
            f"{label}: expected a positive inclusive [start, end] range"
        )
    return list(value)


def confirmed_inference_ranges(units: Sequence[Mapping[str, Any]]) -> list[list[int]]:
    """Return the inventory-compatible interval union for linked units."""

    ranges = sorted(
        validate_measure_range(
            value,
            label=f"{unit['knowledge_unit_id']}.measure_ranges",
        )
        for unit in units
        if unit.get("measure_status") == "specific"
        for value in unit.get("measure_ranges") or []
    )
    merged: list[list[int]] = []
    for start, end in ranges:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def _unit_reference(unit: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "knowledge_unit_id": unit["knowledge_unit_id"],
        "source_ids": list(unit["source_ids"]),
        "answer": unit["answer"],
        "rewrite_status": unit["rewrite_status"],
        "rewrite_notes": unit.get("rewrite_notes", ""),
        "measure_status": unit["measure_status"],
        "measure_ranges": deepcopy(unit.get("measure_ranges") or []),
        "measure_notes": unit.get("measure_notes", ""),
    }


def _unit_is_retrieval_eligible(
    unit: Mapping[str, Any],
    *,
    source_included: bool,
) -> bool:
    return (
        source_included
        and unit.get("rewrite_status") == "ready"
        and unit.get("measure_status") in ELIGIBLE_MEASURE_STATUSES
    )


def _range_applicable_unit_ids(
    units: Sequence[Mapping[str, Any]],
    measure_range: list[int] | None,
) -> list[str]:
    if measure_range is None:
        return [unit["knowledge_unit_id"] for unit in units]
    applicable = []
    for unit in units:
        if unit.get("measure_status") == "whole_piece":
            applicable.append(unit["knowledge_unit_id"])
        elif unit.get("measure_status") == "specific" and any(
            ranges_overlap(candidate, measure_range)
            for candidate in unit.get("measure_ranges") or []
        ):
            applicable.append(unit["knowledge_unit_id"])
    return applicable


def _case_id(source_id: str, measure_range: list[int] | None) -> str:
    if measure_range is None:
        return f"{source_id}__no-range"
    return f"{source_id}__m{measure_range[0]}-{measure_range[1]}"


def load_qualitative_questions(
    dataset_root: Path,
    *,
    piece_ids: Sequence[str] = PIECE_IDS,
) -> tuple[list[dict[str, Any]], list[Path]]:
    """Load every answerable annotator paraphrase and expand its cases."""

    questions: list[dict[str, Any]] = []
    input_paths: list[Path] = []
    inventory_root = dataset_root / "expert_curation" / "evaluation_questions"
    review_root = dataset_root / "expert_curation" / "review"

    for piece_id in piece_ids:
        inventory_path = inventory_root / f"{piece_id}.json"
        review_path = review_root / f"{piece_id}.json"
        inventory = load_json(inventory_path)
        review = load_json(review_path)
        input_paths.extend((inventory_path, review_path))
        if inventory.get("piece_id") != piece_id:
            raise QualitativeEvaluationInputError(
                f"{inventory_path}: piece_id does not match filename"
            )
        if review.get("piece_id") != piece_id:
            raise QualitativeEvaluationInputError(
                f"{review_path}: piece_id does not match filename"
            )

        sources = {
            source["source_id"]: source
            for source in review["source_annotations"]
        }
        units = {
            unit["knowledge_unit_id"]: unit
            for unit in review["knowledge_units"]
        }

        for item in inventory["questions"]:
            source_id = item["source_id"]
            try:
                source = sources[source_id]
            except KeyError as error:
                raise QualitativeEvaluationInputError(
                    f"{inventory_path}: unknown source_id {source_id}"
                ) from error
            if item["original_question"] != source["question"]:
                raise QualitativeEvaluationInputError(
                    f"{source_id}: original question is not verbatim"
                )

            # The inventory preserves unanswerable questions for provenance;
            # qualitative inference includes every other nonempty expert answer.
            if source.get("curation_status") == "excluded_unanswerable":
                continue
            if source.get("curation_status") != "included":
                raise QualitativeEvaluationInputError(
                    f"{source_id}: unsupported curation_status"
                )
            if not source.get("answer", "").strip():
                raise QualitativeEvaluationInputError(
                    f"{source_id}: included source has no expert answer"
                )
            question_text = item.get("paraphrased_question", "")
            if not isinstance(question_text, str) or not question_text.strip():
                raise QualitativeEvaluationInputError(
                    f"{source_id}: paraphrased question is empty"
                )

            linked_units = []
            for unit_id in item["knowledge_unit_ids"]:
                try:
                    unit = units[unit_id]
                except KeyError as error:
                    raise QualitativeEvaluationInputError(
                        f"{source_id}: unknown knowledge unit {unit_id}"
                    ) from error
                if source_id not in unit["source_ids"]:
                    raise QualitativeEvaluationInputError(
                        f"{source_id}: {unit_id} does not link back to source"
                    )
                linked_units.append(unit)
            if not linked_units:
                raise QualitativeEvaluationInputError(
                    f"{source_id}: answerable source has no linked units"
                )

            inference_ranges = [
                validate_measure_range(
                    value,
                    label=f"{source_id}.inference_measure_ranges",
                )
                for value in item.get("inference_measure_ranges", [])
            ]
            expected_ranges = confirmed_inference_ranges(linked_units)
            if inference_ranges != expected_ranges:
                raise QualitativeEvaluationInputError(
                    f"{source_id}: inference_measure_ranges are stale; "
                    f"expected {expected_ranges}, found {inference_ranges}"
                )

            eligible_ids = [
                unit["knowledge_unit_id"]
                for unit in linked_units
                if _unit_is_retrieval_eligible(
                    unit,
                    source_included=True,
                )
            ]
            if list(item["retrieval_eligible_knowledge_unit_ids"]) != eligible_ids:
                raise QualitativeEvaluationInputError(
                    f"{source_id}: retrieval eligibility is stale"
                )

            excluded_inference_ranges = [
                {
                    "measure_range": deepcopy(measure_range),
                    "reason": EXCLUDED_INFERENCE_CASES[
                        (source_id, tuple(measure_range))
                    ],
                }
                for measure_range in inference_ranges
                if (source_id, tuple(measure_range))
                in EXCLUDED_INFERENCE_CASES
            ]
            active_ranges = [
                measure_range
                for measure_range in inference_ranges
                if (source_id, tuple(measure_range))
                not in EXCLUDED_INFERENCE_CASES
            ]
            case_ranges: list[list[int] | None] = (
                active_ranges if inference_ranges else [None]
            )
            cases = []
            for measure_range in case_ranges:
                applicable_ids = _range_applicable_unit_ids(
                    linked_units,
                    measure_range,
                )
                applicable_eligible_ids = [
                    unit_id
                    for unit_id in eligible_ids
                    if unit_id in applicable_ids
                ]
                cases.append(
                    {
                        "case_id": _case_id(source_id, measure_range),
                        "status": "pending",
                        "attempt_count": 0,
                        "last_attempt_at": None,
                        "inference_input": {
                            "piece_id": piece_id,
                            "question": question_text,
                            "measure_range": deepcopy(measure_range),
                            "measure_range_applied": measure_range is not None,
                            "measure_range_provenance": (
                                "confirmed_linked_specific_knowledge_unit_ranges"
                                if measure_range is not None
                                else None
                            ),
                        },
                        "expected_retrieval": {
                            "source_id": source_id,
                            "linked_knowledge_unit_ids": [
                                unit["knowledge_unit_id"]
                                for unit in linked_units
                            ],
                            "retrieval_eligible_knowledge_unit_ids": eligible_ids,
                            "range_applicable_knowledge_unit_ids": applicable_ids,
                            "range_applicable_retrieval_eligible_knowledge_unit_ids": (
                                applicable_eligible_ids
                            ),
                        },
                        "pipeline_result": None,
                        "retrieval_diagnostics": None,
                        "error": None,
                    }
                )

            questions.append(
                {
                    "source_id": source_id,
                    "piece_id": piece_id,
                    "annotator": item["annotator"],
                    "inventory_schema_version": inventory["schema_version"],
                    "original_question": item["original_question"],
                    "paraphrased_question": question_text,
                    "review_status": item["review_status"],
                    "knowledge_unit_ids": list(item["knowledge_unit_ids"]),
                    "retrieval_eligible_knowledge_unit_ids": eligible_ids,
                    "measure_range_hints": deepcopy(
                        item.get("measure_range_hints", [])
                    ),
                    "inference_measure_ranges": deepcopy(inference_ranges),
                    "excluded_inference_measure_ranges": (
                        excluded_inference_ranges
                    ),
                    "inference_scope": item["inference_scope"],
                    "reference_material": {
                        "source_answer": source["answer"],
                        "source_text": source["source_text"],
                        "curation_status": source["curation_status"],
                        "curation_notes": source.get("curation_notes", ""),
                        "reference_claim_scope": deepcopy(
                            item.get("reference_claim_scope")
                        ),
                        "range_contrast_claims": deepcopy(
                            item.get("range_contrast_claims", [])
                        ),
                        "linked_knowledge_units": [
                            _unit_reference(unit) for unit in linked_units
                        ],
                    },
                    "inference_runs": cases,
                }
            )

    return questions, input_paths


def build_retrieval_diagnostics(
    case: Mapping[str, Any],
    pipeline_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Build deterministic linked-KU and source-retrieval diagnostics."""

    evidence = pipeline_result.get("evidence") or []
    evidence_ids = [str(item.get("id", "")) for item in evidence]
    expert_evidence = [
        item
        for item in evidence
        if item.get("kind") == "expert"
        or item.get("evidence_type") == "expert_annotation"
    ]
    expert_ids = [str(item.get("id", "")) for item in expert_evidence]
    expert_source_ids = sorted({
        str(source_id)
        for item in expert_evidence
        for source_id in item.get("source_ids") or []
    })
    ranks = {
        evidence_id: index
        for index, evidence_id in enumerate(evidence_ids, start=1)
        if evidence_id
    }
    expected = case["expected_retrieval"]
    linked_ids = list(expected["linked_knowledge_unit_ids"])
    target_ids = list(
        expected[
            "range_applicable_retrieval_eligible_knowledge_unit_ids"
        ]
    )
    retrieved_targets = [unit_id for unit_id in target_ids if unit_id in ranks]
    missing_targets = [unit_id for unit_id in target_ids if unit_id not in ranks]
    source_id = expected["source_id"]
    source_ranks = [
        index
        for index, item in enumerate(evidence, start=1)
        if source_id in (item.get("source_ids") or [])
    ]

    return {
        "retrieved_evidence_ids": evidence_ids,
        "retrieved_expert_knowledge_unit_ids": expert_ids,
        "retrieved_expert_source_ids": expert_source_ids,
        "expected_linked_knowledge_unit_ids": linked_ids,
        "expected_applicable_retrieval_eligible_knowledge_unit_ids": target_ids,
        "retrieved_expected_knowledge_unit_ids": retrieved_targets,
        "missing_expected_knowledge_unit_ids": missing_targets,
        "unexpected_expert_knowledge_unit_ids": [
            unit_id for unit_id in expert_ids if unit_id not in linked_ids
        ],
        "expected_knowledge_unit_ranks": {
            unit_id: ranks.get(unit_id) for unit_id in target_ids
        },
        "retrieval_target_available": bool(target_ids),
        "any_expected_knowledge_unit_retrieved": (
            bool(retrieved_targets) if target_ids else None
        ),
        "all_expected_knowledge_units_retrieved": (
            not missing_targets if target_ids else None
        ),
        "source_retrieved": source_id in expert_source_ids,
        "source_first_rank": min(source_ranks) if source_ranks else None,
    }


def _all_cases(snapshot: Mapping[str, Any]):
    for question in snapshot["questions"]:
        for case in question["inference_runs"]:
            yield question, case


def refresh_summary(snapshot: dict[str, Any]) -> None:
    statuses: Counter[str] = Counter()
    generation_modes: Counter[str] = Counter()
    answer_bases: Counter[str] = Counter()
    per_piece: dict[str, Counter[str]] = {
        piece_id: Counter() for piece_id in PIECE_IDS
    }
    source_hits = 0
    target_cases = 0
    target_hits = 0
    target_all_hits = 0
    excluded_cases = [
        {
            "source_id": question["source_id"],
            **deepcopy(excluded),
        }
        for question in snapshot["questions"]
        for excluded in question.get("excluded_inference_measure_ranges", [])
    ]

    for question, case in _all_cases(snapshot):
        status = case["status"]
        statuses[status] += 1
        per_piece[question["piece_id"]][status] += 1
        if status != "completed":
            continue
        result = case["pipeline_result"] or {}
        diagnostics = case["retrieval_diagnostics"] or {}
        generation_modes[str(result.get("generation_mode", "unknown"))] += 1
        answer_bases[str(result.get("answer_basis", "unknown"))] += 1
        source_hits += int(bool(diagnostics.get("source_retrieved")))
        if diagnostics.get("retrieval_target_available"):
            target_cases += 1
            target_hits += int(bool(
                diagnostics.get("any_expected_knowledge_unit_retrieved")
            ))
            target_all_hits += int(bool(
                diagnostics.get("all_expected_knowledge_units_retrieved")
            ))

    total = sum(statuses.values())
    finished = statuses["completed"] + statuses["error"]
    if total and statuses["completed"] == total:
        run_status = "completed"
    elif total and finished == total and statuses["error"]:
        run_status = "complete_with_errors"
    elif finished:
        run_status = "in_progress"
    else:
        run_status = "not_started"

    snapshot["run"]["status"] = run_status
    snapshot["run"]["updated_at"] = utc_now()
    snapshot["run"]["finished_at"] = (
        snapshot["run"]["updated_at"]
        if run_status in {"completed", "complete_with_errors"}
        else None
    )
    snapshot["summary"] = {
        "question_count": len(snapshot["questions"]),
        "case_count": total,
        "excluded_inference_case_count": len(excluded_cases),
        "excluded_inference_cases": excluded_cases,
        "case_statuses": dict(sorted(statuses.items())),
        "per_piece_case_statuses": {
            piece_id: dict(sorted(counts.items()))
            for piece_id, counts in per_piece.items()
        },
        "completed_generation_modes": dict(sorted(generation_modes.items())),
        "completed_answer_bases": dict(sorted(answer_bases.items())),
        "retrieval_diagnostics": {
            "completed_cases": statuses["completed"],
            "source_retrieved_cases": source_hits,
            "cases_with_expected_eligible_target": target_cases,
            "cases_retrieving_any_expected_target": target_hits,
            "cases_retrieving_all_expected_targets": target_all_hits,
        },
        "automatic_quality_verdict": None,
        "automatic_quality_verdict_note": (
            "This qualitative artifact records diagnostics only; it is not "
            "the strict schema-1.3 automatic-pass protocol."
        ),
    }


def build_input_fingerprint(
    *,
    input_files: Sequence[Mapping[str, str]],
    generate: bool,
    top_k: int,
    retrieval_embedding_model: Mapping[str, Any] | None = None,
) -> str:
    payload = {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "piece_ids": PIECE_IDS,
        "generate": generate,
        "top_k": top_k,
        "input_files": list(input_files),
        "retrieval_embedding_model": (
            dict(retrieval_embedding_model)
            if retrieval_embedding_model is not None
            else None
        ),
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def new_snapshot(
    *,
    questions: list[dict[str, Any]],
    dataset_root: Path,
    input_files: list[dict[str, str]],
    input_fingerprint: str,
    generate: bool,
    top_k: int,
    retrieval_embedding_model: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    now = utc_now()
    snapshot = {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "protocol": {
            "name": "five-piece qualitative evaluation",
            "purpose": (
                "Preserve real RAG+LLM outputs, linked expert references, "
                "and deterministic retrieval diagnostics for manual or "
                "later LLM-as-a-judge review."
            ),
            "strict_schema_1_3_automatic_pass_protocol": False,
            "issues_automatic_quality_verdicts": False,
            "internal_model_knowledge_fallback": False,
            "documented_inference_case_exclusions": [
                {
                    "source_id": source_id,
                    "measure_range": list(measure_range),
                    "reason": reason,
                }
                for (source_id, measure_range), reason in sorted(
                    EXCLUDED_INFERENCE_CASES.items()
                )
            ],
        },
        "run": {
            "status": "not_started",
            "started_at": now,
            "updated_at": now,
            "finished_at": None,
            "dataset_root": str(dataset_root.resolve()),
            "target_pieces": list(PIECE_IDS),
            "generate": generate,
            "top_k": top_k,
            "input_fingerprint": input_fingerprint,
            "input_files": input_files,
            "retrieval_embedding_model": (
                dict(retrieval_embedding_model)
                if retrieval_embedding_model is not None
                else None
            ),
        },
        "questions": questions,
        "summary": {},
    }
    refresh_summary(snapshot)
    return snapshot


def validate_resume_snapshot(
    snapshot: Mapping[str, Any],
    *,
    input_fingerprint: str,
) -> None:
    if snapshot.get("artifact_type") != ARTIFACT_TYPE:
        raise QualitativeEvaluationInputError(
            "Output is not a five-piece qualitative evaluation artifact"
        )
    if snapshot.get("schema_version") != SCHEMA_VERSION:
        raise QualitativeEvaluationInputError(
            "Output schema version is not resumable by this runner"
        )
    if snapshot.get("run", {}).get("input_fingerprint") != input_fingerprint:
        raise QualitativeEvaluationInputError(
            "Output inputs or run configuration changed; choose a new "
            "--output path rather than mixing incomparable results"
        )


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(canonical_json_bytes(value))
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


@contextmanager
def exclusive_output_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"Another qualitative evaluation is writing {path}"
            ) from error
        yield


def selected_incomplete_cases(
    snapshot: Mapping[str, Any],
    *,
    piece_ids: Sequence[str],
    limit: int | None,
) -> list[dict[str, Any]]:
    selected_pieces = set(piece_ids)
    pending = [
        case
        for question, case in _all_cases(snapshot)
        if question["piece_id"] in selected_pieces
        and case["status"] != "completed"
    ]
    return pending[:limit] if limit is not None else pending


def run_selected_cases(
    snapshot: dict[str, Any],
    *,
    ask_fn: Callable[..., dict[str, Any]],
    piece_ids: Sequence[str],
    limit: int | None,
    generate: bool,
    top_k: int,
    checkpoint: Callable[[], None],
) -> int:
    """Run and checkpoint selected incomplete cases; return attempt count."""

    cases = selected_incomplete_cases(
        snapshot,
        piece_ids=piece_ids,
        limit=limit,
    )
    for case in cases:
        inference_input = case["inference_input"]
        case["attempt_count"] += 1
        case["last_attempt_at"] = utc_now()
        case["error"] = None
        try:
            measure_range = inference_input["measure_range"]
            result = ask_fn(
                piece_id=inference_input["piece_id"],
                question=inference_input["question"],
                measure_range=(
                    tuple(measure_range) if measure_range is not None else None
                ),
                generate=generate,
                top_k=top_k,
                allow_internal_knowledge=False,
            )
            case["pipeline_result"] = result
            case["retrieval_diagnostics"] = build_retrieval_diagnostics(
                case,
                result,
            )
            case["status"] = "completed"
        except Exception as error:  # preserve the failure and continue safely
            case["pipeline_result"] = None
            case["retrieval_diagnostics"] = None
            case["status"] = "error"
            case["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
        checkpoint()
    return len(cases)


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    settings = load_settings(use_legacy_dataset_env=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(settings["dataset_root"]),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--piece",
        action="append",
        choices=PIECE_IDS,
        help="Run only this piece in this invocation; repeat as needed.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Run at most N incomplete selected cases in this invocation.",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Ask the configured local LLM to generate grounded answers.",
    )
    parser.add_argument("--top-k", type=int, default=6)
    arguments = parser.parse_args(argv)
    if arguments.limit is not None and arguments.limit < 1:
        parser.error("--limit must be positive")
    if arguments.top_k < 1:
        parser.error("--top-k must be positive")
    return arguments


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    dataset_root = arguments.dataset_root.resolve()
    output = arguments.output.resolve()
    settings = load_settings(use_legacy_dataset_env=False)
    settings["dataset_root"] = str(dataset_root)
    os.environ["SOPRANO_QA_RAG_DATASET_ROOT"] = str(dataset_root)
    # Make the service corpus current before hashing it.  Otherwise a first
    # service call could rebuild the corpus after the resume fingerprint was
    # captured and make the next invocation incomparable to its own output.
    from soprano_qa.answer import ensure_corpus

    ensure_corpus(settings, rebuild=False)
    questions, dataset_paths = load_qualitative_questions(dataset_root)
    implementation_paths = [
        Path(__file__).resolve(),
        PROJECT_ROOT / "soprano_qa" / "answer.py",
        PROJECT_ROOT / "soprano_qa" / "corpus.py",
        PROJECT_ROOT / "soprano_qa" / "dense.py",
        PROJECT_ROOT / "soprano_qa" / "retrieval.py",
        PROJECT_ROOT / "soprano_qa" / "service.py",
        PROJECT_ROOT / "soprano_qa" / "settings.py",
        PROJECT_ROOT / "config" / "settings.json",
        Path(settings["corpus_path"]),
        Path(settings["stats_path"]),
    ]
    input_files = input_file_records([*dataset_paths, *implementation_paths])
    retrieval_embedding_model = embedding_model_record(
        Path(settings["embedding_model_path"])
    )
    input_fingerprint = build_input_fingerprint(
        input_files=input_files,
        generate=arguments.generate,
        top_k=arguments.top_k,
        retrieval_embedding_model=retrieval_embedding_model,
    )

    with exclusive_output_lock(output):
        if output.exists():
            snapshot = load_json(output)
            validate_resume_snapshot(
                snapshot,
                input_fingerprint=input_fingerprint,
            )
        else:
            snapshot = new_snapshot(
                questions=questions,
                dataset_root=dataset_root,
                input_files=input_files,
                input_fingerprint=input_fingerprint,
                generate=arguments.generate,
                top_k=arguments.top_k,
                retrieval_embedding_model=retrieval_embedding_model,
            )
            atomic_write_json(output, snapshot)

        from soprano_qa.service import ask

        selected_pieces = arguments.piece or list(PIECE_IDS)

        def checkpoint() -> None:
            refresh_summary(snapshot)
            atomic_write_json(output, snapshot)

        attempted = run_selected_cases(
            snapshot,
            ask_fn=ask,
            piece_ids=selected_pieces,
            limit=arguments.limit,
            generate=arguments.generate,
            top_k=arguments.top_k,
            checkpoint=checkpoint,
        )
        checkpoint()

    print(
        json.dumps(
            {
                "artifact": str(output),
                "attempted_cases": attempted,
                "run_status": snapshot["run"]["status"],
                "summary": snapshot["summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if snapshot["summary"]["case_statuses"].get("error", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
