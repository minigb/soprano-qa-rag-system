#!/usr/bin/env python3
"""Run the five-piece qualitative RAG answer evaluation.

It checkpoints the real service response plus deterministic retrieval
diagnostics and authoritative reference material for later human inspection
or direct semantic review. It does not issue an automatic quality verdict or
replace a generated answer with a different fallback answer.
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

from evaluation.no_range_semantic_policy import (  # noqa: E402
    COHORT_NAME as NO_RANGE_COHORT_NAME,
    INTRINSIC_RANGE_ONLY_SOURCES,
    NO_RANGE_SEMANTIC_CASES,
    NO_RANGE_SEMANTIC_EXCLUSIONS,
    REVIEWED_HINT_CONTEXT_RANGE_SOURCES,
    REPORTED_CASE_KIND,
    REPORTED_REGRESSION_FORMULATIONS,
    SHADOW_CASE_KIND,
)
from evaluation.representative_range import (  # noqa: E402
    REPRESENTATIVE_RANGE_POLICY,
    REPRESENTATIVE_RANGE_PROVENANCE,
    REVIEWED_HINT_CONTEXT_CASE_KIND,
    REVIEWED_HINT_CONTEXT_RANGE_POLICY,
    REVIEWED_HINT_CONTEXT_RANGE_PROVENANCE,
    select_representative_range,
)
from soprano_qa.answer import (  # noqa: E402
    is_grounded_insufficiency_answer,
)
from soprano_qa.dense import validate_retrieval_requirements  # noqa: E402
from soprano_qa.llm import validate_generation_requirements  # noqa: E402
from soprano_qa.settings import load_settings  # noqa: E402


ARTIFACT_TYPE = "five_piece_qualitative_rag_llm_evaluation"
SCHEMA_VERSION = "1.2"
PIECE_IDS = (
    "die-forelle",
    "in-flowery-clouds",
    "la-capinera",
    "nella-fantasia",
    "una-voce-poco-fa",
)
EXPECTED_PRODUCTION_CASE_COUNTS = {
    "die-forelle": 13,
    "in-flowery-clouds": 16,
    "la-capinera": 26,
    "nella-fantasia": 18,
    "una-voce-poco-fa": 32,
}
EXPECTED_PRODUCTION_CASE_COUNT = sum(
    EXPECTED_PRODUCTION_CASE_COUNTS.values()
)
ELIGIBLE_MEASURE_STATUSES = {"specific", "whole_piece", "unspecified"}
GENERATED_ANSWER_PATH = ("llm", "retrieved_evidence")
REMOVED_GENERATION_RESPONSE_FIELDS = {
    "generation_fallback_reason",
    "context_limited",
}
REMOVED_RETRIEVAL_RESPONSE_FIELDS = {
    "fallback_reason",
    "fallback_used",
}
REMOVED_MODEL_STATUS_FIELDS = {"llama_cpp_available"}
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation" / "qualitative.json"
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


def model_checkpoint_record(path: Path, *, backend: str) -> dict[str, Any]:
    """Fingerprint one configured GGUF checkpoint for the run contract."""

    resolved = path.resolve()
    if resolved.is_file():
        return {
            "path": str(resolved),
            "checkpoint_exists": True,
            "kind": "file",
            "backend": backend,
            "sha256": file_sha256(resolved),
        }
    return {
        "path": str(resolved),
        "checkpoint_exists": resolved.exists(),
        "kind": "directory" if resolved.is_dir() else "missing",
        "backend": backend,
        "sha256": None,
    }


def embedding_model_record(path: Path) -> dict[str, Any]:
    return model_checkpoint_record(path, backend="llama-cpp-embedding")


def generation_model_record(path: Path) -> dict[str, Any]:
    return model_checkpoint_record(path, backend="llama-cpp-python")


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


def evaluation_inference_ranges(
    *,
    source_id: str,
    piece_id: str,
    source: Mapping[str, Any],
    units: Sequence[Mapping[str, Any]],
) -> list[list[int]]:
    """Return KU ranges or an explicit hint-backed UI context override."""

    confirmed = confirmed_inference_ranges(units)
    override = REVIEWED_HINT_CONTEXT_RANGE_SOURCES.get(source_id)
    if override is None:
        return confirmed
    if confirmed:
        raise QualitativeEvaluationInputError(
            f"{source_id}: reviewed-hint context override is redundant"
        )
    if override["piece_id"] != piece_id:
        raise QualitativeEvaluationInputError(
            f"{source_id}: reviewed-hint context piece mismatch"
        )
    unit_ids = tuple(str(unit["knowledge_unit_id"]) for unit in units)
    if unit_ids != tuple(override["knowledge_unit_ids"]):
        raise QualitativeEvaluationInputError(
            f"{source_id}: reviewed-hint context KU binding drift"
        )
    ranges = [
        validate_measure_range(
            list(value),
            label=f"{source_id}.reviewed_hint_context_ranges",
        )
        for value in override["measure_ranges"]
    ]
    source_hints = [
        validate_measure_range(
            value,
            label=f"{source_id}.legacy_measure_ranges",
        )
        for value in source.get("legacy_measure_ranges") or []
    ]
    unit_hints = sorted({
        tuple(validate_measure_range(
            value,
            label=f"{unit['knowledge_unit_id']}.measure_range_hints",
        ))
        for unit in units
        for value in unit.get("measure_range_hints") or []
    })
    if source_hints != ranges or [list(value) for value in unit_hints] != ranges:
        raise QualitativeEvaluationInputError(
            f"{source_id}: reviewed-hint context no longer matches its "
            "source and KU hints"
        )
    if any(
        unit.get("measure_status") != "whole_piece"
        or unit.get("measure_ranges")
        for unit in units
    ):
        raise QualitativeEvaluationInputError(
            f"{source_id}: reviewed-hint context must not replace canonical "
            "specific KU ranges"
        )
    return ranges


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


def _case_reference_authority(
    *,
    source_id: str,
    units: Sequence[Mapping[str, Any]],
    measure_range: list[int] | None,
    applicable_ids: Sequence[str],
    semantic_required_any_ids: Sequence[str],
    semantic_supporting_ids: Sequence[str],
) -> dict[str, Any]:
    """Use only finalized, range-applicable KU answers as case authority."""

    range_label = (
        "whole-piece case"
        if measure_range is None
        else f"measure range {measure_range[0]}-{measure_range[1]}"
    )
    units_by_id = {
        str(unit["knowledge_unit_id"]): unit for unit in units
    }
    missing_ids = [
        unit_id for unit_id in applicable_ids if unit_id not in units_by_id
    ]
    if missing_ids:
        raise QualitativeEvaluationInputError(
            f"{source_id} ({range_label}): case authority refers to "
            f"unlinked knowledge units {missing_ids}"
        )

    required_ids = list(semantic_required_any_ids)
    supporting_ids = list(semantic_supporting_ids)
    required_set = set(required_ids)
    supporting_set = set(supporting_ids)
    applicable_set = set(applicable_ids)
    if required_ids or supporting_ids:
        if (
            not required_ids
            or len(required_set) != len(required_ids)
            or len(supporting_set) != len(supporting_ids)
            or required_set & supporting_set
            or required_set | supporting_set != applicable_set
        ):
            raise QualitativeEvaluationInputError(
                f"{source_id} ({range_label}): semantic reference roles "
                "must partition the applicable knowledge units into a "
                "nonempty required-any set and a disjoint supporting set"
            )

    items: list[dict[str, Any]] = []
    for unit_id in applicable_ids:
        unit = units_by_id[unit_id]
        rewrite_status = unit.get("rewrite_status")
        measure_status = unit.get("measure_status")
        if (
            rewrite_status != "ready"
            or measure_status not in ELIGIBLE_MEASURE_STATUSES
        ):
            raise QualitativeEvaluationInputError(
                f"{source_id} ({range_label}): {unit_id} is not a "
                "finalized knowledge unit "
                f"(rewrite_status={rewrite_status!r}, "
                f"measure_status={measure_status!r})"
            )
        answer = unit.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise QualitativeEvaluationInputError(
                f"{source_id} ({range_label}): finalized knowledge unit "
                f"{unit_id} has no answer"
            )
        raw_measure_ranges = unit.get("measure_ranges") or []
        if not isinstance(raw_measure_ranges, list):
            raise QualitativeEvaluationInputError(
                f"{source_id} ({range_label}): {unit_id} has invalid "
                "canonical measure ranges"
            )
        canonical_measure_ranges = [
            validate_measure_range(
                value,
                label=f"{unit_id}.measure_ranges",
            )
            for value in raw_measure_ranges
        ]
        if (
            measure_status == "specific"
            and not canonical_measure_ranges
        ) or (
            measure_status != "specific"
            and canonical_measure_ranges
        ):
            raise QualitativeEvaluationInputError(
                f"{source_id} ({range_label}): {unit_id} has canonical "
                f"measure ranges inconsistent with measure_status "
                f"{measure_status!r}"
            )
        reference_role = (
            "required_any"
            if not required_ids or unit_id in required_set
            else "supporting"
        )
        items.append(
            {
                "reference_id": f"R{len(items) + 1:03d}",
                "text": answer.strip(),
                "scope": "curated_knowledge_unit_answer",
                "flags": [],
                "scope_authority": "finalized_knowledge_unit",
                "reference_role": reference_role,
                "measure_status": measure_status,
                "measure_ranges": canonical_measure_ranges,
                "source_sentence_index": None,
                "source_claim_index": None,
                "knowledge_unit_id": unit_id,
            }
        )
    if not items:
        raise QualitativeEvaluationInputError(
            f"{source_id} ({range_label}): no finalized knowledge-unit "
            "answer is available as case authority"
        )
    return {
        "source": "applicable_finalized_knowledge_units",
        "items": items,
        "manual_review_required": False,
        "manual_review_reason": None,
    }


def _case_id(source_id: str, measure_range: list[int] | None) -> str:
    if measure_range is None:
        return f"{source_id}__no-range"
    return f"{source_id}__m{measure_range[0]}-{measure_range[1]}"


def validate_production_case_contract(
    questions: Sequence[Mapping[str, Any]],
) -> None:
    """Require the canonical complete five-piece original-case inventory."""

    observed_counts: Counter[str] = Counter()
    seen_case_ids: set[str] = set()
    for question_index, question in enumerate(questions):
        piece_id = question.get("piece_id")
        if piece_id not in EXPECTED_PRODUCTION_CASE_COUNTS:
            raise QualitativeEvaluationInputError(
                "Production question has an unsupported piece_id at index "
                f"{question_index}: {piece_id!r}"
            )
        cases = question.get("inference_runs")
        if not isinstance(cases, list):
            raise QualitativeEvaluationInputError(
                f"Production question {question.get('source_id')!r} has "
                "invalid inference_runs"
            )
        for case in cases:
            if not isinstance(case, Mapping):
                raise QualitativeEvaluationInputError(
                    f"Production question {question.get('source_id')!r} "
                    "contains a non-object case"
                )
            case_id = case.get("case_id")
            if not isinstance(case_id, str) or not case_id:
                raise QualitativeEvaluationInputError(
                    "Production case_id must be a non-empty string"
                )
            if case_id in seen_case_ids:
                raise QualitativeEvaluationInputError(
                    f"Duplicate production case_id: {case_id}"
                )
            seen_case_ids.add(case_id)
            inference_input = case.get("inference_input")
            if (
                not isinstance(inference_input, Mapping)
                or inference_input.get("piece_id") != piece_id
            ):
                raise QualitativeEvaluationInputError(
                    f"{case_id}: case piece_id does not match its question"
                )
            observed_counts[piece_id] += 1

    observed_total = sum(observed_counts.values())
    if observed_total != EXPECTED_PRODUCTION_CASE_COUNT:
        raise QualitativeEvaluationInputError(
            f"Expected {EXPECTED_PRODUCTION_CASE_COUNT} production cases, "
            f"found {observed_total}"
        )
    observed_by_piece = {
        piece_id: observed_counts[piece_id]
        for piece_id in PIECE_IDS
    }
    if observed_by_piece != EXPECTED_PRODUCTION_CASE_COUNTS:
        raise QualitativeEvaluationInputError(
            "Production case counts by piece drifted: expected "
            f"{EXPECTED_PRODUCTION_CASE_COUNTS}, found {observed_by_piece}"
        )


def load_qualitative_questions(
    dataset_root: Path,
    *,
    piece_ids: Sequence[str] = PIECE_IDS,
    enforce_no_range_semantic_policy: bool = True,
) -> tuple[list[dict[str, Any]], list[Path]]:
    """Load answerable questions with one representative ranged case each."""

    questions: list[dict[str, Any]] = []
    input_paths: list[Path] = []
    seen_no_range_policy_sources: set[str] = set()
    seen_no_range_exclusions: set[str] = set()
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
            evaluation_ranges = evaluation_inference_ranges(
                source_id=source_id,
                piece_id=piece_id,
                source=source,
                units=linked_units,
            )

            no_range_policy = NO_RANGE_SEMANTIC_CASES.get(source_id)
            no_range_exclusion = NO_RANGE_SEMANTIC_EXCLUSIONS.get(source_id)
            if not enforce_no_range_semantic_policy:
                no_range_policy = None
                no_range_exclusion = None
            if evaluation_ranges:
                if (
                    enforce_no_range_semantic_policy
                    and (no_range_policy is None)
                    == (no_range_exclusion is None)
                ):
                    raise QualitativeEvaluationInputError(
                        f"{source_id}: every measure-scoped question must "
                        "have exactly one no-range semantic policy decision"
                    )
                decision = no_range_policy or no_range_exclusion
                if decision is not None and decision["piece_id"] != piece_id:
                    raise QualitativeEvaluationInputError(
                        f"{source_id}: no-range policy piece mismatch"
                    )
                if no_range_policy is not None:
                    required_ids = list(
                        no_range_policy[
                            "required_any_knowledge_unit_ids"
                        ]
                    )
                    supporting_ids = list(
                        no_range_policy["supporting_knowledge_unit_ids"]
                    )
                    linked_ids = [
                        unit["knowledge_unit_id"] for unit in linked_units
                    ]
                    if (
                        not required_ids
                        or set(required_ids) & set(supporting_ids)
                        or not set(required_ids + supporting_ids).issubset(
                            linked_ids
                        )
                    ):
                        raise QualitativeEvaluationInputError(
                            f"{source_id}: invalid no-range semantic KU policy"
                        )
                    seen_no_range_policy_sources.add(source_id)
                elif no_range_exclusion is not None:
                    seen_no_range_exclusions.add(source_id)
            elif no_range_policy is not None or no_range_exclusion is not None:
                raise QualitativeEvaluationInputError(
                    f"{source_id}: no-range semantic policy requires "
                    "measure-scoped finalized KUs"
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
            if no_range_policy is not None:
                policy_ids = set(
                    no_range_policy["required_any_knowledge_unit_ids"]
                ) | set(no_range_policy["supporting_knowledge_unit_ids"])
                if not policy_ids.issubset(eligible_ids):
                    raise QualitativeEvaluationInputError(
                        f"{source_id}: no-range semantic targets must all be "
                        "finalized and retrieval-eligible"
                    )

            excluded_inference_ranges = [
                {
                    "measure_range": deepcopy(measure_range),
                    "reason": EXCLUDED_INFERENCE_CASES[
                        (source_id, tuple(measure_range))
                    ],
                }
                for measure_range in evaluation_ranges
                if (source_id, tuple(measure_range))
                in EXCLUDED_INFERENCE_CASES
            ]
            active_ranges = [
                measure_range
                for measure_range in evaluation_ranges
                if (source_id, tuple(measure_range))
                not in EXCLUDED_INFERENCE_CASES
            ]
            representative_range = select_representative_range(
                source_id,
                active_ranges,
            )
            if evaluation_ranges and representative_range is None:
                raise QualitativeEvaluationInputError(
                    f"{source_id}: every confirmed inference range is "
                    "excluded; no representative ranged case can be built"
                )
            case_specs: list[dict[str, Any]] = []
            if no_range_policy is not None:
                case_specs.append(
                    {
                        "case_id": _case_id(source_id, None),
                        "question": question_text,
                        "measure_range": None,
                        "case_kind": SHADOW_CASE_KIND,
                        "question_provenance": "inventory_paraphrase",
                        "evaluation_cohort": NO_RANGE_COHORT_NAME,
                    }
                )
            evaluated_range = (
                representative_range if evaluation_ranges else None
            )
            evaluated_range_uses_reviewed_hint = (
                source_id in REVIEWED_HINT_CONTEXT_RANGE_SOURCES
                and evaluated_range is not None
            )
            case_specs.append(
                {
                    "case_id": _case_id(source_id, evaluated_range),
                    "question": question_text,
                    "measure_range": evaluated_range,
                    "case_kind": (
                        REVIEWED_HINT_CONTEXT_CASE_KIND
                        if evaluated_range_uses_reviewed_hint
                        else (
                            "confirmed_measure_range"
                            if evaluated_range is not None
                            else "native_no_range"
                        )
                    ),
                    "question_provenance": "inventory_paraphrase",
                    "evaluation_cohort": None,
                    "measure_range_provenance": (
                        REVIEWED_HINT_CONTEXT_RANGE_PROVENANCE
                        if evaluated_range_uses_reviewed_hint
                        else REPRESENTATIVE_RANGE_PROVENANCE
                    ) if evaluated_range is not None else None,
                    "measure_range_selection_policy": (
                        REVIEWED_HINT_CONTEXT_RANGE_POLICY
                        if evaluated_range_uses_reviewed_hint
                        else REPRESENTATIVE_RANGE_POLICY
                    ) if evaluated_range is not None else None,
                }
            )
            for reported in REPORTED_REGRESSION_FORMULATIONS.get(
                source_id,
                (),
            ):
                if no_range_policy is None:
                    raise QualitativeEvaluationInputError(
                        f"{source_id}: reported regression is not in the "
                        "no-range semantic cohort"
                    )
                case_specs.append(
                    {
                        "case_id": (
                            f"{source_id}__{reported['case_suffix']}"
                        ),
                        "question": reported["question"],
                        "measure_range": None,
                        "case_kind": REPORTED_CASE_KIND,
                        "question_provenance": "reported_regression",
                        "evaluation_cohort": NO_RANGE_COHORT_NAME,
                    }
                )
            cases = []
            for case_spec in case_specs:
                measure_range = case_spec["measure_range"]
                applicable_ids = _range_applicable_unit_ids(
                    linked_units,
                    measure_range,
                )
                applicable_eligible_ids = [
                    unit_id
                    for unit_id in eligible_ids
                    if unit_id in applicable_ids
                ]
                semantic_required_ids = (
                    list(
                        no_range_policy[
                            "required_any_knowledge_unit_ids"
                        ]
                    )
                    if case_spec["evaluation_cohort"]
                    else []
                )
                semantic_supporting_ids = (
                    list(
                        no_range_policy["supporting_knowledge_unit_ids"]
                    )
                    if case_spec["evaluation_cohort"]
                    else []
                )
                case_reference_authority = _case_reference_authority(
                    source_id=source_id,
                    units=linked_units,
                    measure_range=measure_range,
                    applicable_ids=applicable_ids,
                    semantic_required_any_ids=semantic_required_ids,
                    semantic_supporting_ids=semantic_supporting_ids,
                )
                cases.append(
                    {
                        "case_id": case_spec["case_id"],
                        "status": "pending",
                        "attempt_count": 0,
                        "last_attempt_at": None,
                        "inference_input": {
                            "piece_id": piece_id,
                            "question": case_spec["question"],
                            "measure_range": deepcopy(measure_range),
                            "measure_range_applied": measure_range is not None,
                            "measure_range_provenance": case_spec.get(
                                "measure_range_provenance"
                            ),
                            "measure_range_selection_policy": case_spec.get(
                                "measure_range_selection_policy"
                            ),
                            "case_kind": case_spec["case_kind"],
                            "question_provenance": case_spec[
                                "question_provenance"
                            ],
                            "evaluation_cohort": case_spec[
                                "evaluation_cohort"
                            ],
                            "semantic_required_any_knowledge_unit_ids": (
                                semantic_required_ids
                            ),
                            "semantic_supporting_knowledge_unit_ids": (
                                semantic_supporting_ids
                            ),
                            "case_reference_authority": (
                                case_reference_authority
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
                            "semantic_required_any_knowledge_unit_ids": (
                                semantic_required_ids
                            ),
                            "semantic_supporting_knowledge_unit_ids": (
                                semantic_supporting_ids
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
                    "evaluation_context_measure_ranges": deepcopy(
                        evaluation_ranges
                    ),
                    "representative_inference_measure_range": deepcopy(
                        representative_range
                    ),
                    "excluded_inference_measure_ranges": (
                        excluded_inference_ranges
                    ),
                    "inference_scope": item["inference_scope"],
                    "evaluation_context_scope": (
                        "measure_range" if evaluation_ranges else "no_range"
                    ),
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

    expected_policy_sources = {
        source_id
        for source_id, policy in NO_RANGE_SEMANTIC_CASES.items()
        if policy["piece_id"] in piece_ids
    }
    expected_exclusions = {
        source_id
        for source_id, policy in NO_RANGE_SEMANTIC_EXCLUSIONS.items()
        if policy["piece_id"] in piece_ids
    }
    if (
        enforce_no_range_semantic_policy
        and seen_no_range_policy_sources != expected_policy_sources
    ):
        raise QualitativeEvaluationInputError(
            "No-range semantic inclusion policy drift: expected "
            f"{sorted(expected_policy_sources)}, found "
            f"{sorted(seen_no_range_policy_sources)}"
        )
    if (
        enforce_no_range_semantic_policy
        and seen_no_range_exclusions != expected_exclusions
    ):
        raise QualitativeEvaluationInputError(
            "No-range semantic exclusion policy drift: expected "
            f"{sorted(expected_exclusions)}, found "
            f"{sorted(seen_no_range_exclusions)}"
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
    semantic_target_ids = list(
        expected.get("semantic_required_any_knowledge_unit_ids") or []
    )
    retrieved_semantic_targets = [
        unit_id for unit_id in semantic_target_ids if unit_id in ranks
    ]
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
        "semantic_target_available": bool(semantic_target_ids),
        "semantic_required_any_knowledge_unit_ids": semantic_target_ids,
        "retrieved_semantic_target_knowledge_unit_ids": (
            retrieved_semantic_targets
        ),
        "semantic_target_ranks": {
            unit_id: ranks.get(unit_id) for unit_id in semantic_target_ids
        },
        "any_semantic_target_retrieved": (
            bool(retrieved_semantic_targets)
            if semantic_target_ids
            else None
        ),
        "semantic_target_first_rank": (
            min(ranks[unit_id] for unit_id in retrieved_semantic_targets)
            if retrieved_semantic_targets
            else None
        ),
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
    cohort_statuses: Counter[str] = Counter()
    cohort_per_piece: dict[str, Counter[str]] = {
        piece_id: Counter() for piece_id in PIECE_IDS
    }
    semantic_target_cases = 0
    semantic_target_hits = 0
    semantic_target_hit_at_1 = 0
    semantic_target_hit_at_3 = 0
    semantic_reciprocal_rank_sum = 0.0
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
        in_cohort = (
            case["inference_input"].get("evaluation_cohort")
            == NO_RANGE_COHORT_NAME
        )
        if in_cohort:
            cohort_statuses[status] += 1
            cohort_per_piece[question["piece_id"]][status] += 1
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
        if in_cohort and diagnostics.get("semantic_target_available"):
            semantic_target_cases += 1
            first_rank = diagnostics.get("semantic_target_first_rank")
            if isinstance(first_rank, int) and first_rank > 0:
                semantic_target_hits += 1
                semantic_target_hit_at_1 += int(first_rank == 1)
                semantic_target_hit_at_3 += int(first_rank <= 3)
                semantic_reciprocal_rank_sum += 1 / first_rank

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
        "representative_measure_range_case_count": sum(
            case["inference_input"].get("case_kind")
            == "confirmed_measure_range"
            for _, case in _all_cases(snapshot)
        ),
        "reviewed_hint_context_range_case_count": sum(
            case["inference_input"].get("case_kind")
            == REVIEWED_HINT_CONTEXT_CASE_KIND
            for _, case in _all_cases(snapshot)
        ),
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
        "evaluation_cohorts": {
            NO_RANGE_COHORT_NAME: {
                "case_count": sum(cohort_statuses.values()),
                "case_statuses": dict(sorted(cohort_statuses.items())),
                "per_piece_case_statuses": {
                    piece_id: dict(sorted(counts.items()))
                    for piece_id, counts in cohort_per_piece.items()
                },
                "semantic_target_cases_completed": semantic_target_cases,
                "semantic_target_hit_at_1_cases": semantic_target_hit_at_1,
                "semantic_target_hit_at_3_cases": semantic_target_hit_at_3,
                "semantic_target_hit_at_3_rate": (
                    round(
                        semantic_target_hit_at_3
                        / semantic_target_cases,
                        6,
                    )
                    if semantic_target_cases
                    else None
                ),
                "semantic_target_hit_at_k_cases": semantic_target_hits,
                "semantic_target_mean_reciprocal_rank": (
                    round(
                        semantic_reciprocal_rank_sum
                        / semantic_target_cases,
                        6,
                    )
                    if semantic_target_cases
                    else None
                ),
            }
        },
        "automatic_quality_verdict": None,
        "automatic_quality_verdict_note": (
            "This qualitative artifact records inference and retrieval "
            "diagnostics only; answer-fidelity judgment is a separate run."
        ),
    }


def build_input_fingerprint(
    *,
    input_files: Sequence[Mapping[str, str]],
    top_k: int,
    retrieval_embedding_model: Mapping[str, Any] | None = None,
    generation_model: Mapping[str, Any] | None = None,
) -> str:
    payload = {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "piece_ids": PIECE_IDS,
        "generate": True,
        "top_k": top_k,
        "input_files": list(input_files),
        "retrieval_embedding_model": (
            dict(retrieval_embedding_model)
            if retrieval_embedding_model is not None
            else None
        ),
        "generation_model": (
            dict(generation_model)
            if generation_model is not None
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
    top_k: int,
    retrieval_embedding_model: Mapping[str, Any] | None = None,
    generation_model: Mapping[str, Any] | None = None,
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
            "strict_grounded_answer_paths": True,
            "measure_range_sampling": {
                "policy": REPRESENTATIVE_RANGE_POLICY,
                "provenance": REPRESENTATIVE_RANGE_PROVENANCE,
                "cases_per_measure_scoped_question": 1,
                "candidate_ranges": (
                    "Sorted confirmed linked knowledge-unit ranges after "
                    "documented exclusions. Full candidates remain in each "
                    "question's inference_measure_ranges metadata. Exact-"
                    "lyric allowlist exceptions use an explicit reviewed "
                    "hint only as selector context with distinct provenance."
                ),
            },
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
            "no_range_semantic_evaluation": {
                "cohort": NO_RANGE_COHORT_NAME,
                "policy": (
                    "Only semantically self-contained questions whose "
                    "finalized expert KUs are measure-specific receive a "
                    "no-selected-range shadow case. The KU remains locally "
                    "scoped and does not become whole-piece authority. "
                    "Questions naming a specific lyric, word, or syllable "
                    "are representative-range-only."
                ),
                "included_source_count": len(NO_RANGE_SEMANTIC_CASES),
                "intrinsic_range_only_source_count": len(
                    INTRINSIC_RANGE_ONLY_SOURCES
                ),
                "reviewed_hint_context_source_count": len(
                    REVIEWED_HINT_CONTEXT_RANGE_SOURCES
                ),
                "reviewed_hint_context_policy": (
                    "An exact-lyric question whose KU has whole-piece claim "
                    "scope uses an explicit reviewed measure hint only as "
                    "the score-selector evaluation context."
                ),
                "reported_regression_case_count": sum(
                    len(items)
                    for items in REPORTED_REGRESSION_FORMULATIONS.values()
                ),
                "excluded_sources": [
                    {
                        "source_id": source_id,
                        "piece_id": decision["piece_id"],
                        "reason": decision["reason"],
                    }
                    for source_id, decision in sorted(
                        NO_RANGE_SEMANTIC_EXCLUSIONS.items()
                    )
                ],
            },
        },
        "run": {
            "status": "not_started",
            "started_at": now,
            "updated_at": now,
            "finished_at": None,
            "dataset_root": str(dataset_root.resolve()),
            "target_pieces": list(PIECE_IDS),
            "generate": True,
            "top_k": top_k,
            "input_fingerprint": input_fingerprint,
            "input_files": input_files,
            "retrieval_embedding_model": (
                dict(retrieval_embedding_model)
                if retrieval_embedding_model is not None
                else None
            ),
            "generation_model": (
                dict(generation_model)
                if generation_model is not None
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
    top_k: int,
    retrieval_embedding_model: Mapping[str, Any],
    generation_model: Mapping[str, Any],
) -> None:
    if snapshot.get("artifact_type") != ARTIFACT_TYPE:
        raise QualitativeEvaluationInputError(
            "Output is not a five-piece qualitative evaluation artifact"
        )
    if snapshot.get("schema_version") != SCHEMA_VERSION:
        raise QualitativeEvaluationInputError(
            "Output schema version is not resumable by this runner"
        )
    run = snapshot.get("run")
    if not isinstance(run, Mapping):
        raise QualitativeEvaluationInputError(
            "Output run metadata must be an object"
        )
    if run.get("input_fingerprint") != input_fingerprint:
        raise QualitativeEvaluationInputError(
            "Output inputs or run configuration changed; choose a new "
            "--output path rather than mixing incomparable results"
        )
    removed_run_fields = {"allow_internal_knowledge"} & set(run)
    if removed_run_fields:
        raise QualitativeEvaluationInputError(
            "Output uses removed run fields: "
            + ", ".join(sorted(removed_run_fields))
        )
    if run.get("generate") is not True:
        raise QualitativeEvaluationInputError(
            "Output run must record generation as mandatory"
        )
    if run.get("top_k") != top_k:
        raise QualitativeEvaluationInputError(
            "Output top_k does not match this run"
        )
    if run.get("retrieval_embedding_model") != dict(
        retrieval_embedding_model
    ):
        raise QualitativeEvaluationInputError(
            "Output retrieval embedding model does not match this run"
        )
    if run.get("generation_model") != dict(generation_model):
        raise QualitativeEvaluationInputError(
            "Output generation model does not match this run"
        )
    validate_resume_case_states(
        snapshot,
        retrieval_embedding_model=retrieval_embedding_model,
        generation_model=generation_model,
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


def validate_generated_response(result: Any) -> dict[str, Any]:
    """Require a generated answer grounded in retrieved corpus evidence."""

    if not isinstance(result, Mapping):
        raise QualitativeEvaluationInputError(
            "Pipeline ask() returned a non-object"
        )
    removed_fields = REMOVED_GENERATION_RESPONSE_FIELDS & set(result)
    if removed_fields:
        raise QualitativeEvaluationInputError(
            "Pipeline response uses removed generation fields: "
            + ", ".join(sorted(removed_fields))
        )
    mode = result.get("generation_mode")
    basis = result.get("answer_basis")
    if mode == "unavailable":
        raise QualitativeEvaluationInputError(
            "Pipeline returned unavailable instead of a generated RAG+LLM answer"
        )
    if (mode, basis) != GENERATED_ANSWER_PATH:
        raise QualitativeEvaluationInputError(
            "Benchmark answers must use generation_mode='llm' and "
            "answer_basis='retrieved_evidence': "
            f"generation_mode={mode!r}, answer_basis={basis!r}"
        )
    answer = result.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise QualitativeEvaluationInputError(
            "Pipeline response answer must be a non-empty string"
        )
    if is_grounded_insufficiency_answer(answer):
        raise QualitativeEvaluationInputError(
            "Pipeline response answer is a grounded-insufficiency refusal"
        )
    evidence = result.get("evidence")
    if not isinstance(evidence, list) or any(
        not isinstance(item, Mapping) for item in evidence
    ):
        raise QualitativeEvaluationInputError(
            "Pipeline response evidence must be a list of objects"
        )
    if result.get("unavailable_reason") is not None:
        raise QualitativeEvaluationInputError(
            "Generated benchmark answers cannot have unavailable_reason"
        )
    if not evidence:
        raise QualitativeEvaluationInputError(
            "Retrieved-evidence LLM answers require evidence"
        )
    return dict(result)


def _expected_checkpoint_path(
    record: Mapping[str, Any],
    *,
    label: str,
    backend: str,
) -> Path:
    if (
        record.get("checkpoint_exists") is not True
        or record.get("kind") != "file"
        or record.get("backend") != backend
        or not isinstance(record.get("sha256"), str)
        or not record.get("sha256")
    ):
        raise QualitativeEvaluationInputError(
            f"{label} checkpoint contract is invalid"
        )
    path = record.get("path")
    if not isinstance(path, str) or not path:
        raise QualitativeEvaluationInputError(
            f"{label} checkpoint path is invalid"
        )
    return Path(path).resolve()


def validate_model_contract(
    result: Mapping[str, Any],
    *,
    generation_model: Mapping[str, Any],
) -> None:
    expected_path = _expected_checkpoint_path(
        generation_model,
        label="Generation model",
        backend="llama-cpp-python",
    )
    status = result.get("model")
    if not isinstance(status, Mapping):
        raise QualitativeEvaluationInputError(
            "Pipeline response omitted model status"
        )
    removed_fields = REMOVED_MODEL_STATUS_FIELDS & set(status)
    if removed_fields:
        raise QualitativeEvaluationInputError(
            "Pipeline model status uses removed fields: "
            + ", ".join(sorted(removed_fields))
        )
    if status.get("checkpoint_exists") is not True:
        raise QualitativeEvaluationInputError(
            "Pipeline generation checkpoint is not available"
        )
    if status.get("backend") != "llama-cpp-python":
        raise QualitativeEvaluationInputError(
            "Pipeline generation backend does not match the run contract"
        )
    status_path = status.get("path")
    if (
        not isinstance(status_path, str)
        or Path(status_path).resolve() != expected_path
    ):
        raise QualitativeEvaluationInputError(
            "Pipeline generation model path does not match the run contract"
        )


def validate_retrieval_contract(
    result: Mapping[str, Any],
    *,
    retrieval_embedding_model: Mapping[str, Any],
) -> None:
    expected_path = _expected_checkpoint_path(
        retrieval_embedding_model,
        label="Retrieval embedding model",
        backend="llama-cpp-embedding",
    )
    diagnostics = result.get("retrieval")
    if not isinstance(diagnostics, Mapping):
        raise QualitativeEvaluationInputError(
            "Pipeline response omitted retrieval diagnostics"
        )
    removed_fields = REMOVED_RETRIEVAL_RESPONSE_FIELDS & set(diagnostics)
    if removed_fields:
        raise QualitativeEvaluationInputError(
            "Pipeline retrieval diagnostics use removed fields: "
            + ", ".join(sorted(removed_fields))
        )
    if diagnostics.get("configured_mode") != "hybrid":
        raise QualitativeEvaluationInputError(
            "Pipeline retrieval configured_mode must be hybrid"
        )
    if diagnostics.get("active_mode") != "hybrid":
        raise QualitativeEvaluationInputError(
            "Pipeline retrieval active_mode must be hybrid"
        )
    if diagnostics.get("dense_available") is not True:
        raise QualitativeEvaluationInputError(
            "Pipeline dense retrieval must be available"
        )
    embedding_path = diagnostics.get("embedding_model")
    if (
        not isinstance(embedding_path, str)
        or Path(embedding_path).resolve() != expected_path
    ):
        raise QualitativeEvaluationInputError(
            "Pipeline embedding model path does not match the run contract"
        )
    dimension = diagnostics.get("embedding_dimension")
    if (
        isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or dimension < 1
    ):
        raise QualitativeEvaluationInputError(
            "Pipeline embedding dimension must be a positive integer"
        )
    if diagnostics.get("last_dense_error") is not None:
        raise QualitativeEvaluationInputError(
            "Pipeline retrieval reported a dense error"
        )
    last_search = diagnostics.get("last_search_mode")
    query_route = diagnostics.get("query_route")
    allowed_route_states = {
        ("hybrid", "hybrid"),
        ("hybrid_no_dense_match", "hybrid"),
        ("lexical_route", "lexical"),
    }
    if (last_search, query_route) not in allowed_route_states:
        raise QualitativeEvaluationInputError(
            "Pipeline retrieval route is inconsistent with hybrid routing"
        )
    dense_attempted = diagnostics.get("dense_attempted")
    dense_contributed = diagnostics.get("dense_contributed")
    if not isinstance(dense_attempted, bool) or not isinstance(
        dense_contributed,
        bool,
    ):
        raise QualitativeEvaluationInputError(
            "Pipeline retrieval diagnostics must include boolean dense state"
        )
    if query_route == "hybrid" and dense_attempted is not True:
        raise QualitativeEvaluationInputError(
            "Hybrid query route must attempt dense retrieval"
        )
    if last_search == "hybrid" and dense_contributed is not True:
        raise QualitativeEvaluationInputError(
            "Hybrid retrieval result must report dense contribution"
        )
    if query_route == "lexical" and (
        dense_attempted is not False or dense_contributed is not False
    ):
        raise QualitativeEvaluationInputError(
            "Lexical query route cannot report dense work"
        )
    if last_search == "hybrid_no_dense_match" and dense_contributed is not False:
        raise QualitativeEvaluationInputError(
            "No-dense-match route cannot report dense contribution"
        )
    if not isinstance(diagnostics.get("route_reason"), str) or not diagnostics.get(
        "route_reason"
    ):
        raise QualitativeEvaluationInputError(
            "Pipeline retrieval route_reason must be a non-empty string"
        )
    ambiguous = diagnostics.get("ambiguous_candidates")
    if not isinstance(ambiguous, list):
        raise QualitativeEvaluationInputError(
            "Pipeline ambiguous_candidates must be a list"
        )


def validate_case_response(
    case: Mapping[str, Any],
    result: Any,
    *,
    retrieval_embedding_model: Mapping[str, Any],
    generation_model: Mapping[str, Any],
) -> dict[str, Any]:
    validated = validate_generated_response(result)
    inference_input = case.get("inference_input")
    if not isinstance(inference_input, Mapping):
        raise QualitativeEvaluationInputError(
            "Evaluation case inference_input must be an object"
        )
    expected_range = inference_input.get("measure_range")
    expected_scope = "range" if expected_range is not None else "whole_piece"
    if validated.get("piece_id") != inference_input.get("piece_id"):
        raise QualitativeEvaluationInputError(
            "Pipeline response piece_id does not match the case"
        )
    if validated.get("measure_range") != expected_range:
        raise QualitativeEvaluationInputError(
            "Pipeline response measure_range does not match the case"
        )
    if validated.get("scope") != expected_scope:
        raise QualitativeEvaluationInputError(
            "Pipeline response scope does not match the case"
        )
    if validated.get("pipeline") != "soprano_qa":
        raise QualitativeEvaluationInputError(
            "Pipeline response does not identify soprano_qa"
        )
    for field in ("has_primary_grounding", "has_confirmed_local_examples"):
        if not isinstance(validated.get(field), bool):
            raise QualitativeEvaluationInputError(
                f"Pipeline response {field} must be boolean"
            )
    if validated.get("has_primary_grounding") is not True:
        raise QualitativeEvaluationInputError(
            "Grounded answer path must report primary grounding"
        )
    selected_grounding = validated.get("has_selected_range_grounding")
    if expected_range is None:
        if selected_grounding is not None:
            raise QualitativeEvaluationInputError(
                "Whole-piece response cannot report selected-range grounding"
            )
    elif not isinstance(selected_grounding, bool):
        raise QualitativeEvaluationInputError(
            "Range response must report selected-range grounding"
        )
    validate_model_contract(validated, generation_model=generation_model)
    validate_retrieval_contract(
        validated,
        retrieval_embedding_model=retrieval_embedding_model,
    )
    return validated


def validate_resume_case_states(
    snapshot: Mapping[str, Any],
    *,
    retrieval_embedding_model: Mapping[str, Any],
    generation_model: Mapping[str, Any],
) -> None:
    questions = snapshot.get("questions")
    if not isinstance(questions, list):
        raise QualitativeEvaluationInputError(
            "Output questions must be a list"
        )
    case_ids: set[str] = set()
    for question in questions:
        if not isinstance(question, Mapping):
            raise QualitativeEvaluationInputError(
                "Output question entries must be objects"
            )
        cases = question.get("inference_runs")
        if not isinstance(cases, list):
            raise QualitativeEvaluationInputError(
                "Output inference_runs must be a list"
            )
        for case in cases:
            if not isinstance(case, Mapping):
                raise QualitativeEvaluationInputError(
                    "Output inference cases must be objects"
                )
            case_id = case.get("case_id")
            if not isinstance(case_id, str) or not case_id or case_id in case_ids:
                raise QualitativeEvaluationInputError(
                    "Output case IDs must be unique non-empty strings"
                )
            case_ids.add(case_id)
            status = case.get("status")
            if status not in {"pending", "completed", "error"}:
                raise QualitativeEvaluationInputError(
                    f"Output {case_id}: invalid case status"
                )
            attempts = case.get("attempt_count")
            if (
                isinstance(attempts, bool)
                or not isinstance(attempts, int)
                or attempts < 0
            ):
                raise QualitativeEvaluationInputError(
                    f"Output {case_id}: invalid attempt_count"
                )
            if status == "completed":
                if attempts < 1 or case.get("error") is not None:
                    raise QualitativeEvaluationInputError(
                        f"Output {case_id}: invalid completed case state"
                    )
                result = validate_case_response(
                    case,
                    case.get("pipeline_result"),
                    retrieval_embedding_model=retrieval_embedding_model,
                    generation_model=generation_model,
                )
                expected_diagnostics = build_retrieval_diagnostics(case, result)
                if case.get("retrieval_diagnostics") != expected_diagnostics:
                    raise QualitativeEvaluationInputError(
                        f"Output {case_id}: retrieval diagnostics are stale"
                    )
            elif status == "error":
                error = case.get("error")
                if (
                    attempts < 1
                    or case.get("pipeline_result") is not None
                    or case.get("retrieval_diagnostics") is not None
                    or not isinstance(error, Mapping)
                    or not isinstance(error.get("type"), str)
                    or not isinstance(error.get("message"), str)
                ):
                    raise QualitativeEvaluationInputError(
                        f"Output {case_id}: invalid error case state"
                    )
            elif (
                case.get("pipeline_result") is not None
                or case.get("retrieval_diagnostics") is not None
                or case.get("error") is not None
            ):
                raise QualitativeEvaluationInputError(
                    f"Output {case_id}: pending case contains output state"
                )


def run_selected_cases(
    snapshot: dict[str, Any],
    *,
    ask_fn: Callable[..., dict[str, Any]],
    piece_ids: Sequence[str],
    limit: int | None,
    top_k: int,
    retrieval_embedding_model: Mapping[str, Any],
    generation_model: Mapping[str, Any],
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
        case["status"] = "pending"
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
                generate=True,
                top_k=top_k,
            )
            result = validate_case_response(
                case,
                result,
                retrieval_embedding_model=retrieval_embedding_model,
                generation_model=generation_model,
            )
            case["pipeline_result"] = result
            case["retrieval_diagnostics"] = build_retrieval_diagnostics(
                case,
                result,
            )
            case["status"] = "completed"
        except KeyboardInterrupt:
            checkpoint()
            raise
        except Exception as error:
            case["pipeline_result"] = None
            case["retrieval_diagnostics"] = None
            case["status"] = "error"
            case["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
            checkpoint()
            raise
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
    retrieval = settings.get("retrieval")
    if (
        not isinstance(retrieval, Mapping)
        or str(retrieval.get("mode") or "").lower() != "hybrid"
    ):
        raise QualitativeEvaluationInputError(
            "Qualitative evaluation requires retrieval.mode='hybrid'"
        )
    validate_retrieval_requirements(settings)
    validate_generation_requirements(
        str(settings.get("model_path") or ""),
        settings.get("llm") or {},
    )
    # Make the service corpus current before hashing it.  Otherwise a first
    # service call could rebuild the corpus after the resume fingerprint was
    # captured and make the next invocation incomparable to its own output.
    from soprano_qa.answer import ensure_corpus

    ensure_corpus(settings, rebuild=False)
    questions, dataset_paths = load_qualitative_questions(dataset_root)
    validate_production_case_contract(questions)
    implementation_paths = [
        Path(__file__).resolve(),
        Path(__file__).with_name("no_range_semantic_policy.py").resolve(),
        Path(__file__).with_name("representative_range.py").resolve(),
        PROJECT_ROOT / "soprano_qa" / "answer.py",
        PROJECT_ROOT / "soprano_qa" / "corpus.py",
        PROJECT_ROOT / "soprano_qa" / "dense.py",
        PROJECT_ROOT / "soprano_qa" / "llm.py",
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
    generation_model = generation_model_record(
        Path(settings["model_path"])
    )
    _expected_checkpoint_path(
        retrieval_embedding_model,
        label="Retrieval embedding model",
        backend="llama-cpp-embedding",
    )
    _expected_checkpoint_path(
        generation_model,
        label="Generation model",
        backend="llama-cpp-python",
    )
    input_fingerprint = build_input_fingerprint(
        input_files=input_files,
        top_k=arguments.top_k,
        retrieval_embedding_model=retrieval_embedding_model,
        generation_model=generation_model,
    )

    with exclusive_output_lock(output):
        if output.exists():
            snapshot = load_json(output)
            validate_resume_snapshot(
                snapshot,
                input_fingerprint=input_fingerprint,
                top_k=arguments.top_k,
                retrieval_embedding_model=retrieval_embedding_model,
                generation_model=generation_model,
            )
        else:
            snapshot = new_snapshot(
                questions=questions,
                dataset_root=dataset_root,
                input_files=input_files,
                input_fingerprint=input_fingerprint,
                top_k=arguments.top_k,
                retrieval_embedding_model=retrieval_embedding_model,
                generation_model=generation_model,
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
            top_k=arguments.top_k,
            retrieval_embedding_model=retrieval_embedding_model,
            generation_model=generation_model,
            checkpoint=checkpoint,
        )

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
