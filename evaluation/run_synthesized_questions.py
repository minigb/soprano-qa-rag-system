#!/usr/bin/env python3
"""Run the five-piece synthesized-question hybrid RAG+LLM evaluation.

The benchmark manifest lives in the dataset repository, while the evaluated
implementation is imported exclusively from ``--system-root``. Corpus,
statistics, and embedding-cache writes are redirected to a temporary runtime
directory so evaluating a worktree does not modify it. Every inference call
uses grounded generation with internal model knowledge disabled.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
from pathlib import Path
import re
import shutil
import sys
import tempfile
from types import ModuleType
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT.parent / "soprano-qa-dataset-evaluation-set-synthesized"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation" / "synthesized_question_results.json"
ARTIFACT_TYPE = "soprano_qa_synthesized_hybrid_rag_llm_evaluation"
SCHEMA_VERSION = "1.0"
VARIANT_SCHEMA_VERSION = "1.0"
BASE_SCHEMA_VERSION = "1.3"
SUPPORTED_BASE_SCHEMA_VERSIONS = {"1.1", "1.3"}
TARGET_PIECES = (
    "die-forelle",
    "in-flowery-clouds",
    "la-capinera",
    "nella-fantasia",
    "una-voce-poco-fa",
)
EXPECTED_QUESTION_COUNTS = {
    "die-forelle": 10,
    "in-flowery-clouds": 10,
    "la-capinera": 20,
    "nella-fantasia": 15,
    "una-voce-poco-fa": 25,
}
EXPECTED_BASE_QUESTION_COUNT = 80
EXPECTED_ACTIVE_QUESTION_COUNT = 79
EXPECTED_VARIANT_FORMULATION_COUNT = 237
EXPECTED_CANONICAL_CASE_COUNT = 124
VARIANTS_PER_QUESTION = 3
EXPECTED_SYNTHESIZED_CASE_COUNT = 372
TRANSFORMATIONS = {
    "colloquial_reframing",
    "information_structure",
    "lexical_substitution",
    "syntactic_reframing",
    "word_order",
}
OPERATIONAL_GENERATION_FAILURE_REASONS = {
    "local generation failed",
    "model checkpoint not found",
    "llama-cpp-python is unavailable",
}
VARIANT_TOP_LEVEL_FIELDS = (
    "schema_version",
    "piece_id",
    "source_inventory_schema_version",
    "variants_per_question",
    "questions",
)
VARIANT_QUESTION_FIELDS = (
    "source_id",
    "original_question",
    "base_paraphrased_question",
    "variants",
)
VARIANT_FIELDS = (
    "variant_id",
    "question",
    "transformations",
)
ABSOLUTE_MEASURE_RE = re.compile(
    r"(?:\d+\s*(?:[-–—~]\s*\d+\s*)?(?:번째\s*)?마디|마디)"
)
EXCLUDED_INFERENCE_CASES = {
    ("kim-la-capinera-01", (78, 81)): (
        "The linked unit transfers an opening pp idea to this reprise, but "
        "its lyric anchor does not match the score at measures 78-81."
    ),
}
EXCLUDED_QUESTIONS = {
    "kim-nella-fantasia-01": {
        "piece_id": "nella-fantasia",
        "review_status": "excluded_unanswerable",
        "curation_status": "excluded_unanswerable",
        "reason": (
            "The preserved source contains a request for web material, not "
            "an expert answer to the question."
        ),
    },
}


class SynthesizedEvaluationError(RuntimeError):
    """Base class for deterministic evaluator failures."""


class EvaluationInputError(SynthesizedEvaluationError, ValueError):
    """Raised when benchmark data violates its immutable contract."""


class ResumeMismatchError(SynthesizedEvaluationError, ValueError):
    """Raised when an output belongs to different inputs or a different system."""


class RetrievalModeMismatchError(SynthesizedEvaluationError):
    """Raised when a requested retrieval mode is unavailable or fell back."""


class ModelPreflightError(SynthesizedEvaluationError):
    """Raised when grounded hybrid generation cannot start safely."""


class PipelineExecutionError(SynthesizedEvaluationError):
    """Raised when a case fails before its pipeline mode can be authenticated."""


class TargetImportError(SynthesizedEvaluationError, ImportError):
    """Raised when ``soprano_qa`` cannot be isolated to the target worktree."""


class IntegrityValidationError(SynthesizedEvaluationError):
    """Raised when authenticated inputs mutate during an evaluation."""


@dataclass(frozen=True)
class Benchmark:
    results: list[dict[str, Any]]
    input_paths: list[Path]
    exclusions: list[dict[str, Any]]


@dataclass(frozen=True)
class TargetRuntime:
    ask: Callable[..., Mapping[str, Any]]
    model_status: Callable[[], Mapping[str, Any]]
    corpus_stats: Callable[[], Mapping[str, Any]]
    load_generation_model: Callable[..., Any]
    pipeline_settings: dict[str, Any]
    pipeline_dataset_root: Path
    module_paths: dict[str, str]
    pipeline_corpus_state: dict[str, Any] = field(default_factory=dict)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError as error:
        raise EvaluationInputError(f"Required input is missing: {path}") from error
    except json.JSONDecodeError as error:
        raise EvaluationInputError(f"Invalid JSON in {path}: {error}") from error


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def pretty_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, *, required: bool = True) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.exists():
        if required:
            raise EvaluationInputError(f"Required file is missing: {resolved}")
        return {
            "path": str(resolved),
            "exists": False,
            "kind": "missing",
            "size": None,
            "sha256": None,
        }
    if not resolved.is_file():
        if required:
            raise EvaluationInputError(f"Expected a file: {resolved}")
        return {
            "path": str(resolved),
            "exists": True,
            "kind": "directory" if resolved.is_dir() else "other",
            "size": None,
            "sha256": None,
        }
    return {
        "path": str(resolved),
        "exists": True,
        "kind": "file",
        "size": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def input_file_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    return [
        file_record(path)
        for path in sorted({path.resolve() for path in paths})
    ]


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(pretty_json_bytes(value))
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    except Exception:
        try:
            temporary_path.unlink()
        except OSError:
            pass
        raise


@contextmanager
def exclusive_output_lock(output: Path) -> Iterator[None]:
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.with_name(f"{output.name}.lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def validate_measure_range(value: Any, *, label: str) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        or value[0] < 1
        or value[1] < value[0]
    ):
        raise EvaluationInputError(
            f"{label}: expected a positive inclusive [start, end] range"
        )
    return list(value)


def ranges_overlap(left: Sequence[int], right: Sequence[int]) -> bool:
    return left[0] <= right[1] and right[0] <= left[1]


def _require_string_list(value: Any, *, label: str) -> list[str]:
    if (
        not isinstance(value, list)
        or any(not isinstance(item, str) or not item for item in value)
        or len(set(value)) != len(value)
    ):
        raise EvaluationInputError(f"{label}: expected unique non-empty strings")
    return list(value)


def _unique_mapping(
    values: Any,
    *,
    key: str,
    label: str,
) -> dict[str, Mapping[str, Any]]:
    if not isinstance(values, list):
        raise EvaluationInputError(f"{label}: expected a list")
    result: dict[str, Mapping[str, Any]] = {}
    for index, value in enumerate(values):
        if not isinstance(value, Mapping):
            raise EvaluationInputError(f"{label}[{index}]: expected an object")
        identity = value.get(key)
        if not isinstance(identity, str) or not identity:
            raise EvaluationInputError(f"{label}[{index}].{key}: expected a string")
        if identity in result:
            raise EvaluationInputError(f"{label}: duplicate {key} {identity}")
        result[identity] = value
    return result


def _case_suffix(measure_range: Sequence[int] | None) -> str:
    if measure_range is None:
        return "no-range"
    return f"m{measure_range[0]}-{measure_range[1]}"


def _range_applicable_unit_ids(
    units: Sequence[Mapping[str, Any]],
    measure_range: list[int] | None,
) -> list[str]:
    if measure_range is None:
        return [str(unit["knowledge_unit_id"]) for unit in units]
    applicable: list[str] = []
    for unit in units:
        if unit.get("measure_status") == "whole_piece":
            applicable.append(str(unit["knowledge_unit_id"]))
        elif unit.get("measure_status") == "specific" and any(
            ranges_overlap(
                validate_measure_range(
                    candidate,
                    label=f"{unit['knowledge_unit_id']}.measure_ranges",
                ),
                measure_range,
            )
            for candidate in unit.get("measure_ranges") or []
        ):
            applicable.append(str(unit["knowledge_unit_id"]))
    return applicable


def _range_source_unit_ids(
    units: Sequence[Mapping[str, Any]],
    measure_range: list[int] | None,
) -> list[str]:
    if measure_range is None:
        return []
    return [
        str(unit["knowledge_unit_id"])
        for unit in units
        if unit.get("measure_status") == "specific"
        and any(
            ranges_overlap(
                validate_measure_range(
                    candidate,
                    label=f"{unit['knowledge_unit_id']}.measure_ranges",
                ),
                measure_range,
            )
            for candidate in unit.get("measure_ranges") or []
        )
    ]


def _confirmed_inference_ranges(
    units: Sequence[Mapping[str, Any]],
) -> list[list[int]]:
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


def _knowledge_unit_reference(
    unit: Mapping[str, Any],
    *,
    sources: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    contributor_records = []
    for source_id in unit.get("source_ids") or []:
        try:
            source = sources[str(source_id)]
        except KeyError as error:
            raise EvaluationInputError(
                f"{unit['knowledge_unit_id']}: unknown source {source_id}"
            ) from error
        contributor_records.append(
            {
                "source_id": source_id,
                "question": source.get("question", ""),
                "answer": source.get("answer", ""),
                "legacy_measure_range_hints": deepcopy(
                    source.get("legacy_measure_ranges") or []
                ),
                "curation_status": source.get("curation_status"),
                "curation_notes": source.get("curation_notes", ""),
            }
        )
    return {
        "knowledge_unit_id": unit["knowledge_unit_id"],
        "source_ids": list(unit.get("source_ids") or []),
        "answer": unit.get("answer", ""),
        "rewrite_status": unit.get("rewrite_status"),
        "rewrite_notes": unit.get("rewrite_notes", ""),
        "measure_status": unit.get("measure_status"),
        "measure_ranges": deepcopy(unit.get("measure_ranges") or []),
        "measure_notes": unit.get("measure_notes", ""),
        "contributing_source_answers": contributor_records,
    }


def _reference_claim_items(
    claim_scope: Mapping[str, Any],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    raw_items = claim_scope.get("sentence_items")
    if not isinstance(raw_items, list):
        raise EvaluationInputError(
            "reference_claim_scope.sentence_items must be a list"
        )
    for sentence in raw_items:
        if not isinstance(sentence, Mapping):
            raise EvaluationInputError(
                "reference_claim_scope sentence item must be an object"
            )
        sentence_index = sentence.get("sentence_index")
        text = sentence.get("text")
        scope = sentence.get("scope")
        flags = sentence.get("flags")
        if (
            isinstance(sentence_index, bool)
            or not isinstance(sentence_index, int)
            or sentence_index < 1
            or not isinstance(text, str)
            or not text.strip()
            or scope not in {
                "direct_required",
                "optional_background",
                "mixed_or_ambiguous",
            }
            or not isinstance(flags, list)
            or any(not isinstance(flag, str) for flag in flags)
        ):
            raise EvaluationInputError(
                "reference_claim_scope contains an invalid sentence item"
            )
        atoms = sentence.get("claim_items")
        if atoms is None:
            items.append(
                {
                    "text": text,
                    "scope": scope,
                    "flags": list(flags),
                    "source_sentence_index": sentence_index,
                    "source_claim_index": None,
                }
            )
            continue
        if not isinstance(atoms, list) or not atoms:
            raise EvaluationInputError(
                "reference_claim_scope.claim_items must be a non-empty list"
            )
        for atom in atoms:
            if (
                not isinstance(atom, Mapping)
                or not isinstance(atom.get("claim_index"), int)
                or not isinstance(atom.get("text"), str)
                or not atom["text"].strip()
                or atom.get("scope")
                not in {"direct_required", "optional_background"}
            ):
                raise EvaluationInputError(
                    "reference_claim_scope contains an invalid atomic claim"
                )
            items.append(
                {
                    "text": atom["text"],
                    "scope": atom["scope"],
                    "flags": list(flags),
                    "source_sentence_index": sentence_index,
                    "source_claim_index": atom["claim_index"],
                }
            )
    return items


def _case_reference_authority(
    result: Mapping[str, Any],
    *,
    measure_range: list[int] | None,
    applicable_ids: Sequence[str],
) -> dict[str, Any]:
    """Expose exact source claims only when they apply to this case."""

    reference = result["authoritative_reference"]
    linked_units = reference["linked_knowledge_units"]
    applicable = [
        unit
        for unit in linked_units
        if unit["knowledge_unit_id"] in set(applicable_ids)
    ]
    source_id = result["source_id"]
    non_applicable_source_split = any(
        source_id in unit.get("source_ids", [])
        and unit["knowledge_unit_id"] not in set(applicable_ids)
        for unit in linked_units
    )
    source_hints = reference.get("source_legacy_measure_range_hints") or []
    source_outside_range = bool(
        measure_range is not None
        and source_hints
        and not any(ranges_overlap(hint, measure_range) for hint in source_hints)
    )
    exact_scope_applies = not source_outside_range and not non_applicable_source_split
    items: list[dict[str, Any]] = []
    if exact_scope_applies:
        claim_scope = reference.get("reference_claim_scope")
        if isinstance(claim_scope, Mapping):
            claim_items = _reference_claim_items(claim_scope)
            items = [
                {
                    "reference_id": f"R{index:03d}",
                    **deepcopy(item),
                    "scope_authority": "question_level_curator",
                }
                for index, item in enumerate(claim_items, start=1)
            ]
            source = "reference_claim_scope"
            review_required = any(
                item["scope"] == "mixed_or_ambiguous" for item in items
            )
            reason = (
                "mixed_or_ambiguous source claim requires human review"
                if review_required
                else None
            )
        else:
            items = [
                {
                    "reference_id": "R001",
                    "text": reference["source_answer"],
                    "scope": "unscoped_source_answer",
                    "flags": [],
                    "scope_authority": (
                        "source_answer_without_claim_partition"
                    ),
                    "source_sentence_index": None,
                    "source_claim_index": None,
                }
            ]
            source = "unscoped_source_answer"
            review_required = True
            reason = (
                "inventory schema does not provide exact claim-scope atoms"
            )
    else:
        for unit in applicable:
            answer = unit.get("answer")
            if isinstance(answer, str) and answer.strip():
                items.append(
                    {
                        "reference_id": f"R{len(items) + 1:03d}",
                        "text": answer.strip(),
                        "scope": "optional_background",
                        "flags": [],
                        "scope_authority": "range_applicable_support_only",
                        "source_sentence_index": None,
                        "source_claim_index": None,
                        "knowledge_unit_id": unit["knowledge_unit_id"],
                    }
                )
        source = "range_applicable_linked_knowledge_units"
        review_required = True
        reason = (
            "original source answer is outside the selected range"
            if source_outside_range
            else "original source answer contains non-applicable split claims"
        )
    return {
        "source": source,
        "items": items,
        "manual_review_required": review_required,
        "manual_review_reason": reason,
    }


def _range_contrast_claims(
    value: Any,
    *,
    source_id: str,
) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise EvaluationInputError(
            f"{source_id}.range_contrast_claims: expected a list"
        )
    validated = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise EvaluationInputError(
                f"{source_id}.range_contrast_claims[{index}]: expected object"
            )
        measure_range = validate_measure_range(
            item.get("measure_range"),
            label=f"{source_id}.range_contrast_claims[{index}].measure_range",
        )
        claims = _require_string_list(
            item.get("claims"),
            label=f"{source_id}.range_contrast_claims[{index}].claims",
        )
        if not claims:
            raise EvaluationInputError(
                f"{source_id}.range_contrast_claims[{index}]: empty claims"
            )
        validated.append({"measure_range": measure_range, "claims": claims})
    return validated


def load_benchmark(
    dataset_root: Path,
    *,
    piece_ids: Sequence[str] = TARGET_PIECES,
    enforce_expected_counts: bool = True,
) -> Benchmark:
    """Bind all variant shards to answerable schema-1.3 source questions."""

    dataset_root = dataset_root.resolve()
    inventory_root = dataset_root / "expert_curation" / "evaluation_questions"
    review_root = dataset_root / "expert_curation" / "review"
    variant_root = (
        dataset_root / "expert_curation" / "evaluation_question_variants"
    )
    expected_variant_files = {f"{piece_id}.json" for piece_id in piece_ids}
    actual_variant_files = {path.name for path in variant_root.glob("*.json")}
    if actual_variant_files != expected_variant_files:
        raise EvaluationInputError(
            f"{variant_root}: expected exactly {sorted(expected_variant_files)}, "
            f"found {sorted(actual_variant_files)}"
        )

    results: list[dict[str, Any]] = []
    input_paths: list[Path] = []
    exclusions: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    seen_variant_ids: set[str] = set()
    seen_variant_texts: set[str] = set()
    base_question_count = 0
    active_question_count = 0
    variant_formulation_count = 0
    canonical_case_ids: set[str] = set()
    seen_excluded_question_ids: set[str] = set()

    for piece_id in piece_ids:
        inventory_path = inventory_root / f"{piece_id}.json"
        review_path = review_root / f"{piece_id}.json"
        variant_path = variant_root / f"{piece_id}.json"
        inventory = load_json(inventory_path)
        review = load_json(review_path)
        shard = load_json(variant_path)
        input_paths.extend((inventory_path, review_path, variant_path))

        if not isinstance(inventory, Mapping) or inventory.get(
            "schema_version"
        ) not in SUPPORTED_BASE_SCHEMA_VERSIONS:
            raise EvaluationInputError(
                f"{inventory_path}: expected one of schemas "
                f"{sorted(SUPPORTED_BASE_SCHEMA_VERSIONS)}"
            )
        if inventory.get("piece_id") != piece_id:
            raise EvaluationInputError(
                f"{inventory_path}: piece_id does not match filename"
            )
        if not isinstance(review, Mapping) or review.get("piece_id") != piece_id:
            raise EvaluationInputError(
                f"{review_path}: piece_id does not match filename"
            )
        if not isinstance(shard, Mapping) or tuple(shard) != VARIANT_TOP_LEVEL_FIELDS:
            raise EvaluationInputError(
                f"{variant_path}: non-canonical top-level fields"
            )
        if shard.get("schema_version") != VARIANT_SCHEMA_VERSION:
            raise EvaluationInputError(
                f"{variant_path}: expected variant schema {VARIANT_SCHEMA_VERSION}"
            )
        if shard.get("piece_id") != piece_id:
            raise EvaluationInputError(
                f"{variant_path}: piece_id does not match filename"
            )
        if shard.get("source_inventory_schema_version") != inventory.get(
            "schema_version"
        ):
            raise EvaluationInputError(
                f"{variant_path}: source inventory schema version drift"
            )
        if shard.get("variants_per_question") != VARIANTS_PER_QUESTION:
            raise EvaluationInputError(
                f"{variant_path}: expected exactly {VARIANTS_PER_QUESTION} variants"
            )

        base_questions = inventory.get("questions")
        variant_questions = shard.get("questions")
        if not isinstance(base_questions, list) or not isinstance(
            variant_questions, list
        ):
            raise EvaluationInputError(
                f"{piece_id}: questions and variant questions must be lists"
            )
        if enforce_expected_counts and len(base_questions) != EXPECTED_QUESTION_COUNTS[
            piece_id
        ]:
            raise EvaluationInputError(
                f"{inventory_path}: unexpected question count {len(base_questions)}"
            )

        sources = _unique_mapping(
            review.get("source_annotations"),
            key="source_id",
            label=f"{review_path}.source_annotations",
        )
        units_by_id = _unique_mapping(
            review.get("knowledge_units"),
            key="knowledge_unit_id",
            label=f"{review_path}.knowledge_units",
        )
        active_base_questions: list[tuple[int, Mapping[str, Any]]] = []
        for base_index, base in enumerate(base_questions):
            base_label = f"{inventory_path}.questions[{base_index}]"
            if not isinstance(base, Mapping):
                raise EvaluationInputError(f"{base_label}: expected object")
            source_id = base.get("source_id")
            if not isinstance(source_id, str) or not source_id:
                raise EvaluationInputError(f"{base_label}: invalid source_id")
            try:
                source = sources[source_id]
            except KeyError as error:
                raise EvaluationInputError(
                    f"{inventory_path}: unknown source_id {source_id}"
                ) from error
            if base.get("original_question") != source.get("question"):
                raise EvaluationInputError(
                    f"{source_id}: original question is not verbatim review text"
                )
            base_question = base.get("paraphrased_question")
            if not isinstance(base_question, str) or not base_question.strip():
                raise EvaluationInputError(f"{source_id}: empty base paraphrase")
            curation_status = source.get("curation_status")
            if curation_status == "excluded_unanswerable":
                exclusion = EXCLUDED_QUESTIONS.get(source_id)
                if exclusion is None or exclusion["piece_id"] != piece_id:
                    raise EvaluationInputError(
                        f"{source_id}: unexpected excluded question"
                    )
                if base.get("review_status") != exclusion["review_status"]:
                    raise EvaluationInputError(
                        f"{source_id}: excluded inventory review_status drift"
                    )
                for field in (
                    "knowledge_unit_ids",
                    "retrieval_eligible_knowledge_unit_ids",
                    "inference_measure_ranges",
                ):
                    if base.get(field) != []:
                        raise EvaluationInputError(
                            f"{source_id}: excluded question {field} must be empty"
                        )
                seen_excluded_question_ids.add(source_id)
                exclusions.append(
                    {
                        "kind": "question",
                        "piece_id": piece_id,
                        "source_id": source_id,
                        "review_status": base.get("review_status"),
                        "curation_status": curation_status,
                        "reason": exclusion["reason"],
                    }
                )
                continue
            if source_id in EXCLUDED_QUESTIONS:
                raise EvaluationInputError(
                    f"{source_id}: expected excluded curation_status"
                )
            if curation_status != "included":
                raise EvaluationInputError(
                    f"{source_id}: unsupported curation_status "
                    f"{curation_status!r}"
                )
            active_base_questions.append((base_index, base))
        if len(active_base_questions) != len(variant_questions):
            raise EvaluationInputError(
                f"{variant_path}: expected one variant entry for each active "
                f"question ({len(active_base_questions)}), found "
                f"{len(variant_questions)}"
            )
        base_question_count += len(base_questions)

        for variant_question_index, ((question_index, base), variant_entry) in enumerate(
            zip(active_base_questions, variant_questions)
        ):
            label = f"{variant_path}.questions[{variant_question_index}]"
            if not isinstance(variant_entry, Mapping) or tuple(
                variant_entry
            ) != VARIANT_QUESTION_FIELDS:
                raise EvaluationInputError(f"{label}: non-canonical fields")
            source_id = base.get("source_id")
            if not isinstance(source_id, str) or not source_id:
                raise EvaluationInputError(f"{label}: invalid source_id")
            try:
                source = sources[source_id]
            except KeyError as error:
                raise EvaluationInputError(
                    f"{inventory_path}: unknown source_id {source_id}"
                ) from error
            original_question = base.get("original_question")
            base_question = base.get("paraphrased_question")
            if original_question != source.get("question"):
                raise EvaluationInputError(
                    f"{source_id}: original question is not verbatim review text"
                )
            if variant_entry.get("source_id") != source_id:
                raise EvaluationInputError(f"{label}: source ID or ordering drift")
            if variant_entry.get("original_question") != original_question:
                raise EvaluationInputError(f"{label}: original question binding drift")
            if variant_entry.get("base_paraphrased_question") != base_question:
                raise EvaluationInputError(f"{label}: base paraphrase binding drift")
            if not isinstance(base_question, str) or not base_question.strip():
                raise EvaluationInputError(f"{source_id}: empty base paraphrase")

            variants = variant_entry.get("variants")
            if not isinstance(variants, list) or len(variants) != VARIANTS_PER_QUESTION:
                raise EvaluationInputError(f"{label}: expected exactly three variants")
            validated_variants: list[dict[str, Any]] = []
            for variant_index, variant in enumerate(variants, start=1):
                variant_label = f"{label}.variants[{variant_index - 1}]"
                if not isinstance(variant, Mapping) or tuple(variant) != VARIANT_FIELDS:
                    raise EvaluationInputError(
                        f"{variant_label}: non-canonical fields"
                    )
                expected_variant_id = f"{source_id}-syn-{variant_index:02d}"
                if variant.get("variant_id") != expected_variant_id:
                    raise EvaluationInputError(
                        f"{variant_label}: expected ID {expected_variant_id}"
                    )
                if expected_variant_id in seen_variant_ids:
                    raise EvaluationInputError(
                        f"{variant_label}: duplicate variant ID"
                    )
                seen_variant_ids.add(expected_variant_id)
                question = variant.get("question")
                if not isinstance(question, str) or not question.strip():
                    raise EvaluationInputError(f"{variant_label}: empty question")
                if question in {original_question, base_question}:
                    raise EvaluationInputError(
                        f"{variant_label}: synthesized question must differ from base"
                    )
                if ABSOLUTE_MEASURE_RE.search(question):
                    raise EvaluationInputError(
                        f"{variant_label}: question contains a measure locator"
                    )
                normalized_question = " ".join(question.split()).casefold()
                if normalized_question in seen_variant_texts:
                    raise EvaluationInputError(
                        f"{variant_label}: duplicate synthesized question text"
                    )
                seen_variant_texts.add(normalized_question)
                transformations = variant.get("transformations")
                if (
                    not isinstance(transformations, list)
                    or not transformations
                    or transformations != sorted(set(transformations))
                    or any(item not in TRANSFORMATIONS for item in transformations)
                ):
                    raise EvaluationInputError(
                        f"{variant_label}: invalid transformation tags"
                    )
                validated_variants.append(
                    {
                        "variant_id": expected_variant_id,
                        "question": question,
                        "transformations": list(transformations),
                    }
                )

            curation_status = source.get("curation_status")
            if curation_status != "included":
                raise EvaluationInputError(
                    f"{source_id}: unsupported curation_status {curation_status!r}"
                )
            if not isinstance(source.get("answer"), str) or not source["answer"].strip():
                raise EvaluationInputError(
                    f"{source_id}: included source has no expert answer"
                )

            expected_ids = _require_string_list(
                base.get("knowledge_unit_ids"),
                label=f"{source_id}.knowledge_unit_ids",
            )
            eligible_ids = _require_string_list(
                base.get("retrieval_eligible_knowledge_unit_ids"),
                label=f"{source_id}.retrieval_eligible_knowledge_unit_ids",
            )
            if not set(eligible_ids).issubset(expected_ids):
                raise EvaluationInputError(
                    f"{source_id}: retrieval-eligible IDs must be expected IDs"
                )
            linked_units: list[Mapping[str, Any]] = []
            for unit_id in expected_ids:
                try:
                    unit = units_by_id[unit_id]
                except KeyError as error:
                    raise EvaluationInputError(
                        f"{source_id}: unknown knowledge unit {unit_id}"
                    ) from error
                if source_id not in (unit.get("source_ids") or []):
                    raise EvaluationInputError(
                        f"{source_id}: {unit_id} does not link back to source"
                    )
                linked_units.append(unit)

            computed_eligible_ids = [
                str(unit["knowledge_unit_id"])
                for unit in linked_units
                if unit.get("rewrite_status") == "ready"
                and unit.get("measure_status")
                in {"specific", "whole_piece", "unspecified"}
            ]
            if eligible_ids != computed_eligible_ids:
                raise EvaluationInputError(
                    f"{source_id}: retrieval eligibility is stale"
                )

            raw_ranges = base.get("inference_measure_ranges")
            if not isinstance(raw_ranges, list):
                raise EvaluationInputError(
                    f"{source_id}.inference_measure_ranges: expected list"
                )
            ranges = [
                validate_measure_range(
                    value,
                    label=f"{source_id}.inference_measure_ranges",
                )
                for value in raw_ranges
            ]
            if ranges != sorted(ranges) or len({tuple(item) for item in ranges}) != len(
                ranges
            ):
                raise EvaluationInputError(
                    f"{source_id}: inference ranges must be sorted and unique"
                )
            expected_ranges = _confirmed_inference_ranges(linked_units)
            if ranges != expected_ranges:
                raise EvaluationInputError(
                    f"{source_id}: inference ranges are stale; expected "
                    f"{expected_ranges}, found {ranges}"
                )
            contrast_claims = _range_contrast_claims(
                base.get("range_contrast_claims"),
                source_id=source_id,
            )
            active_ranges: list[list[int]] = []
            for measure_range in ranges:
                excluded_reason = EXCLUDED_INFERENCE_CASES.get(
                    (source_id, tuple(measure_range))
                )
                if excluded_reason:
                    exclusions.append(
                        {
                            "kind": "inference_range",
                            "piece_id": piece_id,
                            "source_id": source_id,
                            "measure_range": measure_range,
                            "reason": excluded_reason,
                        }
                    )
                else:
                    active_ranges.append(measure_range)
            case_ranges: list[list[int] | None] = (
                active_ranges if ranges else [None]
            )
            raw_claim_scope = base.get("reference_claim_scope")
            if inventory["schema_version"] == "1.3":
                if not isinstance(raw_claim_scope, Mapping):
                    raise EvaluationInputError(
                        f"{source_id}.reference_claim_scope: expected object"
                    )
                expected_answer_hash = hashlib.sha256(
                    source["answer"].encode("utf-8")
                ).hexdigest()
                if raw_claim_scope.get(
                    "source_answer_sha256"
                ) != expected_answer_hash:
                    raise EvaluationInputError(
                        f"{source_id}: reference claim hash does not match "
                        "source answer"
                    )
                _reference_claim_items(raw_claim_scope)
                claim_scope: dict[str, Any] | None = dict(raw_claim_scope)
            else:
                if raw_claim_scope is not None:
                    raise EvaluationInputError(
                        f"{source_id}: schema 1.1 must not contain "
                        "reference_claim_scope"
                    )
                claim_scope = None
            authoritative_reference = {
                "source_answer": source["answer"],
                "source_text": source.get("source_text", ""),
                "source_legacy_measure_range_hints": deepcopy(
                    source.get("legacy_measure_ranges") or []
                ),
                "curation_status": curation_status,
                "curation_notes": source.get("curation_notes", ""),
                "reference_claim_scope": deepcopy(claim_scope),
                "range_contrast_claims": deepcopy(contrast_claims),
                "linked_knowledge_units": [
                    _knowledge_unit_reference(unit, sources=sources)
                    for unit in linked_units
                ],
            }
            result: dict[str, Any] = {
                "source_id": source_id,
                "piece_id": piece_id,
                "annotator": base.get("annotator"),
                "inventory_schema_version": inventory["schema_version"],
                "original_question": original_question,
                "paraphrased_question": base_question,
                "review_status": base.get("review_status"),
                "knowledge_unit_ids": expected_ids,
                "expected_retrieval_eligible_knowledge_unit_ids": eligible_ids,
                "measure_range_hints": deepcopy(
                    base.get("measure_range_hints") or []
                ),
                "inference_measure_ranges": deepcopy(ranges),
                "excluded_inference_measure_ranges": [
                    deepcopy(item)
                    for item in exclusions
                    if item["source_id"] == source_id
                    and item["kind"] == "inference_range"
                ],
                "inference_scope": base.get("inference_scope"),
                "range_contrast_claims": deepcopy(contrast_claims),
                "reference_claim_scope": deepcopy(claim_scope),
                "authoritative_reference": authoritative_reference,
                "synthesized_question_variants": deepcopy(validated_variants),
                "inference_runs": [],
            }
            for variant_index, variant in enumerate(validated_variants, start=1):
                expected_variant_id = variant["variant_id"]
                question = variant["question"]
                for measure_range in case_ranges:
                    suffix = _case_suffix(measure_range)
                    formulation_id = f"{source_id}__{suffix}"
                    case_id = f"{expected_variant_id}__{suffix}"
                    if case_id in seen_case_ids:
                        raise EvaluationInputError(f"Duplicate case ID {case_id}")
                    seen_case_ids.add(case_id)
                    canonical_case_ids.add(formulation_id)
                    applicable_ids = _range_applicable_unit_ids(
                        linked_units,
                        measure_range,
                    )
                    if not applicable_ids:
                        raise EvaluationInputError(
                            f"{case_id}: no range-applicable expected knowledge unit"
                        )
                    range_source_ids = _range_source_unit_ids(
                        linked_units,
                        measure_range,
                    )
                    allowed_contrast = [
                        claim
                        for scoped in contrast_claims
                        if scoped["measure_range"] == measure_range
                        for claim in scoped["claims"]
                    ]
                    excluded_contrast = [
                        claim
                        for scoped in contrast_claims
                        if scoped["measure_range"] != measure_range
                        for claim in scoped["claims"]
                    ]
                    inference_input = {
                        "piece_id": piece_id,
                        "source_id": source_id,
                        "question": question,
                        "synthesized_variant_id": expected_variant_id,
                        "variant_index": variant_index,
                        "transformations": list(variant["transformations"]),
                        "measure_range": deepcopy(measure_range),
                        "measure_range_applied": measure_range is not None,
                        "measure_range_provenance": (
                            "confirmed_linked_specific_knowledge_unit_ranges"
                            if measure_range is not None
                            else None
                        ),
                        "range_source_knowledge_unit_ids": range_source_ids,
                        "range_applicable_knowledge_unit_ids": applicable_ids,
                        "expected_knowledge_unit_ids": expected_ids,
                        "expected_retrieval_eligible_knowledge_unit_ids": (
                            eligible_ids
                        ),
                        "allowed_range_contrast_claims": allowed_contrast,
                        "excluded_range_contrast_claims": excluded_contrast,
                    }
                    inference_input["case_reference_authority"] = (
                        _case_reference_authority(
                            result,
                            measure_range=measure_range,
                            applicable_ids=applicable_ids,
                        )
                    )
                    result["inference_runs"].append(
                        {
                            "case_id": case_id,
                            "status": "pending",
                            "attempt_count": 0,
                            "last_attempt_at": None,
                            "completed_at": None,
                            "inference_input": inference_input,
                            "retrieval_probe": None,
                            "generated_answer": None,
                            "error": None,
                        }
                    )
            results.append(result)
            active_question_count += 1
            variant_formulation_count += len(validated_variants)

    expected_excluded_ids = {
        source_id
        for source_id, exclusion in EXCLUDED_QUESTIONS.items()
        if exclusion["piece_id"] in piece_ids
    }
    if seen_excluded_question_ids != expected_excluded_ids:
        raise EvaluationInputError(
            "Excluded question set drift: expected "
            f"{sorted(expected_excluded_ids)}, found "
            f"{sorted(seen_excluded_question_ids)}"
        )

    if enforce_expected_counts:
        if base_question_count != EXPECTED_BASE_QUESTION_COUNT:
            raise EvaluationInputError(
                f"Expected {EXPECTED_BASE_QUESTION_COUNT} base questions, "
                f"found {base_question_count}"
            )
        if active_question_count != EXPECTED_ACTIVE_QUESTION_COUNT:
            raise EvaluationInputError(
                f"Expected {EXPECTED_ACTIVE_QUESTION_COUNT} active questions, "
                f"found {active_question_count}"
            )
        if variant_formulation_count != EXPECTED_VARIANT_FORMULATION_COUNT:
            raise EvaluationInputError(
                f"Expected {EXPECTED_VARIANT_FORMULATION_COUNT} active variants, "
                f"found {variant_formulation_count}"
            )
        if len(canonical_case_ids) != EXPECTED_CANONICAL_CASE_COUNT:
            raise EvaluationInputError(
                f"Expected {EXPECTED_CANONICAL_CASE_COUNT} canonical cases, "
                f"found {len(canonical_case_ids)}"
            )
        synthesized_case_count = sum(
            len(result["inference_runs"]) for result in results
        )
        if synthesized_case_count != EXPECTED_SYNTHESIZED_CASE_COUNT:
            raise EvaluationInputError(
                f"Expected {EXPECTED_SYNTHESIZED_CASE_COUNT} synthesized cases, "
                f"found {synthesized_case_count}"
            )
    return Benchmark(
        results=results,
        input_paths=input_paths,
        exclusions=exclusions,
    )


def _resolve_settings_path(config_path: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    expanded = Path(os.path.expandvars(os.path.expanduser(value)))
    if not expanded.is_absolute():
        expanded = config_path.parent / expanded
    return expanded.resolve()


def collect_system_state(system_root: Path) -> dict[str, Any]:
    """Fingerprint pipeline code, corpus, configuration, and model assets."""

    root = system_root.resolve()
    package_root = root / "soprano_qa"
    config_path = root / "config" / "settings.json"
    if not package_root.is_dir():
        raise EvaluationInputError(f"Missing target package: {package_root}")
    config = load_json(config_path)
    if not isinstance(config, Mapping):
        raise EvaluationInputError(f"{config_path}: expected an object")

    mandatory_paths = sorted(package_root.rglob("*.py"))
    mandatory_paths.extend((config_path, root / "requirements.txt"))
    for key in ("corpus_path", "stats_path", "feature_overrides_path"):
        path = _resolve_settings_path(config_path, config.get(key))
        if path is not None:
            mandatory_paths.append(path)
    records = input_file_records(mandatory_paths)

    embedding_override = os.environ.get("SOPRANO_QA_EMBEDDING_MODEL_PATH")
    if embedding_override:
        embedding_path = Path(os.path.expanduser(embedding_override))
        if not embedding_path.is_absolute():
            embedding_path = root / embedding_path
        embedding_path = embedding_path.resolve()
    else:
        embedding_path = _resolve_settings_path(
            config_path,
            config.get("embedding_model_path"),
        )
    embedding_cache_path = _resolve_settings_path(
        config_path,
        config.get("embedding_cache_path"),
    )
    model_override = os.environ.get("SOPRANO_QA_MODEL_PATH")
    if model_override:
        model_path = Path(os.path.expanduser(model_override))
        if not model_path.is_absolute():
            model_path = root / model_path
        model_path = model_path.resolve()
    else:
        model_path = _resolve_settings_path(config_path, config.get("model_path"))
    optional_assets = {
        "embedding_model": (
            file_record(embedding_path, required=False)
            if embedding_path is not None
            else None
        ),
        "embedding_cache": (
            file_record(embedding_cache_path, required=False)
            if embedding_cache_path is not None
            else None
        ),
        "generation_model": (
            file_record(model_path, required=False)
            if model_path is not None
            else None
        ),
    }
    payload = {
        "system_root": str(root),
        "files": records,
        "optional_assets": optional_assets,
        "embedding_model_override": embedding_override,
        "generation_model_override": model_override,
    }
    return {**payload, "fingerprint": sha256_value(payload)}


def collect_runtime_state() -> dict[str, Any]:
    """Return a stable host/runtime record used by the resumability contract."""

    try:
        llama_cpp_version = importlib.metadata.version("llama-cpp-python")
    except importlib.metadata.PackageNotFoundError:
        llama_cpp_version = None
    payload = {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "llama_cpp_python_version": llama_cpp_version,
    }
    return {**payload, "fingerprint": sha256_value(payload)}


def validate_hybrid_generation_preflight(
    settings: Mapping[str, Any],
    *,
    system_state: Mapping[str, Any],
    model_status: Mapping[str, Any],
    runtime_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Require hybrid retrieval and both local checkpoints before output."""

    retrieval = settings.get("retrieval") or {}
    if not isinstance(retrieval, Mapping):
        raise ModelPreflightError("retrieval settings must be an object")
    mode = str(retrieval.get("mode") or "lexical").lower()
    if mode != "hybrid":
        raise ModelPreflightError(
            "Synthesized RAG+LLM evaluation requires retrieval.mode='hybrid'; "
            f"found {mode!r}"
        )
    if bool(retrieval.get("fallback_to_lexical", False)):
        raise ModelPreflightError(
            "Hybrid retrieval fallback_to_lexical must be disabled"
        )

    assets = system_state.get("optional_assets") or {}
    embedding = assets.get("embedding_model")
    generation = assets.get("generation_model")
    for label, record in (
        ("embedding", embedding),
        ("generation", generation),
    ):
        if not isinstance(record, Mapping) or record.get("kind") != "file":
            configured = (
                settings.get("embedding_model_path")
                if label == "embedding"
                else settings.get("model_path")
            )
            raise ModelPreflightError(
                f"Required {label} model checkpoint not found: "
                f"{configured or '(not configured)'}"
            )
        if not isinstance(record.get("sha256"), str) or not record["sha256"]:
            raise ModelPreflightError(
                f"Required {label} model could not be fingerprinted"
            )

    configured_model = Path(str(settings.get("model_path") or "")).resolve()
    status_path = Path(str(model_status.get("path") or "")).resolve()
    if status_path != configured_model:
        raise ModelPreflightError(
            "Service generation model status does not match configured model_path"
        )
    if model_status.get("checkpoint_exists") is not True:
        raise ModelPreflightError(
            f"Required generation model checkpoint not found: {configured_model}"
        )
    if model_status.get("llama_cpp_available") is not True:
        raise ModelPreflightError(
            "Required llama-cpp-python generation backend is unavailable"
        )
    if not isinstance(settings.get("llm"), Mapping):
        raise ModelPreflightError("llm settings must be configured")

    return {
        "retrieval_embedding_model": deepcopy(dict(embedding)),
        "generation_model": deepcopy(dict(generation)),
        "generation_model_status": deepcopy(dict(model_status)),
        "runtime": deepcopy(dict(runtime_state)),
    }


def initialize_target_models(
    target: TargetRuntime,
) -> dict[str, Any]:
    """Eagerly initialize dense retrieval and generation before output."""

    try:
        stats = target.corpus_stats()
    except Exception as error:
        raise ModelPreflightError(
            "Hybrid retrieval model/index initialization failed: "
            f"{type(error).__name__}: {error}"
        ) from error
    if not isinstance(stats, Mapping):
        raise ModelPreflightError("Target corpus_stats() returned a non-object")
    raw_retrieval = stats.get("retrieval")
    if not isinstance(raw_retrieval, Mapping):
        raise ModelPreflightError(
            "Target corpus_stats() omitted retrieval diagnostics"
        )
    configured = raw_retrieval.get("configured_mode")
    active = raw_retrieval.get("active_mode")
    problems = []
    if configured != "hybrid":
        problems.append(f"configured_mode={configured!r}")
    if active != "hybrid":
        problems.append(f"active_mode={active!r}")
    if raw_retrieval.get("dense_available") is not True:
        problems.append("dense_available is not true")
    if raw_retrieval.get("fallback_reason"):
        problems.append(
            f"fallback_reason={raw_retrieval.get('fallback_reason')!r}"
        )
    if raw_retrieval.get("last_dense_error"):
        problems.append(
            f"last_dense_error={raw_retrieval.get('last_dense_error')!r}"
        )
    if problems:
        raise ModelPreflightError(
            "Hybrid retrieval initialization was not clean: "
            + ", ".join(problems)
        )

    settings = target.pipeline_settings
    llm_settings = settings.get("llm") or {}
    try:
        model = target.load_generation_model(
            model_path=str(settings["model_path"]),
            n_ctx=int(llm_settings.get("n_ctx", 8192)),
            n_gpu_layers=int(llm_settings.get("n_gpu_layers", -1)),
            chat_format=str(llm_settings.get("chat_format", "chatml")),
        )
    except Exception as error:
        raise ModelPreflightError(
            "Generation model initialization failed: "
            f"{type(error).__name__}: {error}"
        ) from error
    if model is None:
        raise ModelPreflightError(
            "Generation model initialization returned no model"
        )
    return {
        "retrieval": deepcopy(dict(raw_retrieval)),
        "generation": {
            "status": "loaded",
            "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
            "model_path": str(Path(str(settings["model_path"])).resolve()),
            "n_ctx": int(llm_settings.get("n_ctx", 8192)),
            "n_gpu_layers": int(llm_settings.get("n_gpu_layers", -1)),
            "chat_format": str(llm_settings.get("chat_format", "chatml")),
        },
    }


def _labeled_input_file_records(
    paths: Iterable[tuple[str, str | Path]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_labels: set[str] = set()
    for label, path in paths:
        normalized_label = str(label)
        if normalized_label in seen_labels:
            raise EvaluationInputError(
                f"Duplicate corpus input label: {normalized_label}"
            )
        seen_labels.add(normalized_label)
        records.append(
            {
                "label": normalized_label,
                **file_record(Path(path)),
            }
        )
    return sorted(records, key=lambda record: record["label"])


def authenticate_pipeline_corpus(
    settings: Mapping[str, Any],
    corpus_module: ModuleType,
) -> dict[str, Any]:
    """Bind the queried corpus to the exact dataset inputs that built it."""

    fingerprint_fn = getattr(corpus_module, "corpus_input_fingerprint", None)
    input_paths_fn = getattr(corpus_module, "corpus_input_paths", None)
    if not callable(fingerprint_fn) or not callable(input_paths_fn):
        raise EvaluationInputError(
            "Selected system must expose corpus_input_fingerprint and "
            "corpus_input_paths"
        )
    corpus_path = Path(str(settings["corpus_path"])).resolve()
    stats_path = Path(str(settings["stats_path"])).resolve()
    if not corpus_path.is_file() or not stats_path.is_file():
        raise EvaluationInputError(
            "The selected system needs a prebuilt corpus and stats before "
            "evaluation"
        )
    try:
        stats = load_json(stats_path)
        expected_fingerprint = fingerprint_fn(dict(settings))
        labeled_paths = input_paths_fn(dict(settings))
        input_files = _labeled_input_file_records(labeled_paths)
    except (OSError, ValueError, TypeError, FileNotFoundError) as error:
        raise EvaluationInputError(
            "Could not authenticate the selected system's corpus inputs"
        ) from error
    expected_exports = settings.get("web_export_files", ["research-open.jsonl"])
    checks = {
        "stats_object": isinstance(stats, dict),
        "schema": isinstance(stats, dict)
        and stats.get("corpus_schema_version") == 6,
        "web_exports": isinstance(stats, dict)
        and stats.get("web_export_files") == expected_exports,
        "dataset_root": isinstance(stats, dict)
        and stats.get("dataset_root")
        == os.path.realpath(str(settings["dataset_root"])),
        "input_fingerprint": isinstance(stats, dict)
        and stats.get("input_fingerprint") == expected_fingerprint,
        "corpus_sha256": isinstance(stats, dict)
        and stats.get("corpus_sha256") == file_sha256(corpus_path),
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise EvaluationInputError(
            "The selected system's corpus is stale or unauthenticated "
            f"({', '.join(failures)}). Rebuild it before evaluation; the "
            "evaluator will not mutate trusted inputs."
        )
    return {
        "dataset_root": str(Path(str(settings["dataset_root"])).resolve()),
        "input_fingerprint": expected_fingerprint,
        "input_files": input_files,
        "corpus": file_record(corpus_path),
        "stats": file_record(stats_path),
    }


def refresh_pipeline_input_records(
    state: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return _labeled_input_file_records(
        (record["label"], record["path"])
        for record in state.get("input_files", [])
    )


def _module_is_within(module: ModuleType, root: Path) -> bool:
    path = getattr(module, "__file__", None)
    if not path:
        return True
    try:
        Path(path).resolve().relative_to(root)
        return True
    except ValueError:
        return False


def _remove_target_modules() -> None:
    for name in list(sys.modules):
        if name == "soprano_qa" or name.startswith("soprano_qa."):
            sys.modules.pop(name, None)


def validate_target_retrieval_requirements(
    settings: Mapping[str, Any],
    dense_module: ModuleType | None,
) -> None:
    """Preflight both current and pre-validator target worktrees."""

    validator = getattr(
        dense_module,
        "validate_retrieval_requirements",
        None,
    )
    if callable(validator):
        try:
            validator(dict(settings))
        except Exception as error:
            raise ModelPreflightError(
                "Hybrid retrieval requirements failed: "
                f"{type(error).__name__}: {error}"
            ) from error
        return

    retrieval = settings.get("retrieval") or {}
    mode = str(retrieval.get("mode") or "lexical").lower()
    if mode == "lexical":
        return
    if mode != "hybrid":
        raise ModelPreflightError(
            f"Unsupported retrieval.mode for evaluation: {mode!r}"
        )
    configured_path = str(
        settings.get("embedding_model_path") or ""
    ).strip()
    if configured_path and Path(configured_path).expanduser().is_file():
        return
    display_path = configured_path or (
        "(embedding_model_path is not configured)"
    )
    raise ModelPreflightError(
        "Required hybrid-retrieval embedding checkpoint not found: "
        f"{display_path}. Download the embedding model before running "
        "retrieval evaluation, or explicitly configure retrieval.mode "
        "as `lexical`."
    )


@contextmanager
def target_runtime(
    system_root: Path,
    *,
    pipeline_dataset_root: Path | None = None,
) -> Iterator[TargetRuntime]:
    """Import one target package and redirect all known target-local writes."""

    root = system_root.resolve()
    package_init = root / "soprano_qa" / "__init__.py"
    if not package_init.is_file():
        raise TargetImportError(f"Target package is missing: {package_init}")

    # Programmatic callers (including a full unit-test process) may already
    # have imported the local package.  Snapshot and evict that complete
    # module family so target imports cannot mix worktrees, then restore the
    # caller's exact module objects on exit.
    original_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "soprano_qa" or name.startswith("soprano_qa.")
    }
    original_path = list(sys.path)
    original_dont_write = sys.dont_write_bytecode
    tracked_env = (
        "SOPRANO_QA_RAG_DATASET_ROOT",
        "SOPRANO_QA_DATASET_ROOT",
    )
    original_env = {name: os.environ.get(name) for name in tracked_env}
    try:
        _remove_target_modules()
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(root))
        os.environ.pop("SOPRANO_QA_DATASET_ROOT", None)
        if pipeline_dataset_root is None:
            os.environ.pop("SOPRANO_QA_RAG_DATASET_ROOT", None)
        else:
            os.environ["SOPRANO_QA_RAG_DATASET_ROOT"] = str(
                pipeline_dataset_root.resolve()
            )
        importlib.invalidate_caches()
        specification = importlib.util.find_spec("soprano_qa.service")
        if specification is None or specification.origin is None:
            raise TargetImportError(f"Could not resolve soprano_qa.service in {root}")
        try:
            Path(specification.origin).resolve().relative_to(root)
        except ValueError as error:
            raise TargetImportError(
                "soprano_qa.service resolved outside --system-root: "
                f"{specification.origin}"
            ) from error
        service = importlib.import_module("soprano_qa.service")
        corpus_module = importlib.import_module("soprano_qa.corpus")
        llm_module = importlib.import_module("soprano_qa.llm")
        try:
            dense_module = importlib.import_module("soprano_qa.dense")
        except ModuleNotFoundError as error:
            if error.name != "soprano_qa.dense":
                raise
            dense_module = None
        foreign_modules = {
            name: str(getattr(module, "__file__", "<unknown>"))
            for name, module in sys.modules.items()
            if (name == "soprano_qa" or name.startswith("soprano_qa."))
            and isinstance(module, ModuleType)
            and not _module_is_within(module, root)
        }
        if foreign_modules:
            raise TargetImportError(
                "Target import mixed soprano_qa packages: "
                + json.dumps(foreign_modules, sort_keys=True)
            )
        ask = getattr(service, "ask", None)
        model_status = getattr(service, "model_status", None)
        corpus_stats = getattr(service, "corpus_stats", None)
        load_generation_model = getattr(llm_module, "load_llama", None)
        settings = getattr(service, "SETTINGS", None)
        if (
            not callable(ask)
            or not callable(model_status)
            or not callable(corpus_stats)
            or not callable(load_generation_model)
            or not isinstance(settings, dict)
        ):
            raise TargetImportError(
                "Target soprano_qa.service must expose callable ask, "
                "model_status, corpus_stats, and SETTINGS; soprano_qa.llm "
                "must expose load_llama"
            )
        original_settings = deepcopy(settings)
        validate_target_retrieval_requirements(
            original_settings,
            dense_module,
        )
        actual_dataset_root = Path(settings["dataset_root"]).resolve()
        pipeline_corpus_state = authenticate_pipeline_corpus(
            original_settings,
            corpus_module,
        )
        module_paths = {
            name: str(Path(module.__file__).resolve())
            for name, module in sorted(sys.modules.items())
            if (name == "soprano_qa" or name.startswith("soprano_qa."))
            and isinstance(module, ModuleType)
            and getattr(module, "__file__", None)
        }

        with tempfile.TemporaryDirectory(prefix="soprano-synth-retrieval-") as temp:
            runtime_root = Path(temp)
            patched: dict[str, str] = {}
            for key, filename in (
                ("corpus_path", "corpus.json"),
                ("stats_path", "corpus_stats.json"),
            ):
                source = Path(settings[key]).resolve()
                if not source.is_file():
                    raise EvaluationInputError(
                        f"Target {key} must be built before evaluation: {source}"
                    )
                destination = runtime_root / filename
                shutil.copy2(source, destination)
                patched[key] = str(destination)
            cache_value = settings.get("embedding_cache_path")
            if cache_value:
                cache_source = Path(cache_value).resolve()
                cache_destination = runtime_root / "embedding_cache.json"
                if cache_source.is_file():
                    shutil.copy2(cache_source, cache_destination)
                patched["embedding_cache_path"] = str(cache_destination)
            settings.update(patched)
            try:
                yield TargetRuntime(
                    ask=ask,
                    model_status=model_status,
                    corpus_stats=corpus_stats,
                    load_generation_model=load_generation_model,
                    pipeline_settings=original_settings,
                    pipeline_dataset_root=actual_dataset_root,
                    module_paths=module_paths,
                    pipeline_corpus_state=pipeline_corpus_state,
                )
            finally:
                try:
                    mutated = []
                    for key, state_key in (
                        ("corpus_path", "corpus"),
                        ("stats_path", "stats"),
                    ):
                        current = file_record(Path(settings[key]))
                        expected = pipeline_corpus_state[state_key]
                        if current["sha256"] != expected["sha256"]:
                            mutated.append(key)
                    if mutated:
                        raise IntegrityValidationError(
                            "The target mutated its authenticated temporary "
                            "corpus during evaluation: " + ", ".join(mutated)
                        )
                except IntegrityValidationError:
                    raise
                except Exception as error:
                    raise IntegrityValidationError(
                        "Could not reauthenticate the target's temporary "
                        "corpus after evaluation"
                    ) from error
    finally:
        for name, value in original_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        sys.path[:] = original_path
        sys.dont_write_bytecode = original_dont_write
        _remove_target_modules()
        sys.modules.update(original_modules)
        importlib.invalidate_caches()


def normalize_retrieval_diagnostics(result: Mapping[str, Any]) -> dict[str, Any]:
    raw = result.get("retrieval")
    if raw is None:
        return {
            "configured_mode": "lexical",
            "active_mode": "lexical",
            "dense_available": False,
            "embedding_model": None,
            "embedding_dimension": None,
            "fallback_reason": None,
            "last_dense_error": None,
            "last_search_mode": "lexical",
            "query_route": "lexical",
            "route_reason": "implicit_legacy_lexical",
            "dense_attempted": False,
            "dense_contributed": False,
            "fallback_used": False,
            "diagnostic_source": "implicit_legacy_lexical",
        }
    if not isinstance(raw, Mapping):
        raise EvaluationInputError("Pipeline retrieval diagnostics must be an object")
    active = raw.get("active_mode")
    configured = raw.get("configured_mode", active)
    if not isinstance(active, str) or not isinstance(configured, str):
        raise EvaluationInputError(
            "Pipeline retrieval diagnostics must name configured and active modes"
        )
    last_search = raw.get("last_search_mode")
    query_route = raw.get("query_route")
    if query_route is None:
        if last_search in {"hybrid", "hybrid_no_dense_match"}:
            query_route = "hybrid"
        elif last_search in {
            "lexical",
            "lexical_route",
            "lexical_fallback",
        }:
            query_route = "lexical"
        else:
            query_route = "none"
    if query_route not in {"hybrid", "lexical", "none"}:
        raise EvaluationInputError(
            "Pipeline retrieval diagnostics contain an invalid query route"
        )
    fallback_reason = raw.get("fallback_reason")
    last_dense_error = raw.get("last_dense_error")
    fallback_used = raw.get("fallback_used")
    if fallback_used is None:
        fallback_used = bool(
            fallback_reason
            or last_dense_error
            or last_search == "lexical_fallback"
        )
    dense_attempted = raw.get("dense_attempted")
    if dense_attempted is None:
        dense_attempted = last_search in {
            "hybrid",
            "hybrid_no_dense_match",
            "lexical_fallback",
        }
    dense_contributed = raw.get("dense_contributed")
    if dense_contributed is not None:
        dense_contributed = bool(dense_contributed)
    return {
        "configured_mode": configured,
        "active_mode": active,
        "dense_available": bool(raw.get("dense_available", active == "hybrid")),
        "embedding_model": raw.get("embedding_model"),
        "embedding_dimension": raw.get("embedding_dimension"),
        "fallback_reason": fallback_reason,
        "last_dense_error": last_dense_error,
        "last_search_mode": last_search,
        "query_route": query_route,
        "route_reason": raw.get("route_reason"),
        "dense_attempted": bool(dense_attempted),
        "dense_contributed": dense_contributed,
        "fallback_used": bool(fallback_used),
        "diagnostic_source": "pipeline",
    }


def require_retrieval_mode(
    diagnostics: Mapping[str, Any],
    required_mode: str | None,
) -> None:
    if required_mode is None:
        return
    active = diagnostics.get("active_mode")
    configured = diagnostics.get("configured_mode")
    fallback = diagnostics.get("fallback_reason")
    last_error = diagnostics.get("last_dense_error")
    last_search = diagnostics.get("last_search_mode")
    problems: list[str] = []
    if active != required_mode:
        problems.append(f"active_mode={active!r}")
    if configured != required_mode:
        problems.append(f"configured_mode={configured!r}")
    if fallback:
        problems.append(f"fallback_reason={fallback!r}")
    if last_error:
        problems.append(f"last_dense_error={last_error!r}")
    if diagnostics.get("fallback_used") is True:
        problems.append("fallback_used=true")
    if last_search == "lexical_fallback":
        problems.append("last_search_mode='lexical_fallback'")
    if required_mode == "hybrid" and diagnostics.get("dense_available") is not True:
        problems.append("dense_available is not true")
    if required_mode == "hybrid" and last_search not in {
        "hybrid",
        "hybrid_no_dense_match",
        "lexical_route",
    }:
        problems.append(f"last_search_mode={last_search!r}")
    if required_mode == "lexical" and last_search != "lexical":
        problems.append(f"last_search_mode={last_search!r}")
    if problems:
        raise RetrievalModeMismatchError(
            f"Required clean {required_mode} retrieval, but " + ", ".join(problems)
        )


def _rank_metrics(
    target_ids: Sequence[str],
    evidence_ids: Sequence[str],
    *,
    top_k: int,
) -> dict[str, Any]:
    ranks: dict[str, int] = {}
    for rank, evidence_id in enumerate(evidence_ids, start=1):
        ranks.setdefault(evidence_id, rank)
    found = [
        ranks[item]
        for item in target_ids
        if item in ranks and ranks[item] <= top_k
    ]
    first_rank = min(found) if found else None
    return {
        "first_rank": first_rank,
        "reciprocal_rank": (
            round(1.0 / first_rank, 10) if first_rank is not None else 0.0
        ),
        "hit_at_1": first_rank == 1,
        "hit_at_k": first_rank is not None and first_rank <= top_k,
        "all_retrieved": bool(target_ids)
        and all(item in ranks and ranks[item] <= top_k for item in target_ids),
        "retrieved_ids": [
            item for item in target_ids if item in ranks and ranks[item] <= top_k
        ],
    }


def diagnose_result(
    result: Mapping[str, Any],
    case_input: Mapping[str, Any],
    *,
    top_k: int,
    required_mode: str | None,
) -> dict[str, Any]:
    evidence = result.get("evidence") or []
    if not isinstance(evidence, list) or any(
        not isinstance(item, Mapping) for item in evidence
    ):
        raise EvaluationInputError("Pipeline evidence must be a list of objects")
    evidence_ids: list[str] = []
    evidence_retrieval_modes: list[str | None] = []
    for rank, item in enumerate(evidence, start=1):
        evidence_id = item.get("id")
        if not isinstance(evidence_id, str) or not evidence_id:
            raise EvaluationInputError(f"Evidence rank {rank} has no stable ID")
        evidence_ids.append(evidence_id)
        retrieval_mode = item.get("retrieval_mode")
        if retrieval_mode is not None and not isinstance(retrieval_mode, str):
            raise EvaluationInputError(
                f"Evidence rank {rank} has an invalid retrieval mode"
            )
        evidence_retrieval_modes.append(retrieval_mode)
    diagnostics = normalize_retrieval_diagnostics(result)
    require_retrieval_mode(diagnostics, required_mode)
    expected = _rank_metrics(
        case_input["expected_knowledge_unit_ids"],
        evidence_ids,
        top_k=top_k,
    )
    applicable = _rank_metrics(
        case_input["applicable_knowledge_unit_ids"],
        evidence_ids,
        top_k=top_k,
    )
    source_id = case_input["source_id"]
    target_source_grounded = any(
        (
            item.get("kind") == "expert"
            or item.get("evidence_type") == "expert_annotation"
        )
        and source_id in (item.get("source_ids") or [])
        and item.get("in_requested_scope") is True
        for item in evidence
    )
    return {
        "ordered_evidence_ids": evidence_ids,
        "ordered_evidence_retrieval_modes": evidence_retrieval_modes,
        "expected": expected,
        "applicable": applicable,
        "target_source_grounded": target_source_grounded,
        "retrieval_diagnostics": diagnostics,
    }


def _rate(count: int, total: int) -> float | None:
    return round(count / total, 6) if total else None


def _all_cases(
    snapshot: Mapping[str, Any],
) -> Iterator[tuple[Mapping[str, Any], dict[str, Any]]]:
    for result in snapshot.get("results") or []:
        for case in result.get("inference_runs") or []:
            yield result, case


def _rollup(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    completed = [case for case in cases if case.get("status") == "completed"]
    errors = [case for case in cases if case.get("status") == "error"]
    diagnostics = [
        case["retrieval_probe"]["diagnostics"]
        for case in completed
    ]
    retrieval_diagnostics = [
        item["retrieval_diagnostics"] for item in diagnostics
    ]
    generated = [case["generated_answer"] for case in completed]
    result: dict[str, Any] = {
        "cases": len(cases),
        "completed_cases": len(completed),
        "error_cases": len(errors),
        "generation_modes": dict(
            sorted(Counter(
                str(item.get("generation_mode") or "unknown")
                for item in generated
            ).items())
        ),
        "answer_bases": dict(
            sorted(Counter(
                str(item.get("answer_basis") or "unknown")
                for item in generated
            ).items())
        ),
        "generation_fallback_reasons": dict(
            sorted(Counter(
                str(item.get("generation_fallback_reason") or "none")
                for item in generated
            ).items())
        ),
        "retrieval_modes": dict(
            sorted(Counter(
                item.get("last_search_mode") or "unknown"
                for item in retrieval_diagnostics
            ).items())
        ),
        "query_routes": dict(
            sorted(Counter(
                item.get("query_route") or "unknown"
                for item in retrieval_diagnostics
            ).items())
        ),
        "dense_attempted_cases": sum(
            item.get("dense_attempted") is True
            for item in retrieval_diagnostics
        ),
        "dense_contributed_cases": sum(
            item.get("dense_contributed") is True
            for item in retrieval_diagnostics
        ),
        "fallback_used_cases": sum(
            item.get("fallback_used") is True
            for item in retrieval_diagnostics
        ),
    }
    for lane in (
        "expected",
        "applicable",
        "retrieval_eligible",
        "applicable_retrieval_eligible",
    ):
        values = [
            item[lane]
            for item in diagnostics
            if item["target_availability"][lane]
        ]
        hit_1 = sum(value["hit_at_1"] for value in values)
        hit_k = sum(value["hit_at_k"] for value in values)
        all_retrieved = sum(value["all_retrieved"] for value in values)
        result[lane] = {
            "target_available_cases": len(values),
            "target_unavailable_cases": len(completed) - len(values),
            "hit_at_1_cases": hit_1,
            "hit_at_1_rate": _rate(hit_1, len(values)),
            "hit_at_k_cases": hit_k,
            "hit_at_k_rate": _rate(hit_k, len(values)),
            "all_retrieved_cases": all_retrieved,
            "all_retrieved_rate": _rate(all_retrieved, len(values)),
            "mean_reciprocal_rank": (
                round(
                    sum(value["reciprocal_rank"] for value in values)
                    / len(values),
                    6,
                )
                if values
                else None
            ),
        }
    grounded = sum(item["target_source_grounded"] for item in diagnostics)
    result["target_source_grounded_cases"] = grounded
    result["target_source_grounded_rate"] = _rate(grounded, len(completed))
    return result


def _formulation_consistency(
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for case in cases:
        case_input = case["inference_input"]
        formulation_id = (
            f"{case_input['source_id']}__"
            f"{_case_suffix(case_input['measure_range'])}"
        )
        grouped[formulation_id].append(case)
    completed_groups = 0
    expected_consistent = 0
    applicable_consistent = 0
    inconsistent: list[dict[str, Any]] = []
    for formulation_id in sorted(grouped):
        group = sorted(
            grouped[formulation_id],
            key=lambda case: case["inference_input"]["variant_index"],
        )
        indices = [case["inference_input"]["variant_index"] for case in group]
        if indices != [1, 2, 3]:
            raise EvaluationInputError(
                f"{formulation_id}: expected variant indices [1, 2, 3], "
                f"found {indices}"
            )
        if any(case.get("status") != "completed" for case in group):
            continue
        completed_groups += 1
        expected_hits = [
            case["retrieval_probe"]["diagnostics"]["expected"]["hit_at_k"]
            for case in group
        ]
        applicable_hits = [
            case["retrieval_probe"]["diagnostics"]["applicable"]["hit_at_k"]
            for case in group
        ]
        all_expected = all(expected_hits)
        all_applicable = all(applicable_hits)
        expected_consistent += all_expected
        applicable_consistent += all_applicable
        if not all_expected or not all_applicable:
            inconsistent.append(
                {
                    "formulation_id": formulation_id,
                    "variant_ids": [
                        case["inference_input"]["synthesized_variant_id"]
                        for case in group
                    ],
                    "expected_hit_at_k": expected_hits,
                    "applicable_hit_at_k": applicable_hits,
                }
            )
    return {
        "groups": len(grouped),
        "completed_groups": completed_groups,
        "all_variants_hit_at_k_groups": expected_consistent,
        "all_variants_hit_at_k_rate": _rate(expected_consistent, completed_groups),
        "all_variants_applicable_hit_at_k_groups": applicable_consistent,
        "all_variants_applicable_hit_at_k_rate": _rate(
            applicable_consistent,
            completed_groups,
        ),
        "inconsistent_groups": inconsistent,
    }


def refresh_summary(snapshot: dict[str, Any]) -> None:
    cases = [case for _, case in _all_cases(snapshot)]
    integrity = snapshot["run"].setdefault(
        "integrity",
        {
            "status": "pending",
            "validated_at": None,
            "error": None,
        },
    )
    integrity_status = integrity.get("status", "pending")
    if integrity_status not in {"pending", "validated", "invalid"}:
        raise EvaluationInputError(
            f"Unsupported artifact integrity status: {integrity_status}"
        )
    statuses: dict[str, int] = {}
    for case in cases:
        status = case["status"]
        statuses[status] = statuses.get(status, 0) + 1
    by_piece = {
        piece_id: _rollup(
            [
                case
                for case in cases
                if case["inference_input"]["piece_id"] == piece_id
            ]
        )
        for piece_id in TARGET_PIECES
        if any(
            case["inference_input"]["piece_id"] == piece_id
            for case in cases
        )
    }
    by_variant_index = {
        str(index): _rollup(
            [
                case
                for case in cases
                if case["inference_input"]["variant_index"] == index
            ]
        )
        for index in range(1, VARIANTS_PER_QUESTION + 1)
    }
    all_cases_completed = statuses.get("completed", 0) == len(cases)
    summary_status = (
        "invalid"
        if integrity_status == "invalid"
        else (
            "complete"
            if all_cases_completed and integrity_status == "validated"
            else "partial"
        )
    )
    snapshot["summary"] = {
        "status": summary_status,
        "integrity_status": integrity_status,
        "question_count": len(snapshot["results"]),
        "variant_formulation_count": sum(
            len(result["synthesized_question_variants"])
            for result in snapshot["results"]
        ),
        "inference_run_count": len(cases),
        "inference_runs_completed": statuses.get("completed", 0),
        "case_statuses": dict(sorted(statuses.items())),
        "overall": _rollup(cases),
        "by_piece": by_piece,
        "by_variant_index": by_variant_index,
        "formulation_consistency": _formulation_consistency(cases),
    }
    snapshot["run"]["status"] = snapshot["summary"]["status"]
    snapshot["run"]["updated_at"] = utc_now()


def new_snapshot(
    *,
    benchmark: Benchmark,
    benchmark_root: Path,
    system_state: Mapping[str, Any],
    target: TargetRuntime,
    model_contract: Mapping[str, Any],
    top_k: int,
    input_files: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    input_payload = {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "benchmark_root": str(benchmark_root.resolve()),
        "input_files": list(input_files),
        "benchmark_results": benchmark.results,
        "exclusions": benchmark.exclusions,
    }
    input_fingerprint = sha256_value(input_payload)
    runtime_payload = {
        "input_fingerprint": input_fingerprint,
        "system_fingerprint": system_state["fingerprint"],
        "pipeline_settings": target.pipeline_settings,
        "pipeline_dataset_root": str(target.pipeline_dataset_root),
        "pipeline_corpus_state": target.pipeline_corpus_state,
        "model_contract": model_contract,
        "top_k": top_k,
        "required_retrieval_mode": "hybrid",
        "generate": True,
        "allow_internal_knowledge": False,
    }
    run_fingerprint = sha256_value(runtime_payload)
    now = utc_now()
    snapshot = {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "protocol": {
            "name": "five-piece synthesized-question hybrid RAG+LLM evaluation",
            "target_pieces": list(TARGET_PIECES),
            "requires_all_five_variant_shards": True,
            "generate": True,
            "allow_internal_knowledge": False,
            "required_retrieval_mode": "hybrid",
            "issues_automatic_quality_verdicts": False,
        },
        "run": {
            "created_at": now,
            "updated_at": now,
            "status": "partial",
            "benchmark_dataset_root": str(benchmark_root.resolve()),
            "pipeline_dataset_root": str(target.pipeline_dataset_root),
            "system_root": system_state["system_root"],
            "target_pieces": list(TARGET_PIECES),
            "top_k": top_k,
            "generate": True,
            "allow_internal_knowledge": False,
            "required_retrieval_mode": "hybrid",
            "input_fingerprint": input_fingerprint,
            "system_fingerprint": system_state["fingerprint"],
            "run_fingerprint": run_fingerprint,
            "input_files": list(input_files),
            "system_state": deepcopy(dict(system_state)),
            "model_contract": deepcopy(dict(model_contract)),
            "runtime_fingerprint": model_contract["runtime"]["fingerprint"],
            "pipeline_settings": deepcopy(target.pipeline_settings),
            "pipeline_corpus_state": deepcopy(target.pipeline_corpus_state),
            "target_module_paths": deepcopy(target.module_paths),
            "excluded_cases": deepcopy(benchmark.exclusions),
            "pipeline_error": None,
            "integrity": {
                "status": "pending",
                "validated_at": None,
                "error": None,
            },
        },
        "results": deepcopy(benchmark.results),
        "summary": {},
    }
    refresh_summary(snapshot)
    return snapshot


def _resume_definitions(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_results = value.get("results")
    if not isinstance(raw_results, list):
        raise ResumeMismatchError("Output results must be a list")
    output = []
    seen_source_ids: set[str] = set()
    seen_case_ids: set[str] = set()
    for result_index, result in enumerate(raw_results):
        if not isinstance(result, Mapping):
            raise ResumeMismatchError(
                f"Output results[{result_index}] must be an object"
            )
        source_id = result.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise ResumeMismatchError(
                f"Output results[{result_index}] has no source_id"
            )
        if source_id in seen_source_ids:
            raise ResumeMismatchError(
                f"Output contains duplicate source_id {source_id}"
            )
        seen_source_ids.add(source_id)
        raw_cases = result.get("inference_runs")
        if not isinstance(raw_cases, list) or not raw_cases:
            raise ResumeMismatchError(
                f"Output {source_id}.inference_runs must be a non-empty list"
            )
        definition = {
            key: deepcopy(item)
            for key, item in result.items()
            if key != "inference_runs"
        }
        definition["inference_runs"] = []
        for case_index, case in enumerate(raw_cases):
            if not isinstance(case, Mapping):
                raise ResumeMismatchError(
                    f"Output {source_id}.inference_runs[{case_index}] must "
                    "be an object"
                )
            case_id = case.get("case_id")
            if not isinstance(case_id, str) or not case_id:
                raise ResumeMismatchError(
                    f"Output {source_id}.inference_runs[{case_index}] has "
                    "no case_id"
                )
            if case_id in seen_case_ids:
                raise ResumeMismatchError(
                    f"Output contains duplicate case_id {case_id}"
                )
            seen_case_ids.add(case_id)
            case_input = case.get("inference_input")
            if not isinstance(case_input, Mapping):
                raise ResumeMismatchError(
                    f"Output {case_id}.inference_input must be an object"
                )
            definition["inference_runs"].append(
                {
                    "case_id": case_id,
                    "inference_input": deepcopy(dict(case_input)),
                }
            )
        output.append(definition)
    return output


def _validate_completed_resume_outputs(
    case: Mapping[str, Any],
    *,
    expected_model_path: str,
) -> None:
    case_id = str(case["case_id"])
    case_input = case["inference_input"]
    retrieval_probe = case.get("retrieval_probe")
    generated = case.get("generated_answer")
    if not isinstance(retrieval_probe, Mapping):
        raise ResumeMismatchError(
            f"Output {case_id}: completed case has no retrieval_probe"
        )
    if not isinstance(generated, Mapping):
        raise ResumeMismatchError(
            f"Output {case_id}: completed case has no generated_answer"
        )
    expected_piece = case_input["piece_id"]
    expected_range = case_input["measure_range"]
    expected_scope = "range" if expected_range is not None else "whole_piece"
    for label, output in (
        ("retrieval_probe", retrieval_probe),
        ("generated_answer", generated),
    ):
        if output.get("piece_id") != expected_piece:
            raise ResumeMismatchError(
                f"Output {case_id}: {label} piece_id does not match input"
            )
        if output.get("measure_range") != expected_range:
            raise ResumeMismatchError(
                f"Output {case_id}: {label} measure_range does not match input"
            )
        if output.get("scope") != expected_scope:
            raise ResumeMismatchError(
                f"Output {case_id}: {label} scope does not match input"
            )
        evidence = output.get("evidence")
        if not isinstance(evidence, list) or any(
            not isinstance(item, Mapping) for item in evidence
        ):
            raise ResumeMismatchError(
                f"Output {case_id}: {label} evidence must be a list of objects"
            )
        validation = output.get("retrieval_validation")
        if not isinstance(validation, Mapping) or (
            validation.get("required_mode") != "hybrid"
            or validation.get("passed") is not True
        ):
            raise ResumeMismatchError(
                f"Output {case_id}: {label} retrieval validation is invalid"
            )
    for field in ("answer", "generation_mode", "answer_basis"):
        if not isinstance(generated.get(field), str):
            raise ResumeMismatchError(
                f"Output {case_id}: generated_answer.{field} must be a string"
            )
    reason = generated.get("generation_fallback_reason")
    if reason is not None and not isinstance(reason, str):
        raise ResumeMismatchError(
            f"Output {case_id}: generation fallback reason is invalid"
        )
    if reason in OPERATIONAL_GENERATION_FAILURE_REASONS:
        raise ResumeMismatchError(
            f"Output {case_id}: operational generation failure was marked "
            "completed"
        )
    if generated.get("answer_basis") == "internal_knowledge":
        raise ResumeMismatchError(
            f"Output {case_id}: internal-knowledge answer is not allowed"
        )
    try:
        _validate_response_model(
            generated,
            expected_model_path=expected_model_path,
        )
    except ModelPreflightError as error:
        raise ResumeMismatchError(
            f"Output {case_id}: invalid generated model status: {error}"
        ) from error
    probe_diagnostics = retrieval_probe.get("diagnostics")
    if not isinstance(probe_diagnostics, Mapping):
        raise ResumeMismatchError(
            f"Output {case_id}: retrieval diagnostics are missing"
        )
    pipeline_diagnostics = probe_diagnostics.get("retrieval_diagnostics")
    if not isinstance(pipeline_diagnostics, Mapping):
        raise ResumeMismatchError(
            f"Output {case_id}: pipeline retrieval diagnostics are missing"
        )
    try:
        require_retrieval_mode(pipeline_diagnostics, "hybrid")
    except RetrievalModeMismatchError as error:
        raise ResumeMismatchError(
            f"Output {case_id}: stored retrieval mode is invalid: {error}"
        ) from error
    if generated.get("diagnostics") != probe_diagnostics:
        raise ResumeMismatchError(
            f"Output {case_id}: generated/retrieval diagnostics mismatch"
        )
    if generated.get("retrieval") != retrieval_probe.get("retrieval"):
        raise ResumeMismatchError(
            f"Output {case_id}: generated/retrieval runtime metadata mismatch"
        )
    if generated.get("evidence") != retrieval_probe.get("evidence"):
        raise ResumeMismatchError(
            f"Output {case_id}: generated/retrieval evidence mismatch"
        )


def validate_resume_case_states(
    snapshot: Mapping[str, Any],
    *,
    expected_model_path: str,
) -> None:
    for _, case in _all_cases(snapshot):
        case_id = str(case.get("case_id") or "<unknown>")
        status = case.get("status")
        if status not in {"pending", "running", "error", "completed"}:
            raise ResumeMismatchError(
                f"Output {case_id}: invalid case status {status!r}"
            )
        attempt_count = case.get("attempt_count")
        if (
            isinstance(attempt_count, bool)
            or not isinstance(attempt_count, int)
            or attempt_count < 0
        ):
            raise ResumeMismatchError(
                f"Output {case_id}: attempt_count must be a non-negative integer"
            )
        last_attempt = case.get("last_attempt_at")
        completed_at = case.get("completed_at")
        error = case.get("error")
        retrieval_probe = case.get("retrieval_probe")
        generated = case.get("generated_answer")
        if status == "pending":
            if (
                attempt_count != 0
                or last_attempt is not None
                or completed_at is not None
                or error is not None
                or retrieval_probe is not None
                or generated is not None
            ):
                raise ResumeMismatchError(
                    f"Output {case_id}: pending case state is inconsistent"
                )
            continue
        if attempt_count < 1 or not isinstance(last_attempt, str) or not last_attempt:
            raise ResumeMismatchError(
                f"Output {case_id}: attempted case has invalid attempt metadata"
            )
        if status == "completed":
            if (
                not isinstance(completed_at, str)
                or not completed_at
                or error is not None
            ):
                raise ResumeMismatchError(
                    f"Output {case_id}: completed case state is inconsistent"
                )
            _validate_completed_resume_outputs(
                case,
                expected_model_path=expected_model_path,
            )
            continue
        if completed_at is not None or retrieval_probe is not None or generated is not None:
            raise ResumeMismatchError(
                f"Output {case_id}: incomplete case contains completed outputs"
            )
        if status == "running" and error is not None:
            raise ResumeMismatchError(
                f"Output {case_id}: running case must not contain an error"
            )
        if status == "error" and (
            not isinstance(error, Mapping)
            or not isinstance(error.get("type"), str)
            or not error["type"]
            or not isinstance(error.get("message"), str)
        ):
            raise ResumeMismatchError(
                f"Output {case_id}: error case has invalid error metadata"
            )


def validate_resume_snapshot(
    snapshot: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    if snapshot.get("artifact_type") != ARTIFACT_TYPE:
        raise ResumeMismatchError(
            "Output is not a synthesized hybrid RAG+LLM artifact"
        )
    if snapshot.get("schema_version") != SCHEMA_VERSION:
        raise ResumeMismatchError("Output schema version is incompatible")
    current_run = snapshot.get("run")
    expected_run = expected.get("run")
    if not isinstance(current_run, Mapping) or not isinstance(
        expected_run,
        Mapping,
    ):
        raise ResumeMismatchError("Output run metadata must be an object")
    integrity = current_run.get("integrity")
    if not isinstance(integrity, Mapping) or integrity.get("status") not in {
        "pending",
        "validated",
        "invalid",
    }:
        raise ResumeMismatchError("Output integrity metadata is invalid")
    if integrity.get("status") == "invalid":
        raise ResumeMismatchError(
            "Output failed integrity validation; choose a new output path"
        )
    for field in (
        "input_fingerprint",
        "system_fingerprint",
        "run_fingerprint",
    ):
        if current_run.get(field) != expected_run.get(field):
            raise ResumeMismatchError(f"Output {field} does not match this run")
    for field in (
        "benchmark_dataset_root",
        "pipeline_dataset_root",
        "system_root",
        "target_pieces",
        "top_k",
        "generate",
        "allow_internal_knowledge",
        "required_retrieval_mode",
        "input_files",
        "system_state",
        "model_contract",
        "runtime_fingerprint",
        "pipeline_settings",
        "pipeline_corpus_state",
        "target_module_paths",
        "excluded_cases",
    ):
        if current_run.get(field) != expected_run.get(field):
            raise ResumeMismatchError(
                f"Output run.{field} does not match this run"
            )
    current_inputs = _resume_definitions(snapshot)
    expected_inputs = _resume_definitions(expected)
    if current_inputs != expected_inputs:
        raise ResumeMismatchError("Output case definitions do not match benchmark")
    expected_model = expected_run["model_contract"]["generation_model"]
    validate_resume_case_states(
        snapshot,
        expected_model_path=str(expected_model["path"]),
    )


def _validate_response_model(
    response: Mapping[str, Any],
    *,
    expected_model_path: str,
) -> None:
    status = response.get("model")
    if not isinstance(status, Mapping):
        raise ModelPreflightError("Pipeline response omitted model status")
    if status.get("checkpoint_exists") is not True:
        raise ModelPreflightError(
            "Generation model checkpoint became unavailable during evaluation"
        )
    if status.get("llama_cpp_available") is not True:
        raise ModelPreflightError(
            "llama-cpp-python generation backend became unavailable"
        )
    if Path(str(status.get("path") or "")).resolve() != Path(
        expected_model_path
    ).resolve():
        raise ModelPreflightError(
            "Pipeline response used a different generation model path"
        )


def _pipeline_outputs(
    response: Mapping[str, Any],
    *,
    case_input: Mapping[str, Any],
    top_k: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_range = case_input["measure_range"]
    expected_scope = "range" if expected_range is not None else "whole_piece"
    if response.get("piece_id") != case_input["piece_id"]:
        raise EvaluationInputError(
            "Pipeline response piece_id does not match inference input"
        )
    if response.get("measure_range") != expected_range:
        raise EvaluationInputError(
            "Pipeline response measure_range does not match inference input"
        )
    if response.get("scope") != expected_scope:
        raise EvaluationInputError(
            "Pipeline response scope does not match inference input"
        )
    for field in ("answer", "generation_mode", "answer_basis"):
        if not isinstance(response.get(field), str):
            raise EvaluationInputError(
                f"Pipeline response {field} must be a string"
            )
    reason = response.get("generation_fallback_reason")
    if reason is not None and not isinstance(reason, str):
        raise EvaluationInputError(
            "Pipeline generation_fallback_reason must be a string or null"
        )
    if reason in OPERATIONAL_GENERATION_FAILURE_REASONS:
        raise PipelineExecutionError(
            f"Operational generation failure: {reason}"
        )
    linked = list(case_input["expected_knowledge_unit_ids"])
    applicable_linked = list(
        case_input["range_applicable_knowledge_unit_ids"]
    )
    eligible = list(
        case_input["expected_retrieval_eligible_knowledge_unit_ids"]
    )
    applicable_eligible = [
        unit_id
        for unit_id in applicable_linked
        if unit_id in eligible
    ]
    diagnostics = diagnose_result(
        response,
        {
            "source_id": case_input["source_id"],
            "expected_knowledge_unit_ids": linked,
            "applicable_knowledge_unit_ids": applicable_linked,
        },
        top_k=top_k,
        required_mode="hybrid",
    )
    ordered_evidence_ids = diagnostics["ordered_evidence_ids"]
    diagnostics["retrieval_eligible"] = _rank_metrics(
        eligible,
        ordered_evidence_ids,
        top_k=top_k,
    )
    diagnostics["applicable_retrieval_eligible"] = _rank_metrics(
        applicable_eligible,
        ordered_evidence_ids,
        top_k=top_k,
    )
    diagnostics["target_availability"] = {
        "expected": bool(linked),
        "applicable": bool(applicable_linked),
        "retrieval_eligible": bool(eligible),
        "applicable_retrieval_eligible": bool(applicable_eligible),
    }
    evidence = deepcopy(response.get("evidence") or [])
    expert_ids = [
        str(item.get("id"))
        for item in evidence
        if item.get("kind") == "expert"
        or item.get("evidence_type") == "expert_annotation"
    ]
    linked_retrieved = [item for item in linked if item in expert_ids]
    eligible_retrieved = [item for item in eligible if item in expert_ids]
    retrieval_probe = {
        "piece_id": response.get("piece_id"),
        "scope": response.get("scope"),
        "measure_range": deepcopy(response.get("measure_range")),
        "pipeline": response.get("pipeline"),
        "answer_basis": "same_call_hybrid_retrieval",
        "retrieval": deepcopy(response.get("retrieval")),
        "evidence": evidence,
        "evidence_notices": deepcopy(response.get("evidence_notices") or []),
        "diagnostics": diagnostics,
        "retrieval_validation": {
            "required_mode": "hybrid",
            "passed": True,
        },
    }
    generated_answer = deepcopy(dict(response))
    generated_answer.update(
        {
            "diagnostics": diagnostics,
            "expert_evidence_ids": expert_ids,
            "linked_knowledge_unit_ids_retrieved": linked_retrieved,
            "retrieval_eligible_knowledge_unit_ids_retrieved": (
                eligible_retrieved
            ),
            "expected_knowledge_unit_ids_retrieved": linked_retrieved,
            "expected_expert_evidence_retrieved": bool(linked_retrieved),
            "all_linked_expert_units_retrieved": bool(linked)
            and all(item in expert_ids for item in linked),
            "all_retrieval_eligible_expert_units_retrieved": (
                bool(eligible)
                and all(item in expert_ids for item in eligible)
            )
            if eligible
            else None,
            "retrieval_eligible_target_available": bool(eligible),
            "all_expected_expert_units_retrieved": bool(linked)
            and all(item in expert_ids for item in linked),
            "rag_llm_succeeded": (
                response.get("generation_mode") == "llm"
                and response.get("answer_basis") == "retrieved_evidence"
                and bool(evidence)
            ),
            "retrieval_validation": {
                "required_mode": "hybrid",
                "passed": True,
            },
        }
    )
    return retrieval_probe, generated_answer


def run_cases(
    snapshot: dict[str, Any],
    *,
    ask: Callable[..., Mapping[str, Any]],
    top_k: int,
    limit: int | None,
    checkpoint: Callable[[], None],
) -> int:
    selected = [
        case
        for _, case in _all_cases(snapshot)
        if case.get("status") != "completed"
    ]
    if limit is not None:
        selected = selected[:limit]
    expected_model_path = snapshot["run"]["model_contract"][
        "generation_model"
    ]["path"]
    for case in selected:
        if case["status"] == "completed":
            continue
        case["attempt_count"] += 1
        case["status"] = "running"
        case["last_attempt_at"] = utc_now()
        case["completed_at"] = None
        case["error"] = None
        case["retrieval_probe"] = None
        case["generated_answer"] = None
        checkpoint()
        case_input = case["inference_input"]
        measure_range = case_input["measure_range"]
        try:
            response = ask(
                piece_id=case_input["piece_id"],
                question=case_input["question"],
                measure_range=(
                    tuple(measure_range) if measure_range is not None else None
                ),
                generate=True,
                allow_internal_knowledge=False,
                top_k=top_k,
            )
            if not isinstance(response, Mapping):
                raise EvaluationInputError("Pipeline ask() returned a non-object")
            _validate_response_model(
                response,
                expected_model_path=expected_model_path,
            )
            retrieval_probe, generated_answer = _pipeline_outputs(
                response,
                case_input=case_input,
                top_k=top_k,
            )
            case["retrieval_probe"] = retrieval_probe
            case["generated_answer"] = generated_answer
            snapshot["run"]["pipeline_error"] = None
            case["status"] = "completed"
            case["completed_at"] = utc_now()
        except KeyboardInterrupt:
            checkpoint()
            raise
        except SynthesizedEvaluationError as error:
            case["status"] = "error"
            case["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
            snapshot["run"]["pipeline_error"] = deepcopy(case["error"])
            checkpoint()
            raise
        except Exception as error:
            case["status"] = "error"
            case["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
            snapshot["run"]["pipeline_error"] = deepcopy(case["error"])
            checkpoint()
            raise PipelineExecutionError(
                f"{case['case_id']}: pipeline execution failed: {error}"
            ) from error
        checkpoint()
    return len(selected)


def run_evaluation(
    *,
    system_root: Path,
    dataset_root: Path,
    output: Path,
    top_k: int,
    limit: int | None = None,
    pipeline_dataset_root: Path | None = None,
) -> dict[str, Any]:
    benchmark = load_benchmark(dataset_root)
    input_files = input_file_records(
        [*benchmark.input_paths, Path(__file__).resolve()]
    )
    system_before = collect_system_state(system_root)

    snapshot: dict[str, Any] | None = None
    with exclusive_output_lock(output):
        try:
            with target_runtime(
                system_root,
                pipeline_dataset_root=pipeline_dataset_root,
            ) as target:
                raw_model_status = target.model_status()
                if not isinstance(raw_model_status, Mapping):
                    raise ModelPreflightError(
                        "Target model_status() returned a non-object"
                    )
                runtime_state = collect_runtime_state()
                model_contract = validate_hybrid_generation_preflight(
                    target.pipeline_settings,
                    system_state=system_before,
                    model_status=raw_model_status,
                    runtime_state=runtime_state,
                )
                model_contract["initialization"] = initialize_target_models(
                    target
                )
                expected = new_snapshot(
                    benchmark=benchmark,
                    benchmark_root=dataset_root,
                    system_state=system_before,
                    target=target,
                    model_contract=model_contract,
                    top_k=top_k,
                    input_files=input_files,
                )
                if output.exists():
                    snapshot = load_json(output)
                    if not isinstance(snapshot, dict):
                        raise ResumeMismatchError("Output root must be an object")
                    validate_resume_snapshot(snapshot, expected)
                else:
                    snapshot = expected
                    atomic_write_json(output, snapshot)

                snapshot["run"]["integrity"] = {
                    "status": "pending",
                    "validated_at": None,
                    "error": None,
                }

                def checkpoint() -> None:
                    assert snapshot is not None
                    refresh_summary(snapshot)
                    atomic_write_json(output, snapshot)

                run_cases(
                    snapshot,
                    ask=target.ask,
                    top_k=top_k,
                    limit=limit,
                    checkpoint=checkpoint,
                )
                checkpoint()

            try:
                system_after = collect_system_state(system_root)
            except Exception as error:
                raise IntegrityValidationError(
                    "Could not reauthenticate target system files after "
                    "evaluation"
                ) from error
            if system_after != system_before:
                raise IntegrityValidationError(
                    "Target system files changed during evaluation; results "
                    "are not reproducible"
                )
            try:
                pipeline_inputs_after = refresh_pipeline_input_records(
                    target.pipeline_corpus_state
                )
            except Exception as error:
                raise IntegrityValidationError(
                    "Could not reauthenticate pipeline corpus inputs after "
                    "evaluation"
                ) from error
            if (
                pipeline_inputs_after
                != target.pipeline_corpus_state["input_files"]
            ):
                raise IntegrityValidationError(
                    "Pipeline corpus inputs changed during evaluation; "
                    "results are not reproducible"
                )
        except IntegrityValidationError as error:
            if snapshot is not None:
                snapshot["run"]["integrity"] = {
                    "status": "invalid",
                    "validated_at": None,
                    "error": {
                        "at": utc_now(),
                        "type": type(error).__name__,
                        "message": str(error),
                    },
                }
                refresh_summary(snapshot)
                atomic_write_json(output, snapshot)
            raise

        assert snapshot is not None
        snapshot["run"]["integrity"] = {
            "status": "validated",
            "validated_at": utc_now(),
            "error": None,
        }
        refresh_summary(snapshot)
        atomic_write_json(output, snapshot)
    return snapshot


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--system-root",
        type=Path,
        default=PROJECT_ROOT,
        help="RAG-system worktree whose soprano_qa package will be evaluated.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Dataset root containing development variant shards.",
    )
    parser.add_argument(
        "--pipeline-dataset-root",
        type=Path,
        help=(
            "Optional corpus dataset override for the target service. By default "
            "the target worktree's configured dataset is preserved."
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument(
        "--limit",
        type=int,
        help="Run at most N incomplete cases, preserving a resumable artifact.",
    )
    arguments = parser.parse_args(argv)
    if arguments.top_k < 1:
        parser.error("--top-k must be positive")
    if arguments.limit is not None and arguments.limit < 1:
        parser.error("--limit must be positive")
    return arguments


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    try:
        snapshot = run_evaluation(
            system_root=arguments.system_root,
            dataset_root=arguments.dataset_root,
            pipeline_dataset_root=arguments.pipeline_dataset_root,
            output=arguments.output.resolve(),
            top_k=arguments.top_k,
            limit=arguments.limit,
        )
    except SynthesizedEvaluationError as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(snapshot["summary"], ensure_ascii=False, indent=2))
    return 0 if snapshot["summary"]["overall"]["error_cases"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
