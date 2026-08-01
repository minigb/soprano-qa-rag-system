#!/usr/bin/env python3
"""Run the three-piece expert-question RAG reliability evaluation.

The evaluator deliberately separates retrieval, answer generation, and
semantic judging.  Every successful case or judge frame is checkpointed with
an atomic replace so an interrupted run can be resumed without repeating
completed model calls.
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
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import threading
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence
import unicodedata


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation" / "three_piece_results.json"
DEFAULT_JUDGE_CALIBRATION = (
    PROJECT_ROOT / "evaluation" / "judge_calibration_results.json"
)
TARGET_PIECES = (
    "die-forelle",
    "in-flowery-clouds",
    "la-capinera",
)
EXPECTED_QUESTION_COUNT = 40
EXPECTED_CASE_COUNT = 51
SCHEMA_VERSION = "8.0"
JUDGE_PROTOCOL_VERSION = (
    "dual-frame-v13-authenticated-disclosure-retrieved-expert-"
    "factuality-contextual-required-s-substitution-nli-literal-anchor-"
    "hybrid-range-guard"
)
ABSOLUTE_MEASURE_RE = re.compile(
    r"(?:\d+\s*(?:[-–—~]\s*\d+\s*)?(?:번째\s*)?마디|마디)"
)
KNOWLEDGE_UNIT_CITATION_RE = re.compile(
    r"\[(?:[a-z0-9]+(?:-[a-z0-9]+)*-ku-\d{3}|E\d+|"
    r"sqa-\d+|webchunk-[A-Za-z0-9*-]+)\]",
    flags=re.IGNORECASE,
)
EVIDENCE_FOOTER_LABEL_RE = re.compile(r"제공된\s+검색\s+근거\s*:")
OMISSION_LANGUAGE_RE = re.compile(
    r"(?:\bomit(?:s|ted|ting)?\b|\bomission\b|누락|생략|언급하지|"
    r"포함하지|빠져|빠졌)",
    flags=re.IGNORECASE,
)
NEGATED_OMISSION_LANGUAGE_RE = re.compile(
    r"(?:누락|생략)(?:이|을|하지)?\s*(?:없(?:이|다)?|않(?:고|다)?)"
    r"|\bwithout\s+(?:an?\s+)?omission\b"
    r"|\bnot\s+omit(?:s|ted|ting)?\b",
    flags=re.IGNORECASE,
)
LATIN_LETTER_CLASS = "A-Za-zÀ-ÖØ-öø-ÿ"
PRONUNCIATION_ANCHOR_RE = re.compile(
    rf"(?<![{LATIN_LETTER_CLASS}'])"
    rf"([{LATIN_LETTER_CLASS}]+(?:['’][{LATIN_LETTER_CLASS}]+)*)"
    r"\s*발음",
    flags=re.IGNORECASE,
)
NUMERIC_FACT_ANCHOR_RE = re.compile(
    r"(?<!\d)(\d+(?:[.,]\d+)?)\s*"
    r"(분|초|시간|회|번|일|주|개월|년|헤르츠|hz|bpm|배)",
    flags=re.IGNORECASE,
)
CALIBRATION_SCHEMA_VERSION = "8.0"
CALIBRATION_CONTRACT_VERSION = (
    "reference-risk-v13-authenticated-disclosure-retrieved-expert-"
    "factuality-contextual-required-s-substitution-nli-literal-anchor-"
    "hybrid-range-guard"
)
CALIBRATION_EXPECTED_CONTROL_COUNT = 25
REVIEW_DISCLOSURE_PREFIX = "검토 주의:"
RANGE_RELATIONS = ("asserted", "negated", "uncertain", "absent")
RANGE_FRAME_NAME = "range_scope"
DEFAULT_RANGE_EMBEDDING_MODEL_PATH = Path(
    "/home/minhee/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-Embedding-0.6B/snapshots/"
    "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
)
DEFAULT_RANGE_NLI_MODEL_PATH = Path(
    "/home/minhee/.cache/huggingface/hub/"
    "models--chunwoolee0--klue_nli_roberta_base_model/snapshots/"
    "d025228dc83570f814626c2b32980d514c65875d"
)
DEFAULT_RANGE_NLI_THRESHOLD = 0.8
DEFAULT_RANGE_EMBEDDING_DELTA_THRESHOLD = 0.05
RANGE_NLI_LABELS = (
    "entailment",
    "neutral",
    "contradiction",
)
EXCLUDED_EVALUATION_CASES = {
    (
        "kim-la-capinera-01",
        (78, 81),
    ): (
        "The reviewed unit transfers an opening pp idea to this reprise, "
        "but its lyric anchor ('Col ritornar') does not match the score at "
        "measures 78-81. Keep the human annotation and reviewed range "
        "unchanged; omit only this invalid question/range pairing from the "
        "annotator-question benchmark."
    ),
}

RELATIONSHIPS = (
    "equivalent",
    "minor_omission",
    "materially_incomplete",
    "materially_unreliable",
    "contradictory",
    "unrelated",
)
CRITICAL_ERROR_TYPES = {
    "contradicts_expert",
    "invents_specific_fact",
    "wrong_measure_application",
    "unsafe_vocal_advice",
    "wrong_piece",
    "refuses_despite_answerable",
    "ungrounded_as_fact",
    "other_material_error",
}
MISPLACED_RELATIONSHIP_ERROR_LABELS = set(RELATIONSHIPS)
SCORE_FIELDS = (
    "core_correctness",
    "core_coverage",
    "factual_safety",
    "question_relevance",
    "range_consistency",
)
JUDGE_FIELDS = (
    "frame",
    "relationship",
    "scores",
    "reference_assessments",
    "candidate_assessments",
    "critical_error_types",
    "confidence",
    "rationale",
)
REFERENCE_ASSESSMENT_STATUSES = (
    "covered",
    "missing",
    "contradicted",
    "not_required",
)
CANDIDATE_ASSESSMENT_STATUSES = (
    "supported",
    "contradicted",
    "unsupported",
    "mixed",
    "irrelevant",
)
RANGE_JUDGE_FIELDS = (
    "frame",
    "excluded_claim_assessments",
    "confidence",
    "rationale",
)
FRAME_NAMES = ("claim_alignment", "contradiction_first")
REFERENCE_CLAIM_SCOPES = {
    "direct_required",
    "optional_background",
    "mixed_or_ambiguous",
}
REFERENCE_CLAIM_FLAGS = {
    "deictic_cross_sentence_or_question_dependency",
    "elliptical_or_run_on_source_text",
    "mixed_required_and_optional_scope",
    "multi_claim",
    "slash_joined_claims",
}
REFERENCE_ATOMIC_CLAIM_FIELDS = (
    "claim_index",
    "text",
    "scope",
)
QUESTION_SCOPE_AUTHORITY = "question_level_curated"
SUPPORTING_SCOPE_AUTHORITY = "supporting_only_not_question_required"
CURATION_REVIEW_SCOPE_AUTHORITY = "curation_review_context"
RETRIEVED_EXPERT_SCOPE_AUTHORITY = "retrieved_expert_factuality_only"

AskFunction = Callable[..., dict[str, Any]]
JudgeFunction = Callable[[list[dict[str, str]]], str]
RangeSignalFunction = Callable[
    [Mapping[str, Any]],
    dict[str, Any],
]
CoverageSignalFunction = Callable[
    [Mapping[str, Any]],
    dict[str, Any],
]
CheckpointFunction = Callable[[], None]


class EvaluationInputError(ValueError):
    """Raised when the immutable evaluation inputs violate the contract."""


class JudgeOutputError(ValueError):
    """Raised when a judge response does not match the strict schema."""


JSON_APOSTROPHE_ESCAPE_NORMALIZATION = (
    "single_invalid_json_apostrophe_escape_removed"
)


def _load_strict_judge_json(raw: str) -> tuple[Any, list[str]]:
    """Parse exact JSON, repairing one decoder-confirmed apostrophe escape."""

    text = raw.strip()
    try:
        return json.loads(text), []
    except json.JSONDecodeError as error:
        repairable = (
            error.msg == "Invalid \\escape"
            and 0 <= error.pos < len(text) - 1
            and text[error.pos : error.pos + 2] == "\\'"
        )
        if not repairable:
            raise JudgeOutputError(
                f"response is not exact JSON: {error}"
            ) from error
        repair_position = error.pos

    repaired = (
        text[:repair_position] + text[repair_position + 1 :]
    )
    try:
        assessment = json.loads(repaired)
    except json.JSONDecodeError as repair_error:
        raise JudgeOutputError(
            f"response is not exact JSON: {repair_error}"
        ) from repair_error
    return assessment, [JSON_APOSTROPHE_ESCAPE_NORMALIZATION]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def load_trusted_expert_evidence_catalog(
    corpus_path: Path,
) -> dict[str, dict[str, Any]]:
    """Load the fingerprinted corpus records used to authenticate S claims."""

    payload = load_json(corpus_path)
    if not isinstance(payload, list):
        raise EvaluationInputError("Corpus must be a JSON list")
    catalog: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(payload):
        if not isinstance(record, dict):
            raise EvaluationInputError(
                f"Corpus record {index} must be an object"
            )
        if record.get("evidence_type") != "expert_annotation":
            continue
        evidence_id = record.get("id")
        if (
            not isinstance(evidence_id, str)
            or not evidence_id
            or evidence_id in catalog
            or not isinstance(record.get("piece"), str)
            or not isinstance(record.get("answer"), str)
            or not record["answer"].strip()
            or not isinstance(record.get("source_ids"), list)
        ):
            raise EvaluationInputError(
                f"Corpus expert record {index} is malformed or duplicated"
            )
        catalog[evidence_id] = deepcopy(record)
    if not catalog:
        raise EvaluationInputError("Corpus has no expert evidence records")
    return catalog


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    """Durably replace ``path`` with canonical UTF-8 JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(canonical_json_bytes(value))
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


@contextmanager
def exclusive_output_lock(path: Path) -> Iterator[None]:
    """Prevent two evaluator processes from mutating one snapshot."""

    lock_path = path.with_name(f"{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"Another evaluator is using {path}"
            ) from error
        yield


def ranges_overlap(
    first: Sequence[int],
    second: Sequence[int],
) -> bool:
    return first[0] <= second[1] and second[0] <= first[1]


def validate_measure_range(
    value: Any,
    *,
    label: str,
) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in value
        )
        or value[0] < 1
        or value[1] < value[0]
    ):
        raise EvaluationInputError(
            f"{label}: expected a positive inclusive [start, end] range"
        )
    return [value[0], value[1]]


def validate_range_contrast_claims(
    value: Any,
    *,
    inference_ranges: Sequence[Sequence[int]],
    label: str,
) -> list[dict[str, Any]]:
    """Validate optional human-authored, range-distinctive benchmark claims."""

    if value is None:
        return []
    if not isinstance(value, list):
        raise EvaluationInputError(f"{label}: expected a list")
    validated = []
    seen_ranges = set()
    for index, item in enumerate(value):
        item_label = f"{label}[{index}]"
        if (
            not isinstance(item, dict)
            or tuple(item) != ("measure_range", "claims")
        ):
            raise EvaluationInputError(
                f"{item_label}: expected measure_range and claims"
            )
        measure_range = validate_measure_range(
            item["measure_range"],
            label=f"{item_label}.measure_range",
        )
        range_key = tuple(measure_range)
        if range_key in seen_ranges:
            raise EvaluationInputError(
                f"{label}: measure ranges must be unique"
            )
        seen_ranges.add(range_key)
        claims = item["claims"]
        if (
            not isinstance(claims, list)
            or not claims
            or any(
                not isinstance(claim, str) or not claim.strip()
                for claim in claims
            )
            or len(claims) != len(set(claims))
        ):
            raise EvaluationInputError(
                f"{item_label}.claims must be unique non-empty strings"
            )
        if any(ABSOLUTE_MEASURE_RE.search(claim) for claim in claims):
            raise EvaluationInputError(
                f"{item_label}.claims must be location-neutral"
            )
        validated.append(
            {
                "measure_range": measure_range,
                "claims": list(claims),
            }
        )
    inference_range_keys = {
        tuple(measure_range)
        for measure_range in inference_ranges
    }
    if validated and (
        len(validated) < 2
        or seen_ranges != inference_range_keys
        or [item["measure_range"] for item in validated]
        != sorted(item["measure_range"] for item in validated)
    ):
        raise EvaluationInputError(
            f"{label}: claims must cover every inference range in order"
        )
    return validated


def _source_answer_sentences(value: str) -> list[str]:
    """Split an immutable source answer without normalizing its text."""

    return [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+", value)
        if part.strip()
    ]


def validate_reference_claim_scope(
    value: Any,
    *,
    source_answer: str,
    label: str,
    schema_version: str,
) -> dict[str, Any]:
    """Validate the question-level, exact source-answer scope partition."""

    if (
        not isinstance(value, dict)
        or tuple(value) != ("source_answer_sha256", "sentence_items")
    ):
        raise EvaluationInputError(
            f"{label}: expected source_answer_sha256 and sentence_items"
        )
    expected_hash = hashlib.sha256(
        source_answer.encode("utf-8")
    ).hexdigest()
    if value["source_answer_sha256"] != expected_hash:
        raise EvaluationInputError(
            f"{label}.source_answer_sha256 does not match the source answer"
        )
    items = value["sentence_items"]
    expected_sentences = _source_answer_sentences(source_answer)
    if not isinstance(items, list) or len(items) != len(expected_sentences):
        raise EvaluationInputError(
            f"{label}.sentence_items must partition every source sentence"
        )

    validated_items = []
    for index, (item, expected_text) in enumerate(
        zip(items, expected_sentences),
        start=1,
    ):
        item_label = f"{label}.sentence_items[{index - 1}]"
        expected_fields = ("sentence_index", "text", "scope", "flags")
        if isinstance(item, dict) and "claim_items" in item:
            expected_fields += ("claim_items",)
        if not isinstance(item, dict) or tuple(item) != expected_fields:
            raise EvaluationInputError(
                f"{item_label}: non-canonical fields"
            )
        if item["sentence_index"] != index:
            raise EvaluationInputError(
                f"{item_label}.sentence_index must be {index}"
            )
        if item["text"] != expected_text:
            raise EvaluationInputError(
                f"{item_label}.text is not the exact source sentence"
            )
        if item["scope"] not in REFERENCE_CLAIM_SCOPES:
            raise EvaluationInputError(
                f"{item_label}.scope is invalid"
            )
        flags = item["flags"]
        if (
            not isinstance(flags, list)
            or any(not isinstance(flag, str) for flag in flags)
            or len(flags) != len(set(flags))
            or any(flag not in REFERENCE_CLAIM_FLAGS for flag in flags)
        ):
            raise EvaluationInputError(
                f"{item_label}.flags must be unique and recognized"
            )
        if (
            item["scope"] == "mixed_or_ambiguous"
            and "mixed_required_and_optional_scope" not in flags
        ):
            raise EvaluationInputError(
                f"{item_label}: mixed scope requires its review flag"
            )
        if (
            item["scope"] != "mixed_or_ambiguous"
            and "mixed_required_and_optional_scope" in flags
        ):
            raise EvaluationInputError(
                f"{item_label}: mixed review flag requires mixed scope"
            )
        claim_items = item.get("claim_items")
        if claim_items is None:
            if (
                schema_version == "1.3"
                and item["scope"] == "mixed_or_ambiguous"
            ):
                raise EvaluationInputError(
                    f"{item_label}: schema 1.3 mixed scope requires "
                    "claim_items"
                )
            validated_claims = None
        else:
            if (
                schema_version != "1.3"
                or "multi_claim" not in flags
                or not isinstance(claim_items, list)
                or len(claim_items) < 2
            ):
                raise EvaluationInputError(
                    f"{item_label}.claim_items require schema 1.3 "
                    "multi_claim and at least two atoms"
                )
            validated_claims = []
            atomic_scopes = set()
            for claim_index, claim in enumerate(claim_items, start=1):
                claim_label = (
                    f"{item_label}.claim_items[{claim_index - 1}]"
                )
                if (
                    not isinstance(claim, dict)
                    or tuple(claim) != REFERENCE_ATOMIC_CLAIM_FIELDS
                    or claim["claim_index"] != claim_index
                    or not isinstance(claim["text"], str)
                    or not claim["text"].strip()
                    or claim["text"] != claim["text"].strip()
                    or claim["scope"] not in {
                        "direct_required",
                        "optional_background",
                    }
                ):
                    raise EvaluationInputError(
                        f"{claim_label}: invalid atomic claim"
                    )
                validated_claims.append(
                    {
                        "claim_index": claim_index,
                        "text": claim["text"],
                        "scope": claim["scope"],
                    }
                )
                atomic_scopes.add(claim["scope"])
            if " ".join(
                claim["text"] for claim in validated_claims
            ) != expected_text:
                raise EvaluationInputError(
                    f"{item_label}.claim_items must reconstruct the exact "
                    "parent sentence"
                )
            if (
                item["scope"] == "mixed_or_ambiguous"
                and atomic_scopes
                != {"direct_required", "optional_background"}
            ):
                raise EvaluationInputError(
                    f"{item_label}.claim_items must resolve direct and "
                    "optional content"
                )
            if (
                item["scope"] != "mixed_or_ambiguous"
                and atomic_scopes != {item["scope"]}
            ):
                raise EvaluationInputError(
                    f"{item_label}.claim_items must inherit parent scope"
                )
        validated_items.append(
            {
                "sentence_index": index,
                "text": expected_text,
                "scope": item["scope"],
                "flags": list(flags),
                **(
                    {"claim_items": validated_claims}
                    if validated_claims is not None
                    else {}
                ),
            }
        )
    return {
        "source_answer_sha256": expected_hash,
        "sentence_items": validated_items,
    }


def _knowledge_unit_reference(
    unit: Mapping[str, Any],
    *,
    sources: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "knowledge_unit_id": unit["knowledge_unit_id"],
        "source_ids": list(unit["source_ids"]),
        "answer": unit["answer"],
        "rewrite_status": unit["rewrite_status"],
        "rewrite_notes": unit.get("rewrite_notes", ""),
        "measure_status": unit["measure_status"],
        "measure_ranges": deepcopy(unit.get("measure_ranges") or []),
        "measure_notes": unit.get("measure_notes", ""),
        "contributing_source_answers": [
            {
                "source_id": source_id,
                "question": sources[source_id]["question"],
                "answer": sources[source_id]["answer"],
                "legacy_measure_range_hints": deepcopy(
                    sources[source_id]["legacy_measure_ranges"]
                ),
                "curation_status": sources[source_id][
                    "curation_status"
                ],
                "curation_notes": sources[source_id].get(
                    "curation_notes",
                    "",
                ),
            }
            for source_id in unit["source_ids"]
        ],
    }


def _case_id(source_id: str, measure_range: list[int] | None) -> str:
    if measure_range is None:
        return f"{source_id}__no-range"
    return f"{source_id}__m{measure_range[0]}-{measure_range[1]}"


def _range_source_unit_ids(
    units: Sequence[Mapping[str, Any]],
    measure_range: list[int] | None,
) -> list[str]:
    if measure_range is None:
        return []
    return [
        unit["knowledge_unit_id"]
        for unit in units
        if unit["measure_status"] == "specific"
        and any(
            ranges_overlap(candidate, measure_range)
            for candidate in unit.get("measure_ranges") or []
        )
    ]


def _range_applicable_unit_ids(
    units: Sequence[Mapping[str, Any]],
    measure_range: list[int] | None,
) -> list[str]:
    if measure_range is None:
        return [unit["knowledge_unit_id"] for unit in units]
    applicable: list[str] = []
    for unit in units:
        if unit["measure_status"] == "whole_piece":
            applicable.append(unit["knowledge_unit_id"])
        elif unit["measure_status"] == "specific" and any(
            ranges_overlap(candidate, measure_range)
            for candidate in unit.get("measure_ranges") or []
        ):
            applicable.append(unit["knowledge_unit_id"])
    return applicable


def load_evaluation_questions(
    dataset_root: Path,
    *,
    enforce_expected_counts: bool = True,
) -> tuple[list[dict[str, Any]], list[Path]]:
    """Load questions and authoritative source answers from the dataset."""

    inventory_root = dataset_root / "expert_curation" / "evaluation_questions"
    review_root = dataset_root / "expert_curation" / "review"
    questions: list[dict[str, Any]] = []
    input_paths: list[Path] = []

    for piece_id in TARGET_PIECES:
        inventory_path = inventory_root / f"{piece_id}.json"
        review_path = review_root / f"{piece_id}.json"
        inventory = load_json(inventory_path)
        review = load_json(review_path)
        input_paths.extend((inventory_path, review_path))

        if inventory.get("schema_version") != "1.3":
            raise EvaluationInputError(
                f"{inventory_path}: expected evaluation-question schema 1.3"
            )
        if inventory.get("piece_id") != piece_id:
            raise EvaluationInputError(
                f"{inventory_path}: piece_id does not match filename"
            )
        if review.get("piece_id") != piece_id:
            raise EvaluationInputError(
                f"{review_path}: piece_id does not match filename"
            )

        sources = {
            source["source_id"]: source
            for source in review["source_annotations"]
        }
        units_by_id = {
            unit["knowledge_unit_id"]: unit
            for unit in review["knowledge_units"]
        }

        for item in inventory["questions"]:
            source_id = item["source_id"]
            if source_id not in sources:
                raise EvaluationInputError(
                    f"{inventory_path}: unknown source_id {source_id}"
                )
            source = sources[source_id]
            question = item["paraphrased_question"]
            if item["original_question"] != source["question"]:
                raise EvaluationInputError(
                    f"{source_id}: original question is not verbatim"
                )
            if not isinstance(question, str) or not question.strip():
                raise EvaluationInputError(
                    f"{source_id}: paraphrased question is empty"
                )
            if ABSOLUTE_MEASURE_RE.search(question):
                raise EvaluationInputError(
                    f"{source_id}: question text contains a measure locator"
                )
            if not source["answer"].strip():
                raise EvaluationInputError(
                    f"{source_id}: authoritative source answer is empty"
                )
            reference_claim_scope = validate_reference_claim_scope(
                item.get("reference_claim_scope"),
                source_answer=source["answer"],
                label=f"{source_id}.reference_claim_scope",
                schema_version=inventory["schema_version"],
            )

            linked_units: list[dict[str, Any]] = []
            for unit_id in item["knowledge_unit_ids"]:
                try:
                    unit = units_by_id[unit_id]
                except KeyError as error:
                    raise EvaluationInputError(
                        f"{source_id}: unknown knowledge unit {unit_id}"
                    ) from error
                if source_id not in unit["source_ids"]:
                    raise EvaluationInputError(
                        f"{source_id}: {unit_id} does not link back to source"
                    )
                linked_units.append(unit)

            ranges = [
                validate_measure_range(
                    value,
                    label=f"{source_id}.inference_measure_ranges",
                )
                for value in item["inference_measure_ranges"]
            ]
            if ranges != sorted(ranges) or len({
                tuple(value) for value in ranges
            }) != len(ranges):
                raise EvaluationInputError(
                    f"{source_id}: inference ranges must be sorted and unique"
                )
            range_contrast_claims = validate_range_contrast_claims(
                item.get("range_contrast_claims"),
                inference_ranges=ranges,
                label=f"{source_id}.range_contrast_claims",
            )
            excluded_ranges = [
                {
                    "measure_range": deepcopy(measure_range),
                    "reason": EXCLUDED_EVALUATION_CASES[
                        (source_id, tuple(measure_range))
                    ],
                }
                for measure_range in ranges
                if (source_id, tuple(measure_range))
                in EXCLUDED_EVALUATION_CASES
            ]
            active_ranges = [
                measure_range
                for measure_range in ranges
                if (source_id, tuple(measure_range))
                not in EXCLUDED_EVALUATION_CASES
            ]
            case_ranges: list[list[int] | None] = active_ranges or [None]
            reference_units = [
                _knowledge_unit_reference(unit, sources=sources)
                for unit in linked_units
            ]
            inference_runs = []
            for measure_range in case_ranges:
                inference_runs.append(
                    {
                        "case_id": _case_id(source_id, measure_range),
                        "inference_input": {
                            "question": question,
                            "measure_range": deepcopy(measure_range),
                            "measure_range_applied": measure_range is not None,
                            "measure_range_provenance": (
                                "confirmed_linked_specific_knowledge_unit_ranges"
                                if measure_range is not None
                                else None
                            ),
                            "range_source_knowledge_unit_ids": (
                                _range_source_unit_ids(
                                    linked_units,
                                    measure_range,
                                )
                            ),
                            "range_applicable_knowledge_unit_ids": (
                                _range_applicable_unit_ids(
                                    linked_units,
                                    measure_range,
                                )
                            ),
                            "expected_knowledge_unit_ids": list(
                                item["knowledge_unit_ids"]
                            ),
                            "expected_retrieval_eligible_knowledge_unit_ids": (
                                list(
                                    item[
                                        "retrieval_eligible_knowledge_unit_ids"
                                    ]
                                )
                            ),
                            "allowed_range_contrast_claims": [
                                claim
                                for scoped in range_contrast_claims
                                if scoped["measure_range"] == measure_range
                                for claim in scoped["claims"]
                            ],
                            "excluded_range_contrast_claims": [
                                claim
                                for scoped in range_contrast_claims
                                if scoped["measure_range"] != measure_range
                                for claim in scoped["claims"]
                            ],
                        },
                        "retrieval_probe": None,
                        "generated_answer": None,
                        "semantic_evaluation": None,
                        "phase_errors": {},
                    }
                )

            questions.append(
                {
                    "source_id": source_id,
                    "piece_id": piece_id,
                    "annotator": item["annotator"],
                    "original_question": item["original_question"],
                    "paraphrased_question": question,
                    "review_status": item["review_status"],
                    "knowledge_unit_ids": list(item["knowledge_unit_ids"]),
                    "expected_retrieval_eligible_knowledge_unit_ids": list(
                        item["retrieval_eligible_knowledge_unit_ids"]
                    ),
                    "measure_range_hints": deepcopy(
                        item["measure_range_hints"]
                    ),
                    "inference_measure_ranges": deepcopy(active_ranges),
                    "range_contrast_claims": deepcopy(
                        range_contrast_claims
                    ),
                    "reference_claim_scope": deepcopy(
                        reference_claim_scope
                    ),
                    "excluded_inference_measure_ranges": excluded_ranges,
                    "inference_scope": item["inference_scope"],
                    "authoritative_reference": {
                        "source_answer": source["answer"],
                        "source_legacy_measure_range_hints": deepcopy(
                            source["legacy_measure_ranges"]
                        ),
                        "curation_status": source["curation_status"],
                        "curation_notes": source.get("curation_notes", ""),
                        "reference_claim_scope": deepcopy(
                            reference_claim_scope
                        ),
                        "linked_knowledge_units": reference_units,
                    },
                    "inference_runs": inference_runs,
                }
            )

    case_count = sum(
        len(question["inference_runs"])
        for question in questions
    )
    if enforce_expected_counts and (
        len(questions) != EXPECTED_QUESTION_COUNT
        or case_count != EXPECTED_CASE_COUNT
    ):
        raise EvaluationInputError(
            "Expected exactly "
            f"{EXPECTED_QUESTION_COUNT} questions/{EXPECTED_CASE_COUNT} "
            f"cases, found {len(questions)}/{case_count}"
        )
    return questions, input_paths


def input_file_records(paths: Iterable[Path]) -> list[dict[str, str]]:
    records = []
    for path in sorted({path.resolve() for path in paths}):
        records.append(
            {
                "path": str(path),
                "sha256": file_sha256(path),
            }
        )
    return records


def build_input_fingerprint(
    *,
    input_files: Sequence[Mapping[str, str]],
    top_k: int,
    generator_runtime_settings: Mapping[str, Any],
    generator_model_sha256: str | None,
    judge_model_sha256: str | None,
    judge_backend: str,
    range_guard: Mapping[str, Any],
    judge_calibration_sha256: str | None = None,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "judge_protocol": JUDGE_PROTOCOL_VERSION,
        "target_pieces": TARGET_PIECES,
        "top_k": top_k,
        "generator_runtime_settings": deepcopy(
            dict(generator_runtime_settings)
        ),
        "input_files": list(input_files),
        "generator_model_sha256": generator_model_sha256,
        "judge_model_sha256": judge_model_sha256,
        "judge_backend": judge_backend,
        "range_guard": dict(range_guard),
        "judge_calibration_sha256": judge_calibration_sha256,
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def new_snapshot(
    *,
    questions: list[dict[str, Any]],
    dataset_root: Path,
    input_files: list[dict[str, str]],
    input_fingerprint: str,
    top_k: int,
    generator_model: Mapping[str, Any],
    judge_model: Mapping[str, Any],
    range_guard: Mapping[str, Any],
) -> dict[str, Any]:
    now = utc_now()
    snapshot = {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "status": "in_progress",
            "started_at": now,
            "updated_at": now,
            "finished_at": None,
            "target_pieces": list(TARGET_PIECES),
            "dataset_root": str(dataset_root.resolve()),
            "input_fingerprint": input_fingerprint,
            "input_files": input_files,
            "top_k": top_k,
            "phases": {
                phase: {
                    "status": "pending",
                    "completed_cases": 0,
                    "error_cases": 0,
                    "started_at": None,
                    "finished_at": None,
                }
                for phase in ("retrieval", "generate", "judge")
            },
            "generator_model": dict(generator_model),
            "judge_model": dict(judge_model),
            "range_guard": deepcopy(dict(range_guard)),
            "judge_protocol": {
                "version": JUDGE_PROTOCOL_VERSION,
                "frames": list(FRAME_NAMES),
                "same_model_dual_prompt_is_independent": False,
                "role": "conservative automated triage; human review remains "
                "authoritative",
                "candidate_claims_require_exact_quotes": True,
                "reference_claims_use_validated_ids": True,
                "reference_claims_use_question_level_curator_scope": True,
                "optional_background_is_factuality_only": True,
                "mixed_reference_claims_require_full_two_frame_coverage": (
                    True
                ),
                "other_source_claims_are_not_completeness_targets": True,
                "semantic_packet_includes_retrieved_expert_factuality": True,
                "retrieved_expert_claims_are_completeness_targets": False,
                "retrieved_expert_scope_must_match_case": True,
                "semantic_packet_includes_excluded_range_text": False,
                "hybrid_range_guard": True,
                "atomic_claim_entailment_guard": True,
                "supplemental_substitution_entailment_guard": True,
                "answer_quality_precedes_curation_adjudication_gate": True,
                "grounded_pass_requires_target_expert_source": False,
                "target_source_retrieval_is_diagnostic_only": True,
                "pass_threshold": 80.0,
                "minimum_dimension_score": 3,
                "minimum_confidence": 0.7,
            },
        },
        "summary": {},
        "results": questions,
    }
    refresh_summary(snapshot)
    return snapshot


def all_cases(
    snapshot: Mapping[str, Any],
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    for question in snapshot["results"]:
        for case in question["inference_runs"]:
            yield question, case


def _diagnose_pipeline_result(
    result: Mapping[str, Any],
    *,
    source_id: str,
    expected_ids: Sequence[str],
    eligible_ids: Sequence[str],
    range_applicable_ids: Sequence[str],
) -> dict[str, Any]:
    evidence = result.get("evidence") or []
    expert_evidence = [
        item
        for item in evidence
        if item.get("kind") == "expert"
        or item.get("evidence_type") == "expert_annotation"
    ]
    expert_ids = [
        item["id"]
        for item in expert_evidence
        if isinstance(item.get("id"), str)
    ]
    other_range_context_ids = [
        item["id"]
        for item in expert_evidence
        if (
            item.get("scope_match") == "other_range_context"
            and isinstance(item.get("id"), str)
        )
    ]
    retrieved_expected = [
        unit_id for unit_id in expected_ids if unit_id in expert_ids
    ]
    retrieved_eligible = [
        unit_id for unit_id in eligible_ids if unit_id in expert_ids
    ]
    retrieved_applicable = [
        unit_id
        for unit_id in range_applicable_ids
        if unit_id in expert_ids
    ]
    target_source_grounded = any(
        item.get("in_requested_scope") is True
        and source_id in (item.get("source_ids") or [])
        for item in expert_evidence
    )
    return {
        "expert_evidence_ids": expert_ids,
        "other_range_context_evidence_ids": other_range_context_ids,
        "expected_knowledge_unit_ids_retrieved": retrieved_expected,
        "any_expected_knowledge_unit_retrieved": bool(retrieved_expected),
        "all_expected_knowledge_units_retrieved": (
            bool(expected_ids)
            and set(expected_ids).issubset(expert_ids)
        ),
        "range_applicable_knowledge_unit_ids_retrieved": (
            retrieved_applicable
        ),
        "all_range_applicable_knowledge_units_retrieved": (
            bool(range_applicable_ids)
            and set(range_applicable_ids).issubset(expert_ids)
        ),
        "target_source_grounded": target_source_grounded,
        # Compatibility with the original qualitative-viewer snapshot.
        "expected_expert_evidence_retrieved": bool(retrieved_eligible),
        "all_expected_expert_units_retrieved": (
            bool(eligible_ids)
            and set(eligible_ids).issubset(expert_ids)
        ),
    }


def enrich_pipeline_result(
    result: Mapping[str, Any],
    *,
    question: Mapping[str, Any],
    case: Mapping[str, Any],
) -> dict[str, Any]:
    enriched = deepcopy(dict(result))
    inference_input = case["inference_input"]
    diagnostics = _diagnose_pipeline_result(
        enriched,
        source_id=question["source_id"],
        expected_ids=inference_input["expected_knowledge_unit_ids"],
        eligible_ids=inference_input[
            "expected_retrieval_eligible_knowledge_unit_ids"
        ],
        range_applicable_ids=inference_input[
            "range_applicable_knowledge_unit_ids"
        ],
    )
    enriched["diagnostics"] = diagnostics
    for compatibility_field in (
        "expert_evidence_ids",
        "expected_expert_evidence_retrieved",
        "all_expected_expert_units_retrieved",
    ):
        enriched[compatibility_field] = diagnostics[compatibility_field]
    return enriched


def _phase_field(phase: str) -> str:
    return {
        "retrieval": "retrieval_probe",
        "generate": "generated_answer",
        "judge": "semantic_evaluation",
    }[phase]


def _invalidate_case_from_phase(
    case: dict[str, Any],
    phase: str,
) -> None:
    order = ("retrieval", "generate", "judge")
    start = order.index(phase)
    for affected_phase in order[start:]:
        case[_phase_field(affected_phase)] = None
        case["phase_errors"].pop(affected_phase, None)


def _selected_case(
    case: Mapping[str, Any],
    *,
    selected_case_ids: set[str] | None,
) -> bool:
    return selected_case_ids is None or case["case_id"] in selected_case_ids


def run_pipeline_phase(
    snapshot: dict[str, Any],
    *,
    phase: str,
    ask_fn: AskFunction,
    checkpoint: CheckpointFunction,
    top_k: int,
    selected_case_ids: set[str] | None = None,
    rerun: bool = False,
) -> None:
    if phase not in {"retrieval", "generate"}:
        raise ValueError(f"Unsupported pipeline phase {phase}")
    generate = phase == "generate"
    field = _phase_field(phase)
    phase_state = snapshot["run"]["phases"][phase]
    if phase_state["started_at"] is None:
        phase_state["started_at"] = utc_now()
    phase_state["status"] = "running"
    checkpoint()

    for question, case in all_cases(snapshot):
        if not _selected_case(
            case,
            selected_case_ids=selected_case_ids,
        ):
            continue
        if rerun:
            _invalidate_case_from_phase(case, phase)
            refresh_summary(snapshot)
            checkpoint()
        if case[field] is not None:
            continue

        inference_input = case["inference_input"]
        measure_range = inference_input["measure_range"]
        try:
            result = ask_fn(
                piece_id=question["piece_id"],
                question=inference_input["question"],
                measure_range=(
                    tuple(measure_range)
                    if measure_range is not None
                    else None
                ),
                generate=generate,
                allow_internal_knowledge=False,
                top_k=top_k,
            )
            case[field] = enrich_pipeline_result(
                result,
                question=question,
                case=case,
            )
            case["phase_errors"].pop(phase, None)
        except KeyboardInterrupt:
            refresh_summary(snapshot)
            checkpoint()
            raise
        except Exception as error:  # keep other cases resumable
            case["phase_errors"][phase] = {
                "at": utc_now(),
                "type": type(error).__name__,
                "message": str(error),
            }
        refresh_summary(snapshot)
        checkpoint()

    refresh_summary(snapshot)
    checkpoint()


def _validated_retrieved_expert_evidence(
    generated: Mapping[str, Any],
    *,
    piece_id: str,
    selected_measure_range: list[int] | None,
    trusted_expert_catalog: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Authenticate actual retrieval IDs and reconstruct prompt-visible facts."""

    generated_piece_missing = "piece_id" not in generated
    generated_piece = generated.get("piece_id")
    if (
        (
            trusted_expert_catalog is not None
            and generated_piece_missing
        )
        or (
            not generated_piece_missing
            and generated_piece != piece_id
        )
    ):
        piece_failure_reason = (
            "generated_answer_piece_missing"
            if generated_piece_missing
            else "generated_answer_piece_mismatch"
        )
        return {
            "status": "invalid",
            "auto_pass_eligible": False,
            "accepted_evidence": [],
            "accepted_secondary_context": [],
            "rejected_evidence": [
                {
                    "evidence_id": None,
                    "reason": piece_failure_reason,
                    "severity": "integrity_error",
                }
            ],
            "review_disclosure_expectation": {
                "status": "untrusted_evidence",
                "auto_pass_eligible": False,
                "expected_text": "",
                "evidence_ids": [],
                "all_contributing_expert_evidence_validated": False,
                "failure_reason": piece_failure_reason,
            },
        }
    generated_range_missing = "measure_range" not in generated
    generated_range = generated.get("measure_range")
    if (
        (
            trusted_expert_catalog is not None
            and generated_range_missing
        )
        or (
            not generated_range_missing
            and generated_range != selected_measure_range
        )
    ):
        range_failure_reason = (
            "generated_answer_measure_range_missing"
            if generated_range_missing
            else "generated_answer_measure_range_mismatch"
        )
        return {
            "status": "invalid",
            "auto_pass_eligible": False,
            "accepted_evidence": [],
            "accepted_secondary_context": [],
            "rejected_evidence": [
                {
                    "evidence_id": None,
                    "reason": range_failure_reason,
                    "severity": "integrity_error",
                }
            ],
            "review_disclosure_expectation": {
                "status": "untrusted_evidence",
                "auto_pass_eligible": False,
                "expected_text": "",
                "evidence_ids": [],
                "all_contributing_expert_evidence_validated": False,
                "failure_reason": range_failure_reason,
            },
        }
    evidence = generated.get("evidence") or []
    if not isinstance(evidence, list):
        return {
            "status": "invalid",
            "auto_pass_eligible": False,
            "accepted_evidence": [],
            "accepted_secondary_context": [],
            "rejected_evidence": [
                {
                    "evidence_id": None,
                    "reason": "generated_answer_evidence_not_list",
                    "severity": "integrity_error",
                }
            ],
            "review_disclosure_expectation": {
                "status": "untrusted_evidence",
                "auto_pass_eligible": False,
                "expected_text": "",
                "evidence_ids": [],
                "all_contributing_expert_evidence_validated": False,
                "failure_reason": "generated_answer_evidence_not_list",
            },
        }
    if trusted_expert_catalog is None:
        return {
            "status": "not_configured",
            "auto_pass_eligible": True,
            "accepted_evidence": [],
            "accepted_secondary_context": [],
            "rejected_evidence": [],
            "review_disclosure_expectation": {
                "status": "not_configured",
                "auto_pass_eligible": True,
                "expected_text": "",
                "evidence_ids": [],
                "all_contributing_expert_evidence_validated": None,
                "failure_reason": None,
            },
        }

    from soprano_qa.answer import (
        build_review_disclosure_from_records,
        expert_prompt_factuality_material,
    )
    from soprano_qa.retrieval import (
        measure_scope_match,
        scope_evidence_role,
    )

    selected_ranges = (
        [selected_measure_range]
        if selected_measure_range is not None
        else []
    )
    accepted = []
    accepted_records = []
    accepted_secondary_context = []
    rejected = []
    seen_ids = set()
    contributing_expert_evidence_count = 0

    def reject(
        evidence_id: Any,
        reason: str,
        *,
        severity: str = "integrity_error",
    ) -> None:
        rejected.append(
            {
                "evidence_id": (
                    evidence_id
                    if isinstance(evidence_id, str)
                    else None
                ),
                "reason": reason,
                "severity": severity,
            }
        )

    for index, item in enumerate(evidence):
        if not isinstance(item, Mapping):
            reject(None, f"evidence_{index}_not_object")
            continue
        is_expert = (
            item.get("kind") == "expert"
            or item.get("evidence_type") == "expert_annotation"
        )
        if not is_expert:
            continue
        contributing_expert_evidence_count += 1
        if (
            item.get("kind") != "expert"
            or item.get("evidence_type") != "expert_annotation"
        ):
            reject(item.get("id"), "inconsistent_expert_type")
            continue
        evidence_id = item.get("id")
        if not isinstance(evidence_id, str) or not evidence_id:
            reject(evidence_id, "missing_expert_evidence_id")
            continue
        if evidence_id in seen_ids:
            reject(evidence_id, "duplicate_expert_evidence_id")
            continue
        seen_ids.add(evidence_id)
        record = trusted_expert_catalog.get(evidence_id)
        if record is None:
            reject(evidence_id, "unknown_expert_evidence_id")
            continue
        if record.get("piece") != piece_id:
            reject(evidence_id, "expert_evidence_piece_mismatch")
            continue
        equality_checks = {
            "piece_id": record.get("piece"),
            "text": record.get("answer"),
            "source_ids": record.get("source_ids", []),
            "measure_ranges": record.get("measure_range", []),
            "rewrite_status": record.get("rewrite_status"),
            "rewrite_notes": record.get("rewrite_notes", ""),
            "measure_status": record.get("measure_status"),
            "measure_notes": record.get("measure_notes", ""),
            "retrieval_review_warning": record.get(
                "retrieval_review_warning",
                "",
            ),
        }
        expected_scope_match = measure_scope_match(
            dict(record),
            selected_ranges,
        )
        equality_checks.update(
            {
                "scope_match": expected_scope_match,
                "generation_role": scope_evidence_role(
                    expected_scope_match
                ),
                "in_requested_scope": (
                    expected_scope_match == "overlaps_query_range"
                    if selected_measure_range is not None
                    else expected_scope_match
                    in {
                        "general_evidence",
                        "local_example",
                        "unscoped_pending_review",
                        "unspecified_scope",
                    }
                ),
                "selected_range_claim_authority": (
                    expected_scope_match == "overlaps_query_range"
                    if selected_measure_range is not None
                    else None
                ),
            }
        )
        mismatched_fields = [
            field
            for field, expected in equality_checks.items()
            if item.get(field) != expected
        ]
        if mismatched_fields:
            reject(
                evidence_id,
                "expert_evidence_metadata_mismatch:"
                + ",".join(mismatched_fields),
            )
            continue

        measure_status = record.get("measure_status")
        support_requires_review = (
            record.get("rewrite_status") != "ready"
            or bool(record.get("retrieval_review_warning"))
            or measure_status in {"waiting_for_review", "unspecified"}
        )
        is_other_range_context = (
            selected_measure_range is not None
            and measure_status == "specific"
            and expected_scope_match == "other_range_context"
        )
        if selected_measure_range is None:
            if (
                measure_status == "specific"
                and expected_scope_match != "local_example"
            ):
                reject(
                    evidence_id,
                    "specific_evidence_missing_local_example_role",
                )
                continue
        elif measure_status in {"waiting_for_review", "unspecified"}:
            reject(
                evidence_id,
                "unconfirmed_scope_context_not_runtime_eligible",
            )
            continue
        elif measure_status not in {
            "specific",
            "whole_piece",
            "waiting_for_review",
            "unspecified",
        }:
            reject(evidence_id, "invalid_expert_measure_status")
            continue

        material = expert_prompt_factuality_material(
            dict(record),
            selected_ranges,
        )
        serialized_material = item.get("generation_expert_authority")
        if serialized_material != material:
            reject(
                evidence_id,
                "generation_expert_authority_mismatch",
            )
            continue
        if is_other_range_context:
            accepted_secondary_context.append(
                {
                    "evidence_id": evidence_id,
                    "answer_texts": list(material["answer_texts"]),
                    "question_contexts": list(
                        material["question_contexts"]
                    ),
                    "source_ids": list(record.get("source_ids", [])),
                    "scope_match": expected_scope_match,
                    "generation_role": scope_evidence_role(
                        expected_scope_match
                    ),
                    "selected_range_claim_authority": False,
                    "range_partitioned": material["range_partitioned"],
                }
            )
            continue
        if not material["answer_texts"]:
            reject(
                evidence_id,
                "no_prompt_visible_expert_answer_text",
                severity="not_auto_authority",
            )
            continue
        accepted.append(
            {
                "evidence_id": evidence_id,
                "answer_texts": list(material["answer_texts"]),
                "question_contexts": list(
                    material["question_contexts"]
                ),
                "source_ids": list(record.get("source_ids", [])),
                "scope_match": item.get("scope_match"),
                "generation_role": scope_evidence_role(
                    expected_scope_match
                ),
                "measure_ranges": deepcopy(
                    record.get("measure_range", [])
                ),
                "whole_piece_claim_authority": (
                    expected_scope_match == "general_evidence"
                ),
                "rewrite_status": record.get("rewrite_status"),
                "measure_status": measure_status,
                "retrieval_review_warning": record.get(
                    "retrieval_review_warning",
                    "",
                ),
                "support_requires_review": support_requires_review,
                "range_partitioned": material["range_partitioned"],
            }
        )
        accepted_records.append(record)
    integrity_errors = [
        item
        for item in rejected
        if item["severity"] == "integrity_error"
    ]
    all_expert_evidence_validated = (
        not rejected
        and contributing_expert_evidence_count
        == len(accepted_records) + len(accepted_secondary_context)
    )
    disclosure_evidence_ids = [
        record["id"]
        for record in accepted_records
        if record.get("retrieval_review_warning")
    ]
    expected_disclosure = (
        build_review_disclosure_from_records(
            accepted_records,
            neutralize_specific_measure_locators=any(
                item.get("generation_role") == "local_example"
                for item in accepted
            ),
        )
        if all_expert_evidence_validated
        else ""
    )
    disclosure_expectation = {
        "status": (
            "required"
            if expected_disclosure
            else (
                "none_required"
                if all_expert_evidence_validated
                else "untrusted_evidence"
            )
        ),
        "auto_pass_eligible": all_expert_evidence_validated,
        "expected_text": expected_disclosure,
        "evidence_ids": disclosure_evidence_ids,
        "all_contributing_expert_evidence_validated": (
            all_expert_evidence_validated
        ),
        "failure_reason": (
            None
            if all_expert_evidence_validated
            else "not_all_contributing_expert_evidence_validated"
        ),
    }
    return {
        "status": "invalid" if integrity_errors else "valid",
        "auto_pass_eligible": not integrity_errors,
        "accepted_evidence": accepted,
        "accepted_secondary_context": accepted_secondary_context,
        "rejected_evidence": rejected,
        "review_disclosure_expectation": disclosure_expectation,
    }


def _authenticate_and_strip_review_disclosure(
    answer: Any,
    expectation: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Authenticate one trusted leading disclosure and exclude it from C.

    Only a byte-for-byte disclosure reconstructed from authenticated corpus
    records may be removed. Generic or serialized ``검토 주의:`` prose is
    never stripped merely because it looks like a disclosure.
    """

    expected = expectation.get("expected_text")
    evidence_ids = expectation.get("evidence_ids")
    if not isinstance(expected, str):
        expected = ""
    if not isinstance(evidence_ids, list):
        evidence_ids = []
    audit = {
        "status": "invalid",
        "expected_text": expected,
        "evidence_ids": list(evidence_ids),
        "stripped_from_semantic_candidate": False,
        "auto_pass_eligible": False,
        "failure_reason": None,
        "semantic_candidate_empty": False,
        "additional_disclosure_remains": False,
    }
    if not isinstance(answer, str):
        audit["failure_reason"] = "generated_answer_not_string"
        return audit, ""

    expectation_status = expectation.get("status")
    if expectation_status == "not_configured":
        candidate = normalize_candidate_answer_for_judge(answer)
        audit.update(
            {
                "status": "not_configured",
                "auto_pass_eligible": True,
                "semantic_candidate_empty": not bool(candidate),
            }
        )
        return audit, candidate
    if expectation.get("auto_pass_eligible") is not True:
        candidate = normalize_candidate_answer_for_judge(answer)
        audit.update(
            {
                "status": "untrusted_evidence",
                "failure_reason": (
                    expectation.get("failure_reason")
                    or "review_disclosure_expectation_not_authenticated"
                ),
                "semantic_candidate_empty": not bool(candidate),
            }
        )
        return audit, candidate

    if not expected:
        candidate = normalize_candidate_answer_for_judge(answer)
        if REVIEW_DISCLOSURE_PREFIX in candidate:
            audit.update(
                {
                    "status": "unexpected_disclosure",
                    "failure_reason": (
                        "review_disclosure_present_without_trusted_warning"
                    ),
                    "semantic_candidate_empty": not bool(candidate),
                    "additional_disclosure_remains": True,
                }
            )
        else:
            audit.update(
                {
                    "status": "none_required",
                    "auto_pass_eligible": True,
                    "semantic_candidate_empty": not bool(candidate),
                }
            )
        return audit, candidate

    if not answer.startswith(expected):
        candidate = normalize_candidate_answer_for_judge(answer)
        if expected in answer:
            reason = "trusted_review_disclosure_not_leading"
        elif candidate.startswith(REVIEW_DISCLOSURE_PREFIX):
            reason = "trusted_review_disclosure_mismatch"
        else:
            reason = "trusted_review_disclosure_missing"
        audit.update(
            {
                "status": "invalid",
                "failure_reason": reason,
                "semantic_candidate_empty": not bool(candidate),
                "additional_disclosure_remains": (
                    REVIEW_DISCLOSURE_PREFIX in candidate
                ),
            }
        )
        return audit, candidate
    if len(answer) > len(expected) and not answer[len(expected)].isspace():
        candidate = normalize_candidate_answer_for_judge(answer)
        audit.update(
            {
                "status": "invalid",
                "failure_reason": (
                    "trusted_review_disclosure_missing_end_boundary"
                ),
                "semantic_candidate_empty": not bool(candidate),
                "additional_disclosure_remains": True,
            }
        )
        return audit, candidate

    semantic_source = answer[len(expected):].lstrip()
    candidate = normalize_candidate_answer_for_judge(semantic_source)
    additional_disclosure = REVIEW_DISCLOSURE_PREFIX in candidate
    empty = not bool(candidate)
    audit.update(
        {
            "status": (
                "authenticated_stripped"
                if not additional_disclosure and not empty
                else (
                    "authenticated_stripped_empty_answer"
                    if empty
                    else "authenticated_stripped_additional_disclosure"
                )
            ),
            "stripped_from_semantic_candidate": True,
            "auto_pass_eligible": not additional_disclosure and not empty,
            "failure_reason": (
                "disclosure_only_answer"
                if empty
                else (
                    "additional_review_disclosure_remains_in_answer"
                    if additional_disclosure
                    else None
                )
            ),
            "semantic_candidate_empty": empty,
            "additional_disclosure_remains": additional_disclosure,
        }
    )
    return audit, candidate


def _reference_packet(
    question: Mapping[str, Any],
    case: Mapping[str, Any],
    *,
    trusted_expert_catalog: (
        Mapping[str, Mapping[str, Any]] | None
    ) = None,
) -> dict[str, Any]:
    generated = case["generated_answer"]
    reference = question["authoritative_reference"]
    applicable_ids = set(
        case["inference_input"]["range_applicable_knowledge_unit_ids"]
    )
    applicable_units = [
        unit
        for unit in reference["linked_knowledge_units"]
        if unit["knowledge_unit_id"] in applicable_ids
    ]
    selected_range = case["inference_input"]["measure_range"]
    source_hints = reference.get(
        "source_legacy_measure_range_hints",
        [],
    )
    if selected_range is None:
        source_range_relation = "no_selected_range"
    elif not source_hints:
        source_range_relation = "source_range_unscoped"
    elif any(
        ranges_overlap(hint, selected_range)
        for hint in source_hints
    ):
        source_range_relation = "overlaps_selected_range_review_hint"
    else:
        source_range_relation = "outside_selected_range_review_hint"

    contributors = []
    seen_contributor_ids = set()
    for unit in applicable_units:
        for contributor in unit["contributing_source_answers"]:
            source_id = contributor["source_id"]
            if source_id in seen_contributor_ids:
                continue
            seen_contributor_ids.add(source_id)
            contributors.append(contributor)
    range_scoped_contributors = contributors
    range_disambiguation_applied = False
    mixed_contributor_range_disambiguation = False
    overlapping_contributors = []
    nonoverlapping_hinted_contributors = []
    if selected_range is not None:
        overlapping_contributors = [
            contributor
            for contributor in contributors
            if (
                contributor["legacy_measure_range_hints"]
                and any(
                    ranges_overlap(hint, selected_range)
                    for hint in contributor[
                        "legacy_measure_range_hints"
                    ]
                )
            )
        ]
        nonoverlapping_hinted_contributors = [
            contributor
            for contributor in contributors
            if (
                contributor["legacy_measure_range_hints"]
                and not any(
                    ranges_overlap(hint, selected_range)
                    for hint in contributor[
                        "legacy_measure_range_hints"
                    ]
                )
            )
        ]
        if nonoverlapping_hinted_contributors:
            range_disambiguation_applied = True
            range_scoped_contributors = [
                *overlapping_contributors,
                *[
                    contributor
                    for contributor in contributors
                    if not contributor[
                        "legacy_measure_range_hints"
                    ]
                ],
            ]
            mixed_contributor_range_disambiguation = bool(
                overlapping_contributors
            )
    range_override = (
        source_range_relation == "outside_selected_range_review_hint"
        and bool(applicable_units)
    )
    non_applicable_ids = [
        unit["knowledge_unit_id"]
        for unit in reference["linked_knowledge_units"]
        if unit["knowledge_unit_id"] not in applicable_ids
    ]
    units_by_source: dict[str, set[str]] = {}
    for unit in reference["linked_knowledge_units"]:
        for source_id in unit["source_ids"]:
            units_by_source.setdefault(source_id, set()).add(
                unit["knowledge_unit_id"]
            )
    split_source_ids = {
        source_id
        for source_id, unit_ids in units_by_source.items()
        if len(unit_ids) > 1
    }
    original_source_has_non_applicable_split = any(
        question["source_id"] in unit["source_ids"]
        and unit["knowledge_unit_id"] in non_applicable_ids
        for unit in reference["linked_knowledge_units"]
    )
    original_answer_withheld_reason = None
    if range_override:
        original_answer_withheld_reason = (
            "original_source_range_does_not_apply_to_selected_range"
        )
    elif original_source_has_non_applicable_split:
        original_answer_withheld_reason = (
            "original_source_contains_non_applicable_split_claims"
        )
    reference_claim_scope = deepcopy(
        reference.get("reference_claim_scope")
    )
    question_scope_applies = (
        reference_claim_scope is not None
        and original_answer_withheld_reason is None
    )

    judge_units = deepcopy(applicable_units)
    judge_contributors = deepcopy(range_scoped_contributors)
    excluded_other_range_source_answers = []
    excluded_non_applicable_unit_answers = []
    if mixed_contributor_range_disambiguation:
        excluded_other_range_source_answers = deepcopy(
            nonoverlapping_hinted_contributors
        )
        for unit in judge_units:
            unit["answer"] = ""
            unit["answer_withheld_reason"] = (
                "combined_multi_range_answer_contains_other_range_claims"
            )
            unit["contributing_source_answers"] = [
                deepcopy(contributor)
                for contributor in range_scoped_contributors
                if contributor["source_id"] in unit["source_ids"]
            ]
    else:
        judge_contributors = [
            contributor
            for contributor in judge_contributors
            if contributor["source_id"] not in split_source_ids
        ]
        range_scoped_source_ids = {
            contributor["source_id"]
            for contributor in range_scoped_contributors
        }
        for unit in judge_units:
            withheld_ids = [
                contributor["source_id"]
                for contributor in unit["contributing_source_answers"]
                if (
                    contributor["source_id"] in split_source_ids
                    or (
                        range_disambiguation_applied
                        and contributor["source_id"]
                        not in range_scoped_source_ids
                    )
                )
            ]
            unit["contributing_source_answers"] = [
                contributor
                for contributor in unit["contributing_source_answers"]
                if (
                    contributor["source_id"] not in split_source_ids
                    and (
                        not range_disambiguation_applied
                        or contributor["source_id"]
                        in range_scoped_source_ids
                    )
                )
            ]
            if withheld_ids:
                unit["withheld_split_source_answer_ids"] = withheld_ids
    excluded_non_applicable_unit_answers = [
        {
            "knowledge_unit_id": unit["knowledge_unit_id"],
            "measure_ranges": deepcopy(unit["measure_ranges"]),
            "answer": unit["answer"],
        }
        for unit in reference["linked_knowledge_units"]
        if (
            unit["knowledge_unit_id"] in non_applicable_ids
            and question["source_id"] in unit["source_ids"]
        )
    ]

    supplemental_validation = _validated_retrieved_expert_evidence(
        generated,
        piece_id=question["piece_id"],
        selected_measure_range=selected_range,
        trusted_expert_catalog=trusted_expert_catalog,
    )
    (
        review_disclosure_validation,
        semantic_candidate_answer,
    ) = _authenticate_and_strip_review_disclosure(
        (generated or {}).get("answer", ""),
        supplemental_validation.get(
            "review_disclosure_expectation",
            {
                "status": "untrusted_evidence",
                "auto_pass_eligible": False,
                "expected_text": "",
                "evidence_ids": [],
                "failure_reason": (
                    "review_disclosure_expectation_missing"
                ),
            },
        ),
    )

    return {
        "piece_id": question["piece_id"],
        "source_id": question["source_id"],
        "question_review_status": question["review_status"],
        "question": case["inference_input"]["question"],
        "selected_measure_range": selected_range,
        "authoritative_original_expert_answer": (
            None
            if original_answer_withheld_reason
            else reference["source_answer"]
        ),
        "authoritative_original_expert_question_context": (
            None
            if original_answer_withheld_reason
            else question["original_question"]
        ),
        "authoritative_original_expert_answer_withheld_reason": (
            original_answer_withheld_reason
        ),
        "reference_claim_scope": reference_claim_scope,
        "question_level_reference_scope_applies": question_scope_applies,
        "authoritative_source_legacy_measure_range_hints": source_hints,
        "authoritative_source_range_relation": source_range_relation,
        "range_disambiguation_applied": range_disambiguation_applied,
        "mixed_contributor_range_disambiguation": (
            mixed_contributor_range_disambiguation
        ),
        "source_curation_status": reference["curation_status"],
        "source_curation_notes": reference["curation_notes"],
        "required_claim_scope": (
            (
                "Use the curator-approved question-level exact-span claim "
                "scopes. Multi-claim source sentences may be split into "
                "independent atomic R items; coverage of one atom never "
                "covers another. "
                "direct_required is hard completeness; optional_background "
                "is factuality authority but omission is allowed; "
                "mixed_or_ambiguous requires manual review. Claims supplied "
                "only by other sources or linked units are never required."
            )
            if question_scope_applies
            else (
                "The question-level source answer is outside this selected "
                "range. Treat range-scoped contributor and curated-unit text "
                "as factuality support only, never as required completeness "
                "claims. Manual reference review is required."
            )
            if mixed_contributor_range_disambiguation
            else (
                "The original source answer contains a locator outside this "
                "confirmed recurrence. Treat the location-neutral, "
                "range-applicable curated knowledge-unit answer as "
                "factuality support only, never as a required completeness "
                "claim. Manual reference review is required."
            )
            if range_override
            else (
                "No applicable curator-approved question-level source claim "
                "is available. Supporting text is factuality authority only; "
                "manual reference review is required."
            )
        ),
        "range_scoped_contributing_source_answers": judge_contributors,
        "range_applicable_linked_knowledge_units": judge_units,
        "non_applicable_linked_knowledge_unit_ids": non_applicable_ids,
        "excluded_other_range_source_answers": (
            excluded_other_range_source_answers
        ),
        "excluded_non_applicable_unit_answers": (
            excluded_non_applicable_unit_answers
        ),
        "allowed_range_contrast_claims": deepcopy(
            case["inference_input"].get(
                "allowed_range_contrast_claims",
                [],
            )
        ),
        "excluded_range_contrast_claims": deepcopy(
            case["inference_input"].get(
                "excluded_range_contrast_claims",
                [],
            )
        ),
        "retrieved_expert_factuality_evidence": deepcopy(
            supplemental_validation["accepted_evidence"]
        ),
        "supplemental_evidence_validation": (
            supplemental_validation
        ),
        "review_disclosure_validation": (
            review_disclosure_validation
        ),
        "candidate_answer": semantic_candidate_answer,
    }


def _authoritative_reference_texts(
    packet: Mapping[str, Any],
) -> list[str]:
    """Return only answer text that is authoritative for this exact case."""

    values: list[str] = []
    if packet.get("question_review_status") != "retrievable":
        for unit in packet.get(
            "range_applicable_linked_knowledge_units",
            [],
        ):
            answer = unit.get("answer")
            if isinstance(answer, str) and answer.strip():
                values.append(answer.strip())
        if values:
            # Pending curated units preserve known conflicts/options that a
            # single source answer may not contain. They remain manually
            # review-gated by reference_review_requirement().
            return list(dict.fromkeys(values))
    original = packet.get("authoritative_original_expert_answer")
    if isinstance(original, str) and original.strip():
        # The original expert answer is the primary gold text. Curated units
        # are fallback authority only when range/split disambiguation has
        # deliberately withheld that original answer.
        return [original.strip()]
    for contributor in packet.get(
        "range_scoped_contributing_source_answers",
        [],
    ):
        answer = contributor.get("answer")
        if isinstance(answer, str) and answer.strip():
            values.append(answer.strip())
    for unit in packet.get(
        "range_applicable_linked_knowledge_units",
        [],
    ):
        answer = unit.get("answer")
        if isinstance(answer, str) and answer.strip():
            values.append(answer.strip())
        for contributor in unit.get("contributing_source_answers", []):
            answer = contributor.get("answer")
            if isinstance(answer, str) and answer.strip():
                values.append(answer.strip())
    return list(dict.fromkeys(values))


def _authoritative_question_contexts(
    packet: Mapping[str, Any],
) -> list[str]:
    """Return source-question premise text allowed for this exact case."""

    values = []
    original = packet.get(
        "authoritative_original_expert_question_context"
    )
    if isinstance(original, str) and original.strip():
        values.append(original.strip())
    if not values:
        for contributor in packet.get(
            "range_scoped_contributing_source_answers",
            [],
        ):
            question = contributor.get("question")
            if isinstance(question, str) and question.strip():
                values.append(question.strip())
    return list(dict.fromkeys(values))


def _authoritative_reference_items(
    packet: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return stable claims with curator scope when it applies to the case."""

    claim_scope = packet.get("reference_claim_scope")
    if (
        packet.get("question_level_reference_scope_applies")
        and isinstance(claim_scope, Mapping)
    ):
        scoped_claims = []
        for sentence in claim_scope["sentence_items"]:
            atoms = sentence.get("claim_items")
            if atoms:
                for atom in atoms:
                    scoped_claims.append(
                        {
                            "text": atom["text"],
                            "scope": atom["scope"],
                            "flags": list(sentence["flags"]),
                            "source_sentence_index": sentence[
                                "sentence_index"
                            ],
                            "source_claim_index": atom["claim_index"],
                        }
                    )
            else:
                scoped_claims.append(
                    {
                        "text": sentence["text"],
                        "scope": sentence["scope"],
                        "flags": list(sentence["flags"]),
                        "source_sentence_index": sentence[
                            "sentence_index"
                        ],
                        "source_claim_index": None,
                    }
                )
        items = [
            {
                "reference_id": f"R{index:03d}",
                "text": item["text"],
                "scope": item["scope"],
                "flags": list(item["flags"]),
                "scope_authority": QUESTION_SCOPE_AUTHORITY,
                "source_sentence_index": item["source_sentence_index"],
                "source_claim_index": item["source_claim_index"],
            }
            for index, item in enumerate(scoped_claims, start=1)
        ]
        review_texts = []
        for unit in packet.get(
            "range_applicable_linked_knowledge_units",
            [],
        ):
            note = unit.get("rewrite_notes")
            if (
                unit.get("rewrite_status") != "ready"
                and isinstance(note, str)
                and note.strip()
                and note.strip() not in {
                    item["text"] for item in review_texts
                }
            ):
                review_texts.append(
                    {
                        "text": note.strip(),
                        "knowledge_unit_id": unit["knowledge_unit_id"],
                    }
                )
        for review in review_texts:
            items.append(
                {
                    "reference_id": f"R{len(items) + 1:03d}",
                    "text": review["text"],
                    "scope": "optional_background",
                    "flags": [],
                    "scope_authority": CURATION_REVIEW_SCOPE_AUTHORITY,
                    "source_sentence_index": None,
                    "source_claim_index": None,
                    "knowledge_unit_id": review["knowledge_unit_id"],
                    "review_context_kind": "unresolved_curation_note",
                }
            )
        return items

    items = _sentence_reference_items(
        _authoritative_reference_texts(packet),
        prefix="R",
    )
    if claim_scope is not None:
        # The target source answer was deliberately withheld for this case.
        # Other-source and curated-unit prose may support factuality checks,
        # but question-level curators did not approve it as required content.
        for item in items:
            item.update(
                scope="optional_background",
                flags=[],
                scope_authority=SUPPORTING_SCOPE_AUTHORITY,
                source_sentence_index=None,
                source_claim_index=None,
            )
    return items


def _supplemental_factuality_items(
    packet: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build optional claims from expert evidence actually used by generation.

    These S items can establish that a generated candidate claim is grounded
    in reviewed, range-applicable expert evidence. They can never satisfy or
    replace the completeness obligation attached to a question-scoped R item.
    """

    primary_texts = {
        _normalized_support_text(item["text"])
        for item in _authoritative_reference_items(packet)
    }
    claims: list[dict[str, Any]] = []
    claim_index_by_scope: dict[
        tuple[str, str, tuple[tuple[int, int], ...]],
        int,
    ] = {}
    for evidence in packet.get(
        "retrieved_expert_factuality_evidence",
        [],
    ):
        support_requires_review = evidence[
            "support_requires_review"
        ]
        for sentence in _sentence_reference_items(
            evidence["answer_texts"],
            prefix="unused",
        ):
            text = sentence["text"]
            if (
                text in primary_texts
                and evidence.get("generation_role") != "local_example"
            ):
                continue
            measure_ranges = tuple(
                (int(start), int(end))
                for start, end in evidence.get("measure_ranges", [])
            )
            claim_key = (
                text,
                evidence.get("generation_role", ""),
                measure_ranges,
            )
            existing_index = claim_index_by_scope.get(claim_key)
            if existing_index is not None:
                existing = claims[existing_index]
                if evidence["evidence_id"] not in existing["evidence_ids"]:
                    existing["evidence_ids"].append(evidence["evidence_id"])
                existing["source_ids"] = list(dict.fromkeys(
                    [
                        *existing["source_ids"],
                        *evidence["source_ids"],
                    ]
                ))
                existing["support_requires_review"] = (
                    existing["support_requires_review"]
                    and support_requires_review
                )
                warnings = existing["retrieval_review_warnings"]
                warning = evidence["retrieval_review_warning"]
                if warning and warning not in warnings:
                    warnings.append(warning)
                continue
            claim_index_by_scope[claim_key] = len(claims)
            claims.append(
                {
                    "text": text,
                    "scope": "optional_background",
                    "flags": [],
                    "scope_authority": (
                        RETRIEVED_EXPERT_SCOPE_AUTHORITY
                    ),
                    "source_sentence_index": None,
                    "source_claim_index": None,
                    "evidence_ids": [evidence["evidence_id"]],
                    "source_ids": list(evidence["source_ids"]),
                    "generation_role": evidence.get(
                        "generation_role",
                        "",
                    ),
                    "applicable_measure_ranges": [
                        [start, end]
                        for start, end in measure_ranges
                    ],
                    "whole_piece_claim_authority": evidence.get(
                        "whole_piece_claim_authority",
                        False,
                    ),
                    "support_requires_review": support_requires_review,
                    "retrieval_review_warnings": (
                        [evidence["retrieval_review_warning"]]
                        if evidence["retrieval_review_warning"]
                        else []
                    ),
                }
            )
    return [
        {
            "reference_id": f"S{index:03d}",
            **claim,
        }
        for index, claim in enumerate(claims, start=1)
    ]


def _supplemental_question_contexts(
    packet: Mapping[str, Any],
) -> list[str]:
    return list(dict.fromkeys(
        context
        for evidence in packet.get(
            "retrieved_expert_factuality_evidence",
            [],
        )
        for context in evidence["question_contexts"]
        if context
    ))


def _all_semantic_reference_items(
    packet: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return completeness R items plus factuality-only retrieved S items."""

    return [
        *_authoritative_reference_items(packet),
        *_supplemental_factuality_items(packet),
    ]


def _sentence_reference_items(
    texts: Sequence[str],
    *,
    prefix: str,
) -> list[dict[str, Any]]:
    """Split prose into stable, deduplicated sentence-level items."""

    claims = []
    seen = set()
    for text in texts:
        parts = re.split(r"(?<=[.!?])\s+", text)
        for part in parts:
            normalized = _normalized_support_text(part)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            claims.append(normalized)
    return [
        {
            "reference_id": f"{prefix}{index:03d}",
            "text": claim,
        }
        for index, claim in enumerate(claims, start=1)
    ]


def _excluded_reference_items(
    packet: Mapping[str, Any],
) -> list[dict[str, str]]:
    """Return reviewed atomic other-range claims for the range NLI frame."""

    explicit_claims = packet.get("excluded_range_contrast_claims", [])
    return [
        {
            "reference_id": f"X{index:03d}",
            "text": claim,
        }
        for index, claim in enumerate(explicit_claims, start=1)
    ]


def _allowed_range_contrast_items(
    packet: Mapping[str, Any],
) -> list[dict[str, str]]:
    """Return reviewed distinctive claims for the selected range."""

    return [
        {
            "reference_id": f"A{index:03d}",
            "text": claim,
        }
        for index, claim in enumerate(
            packet.get("allowed_range_contrast_claims", []),
            start=1,
        )
    ]


def _candidate_answer_items(
    packet: Mapping[str, Any],
) -> list[dict[str, str]]:
    """Split the generated answer so every candidate segment is classified."""

    return [
        {
            "candidate_id": item["reference_id"],
            "text": item["text"],
        }
        for item in _sentence_reference_items(
            [packet.get("candidate_answer", "")],
            prefix="C",
        )
    ]


def _semantic_reference_packet(
    packet: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the compact, positive, case-applicable semantic packet."""

    return {
        "piece_id": packet["piece_id"],
        "question": packet["question"],
        "selected_measure_range": packet["selected_measure_range"],
        "authoritative_reference_items": (
            _authoritative_reference_items(packet)
        ),
        "authoritative_source_question_context": (
            _authoritative_question_contexts(packet)
        ),
        "supplemental_source_question_context": (
            _supplemental_question_contexts(packet)
        ),
        "supplemental_factuality_items": (
            _supplemental_factuality_items(packet)
        ),
        "candidate_answer_items": _candidate_answer_items(packet),
        "required_claim_scope": packet["required_claim_scope"],
    }


def reference_review_requirement(
    packet: Mapping[str, Any],
) -> dict[str, Any]:
    """Surface unresolved reference curation without treating notes as facts."""

    reasons = []
    status = packet.get("question_review_status")
    if status != "retrievable":
        reasons.append(f"question_review_status:{status}")
    for unit in packet.get(
        "range_applicable_linked_knowledge_units",
        [],
    ):
        unit_id = unit["knowledge_unit_id"]
        if unit.get("rewrite_status") != "ready":
            reasons.append(
                f"{unit_id}:rewrite_status:{unit.get('rewrite_status')}"
            )
        if unit.get("measure_status") == "waiting_for_review":
            reasons.append(
                f"{unit_id}:measure_status:waiting_for_review"
            )
    reference_items = _authoritative_reference_items(packet)
    if (
        packet.get("reference_claim_scope") is not None
        and not packet.get("question_level_reference_scope_applies")
    ):
        reasons.append(
            "question_level_reference_scope_not_applicable_to_case"
        )
    mixed_ids = [
        item["reference_id"]
        for item in reference_items
        if (
            item.get("scope_authority") == QUESTION_SCOPE_AUTHORITY
            and item.get("scope") == "mixed_or_ambiguous"
        )
    ]
    scoped_items = [
        item
        for item in reference_items
        if item.get("scope") in REFERENCE_CLAIM_SCOPES
    ]
    if scoped_items and not any(
        item.get("scope_authority") == QUESTION_SCOPE_AUTHORITY
        and item.get("scope") == "direct_required"
        for item in scoped_items
    ):
        reasons.append("no_applicable_direct_required_reference")
    return {
        "required": bool(reasons),
        "reasons": list(dict.fromkeys(reasons)),
        "conditional_mixed_reference_ids": mixed_ids,
        "policy": (
            "Automated semantic results remain diagnostic; unresolved "
            "reference curation requires manual adjudication. A curated "
            "mixed_or_ambiguous sentence may pass automatically only when "
            "both judge frames classify its complete proposition as covered."
        ),
    }


def range_scope_requirement(
    packet: Mapping[str, Any],
) -> dict[str, Any]:
    """Describe whether a separate semantic range-contrast judgment is needed."""

    excluded_items = _excluded_reference_items(packet)
    if not excluded_items:
        return {
            "status": "not_applicable",
            "applicable": False,
            "wrong_measure_application": False,
            "excluded_reference_items": [],
            "reason": "no_explicit_other_range_or_split_claims",
        }
    return {
        "status": "judge_required",
        "applicable": True,
        "wrong_measure_application": None,
        "excluded_reference_items": excluded_items,
        "reason": "semantic_range_contrast_frame_required",
    }


JUDGE_SYSTEM_COMMON = """You are a conservative evaluator of a Korean musical-performance QA answer.

The input contains sentence-level authoritative_reference_items (R IDs),
supplemental_factuality_items (S IDs), and candidate_answer_items (C IDs).
Evaluate semantic meaning, not word overlap. Do not reward a retrieved ID,
and never infer candidate content from a reference.
authoritative_source_question_context may contain notation or lyric premises
from the original expert question. It is allowed context that the candidate
may restate, but it is not a required answer claim and has no R ID.
supplemental_source_question_context has the same limited role for retrieved
S evidence: it may resolve a premise or referent, but it cannot cover an R
item or become a required claim.

R items are the only completeness targets. S items come only from expert
annotation evidence actually supplied to generation and authenticated for
their stated role. They may support the factuality of a C claim, but they
never make a missing R item covered and never reduce the required R coverage.
An S item with generation_role=local_example and nonempty
applicable_measure_ranges supports only a candidate claim explicitly limited
to those measures as a local example. It cannot support an unlocalized
whole-piece or frequency claim such as "곡 전반", "자주", or "많이"; that
requires separate evidence with whole_piece_claim_authority=true. If a local
example's exact measure numbers are stated, they must agree with
applicable_measure_ranges. If a C item states an S claim within its allowed
scope, classify and link it normally. If it does not, classify the S item
not_required. If the same candidate meaning entails both an R and an S item,
assess both independently; do not link only to S in order to avoid an R
completeness obligation.
A candidate may therefore be factually supported by S and still be materially
incomplete because it omits R. For example, if R says that two score versions
are both possible and one is commonly used, while S recommends comparing the
original score, a candidate that only recommends comparison covers S but
leaves R missing. Shared topic and relevance to the same question never turn
that S-supported recommendation into coverage of R.

Each R item may include a curator-approved scope:
- direct_required: hard completeness. It must be covered, missing, or
  contradicted; never classify it not_required.
- optional_background: omission alone is allowed, so missing or not_required
  does not reduce completeness. It remains factual authority: if a C item
  states, changes, or contradicts its content, assess and link that C claim
  normally. Do not call candidate content unsupported merely because its
  supporting R item is optional.
- mixed_or_ambiguous: the source sentence mixes required and optional content.
  Assess its full meaning as covered, missing, or contradicted; never classify
  it not_required. Automatic pass is possible only if both judge frames cover
  the sentence's complete proposition; otherwise the evaluator routes it to
  human review.
scope_authority "supporting_only_not_question_required" means the item is
factuality support from another source or curated unit, not a completeness
requirement; treat it like optional_background.
scope_authority "curation_review_context" is a version-controlled curator note
that records an unresolved conflict between source claims. It is not required
answer content and cannot resolve the conflict. It may support a candidate's
accurate caution that the sources disagree or that the exact point remains
unsettled. A candidate that asserts one disputed option as certain conflicts
with this review context even if that option appears in one source R item.
scope_authority "retrieved_expert_factuality_only" marks an S item from actual
in-scope expert evidence. It can support a candidate claim within its
generation_role and measure binding, but is optional and can never count
toward R completeness. support_requires_review=true means that support is
provisional; accurate reliance on it is routed to human review rather than
automatic pass.
An R item with source_claim_index is an exact, human-reviewed atom from a
multi-claim source sentence. Judge that atom independently. Text inherited by
an adjacent atom, or a shared subject in the parent sentence, cannot supply an
unstated mechanism, action, qualifier, or effect.

Judge every R item independently. A C item covers an R item only when the C
item's own text, using the question only to resolve referents, entails the
complete proposition in that R item. Shared topic, shared terminology,
adjacency between R items, or coverage of another R item cannot supply an
unstated detail. Never transfer support between R claims.
When one C item seems related to several R items, test each link separately:
if all other R items were hidden, would that C item still establish this R
claim? Link it only to the R items that pass this test. Do not restate a fact
visible only in an R item as though the C item had said it.

Generic example: if one R item says to connect adjacent vowels and another R
item says to place a consonant with the following syllable, a candidate that
only says to connect the vowels covers the first R item, leaves the consonant
R item missing, and links only to the first R item. Conversely, "let the final
consonant flow into the next vowel" does cover the consonant R item even
though it is a semantic paraphrase rather than a word-for-word match.
Likewise, an answer that names the image or expressive effect of a musical
gesture does not thereby cover a separate atomic claim about the notation or
mechanism that produces it (for example, register/clef movement, repetition,
rhythm, articulation, or dynamics). Cover that mechanism only when the
candidate actually states or necessarily entails it.

Classify EVERY R and S ID exactly once:
- covered: one or more C items semantically preserve this relevant claim;
- missing: the claim is absent (a hard omission only for direct_required or
  mixed_or_ambiguous);
- contradicted: a C item positively conflicts with the claim;
- not_required: only curator-scoped optional_background or an unasserted S
  item that is not needed to answer the supplied question.

Classify EVERY C ID exactly once:
- supported: its material claims are supported by linked R or S items;
- contradicted: it positively conflicts with linked R or S items;
- unsupported: it adds a material factual, musical, or vocal claim absent
  from the authority;
- mixed: the same segment combines supported content with a material
  unsupported or contradictory addition;
- irrelevant: harmless text that neither answers the question nor makes a
  material factual claim.

candidate_ids/reference_ids must express the same semantic link in both
directions. A covered R or S links only to supported or mixed C items. A
contradicted R or S links only to contradicted or mixed C items. A supported C
links only to covered R or S items. A contradicted C links only to
contradicted R or S items. Only covered R items count toward completeness.
Unsupported and irrelevant C items have no reference_ids. Mixed C items link
to every R or S item whose content they preserve or contradict.
A critical_error_type must be anchored to an exact material claim in a C
item classified contradicted, unsupported, or mixed. Never invent candidate
wording in the rationale. If every C item is supported or irrelevant, return
an empty critical_error_types list.
minor_omission is a relationship label, never a critical_error_type.

An omission is not a contradiction. Negating an incorrect proposition is not
asserting it. A clause that only denies an outside proposition is not an
unsupported positive factual claim; classify that clause as irrelevant when
it adds no affirmative advice or musical fact. Use "equivalent" only when all
question-relevant expert claims
are covered, no candidate segment is contradicted/unsupported/mixed, and the
answer directly resolves the question. Use "minor_omission" only for a
secondary missing required detail. Curator-approved optional background may be
missing or not_required without making the answer an omission, but at least
one direct_required reference must be covered for equivalent or
minor_omission. A candidate that
merely repeats expert background while missing the answer is unrelated or
materially incomplete.

Do not require printed measure numbers. range_consistency is an integer 0-4
when a selected range exists and null otherwise. Preserve uncertainty,
options, foreign terms, diction examples, performance qualifiers, and
hedging. A confident unsupported performance or vocal-technique instruction
is a factual-safety problem. Confidence describes your certainty in the
classification, not candidate quality.

Return exactly one JSON object, without Markdown, using exactly this shape:
{
  "frame": "FRAME_NAME",
  "relationship": "equivalent|minor_omission|materially_incomplete|materially_unreliable|contradictory|unrelated",
  "scores": {
    "core_correctness": 0,
    "core_coverage": 0,
    "factual_safety": 0,
    "question_relevance": 0,
    "range_consistency": null
  },
  "reference_assessments": [
    {"reference_id": "R001", "status": "covered", "candidate_ids": ["C001"]},
    {"reference_id": "S001", "status": "covered", "candidate_ids": ["C001"]}
  ],
  "candidate_assessments": [
    {"candidate_id": "C001", "status": "supported", "reference_ids": ["R001", "S001"]}
  ],
  "critical_error_types": [],
  "confidence": 0.0,
  "rationale": ""
}

Scores are integers 0-4. Confidence is 0-1. Allowed critical_error_types:
contradicts_expert, invents_specific_fact, wrong_measure_application,
unsafe_vocal_advice, wrong_piece, refuses_despite_answerable,
ungrounded_as_fact, other_material_error.

Cross-field rules: equivalent requires all applicable scores 4, no missing or
contradicted reference, no contradicted/unsupported/mixed candidate, and no
critical error. Minor omission requires all scores >=3, at least one covered
reference, and no material error. Any contradicted reference/candidate
requires contradictory or materially_unreliable, a material critical error,
and core_correctness/factual_safety below 3. Any unsupported or mixed
candidate requires materially_unreliable (unless contradiction is stronger),
a material critical error, and factual_safety below 3.
"""

FRAME_INSTRUCTIONS = {
    "claim_alignment": """Frame: claim_alignment
Begin with the R completeness targets, then classify every S factuality item
and every C item and verify all links. Decide which R items are required by
the question; S items are never required. Minor wording compression is not an
omission. Set frame exactly to "claim_alignment".""",
    "contradiction_first": """Frame: contradiction_first
Begin with the C items. Classify every candidate segment for contradiction,
overstatement, invented specificity, unsafe advice, wrong-piece leakage, or
unsupported content; then classify every R and every S item and verify all
links. Absence of a problem must be earned rather than assumed. Set frame
exactly to "contradiction_first".""",
}


def build_judge_messages(
    frame: str,
    packet: Mapping[str, Any],
    *,
    previous_error: str | None = None,
) -> list[dict[str, str]]:
    if frame not in FRAME_NAMES:
        raise ValueError(f"Unknown judge frame {frame}")
    repair = ""
    if previous_error:
        repair = (
            "\nYour previous response was invalid: "
            f"{previous_error}. Correct the schema in this attempt."
        )
    reference_packet = _semantic_reference_packet(packet)
    return [
        {
            "role": "system",
            "content": (
                JUDGE_SYSTEM_COMMON.replace("FRAME_NAME", frame)
                + "\n"
                + FRAME_INSTRUCTIONS[frame]
                + repair
            ),
        },
        {
            "role": "user",
            "content": (
                "SEMANTIC_EVALUATION_PACKET (JSON):\n"
                + json.dumps(
                    reference_packet,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\nClassify every listed R ID, S ID, and C ID exactly "
                "once. Use only IDs present in this packet and make every "
                "semantic link bidirectionally consistent."
                + "\n/no_think"
            ),
        },
    ]


RANGE_JUDGE_SYSTEM = """You are checking whether a Korean musical-performance answer applies an other-range claim, using direct semantic NLI.

allowed_range_contrast_items (A IDs) are human-reviewed atomic claims for the
selected range. excluded_reference_items (X IDs) are human-reviewed atomic,
location-neutral claims distinctive to another range. The shared effects and
generic vocabulary have already been removed from X. Do not recompute,
weaken, or reinterpret an X item as shared content. candidate_answer_items
(C IDs) are the generated answer.

For EVERY X ID, compare the C text directly to the complete X proposition:
- asserted: C affirms or semantically entails X;
- negated: C mentions X only to deny or correct it;
- uncertain: C is genuinely ambiguous, modal, or double-negated about X;
- absent: C does not express X.

Assertion does not require a contradiction with A. If X says that dynamics
alternate rapidly and C also says they alternate rapidly, X is asserted even
when A describes a different structure. If A says two dynamic blocks contrast
and C says two blocks contrast, an X about rapid alternation is absent.

Judge paraphrase, ordinary negation, and double negation carefully. Shared
words such as ff/pp,
contrast, expression, echo, or bird do not assert X unless C expresses X's
complete distinctive proposition. "Not X but A" is negated. "Not not X" is
asserted or uncertain, never absent. Link asserted, negated, and uncertain
relations to every relevant C ID. For absent, candidate_ids MUST be [].

Semantic examples:
- X "강약이 두 구간으로 나뉘어 대비된다"; C "강한 부분과 여린
  부분을 두 블록으로 나누어 대비한다" => asserted.
- X "강약이 빠르게 번갈아 등장한다"; C "강약이 쉴 새 없이
  교대한다" => asserted.
- X "강약이 빠르게 번갈아 등장한다"; C "강약을 두 블록으로
  대비한다" => absent with candidate_ids [].
- X "강약이 빠르게 번갈아 등장한다"; C "빠르게 번갈아 나오는
  것이 아니다" => negated.

Return exactly one JSON object, no Markdown:
{
  "frame": "range_scope",
  "excluded_claim_assessments": [
    {"excluded_id": "X001", "relation": "asserted|negated|uncertain|absent", "candidate_ids": ["C001"]}
  ],
  "confidence": 0.0,
  "rationale": ""
}
Confidence is 0-1 and reflects certainty in the range classification.
"""


def build_range_judge_messages(
    packet: Mapping[str, Any],
    *,
    previous_error: str | None = None,
) -> list[dict[str, str]]:
    requirement = range_scope_requirement(packet)
    if not requirement["applicable"]:
        raise ValueError("range-scope judge requested without a contrast")
    repair = ""
    if previous_error:
        repair = (
            "\nYour previous response was invalid: "
            f"{previous_error}. Correct the schema in this attempt."
        )
    range_packet = {
        "piece_id": packet["piece_id"],
        "question": packet["question"],
        "selected_measure_range": packet["selected_measure_range"],
        "allowed_range_contrast_items": (
            _allowed_range_contrast_items(packet)
        ),
        "excluded_reference_items": (
            requirement["excluded_reference_items"]
        ),
        "candidate_answer_items": _candidate_answer_items(packet),
    }
    return [
        {
            "role": "system",
            "content": RANGE_JUDGE_SYSTEM + repair,
        },
        {
            "role": "user",
            "content": (
                "RANGE_CONTRAST_PACKET (JSON):\n"
                + json.dumps(
                    range_packet,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\nClassify every X ID exactly once using only listed C "
                "IDs.\n/no_think"
            ),
        },
    ]


def _validate_llm_range_judge_assessment(
    raw: str,
    *,
    packet: Mapping[str, Any],
    hybrid_signals: Mapping[str, Mapping[str, Any]],
    nli_threshold: float,
) -> dict[str, Any]:
    """Validate the Qwen range frame without assigning the final relation."""

    assessment, json_normalizations = _load_strict_judge_json(raw)
    if (
        not isinstance(assessment, dict)
        or set(assessment) != set(RANGE_JUDGE_FIELDS)
    ):
        raise JudgeOutputError(
            f"range-scope keys must be exactly {RANGE_JUDGE_FIELDS}"
        )
    if assessment["frame"] != RANGE_FRAME_NAME:
        raise JudgeOutputError(
            f"frame must be {RANGE_FRAME_NAME!r}"
        )
    excluded_items = {
        item["reference_id"]: item["text"]
        for item in _excluded_reference_items(packet)
    }
    candidate_items = {
        item["candidate_id"]: item["text"]
        for item in _candidate_answer_items(packet)
    }
    values = assessment["excluded_claim_assessments"]
    if not isinstance(values, list) or not values:
        raise JudgeOutputError(
            "excluded_claim_assessments must classify every X ID once"
        )
    observations: dict[str, list[dict[str, Any]]] = {
        excluded_id: [] for excluded_id in excluded_items
    }
    deterministic_normalizations = list(json_normalizations)
    for index, item in enumerate(values):
        if (
            not isinstance(item, dict)
            or set(item)
            != {"excluded_id", "relation", "candidate_ids"}
        ):
            raise JudgeOutputError(
                f"excluded_claim_assessments[{index}] has invalid fields"
            )
        excluded_id = item["excluded_id"]
        relation = item["relation"]
        candidate_ids = item["candidate_ids"]
        if excluded_id not in excluded_items:
            raise JudgeOutputError(
                "excluded_claim_assessments has an unknown X ID"
            )
        if relation not in RANGE_RELATIONS:
            raise JudgeOutputError(
                f"excluded_claim_assessments[{index}].relation is invalid"
            )
        if (
            not isinstance(candidate_ids, list)
            or any(
                not isinstance(candidate_id, str)
                or candidate_id not in candidate_items
                for candidate_id in candidate_ids
            )
            or len(candidate_ids) != len(set(candidate_ids))
        ):
            raise JudgeOutputError(
                f"excluded_claim_assessments[{index}].candidate_ids is "
                "inconsistent with its relation"
            )
        if relation == "absent" and candidate_ids:
            raise JudgeOutputError(
                f"excluded_claim_assessments[{index}]: absent relation "
                "must not contain candidate_ids"
            )
        elif relation != "absent" and not candidate_ids:
            raise JudgeOutputError(
                f"excluded_claim_assessments[{index}].candidate_ids is "
                "inconsistent with its relation"
            )
        observations[excluded_id].append(
            {
                "excluded_id": excluded_id,
                "excluded_text": excluded_items[excluded_id],
                "llm_relation": relation,
                "candidate_ids": list(candidate_ids),
                "candidate_texts": [
                    candidate_items[candidate_id]
                    for candidate_id in candidate_ids
                ],
            }
        )
    if any(not items for items in observations.values()):
        raise JudgeOutputError(
            "excluded_claim_assessments must classify every X ID once"
        )
    validated = []
    for excluded_id, raw_items in observations.items():
        unique = []
        for item in raw_items:
            identity = (item["llm_relation"], tuple(item["candidate_ids"]))
            if any(
                identity
                == (
                    existing["llm_relation"],
                    tuple(existing["candidate_ids"]),
                )
                for existing in unique
            ):
                continue
            unique.append(item)
        duplicate_conflict = len(unique) > 1
        if len(raw_items) > 1 and not duplicate_conflict:
            deterministic_normalizations.append(
                f"{excluded_id}:exact_duplicate_observations_deduplicated"
            )
        if duplicate_conflict:
            scores = hybrid_signals[excluded_id]["candidate_nli"]
            entailed_ids = [
                item["candidate_id"]
                for item in scores
                if item["entailment"] >= nli_threshold
            ]
            contradicted_ids = [
                item["candidate_id"]
                for item in scores
                if item["contradiction"] >= nli_threshold
            ]
            if entailed_ids:
                relation = "asserted"
                candidate_ids = entailed_ids
                resolution = "clause_nli_entailment"
            elif contradicted_ids:
                relation = "negated"
                candidate_ids = contradicted_ids
                resolution = "clause_nli_contradiction"
            else:
                relation = "uncertain"
                candidate_ids = list(dict.fromkeys(
                    candidate_id
                    for item in unique
                    for candidate_id in item["candidate_ids"]
                ))
                if not candidate_ids:
                    candidate_ids = list(candidate_items)
                resolution = "unresolved"
            deterministic_normalizations.append(
                f"{excluded_id}:conflicting_duplicate_observations:"
                f"{resolution}"
            )
            consolidated = {
                "excluded_id": excluded_id,
                "excluded_text": excluded_items[excluded_id],
                "llm_relation": relation,
                "candidate_ids": candidate_ids,
                "candidate_texts": [
                    candidate_items[candidate_id]
                    for candidate_id in candidate_ids
                ],
            }
        else:
            consolidated = deepcopy(unique[0])
        consolidated["raw_observations"] = deepcopy(raw_items)
        consolidated["duplicate_conflict"] = duplicate_conflict
        validated.append(consolidated)
    confidence = _strict_number(
        assessment["confidence"],
        label="confidence",
        minimum=0.0,
        maximum=1.0,
    )
    if (
        not isinstance(assessment["rationale"], str)
        or not assessment["rationale"].strip()
    ):
        raise JudgeOutputError("rationale must be a non-empty string")
    return {
        "frame": RANGE_FRAME_NAME,
        "excluded_claim_assessments": validated,
        "confidence": confidence,
        "rationale": assessment["rationale"].strip(),
        "deterministic_normalizations": deterministic_normalizations,
    }


def _validate_range_signal_scores(
    value: Any,
    *,
    packet: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Validate deterministic embedding/NLI scores for every X claim."""

    if (
        not isinstance(value, dict)
        or set(value) != {"excluded_claim_signals"}
        or not isinstance(value["excluded_claim_signals"], list)
    ):
        raise JudgeOutputError(
            "hybrid range signals must contain excluded_claim_signals"
        )
    excluded_ids = [
        item["reference_id"]
        for item in _excluded_reference_items(packet)
    ]
    allowed_ids = [
        item["reference_id"]
        for item in _allowed_range_contrast_items(packet)
    ]
    candidate_ids = [
        item["candidate_id"]
        for item in _candidate_answer_items(packet)
    ]
    if not allowed_ids:
        raise JudgeOutputError(
            "hybrid range guard requires at least one allowed A claim"
        )
    signals = value["excluded_claim_signals"]
    if len(signals) != len(excluded_ids):
        raise JudgeOutputError(
            "hybrid range signals must classify every X ID once"
        )
    validated: dict[str, dict[str, Any]] = {}
    for index, signal in enumerate(signals):
        if (
            not isinstance(signal, dict)
            or set(signal)
            != {"excluded_id", "embedding", "nli", "candidate_nli"}
        ):
            raise JudgeOutputError(
                f"excluded_claim_signals[{index}] has invalid fields"
            )
        excluded_id = signal["excluded_id"]
        if excluded_id not in excluded_ids or excluded_id in validated:
            raise JudgeOutputError(
                "excluded_claim_signals has unknown or duplicate X ID"
            )
        embedding = signal["embedding"]
        if (
            not isinstance(embedding, dict)
            or set(embedding)
            != {
                "candidate_vs_excluded",
                "allowed_similarities",
                "best_allowed_id",
                "candidate_vs_best_allowed",
                "contrast_delta",
            }
        ):
            raise JudgeOutputError(
                f"{excluded_id}.embedding has invalid fields"
            )
        excluded_similarity = _strict_number(
            embedding["candidate_vs_excluded"],
            label=f"{excluded_id}.candidate_vs_excluded",
            minimum=-1.0,
            maximum=1.0,
        )
        allowed_values = embedding["allowed_similarities"]
        if (
            not isinstance(allowed_values, list)
            or len(allowed_values) != len(allowed_ids)
        ):
            raise JudgeOutputError(
                f"{excluded_id}.allowed_similarities must cover every A ID"
            )
        allowed_scores: list[dict[str, Any]] = []
        seen_allowed = set()
        for allowed_index, allowed in enumerate(allowed_values):
            if (
                not isinstance(allowed, dict)
                or set(allowed) != {"allowed_id", "cosine_similarity"}
            ):
                raise JudgeOutputError(
                    f"{excluded_id}.allowed_similarities[{allowed_index}] "
                    "has invalid fields"
                )
            allowed_id = allowed["allowed_id"]
            if (
                allowed_id not in allowed_ids
                or allowed_id in seen_allowed
            ):
                raise JudgeOutputError(
                    f"{excluded_id}.allowed_similarities has an unknown "
                    "or duplicate A ID"
                )
            seen_allowed.add(allowed_id)
            allowed_scores.append(
                {
                    "allowed_id": allowed_id,
                    "cosine_similarity": _strict_number(
                        allowed["cosine_similarity"],
                        label=(
                            f"{excluded_id}.{allowed_id}."
                            "cosine_similarity"
                        ),
                        minimum=-1.0,
                        maximum=1.0,
                    ),
                }
            )
        if [item["allowed_id"] for item in allowed_scores] != allowed_ids:
            raise JudgeOutputError(
                f"{excluded_id}.allowed_similarities must preserve A order"
            )
        best = max(
            allowed_scores,
            key=lambda item: item["cosine_similarity"],
        )
        best_similarity = _strict_number(
            embedding["candidate_vs_best_allowed"],
            label=f"{excluded_id}.candidate_vs_best_allowed",
            minimum=-1.0,
            maximum=1.0,
        )
        delta = _strict_number(
            embedding["contrast_delta"],
            label=f"{excluded_id}.contrast_delta",
            minimum=-2.0,
            maximum=2.0,
        )
        if (
            embedding["best_allowed_id"] != best["allowed_id"]
            or not math.isclose(
                best_similarity,
                best["cosine_similarity"],
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or not math.isclose(
                delta,
                excluded_similarity - best_similarity,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            raise JudgeOutputError(
                f"{excluded_id}.embedding derived values are inconsistent"
            )
        nli = signal["nli"]
        if not isinstance(nli, dict) or set(nli) != set(RANGE_NLI_LABELS):
            raise JudgeOutputError(
                f"{excluded_id}.nli must have exact KLUE NLI labels"
            )
        nli_scores = {
            label: _strict_number(
                nli[label],
                label=f"{excluded_id}.nli.{label}",
                minimum=0.0,
                maximum=1.0,
            )
            for label in RANGE_NLI_LABELS
        }
        if not math.isclose(
            sum(nli_scores.values()),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-4,
        ):
            raise JudgeOutputError(
                f"{excluded_id}.nli probabilities must sum to one"
            )
        candidate_nli_values = signal["candidate_nli"]
        if (
            not isinstance(candidate_nli_values, list)
            or len(candidate_nli_values) != len(candidate_ids)
        ):
            raise JudgeOutputError(
                f"{excluded_id}.candidate_nli must cover every C ID"
            )
        candidate_nli = []
        for candidate_index, candidate_signal in enumerate(
            candidate_nli_values
        ):
            if (
                not isinstance(candidate_signal, dict)
                or set(candidate_signal)
                != {"candidate_id", *RANGE_NLI_LABELS}
                or candidate_signal["candidate_id"]
                != candidate_ids[candidate_index]
            ):
                raise JudgeOutputError(
                    f"{excluded_id}.candidate_nli[{candidate_index}] "
                    "has invalid fields or order"
                )
            candidate_scores = {
                label: _strict_number(
                    candidate_signal[label],
                    label=(
                        f"{excluded_id}.{candidate_ids[candidate_index]}."
                        f"{label}"
                    ),
                    minimum=0.0,
                    maximum=1.0,
                )
                for label in RANGE_NLI_LABELS
            }
            if not math.isclose(
                sum(candidate_scores.values()),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-4,
            ):
                raise JudgeOutputError(
                    f"{excluded_id}.{candidate_ids[candidate_index]} "
                    "NLI probabilities must sum to one"
                )
            candidate_nli.append(
                {
                    "candidate_id": candidate_ids[candidate_index],
                    **candidate_scores,
                }
            )
        validated[excluded_id] = {
            "embedding": {
                "candidate_vs_excluded": excluded_similarity,
                "allowed_similarities": allowed_scores,
                "best_allowed_id": best["allowed_id"],
                "candidate_vs_best_allowed": best_similarity,
                "contrast_delta": delta,
            },
            "nli": nli_scores,
            "candidate_nli": candidate_nli,
        }
    if list(validated) != excluded_ids:
        raise JudgeOutputError(
            "hybrid range signals must preserve X claim order"
        )
    return validated


def _hybrid_range_relation(
    *,
    llm_relation: str,
    embedding_delta: float,
    nli_entailment: float,
    nli_contradiction: float,
    nli_threshold: float,
    embedding_delta_threshold: float,
) -> tuple[str, list[str]]:
    """Apply the calibrated conservative range-risk decision table."""

    reasons = []
    if llm_relation == "asserted":
        reasons.append("llm_asserted")
    if nli_entailment >= nli_threshold:
        reasons.append("nli_entailment_at_or_above_threshold")
    if reasons:
        return "asserted", reasons
    if llm_relation == "uncertain":
        return "uncertain", ["llm_uncertain"]
    if (
        llm_relation == "negated"
        or nli_contradiction >= nli_threshold
    ):
        if llm_relation == "negated":
            reasons.append("llm_negated")
        if nli_contradiction >= nli_threshold:
            reasons.append("nli_contradiction_at_or_above_threshold")
        if embedding_delta >= embedding_delta_threshold:
            reasons.append("embedding_contrast_risk_at_or_above_threshold")
            return "uncertain", reasons
        reasons.append("embedding_contrast_risk_below_threshold")
        return "negated", reasons
    if llm_relation != "absent":
        raise JudgeOutputError(
            f"unsupported raw range relation: {llm_relation}"
        )
    if embedding_delta >= embedding_delta_threshold:
        return "uncertain", [
            "llm_absent",
            "embedding_contrast_risk_at_or_above_threshold",
        ]
    return "absent", [
        "llm_absent",
        "embedding_contrast_risk_below_threshold",
    ]


def validate_range_judge_assessment(
    raw: str,
    *,
    packet: Mapping[str, Any],
    hybrid_signals: Mapping[str, Any],
    nli_threshold: float,
    embedding_delta_threshold: float,
) -> dict[str, Any]:
    """Validate Qwen output and apply the calibrated hybrid range guard."""

    for label, threshold in (
        ("nli_threshold", nli_threshold),
        ("embedding_delta_threshold", embedding_delta_threshold),
    ):
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
        ):
            raise JudgeOutputError(f"{label} must be finite")
    if not 0.0 <= nli_threshold <= 1.0:
        raise JudgeOutputError("nli_threshold must be between zero and one")
    if not -2.0 <= embedding_delta_threshold <= 2.0:
        raise JudgeOutputError(
            "embedding_delta_threshold must be between -2 and 2"
        )
    signals = _validate_range_signal_scores(
        hybrid_signals,
        packet=packet,
    )
    llm_assessment = _validate_llm_range_judge_assessment(
        raw,
        packet=packet,
        hybrid_signals=signals,
        nli_threshold=float(nli_threshold),
    )
    all_candidate_items = _candidate_answer_items(packet)
    all_candidate_ids = [
        item["candidate_id"] for item in all_candidate_items
    ]
    candidate_text_by_id = {
        item["candidate_id"]: item["text"]
        for item in all_candidate_items
    }
    assessments = []
    for raw_item in llm_assessment["excluded_claim_assessments"]:
        excluded_id = raw_item["excluded_id"]
        scores = signals[excluded_id]
        relation, reasons = _hybrid_range_relation(
            llm_relation=raw_item["llm_relation"],
            embedding_delta=scores["embedding"]["contrast_delta"],
            nli_entailment=scores["nli"]["entailment"],
            nli_contradiction=scores["nli"]["contradiction"],
            nli_threshold=float(nli_threshold),
            embedding_delta_threshold=float(
                embedding_delta_threshold
            ),
        )
        clause_entailment_ids = [
            item["candidate_id"]
            for item in scores["candidate_nli"]
            if item["entailment"] >= float(nli_threshold)
        ]
        clause_contradiction_ids = [
            item["candidate_id"]
            for item in scores["candidate_nli"]
            if item["contradiction"] >= float(nli_threshold)
        ]
        if clause_entailment_ids:
            relation = "asserted"
            reasons.append(
                "candidate_clause_nli_entailment_at_or_above_threshold"
            )
        llm_candidate_ids = list(raw_item["candidate_ids"])
        if relation == "absent":
            candidate_ids = []
            candidate_link_provenance = "not_applicable_absent"
        elif clause_entailment_ids:
            candidate_ids = clause_entailment_ids
            candidate_link_provenance = "clause_nli_localized"
        elif relation == "negated" and clause_contradiction_ids:
            candidate_ids = clause_contradiction_ids
            candidate_link_provenance = "clause_nli_localized"
            reasons.append(
                "candidate_clause_nli_contradiction_at_or_above_threshold"
            )
        elif llm_candidate_ids:
            candidate_ids = llm_candidate_ids
            candidate_link_provenance = "llm_localized"
        else:
            candidate_ids = list(all_candidate_ids)
            candidate_link_provenance = (
                "synthesized_full_answer_unlocalized"
            )
            reasons.append("candidate_links_synthesized_from_full_answer")
        assessments.append(
            {
                "excluded_id": excluded_id,
                "excluded_text": raw_item["excluded_text"],
                "relation": relation,
                "candidate_ids": candidate_ids,
                "candidate_texts": [
                    candidate_text_by_id[candidate_id]
                    for candidate_id in candidate_ids
                ],
                "llm_relation": raw_item["llm_relation"],
                "llm_candidate_ids": llm_candidate_ids,
                "llm_candidate_texts": list(
                    raw_item["candidate_texts"]
                ),
                "llm_raw_observations": deepcopy(
                    raw_item["raw_observations"]
                ),
                "llm_duplicate_conflict": raw_item[
                    "duplicate_conflict"
                ],
                "candidate_link_provenance": candidate_link_provenance,
                "model_scores": scores,
                "decision_reasons": reasons,
            }
        )
    relations = {item["relation"] for item in assessments}
    unlocalized_negation = any(
        item["relation"] == "negated"
        and item["candidate_link_provenance"]
        == "synthesized_full_answer_unlocalized"
        for item in assessments
    )
    duplicate_conflict = any(
        item["llm_duplicate_conflict"]
        for item in assessments
    )
    confidence = llm_assessment["confidence"]
    if "asserted" in relations:
        status = "fail"
    elif (
        "uncertain" in relations
        or unlocalized_negation
        or duplicate_conflict
        or confidence < 0.7
    ):
        status = "human_review"
    else:
        status = "pass"
    return {
        "frame": RANGE_FRAME_NAME,
        "status": status,
        "wrong_measure_application": "asserted" in relations,
        "excluded_claim_assessments": assessments,
        "confidence": confidence,
        "rationale": llm_assessment["rationale"],
        "guard_policy": {
            "version": JUDGE_PROTOCOL_VERSION,
            "nli_threshold": float(nli_threshold),
            "embedding_delta_threshold": float(
                embedding_delta_threshold
            ),
        },
        "deterministic_normalizations": llm_assessment[
            "deterministic_normalizations"
        ],
    }


def _strict_number(
    value: Any,
    *,
    label: str,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise JudgeOutputError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise JudgeOutputError(
            f"{label} must be between {minimum} and {maximum}"
        )
    return number


def _strict_string_list(
    value: Any,
    *,
    label: str,
) -> list[str]:
    if (
        not isinstance(value, list)
        or any(
            not isinstance(item, str) or not item.strip()
            for item in value
        )
    ):
        raise JudgeOutputError(f"{label} must be a list of non-empty strings")
    return list(value)


def normalize_candidate_answer_for_judge(answer: str) -> str:
    """Remove provenance labels that are not substantive answer content."""

    without_citations = KNOWLEDGE_UNIT_CITATION_RE.sub("", answer)
    without_footer = EVIDENCE_FOOTER_LABEL_RE.sub("", without_citations)
    return re.sub(r"\s+", " ", without_footer).strip()


def _normalized_support_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _candidate_with_question_subject_context(
    question: str,
    candidate: str,
) -> str:
    """Resolve a Korean question subject without adding answer content."""

    normalized_question = _normalized_support_text(question).rstrip("?!。")
    match = re.match(
        r"^(.+?)(?:은|는|이|가)\s*"
        r"(?:무엇|어떻게|왜|어떤|어느|언제|어디|무슨|어째서)",
        normalized_question,
    )
    if not match:
        return candidate
    subject = match.group(1).strip(" ,.")
    if not subject:
        return candidate
    return f"{subject}에 대해 말하면, {candidate}"


def _normalized_literal_text(value: str) -> str:
    return (
        unicodedata.normalize("NFKC", value)
        .replace("’", "'")
        .casefold()
    )


def _pronunciation_anchors(value: str) -> list[str]:
    """Return exact Latin tokens explicitly identified as pronunciations."""

    normalized = _normalized_literal_text(value)
    return list(dict.fromkeys(
        match.group(1)
        for match in PRONUNCIATION_ANCHOR_RE.finditer(normalized)
    ))


def _contains_standalone_latin_anchor(value: str, anchor: str) -> bool:
    normalized = _normalized_literal_text(value)
    escaped = re.escape(_normalized_literal_text(anchor))
    return bool(re.search(
        rf"(?<![{LATIN_LETTER_CLASS}']){escaped}"
        rf"(?![{LATIN_LETTER_CLASS}'])",
        normalized,
        flags=re.IGNORECASE,
    ))


def _literal_anchor_coverage_warnings(
    *,
    references: Sequence[Mapping[str, Any]],
    candidates_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Find covered pronunciation claims whose linked C text omits the token.

    This is deliberately an automatic-pass guard rather than a semantic
    reclassification. A natural paraphrase may still be defensible, but a
    judge may not silently transfer a foreign-letter pronunciation detail
    from the reference into generic candidate wording.
    """

    warnings = []
    for reference in references:
        if reference["status"] != "covered":
            continue
        required = _pronunciation_anchors(reference["reference_text"])
        if not required:
            continue
        candidate_ids = list(reference["candidate_ids"])
        linked_texts = [
            candidates_by_id[candidate_id]["candidate_text"]
            for candidate_id in candidate_ids
        ]
        missing = [
            anchor
            for anchor in required
            if not any(
                _contains_standalone_latin_anchor(text, anchor)
                for text in linked_texts
            )
        ]
        if not missing:
            continue
        warnings.append(
            {
                "reference_id": reference["reference_id"],
                "reference_text": reference["reference_text"],
                "candidate_ids": candidate_ids,
                "candidate_texts": linked_texts,
                "required_anchors": required,
                "missing_anchors": missing,
                "description": (
                    "Covered pronunciation claim was not accepted for "
                    "automatic pass because linked candidate text omitted "
                    f"literal anchor(s): {', '.join(missing)}"
                ),
            }
        )
    return warnings


def _numeric_fact_anchors(value: str) -> list[str]:
    normalized = _normalized_literal_text(value)
    return list(dict.fromkeys(
        f"{match.group(1)}{match.group(2)}"
        for match in NUMERIC_FACT_ANCHOR_RE.finditer(normalized)
    ))


def _novel_numeric_fact_warnings(
    *,
    candidates: Sequence[Mapping[str, Any]],
    authoritative_reference_items: Sequence[Mapping[str, Any]],
    allowed_context_texts: Sequence[str],
) -> list[dict[str, Any]]:
    """Find candidate-specific numeric prescriptions absent from authority."""

    allowed_anchors = {
        anchor
        for item in authoritative_reference_items
        for anchor in _numeric_fact_anchors(item["text"])
    }
    allowed_anchors.update(
        anchor
        for text in allowed_context_texts
        for anchor in _numeric_fact_anchors(text)
    )
    warnings = []
    for candidate in candidates:
        candidate_anchors = _numeric_fact_anchors(
            candidate["candidate_text"]
        )
        novel = [
            anchor
            for anchor in candidate_anchors
            if anchor not in allowed_anchors
        ]
        if not novel:
            continue
        warnings.append(
            {
                "candidate_id": candidate["candidate_id"],
                "candidate_text": candidate["candidate_text"],
                "novel_anchors": novel,
                "description": (
                    "Candidate introduced numeric fact anchor(s) absent "
                    f"from expert authority and question context: "
                    f"{', '.join(novel)}"
                ),
            }
        )
    return warnings


def _validate_nli_probabilities(
    value: Any,
    *,
    label: str,
) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) != set(RANGE_NLI_LABELS):
        raise JudgeOutputError(
            f"{label} must have exact KLUE NLI labels"
        )
    scores = {
        name: _strict_number(
            value[name],
            label=f"{label}.{name}",
            minimum=0.0,
            maximum=1.0,
        )
        for name in RANGE_NLI_LABELS
    }
    if not math.isclose(
        sum(scores.values()),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-4,
    ):
        raise JudgeOutputError(
            f"{label} probabilities must sum to one"
        )
    return scores


def _coverage_guard_reference_items(
    authoritative_reference_items: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Select R claims that need independent coverage corroboration.

    Curator-split atomic R claims are always guarded. When authenticated S
    claims are present, direct or mixed question-level R claims are also
    scored so a later validator can detect an S-supported candidate segment
    being incorrectly reused as R completeness.
    """

    has_supplemental = any(
        item.get("scope_authority")
        == RETRIEVED_EXPERT_SCOPE_AUTHORITY
        for item in authoritative_reference_items
    )
    return [
        dict(item)
        for item in authoritative_reference_items
        if (
            item.get("source_claim_index") is not None
            or (
                has_supplemental
                and item.get("scope_authority")
                == QUESTION_SCOPE_AUTHORITY
                and item.get("scope")
                in {"direct_required", "mixed_or_ambiguous"}
            )
        )
    ]


def _validate_atomic_claim_entailment_signals(
    value: Any,
    *,
    authoritative_reference_items: Sequence[Mapping[str, Any]],
    candidate_answer_items: Sequence[Mapping[str, Any]],
    question_context: str,
) -> dict[str, dict[str, Any]]:
    """Validate local NLI corroboration for guarded R claims."""

    if (
        not isinstance(value, dict)
        or set(value) != {"atomic_claim_signals"}
        or not isinstance(value["atomic_claim_signals"], list)
    ):
        raise JudgeOutputError(
            "atomic coverage signals must contain atomic_claim_signals"
        )
    guarded_reference_ids = [
        item["reference_id"]
        for item in _coverage_guard_reference_items(
            authoritative_reference_items
        )
    ]
    candidate_ids = [
        item["candidate_id"] for item in candidate_answer_items
    ]
    candidate_text_by_id = {
        item["candidate_id"]: item["text"]
        for item in candidate_answer_items
    }
    values = value["atomic_claim_signals"]
    if len(values) != len(guarded_reference_ids):
        raise JudgeOutputError(
            "coverage signals must classify every guarded R ID"
        )
    validated: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(values):
        if (
            not isinstance(item, dict)
            or set(item)
            != {
                "reference_id",
                "full_answer_nli",
                "candidate_nli",
                "contextualized_candidate_nli",
            }
        ):
            raise JudgeOutputError(
                f"atomic_claim_signals[{index}] has invalid fields"
            )
        reference_id = item["reference_id"]
        if (
            reference_id not in guarded_reference_ids
            or reference_id in validated
        ):
            raise JudgeOutputError(
                "atomic coverage signals have an unknown or duplicate R ID"
            )
        candidate_values = item["candidate_nli"]
        if (
            not isinstance(candidate_values, list)
            or len(candidate_values) != len(candidate_ids)
        ):
            raise JudgeOutputError(
                f"{reference_id}.candidate_nli must cover every C ID"
            )
        candidate_nli = []
        for candidate_index, candidate_value in enumerate(candidate_values):
            candidate_id = candidate_ids[candidate_index]
            if (
                not isinstance(candidate_value, dict)
                or set(candidate_value)
                != {"candidate_id", *RANGE_NLI_LABELS}
                or candidate_value["candidate_id"] != candidate_id
            ):
                raise JudgeOutputError(
                    f"{reference_id}.candidate_nli[{candidate_index}] "
                    "has invalid fields or order"
                )
            candidate_nli.append(
                {
                    "candidate_id": candidate_id,
                    **_validate_nli_probabilities(
                        {
                            name: candidate_value[name]
                            for name in RANGE_NLI_LABELS
                        },
                        label=f"{reference_id}.{candidate_id}",
                    ),
                }
            )
        contextual_values = item["contextualized_candidate_nli"]
        if (
            not isinstance(contextual_values, list)
            or len(contextual_values) != len(candidate_ids)
        ):
            raise JudgeOutputError(
                f"{reference_id}.contextualized_candidate_nli must "
                "cover every C ID"
            )
        contextualized_candidate_nli = []
        for candidate_index, candidate_value in enumerate(
            contextual_values
        ):
            candidate_id = candidate_ids[candidate_index]
            if (
                not isinstance(candidate_value, dict)
                or set(candidate_value)
                != {
                    "candidate_id",
                    "context_text",
                    *RANGE_NLI_LABELS,
                }
                or candidate_value["candidate_id"] != candidate_id
                or not isinstance(candidate_value["context_text"], str)
                or not candidate_value["context_text"].strip()
                or candidate_value["context_text"].strip()
                != _candidate_with_question_subject_context(
                    question_context,
                    candidate_text_by_id[candidate_id],
                )
            ):
                raise JudgeOutputError(
                    f"{reference_id}.contextualized_candidate_nli"
                    f"[{candidate_index}] has invalid fields or order"
                )
            contextualized_candidate_nli.append(
                {
                    "candidate_id": candidate_id,
                    "context_text": candidate_value[
                        "context_text"
                    ].strip(),
                    **_validate_nli_probabilities(
                        {
                            name: candidate_value[name]
                            for name in RANGE_NLI_LABELS
                        },
                        label=(
                            f"{reference_id}.{candidate_id}."
                            "contextualized"
                        ),
                    ),
                }
            )
        validated[reference_id] = {
            "full_answer_nli": _validate_nli_probabilities(
                item["full_answer_nli"],
                label=f"{reference_id}.full_answer_nli",
            ),
            "candidate_nli": candidate_nli,
            "contextualized_candidate_nli": (
                contextualized_candidate_nli
            ),
        }
    if list(validated) != guarded_reference_ids:
        raise JudgeOutputError(
            "coverage signals must preserve guarded R order"
        )
    return validated


def _atomic_claim_entailment_warnings(
    *,
    references: Sequence[Mapping[str, Any]],
    candidates_by_id: Mapping[str, Mapping[str, Any]],
    authoritative_reference_items: Sequence[Mapping[str, Any]],
    coverage_signals: Mapping[str, Any] | None,
    nli_threshold: float | None,
    question_context: str,
) -> list[dict[str, Any]]:
    """Guard automatic pass when local NLI does not corroborate R coverage.

    Curator-split atoms are always checked. Whenever authenticated S context
    is present, every covered non-atomic required R is also checked so the
    same judge cannot evade the gate by omitting an S link. The signal never
    rewrites the semantic classification; it only prevents automatic pass.
    Human review remains authoritative for neutral or conflicting local-model
    evidence.
    """

    if coverage_signals is None and nli_threshold is None:
        return []
    if coverage_signals is None or nli_threshold is None:
        raise JudgeOutputError(
            "atomic coverage signals and threshold must be supplied together"
        )
    threshold = _strict_number(
        nli_threshold,
        label="atomic_coverage_nli_threshold",
        minimum=0.0,
        maximum=1.0,
    )
    validated = _validate_atomic_claim_entailment_signals(
        coverage_signals,
        authoritative_reference_items=authoritative_reference_items,
        candidate_answer_items=[
            {
                "candidate_id": item["candidate_id"],
                "text": item["candidate_text"],
            }
            for item in candidates_by_id.values()
        ],
        question_context=question_context,
    )
    guarded_ids = set(validated)
    warnings = []
    for reference in references:
        reference_id = reference["reference_id"]
        if (
            reference["status"] != "covered"
            or reference_id not in guarded_ids
        ):
            continue
        is_atomic = any(
            item["reference_id"] == reference_id
            and item.get("source_claim_index") is not None
            for item in authoritative_reference_items
        )
        signal = validated[reference_id]
        linked_ids = list(reference["candidate_ids"])
        linked_scores = [
            item
            for item in signal["candidate_nli"]
            if item["candidate_id"] in linked_ids
        ]
        contextualized_linked_scores = [
            item
            for item in signal["contextualized_candidate_nli"]
            if item["candidate_id"] in linked_ids
        ]
        best_entailment = max(
            [
                signal["full_answer_nli"]["entailment"],
                *(item["entailment"] for item in linked_scores),
                *(
                    item["entailment"]
                    for item in contextualized_linked_scores
                ),
            ],
            default=0.0,
        )
        if best_entailment >= threshold:
            continue
        warnings.append(
            {
                "reference_id": reference_id,
                "reference_text": reference["reference_text"],
                "candidate_ids": linked_ids,
                "candidate_texts": [
                    candidates_by_id[candidate_id]["candidate_text"]
                    for candidate_id in linked_ids
                ],
                "full_answer_nli": signal["full_answer_nli"],
                "linked_candidate_nli": linked_scores,
                "contextualized_linked_candidate_nli": (
                    contextualized_linked_scores
                ),
                "best_entailment": best_entailment,
                "threshold": threshold,
                "guard_scope": (
                    "curator_split_atomic_claim"
                    if is_atomic
                    else "supplemental_substitution_required_claim"
                ),
                "description": (
                    "Covered required claim was not accepted for automatic "
                    "pass because independent local NLI did not corroborate "
                    "the candidate-to-R entailment."
                ),
            }
        )
    return warnings


def _has_positive_omission_language(value: str) -> bool:
    without_negated_phrases = NEGATED_OMISSION_LANGUAGE_RE.sub(" ", value)
    return bool(OMISSION_LANGUAGE_RE.search(without_negated_phrases))


def _strict_exhaustive_assessments(
    value: Any,
    *,
    item_kind: str,
    expected_items: Sequence[Mapping[str, str]],
    linked_items: Sequence[Mapping[str, str]],
    normalizations: list[str] | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise JudgeOutputError(f"{item_kind}_assessments must be a list")
    if item_kind == "reference":
        id_key = "reference_id"
        link_key = "candidate_ids"
        expected_id_key = "reference_id"
        linked_id_key = "candidate_id"
        statuses = REFERENCE_ASSESSMENT_STATUSES
    elif item_kind == "candidate":
        id_key = "candidate_id"
        link_key = "reference_ids"
        expected_id_key = "candidate_id"
        linked_id_key = "reference_id"
        statuses = CANDIDATE_ASSESSMENT_STATUSES
    else:
        raise ValueError(f"Unknown assessment kind: {item_kind}")
    expected = {
        item[expected_id_key]: dict(item)
        for item in expected_items
    }
    linked_ids = {
        item[linked_id_key]
        for item in linked_items
    }
    if len(value) != len(expected):
        observed_ids = [
            item.get(id_key)
            for item in value
            if isinstance(item, dict)
        ]
        missing_ids = sorted(set(expected) - set(observed_ids))
        raise JudgeOutputError(
            f"{item_kind}_assessments must classify every item exactly "
            f"once; expected IDs={sorted(expected)}, "
            f"observed IDs={observed_ids}, missing IDs={missing_ids}"
        )
    validated = []
    seen = set()
    for index, item in enumerate(value):
        if (
            not isinstance(item, dict)
            or set(item) != {id_key, "status", link_key}
        ):
            raise JudgeOutputError(
                f"{item_kind}_assessments[{index}] has invalid fields"
            )
        item_id = item[id_key]
        status = item["status"]
        links = item[link_key]
        if item_id not in expected or item_id in seen:
            raise JudgeOutputError(
                f"{item_kind}_assessments has unknown or duplicate ID "
                f"{item_id!r}"
            )
        seen.add(item_id)
        expected_item = expected[item_id]
        if (
            item_kind == "reference"
            and status == "optional_background"
            and expected_item.get("scope") == "optional_background"
        ):
            status = "not_required"
            if normalizations is not None:
                normalizations.append(
                    f"{item_id}:optional_scope_label_normalized_to_not_required"
                )
        if status not in statuses:
            raise JudgeOutputError(
                f"{item_kind}_assessments[{index}].status is invalid"
            )
        if (
            item_kind == "reference"
            and expected_item.get("scope")
            in {"direct_required", "mixed_or_ambiguous"}
            and status == "not_required"
        ):
            raise JudgeOutputError(
                f"{item_kind}_assessments[{index}]: curator-scoped "
                f"{expected_item['scope']} cannot be not_required"
            )
        if (
            not isinstance(links, list)
            or any(not isinstance(link, str) for link in links)
            or len(links) != len(set(links))
            or any(link not in linked_ids for link in links)
        ):
            raise JudgeOutputError(
                f"{item_kind}_assessments[{index}].{link_key} is invalid"
            )
        requires_links = (
            status in {"covered", "contradicted"}
            if item_kind == "reference"
            else status in {"supported", "contradicted", "mixed"}
        )
        if requires_links and not links:
            raise JudgeOutputError(
                f"{item_kind}_assessments[{index}] link cardinality does "
                "not match its status"
            )
        validated_item = {
            id_key: item_id,
            f"{item_kind}_text": expected_item["text"],
            "status": status,
            link_key: list(links),
        }
        if item_kind == "reference" and "scope" in expected_item:
            scope = expected_item["scope"]
            if scope not in REFERENCE_CLAIM_SCOPES:
                raise JudgeOutputError(
                    f"{item_kind}_assessments[{index}]: invalid expected "
                    "reference scope"
                )
            validated_item.update(
                curator_scope=scope,
                curator_flags=list(expected_item.get("flags", [])),
                scope_authority=expected_item.get(
                    "scope_authority",
                    QUESTION_SCOPE_AUTHORITY,
                ),
            )
            if (
                expected_item.get("scope_authority")
                == RETRIEVED_EXPERT_SCOPE_AUTHORITY
            ):
                validated_item.update(
                    evidence_ids=list(expected_item["evidence_ids"]),
                    source_ids=list(expected_item["source_ids"]),
                    support_requires_review=bool(
                        expected_item["support_requires_review"]
                    ),
                    retrieval_review_warnings=list(
                        expected_item["retrieval_review_warnings"]
                    ),
                )
        validated.append(validated_item)
    return validated


def _repair_missing_supplemental_reference_assessments(
    reference_assessments: Any,
    *,
    candidate_assessments: Any,
    expected_items: Sequence[Mapping[str, Any]],
    normalizations: list[str],
) -> Any:
    """Repair only redundant, one-way bookkeeping for optional S items.

    Qwen can correctly link a candidate to an S item while omitting the
    reciprocal S row. Because S items are never completeness targets, the
    missing row is recoverable without a new semantic inference when every
    linked candidate has the same unambiguous status. Missing R rows,
    malformed/duplicate IDs, and mixed candidate links remain hard errors.
    """

    if (
        not isinstance(reference_assessments, list)
        or not isinstance(candidate_assessments, list)
    ):
        return reference_assessments
    expected_by_id = {
        item.get("reference_id"): item
        for item in expected_items
        if isinstance(item.get("reference_id"), str)
    }
    observed_ids = []
    for item in reference_assessments:
        if (
            not isinstance(item, dict)
            or set(item)
            != {"reference_id", "status", "candidate_ids"}
            or not isinstance(item.get("reference_id"), str)
        ):
            return reference_assessments
        observed_ids.append(item["reference_id"])
    if (
        len(observed_ids) != len(set(observed_ids))
        or any(item_id not in expected_by_id for item_id in observed_ids)
    ):
        return reference_assessments
    missing_ids = [
        item_id
        for item_id in expected_by_id
        if item_id not in observed_ids
    ]
    if not missing_ids:
        return reference_assessments
    if any(
        expected_by_id[item_id].get("scope_authority")
        != RETRIEVED_EXPERT_SCOPE_AUTHORITY
        or expected_by_id[item_id].get("scope")
        != "optional_background"
        for item_id in missing_ids
    ):
        return reference_assessments

    candidate_rows = []
    for item in candidate_assessments:
        if (
            not isinstance(item, dict)
            or set(item)
            != {"candidate_id", "status", "reference_ids"}
            or not isinstance(item.get("candidate_id"), str)
            or not isinstance(item.get("reference_ids"), list)
            or any(
                not isinstance(reference_id, str)
                for reference_id in item["reference_ids"]
            )
        ):
            return reference_assessments
        candidate_rows.append(item)

    additions = []
    for reference_id in missing_ids:
        linked = [
            item
            for item in candidate_rows
            if reference_id in item["reference_ids"]
        ]
        if not linked:
            status = "not_required"
        elif all(item["status"] == "supported" for item in linked):
            status = "covered"
        elif all(item["status"] == "contradicted" for item in linked):
            status = "contradicted"
        else:
            return reference_assessments
        additions.append(
            {
                "reference_id": reference_id,
                "status": status,
                "candidate_ids": [
                    item["candidate_id"] for item in linked
                ],
            }
        )

    for addition in additions:
        normalizations.append(
            f"{addition['reference_id']}:"
            "missing_supplemental_assessment_reconstructed_from_"
            "candidate_links"
        )
    return [*reference_assessments, *additions]


def validate_judge_assessment(
    raw: str,
    *,
    expected_frame: str,
    has_measure_range: bool,
    candidate_answer: str,
    authoritative_reference_items: Sequence[Mapping[str, str]],
    candidate_answer_items: Sequence[Mapping[str, str]] | None = None,
    allowed_context_texts: Sequence[str] = (),
    atomic_coverage_signals: Mapping[str, Any] | None = None,
    atomic_coverage_nli_threshold: float | None = None,
) -> dict[str, Any]:
    del candidate_answer  # Exact candidate content is represented by C IDs.
    candidate_answer_items = list(candidate_answer_items or [])
    assessment, preliminary_normalizations = _load_strict_judge_json(raw)
    if not isinstance(assessment, dict):
        raise JudgeOutputError("response must be a JSON object")
    if (
        len(assessment) != len(JUDGE_FIELDS)
        or set(assessment) != set(JUDGE_FIELDS)
    ):
        raise JudgeOutputError(
            f"top-level keys must be exactly {JUDGE_FIELDS}"
        )
    if assessment["frame"] != expected_frame:
        raise JudgeOutputError(f"frame must be {expected_frame!r}")
    if assessment["relationship"] not in RELATIONSHIPS:
        raise JudgeOutputError("relationship is invalid")

    scores = assessment["scores"]
    if (
        not isinstance(scores, dict)
        or set(scores) != set(SCORE_FIELDS)
    ):
        raise JudgeOutputError(f"score keys must be exactly {SCORE_FIELDS}")
    validated_scores: dict[str, int | None] = {}
    for name in SCORE_FIELDS[:-1]:
        value = scores[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= 4
        ):
            raise JudgeOutputError(f"scores.{name} must be an integer 0-4")
        validated_scores[name] = value
    range_value = scores["range_consistency"]
    if has_measure_range:
        if range_value is None:
            range_value = 2
            preliminary_normalizations.append(
                "null_range_consistency_normalized_to_uncertain"
            )
        if (
            isinstance(range_value, bool)
            or not isinstance(range_value, int)
            or not 0 <= range_value <= 4
        ):
            raise JudgeOutputError(
                "scores.range_consistency must be an integer 0-4"
            )
        validated_scores["range_consistency"] = range_value
    elif range_value is not None:
        raise JudgeOutputError(
            "scores.range_consistency must be null without a range"
        )
    else:
        validated_scores["range_consistency"] = None

    repaired_reference_assessments = (
        _repair_missing_supplemental_reference_assessments(
            assessment["reference_assessments"],
            candidate_assessments=assessment["candidate_assessments"],
            expected_items=authoritative_reference_items,
            normalizations=preliminary_normalizations,
        )
    )
    references = _strict_exhaustive_assessments(
        repaired_reference_assessments,
        item_kind="reference",
        expected_items=authoritative_reference_items,
        linked_items=candidate_answer_items,
        normalizations=preliminary_normalizations,
    )
    candidates = _strict_exhaustive_assessments(
        assessment["candidate_assessments"],
        item_kind="candidate",
        expected_items=candidate_answer_items,
        linked_items=authoritative_reference_items,
        normalizations=preliminary_normalizations,
    )
    candidates_by_id = {
        item["candidate_id"]: item for item in candidates
    }
    for reference in references:
        if reference["status"] != "covered":
            continue
        reciprocal_ids = []
        for candidate_id in reference["candidate_ids"]:
            candidate = candidates_by_id[candidate_id]
            if reference["reference_id"] in candidate["reference_ids"]:
                reciprocal_ids.append(candidate_id)
        can_prune = (
            bool(reciprocal_ids)
            or reference.get("curator_scope") == "optional_background"
        )
        if can_prune and reciprocal_ids != reference["candidate_ids"]:
            removed_ids = [
                candidate_id
                for candidate_id in reference["candidate_ids"]
                if candidate_id not in reciprocal_ids
            ]
            for candidate_id in removed_ids:
                preliminary_normalizations.append(
                    f"{reference['reference_id']}:{candidate_id}:"
                    "one_way_redundant_reference_link_removed"
                )
            reference["candidate_ids"] = reciprocal_ids
            if not reciprocal_ids:
                reference["status"] = "not_required"
    references_by_id = {
        item["reference_id"]: item for item in references
    }
    link_warnings = []
    valid_link_pairs: set[tuple[str, str]] = set()
    for reference in references:
        expected_candidate_statuses = (
            {"supported", "mixed"}
            if reference["status"] == "covered"
            else {"contradicted", "mixed"}
        )
        for candidate_id in reference["candidate_ids"]:
            candidate = candidates_by_id[candidate_id]
            if (
                candidate["status"] not in expected_candidate_statuses
                or reference["reference_id"]
                not in candidate["reference_ids"]
            ):
                link_warnings.append(
                    "reference_link_inconsistent:"
                    f"{reference['reference_id']}:{candidate_id}"
                )
            else:
                valid_link_pairs.add(
                    (reference["reference_id"], candidate_id)
                )
    for candidate in candidates:
        expected_reference_statuses = (
            {"covered"}
            if candidate["status"] == "supported"
            else {"contradicted"}
            if candidate["status"] == "contradicted"
            else {"covered", "contradicted"}
        )
        for reference_id in candidate["reference_ids"]:
            reference = references_by_id[reference_id]
            if (
                reference["status"] not in expected_reference_statuses
                or candidate["candidate_id"]
                not in reference["candidate_ids"]
            ):
                link_warnings.append(
                    "candidate_link_inconsistent:"
                    f"{candidate['candidate_id']}:{reference_id}"
                )
            else:
                valid_link_pairs.add(
                    (reference_id, candidate["candidate_id"])
                )
    link_orphans = [
        f"reference:{reference['reference_id']}"
        for reference in references
        if (
            reference["status"] in {"covered", "contradicted"}
            and not any(
                reference_id == reference["reference_id"]
                for reference_id, _ in valid_link_pairs
            )
        )
    ] + [
        f"candidate:{candidate['candidate_id']}"
        for candidate in candidates
        if (
            candidate["status"] in {"supported", "contradicted", "mixed"}
            and not any(
                candidate_id == candidate["candidate_id"]
                for _, candidate_id in valid_link_pairs
            )
        )
    ]
    literal_anchor_warnings = _literal_anchor_coverage_warnings(
        references=references,
        candidates_by_id=candidates_by_id,
    )
    atomic_entailment_warnings = _atomic_claim_entailment_warnings(
        references=references,
        candidates_by_id=candidates_by_id,
        authoritative_reference_items=authoritative_reference_items,
        coverage_signals=atomic_coverage_signals,
        nli_threshold=atomic_coverage_nli_threshold,
        question_context=(
            allowed_context_texts[0] if allowed_context_texts else ""
        ),
    )
    novel_numeric_warnings = _novel_numeric_fact_warnings(
        candidates=candidates,
        authoritative_reference_items=authoritative_reference_items,
        allowed_context_texts=allowed_context_texts,
    )

    critical_errors = _strict_string_list(
        assessment["critical_error_types"],
        label="critical_error_types",
    )
    if len(critical_errors) != len(set(critical_errors)):
        raise JudgeOutputError("critical_error_types must be unique")
    misplaced_relationship_error_labels = [
        item
        for item in critical_errors
        if item in MISPLACED_RELATIONSHIP_ERROR_LABELS
    ]
    unknown_errors = (
        set(critical_errors)
        - CRITICAL_ERROR_TYPES
        - MISPLACED_RELATIONSHIP_ERROR_LABELS
    )
    if unknown_errors:
        raise JudgeOutputError(
            f"unknown critical_error_types: {sorted(unknown_errors)}"
        )
    critical_errors = [
        item
        for item in critical_errors
        if item in CRITICAL_ERROR_TYPES
    ]
    confidence = _strict_number(
        assessment["confidence"],
        label="confidence",
        minimum=0.0,
        maximum=1.0,
    )
    if (
        not isinstance(assessment["rationale"], str)
        or not assessment["rationale"].strip()
    ):
        raise JudgeOutputError("rationale must be a non-empty string")

    normalizations = list(preliminary_normalizations)
    normalizations.extend(
        "relationship_label_removed_from_critical_errors:" + item
        for item in misplaced_relationship_error_labels
    )

    def normalize_field(field: str, value: Any, reason: str) -> None:
        if assessment[field] == value:
            return
        assessment[field] = value
        normalizations.append(reason)

    normalized_numeric_candidate_ids = []
    for warning in novel_numeric_warnings:
        candidate = candidates_by_id[warning["candidate_id"]]
        if candidate["status"] != "supported":
            continue
        candidate["status"] = "mixed"
        normalized_numeric_candidate_ids.append(candidate["candidate_id"])
        normalizations.append(
            "novel_numeric_fact_forced_mixed_candidate:"
            f"{candidate['candidate_id']}"
        )
    if normalized_numeric_candidate_ids:
        normalize_field(
            "relationship",
            "materially_unreliable",
            "novel_numeric_fact_forced_relationship",
        )
        if "invents_specific_fact" not in critical_errors:
            critical_errors.append("invents_specific_fact")
            normalizations.append(
                "novel_numeric_fact_added_critical_error"
            )
        if validated_scores["factual_safety"] >= 3:
            validated_scores["factual_safety"] = 2
            normalizations.append(
                "novel_numeric_fact_capped_factual_safety"
            )

    missing = [
        item for item in references if item["status"] == "missing"
    ]
    hard_missing = [
        item
        for item in missing
        if item.get("curator_scope") != "optional_background"
    ]
    covered = [
        item for item in references if item["status"] == "covered"
    ]
    contradictions = [
        item for item in references if item["status"] == "contradicted"
    ]
    contradicted_candidates = [
        item for item in candidates if item["status"] == "contradicted"
    ]
    unsupported_candidates = [
        item
        for item in candidates
        if item["status"] in {"unsupported", "mixed"}
    ]
    material_errors = (
        contradictions
        or contradicted_candidates
        or unsupported_candidates
    )
    relationship = assessment["relationship"]
    unanchored_critical_errors: list[str] = []

    if contradictions or contradicted_candidates:
        normalize_field(
            "relationship",
            "contradictory",
            "exhaustive_contradiction_forced_relationship",
        )
        if not {
            "contradicts_expert",
            "wrong_measure_application",
            "wrong_piece",
            "other_material_error",
        }.intersection(critical_errors):
            critical_errors.append("contradicts_expert")
            normalizations.append(
                "exhaustive_contradiction_added_critical_error"
            )
        for name in ("core_correctness", "factual_safety"):
            if validated_scores[name] >= 3:
                validated_scores[name] = 2
                normalizations.append(
                    f"exhaustive_contradiction_capped_{name}"
                )
    if unsupported_candidates:
        if not (contradictions or contradicted_candidates):
            normalize_field(
                "relationship",
                "materially_unreliable",
                "exhaustive_unsupported_forced_relationship",
            )
        if not critical_errors:
            critical_errors.append("ungrounded_as_fact")
            normalizations.append(
                "exhaustive_unsupported_added_critical_error"
            )
        if validated_scores["factual_safety"] >= 3:
            validated_scores["factual_safety"] = 2
            normalizations.append(
                "exhaustive_unsupported_capped_factual_safety"
            )
    if not material_errors:
        applicable_scores = [
            value
            for value in validated_scores.values()
            if value is not None
        ]
        if critical_errors:
            unanchored_critical_errors = list(critical_errors)
            critical_errors = []
            normalizations.append(
                "unanchored_critical_errors_routed_to_review:"
                + ",".join(unanchored_critical_errors)
            )
        if relationship in {"contradictory", "materially_unreliable"}:
            if (
                unanchored_critical_errors
                or misplaced_relationship_error_labels
            ):
                normalize_field(
                    "relationship",
                    "materially_incomplete",
                    "unanchored_critical_error_relationship_routed_to_review",
                )
            else:
                raise JudgeOutputError(
                    f"{relationship} requires an exhaustive material error"
                )
        if relationship in {"equivalent", "minor_omission"} and not covered:
            raise JudgeOutputError(
                f"{relationship} requires at least one covered reference"
            )
        if relationship == "equivalent" and (
            hard_missing
            or _has_positive_omission_language(assessment["rationale"])
            or any(value != 4 for value in applicable_scores)
        ):
            normalize_field(
                "relationship",
                (
                    "minor_omission"
                    if all(value >= 3 for value in applicable_scores)
                    else "materially_incomplete"
                ),
                "equivalent_with_omission_or_low_score_was_downgraded",
            )
        elif relationship == "minor_omission" and any(
            value < 3 for value in applicable_scores
        ):
            normalize_field(
                "relationship",
                "materially_incomplete",
                "low_scoring_minor_omission_was_downgraded",
            )

    assessment["scores"] = validated_scores
    assessment["reference_assessments"] = references
    assessment["candidate_assessments"] = candidates
    assessment["critical_error_types"] = critical_errors
    assessment["unanchored_critical_error_warnings"] = (
        unanchored_critical_errors
    )
    assessment["misplaced_relationship_error_warnings"] = (
        misplaced_relationship_error_labels
    )
    assessment["confidence"] = confidence
    assessment["semantic_link_warnings"] = list(
        dict.fromkeys(link_warnings)
    )
    assessment["semantic_link_orphans"] = link_orphans
    assessment["literal_anchor_coverage_warnings"] = (
        literal_anchor_warnings
    )
    assessment["atomic_claim_entailment_warnings"] = (
        atomic_entailment_warnings
    )
    assessment["novel_numeric_fact_warnings"] = novel_numeric_warnings
    assessment["deterministic_normalizations"] = normalizations
    return assessment


def compute_frame_result(
    assessment: Mapping[str, Any],
) -> dict[str, Any]:
    scores = assessment["scores"]
    reference_assessments = assessment["reference_assessments"]
    has_curator_scope = any(
        "curator_scope" in item
        for item in reference_assessments
    )
    weighted_dimensions = (
        ("core_correctness", 0.35),
        ("core_coverage", 0.25),
        ("factual_safety", 0.25),
        ("question_relevance", 0.10),
        ("range_consistency", 0.05),
    )
    active = [
        (name, weight)
        for name, weight in weighted_dimensions
        if scores[name] is not None
    ]
    weight_total = sum(weight for _, weight in active)
    score = (
        sum(scores[name] / 4 * weight for name, weight in active)
        / weight_total
        * 100
    )
    low_dimensions = [
        name
        for name, _ in active
        if scores[name] < 3
    ]
    reasons = []
    if score < 80:
        reasons.append("weighted_score_below_80")
    if low_dimensions:
        reasons.append("dimension_below_3")
    if assessment["critical_error_types"]:
        reasons.append("critical_error")
    if assessment.get("unanchored_critical_error_warnings"):
        reasons.append("unanchored_critical_error_requires_review")
    if assessment.get("misplaced_relationship_error_warnings"):
        reasons.append(
            "misplaced_relationship_error_label_requires_review"
        )
    if (
        assessment.get("semantic_link_orphans")
        or assessment.get("semantic_link_warnings")
    ):
        reasons.append("semantic_link_inconsistency")
    if assessment.get("literal_anchor_coverage_warnings"):
        reasons.append("unverified_literal_anchor_coverage")
    if assessment.get("atomic_claim_entailment_warnings"):
        reasons.append("unverified_atomic_claim_entailment")
    if any(
        item["status"] == "contradicted"
        for item in reference_assessments
    ) or any(
        item["status"] == "contradicted"
        for item in assessment["candidate_assessments"]
    ):
        reasons.append("reported_contradiction")
    if any(
        item["status"] in {"unsupported", "mixed"}
        for item in assessment["candidate_assessments"]
    ):
        reasons.append("reported_unsupported_material_claim")
    if any(
        item["status"] == "missing"
        and item.get("curator_scope") != "optional_background"
        for item in reference_assessments
    ):
        reasons.append("missing_authoritative_reference")
    if any(
        item["status"] == "not_required"
        and (
            not has_curator_scope
            or item.get("curator_scope") != "optional_background"
        )
        for item in reference_assessments
    ):
        # Legacy/unscoped fixtures remain fail-closed. Curated
        # optional_background is the only approved completeness exemption.
        reasons.append("unapproved_not_required_reference")
    if has_curator_scope and not any(
        item.get("curator_scope") == "direct_required"
        and item.get("scope_authority") == QUESTION_SCOPE_AUTHORITY
        and item["status"] == "covered"
        for item in reference_assessments
    ):
        reasons.append("no_covered_direct_required_reference")
    if (
        assessment["relationship"] == "equivalent"
        and any(
            item["status"] == "missing"
            and item.get("curator_scope") != "optional_background"
            for item in reference_assessments
        )
    ):
        reasons.append("inconsistent_equivalent_with_omission")
    if assessment["confidence"] < 0.7:
        reasons.append("weak_confidence")
    if assessment["relationship"] not in {
        "equivalent",
        "minor_omission",
    }:
        reasons.append("material_relationship")
    return {
        "weighted_score": round(score, 2),
        "pass": not reasons,
        "failure_reasons": reasons,
        "low_dimensions": low_dimensions,
    }


def aggregate_judge_frames(
    frames: Mapping[str, Mapping[str, Any]],
    *,
    generated_answer: Mapping[str, Any],
    range_scope_evaluation: Mapping[str, Any] | None = None,
    reference_review: Mapping[str, Any] | None = None,
    supplemental_evidence_validation: Mapping[str, Any] | None = None,
    review_disclosure_validation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    assessments = {
        frame: frames[frame]["assessment"]
        for frame in FRAME_NAMES
    }
    computed = {
        frame: frames[frame]["computed"]
        for frame in FRAME_NAMES
    }
    relationships = {
        assessment["relationship"]
        for assessment in assessments.values()
    }
    pass_values = {
        result["pass"] for result in computed.values()
    }
    critical_sets = {
        tuple(sorted(assessment["critical_error_types"]))
        for assessment in assessments.values()
    }
    disagreement_reasons = []
    if len(relationships) != 1:
        disagreement_reasons.append("relationship_disagreement")
    if len(pass_values) != 1:
        disagreement_reasons.append("pass_disagreement")
    if len(critical_sets) != 1:
        disagreement_reasons.append("critical_error_disagreement")
    shared_critical_errors = set.intersection(
        *(
            set(assessment["critical_error_types"])
            for assessment in assessments.values()
        )
    )

    weak_confidence_frames = [
        frame
        for frame, assessment in assessments.items()
        if assessment["confidence"] < 0.7
    ]
    substantive_passes = {
        frame: set(result["failure_reasons"]).issubset(
            {"weak_confidence"}
        )
        for frame, result in computed.items()
    }
    reference_coverage_only_failures = {
        "missing_authoritative_reference",
        "unapproved_not_required_reference",
        "no_covered_direct_required_reference",
        "inconsistent_equivalent_with_omission",
        # A one-frame link bookkeeping defect is not evidence that the
        # candidate is semantically wrong.  It must still block an automatic
        # pass, but absent a material claim error it belongs in manual review
        # rather than the failure bucket.
        "semantic_link_inconsistency",
        "unverified_literal_anchor_coverage",
        "unverified_atomic_claim_entailment",
        "weak_confidence",
    }
    coverage_review_frames = {
        frame: bool(result["failure_reasons"])
        and set(result["failure_reasons"]).issubset(
            reference_coverage_only_failures
        )
        for frame, result in computed.items()
    }
    unanchored_critical_review_frames = {
        frame: bool(
            assessment.get("unanchored_critical_error_warnings")
        )
        and set(computed[frame]["failure_reasons"]).issubset(
            {
                "unanchored_critical_error_requires_review",
                "material_relationship",
                "weak_confidence",
            }
        )
        for frame, assessment in assessments.items()
    }
    structurally_unsubstantiated_failure_frames = {}
    for frame, assessment in assessments.items():
        frame_references = assessment["reference_assessments"]
        has_curator_scope = any(
            "curator_scope" in item for item in frame_references
        )
        has_covered_required = any(
            item.get("curator_scope") == "direct_required"
            and item["status"] == "covered"
            for item in frame_references
        )
        has_uncovered_required = any(
            item.get("curator_scope") != "optional_background"
            and item["status"] in {
                "missing",
                "contradicted",
                "not_required",
            }
            for item in frame_references
        )
        has_structured_material_error = (
            any(
                item["status"] == "contradicted"
                for item in frame_references
            )
            or any(
                item["status"]
                in {"contradicted", "unsupported", "mixed"}
                for item in assessment["candidate_assessments"]
            )
            or bool(assessment["critical_error_types"])
        )
        review_only_reasons = {
            "weighted_score_below_80",
            "dimension_below_3",
            "unanchored_critical_error_requires_review",
            "misplaced_relationship_error_label_requires_review",
            "semantic_link_inconsistency",
            "unverified_literal_anchor_coverage",
            "unverified_atomic_claim_entailment",
            "weak_confidence",
            "material_relationship",
        }
        structurally_unsubstantiated_failure_frames[frame] = (
            has_curator_scope
            and has_covered_required
            and not has_uncovered_required
            and not has_structured_material_error
            and bool(computed[frame]["failure_reasons"])
            and set(
                computed[frame]["failure_reasons"]
            ).issubset(review_only_reasons)
        )
    hard_error_frames = [
        frame
        for frame, assessment in assessments.items()
        if (
            any(
                item["status"] == "contradicted"
                for item in assessment["reference_assessments"]
            )
            or any(
                item["status"]
                in {"contradicted", "unsupported", "mixed"}
                for item in assessment["candidate_assessments"]
            )
            or assessment["critical_error_types"]
        )
    ]
    if len(hard_error_frames) == len(FRAME_NAMES):
        status = "fail"
    elif hard_error_frames and not any(substantive_passes.values()):
        status = "fail"
    elif hard_error_frames:
        # One model prompt cannot turn a disputed semantic relation into an
        # automatic failure. It remains ineligible for the primary pass and
        # is surfaced for human review.
        status = "human_review"
    elif all(substantive_passes.values()):
        status = (
            "human_review" if weak_confidence_frames else "pass"
        )
    elif not any(substantive_passes.values()):
        status = (
            "human_review"
            if all(
                coverage_review_frames[frame]
                or unanchored_critical_review_frames[frame]
                or structurally_unsubstantiated_failure_frames[frame]
                for frame in FRAME_NAMES
            )
            else "fail"
        )
    else:
        status = "human_review"

    scope_evaluation = dict(
        range_scope_evaluation
        or {
            "status": "not_applicable",
            "applicable": False,
            "wrong_measure_application": False,
        }
    )
    range_relations_by_candidate: dict[str, set[str]] = {}
    for item in scope_evaluation.get(
        "excluded_claim_assessments",
        [],
    ):
        if (
            item.get(
                "candidate_link_provenance",
                "llm_localized",
            )
            not in {"llm_localized", "clause_nli_localized"}
        ):
            # Only Qwen-localized or clause-NLI-localized relations identify
            # the exact answer segment. Deterministic full-answer signals
            # cannot do that, so synthesized all-C links are never eligible
            # for automatic semantic-error reconciliation.
            continue
        for candidate_id in item.get("candidate_ids", []):
            range_relations_by_candidate.setdefault(
                candidate_id,
                set(),
            ).add(item.get("relation"))

    def errors_confined_to_range_mentions(
        assessment: Mapping[str, Any],
        allowed_candidate_ids: set[str],
        *,
        review_only: bool = False,
    ) -> bool:
        if not allowed_candidate_ids:
            return False
        if assessment.get("literal_anchor_coverage_warnings"):
            return False
        material_candidates = [
            item
            for item in assessment["candidate_assessments"]
            if item["status"]
            in {"contradicted", "unsupported", "mixed"}
        ]
        # Range reconciliation may only correct a narrow range-labeling
        # mistake. It must never erase mixed clauses or independent safety,
        # factuality, piece-identity, or expert-contradiction findings.
        if any(
            item["status"] in {"mixed", "contradicted"}
            for item in material_candidates
        ):
            return False
        allowed_critical_errors = {"wrong_measure_application"}
        if review_only:
            # This path can only downgrade a failure to human review. Local
            # clause NLI has already localized an explicit negation, so these
            # common same-model mislabels are not independently sufficient
            # to force failure. They are never erased for an automatic pass.
            allowed_critical_errors.update(
                {
                    "invents_specific_fact",
                    "ungrounded_as_fact",
                    "unsafe_vocal_advice",
                }
            )
        if (
            set(assessment["critical_error_types"])
            - allowed_critical_errors
        ):
            return False
        contradictions = [
            item
            for item in assessment["reference_assessments"]
            if item["status"] == "contradicted"
        ]
        if contradictions:
            return False
        covered = any(
            item["status"] == "covered"
            for item in assessment["reference_assessments"]
        )
        missing = any(
            item["status"] == "missing"
            for item in assessment["reference_assessments"]
        )
        support_only_review = (
            review_only
            and "no_applicable_direct_required_reference"
            in (reference_review or {}).get("reasons", [])
        )
        hard_missing = any(
            item["status"] == "missing"
            and item.get("curator_scope") != "optional_background"
            for item in assessment["reference_assessments"]
        )
        if (
            not material_candidates
            or hard_missing
            or (
                not review_only
                and (not covered or missing)
            )
            or (
                review_only
                and not covered
                and not support_only_review
            )
        ):
            return False
        if any(
            item["candidate_id"] not in allowed_candidate_ids
            for item in material_candidates
        ):
            return False
        if any(
            not item["candidate_ids"]
            or any(
                candidate_id not in allowed_candidate_ids
                for candidate_id in item["candidate_ids"]
            )
            for item in contradictions
        ):
            return False
        warning_ids = {
            token
            for warning in assessment.get(
                "semantic_link_warnings",
                [],
            )
            for token in warning.split(":")
            if token.startswith("C")
        }
        if review_only and support_only_review:
            # Optional support-only references can be linked inconsistently
            # by the same judge that mislabeled the explicit negation. This
            # remains a review outcome and can never become an automatic pass.
            return True
        return warning_ids.issubset(allowed_candidate_ids)

    negated_candidate_ids = {
        candidate_id
        for candidate_id, relations in range_relations_by_candidate.items()
        if relations and relations.issubset({"negated"})
    }
    uncertain_or_negated_candidate_ids = {
        candidate_id
        for candidate_id, relations in range_relations_by_candidate.items()
        if relations
        and relations.issubset({"negated", "uncertain"})
    }
    if scope_evaluation["status"] == "pass" and negated_candidate_ids:
        reconciled_passes = {
            frame: (
                substantive_passes[frame]
                or errors_confined_to_range_mentions(
                    assessments[frame],
                    negated_candidate_ids,
                )
            )
            for frame in FRAME_NAMES
        }
        if all(reconciled_passes.values()):
            status = (
                "human_review" if weak_confidence_frames else "pass"
            )
            disagreement_reasons.append(
                "negated_other_range_mentions_reconciled"
            )
    elif (
        scope_evaluation["status"] == "human_review"
        and uncertain_or_negated_candidate_ids
    ):
        reconciled_nonfailures = {
            frame: (
                substantive_passes[frame]
                or errors_confined_to_range_mentions(
                    assessments[frame],
                    uncertain_or_negated_candidate_ids,
                    review_only=True,
                )
            )
            for frame in FRAME_NAMES
        }
        if all(reconciled_nonfailures.values()):
            status = "human_review"
            disagreement_reasons.append(
                "uncertain_other_range_mentions_reconciled_for_review"
            )
    if scope_evaluation["status"] == "fail":
        status = "fail"
        disagreement_reasons.append("range_scope_evaluation_failed")
    elif (
        scope_evaluation["status"] == "human_review"
        and status == "pass"
    ):
        status = "human_review"
        disagreement_reasons.append("range_scope_evaluation_needs_review")
    judge_schema_warning_frames = [
        frame
        for frame, assessment in assessments.items()
        if (
            assessment.get("unanchored_critical_error_warnings")
            or assessment.get("misplaced_relationship_error_warnings")
        )
    ]
    if judge_schema_warning_frames:
        # Range reconciliation can localize a genuine range-only error, but
        # it cannot validate a separate unanchored/invalid critical-error
        # declaration made by the judge. Keep that uncertainty review-only
        # after every path that is otherwise allowed to restore a pass.
        if status == "pass":
            status = "human_review"
        disagreement_reasons.append(
            "judge_error_warning_requires_review"
        )
    reference_review_record = dict(
        reference_review
        or {
            "required": False,
            "reasons": [],
            "conditional_mixed_reference_ids": [],
        }
    )
    conditional_mixed_ids = set(
        reference_review_record.get(
            "conditional_mixed_reference_ids",
            [],
        )
    )
    incomplete_mixed_frames = [
        frame
        for frame, assessment in assessments.items()
        if conditional_mixed_ids
        and any(
            item["reference_id"] in conditional_mixed_ids
            and (
                item["status"] != "covered"
                or any(
                    warning["reference_id"] == item["reference_id"]
                    for warning in assessment.get(
                        "literal_anchor_coverage_warnings",
                        [],
                    )
                )
                or any(
                    warning["reference_id"] == item["reference_id"]
                    for warning in assessment.get(
                        "atomic_claim_entailment_warnings",
                        [],
                    )
                )
            )
            for item in assessment["reference_assessments"]
        )
    ]
    reference_review_record["conditional_mixed_fully_covered"] = (
        bool(conditional_mixed_ids)
        and not incomplete_mixed_frames
    )
    reference_review_record["conditional_mixed_incomplete_frames"] = (
        incomplete_mixed_frames
    )
    if incomplete_mixed_frames:
        if status == "pass":
            status = "human_review"
        disagreement_reasons.append(
            "mixed_reference_claim_not_fully_covered"
        )
    provisional_supplemental_usage: dict[str, list[dict[str, Any]]] = {}
    for frame, assessment in assessments.items():
        references_by_id = {
            item["reference_id"]: item
            for item in assessment["reference_assessments"]
        }
        frame_usage = []
        for candidate in assessment["candidate_assessments"]:
            provisional_ids = [
                reference_id
                for reference_id in candidate["reference_ids"]
                if (
                    references_by_id[reference_id].get("scope_authority")
                    == RETRIEVED_EXPERT_SCOPE_AUTHORITY
                    and references_by_id[reference_id].get(
                        "support_requires_review"
                    )
                )
            ]
            if provisional_ids:
                frame_usage.append(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "reference_ids": provisional_ids,
                    }
                )
        if frame_usage:
            provisional_supplemental_usage[frame] = frame_usage
    supplemental_evidence_review = {
        "required": bool(provisional_supplemental_usage),
        "usage_by_frame": provisional_supplemental_usage,
        "policy": (
            "Retrieved expert evidence may support candidate factuality but "
            "never target-reference completeness. Reliance on a retrieved "
            "unit whose rewrite still needs review cannot pass "
            "automatically."
        ),
    }
    if supplemental_evidence_review["required"] and status == "pass":
        status = "human_review"
        disagreement_reasons.append(
            "provisional_retrieved_expert_support_requires_review"
        )
    supplemental_validation_record = deepcopy(
        dict(
            supplemental_evidence_validation
            or {
                "status": "not_configured",
                "auto_pass_eligible": True,
                "accepted_evidence": [],
                "accepted_secondary_context": [],
                "rejected_evidence": [],
            }
        )
    )
    if (
        supplemental_validation_record.get("auto_pass_eligible") is not True
        and status == "pass"
    ):
        status = "human_review"
        disagreement_reasons.append(
            "retrieved_expert_evidence_integrity_requires_review"
        )
    review_disclosure_record = deepcopy(
        dict(
            review_disclosure_validation
            or {
                "status": "not_configured",
                "expected_text": "",
                "evidence_ids": [],
                "stripped_from_semantic_candidate": False,
                "auto_pass_eligible": True,
                "failure_reason": None,
                "semantic_candidate_empty": False,
                "additional_disclosure_remains": False,
            }
        )
    )
    if (
        review_disclosure_record.get("auto_pass_eligible") is not True
        and status == "pass"
    ):
        status = "human_review"
        disagreement_reasons.append(
            "review_disclosure_authentication_requires_review"
        )
    # Preserve the answer-content judgment before unresolved curation state
    # adds a separate adjudication requirement. This is the metric aligned
    # with semantic fidelity to the expert claims; it never converts a judge
    # failure or range uncertainty into a pass.
    answer_quality_status = status
    support_only_reference = (
        "no_applicable_direct_required_reference"
        in reference_review_record.get("reasons", [])
    )
    if (
        reference_review_record.get("required")
        and support_only_reference
        and not hard_error_frames
        and scope_evaluation["status"] != "fail"
    ):
        # An explicitly linked other-source/KU recurrence is factuality
        # authority, but it has no question-level completeness scope.  If
        # neither semantic frame nor the range guard found a real error,
        # uncertainty about relationship/coverage belongs in manual review
        # rather than the failure bucket.
        status = "human_review"
        answer_quality_status = "human_review"
        disagreement_reasons.append(
            "support_only_reference_requires_review"
        )
    elif reference_review_record.get("required") and status == "pass":
        status = "human_review"
        disagreement_reasons.append("reference_curation_needs_review")

    evidence = generated_answer.get("evidence") or []
    expert_evidence = [
        item
        for item in evidence
        if item.get("kind") == "expert"
        or item.get("evidence_type") == "expert_annotation"
    ]
    target_source_grounded = (
        generated_answer.get("diagnostics", {}).get(
            "target_source_grounded"
        )
        is True
    )
    pipeline_is_rag_llm = (
        generated_answer.get("generation_mode") == "llm"
        and generated_answer.get("answer_basis") == "retrieved_evidence"
        and bool(expert_evidence)
        and (
            supplemental_evidence_validation is None
            or supplemental_validation_record.get("status")
            == "not_configured"
            or (
                supplemental_validation_record.get(
                    "auto_pass_eligible"
                )
                is True
                and bool(
                    supplemental_validation_record.get(
                        "accepted_evidence"
                    )
                )
            )
        )
        and (
            review_disclosure_validation is None
            or review_disclosure_record.get("status")
            == "not_configured"
            or review_disclosure_record.get("auto_pass_eligible") is True
        )
    )
    reasons = []
    if status != "pass":
        reasons.append(f"semantic_status_{status}")
    if not pipeline_is_rag_llm:
        reasons.append("not_rag_llm_with_expert_evidence")
    return {
        "status": status,
        "weighted_score": round(
            sum(
                result["weighted_score"]
                for result in computed.values()
            )
            / len(computed),
            2,
        ),
        "minimum_frame_score": min(
            result["weighted_score"]
            for result in computed.values()
        ),
        "disagreement": bool(disagreement_reasons),
        "disagreement_reasons": disagreement_reasons,
        "shared_critical_errors": sorted(shared_critical_errors),
        "union_critical_errors": sorted(set().union(
            *(
                set(assessment["critical_error_types"])
                for assessment in assessments.values()
            )
        )),
        "hard_error_frames": hard_error_frames,
        "weak_confidence_frames": weak_confidence_frames,
        "judge_schema_warning_frames": judge_schema_warning_frames,
        "range_scope_evaluation": scope_evaluation,
        "reference_review": reference_review_record,
        "supplemental_evidence_review": (
            supplemental_evidence_review
        ),
        "supplemental_evidence_validation": (
            supplemental_validation_record
        ),
        "review_disclosure_validation": review_disclosure_record,
        "target_source_grounded": target_source_grounded,
        "pipeline_is_grounded_rag_llm": pipeline_is_rag_llm,
        "answer_quality_status": answer_quality_status,
        "answer_quality_rag_llm_pass": (
            answer_quality_status == "pass" and pipeline_is_rag_llm
        ),
        "reliable_rag_llm_pass": status == "pass" and pipeline_is_rag_llm,
        "reliability_failure_reasons": reasons,
    }


def _run_one_judge_frame(
    *,
    frame: str,
    packet: Mapping[str, Any],
    judge_fn: JudgeFunction,
    coverage_signal_fn: CoverageSignalFunction,
    coverage_nli_threshold: float,
    frame_state: dict[str, Any],
    checkpoint: CheckpointFunction,
    max_attempts: int,
) -> bool:
    previous_error: str | None = None
    for _ in range(max_attempts):
        attempt: dict[str, Any] = {"at": utc_now()}
        try:
            raw = judge_fn(
                build_judge_messages(
                    frame,
                    packet,
                    previous_error=previous_error,
                )
            )
            attempt["raw_response"] = raw
            atomic_coverage_signals = coverage_signal_fn(packet)
            attempt["atomic_coverage_signals"] = (
                atomic_coverage_signals
            )
            assessment = validate_judge_assessment(
                raw,
                expected_frame=frame,
                has_measure_range=(
                    packet["selected_measure_range"] is not None
                ),
                candidate_answer=packet["candidate_answer"],
                authoritative_reference_items=(
                    _all_semantic_reference_items(packet)
                ),
                candidate_answer_items=_candidate_answer_items(packet),
                allowed_context_texts=(
                    [packet["question"]]
                    + _authoritative_question_contexts(packet)
                    + _supplemental_question_contexts(packet)
                ),
                atomic_coverage_signals=atomic_coverage_signals,
                atomic_coverage_nli_threshold=coverage_nli_threshold,
            )
            parser_normalizations = [
                item
                for item in assessment["deterministic_normalizations"]
                if item == JSON_APOSTROPHE_ESCAPE_NORMALIZATION
            ]
            if parser_normalizations:
                attempt["parser_normalizations"] = parser_normalizations
        except KeyboardInterrupt:
            frame_state.setdefault("attempts", []).append(attempt)
            checkpoint()
            raise
        except Exception as error:
            attempt["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
            frame_state.setdefault("attempts", []).append(attempt)
            frame_state["status"] = "error"
            previous_error = str(error)
            checkpoint()
            continue

        frame_state.setdefault("attempts", []).append(attempt)
        frame_state.update(
            {
                "status": "complete",
                "assessment": assessment,
                "computed": compute_frame_result(assessment),
                "completed_at": utc_now(),
            }
        )
        checkpoint()
        return True
    return False


def _run_one_range_frame(
    *,
    packet: Mapping[str, Any],
    judge_fn: JudgeFunction,
    range_signal_fn: RangeSignalFunction,
    nli_threshold: float,
    embedding_delta_threshold: float,
    frame_state: dict[str, Any],
    checkpoint: CheckpointFunction,
    max_attempts: int,
) -> bool:
    previous_error: str | None = None
    for _ in range(max_attempts):
        attempt: dict[str, Any] = {"at": utc_now()}
        try:
            raw = judge_fn(
                build_range_judge_messages(
                    packet,
                    previous_error=previous_error,
                )
            )
            attempt["raw_response"] = raw
            hybrid_signals = range_signal_fn(packet)
            attempt["hybrid_signals"] = hybrid_signals
            assessment = validate_range_judge_assessment(
                raw,
                packet=packet,
                hybrid_signals=hybrid_signals,
                nli_threshold=nli_threshold,
                embedding_delta_threshold=(
                    embedding_delta_threshold
                ),
            )
            parser_normalizations = [
                item
                for item in assessment["deterministic_normalizations"]
                if item == JSON_APOSTROPHE_ESCAPE_NORMALIZATION
            ]
            if parser_normalizations:
                attempt["parser_normalizations"] = parser_normalizations
        except KeyboardInterrupt:
            frame_state.setdefault("attempts", []).append(attempt)
            checkpoint()
            raise
        except Exception as error:
            attempt["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
            frame_state.setdefault("attempts", []).append(attempt)
            frame_state["status"] = "error"
            previous_error = str(error)
            checkpoint()
            continue
        frame_state.setdefault("attempts", []).append(attempt)
        frame_state.update(
            {
                "status": "complete",
                "assessment": assessment,
                "completed_at": utc_now(),
            }
        )
        checkpoint()
        return True
    return False


def run_judge_phase(
    snapshot: dict[str, Any],
    *,
    judge_fn: JudgeFunction,
    range_signal_fn: RangeSignalFunction,
    coverage_signal_fn: CoverageSignalFunction,
    nli_threshold: float,
    embedding_delta_threshold: float,
    checkpoint: CheckpointFunction,
    max_attempts: int,
    selected_case_ids: set[str] | None = None,
    rerun: bool = False,
    trusted_expert_catalog: (
        Mapping[str, Mapping[str, Any]] | None
    ) = None,
) -> None:
    phase = "judge"
    phase_state = snapshot["run"]["phases"][phase]
    if phase_state["started_at"] is None:
        phase_state["started_at"] = utc_now()
    phase_state["status"] = "running"
    checkpoint()

    for question, case in all_cases(snapshot):
        if not _selected_case(
            case,
            selected_case_ids=selected_case_ids,
        ):
            continue
        if rerun:
            _invalidate_case_from_phase(case, phase)
            refresh_summary(snapshot)
            checkpoint()
        if (
            case["semantic_evaluation"] is not None
            and case["semantic_evaluation"].get("aggregate") is not None
        ):
            continue
        if case["generated_answer"] is None:
            continue

        packet = _reference_packet(
            question,
            case,
            trusted_expert_catalog=trusted_expert_catalog,
        )
        reference_authority = (
            "supporting_only_range_scoped_expert_content"
            if (
                packet.get("reference_claim_scope") is not None
                and not packet.get(
                    "question_level_reference_scope_applies"
                )
            )
            else (
                "range_scoped_contributing_expert_source_answers"
                if packet["range_disambiguation_applied"]
                and packet["range_scoped_contributing_source_answers"]
                else (
                    "range_applicable_curated_knowledge_unit_answer"
                    if packet[
                        "authoritative_original_expert_answer_withheld_reason"
                    ]
                    else "curator_scoped_verbatim_original_expert_answer"
                )
            )
        )
        semantic = case["semantic_evaluation"]
        range_requirement = range_scope_requirement(packet)
        reference_review = reference_review_requirement(packet)
        if semantic is None:
            semantic = {
                "protocol": JUDGE_PROTOCOL_VERSION,
                "reference_authority": reference_authority,
                "authoritative_reference_items": (
                    _authoritative_reference_items(packet)
                ),
                "supplemental_factuality_items": (
                    _supplemental_factuality_items(packet)
                ),
                "supplemental_evidence_validation": deepcopy(
                    packet["supplemental_evidence_validation"]
                ),
                "review_disclosure_validation": deepcopy(
                    packet["review_disclosure_validation"]
                ),
                "frames": {
                    frame: {"status": "pending", "attempts": []}
                    for frame in FRAME_NAMES
                },
                "range_scope": (
                    {
                        "status": "pending",
                        "attempts": [],
                        "requirement": range_requirement,
                    }
                    if range_requirement["applicable"]
                    else {
                        **range_requirement,
                        "attempts": [],
                        "assessment": {
                            "status": "not_applicable",
                            "applicable": False,
                            "wrong_measure_application": False,
                        },
                    }
                ),
                "reference_review": reference_review,
                "aggregate": None,
            }
            case["semantic_evaluation"] = semantic
            checkpoint()
        else:
            semantic["reference_authority"] = reference_authority
            semantic["reference_review"] = reference_review
            semantic["supplemental_evidence_validation"] = deepcopy(
                packet["supplemental_evidence_validation"]
            )
            semantic["review_disclosure_validation"] = deepcopy(
                packet["review_disclosure_validation"]
            )

        all_frames_complete = True
        for frame in FRAME_NAMES:
            frame_state = semantic["frames"][frame]
            if frame_state.get("status") == "complete":
                continue
            completed = _run_one_judge_frame(
                frame=frame,
                packet=packet,
                judge_fn=judge_fn,
                coverage_signal_fn=coverage_signal_fn,
                coverage_nli_threshold=nli_threshold,
                frame_state=frame_state,
                checkpoint=checkpoint,
                max_attempts=max_attempts,
            )
            if not completed:
                all_frames_complete = False
                case["phase_errors"][phase] = {
                    "at": utc_now(),
                    "type": "JudgeOutputError",
                    "message": (
                        f"{frame} did not return valid structured output "
                        f"within {max_attempts} attempts"
                    ),
                }
                break

        range_state = semantic["range_scope"]
        if (
            all_frames_complete
            and range_requirement["applicable"]
            and range_state.get("status") != "complete"
        ):
            completed = _run_one_range_frame(
                packet=packet,
                judge_fn=judge_fn,
                range_signal_fn=range_signal_fn,
                nli_threshold=nli_threshold,
                embedding_delta_threshold=embedding_delta_threshold,
                frame_state=range_state,
                checkpoint=checkpoint,
                max_attempts=max_attempts,
            )
            if not completed:
                all_frames_complete = False
                case["phase_errors"][phase] = {
                    "at": utc_now(),
                    "type": "JudgeOutputError",
                    "message": (
                        "range_scope did not return valid structured output "
                        f"within {max_attempts} attempts"
                    ),
                }

        if all_frames_complete and all(
            semantic["frames"][frame].get("status") == "complete"
            for frame in FRAME_NAMES
        ) and (
            not range_requirement["applicable"]
            or range_state.get("status") == "complete"
        ):
            semantic["aggregate"] = aggregate_judge_frames(
                semantic["frames"],
                generated_answer=case["generated_answer"],
                range_scope_evaluation=range_state["assessment"],
                reference_review=reference_review,
                supplemental_evidence_validation=packet[
                    "supplemental_evidence_validation"
                ],
                review_disclosure_validation=packet[
                    "review_disclosure_validation"
                ],
            )
            semantic["completed_at"] = utc_now()
            case["phase_errors"].pop(phase, None)
        refresh_summary(snapshot)
        checkpoint()

    refresh_summary(snapshot)
    checkpoint()


def _phase_counts(
    snapshot: Mapping[str, Any],
    phase: str,
) -> tuple[int, int, int]:
    cases = list(all_cases(snapshot))
    field = _phase_field(phase)
    if phase == "judge":
        completed = sum(
            bool(
                case[field]
                and case[field].get("aggregate") is not None
            )
            for _, case in cases
        )
    else:
        completed = sum(case[field] is not None for _, case in cases)
    errors = sum(
        phase in case["phase_errors"]
        for _, case in cases
    )
    return len(cases), completed, errors


def _question_judge_status(question: Mapping[str, Any]) -> str:
    aggregates = [
        case["semantic_evaluation"]["aggregate"]
        for case in question["inference_runs"]
        if (
            case["semantic_evaluation"]
            and case["semantic_evaluation"].get("aggregate")
        )
    ]
    if len(aggregates) != len(question["inference_runs"]):
        return "pending"
    statuses = {aggregate["status"] for aggregate in aggregates}
    if "fail" in statuses:
        return "fail"
    if "human_review" in statuses:
        return "human_review"
    return "pass"


def _question_answer_quality_status(
    question: Mapping[str, Any],
) -> str:
    aggregates = [
        case["semantic_evaluation"]["aggregate"]
        for case in question["inference_runs"]
        if (
            case["semantic_evaluation"]
            and case["semantic_evaluation"].get("aggregate")
        )
    ]
    if len(aggregates) != len(question["inference_runs"]):
        return "pending"
    statuses = {
        aggregate.get(
            "answer_quality_status",
            aggregate["status"],
        )
        for aggregate in aggregates
    }
    if "fail" in statuses:
        return "fail"
    if "human_review" in statuses:
        return "human_review"
    return "pass"


def refresh_summary(snapshot: dict[str, Any]) -> None:
    cases = list(all_cases(snapshot))
    questions = snapshot["results"]
    phase_summary: dict[str, Any] = {}
    for phase in ("retrieval", "generate", "judge"):
        total, completed, errors = _phase_counts(snapshot, phase)
        if completed == total:
            status = "complete"
        elif completed or errors:
            status = "partial"
        else:
            status = "pending"
        phase_state = snapshot["run"]["phases"][phase]
        phase_state["status"] = status
        phase_state["completed_cases"] = completed
        phase_state["error_cases"] = errors
        if status == "complete" and phase_state["finished_at"] is None:
            phase_state["finished_at"] = utc_now()
        elif status != "complete":
            phase_state["finished_at"] = None
        phase_summary[phase] = {
            "status": status,
            "completed_cases": completed,
            "error_cases": errors,
        }

    retrievals = [
        case["retrieval_probe"]
        for _, case in cases
        if case["retrieval_probe"] is not None
    ]
    generations = [
        case["generated_answer"]
        for _, case in cases
        if case["generated_answer"] is not None
    ]
    aggregates = [
        case["semantic_evaluation"]["aggregate"]
        for _, case in cases
        if (
            case["semantic_evaluation"]
            and case["semantic_evaluation"].get("aggregate")
        )
    ]
    judge_statuses = Counter(
        aggregate["status"] for aggregate in aggregates
    )
    question_statuses = Counter(
        _question_judge_status(question) for question in questions
    )
    answer_quality_statuses = Counter(
        aggregate.get("answer_quality_status", aggregate["status"])
        for aggregate in aggregates
    )
    answer_quality_question_statuses = Counter(
        _question_answer_quality_status(question)
        for question in questions
    )
    judged_count = len(aggregates)
    semantic_passes = judge_statuses["pass"]
    rag_llm_passes = sum(
        aggregate["reliable_rag_llm_pass"]
        for aggregate in aggregates
    )
    answer_quality_rag_llm_passes = sum(
        aggregate.get(
            "answer_quality_rag_llm_pass",
            aggregate["reliable_rag_llm_pass"],
        )
        for aggregate in aggregates
    )
    piece_primary = {}
    piece_semantic = {}
    piece_answer_quality = {}
    for piece_id in TARGET_PIECES:
        piece_questions = [
            question
            for question in questions
            if question["piece_id"] == piece_id
        ]
        piece_aggregates = [
            case["semantic_evaluation"]["aggregate"]
            for question, case in cases
            if (
                question["piece_id"] == piece_id
                and case["semantic_evaluation"]
                and case["semantic_evaluation"].get("aggregate")
            )
        ]
        piece_passes = sum(
            aggregate["reliable_rag_llm_pass"]
            for aggregate in piece_aggregates
        )
        piece_case_statuses = Counter(
            aggregate["status"]
            for aggregate in piece_aggregates
        )
        piece_quality_statuses = Counter(
            aggregate.get(
                "answer_quality_status",
                aggregate["status"],
            )
            for aggregate in piece_aggregates
        )
        piece_question_statuses = Counter(
            _question_judge_status(question)
            for question in piece_questions
        )
        piece_quality_question_statuses = Counter(
            _question_answer_quality_status(question)
            for question in piece_questions
        )
        piece_completed_questions = [
            question
            for question in piece_questions
            if _question_judge_status(question) != "pending"
        ]
        piece_rag_llm_question_passes = sum(
            all(
                case["semantic_evaluation"]["aggregate"][
                    "reliable_rag_llm_pass"
                ]
                for case in question["inference_runs"]
            )
            for question in piece_completed_questions
        )
        piece_semantic_question_passes = piece_question_statuses["pass"]
        piece_semantic_case_passes = piece_case_statuses["pass"]
        piece_quality_case_passes = sum(
            aggregate.get(
                "answer_quality_rag_llm_pass",
                aggregate["reliable_rag_llm_pass"],
            )
            for aggregate in piece_aggregates
        )
        piece_quality_completed_questions = [
            question
            for question in piece_questions
            if _question_answer_quality_status(question) != "pending"
        ]
        piece_quality_question_passes = sum(
            all(
                case["semantic_evaluation"]["aggregate"].get(
                    "answer_quality_rag_llm_pass",
                    case["semantic_evaluation"]["aggregate"][
                        "reliable_rag_llm_pass"
                    ],
                )
                for case in question["inference_runs"]
            )
            for question in piece_quality_completed_questions
        )
        piece_case_total = sum(
            len(question["inference_runs"])
            for question in piece_questions
        )
        piece_primary[piece_id] = {
            "inference_cases": piece_case_total,
            "judged_cases": len(piece_aggregates),
            "passed_cases": piece_passes,
            "case_pass_rate": (
                round(piece_passes / piece_case_total, 6)
                if (
                    piece_case_total
                    and len(piece_aggregates) == piece_case_total
                )
                else None
            ),
            "completed_case_pass_rate": (
                round(piece_passes / len(piece_aggregates), 6)
                if piece_aggregates
                else None
            ),
            "end_to_end_case_pass_rate": (
                round(piece_passes / piece_case_total, 6)
                if piece_case_total
                else None
            ),
            "questions": len(piece_questions),
            "completed_questions": len(piece_completed_questions),
            "passed_questions": piece_rag_llm_question_passes,
            "question_pass_rate": (
                round(
                    piece_rag_llm_question_passes
                    / len(piece_questions),
                    6,
                )
                if (
                    piece_questions
                    and len(piece_completed_questions)
                    == len(piece_questions)
                )
                else None
            ),
            "completed_question_pass_rate": (
                round(
                    piece_rag_llm_question_passes
                    / len(piece_completed_questions),
                    6,
                )
                if piece_completed_questions
                else None
            ),
            "end_to_end_question_pass_rate": (
                round(
                    piece_rag_llm_question_passes / len(piece_questions),
                    6,
                )
                if piece_questions
                else None
            ),
        }
        piece_semantic[piece_id] = {
            "case_statuses": dict(sorted(piece_case_statuses.items())),
            "judged_cases": len(piece_aggregates),
            "passed_cases": piece_semantic_case_passes,
            "case_pass_rate": (
                round(
                    piece_semantic_case_passes
                    / len(piece_aggregates),
                    6,
                )
                if piece_aggregates
                else None
            ),
            "question_statuses": dict(
                sorted(piece_question_statuses.items())
            ),
            "completed_questions": len(piece_completed_questions),
            "passed_questions": piece_semantic_question_passes,
            "question_pass_rate": (
                round(
                    piece_semantic_question_passes
                    / len(piece_completed_questions),
                    6,
                )
                if piece_completed_questions
                else None
            ),
        }
        piece_answer_quality[piece_id] = {
            "case_statuses": dict(sorted(piece_quality_statuses.items())),
            "inference_cases": piece_case_total,
            "judged_cases": len(piece_aggregates),
            "passed_cases": piece_quality_case_passes,
            "case_pass_rate": (
                round(piece_quality_case_passes / piece_case_total, 6)
                if (
                    piece_case_total
                    and len(piece_aggregates) == piece_case_total
                )
                else None
            ),
            "completed_case_pass_rate": (
                round(
                    piece_quality_case_passes / len(piece_aggregates),
                    6,
                )
                if piece_aggregates
                else None
            ),
            "end_to_end_case_pass_rate": (
                round(piece_quality_case_passes / piece_case_total, 6)
                if piece_case_total
                else None
            ),
            "question_statuses": dict(
                sorted(piece_quality_question_statuses.items())
            ),
            "questions": len(piece_questions),
            "completed_questions": len(
                piece_quality_completed_questions
            ),
            "passed_questions": piece_quality_question_passes,
            "question_pass_rate": (
                round(
                    piece_quality_question_passes / len(piece_questions),
                    6,
                )
                if (
                    piece_questions
                    and len(piece_quality_completed_questions)
                    == len(piece_questions)
                )
                else None
            ),
            "completed_question_pass_rate": (
                round(
                    piece_quality_question_passes
                    / len(piece_quality_completed_questions),
                    6,
                )
                if piece_quality_completed_questions
                else None
            ),
            "end_to_end_question_pass_rate": (
                round(
                    piece_quality_question_passes / len(piece_questions),
                    6,
                )
                if piece_questions
                else None
            ),
        }
    completed_questions = [
        question
        for question in questions
        if _question_judge_status(question) != "pending"
    ]
    rag_llm_question_passes = sum(
        all(
            case["semantic_evaluation"]["aggregate"][
                "reliable_rag_llm_pass"
            ]
            for case in question["inference_runs"]
        )
        for question in completed_questions
    )
    answer_quality_completed_questions = [
        question
        for question in questions
        if _question_answer_quality_status(question) != "pending"
    ]
    answer_quality_question_passes = sum(
        all(
            case["semantic_evaluation"]["aggregate"].get(
                "answer_quality_rag_llm_pass",
                case["semantic_evaluation"]["aggregate"][
                    "reliable_rag_llm_pass"
                ],
            )
            for case in question["inference_runs"]
        )
        for question in answer_quality_completed_questions
    )

    snapshot["summary"] = {
        "questions": len(questions),
        "inference_cases": len(cases),
        "phases": phase_summary,
        "primary_metric": {
            "name": "semantically_reliable_rag_llm_answer",
            "judged_cases": judged_count,
            "passed_cases": rag_llm_passes,
            "case_pass_rate": (
                round(rag_llm_passes / len(cases), 6)
                if judged_count == len(cases)
                else None
            ),
            "completed_case_pass_rate": (
                round(rag_llm_passes / judged_count, 6)
                if judged_count
                else None
            ),
            "end_to_end_case_pass_rate": round(
                rag_llm_passes / len(cases),
                6,
            ),
            "completed_questions": len(completed_questions),
            "passed_questions": rag_llm_question_passes,
            "question_pass_rate": (
                round(
                    rag_llm_question_passes
                    / len(questions),
                    6,
                )
                if len(completed_questions) == len(questions)
                else None
            ),
            "completed_question_pass_rate": (
                round(
                    rag_llm_question_passes
                    / len(completed_questions),
                    6,
                )
                if completed_questions
                else None
            ),
            "end_to_end_question_pass_rate": round(
                rag_llm_question_passes / len(questions),
                6,
            ),
            "by_piece": piece_primary,
        },
        "answer_quality_metric": {
            "name": "expert_claim_semantic_fidelity_before_curation_gate",
            "statuses": dict(sorted(answer_quality_statuses.items())),
            "judged_cases": judged_count,
            "passed_cases": answer_quality_rag_llm_passes,
            "case_pass_rate": (
                round(answer_quality_rag_llm_passes / len(cases), 6)
                if judged_count == len(cases)
                else None
            ),
            "completed_case_pass_rate": (
                round(answer_quality_rag_llm_passes / judged_count, 6)
                if judged_count
                else None
            ),
            "end_to_end_case_pass_rate": round(
                answer_quality_rag_llm_passes / len(cases),
                6,
            ),
            "question_statuses": dict(
                sorted(answer_quality_question_statuses.items())
            ),
            "completed_questions": len(
                answer_quality_completed_questions
            ),
            "passed_questions": answer_quality_question_passes,
            "question_pass_rate": (
                round(
                    answer_quality_question_passes / len(questions),
                    6,
                )
                if len(answer_quality_completed_questions)
                == len(questions)
                else None
            ),
            "completed_question_pass_rate": (
                round(
                    answer_quality_question_passes
                    / len(answer_quality_completed_questions),
                    6,
                )
                if answer_quality_completed_questions
                else None
            ),
            "end_to_end_question_pass_rate": round(
                answer_quality_question_passes / len(questions),
                6,
            ),
            "by_piece": piece_answer_quality,
        },
        "semantic_judge": {
            "statuses": dict(sorted(judge_statuses.items())),
            "passed_cases": semantic_passes,
            "pass_rate": (
                round(semantic_passes / judged_count, 6)
                if judged_count
                else None
            ),
            "question_statuses": dict(sorted(question_statuses.items())),
            "by_piece": piece_semantic,
        },
        "retrieval_diagnostics": {
            "completed_cases": len(retrievals),
            "target_source_grounded_cases": sum(
                result["diagnostics"]["target_source_grounded"]
                for result in retrievals
            ),
            "any_expected_knowledge_unit_cases": sum(
                result["diagnostics"][
                    "any_expected_knowledge_unit_retrieved"
                ]
                for result in retrievals
            ),
            "all_expected_knowledge_unit_cases": sum(
                result["diagnostics"][
                    "all_expected_knowledge_units_retrieved"
                ]
                for result in retrievals
            ),
            "all_range_applicable_knowledge_unit_cases": sum(
                result["diagnostics"][
                    "all_range_applicable_knowledge_units_retrieved"
                ]
                for result in retrievals
            ),
        },
        "generation_diagnostics": {
            "completed_cases": len(generations),
            "answer_basis": dict(sorted(Counter(
                result.get("answer_basis", "unknown")
                for result in generations
            ).items())),
            "generation_mode": dict(sorted(Counter(
                result.get("generation_mode", "unknown")
                for result in generations
            ).items())),
            "target_source_grounded_cases": sum(
                result["diagnostics"]["target_source_grounded"]
                for result in generations
            ),
        },
    }
    snapshot["run"]["updated_at"] = utc_now()
    if phase_summary["judge"]["status"] == "complete":
        snapshot["run"]["status"] = "complete"
        if snapshot["run"]["finished_at"] is None:
            snapshot["run"]["finished_at"] = utc_now()
    else:
        snapshot["run"]["status"] = "in_progress"
        snapshot["run"]["finished_at"] = None


def validate_resume_snapshot(
    snapshot: Mapping[str, Any],
    *,
    input_fingerprint: str,
) -> None:
    if snapshot.get("schema_version") != SCHEMA_VERSION:
        raise EvaluationInputError(
            "Existing output uses a different schema; choose a new --output"
        )
    existing = snapshot.get("run", {}).get("input_fingerprint")
    if existing != input_fingerprint:
        raise EvaluationInputError(
            "Evaluation inputs changed since this snapshot was created. "
            "Choose a new --output so results from different corpus/model/code "
            "states are not mixed."
        )


def _directory_identity(path: Path) -> str:
    """Hash a model-directory manifest without rereading every weight shard."""

    files = []
    for candidate in sorted(path.rglob("*")):
        if not candidate.is_file():
            continue
        stat = candidate.stat()
        files.append(
            {
                "path": str(candidate.relative_to(path)),
                "size": stat.st_size,
                "symlink_target": (
                    os.readlink(candidate) if candidate.is_symlink() else None
                ),
            }
        )
    manifest = {
        "resolved_path": str(path.resolve()),
        "snapshot_revision": (
            path.name if path.parent.name == "snapshots" else None
        ),
        "files": files,
    }
    return hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()


def _model_record(path: Path, *, backend: str) -> dict[str, Any]:
    exists = path.is_file() or path.is_dir()
    if path.is_file():
        identity = file_sha256(path)
        model_kind = "file"
    elif path.is_dir():
        identity = _directory_identity(path)
        model_kind = "directory"
    else:
        identity = None
        model_kind = "missing"
    return {
        "path": str(path.resolve()),
        "checkpoint_exists": exists,
        "kind": model_kind,
        "backend": backend,
        "sha256": identity,
    }


def _validate_range_embedding_checkpoint(
    model_path: Path,
) -> dict[str, Any]:
    if not model_path.is_dir():
        raise EvaluationInputError(
            f"Range embedding model directory not found: {model_path}"
        )
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise EvaluationInputError(
            f"Range embedding config.json not found: {config_path}"
        )
    config = load_json(config_path)
    expected = {
        "model_type": "qwen3",
        "hidden_size": 1024,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
    }
    mismatches = {
        name: {"expected": value, "actual": config.get(name)}
        for name, value in expected.items()
        if config.get(name) != value
    }
    if "Qwen3ForCausalLM" not in (config.get("architectures") or []):
        mismatches["architectures"] = {
            "expected": ["Qwen3ForCausalLM"],
            "actual": config.get("architectures"),
        }
    if mismatches:
        raise EvaluationInputError(
            "Range embedding checkpoint is not Qwen3-Embedding-0.6B: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )
    return {
        "declared_model": "Qwen/Qwen3-Embedding-0.6B",
        "config_path": str(config_path.resolve()),
        "architecture_checks": expected,
        "pooling": "last_non_padding_token_l2_normalized",
    }


def _validate_range_nli_checkpoint(model_path: Path) -> dict[str, Any]:
    if not model_path.is_dir():
        raise EvaluationInputError(
            f"Range NLI model directory not found: {model_path}"
        )
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise EvaluationInputError(
            f"Range NLI config.json not found: {config_path}"
        )
    config = load_json(config_path)
    id_to_label = {
        str(index): label
        for index, label in enumerate(RANGE_NLI_LABELS)
    }
    label_to_id = {
        label: index
        for index, label in enumerate(RANGE_NLI_LABELS)
    }
    expected = {
        "model_type": "roberta",
        "architectures": ["RobertaForSequenceClassification"],
        "id2label": id_to_label,
        "label2id": label_to_id,
    }
    mismatches = {
        name: {"expected": value, "actual": config.get(name)}
        for name, value in expected.items()
        if config.get(name) != value
    }
    if mismatches:
        raise EvaluationInputError(
            "Range NLI checkpoint has an unexpected KLUE NLI contract: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )
    return {
        "declared_model": "chunwoolee0/klue_nli_roberta_base_model",
        "config_path": str(config_path.resolve()),
        "architecture_checks": expected,
        "label_order": list(RANGE_NLI_LABELS),
        "inference_direction": (
            "premise=candidate_full_text,hypothesis=excluded_atomic_claim"
        ),
    }


def build_range_guard_record(
    *,
    embedding_model_path: Path,
    nli_model_path: Path,
    nli_threshold: float,
    embedding_delta_threshold: float,
) -> dict[str, Any]:
    """Build the exact immutable model and threshold contract."""

    if not 0.0 <= nli_threshold <= 1.0:
        raise EvaluationInputError(
            "Range NLI threshold must be between zero and one"
        )
    if not -2.0 <= embedding_delta_threshold <= 2.0:
        raise EvaluationInputError(
            "Range embedding delta threshold must be between -2 and 2"
        )
    embedding_model_path = embedding_model_path.resolve()
    nli_model_path = nli_model_path.resolve()
    embedding_model = _model_record(
        embedding_model_path,
        backend="transformers-local",
    )
    embedding_model.update(
        _validate_range_embedding_checkpoint(embedding_model_path)
    )
    nli_model = _model_record(
        nli_model_path,
        backend="transformers-local",
    )
    nli_model.update(_validate_range_nli_checkpoint(nli_model_path))
    return {
        "protocol": JUDGE_PROTOCOL_VERSION,
        "local_files_only": True,
        "candidate_text_scope": "full_normalized_generated_answer",
        "atomic_coverage_text_scope": (
            "full_answer_raw_segments_and_question_subject_contextualized_"
            "segments"
        ),
        "embedding_comparison": (
            "cosine(candidate,X)-max_A(cosine(candidate,A))"
        ),
        "embedding_model": embedding_model,
        "nli_model": nli_model,
        "atomic_coverage_guard": {
            "scope": (
                "curator_split_atomic_R_and_every_required_R_when_"
                "supplemental_context_exists"
            ),
            "direction": "premise=candidate,hypothesis=guarded_R",
            "question_context_role": (
                "referent_resolution_only_no_answer_content"
            ),
            "decision": (
                "neutral_or_contradiction_blocks_automatic_pass_only"
            ),
        },
        "thresholds": {
            "nli_entailment_or_contradiction": float(nli_threshold),
            "embedding_contrast_delta": float(
                embedding_delta_threshold
            ),
        },
    }


def build_local_atomic_coverage_signal_scorer(
    *,
    nli_model_path: Path,
) -> CoverageSignalFunction:
    """Return a cached local NLI scorer for guarded R completeness claims."""

    _validate_range_nli_checkpoint(nli_model_path)
    state: dict[str, Any] = {}
    cache: dict[str, Any] = {}
    lock = threading.Lock()

    def load_model() -> None:
        if state:
            return
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            str(nli_model_path),
            local_files_only=True,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            str(nli_model_path),
            torch_dtype="auto",
            device_map="auto",
            local_files_only=True,
        )
        model.eval()
        state.update(
            {
                "torch": torch,
                "tokenizer": tokenizer,
                "model": model,
            }
        )

    def score(packet: Mapping[str, Any]) -> dict[str, Any]:
        references = _coverage_guard_reference_items(
            _all_semantic_reference_items(packet)
        )
        candidates = _candidate_answer_items(packet)
        if not references:
            return {"atomic_claim_signals": []}
        candidate_answer = normalize_candidate_answer_for_judge(
            packet["candidate_answer"]
        )
        if not candidate_answer or not candidates:
            raise JudgeOutputError(
                "atomic coverage guard requires candidate answer items"
            )
        contextualized_candidates = [
            {
                "candidate_id": candidate["candidate_id"],
                "context_text": _candidate_with_question_subject_context(
                    packet["question"],
                    candidate["text"],
                ),
            }
            for candidate in candidates
        ]
        cache_key = hashlib.sha256(
            canonical_json_bytes(
                {
                    "candidate_answer": candidate_answer,
                    "candidate_items": candidates,
                    "contextualized_candidate_items": (
                        contextualized_candidates
                    ),
                    "guarded_references": references,
                }
            )
        ).hexdigest()
        with lock:
            if cache.get("key") == cache_key:
                return deepcopy(cache["value"])
            load_model()
            torch = state["torch"]
            tokenizer = state["tokenizer"]
            model = state["model"]
            premises = []
            hypotheses = []
            for reference in references:
                premises.append(candidate_answer)
                hypotheses.append(reference["text"])
                for candidate in candidates:
                    premises.append(candidate["text"])
                    hypotheses.append(reference["text"])
                for candidate in contextualized_candidates:
                    premises.append(candidate["context_text"])
                    hypotheses.append(reference["text"])
            inputs = tokenizer(
                premises,
                hypotheses,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
                return_token_type_ids=False,
            )
            device = next(model.parameters()).device
            inputs = {
                name: tensor.to(device)
                for name, tensor in inputs.items()
            }
            with torch.inference_mode():
                probabilities = torch.softmax(
                    model(**inputs).logits.float(),
                    dim=-1,
                ).detach().cpu().tolist()
            stride = 1 + 2 * len(candidates)
            values = []
            for reference_index, reference in enumerate(references):
                offset = reference_index * stride

                def nli_record(row: Sequence[float]) -> dict[str, float]:
                    return {
                        label: round(float(row[index]), 10)
                        for index, label in enumerate(RANGE_NLI_LABELS)
                    }

                values.append(
                    {
                        "reference_id": reference["reference_id"],
                        "full_answer_nli": nli_record(
                            probabilities[offset]
                        ),
                        "candidate_nli": [
                            {
                                "candidate_id": candidate["candidate_id"],
                                **nli_record(
                                    probabilities[
                                        offset + candidate_index + 1
                                    ]
                                ),
                            }
                            for candidate_index, candidate in enumerate(
                                candidates
                            )
                        ],
                        "contextualized_candidate_nli": [
                            {
                                "candidate_id": candidate["candidate_id"],
                                "context_text": candidate["context_text"],
                                **nli_record(
                                    probabilities[
                                        offset
                                        + len(candidates)
                                        + candidate_index
                                        + 1
                                    ]
                                ),
                            }
                            for candidate_index, candidate in enumerate(
                                contextualized_candidates
                            )
                        ],
                    }
                )
            result = {"atomic_claim_signals": values}
            cache.update(key=cache_key, value=deepcopy(result))
            return result

    return score


def build_local_range_signal_scorer(
    *,
    embedding_model_path: Path,
    nli_model_path: Path,
) -> RangeSignalFunction:
    """Return a lazy, local-only embedding and KLUE-NLI signal scorer."""

    _validate_range_embedding_checkpoint(embedding_model_path)
    _validate_range_nli_checkpoint(nli_model_path)
    state: dict[str, Any] = {}
    lock = threading.Lock()

    def load_models() -> None:
        if state:
            return
        import torch
        from transformers import (
            AutoModel,
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        embedding_tokenizer = AutoTokenizer.from_pretrained(
            str(embedding_model_path),
            local_files_only=True,
        )
        if embedding_tokenizer.pad_token_id is None:
            embedding_tokenizer.pad_token_id = (
                embedding_tokenizer.eos_token_id
            )
        embedding_tokenizer.padding_side = "left"
        embedding_model = AutoModel.from_pretrained(
            str(embedding_model_path),
            torch_dtype="auto",
            device_map="auto",
            local_files_only=True,
        )
        embedding_model.eval()
        nli_tokenizer = AutoTokenizer.from_pretrained(
            str(nli_model_path),
            local_files_only=True,
        )
        nli_model = AutoModelForSequenceClassification.from_pretrained(
            str(nli_model_path),
            torch_dtype="auto",
            device_map="auto",
            local_files_only=True,
        )
        nli_model.eval()
        state.update(
            {
                "torch": torch,
                "embedding_tokenizer": embedding_tokenizer,
                "embedding_model": embedding_model,
                "nli_tokenizer": nli_tokenizer,
                "nli_model": nli_model,
            }
        )

    def score(packet: Mapping[str, Any]) -> dict[str, Any]:
        with lock:
            load_models()
            torch = state["torch"]
            embedding_tokenizer = state["embedding_tokenizer"]
            embedding_model = state["embedding_model"]
            nli_tokenizer = state["nli_tokenizer"]
            nli_model = state["nli_model"]
            candidate = normalize_candidate_answer_for_judge(
                packet["candidate_answer"]
            )
            candidate_items = _candidate_answer_items(packet)
            allowed_items = _allowed_range_contrast_items(packet)
            excluded_items = _excluded_reference_items(packet)
            if (
                not candidate
                or not candidate_items
                or not allowed_items
                or not excluded_items
            ):
                raise JudgeOutputError(
                    "hybrid range guard requires candidate, A, and X text"
                )

            embedding_texts = [
                candidate,
                *(item["text"] for item in allowed_items),
                *(item["text"] for item in excluded_items),
            ]
            embedding_inputs = embedding_tokenizer(
                embedding_texts,
                padding=True,
                truncation=True,
                max_length=8192,
                return_tensors="pt",
            )
            embedding_device = next(
                embedding_model.parameters()
            ).device
            embedding_inputs = {
                name: tensor.to(embedding_device)
                for name, tensor in embedding_inputs.items()
            }
            with torch.inference_mode():
                hidden = embedding_model(
                    **embedding_inputs,
                ).last_hidden_state
            mask = embedding_inputs["attention_mask"]
            if bool(torch.all(mask[:, -1] == 1)):
                pooled = hidden[:, -1]
            else:
                sequence_lengths = mask.sum(dim=1) - 1
                pooled = hidden[
                    torch.arange(
                        hidden.shape[0],
                        device=hidden.device,
                    ),
                    sequence_lengths,
                ]
            pooled = torch.nn.functional.normalize(
                pooled.float(),
                p=2,
                dim=1,
            )
            similarities = (
                pooled[1:] @ pooled[0]
            ).detach().cpu().tolist()
            allowed_similarities = similarities[:len(allowed_items)]
            excluded_similarities = similarities[len(allowed_items):]

            nli_premises = [candidate] * len(excluded_items)
            nli_hypotheses = [
                item["text"] for item in excluded_items
            ]
            nli_premises.extend(
                candidate_item["text"]
                for excluded in excluded_items
                for candidate_item in candidate_items
            )
            nli_hypotheses.extend(
                excluded["text"]
                for excluded in excluded_items
                for _ in candidate_items
            )
            nli_inputs = nli_tokenizer(
                nli_premises,
                nli_hypotheses,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
                return_token_type_ids=False,
            )
            nli_device = next(nli_model.parameters()).device
            nli_inputs = {
                name: tensor.to(nli_device)
                for name, tensor in nli_inputs.items()
            }
            with torch.inference_mode():
                probabilities = torch.softmax(
                    nli_model(**nli_inputs).logits.float(),
                    dim=-1,
                ).detach().cpu().tolist()

            results = []
            for excluded_index, excluded in enumerate(excluded_items):
                allowed_scores = [
                    {
                        "allowed_id": item["reference_id"],
                        "cosine_similarity": round(
                            float(allowed_similarities[index]),
                            10,
                        ),
                    }
                    for index, item in enumerate(allowed_items)
                ]
                best = max(
                    allowed_scores,
                    key=lambda item: item["cosine_similarity"],
                )
                excluded_similarity = round(
                    float(excluded_similarities[excluded_index]),
                    10,
                )
                best_similarity = best["cosine_similarity"]
                results.append(
                    {
                        "excluded_id": excluded["reference_id"],
                        "embedding": {
                            "candidate_vs_excluded": excluded_similarity,
                            "allowed_similarities": allowed_scores,
                            "best_allowed_id": best["allowed_id"],
                            "candidate_vs_best_allowed": best_similarity,
                            "contrast_delta": round(
                                excluded_similarity - best_similarity,
                                10,
                            ),
                        },
                        "nli": {
                            label: round(
                                float(
                                    probabilities[excluded_index][index]
                                ),
                                10,
                            )
                            for index, label in enumerate(RANGE_NLI_LABELS)
                        },
                        "candidate_nli": [
                            {
                                "candidate_id": candidate_item[
                                    "candidate_id"
                                ],
                                **{
                                    label: round(
                                        float(
                                            probabilities[
                                                len(excluded_items)
                                                + excluded_index
                                                * len(candidate_items)
                                                + candidate_index
                                            ][label_index]
                                        ),
                                        10,
                                    )
                                    for label_index, label in enumerate(
                                        RANGE_NLI_LABELS
                                    )
                                },
                            }
                            for candidate_index, candidate_item in enumerate(
                                candidate_items
                            )
                        ],
                    }
                )
            return {"excluded_claim_signals": results}

    return score


def validate_judge_calibration(
    path: Path,
    *,
    judge_model: Mapping[str, Any],
    range_guard: Mapping[str, Any],
    dataset_root: Path,
    judge_max_tokens: int,
) -> dict[str, Any]:
    """Reconstruct and verify the complete calibration contract and results."""

    if not path.is_file():
        raise EvaluationInputError(
            f"Judge calibration file not found: {path}"
        )
    from evaluation import calibrate_judge as calibration

    artifact = load_json(path)
    run = artifact.get("run") or {}
    summary = artifact.get("summary") or {}
    controls = artifact.get("controls")
    failures = []
    detail: dict[str, Any] = {}
    try:
        expected = calibration.expected_calibration_inputs(
            dataset_root=dataset_root.resolve(),
            judge_model_path=Path(judge_model["path"]).resolve(),
            range_embedding_model_path=Path(
                range_guard["embedding_model"]["path"]
            ).resolve(),
            range_nli_model_path=Path(
                range_guard["nli_model"]["path"]
            ).resolve(),
            range_nli_threshold=range_guard["thresholds"][
                "nli_entailment_or_contradiction"
            ],
            range_embedding_delta_threshold=range_guard["thresholds"][
                "embedding_contrast_delta"
            ],
            max_tokens=judge_max_tokens,
        )
    except Exception as error:
        raise EvaluationInputError(
            "Cannot reconstruct judge calibration contract: "
            f"{type(error).__name__}: {error}"
        ) from error

    if artifact.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
        failures.append("calibration_schema_mismatch")
    if (
        run.get("control_contract_version")
        != CALIBRATION_CONTRACT_VERSION
    ):
        failures.append("calibration_contract_mismatch")
    if run.get("status") != "complete":
        failures.append("calibration_run_not_complete")
    if summary.get("quality_gate_passed") is not True:
        failures.append("calibration_quality_gate_failed")
    expected_count = CALIBRATION_EXPECTED_CONTROL_COUNT
    if summary.get("total_controls") != expected_count:
        failures.append("calibration_control_count_mismatch")
    if summary.get("completed_controls") != expected_count:
        failures.append("calibration_controls_incomplete")
    if summary.get("matched_controls") != expected_count:
        failures.append("calibration_controls_not_all_matched")
    if summary.get("required_matches") != expected_count:
        failures.append("calibration_required_match_count_mismatch")
    if summary.get("accuracy") != 1.0:
        failures.append("calibration_accuracy_not_one")
    if summary.get("mismatched_control_ids") not in ([], ()):
        failures.append("calibration_reports_mismatches")
    if summary.get("frame_errors") != 0:
        failures.append("calibration_has_frame_errors")
    if run.get("runner_judge_protocol") != JUDGE_PROTOCOL_VERSION:
        failures.append("calibration_protocol_mismatch")
    if run.get("dataset_root") != str(dataset_root.resolve()):
        failures.append("calibration_dataset_root_mismatch")
    if run.get("judge_backend") != "transformers":
        failures.append("calibration_backend_mismatch")
    if run.get("judge_max_tokens") != judge_max_tokens:
        failures.append("calibration_max_tokens_mismatch")
    if run.get("runtime_versions") != expected["runtime_versions"]:
        failures.append("calibration_runtime_mismatch")
    calibrated_model = run.get("judge_model") or {}
    if calibrated_model != expected["judge_model"]:
        failures.append("calibration_model_mismatch")
    if judge_model.get("sha256") != expected["judge_model"].get("sha256"):
        failures.append("evaluation_judge_model_mismatch")
    if calibrated_model.get("distinct_from_generator") is not True:
        failures.append("calibration_judge_not_distinct_from_generator")
    if run.get("range_guard") != expected["range_guard"]:
        failures.append("calibration_range_guard_mismatch")
    if dict(range_guard) != expected["range_guard"]:
        failures.append("evaluation_range_guard_mismatch")
    if run.get("frames") != list(FRAME_NAMES):
        failures.append("calibration_frame_contract_mismatch")
    if run.get("input_files") != expected["input_files"]:
        failures.append("calibration_input_manifest_mismatch")
    if run.get("input_fingerprint") != expected["input_fingerprint"]:
        failures.append("calibration_input_fingerprint_mismatch")

    expected_controls = expected["controls"]
    expected_ids = [
        control["control_id"] for control in expected_controls
    ]
    if not isinstance(controls, list) or [
        control.get("control_id") if isinstance(control, dict) else None
        for control in (controls or [])
    ] != expected_ids:
        failures.append("calibration_control_records_invalid")
        controls = []

    contract_fields = (
        "control_id",
        "description",
        "source_id",
        "piece_id",
        "question",
        "selected_measure_range",
        "candidate_answer",
        "expected",
        "reference_packet",
        "generated_answer",
        "reference_review",
    )
    control_failures = []
    if controls:
        for expected_control, control in zip(expected_controls, controls):
            control_id = expected_control["control_id"]
            if any(
                control.get(field) != expected_control.get(field)
                for field in contract_fields
            ):
                control_failures.append(
                    f"{control_id}:contract_mismatch"
                )
                continue
            frame_states = control.get("frames")
            if (
                not isinstance(frame_states, dict)
                or set(frame_states) != set(FRAME_NAMES)
            ):
                control_failures.append(
                    f"{control_id}:frame_set_invalid"
                )
                continue
            for frame in FRAME_NAMES:
                state = frame_states[frame]
                attempts = state.get("attempts")
                successful = [
                    attempt
                    for attempt in (attempts or [])
                    if (
                        isinstance(attempt, dict)
                        and isinstance(
                            attempt.get("raw_response"),
                            str,
                        )
                        and isinstance(
                            attempt.get("atomic_coverage_signals"),
                            dict,
                        )
                        and "error" not in attempt
                    )
                ]
                if state.get("status") != "complete" or not successful:
                    control_failures.append(
                        f"{control_id}:{frame}:not_complete"
                    )
                    continue
                try:
                    recomputed_assessment = validate_judge_assessment(
                        successful[-1]["raw_response"],
                        expected_frame=frame,
                        has_measure_range=(
                            control["reference_packet"][
                                "selected_measure_range"
                            ]
                            is not None
                        ),
                        candidate_answer=control["reference_packet"][
                            "candidate_answer"
                        ],
                        authoritative_reference_items=(
                            _all_semantic_reference_items(
                                control["reference_packet"]
                            )
                        ),
                        candidate_answer_items=_candidate_answer_items(
                            control["reference_packet"]
                        ),
                        allowed_context_texts=(
                            [control["reference_packet"]["question"]]
                            + _authoritative_question_contexts(
                                control["reference_packet"]
                            )
                            + _supplemental_question_contexts(
                                control["reference_packet"]
                            )
                        ),
                        atomic_coverage_signals=successful[-1][
                            "atomic_coverage_signals"
                        ],
                        atomic_coverage_nli_threshold=expected[
                            "range_guard"
                        ]["thresholds"][
                            "nli_entailment_or_contradiction"
                        ],
                    )
                except Exception:
                    control_failures.append(
                        f"{control_id}:{frame}:raw_response_invalid"
                    )
                    continue
                if state.get("assessment") != recomputed_assessment:
                    control_failures.append(
                        f"{control_id}:{frame}:assessment_mismatch"
                    )
                if state.get("computed") != compute_frame_result(
                    recomputed_assessment
                ):
                    control_failures.append(
                        f"{control_id}:{frame}:computed_mismatch"
                    )
            range_requirement = range_scope_requirement(
                control["reference_packet"]
            )
            range_state = control.get("range_scope") or {}
            if range_requirement["applicable"]:
                attempts = range_state.get("attempts")
                successful = [
                    attempt
                    for attempt in (attempts or [])
                    if (
                        isinstance(attempt, dict)
                        and isinstance(
                            attempt.get("raw_response"),
                            str,
                        )
                        and isinstance(
                            attempt.get("hybrid_signals"),
                            dict,
                        )
                        and "error" not in attempt
                    )
                ]
                if (
                    range_state.get("status") != "complete"
                    or not successful
                ):
                    control_failures.append(
                        f"{control_id}:range_scope:not_complete"
                    )
                    continue
                try:
                    range_assessment = validate_range_judge_assessment(
                        successful[-1]["raw_response"],
                        packet=control["reference_packet"],
                        hybrid_signals=successful[-1][
                            "hybrid_signals"
                        ],
                        nli_threshold=expected["range_guard"][
                            "thresholds"
                        ]["nli_entailment_or_contradiction"],
                        embedding_delta_threshold=expected[
                            "range_guard"
                        ]["thresholds"]["embedding_contrast_delta"],
                    )
                except Exception:
                    control_failures.append(
                        f"{control_id}:range_scope:raw_response_invalid"
                    )
                    continue
                if range_state.get("assessment") != range_assessment:
                    control_failures.append(
                        f"{control_id}:range_scope:assessment_mismatch"
                    )
            else:
                range_assessment = {
                    "status": "not_applicable",
                    "applicable": False,
                    "wrong_measure_application": False,
                }
                if range_state.get("assessment") != range_assessment:
                    control_failures.append(
                        f"{control_id}:range_scope:not_applicable_mismatch"
                    )
            if any(
                state.get("status") != "complete"
                for state in frame_states.values()
            ):
                continue
            recomputed_aggregate = aggregate_judge_frames(
                frame_states,
                generated_answer=control["generated_answer"],
                range_scope_evaluation=range_assessment,
                reference_review=control["reference_review"],
                supplemental_evidence_validation=control[
                    "reference_packet"
                ]["supplemental_evidence_validation"],
                review_disclosure_validation=control[
                    "reference_packet"
                ]["review_disclosure_validation"],
            )
            if control.get("aggregate") != recomputed_aggregate:
                control_failures.append(
                    f"{control_id}:aggregate_mismatch"
                )
                continue
            recomputed_expectation = (
                calibration.evaluate_control_expectation(control)
            )
            if (
                control.get("expectation_evaluation")
                != recomputed_expectation
                or recomputed_expectation.get("match") is not True
            ):
                control_failures.append(
                    f"{control_id}:expectation_mismatch"
                )
        if control_failures:
            failures.append("calibration_control_results_invalid")
            detail["control_failures"] = control_failures

    if controls:
        recomputed_snapshot = deepcopy(artifact)
        calibration.refresh_summary(recomputed_snapshot)
        if recomputed_snapshot.get("summary") != summary:
            failures.append("calibration_summary_mismatch")

    stale_files = []
    input_records = run.get("input_files")
    if not isinstance(input_records, list) or not input_records:
        failures.append("calibration_input_manifest_missing")
        input_records = []
    for record in input_records:
        if not isinstance(record, dict):
            stale_files.append("<invalid-record>")
            continue
        input_path = Path(record.get("path", ""))
        if (
            not input_path.is_file()
            or file_sha256(input_path) != record.get("sha256")
        ):
            stale_files.append(str(input_path))
    if stale_files:
        failures.append("calibration_inputs_changed")
    if failures:
        detail.update(
            {
                "failures": failures,
                "stale_input_files": stale_files,
            }
        )
        raise EvaluationInputError(
            "Judge calibration is not valid for this run: "
            + json.dumps(detail, ensure_ascii=False, sort_keys=True)
        )
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "input_fingerprint": run.get("input_fingerprint"),
        "control_contract_version": run.get(
            "control_contract_version"
        ),
        "total_controls": summary.get("total_controls"),
        "matched_controls": summary.get("matched_controls"),
        "quality_gate_passed": True,
    }


def build_transformers_judge(
    model_path: Path,
    *,
    max_tokens: int,
) -> JudgeFunction:
    """Return a lazy Transformers judge using Qwen's non-thinking template."""

    state: dict[str, Any] = {}
    lock = threading.Lock()

    def generate(messages: list[dict[str, str]]) -> str:
        with lock:
            if not state:
                import torch
                from transformers import (
                    AutoModelForCausalLM,
                    AutoTokenizer,
                )

                tokenizer = AutoTokenizer.from_pretrained(
                    str(model_path),
                )
                model = AutoModelForCausalLM.from_pretrained(
                    str(model_path),
                    torch_dtype="auto",
                    device_map="auto",
                )
                model.eval()
                state.update(
                    {
                        "torch": torch,
                        "tokenizer": tokenizer,
                        "model": model,
                    }
                )

            torch = state["torch"]
            tokenizer = state["tokenizer"]
            model = state["model"]
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            model_inputs = tokenizer(
                [rendered],
                return_tensors="pt",
            ).to(model.device)
            with torch.inference_mode():
                output_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated_ids = output_ids[
                :,
                model_inputs.input_ids.shape[1]:,
            ]
            text = tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )[0]
            text = re.sub(
                r"<think>.*?</think>",
                "",
                text,
                flags=re.DOTALL | re.IGNORECASE,
            )
            return re.sub(
                r"</?think>",
                "",
                text,
                flags=re.IGNORECASE,
            ).strip()

    return generate


def _resolve_selected_cases(
    snapshot: Mapping[str, Any],
    requested: Sequence[str],
    *,
    limit: int | None,
) -> set[str] | None:
    known = [case["case_id"] for _, case in all_cases(snapshot)]
    unknown = set(requested) - set(known)
    if unknown:
        raise EvaluationInputError(
            f"Unknown --case-id values: {sorted(unknown)}"
        )
    if requested:
        selected = [case_id for case_id in known if case_id in requested]
    else:
        selected = known
    if limit is not None:
        selected = selected[:limit]
    if len(selected) == len(known):
        return None
    return set(selected)


def _load_runtime_settings() -> dict[str, Any]:
    from soprano_qa.settings import load_settings

    return load_settings(use_legacy_dataset_env=False)


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    settings = _load_runtime_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("retrieval", "generate", "judge", "all"),
        default="all",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(settings["dataset_root"]),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Run only this stable inference case ID; repeat as needed.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Run only the first N selected cases (for smoke tests).",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Explicitly rerun the selected phase and invalidate downstream "
        "case results.",
    )
    parser.add_argument(
        "--judge-model-path",
        type=Path,
        default=Path(settings["model_path"]),
    )
    parser.add_argument(
        "--judge-calibration",
        type=Path,
        default=DEFAULT_JUDGE_CALIBRATION,
        help="Completed calibration artifact required before judging.",
    )
    parser.add_argument(
        "--judge-backend",
        choices=("auto", "llama-cpp", "transformers"),
        default="auto",
        help="`auto` uses Transformers for a model directory and llama.cpp "
        "for a GGUF file.",
    )
    parser.add_argument("--judge-max-attempts", type=int, default=4)
    parser.add_argument("--judge-max-tokens", type=int, default=2048)
    parser.add_argument(
        "--range-embedding-model-path",
        type=Path,
        default=Path(os.environ.get(
            "SOPRANO_QA_RANGE_EMBEDDING_MODEL_PATH",
            str(DEFAULT_RANGE_EMBEDDING_MODEL_PATH),
        )),
        help="Local-only Qwen3-Embedding-0.6B snapshot.",
    )
    parser.add_argument(
        "--range-nli-model-path",
        type=Path,
        default=Path(os.environ.get(
            "SOPRANO_QA_RANGE_NLI_MODEL_PATH",
            str(DEFAULT_RANGE_NLI_MODEL_PATH),
        )),
        help="Local-only Korean KLUE NLI snapshot.",
    )
    parser.add_argument(
        "--range-nli-threshold",
        type=float,
        default=float(os.environ.get(
            "SOPRANO_QA_RANGE_NLI_THRESHOLD",
            str(DEFAULT_RANGE_NLI_THRESHOLD),
        )),
    )
    parser.add_argument(
        "--range-embedding-delta-threshold",
        type=float,
        default=float(os.environ.get(
            "SOPRANO_QA_RANGE_EMBEDDING_DELTA_THRESHOLD",
            str(DEFAULT_RANGE_EMBEDDING_DELTA_THRESHOLD),
        )),
    )
    parser.add_argument(
        "--minimum-pass-rate",
        type=float,
        default=0.80,
        help="Exit 2 after a complete judge phase below this grounded "
        "RAG+LLM case pass rate.",
    )
    arguments = parser.parse_args(argv)
    if arguments.top_k < 1:
        parser.error("--top-k must be positive")
    if arguments.limit is not None and arguments.limit < 1:
        parser.error("--limit must be positive")
    if arguments.judge_max_attempts < 1:
        parser.error("--judge-max-attempts must be positive")
    if arguments.judge_max_tokens < 128:
        parser.error("--judge-max-tokens must be at least 128")
    if not 0 <= arguments.range_nli_threshold <= 1:
        parser.error("--range-nli-threshold must be between 0 and 1")
    if not -2 <= arguments.range_embedding_delta_threshold <= 2:
        parser.error(
            "--range-embedding-delta-threshold must be between -2 and 2"
        )
    if not 0 <= arguments.minimum_pass_rate <= 1:
        parser.error("--minimum-pass-rate must be between 0 and 1")
    return arguments


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    dataset_root = arguments.dataset_root.resolve()
    output = arguments.output.resolve()
    settings = _load_runtime_settings()
    generator_model_path = Path(settings["model_path"])
    judge_model_path = arguments.judge_model_path.resolve()
    judge_backend = arguments.judge_backend
    if judge_backend == "auto":
        judge_backend = (
            "transformers"
            if judge_model_path.is_dir()
            else "llama-cpp"
        )
    questions, dataset_paths = load_evaluation_questions(dataset_root)

    implementation_paths = [
        PROJECT_ROOT / "soprano_qa" / filename
        for filename in (
            "answer.py",
            "corpus.py",
            "llm.py",
            "retrieval.py",
            "service.py",
            "settings.py",
        )
    ]
    implementation_paths.extend(
        (
            Path(__file__).resolve(),
            PROJECT_ROOT / "config" / "settings.json",
        )
    )
    corpus_paths = [
        Path(settings["corpus_path"]),
        Path(settings["stats_path"]),
    ]
    trusted_expert_catalog = load_trusted_expert_evidence_catalog(
        corpus_paths[0]
    )
    input_files = input_file_records(
        [*dataset_paths, *implementation_paths, *corpus_paths]
    )
    generator_model = _model_record(
        generator_model_path,
        backend="llama-cpp",
    )
    generator_runtime_settings = deepcopy(settings)
    # The explicit evaluation argument is what the service actually receives
    # through SOPRANO_QA_RAG_DATASET_ROOT below, even when the repository
    # settings file names another default dataset.
    generator_runtime_settings["dataset_root"] = str(dataset_root)
    generator_model["runtime_settings"] = deepcopy(
        generator_runtime_settings
    )
    judge_model = _model_record(
        judge_model_path,
        backend=judge_backend,
    )
    range_guard = build_range_guard_record(
        embedding_model_path=(
            arguments.range_embedding_model_path.resolve()
        ),
        nli_model_path=arguments.range_nli_model_path.resolve(),
        nli_threshold=arguments.range_nli_threshold,
        embedding_delta_threshold=(
            arguments.range_embedding_delta_threshold
        ),
    )
    fingerprint = build_input_fingerprint(
        input_files=input_files,
        top_k=arguments.top_k,
        generator_runtime_settings=generator_runtime_settings,
        generator_model_sha256=generator_model["sha256"],
        judge_model_sha256=judge_model["sha256"],
        judge_backend=judge_backend,
        range_guard=range_guard,
        judge_calibration_sha256=(
            file_sha256(arguments.judge_calibration.resolve())
            if arguments.judge_calibration.resolve().is_file()
            else None
        ),
    )

    with exclusive_output_lock(output):
        if output.exists():
            snapshot = load_json(output)
            validate_resume_snapshot(
                snapshot,
                input_fingerprint=fingerprint,
            )
        else:
            snapshot = new_snapshot(
                questions=questions,
                dataset_root=dataset_root,
                input_files=input_files,
                input_fingerprint=fingerprint,
                top_k=arguments.top_k,
                generator_model=generator_model,
                judge_model=judge_model,
                range_guard=range_guard,
            )
            atomic_write_json(output, snapshot)

        selected_case_ids = _resolve_selected_cases(
            snapshot,
            arguments.case_id,
            limit=arguments.limit,
        )

        def checkpoint() -> None:
            refresh_summary(snapshot)
            atomic_write_json(output, snapshot)

        phases = (
            ("retrieval", "generate", "judge")
            if arguments.phase == "all"
            else (arguments.phase,)
        )
        if any(phase in {"retrieval", "generate"} for phase in phases):
            os.environ["SOPRANO_QA_RAG_DATASET_ROOT"] = str(dataset_root)
            from soprano_qa.service import ask

            for phase in phases:
                if phase not in {"retrieval", "generate"}:
                    continue
                run_pipeline_phase(
                    snapshot,
                    phase=phase,
                    ask_fn=ask,
                    checkpoint=checkpoint,
                    top_k=arguments.top_k,
                    selected_case_ids=selected_case_ids,
                    rerun=arguments.rerun,
                )

        if "judge" in phases:
            if not judge_model["checkpoint_exists"]:
                raise FileNotFoundError(
                    f"Judge model not found: {judge_model_path}"
                )
            calibration_record = validate_judge_calibration(
                arguments.judge_calibration.resolve(),
                judge_model=judge_model,
                range_guard=range_guard,
                dataset_root=dataset_root,
                judge_max_tokens=arguments.judge_max_tokens,
            )
            snapshot["run"]["judge_calibration"] = calibration_record
            checkpoint()
            if judge_backend == "transformers":
                judge_fn = build_transformers_judge(
                    judge_model_path,
                    max_tokens=arguments.judge_max_tokens,
                )
            else:
                from soprano_qa.llm import generate as generate_llm

                judge_settings = {
                    **settings["llm"],
                    "temperature": 0.0,
                    "max_tokens": arguments.judge_max_tokens,
                }

                def judge_fn(messages: list[dict[str, str]]) -> str:
                    return generate_llm(
                        str(judge_model_path),
                        messages,
                        judge_settings,
                    )

            range_signal_fn = build_local_range_signal_scorer(
                embedding_model_path=(
                    arguments.range_embedding_model_path.resolve()
                ),
                nli_model_path=(
                    arguments.range_nli_model_path.resolve()
                ),
            )
            coverage_signal_fn = (
                build_local_atomic_coverage_signal_scorer(
                    nli_model_path=(
                        arguments.range_nli_model_path.resolve()
                    ),
                )
            )
            run_judge_phase(
                snapshot,
                judge_fn=judge_fn,
                range_signal_fn=range_signal_fn,
                coverage_signal_fn=coverage_signal_fn,
                nli_threshold=arguments.range_nli_threshold,
                embedding_delta_threshold=(
                    arguments.range_embedding_delta_threshold
                ),
                checkpoint=checkpoint,
                max_attempts=arguments.judge_max_attempts,
                selected_case_ids=selected_case_ids,
                rerun=arguments.rerun,
                trusted_expert_catalog=trusted_expert_catalog,
            )

        checkpoint()

    summary = snapshot["summary"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    judge_phase = summary["phases"]["judge"]
    if judge_phase["status"] == "complete":
        pass_rate = (
            summary["answer_quality_metric"]["case_pass_rate"] or 0.0
        )
        if pass_rate < arguments.minimum_pass_rate:
            print(
                "Quality gate failed: expert-claim semantic-fidelity "
                "RAG+LLM case pass rate "
                f"{pass_rate:.1%} < {arguments.minimum_pass_rate:.1%}",
                file=sys.stderr,
            )
            return 2
    if any(
        summary["phases"][phase]["error_cases"]
        for phase in phases
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
