#!/usr/bin/env python3
"""Judge five-piece answers for qualitative expert-answer fidelity.

This is a lightweight LLM-as-a-judge pass over
``five_piece_qualitative.json``.  It is deliberately *not* the strict
schema-1.3 automatic-pass protocol implemented by
``run_question_evaluation.py``.  Its pass/review/fail labels are conservative
qualitative triage for later human inspection.

The judge sees the generated answer, evaluation question, selected measure
range, exact annotator answer, and linked knowledge-unit answers.  Retrieval
IDs and retrieval-hit diagnostics are intentionally omitted: retrieving an
expected record is not evidence that the final answer is good, and missing an
expected record does not by itself make a faithful final answer bad.
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
import re
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SOURCE_ARTIFACT_TYPE = "five_piece_qualitative_rag_llm_evaluation"
ARTIFACT_TYPE = "five_piece_qualitative_answer_fidelity_judgment"
SCHEMA_VERSION = "1.0"
PROTOCOL_VERSION = "qualitative-answer-fidelity-v2"
PIECE_IDS = (
    "die-forelle",
    "in-flowery-clouds",
    "la-capinera",
    "nella-fantasia",
    "una-voce-poco-fa",
)
DEFAULT_INPUT = PROJECT_ROOT / "evaluation" / "five_piece_qualitative.json"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "evaluation" / "five_piece_qualitative_judged.json"
)
QWEN3_4B_CACHE_ROOT = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--Qwen--Qwen3-4B"
)
VERDICTS = ("pass", "review", "fail")
MAX_REASON_CHARACTERS = 400


JudgeFunction = Callable[[list[dict[str, str]]], str]
CheckpointFunction = Callable[[], None]


class QualitativeJudgeInputError(ValueError):
    """Raised when a source or resume artifact violates this contract."""


class QualitativeJudgeResponseError(ValueError):
    """Raised when the local judge does not return the requested JSON."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_qwen3_4b_path() -> Path:
    """Resolve the locally cached immutable Qwen3-4B snapshot, if present."""

    revision_file = QWEN3_4B_CACHE_ROOT / "refs" / "main"
    if revision_file.is_file():
        revision = revision_file.read_text(encoding="utf-8").strip()
        if revision:
            return QWEN3_4B_CACHE_ROOT / "snapshots" / revision
    return QWEN3_4B_CACHE_ROOT


def _directory_identity(path: Path) -> str:
    """Hash a model manifest without rereading every weight shard."""

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


def model_record(path: Path, *, backend: str) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved.is_dir():
        kind = "directory"
        identity = _directory_identity(resolved)
    elif resolved.is_file():
        kind = "file"
        identity = file_sha256(resolved)
    else:
        kind = "missing"
        identity = None
    return {
        "path": str(resolved),
        "backend": backend,
        "kind": kind,
        "checkpoint_exists": identity is not None,
        "sha256": identity,
    }


def validate_source_artifact(source: Mapping[str, Any]) -> None:
    if source.get("artifact_type") != SOURCE_ARTIFACT_TYPE:
        raise QualitativeJudgeInputError(
            "Input is not a five-piece qualitative RAG+LLM artifact"
        )
    if not isinstance(source.get("questions"), list):
        raise QualitativeJudgeInputError("Input questions must be a list")
    source_run = source.get("run")
    if not isinstance(source_run, Mapping) or source_run.get("generate") is not True:
        raise QualitativeJudgeInputError(
            "Input must be produced by run_five_piece_qualitative.py "
            "with --generate"
        )

    seen_case_ids: set[str] = set()
    for question in source["questions"]:
        if not isinstance(question, Mapping):
            raise QualitativeJudgeInputError("Input question must be an object")
        for key in (
            "source_id",
            "piece_id",
            "paraphrased_question",
            "reference_material",
            "inference_runs",
        ):
            if key not in question:
                raise QualitativeJudgeInputError(
                    f"Input question is missing {key}"
                )
        if question["piece_id"] not in PIECE_IDS:
            raise QualitativeJudgeInputError(
                f"Unsupported piece_id {question['piece_id']}"
            )
        reference = question["reference_material"]
        if not isinstance(reference, Mapping) or not isinstance(
            reference.get("source_answer"), str
        ):
            raise QualitativeJudgeInputError(
                f"{question['source_id']}: exact source answer is missing"
            )
        if not isinstance(reference.get("linked_knowledge_units"), list):
            raise QualitativeJudgeInputError(
                f"{question['source_id']}: linked knowledge units are missing"
            )
        if not isinstance(question["inference_runs"], list):
            raise QualitativeJudgeInputError(
                f"{question['source_id']}: inference_runs must be a list"
            )
        for case in question["inference_runs"]:
            case_id = case.get("case_id") if isinstance(case, Mapping) else None
            if not isinstance(case_id, str) or not case_id:
                raise QualitativeJudgeInputError("Input case_id is missing")
            if case_id in seen_case_ids:
                raise QualitativeJudgeInputError(
                    f"Duplicate input case_id {case_id}"
                )
            seen_case_ids.add(case_id)
            if not isinstance(case.get("inference_input"), Mapping):
                raise QualitativeJudgeInputError(
                    f"{case_id}: inference_input is missing"
                )


def _linked_unit_reference(unit: Mapping[str, Any]) -> dict[str, Any]:
    required = ("knowledge_unit_id", "answer", "rewrite_status", "measure_status")
    missing = [key for key in required if key not in unit]
    if missing:
        raise QualitativeJudgeInputError(
            "Linked knowledge unit is missing " + ", ".join(missing)
        )
    return {
        "knowledge_unit_id": unit["knowledge_unit_id"],
        "answer": unit["answer"],
        "rewrite_status": unit["rewrite_status"],
        "rewrite_notes": unit.get("rewrite_notes", ""),
        "measure_status": unit["measure_status"],
        "measure_ranges": deepcopy(unit.get("measure_ranges") or []),
        "measure_notes": unit.get("measure_notes", ""),
    }


def build_judgment_cases(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build immutable judge packets from the qualitative inference artifact."""

    validate_source_artifact(source)
    judgments: list[dict[str, Any]] = []
    for question in source["questions"]:
        reference = question["reference_material"]
        linked_units = [
            _linked_unit_reference(unit)
            for unit in reference["linked_knowledge_units"]
        ]
        linked_ids = [unit["knowledge_unit_id"] for unit in linked_units]

        for inference_case in question["inference_runs"]:
            case_id = inference_case["case_id"]
            inference_input = inference_case["inference_input"]
            inference_status = inference_case.get("status")
            pipeline_result = inference_case.get("pipeline_result")
            candidate_answer: str | None = None
            supplemental_expert_support: list[dict[str, Any]] = []
            if inference_status == "completed":
                if not isinstance(pipeline_result, Mapping):
                    raise QualitativeJudgeInputError(
                        f"{case_id}: completed case has no pipeline_result"
                    )
                answer = pipeline_result.get("answer")
                if not isinstance(answer, str):
                    raise QualitativeJudgeInputError(
                        f"{case_id}: pipeline answer must be text"
                    )
                candidate_answer = answer
                for evidence in pipeline_result.get("evidence") or []:
                    if not isinstance(evidence, Mapping):
                        continue
                    if not (
                        evidence.get("kind") == "expert"
                        or evidence.get("evidence_type")
                        == "expert_annotation"
                    ):
                        continue
                    text = evidence.get("text")
                    if not isinstance(text, str) or not text.strip():
                        continue
                    supplemental_expert_support.append(
                        {
                            # Content is factuality support only. Retrieval
                            # rank and target-hit diagnostics remain absent.
                            "text": text,
                            "measure_ranges": deepcopy(
                                evidence.get("measure_ranges") or []
                            ),
                            "scope_match": evidence.get("scope_match"),
                            "rewrite_status": evidence.get("rewrite_status"),
                            "retrieval_review_warning": evidence.get(
                                "retrieval_review_warning", ""
                            ),
                        }
                    )

            expected = inference_case.get("expected_retrieval") or {}
            applicable_ids = expected.get(
                "range_applicable_knowledge_unit_ids",
                linked_ids,
            )
            if not isinstance(applicable_ids, list):
                raise QualitativeJudgeInputError(
                    f"{case_id}: applicable knowledge-unit IDs must be a list"
                )
            unknown_applicable = set(applicable_ids) - set(linked_ids)
            if unknown_applicable:
                raise QualitativeJudgeInputError(
                    f"{case_id}: applicable unit is not linked: "
                    f"{sorted(unknown_applicable)}"
                )

            judgments.append(
                {
                    "case_id": case_id,
                    "source_id": question["source_id"],
                    "piece_id": question["piece_id"],
                    "question": inference_input.get(
                        "question", question["paraphrased_question"]
                    ),
                    "original_question": question.get("original_question", ""),
                    "measure_range": deepcopy(
                        inference_input.get("measure_range")
                    ),
                    "candidate_answer": candidate_answer,
                    "reference": {
                        "exact_source_answer": reference["source_answer"],
                        "linked_knowledge_units": deepcopy(linked_units),
                        "range_applicable_knowledge_unit_ids": deepcopy(
                            applicable_ids
                        ),
                        "supplemental_expert_support": (
                            supplemental_expert_support
                        ),
                    },
                    "source_inference_status": inference_status,
                    "judgment": {
                        "status": (
                            "pending"
                            if inference_status == "completed"
                            else "unavailable"
                        ),
                        "verdict": None,
                        "reason": None,
                        "attempt_count": 0,
                        "attempts": [],
                        "started_at": None,
                        "completed_at": None,
                        "error": None,
                    },
                }
            )
    return judgments


def build_judge_messages(case: Mapping[str, Any]) -> list[dict[str, str]]:
    """Build a retrieval-independent qualitative answer-fidelity prompt."""

    reference = case["reference"]
    applicable_ids = set(reference["range_applicable_knowledge_unit_ids"])
    applicable_units = [
        unit
        for unit in reference["linked_knowledge_units"]
        if unit["knowledge_unit_id"] in applicable_ids
    ]
    other_range_units = [
        unit
        for unit in reference["linked_knowledge_units"]
        if unit["knowledge_unit_id"] not in applicable_ids
    ]
    range_selected = case["measure_range"] is not None
    payload = {
        "question": case["question"],
        "selected_measure_range": case["measure_range"],
        "exact_annotator_answer": (
            None if range_selected else reference["exact_source_answer"]
        ),
        "linked_knowledge_units": (
            applicable_units
            if range_selected
            else reference["linked_knowledge_units"]
        ),
        "range_applicable_linked_knowledge_units": applicable_units,
        "other_range_linked_knowledge_units": (
            [] if range_selected else other_range_units
        ),
        "withheld_other_range_reference_count": (
            len(other_range_units) if range_selected else 0
        ),
        "range_applicable_knowledge_unit_ids": reference[
            "range_applicable_knowledge_unit_ids"
        ],
        "supplemental_expert_support": reference.get(
            "supplemental_expert_support", []
        ),
        "generated_answer": case["candidate_answer"],
    }
    system = """\
너는 성악 전문가 주석에 대한 답변 충실도를 보수적으로 판정하는 심사자다.
이 평가는 간단한 정성적 선별이며 schema-1.3 자동 통과 프로토콜이 아니다.

평가 기준:
1. 질문과 선택 마디 범위에 맞게 생성 답변의 핵심 내용이 정확한지 본다.
2. 정확한 주석자 원답변과 연결된 지식 단위 답변을 함께 근거로 사용한다.
   정확한 주석자 답변과 범위에 적용되는 연결 지식 단위가 질문의 목표(R)다.
   연결된 모든 지식 단위의 모든 문장을 답변에 반복해야 한다는 뜻은 아니며,
   질문에 직접 필요한 핵심 조언을 빠짐없이 전달하면 된다.
   supplemental_expert_support는 생성 답변의 추가 내용이 사실인지 확인하는
   보조 근거(S)일 뿐이며, 빠진 목표 내용을 대신 충족할 수 없다.
3. 선택 범위가 있으면 range_applicable_linked_knowledge_units만 그 범위의
   목표(R)다. 원문 답변과 다른 구간 지식 단위는 혼합 범위 주장의 전이를
   막기 위해 입력에서 비공개 처리된다. 비공개 참조의 내용을 추측하거나
   누락으로 판정하지 않는다. 다른 구간 주장을 선택 범위의 사실처럼 옮기면 fail이다.
   범위가 없으면 정확한 주석자 답변과 연결 지식 단위 전체가 목표다.
4. 핵심 뜻을 충실히 전달하면 세부 예시를 모두 반복할 필요는 없다.
5. 핵심 모순, 중요한 근거 없는 추가 주장, 질문의 핵심을 답하지 못한 경우,
   또는 국소 주장을 근거 없이 곡 전체 사실로 확대하면 fail이다.
6. 대체로 타당하지만 부분성ㆍ범위ㆍ주석 검토 상태 때문에 전문가 판단이
   더 필요한 경우 review다. 명확히 충실하면 pass다.
7. 검색 성공 여부, 예상 지식 단위 검색 여부, 검색 순위, 인용 형식은 평가하지
   않는다. 오직 최종 생성 답변의 내용만 평가한다.

반드시 다음 키만 있는 JSON 객체 하나를 출력한다.
{"verdict":"pass|review|fail","reason":"한국어 한 문장"}
reason은 판정의 가장 중요한 근거만 180자 이내로 간결하게 쓴다.
reason에는 일치ㆍ누락ㆍ모순한 구체적인 음악 또는 가창 내용을 적고,
단순히 "근거와 일치한다" 같은 일반적인 문구만 쓰지 않는다.
코드 펜스, 설명, 머리말을 JSON 밖에 쓰지 않는다."""
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, indent=2),
        },
    ]


def validate_judge_response(raw: str) -> dict[str, str]:
    if not isinstance(raw, str) or not raw.strip():
        raise QualitativeJudgeResponseError("Judge response is empty")
    cleaned = re.sub(
        r"<think>.*?</think>",
        "",
        raw,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()
    cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as error:
        raise QualitativeJudgeResponseError(
            f"Judge response is not one JSON object: {error.msg}"
        ) from error
    if not isinstance(value, dict) or set(value) != {"verdict", "reason"}:
        raise QualitativeJudgeResponseError(
            "Judge response must contain exactly verdict and reason"
        )
    verdict = value["verdict"]
    reason = value["reason"]
    if verdict not in VERDICTS:
        raise QualitativeJudgeResponseError(
            f"Unsupported qualitative verdict {verdict!r}"
        )
    if not isinstance(reason, str):
        raise QualitativeJudgeResponseError("Judge reason must be text")
    reason = " ".join(reason.split())
    if not reason:
        raise QualitativeJudgeResponseError("Judge reason is empty")
    if len(reason) > MAX_REASON_CHARACTERS:
        raise QualitativeJudgeResponseError(
            f"Judge reason exceeds {MAX_REASON_CHARACTERS} characters"
        )
    return {"verdict": verdict, "reason": reason}


def build_input_fingerprint(
    *,
    source_judgment_input_sha256: str,
    judge_model: Mapping[str, Any],
    judge_max_tokens: int,
) -> str:
    payload = {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "source_judgment_input_sha256": source_judgment_input_sha256,
        "judge_model": dict(judge_model),
        "judge_max_tokens": judge_max_tokens,
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def immutable_judgment_case_payloads(
    cases: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return exactly the case material that can affect a judgment.

    Checkpoint timestamps and verdicts are deliberately excluded.  The
    qualitative protocol version remains responsible for invalidating saved
    judgments when the judge instructions themselves change.
    """

    return [
        {
            key: deepcopy(value)
            for key, value in case.items()
            if key != "judgment"
        }
        for case in cases
    ]


def source_judgment_input_sha256(source: Mapping[str, Any]) -> str:
    """Hash judge-visible source content, not volatile run timestamps."""

    payload = {
        "artifact_type": source.get("artifact_type"),
        "schema_version": source.get("schema_version"),
        "source_input_fingerprint": source.get("run", {}).get(
            "input_fingerprint"
        ),
        "cases": immutable_judgment_case_payloads(
            build_judgment_cases(source)
        ),
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def new_snapshot(
    *,
    source: Mapping[str, Any],
    source_path: Path,
    source_sha256: str,
    source_judgment_sha256: str | None = None,
    judge_model: Mapping[str, Any],
    judge_max_tokens: int,
    input_fingerprint: str,
    implementation_sha256: str,
) -> dict[str, Any]:
    now = utc_now()
    snapshot = {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "protocol": {
            "name": "five-piece qualitative answer-fidelity judge",
            "version": PROTOCOL_VERSION,
            "purpose": (
                "Conservative pass/review/fail triage of generated answers "
                "against exact expert references."
            ),
            "strict_schema_1_3_automatic_pass_protocol": False,
            "issues_strict_automatic_pass_verdicts": False,
            "retrieval_exactness_is_answer_quality": False,
            "human_review_remains_authoritative": True,
        },
        "run": {
            "status": "not_started",
            "started_at": now,
            "updated_at": now,
            "finished_at": None,
            "source_artifact": {
                "path": str(source_path.resolve()),
                "sha256": source_sha256,
                "sha256_scope": "full_file_at_last_judgment",
                "judgment_input_sha256": source_judgment_sha256,
                "artifact_type": source.get("artifact_type"),
                "schema_version": source.get("schema_version"),
                "source_run_status": source.get("run", {}).get("status"),
            },
            "judge_model": dict(judge_model),
            "judge_max_tokens": judge_max_tokens,
            "input_fingerprint": input_fingerprint,
            "implementation_sha256": implementation_sha256,
            "last_minimum_pass_rate": None,
        },
        "cases": build_judgment_cases(source),
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
        raise QualitativeJudgeInputError(
            "Output is not a five-piece qualitative judgment artifact"
        )
    if snapshot.get("schema_version") != SCHEMA_VERSION:
        raise QualitativeJudgeInputError(
            "Output schema version is not resumable by this judge"
        )
    if snapshot.get("run", {}).get("input_fingerprint") != input_fingerprint:
        raise QualitativeJudgeInputError(
            "Inference artifact, judge model, or judge configuration changed; "
            "choose a new --output path rather than mixing judgments"
        )


def can_migrate_semantically_identical_resume(
    snapshot: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    judge_model: Mapping[str, Any],
    judge_max_tokens: int,
) -> bool:
    """Allow old full-file hashes to migrate after timestamp-only changes."""

    if snapshot.get("artifact_type") != ARTIFACT_TYPE:
        return False
    if snapshot.get("schema_version") != SCHEMA_VERSION:
        return False
    if snapshot.get("protocol", {}).get("version") != PROTOCOL_VERSION:
        return False
    run = snapshot.get("run", {})
    if run.get("judge_model") != dict(judge_model):
        return False
    if run.get("judge_max_tokens") != judge_max_tokens:
        return False
    try:
        expected_cases = build_judgment_cases(source)
    except QualitativeJudgeInputError:
        return False
    return immutable_judgment_case_payloads(
        snapshot.get("cases") or []
    ) == immutable_judgment_case_payloads(expected_cases)


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _piece_summary(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    statuses = Counter(case["judgment"]["status"] for case in cases)
    verdicts = Counter(
        case["judgment"]["verdict"]
        for case in cases
        if case["judgment"]["status"] == "completed"
    )
    judgeable = len(cases) - statuses["unavailable"]
    completed = statuses["completed"]
    piece_complete = (
        bool(judgeable)
        and completed == judgeable
        and not statuses["unavailable"]
    )
    return {
        "case_count": len(cases),
        "judgeable_case_count": judgeable,
        "judgment_statuses": dict(sorted(statuses.items())),
        "verdicts": dict(sorted(verdicts.items())),
        "completed_case_pass_rate": _rate(verdicts["pass"], completed),
        "final_case_pass_rate": (
            _rate(verdicts["pass"], judgeable) if piece_complete else None
        ),
        "complete": piece_complete,
    }


def refresh_summary(snapshot: dict[str, Any]) -> None:
    cases = snapshot.get("cases") or []
    statuses = Counter(case["judgment"]["status"] for case in cases)
    verdicts = Counter(
        case["judgment"]["verdict"]
        for case in cases
        if case["judgment"]["status"] == "completed"
    )
    judgeable = len(cases) - statuses["unavailable"]
    completed = statuses["completed"]

    if statuses["pending"] or statuses["running"]:
        attempted = completed + statuses["error"] + statuses["running"]
        run_status = "in_progress" if attempted else "not_started"
    elif statuses["error"]:
        run_status = "complete_with_errors"
    elif statuses["unavailable"]:
        run_status = "input_incomplete"
    elif judgeable and completed == judgeable:
        run_status = "completed"
    else:
        run_status = "input_incomplete"

    by_piece = {
        piece_id: _piece_summary(
            [case for case in cases if case["piece_id"] == piece_id]
        )
        for piece_id in PIECE_IDS
        if any(case["piece_id"] == piece_id for case in cases)
    }
    final_pass_rate = (
        _rate(verdicts["pass"], judgeable)
        if run_status == "completed"
        else None
    )
    snapshot["run"]["status"] = run_status
    snapshot["run"]["updated_at"] = utc_now()
    snapshot["run"]["finished_at"] = (
        snapshot["run"]["updated_at"]
        if run_status in {
            "completed",
            "complete_with_errors",
            "input_incomplete",
        }
        else None
    )
    snapshot["summary"] = {
        "case_count": len(cases),
        "judgeable_case_count": judgeable,
        "unavailable_case_count": statuses["unavailable"],
        "judgment_statuses": dict(sorted(statuses.items())),
        "verdicts": dict(sorted(verdicts.items())),
        "completed_case_pass_rate": _rate(verdicts["pass"], completed),
        "final_case_pass_rate": final_pass_rate,
        "minimum_pass_rate_gate_eligible": run_status == "completed",
        "by_piece": by_piece,
        "metric_note": (
            "Pass rate measures qualitative generated-answer fidelity to "
            "expert references. It does not use exact retrieval recall and "
            "is not the strict schema-1.3 automatic-pass metric."
        ),
    }


def selected_incomplete_cases(
    snapshot: Mapping[str, Any],
    *,
    piece_ids: Sequence[str] | None = None,
    case_ids: Sequence[str] | None = None,
    limit: int | None = None,
    max_attempts: int,
) -> list[dict[str, Any]]:
    cases = list(snapshot.get("cases") or [])
    known_ids = {case["case_id"] for case in cases}
    requested_ids = set(case_ids or [])
    unknown = requested_ids - known_ids
    if unknown:
        raise QualitativeJudgeInputError(
            f"Unknown --case-id values: {sorted(unknown)}"
        )
    selected_pieces = set(piece_ids or PIECE_IDS)
    selected = [
        case
        for case in cases
        if case["piece_id"] in selected_pieces
        and (not requested_ids or case["case_id"] in requested_ids)
        and case["judgment"]["status"] != "completed"
        and case["judgment"]["status"] != "unavailable"
        and case["judgment"]["attempt_count"] < max_attempts
    ]
    return selected[:limit] if limit is not None else selected


def run_judgments(
    snapshot: dict[str, Any],
    *,
    judge_fn: JudgeFunction,
    checkpoint: CheckpointFunction,
    max_attempts: int,
    piece_ids: Sequence[str] | None = None,
    case_ids: Sequence[str] | None = None,
    limit: int | None = None,
) -> int:
    """Judge selected cases with retry checkpoints; return selected count."""

    selected = selected_incomplete_cases(
        snapshot,
        piece_ids=piece_ids,
        case_ids=case_ids,
        limit=limit,
        max_attempts=max_attempts,
    )
    for case in selected:
        judgment = case["judgment"]
        if judgment["started_at"] is None:
            judgment["started_at"] = utc_now()
        while (
            judgment["status"] != "completed"
            and judgment["attempt_count"] < max_attempts
        ):
            judgment["status"] = "running"
            judgment["attempt_count"] += 1
            judgment["error"] = None
            attempt = {
                "attempt": judgment["attempt_count"],
                "started_at": utc_now(),
                "completed_at": None,
                "raw_response": None,
                "valid": False,
                "error": None,
            }
            judgment["attempts"].append(attempt)
            checkpoint()
            try:
                raw = judge_fn(build_judge_messages(case))
                attempt["raw_response"] = raw
                assessment = validate_judge_response(raw)
                attempt["valid"] = True
                judgment["verdict"] = assessment["verdict"]
                judgment["reason"] = assessment["reason"]
                judgment["status"] = "completed"
                judgment["completed_at"] = utc_now()
            except Exception as error:  # preserve local-model/parser failures
                attempt["error"] = {
                    "type": type(error).__name__,
                    "message": str(error),
                }
                judgment["status"] = "error"
                judgment["error"] = deepcopy(attempt["error"])
            finally:
                attempt["completed_at"] = utc_now()
                checkpoint()
    return len(selected)


def quality_gate_exit_code(
    snapshot: Mapping[str, Any],
    *,
    minimum_pass_rate: float,
) -> int:
    """Return 2 only for a complete run below the qualitative pass target."""

    if snapshot.get("run", {}).get("status") != "completed":
        return 0
    pass_rate = snapshot.get("summary", {}).get("final_case_pass_rate")
    if pass_rate is None:
        return 0
    return 2 if pass_rate < minimum_pass_rate else 0


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
                f"Another qualitative judge is writing {path}"
            ) from error
        yield


def build_local_judge(
    model_path: Path,
    *,
    backend: str,
    max_tokens: int,
) -> JudgeFunction:
    """Build a local deterministic judge, reusing the existing backend."""

    if backend == "transformers":
        from evaluation.run_question_evaluation import build_transformers_judge

        return build_transformers_judge(model_path, max_tokens=max_tokens)
    if backend != "llama-cpp":
        raise QualitativeJudgeInputError(
            f"Unsupported local judge backend {backend}"
        )

    from soprano_qa.llm import generate as generate_llm
    from soprano_qa.settings import load_settings

    settings = load_settings(use_legacy_dataset_env=False)
    judge_settings = {
        **settings["llm"],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }

    def generate(messages: list[dict[str, str]]) -> str:
        return generate_llm(str(model_path), messages, judge_settings)

    return generate


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--judge-model-path",
        type=Path,
        default=default_qwen3_4b_path(),
        help="Local Qwen3-4B Transformers directory or local GGUF file.",
    )
    parser.add_argument(
        "--judge-backend",
        choices=("auto", "transformers", "llama-cpp"),
        default="auto",
    )
    parser.add_argument("--judge-max-tokens", type=int, default=512)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument(
        "--piece",
        action="append",
        choices=PIECE_IDS,
        help="Judge only this piece in this invocation; repeat as needed.",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Judge only this stable inference case ID; repeat as needed.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Judge at most N incomplete selected cases in this invocation.",
    )
    parser.add_argument(
        "--minimum-pass-rate",
        type=float,
        default=0.90,
        help=(
            "Exit 2 only after a complete qualitative judgment run below "
            "this generated-answer fidelity rate."
        ),
    )
    arguments = parser.parse_args(argv)
    if arguments.judge_max_tokens < 64:
        parser.error("--judge-max-tokens must be at least 64")
    if arguments.max_attempts < 1:
        parser.error("--max-attempts must be positive")
    if arguments.limit is not None and arguments.limit < 1:
        parser.error("--limit must be positive")
    if not 0 <= arguments.minimum_pass_rate <= 1:
        parser.error("--minimum-pass-rate must be between 0 and 1")
    return arguments


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    source_path = arguments.input.resolve()
    output_path = arguments.output.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Qualitative inference artifact not found: {source_path}"
        )

    source = load_json(source_path)
    validate_source_artifact(source)
    source_sha256 = file_sha256(source_path)
    source_judgment_sha256 = source_judgment_input_sha256(source)
    judge_model_path = arguments.judge_model_path.resolve()
    judge_backend = arguments.judge_backend
    if judge_backend == "auto":
        judge_backend = (
            "transformers" if judge_model_path.is_dir() else "llama-cpp"
        )
    judge_model = model_record(judge_model_path, backend=judge_backend)
    if not judge_model["checkpoint_exists"]:
        raise FileNotFoundError(f"Local judge model not found: {judge_model_path}")
    if judge_backend == "transformers" and not judge_model_path.is_dir():
        raise QualitativeJudgeInputError(
            "Transformers judge requires a local model directory"
        )
    if judge_backend == "llama-cpp" and not judge_model_path.is_file():
        raise QualitativeJudgeInputError(
            "llama-cpp judge requires a local GGUF file"
        )

    input_fingerprint = build_input_fingerprint(
        source_judgment_input_sha256=source_judgment_sha256,
        judge_model=judge_model,
        judge_max_tokens=arguments.judge_max_tokens,
    )
    implementation_sha256 = file_sha256(Path(__file__).resolve())
    with exclusive_output_lock(output_path):
        migrated_resume = False
        if output_path.exists():
            snapshot = load_json(output_path)
            try:
                validate_resume_snapshot(
                    snapshot,
                    input_fingerprint=input_fingerprint,
                )
            except QualitativeJudgeInputError:
                if not can_migrate_semantically_identical_resume(
                    snapshot,
                    source=source,
                    judge_model=judge_model,
                    judge_max_tokens=arguments.judge_max_tokens,
                ):
                    raise
                snapshot["run"]["input_fingerprint"] = input_fingerprint
                snapshot["run"]["implementation_sha256"] = (
                    implementation_sha256
                )
                snapshot["run"]["source_artifact"].update(
                    {
                        "sha256": source_sha256,
                        "sha256_scope": "full_file_at_last_judgment",
                        "judgment_input_sha256": source_judgment_sha256,
                    }
                )
                migrated_resume = True
        else:
            snapshot = new_snapshot(
                source=source,
                source_path=source_path,
                source_sha256=source_sha256,
                source_judgment_sha256=source_judgment_sha256,
                judge_model=judge_model,
                judge_max_tokens=arguments.judge_max_tokens,
                input_fingerprint=input_fingerprint,
                implementation_sha256=implementation_sha256,
            )
            atomic_write_json(output_path, snapshot)

        minimum_rate_changed = snapshot["run"].get(
            "last_minimum_pass_rate"
        ) != arguments.minimum_pass_rate
        snapshot["run"]["last_minimum_pass_rate"] = arguments.minimum_pass_rate

        def checkpoint() -> None:
            refresh_summary(snapshot)
            atomic_write_json(output_path, snapshot)

        judge_fn = build_local_judge(
            judge_model_path,
            backend=judge_backend,
            max_tokens=arguments.judge_max_tokens,
        )
        attempted = run_judgments(
            snapshot,
            judge_fn=judge_fn,
            checkpoint=checkpoint,
            max_attempts=arguments.max_attempts,
            piece_ids=arguments.piece,
            case_ids=arguments.case_id,
            limit=arguments.limit,
        )
        if attempted or migrated_resume or minimum_rate_changed:
            checkpoint()

    print(
        json.dumps(
            {
                "artifact": str(output_path),
                "attempted_cases": attempted,
                "run_status": snapshot["run"]["status"],
                "summary": snapshot["summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    gate_code = quality_gate_exit_code(
        snapshot,
        minimum_pass_rate=arguments.minimum_pass_rate,
    )
    if gate_code:
        rate = snapshot["summary"]["final_case_pass_rate"]
        print(
            "Qualitative answer-fidelity gate failed: "
            f"{rate:.1%} < {arguments.minimum_pass_rate:.1%}",
            file=sys.stderr,
        )
        return gate_code
    if snapshot["summary"]["judgment_statuses"].get("error", 0):
        return 1
    if snapshot["run"]["status"] == "input_incomplete":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
