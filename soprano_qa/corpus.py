#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a combined expert-and-web retrieval corpus from the external dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from datetime import datetime
from typing import Any, Dict, Iterable, List, Sequence
from urllib.parse import urlsplit

from soprano_qa.settings import load_settings


SONG_ORDER = [
    "una-voce-poco-fa",
    "la-capinera",
    "die-forelle",
    "nella-fantasia",
    "in-flowery-clouds",
]

PIECES: Dict[str, Dict[str, str]] = {
    "una-voce-poco-fa": {
        "work": 'Rossini, Il barbiere di Siviglia - "Una voce poco fa"',
        "title": "Una voce poco fa",
        "composer": "Gioachino Rossini",
        "genre": "opera aria",
        "language": "Italian",
    },
    "la-capinera": {
        "work": 'Julius Benedict - "La capinera"',
        "title": "La capinera",
        "composer": "Julius Benedict",
        "genre": "Italian song",
        "language": "Italian",
    },
    "die-forelle": {
        "work": 'Schubert, D.550 - "Die Forelle"',
        "title": "Die Forelle",
        "composer": "Franz Schubert",
        "genre": "German lied",
        "language": "German",
    },
    "nella-fantasia": {
        "work": 'Morricone - "Nella fantasia"',
        "title": "Nella fantasia",
        "composer": "Ennio Morricone",
        "genre": "crossover",
        "language": "Italian",
    },
    "in-flowery-clouds": {
        "work": "Lee Heung-ryul - In Flowery Clouds",
        "title": "꽃구름 속에",
        "composer": "이흥렬",
        "genre": "Korean art song",
        "language": "Korean",
    },
}

WEB_EXPORT_USAGE_CLASSES = {
    "research-open.jsonl": "research_open",
    "research-conditional.jsonl": "research_conditional",
}

WEB_TOPIC_SEARCH_TERMS = {
    "creator_profiles": (
        "작곡가 작사가 작시가 시인 대본가 창작자 인물 약력 생애 배경 누구 "
        "composer creator biography"
    ),
    "editions_versions_and_arrangements": "판본 악보 비평판 편곡 버전 개정 에디션",
    "genre_and_form": "장르 형식 구조 조성 박자 편성 genre form",
    "identity_aliases_and_catalog_ids": (
        "정식 표제 제목 원제 별칭 작품 번호 작품번호 카탈로그 "
        "title original title catalog number work number"
    ),
    "other_documented_topics": "기타 문헌 역사 대본가 작사가 자료",
    "parent_and_source_works": "원작 원곡 원전 관계 영화 오페라 바탕",
    "performance_practice": "연주 관행 가창 연습",
    "performances_and_recordings": "녹음 음반 공연 연주 실황 수용 이력",
    "primary_source_history": "원전 일차 자료 1차 자료 자필 악보 소장 자료 판본",
    "publication_and_premiere": "출판 발행 초연 작곡 연도 역사",
    "rights_history": (
        "저작권 라이선스 권리 이용 재사용 재배포 조건 허용 사용 조건 관할권 "
        "rights license jurisdiction territorial access conditions terms"
    ),
    "text_language_and_meaning": "언어 가사 텍스트 시 뜻 의미 계보 번역",
}

WEB_SUBTOPIC_SEARCH_TERMS = {
    "character_and_expression": "성격 분위기 표현 음색",
    "dramatic_context": "극적 맥락 캐릭터 장면 오페라",
    "lyric_pronunciation": "가사 발음 딕션 IPA 음성학 언어",
    "piano_and_voice": "성악 피아노 반주 관계 균형 표현",
    "recordings_and_reception": "녹음 음반 공연 수용 이력",
}

WEB_CLAIM_TOPIC_TO_CHUNK_TOPIC = {
    "authority_control": "identity_aliases_and_catalog_ids",
    "authorship": "creator_profiles",
    "career": "creator_profiles",
    "character_and_expression": "performance_practice",
    "composer_biography": "creator_profiles",
    "composer_identity": "creator_profiles",
    "creator_biography": "creator_profiles",
    "creator_roles": "creator_profiles",
    "creators": "creator_profiles",
    "critical_editions": "editions_versions_and_arrangements",
    "dramatic_context": "performance_practice",
    "edition_control": "editions_versions_and_arrangements",
    "editions_and_arrangements": "editions_versions_and_arrangements",
    "editions_and_catalogs": "editions_versions_and_arrangements",
    "evidence_method": "scholarship",
    "form": "genre_and_form",
    "form_and_structure": "genre_and_form",
    "genesis_and_chronology": "genesis_and_chronology",
    "genesis_and_premiere": "publication_and_premiere",
    "genesis_and_versions": "editions_versions_and_arrangements",
    "identity_control": "identity_aliases_and_catalog_ids",
    "language_and_text": "text_language_and_meaning",
    "literary_context": "parent_and_source_works",
    "lyric_pronunciation": "performance_practice",
    "meter": "musical_features",
    "meter_and_rhythm": "musical_features",
    "parent_work_and_film": "parent_and_source_works",
    "parent_work_identity": "parent_and_source_works",
    "ornamentation": "performance_practice",
    "piano_accompaniment": "performance_practice",
    "piano_and_voice": "performance_practice",
    "poet_biography": "creator_profiles",
    "poet_identity": "creator_profiles",
    "primary_sources": "primary_source_history",
    "publication_history": "publication_and_premiere",
    "range_and_edition_conflicts": "instrumentation_and_vocal_range",
    "reception_and_awards": "reception_and_awards",
    "reception_history": "reception_and_awards",
    "recording_history": "performances_and_recordings",
    "recording_metadata_conflicts": "performances_and_recordings",
    "recordings_and_reception": "performances_and_recordings",
    "rights_and_licensing": "rights_history",
    "rights_and_reuse": "rights_history",
    "scholarship": "scholarship",
    "scoring_and_form": "genre_and_form",
    "tempo": "musical_features",
    "title_and_identity": "identity_aliases_and_catalog_ids",
    "tonality": "musical_features",
    "work_identity": "identity_aliases_and_catalog_ids",
    "work_relations": "parent_and_source_works",
    "works_catalog": "identity_aliases_and_catalog_ids",
    "자료와 판본": "primary_source_history",
    "작품 관계와 버전": "editions_versions_and_arrangements",
    "작품 정체성과 표제": "identity_aliases_and_catalog_ids",
    "창작자와 기여자": "creator_profiles",
    "출판·자료 연혁": "publication_and_premiere",
}

OPEN_RIGHTS_STATUSES = {
    "public_domain",
    "cc0",
    "cc_by",
    "kogl_type_1",
    "open_government",
}
CONDITIONAL_RIGHTS_STATUSES = {"cc_by_sa", "cc_by_nc_sa", "permission"}
RESEARCH_REQUIRED_PERMISSIONS = (
    "storage",
    "adaptation",
    "text_mining",
    "redistribution",
)
DEMO_REQUIRED_PERMISSIONS = ("storage", "adaptation", "text_mining")
FACTS_REVIEW_KEYS = {
    "claim_id",
    "claim_record_sha256",
    "evidence_lineage_sha256",
    "status",
    "overlap_review_status",
    "reviewer",
    "reviewed_at",
    "method",
    "notes",
}
FACTS_REVIEW_STATUSES = {"approved", "rejected"}
FACTS_OVERLAP_REVIEW_STATUSES = {
    "no_protected_expression_detected",
    "protected_expression_detected",
    "indeterminate",
}

EXPERT_REVIEW_SCHEMA_VERSION = "1.0"
EXPERT_REVIEW_TOP_LEVEL_KEYS = (
    "schema_version",
    "piece_id",
    "source_files",
    "source_annotations",
    "knowledge_units",
)
EXPERT_SOURCE_KEYS = (
    "source_id",
    "annotator",
    "source_text",
    "question",
    "answer",
    "legacy_measure_ranges",
    "curation_status",
    "curation_notes",
)
EXPERT_UNIT_KEYS = (
    "knowledge_unit_id",
    "source_ids",
    "answer",
    "rewrite_status",
    "rewrite_notes",
    "measure_range_hints",
    "measure_status",
    "measure_ranges",
    "measure_notes",
)
EXPERT_ANNOTATORS = ("kim", "yeon")
EXPERT_SOURCE_STATUSES = {"included", "excluded_unanswerable"}
EXPERT_REWRITE_STATUSES = {"ready", "needs_review"}
EXPERT_MEASURE_STATUSES = {
    "waiting_for_review",
    "specific",
    "whole_piece",
    "unspecified",
}
ABSOLUTE_MEASURE_LOCATOR_RE = re.compile(
    r"(?:"
    r"\d+(?:\s*(?:[-~–—,/·]|및|과|와)\s*\d+)*"
    r"\s*[\])}]?\s*(?:번째\s*)?마디"
    r"|첫\s*마디"
    r"|\bmm?\.\s*\d+(?:\s*[-~–—]\s*\d+)?"
    r"|\bmeasures?\s+\d+(?:\s*[-~–—]\s*\d+)?"
    r")",
    re.IGNORECASE,
)


def load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("Invalid JSONL at %s:%d" % (path, line_number)) from exc
            if not isinstance(record, dict):
                raise ValueError("Expected a JSON object at %s:%d" % (path, line_number))
            yield record


def serialize_json(data: Any) -> bytes:
    return (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def atomic_write(path: str, payload: bytes) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    temporary_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=directory,
            prefix=".%s." % os.path.basename(path),
            delete=False,
        ) as temporary:
            temporary_path = temporary.name
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.chmod(temporary_path, 0o644)
        os.replace(temporary_path, path)
        temporary_path = ""
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def corpus_input_paths(settings: Dict[str, Any]) -> List[tuple[str, str]]:
    dataset_root = os.path.realpath(settings["dataset_root"])
    paths: List[tuple[str, str]] = []
    for song in SONG_ORDER:
        path = os.path.join(
            dataset_root,
            "expert_curation",
            "review",
            song + ".json",
        )
        paths.append((os.path.relpath(path, dataset_root), path))
    for record_file in ("sources.jsonl", "claims.jsonl", "chunks.jsonl"):
        record_path = os.path.join(dataset_root, "database", "records", record_file)
        paths.append((os.path.relpath(record_path, dataset_root), record_path))
    facts_audit_path = os.path.join(
        dataset_root,
        "database",
        "reports",
        "facts-only-release-audit.json",
    )
    paths.append((os.path.relpath(facts_audit_path, dataset_root), facts_audit_path))
    facts_review_path = os.path.join(
        dataset_root,
        "database",
        "facts-only-expression-reviews.json",
    )
    if os.path.exists(facts_review_path):
        paths.append((os.path.relpath(facts_review_path, dataset_root), facts_review_path))
    for export_file in settings.get("web_export_files", ["research-open.jsonl"]):
        path = os.path.join(dataset_root, "database", "exports", export_file)
        paths.append((os.path.relpath(path, dataset_root), path))
    override_path = settings["feature_overrides_path"]
    if os.path.exists(override_path):
        paths.append(("project/feature_overrides.json", override_path))
    return paths


def corpus_input_fingerprint(settings: Dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for label, path in corpus_input_paths(settings):
        if not os.path.isfile(path):
            raise FileNotFoundError("Missing corpus input: %s" % path)
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def normalize_ranges(ranges: Sequence[Sequence[int]]) -> List[List[int]]:
    normalized: List[List[int]] = []
    for item in ranges:
        if len(item) != 2:
            raise ValueError("Invalid measure range: %r" % (item,))
        if any(isinstance(value, bool) or not isinstance(value, int) for value in item):
            raise ValueError("Measure endpoints must be integers: %r" % (item,))
        start_i = item[0]
        end_i = item[1]
        if start_i < 1 or end_i < start_i:
            raise ValueError("Invalid measure range: %r" % (item,))
        normalized.append([start_i, end_i])
    return normalized


def default_measure_scope(measure_range: List[List[int]]) -> str:
    if not measure_range:
        return "global"
    if len(measure_range) > 1:
        return "multi_range"
    return "local"


def annotators_from_sources(source_ids: List[str], source_index: Dict[str, Dict[str, Any]]) -> List[str]:
    annotators = set()
    for source_id in source_ids:
        source_record = source_index.get(source_id)
        if source_record:
            annotators.add(source_record.get("annotator", source_id.split("-", 1)[0]))
        else:
            annotators.add(source_id.split("-", 1)[0])
    return sorted(annotators)


def semantic_relevance_text(record: Dict[str, Any]) -> str:
    """Text allowed to establish answer relevance, excluding provenance labels."""

    question = record.get("question") or ""
    topic = record.get("topic") or ""
    parts = [
        topic,
        WEB_TOPIC_SEARCH_TERMS.get(topic, ""),
        record.get("knowledge_scope") or "",
        question,
        question,
        record["answer"],
        record.get("applicability_note") or "",
    ]
    parts.extend(str(alias) for alias in record.get("retrieval_aliases", []))
    for subtopic in record.get("subtopics", []):
        parts.append(str(subtopic))
        parts.append(WEB_SUBTOPIC_SEARCH_TERMS.get(str(subtopic), ""))
    for feature in record.get("features", []):
        if isinstance(feature, dict):
            parts.extend(str(value) for value in feature.values())
        else:
            parts.append(str(feature))
    return "\n".join(part for part in parts if part)


def retrieval_text(record: Dict[str, Any]) -> str:
    parts = [semantic_relevance_text(record)]
    for source in record.get("sources", []):
        parts.extend(
            [
                str(source.get("title") or ""),
                str(source.get("institution") or ""),
            ]
        )
    return "\n".join(part for part in parts if part)


def load_overrides(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    overrides = load_json(path)
    if not isinstance(overrides, list):
        raise ValueError("Feature overrides must be a JSON list: %s" % path)
    return overrides


def source_match(record: Dict[str, Any], wanted: List[str]) -> bool:
    return set(wanted).issubset(set(record.get("source_ids", [])))


def validate_measure_scope(ranges: List[List[int]], scope: str) -> None:
    if scope == "recurring":
        if not ranges:
            raise ValueError("Recurring measure scope requires at least one exact range")
        return
    expected_scope = default_measure_scope(ranges)
    if scope != expected_scope:
        raise ValueError(
            "measure_scope %r contradicts measure_range %r; expected %r"
            % (scope, ranges, expected_scope)
        )


def apply_overrides(records: List[Dict[str, Any]], overrides: List[Dict[str, Any]]) -> int:
    applied = 0
    by_id = {record["id"]: record for record in records}
    selector_fields = {"id", "match_source_ids"}
    mutation_fields = {"measure_range", "measure_scope", "features", "applicability_note"}
    for override in overrides:
        if not isinstance(override, dict):
            raise ValueError("Each feature override must be a JSON object: %r" % override)
        unknown_fields = set(override) - selector_fields - mutation_fields
        if unknown_fields:
            raise ValueError("Unknown feature override fields: %r" % sorted(unknown_fields))
        has_id = "id" in override
        has_source_match = "match_source_ids" in override
        if has_id == has_source_match:
            raise ValueError("Override needs exactly one of id or match_source_ids: %r" % override)
        if has_id:
            if override["id"] not in by_id:
                raise ValueError("Override references unknown expert corpus id: %s" % override["id"])
            targets = [by_id[override["id"]]]
        else:
            wanted = override["match_source_ids"]
            if (
                not isinstance(wanted, list)
                or not wanted
                or any(not isinstance(source_id, str) or not source_id for source_id in wanted)
                or len(wanted) != len(set(wanted))
            ):
                raise ValueError(
                    "match_source_ids must be a non-empty list of unique strings: %r"
                    % wanted
                )
            targets = [record for record in records if source_match(record, wanted)]
            if not targets:
                raise ValueError("Override matched no expert records for source ids: %r" % wanted)
            if len(targets) != 1:
                raise ValueError(
                    "Override source selector is ambiguous; matched %r; use an explicit id"
                    % [record["id"] for record in targets]
                )
        if not (set(override) & mutation_fields):
            raise ValueError("Feature override must include at least one mutation field")

        for record in targets:
            if "measure_range" in override:
                record["measure_range"] = normalize_ranges(override["measure_range"])
            if "measure_scope" in override:
                record["measure_scope"] = override["measure_scope"]
            elif "measure_range" in override:
                record["measure_scope"] = default_measure_scope(record["measure_range"])
            validate_measure_scope(record["measure_range"], record["measure_scope"])
            if "features" in override:
                if not isinstance(override["features"], list):
                    raise ValueError("Override features must be a list")
                record["features"] = override["features"]
            if "applicability_note" in override:
                if not isinstance(override["applicability_note"], str):
                    raise ValueError("Override applicability_note must be a string")
                record["applicability_note"] = override["applicability_note"]
            record["relevance_text"] = semantic_relevance_text(record)
            record["retrieval_text"] = retrieval_text(record)
            applied += 1
    return applied


def expect_exact_keys(
    value: Any,
    expected_keys: Sequence[str],
    *,
    label: str,
) -> None:
    if not isinstance(value, dict) or tuple(value) != tuple(expected_keys):
        raise ValueError(
            "%s keys must be exactly %r in that order"
            % (label, list(expected_keys))
        )


def validate_review_ranges(
    value: Any,
    *,
    label: str,
    require_sorted_nonoverlapping: bool = True,
) -> List[List[int]]:
    if not isinstance(value, list):
        raise ValueError("%s must be a list" % label)
    for index, item in enumerate(value):
        if not isinstance(item, list):
            raise ValueError("%s/%d must be a [start, end] list" % (label, index))
    ranges = normalize_ranges(value)
    if require_sorted_nonoverlapping:
        previous_end = 0
        for index, (start, end) in enumerate(ranges):
            if start <= previous_end:
                raise ValueError(
                    "%s/%d ranges must be sorted and non-overlapping"
                    % (label, index)
                )
            previous_end = end
    return ranges


def union_source_ranges(
    source_records: Sequence[Dict[str, Any]],
) -> List[List[int]]:
    ranges = sorted(
        {
            tuple(item)
            for source_record in source_records
            for item in source_record["legacy_measure_ranges"]
        }
    )
    merged: List[List[int]] = []
    for start, end in ranges:
        if merged and start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def validate_expert_review_document(
    value: Any,
    path: str,
    piece_id: str,
) -> Dict[str, Dict[str, Any]]:
    label = str(path)
    expect_exact_keys(value, EXPERT_REVIEW_TOP_LEVEL_KEYS, label=label)
    if value["schema_version"] != EXPERT_REVIEW_SCHEMA_VERSION:
        raise ValueError("%s has unsupported expert review schema_version" % label)
    if value["piece_id"] != piece_id:
        raise ValueError("%s piece_id does not match its filename" % label)
    expected_source_files = [
        "annotation/original_txt/%s/%s.txt" % (annotator, piece_id)
        for annotator in EXPERT_ANNOTATORS
    ]
    if value["source_files"] != expected_source_files:
        raise ValueError("%s source_files do not match piece %s" % (label, piece_id))

    source_values = value["source_annotations"]
    if not isinstance(source_values, list) or not source_values:
        raise ValueError("%s source_annotations must be a non-empty list" % label)
    source_index: Dict[str, Dict[str, Any]] = {}
    source_order: Dict[str, int] = {}
    annotator_counts = Counter()
    previous_annotator_index = 0
    for source_position, source in enumerate(source_values):
        source_label = "%s/source_annotations/%d" % (label, source_position)
        expect_exact_keys(source, EXPERT_SOURCE_KEYS, label=source_label)
        annotator = source["annotator"]
        if annotator not in EXPERT_ANNOTATORS:
            raise ValueError("%s has invalid annotator %r" % (source_label, annotator))
        annotator_index = EXPERT_ANNOTATORS.index(annotator)
        if annotator_index < previous_annotator_index:
            raise ValueError("%s source annotations are out of annotator order" % label)
        previous_annotator_index = annotator_index
        annotator_counts[annotator] += 1
        expected_source_id = "%s-%s-%02d" % (
            annotator,
            piece_id,
            annotator_counts[annotator],
        )
        if source["source_id"] != expected_source_id:
            raise ValueError(
                "%s source_id must be %s" % (source_label, expected_source_id)
            )
        for field in ("source_text", "question", "answer"):
            if not isinstance(source[field], str):
                raise ValueError("%s/%s must be a string" % (source_label, field))
        if not source["source_text"].strip():
            raise ValueError("%s/source_text must not be empty" % source_label)
        validate_review_ranges(
            source["legacy_measure_ranges"],
            label="%s/legacy_measure_ranges" % source_label,
            require_sorted_nonoverlapping=False,
        )
        if source["curation_status"] not in EXPERT_SOURCE_STATUSES:
            raise ValueError("%s has invalid curation_status" % source_label)
        if not isinstance(source["curation_notes"], str):
            raise ValueError("%s/curation_notes must be a string" % source_label)
        if (
            source["curation_status"] == "excluded_unanswerable"
            and not source["curation_notes"].strip()
        ):
            raise ValueError("%s excluded source requires curation_notes" % source_label)
        source_id = source["source_id"]
        if source_id in source_index:
            raise ValueError("%s has duplicate source_id %s" % (label, source_id))
        source_index[source_id] = source
        source_order[source_id] = source_position
    if set(annotator_counts) != set(EXPERT_ANNOTATORS):
        raise ValueError("%s must preserve sources from every annotator" % label)

    units = value["knowledge_units"]
    if not isinstance(units, list) or not units:
        raise ValueError("%s knowledge_units must be a non-empty list" % label)
    unit_id_pattern = re.compile(r"^%s-ku-\d{3}$" % re.escape(piece_id))
    seen_unit_ids = set()
    covered_source_ids = set()
    for unit_position, unit in enumerate(units):
        unit_label = "%s/knowledge_units/%d" % (label, unit_position)
        expect_exact_keys(unit, EXPERT_UNIT_KEYS, label=unit_label)
        unit_id = unit["knowledge_unit_id"]
        if not isinstance(unit_id, str) or not unit_id_pattern.fullmatch(unit_id):
            raise ValueError("%s has invalid knowledge_unit_id %r" % (unit_label, unit_id))
        if unit_id in seen_unit_ids:
            raise ValueError("%s has duplicate knowledge_unit_id %s" % (label, unit_id))
        seen_unit_ids.add(unit_id)

        source_ids = unit["source_ids"]
        if (
            not isinstance(source_ids, list)
            or not source_ids
            or any(not isinstance(source_id, str) or not source_id for source_id in source_ids)
            or len(source_ids) != len(set(source_ids))
        ):
            raise ValueError("%s/source_ids must be unique non-empty strings" % unit_label)
        unknown_source_ids = [
            source_id for source_id in source_ids if source_id not in source_index
        ]
        if unknown_source_ids:
            raise ValueError(
                "%s has unknown source_ids %r" % (unit_label, unknown_source_ids)
            )
        if source_ids != sorted(source_ids, key=source_order.__getitem__):
            raise ValueError(
                "%s source_ids must follow source annotation order" % unit_label
            )
        excluded_source_ids = [
            source_id
            for source_id in source_ids
            if source_index[source_id]["curation_status"] != "included"
        ]
        if excluded_source_ids:
            raise ValueError(
                "%s links excluded source_ids %r" % (unit_label, excluded_source_ids)
            )
        covered_source_ids.update(source_ids)

        if not isinstance(unit["answer"], str) or not unit["answer"].strip():
            raise ValueError("%s/answer must be a non-empty string" % unit_label)
        measure_locator = ABSOLUTE_MEASURE_LOCATOR_RE.search(unit["answer"])
        if measure_locator:
            raise ValueError(
                "%s answer contains absolute measure locator %r"
                % (unit_label, measure_locator.group())
            )
        if unit["rewrite_status"] not in EXPERT_REWRITE_STATUSES:
            raise ValueError("%s has invalid rewrite_status" % unit_label)
        if not isinstance(unit["rewrite_notes"], str):
            raise ValueError("%s/rewrite_notes must be a string" % unit_label)
        if (
            unit["rewrite_status"] == "needs_review"
            and not unit["rewrite_notes"].strip()
        ):
            raise ValueError("%s needs_review requires rewrite_notes" % unit_label)

        hints = validate_review_ranges(
            unit["measure_range_hints"],
            label="%s/measure_range_hints" % unit_label,
        )
        expected_hints = union_source_ranges(
            [source_index[source_id] for source_id in source_ids]
        )
        if hints != expected_hints:
            raise ValueError(
                "%s measure_range_hints do not match source legacy ranges"
                % unit_label
            )
        measure_status = unit["measure_status"]
        if measure_status not in EXPERT_MEASURE_STATUSES:
            raise ValueError("%s has invalid measure_status" % unit_label)
        ranges = validate_review_ranges(
            unit["measure_ranges"],
            label="%s/measure_ranges" % unit_label,
        )
        if measure_status == "specific" and not ranges:
            raise ValueError("%s specific status requires measure_ranges" % unit_label)
        if measure_status != "specific" and ranges:
            raise ValueError(
                "%s only specific status may have measure_ranges" % unit_label
            )
        if not isinstance(unit["measure_notes"], str):
            raise ValueError("%s/measure_notes must be a string" % unit_label)

    included_source_ids = {
        source_id
        for source_id, source in source_index.items()
        if source["curation_status"] == "included"
    }
    missing_source_ids = sorted(included_source_ids - covered_source_ids)
    if missing_source_ids:
        raise ValueError(
            "%s included sources are not covered by knowledge units: %r"
            % (label, missing_source_ids)
        )
    excluded_source_ids = set(source_index) - included_source_ids
    unexpectedly_covered = sorted(excluded_source_ids & covered_source_ids)
    if unexpectedly_covered:
        raise ValueError(
            "%s excluded sources are covered by knowledge units: %r"
            % (label, unexpectedly_covered)
        )
    return source_index


def load_expert_review_documents(dataset_root: str) -> Dict[str, Dict[str, Any]]:
    review_dir = os.path.join(dataset_root, "expert_curation", "review")
    documents: Dict[str, Dict[str, Any]] = {}
    for piece_id in SONG_ORDER:
        path = os.path.join(review_dir, piece_id + ".json")
        value = load_json(path)
        validate_expert_review_document(value, path, piece_id)
        documents[piece_id] = value
    return documents


def build_source_index(dataset_root: str) -> Dict[str, Dict[str, Any]]:
    source_index: Dict[str, Dict[str, Any]] = {}
    for piece_id, document in load_expert_review_documents(dataset_root).items():
        for source in document["source_annotations"]:
            source_id = source["source_id"]
            if source_id in source_index:
                raise ValueError("Duplicate expert source id: %s" % source_id)
            indexed_source = dict(source)
            indexed_source["_piece"] = piece_id
            source_index[source_id] = indexed_source
    return source_index


def expert_review_warning(rewrite_status: str, measure_status: str) -> str:
    rewrite_pending = rewrite_status != "ready"
    measure_pending = measure_status == "waiting_for_review"
    if rewrite_pending and measure_pending:
        return "rewrite_and_measure_review_pending"
    if rewrite_pending:
        return "rewrite_review_pending"
    if measure_pending:
        return "measure_review_pending"
    return ""


def build_expert_records(
    dataset_root: str,
    source_index: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    documents = load_expert_review_documents(dataset_root)
    expected_source_ids = {
        source["source_id"]
        for document in documents.values()
        for source in document["source_annotations"]
    }
    if set(source_index) != expected_source_ids:
        raise ValueError("Expert source index does not exactly match review documents")
    for song, document in documents.items():
        for source in document["source_annotations"]:
            expected_source = dict(source)
            expected_source["_piece"] = song
            if source_index[source["source_id"]] != expected_source:
                raise ValueError(
                    "Expert source index differs from review source %s"
                    % source["source_id"]
                )

    for song in SONG_ORDER:
        document = documents[song]
        source_knowledge_unit_ids: Dict[str, List[str]] = {}
        for document_unit in document["knowledge_units"]:
            for document_source_id in document_unit["source_ids"]:
                source_knowledge_unit_ids.setdefault(
                    document_source_id,
                    [],
                ).append(document_unit["knowledge_unit_id"])
        path = os.path.join(
            dataset_root,
            "expert_curation",
            "review",
            song + ".json",
        )
        piece = PIECES[song]
        for unit_index, unit in enumerate(document["knowledge_units"], start=1):
            source_ids = list(unit["source_ids"])
            retrieval_aliases = []
            source_answer_context = []
            for source_id in source_ids:
                source_record = source_index[source_id]
                if source_record["_piece"] != song:
                    raise ValueError(
                        "%s links source %s from piece %s"
                        % (path, source_id, source_record["_piece"])
                    )
                if source_record["curation_status"] != "included":
                    raise ValueError("%s links excluded source %s" % (path, source_id))
                source_question = source_record["question"].strip()
                if source_question and source_question not in retrieval_aliases:
                    retrieval_aliases.append(source_question)
                source_answer_context.append(
                    {
                        "source_id": source_id,
                        "question": source_record["question"],
                        "answer": source_record["answer"],
                        "legacy_measure_range_hints": normalize_ranges(
                            source_record["legacy_measure_ranges"]
                        ),
                        "linked_knowledge_unit_ids": list(
                            source_knowledge_unit_ids[source_id]
                        ),
                    }
                )

            measure_status = unit["measure_status"]
            measure_range = (
                normalize_ranges(unit["measure_ranges"])
                if measure_status == "specific"
                else []
            )
            rewrite_status = unit["rewrite_status"]
            review_warning = expert_review_warning(
                rewrite_status,
                measure_status,
            )
            record = {
                "id": unit["knowledge_unit_id"],
                "evidence_type": "expert_annotation",
                "piece": song,
                "piece_title": piece["title"],
                "work": piece["work"],
                "composer": piece["composer"],
                "genre": piece["genre"],
                "language": piece["language"],
                "unit_index": unit_index,
                "topic": "",
                "measure_range": measure_range,
                "measure_scope": default_measure_scope(measure_range),
                "measure_range_hints": normalize_ranges(unit["measure_range_hints"]),
                "measure_status": measure_status,
                "measure_notes": unit["measure_notes"],
                "features": [],
                "applicability_note": "",
                "question": "",
                "question_source": "none",
                "retrieval_aliases": retrieval_aliases,
                "source_answer_context": source_answer_context,
                "answer": unit["answer"],
                "rewrite_status": rewrite_status,
                "rewrite_notes": unit["rewrite_notes"],
                "source_ids": source_ids,
                "annotators": annotators_from_sources(source_ids, source_index),
                "retrieval_eligible": True,
                "retrieval_exclusion_reason": "",
                "retrieval_review_warning": review_warning,
            }
            record["relevance_text"] = semantic_relevance_text(record)
            record["retrieval_text"] = retrieval_text(record)
            records.append(record)
    return records


def compact_web_source(source: Dict[str, Any]) -> Dict[str, Any]:
    citation = source.get("citation") or {}
    return {
        "web_source_id": source["web_source_id"],
        "title": source["title"],
        "url": source["canonical_url"],
        "institution": citation.get("publisher_or_institution", ""),
        "authority_tier": source.get("authority_tier"),
        "source_type": source.get("source_type", ""),
        "piece_ids": list(source["piece_ids"]),
    }


def build_web_source_index(dataset_root: str) -> Dict[str, Dict[str, Any]]:
    path = os.path.join(dataset_root, "database", "records", "sources.jsonl")
    source_index: Dict[str, Dict[str, Any]] = {}
    for source in iter_jsonl(path):
        source_id = source["web_source_id"]
        if source_id in source_index:
            raise ValueError("Duplicate web source id: %s" % source_id)
        piece_ids = source.get("piece_ids")
        if (
            not isinstance(piece_ids, list)
            or not piece_ids
            or piece_ids != sorted(set(piece_ids))
            or any(piece_id not in PIECES for piece_id in piece_ids)
        ):
            raise ValueError("%s has invalid piece_ids: %r" % (source_id, piece_ids))
        asset_ids = [asset.get("asset_id") for asset in source.get("assets", [])]
        if any(not asset_id for asset_id in asset_ids) or len(asset_ids) != len(set(asset_ids)):
            raise ValueError("%s has invalid or duplicate asset IDs" % source_id)
        source_index[source_id] = source
    return source_index


def resolve_claim_assets(
    claim: Dict[str, Any],
    source_index: Dict[str, Dict[str, Any]],
) -> List[tuple[Dict[str, Any], Dict[str, Any]]]:
    resolved = []
    for evidence_index, evidence in enumerate(claim.get("evidence", [])):
        source_id = evidence.get("web_source_id")
        if source_id not in source_index:
            raise ValueError(
                "%s evidence %d references unknown source %r"
                % (claim["claim_id"], evidence_index, source_id)
            )
        matching_assets = [
            asset
            for asset in source_index[source_id].get("assets", [])
            if asset.get("asset_id") == evidence.get("asset_id")
        ]
        if len(matching_assets) != 1:
            raise ValueError(
                "%s evidence %d does not resolve to one source asset"
                % (claim["claim_id"], evidence_index)
            )
        asset = matching_assets[0]
        if evidence.get("asset_type") != asset.get("asset_type"):
            raise ValueError(
                "%s evidence %d has an asset_type mismatch"
                % (claim["claim_id"], evidence_index)
            )
        resolved.append((evidence, asset))
    return resolved


def asset_research_eligible(asset: Dict[str, Any]) -> bool:
    permissions = asset.get("permissions")
    return (
        asset.get("research_class") in {"research_open", "research_conditional"}
        and isinstance(permissions, dict)
        and all(permissions.get(permission) is True for permission in RESEARCH_REQUIRED_PERMISSIONS)
    )


def effective_source_asset_class(asset: Dict[str, Any]) -> str:
    status = asset.get("rights_status")
    declared = asset.get("research_class")
    jurisdictions = asset.get("jurisdictions")
    if status in OPEN_RIGHTS_STATUSES and asset_research_eligible(asset):
        if declared == "research_conditional":
            return "research_conditional"
        if status == "kogl_type_1":
            return "research_open"
        if not isinstance(jurisdictions, list) or not jurisdictions:
            return "catalog_only"
        if not any(
            isinstance(jurisdiction, str)
            and jurisdiction.strip().casefold() == "worldwide"
            for jurisdiction in jurisdictions
        ):
            return "research_conditional"
        return "research_open"
    if status in CONDITIONAL_RIGHTS_STATUSES and asset_research_eligible(asset):
        return "research_conditional"
    if declared == "facts_only":
        return "facts_only"
    permissions = asset.get("permissions")
    if (
        declared == "demo_local"
        and status == "permission"
        and isinstance(permissions, dict)
        and all(permissions.get(permission) is True for permission in DEMO_REQUIRED_PERMISSIONS)
    ):
        return "demo_local"
    return "catalog_only"


def evidence_expression_allowed(usage_class: str, asset_class: str) -> bool:
    if usage_class == "research_open":
        return asset_class == "research_open"
    if usage_class == "research_conditional":
        return asset_class in {"research_open", "research_conditional"}
    if usage_class == "demo_local":
        return asset_class == "demo_local"
    return False


def validate_claim_asset_rights(
    claim: Dict[str, Any],
    source_index: Dict[str, Dict[str, Any]],
) -> None:
    resolved = resolve_claim_assets(claim, source_index)
    asset_classes = {effective_source_asset_class(asset) for _, asset in resolved}
    usage_class = claim.get("usage_class")
    required_class = {
        "research_open": "research_open",
        "research_conditional": "research_conditional",
        "demo_local": "demo_local",
    }.get(str(usage_class))
    if required_class and required_class not in asset_classes:
        raise ValueError(
            "%s %s claim has no effective %s asset"
            % (claim["claim_id"], usage_class, required_class)
        )
    retained_expression_classes = {
        effective_source_asset_class(asset)
        for evidence, asset in resolved
        if evidence.get("original_text") is not None
    }
    if claim.get("contains_source_expression") is True or (
        claim.get("claim_original") is not None and not retained_expression_classes
    ):
        retained_expression_classes = set(asset_classes)
    allowed_expression_classes = {
        "research_open": {"research_open"},
        "research_conditional": {"research_open", "research_conditional"},
        "demo_local": {"demo_local"},
    }.get(str(usage_class), set())
    if retained_expression_classes - allowed_expression_classes:
        raise ValueError(
            "%s retains claim-level expression from incompatible assets"
            % claim["claim_id"]
        )
    for evidence_index, (evidence, asset) in enumerate(resolved):
        if evidence.get("original_text") is None:
            continue
        asset_class = effective_source_asset_class(asset)
        if not evidence_expression_allowed(str(usage_class), asset_class):
            raise ValueError(
                "%s evidence %d retains expression from incompatible %s asset"
                % (claim["claim_id"], evidence_index, asset_class)
            )
    if usage_class in {"facts_only", "catalog_only"} and (
        claim.get("claim_original") is not None
        or any(evidence.get("original_text") is not None for evidence, _ in resolved)
    ):
        raise ValueError(
            "%s %s claim retains source expression" % (claim["claim_id"], usage_class)
        )


def build_web_claim_index(
    dataset_root: str,
    source_index: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    path = os.path.join(dataset_root, "database", "records", "claims.jsonl")
    claim_index: Dict[str, Dict[str, Any]] = {}
    for claim in iter_jsonl(path):
        claim_id = claim["claim_id"]
        if claim_id in claim_index:
            raise ValueError("Duplicate web claim id: %s" % claim_id)
        song = claim.get("piece_id")
        if song not in PIECES:
            raise ValueError("%s has unknown piece_id %r" % (claim_id, song))
        source_ids = claim.get("web_source_ids")
        if (
            not isinstance(source_ids, list)
            or not source_ids
            or source_ids != sorted(set(source_ids))
        ):
            raise ValueError("%s has invalid web_source_ids" % claim_id)
        missing_sources = [source_id for source_id in source_ids if source_id not in source_index]
        if missing_sources:
            raise ValueError("%s has unknown sources: %r" % (claim_id, missing_sources))
        wrong_piece_sources = [
            source_id
            for source_id in source_ids
            if song not in source_index[source_id]["piece_ids"]
        ]
        if wrong_piece_sources:
            raise ValueError(
                "%s links sources outside piece %s: %r"
                % (claim_id, song, wrong_piece_sources)
            )
        evidence_source_ids = sorted(
            {evidence.get("web_source_id") for evidence in claim.get("evidence", [])}
        )
        if evidence_source_ids != source_ids:
            raise ValueError(
                "%s web_source_ids do not equal its evidence source union" % claim_id
            )
        validate_claim_asset_rights(claim, source_index)
        claim_ranges = normalize_ranges(claim.get("measure_range", []))
        if claim_ranges != claim.get("measure_range", []):
            raise ValueError("%s has non-canonical measure ranges" % claim_id)
        if claim_ranges and (not claim.get("edition_id") or not claim.get("measure_system")):
            raise ValueError(
                "%s has measure ranges without an edition and measure system" % claim_id
            )
        claim_index[claim_id] = claim
    return claim_index


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def successful_web_retrieval(source: Dict[str, Any]) -> bool:
    retrieval = source.get("retrieval") or {}
    status = retrieval.get("http_status")
    return (
        retrieval.get("error") is None
        and isinstance(status, int)
        and not isinstance(status, bool)
        and 200 <= status < 400
        and isinstance(retrieval.get("retrieved_at"), str)
        and bool(retrieval["retrieved_at"].strip())
        and isinstance(retrieval.get("content_hash"), str)
        and retrieval["content_hash"].startswith("sha256:")
    )


def web_provider_identity(source: Dict[str, Any]) -> str:
    retrieval = source.get("retrieval") or {}
    adapter = str(retrieval.get("adapter") or "").casefold()
    institution = str(
        (source.get("citation") or {}).get("publisher_or_institution") or ""
    ).strip()
    institution_folded = institution.casefold()
    hostname = (
        urlsplit(str(source.get("canonical_url") or "")).hostname or ""
    ).casefold()
    if (
        adapter == "musicbrainz"
        or hostname == "musicbrainz.org"
        or hostname.endswith(".musicbrainz.org")
        or "musicbrainz" in institution_folded
        or "metabrainz" in institution_folded
    ):
        return "metabrainz"
    if (
        adapter in {"wikidata", "wikidata_json"}
        or hostname == "wikidata.org"
        or hostname.endswith(".wikidata.org")
        or "wikidata" in institution_folded
        or "wikimedia foundation" in institution_folded
    ):
        return "wikimedia"
    if (
        adapter in {"loc", "loc_sru"}
        or hostname == "loc.gov"
        or hostname.endswith(".loc.gov")
        or institution_folded.startswith("library of congress")
    ):
        return "library-of-congress"
    if institution:
        leading = re.split(r"\s*(?:/|;)\s*", institution, maxsplit=1)[0]
        normalized = "-".join(
            re.findall(r"[^\W_]+", leading.casefold(), flags=re.UNICODE)
        )
        if normalized:
            return "institution:%s" % normalized
    if hostname:
        return "host:%s" % hostname.removeprefix("www.")
    return "source:%s" % source.get("web_source_id", "unknown")


def facts_verification_provenance(
    claim: Dict[str, Any],
    source_index: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    successful = [
        source_index[source_id]
        for source_id in claim.get("web_source_ids", [])
        if successful_web_retrieval(source_index[source_id])
        and isinstance(source_index[source_id].get("authority_tier"), int)
        and source_index[source_id]["authority_tier"] <= 3
    ]
    tier_one_ids = sorted(
        source["web_source_id"]
        for source in successful
        if source["authority_tier"] == 1
    )
    provider_ids = sorted({web_provider_identity(source) for source in successful})
    status = claim.get("verification_status")
    if status == "verified_primary":
        sufficient = bool(tier_one_ids)
        rule = "one_successful_tier_one_source"
    elif status == "verified_multiple":
        sufficient = len(provider_ids) >= 2
        rule = "two_successful_independent_professional_providers"
    else:
        sufficient = False
        rule = "facts_only_release_requires_verified_primary_or_verified_multiple"
    return {
        "claim_verification_status": status,
        "required_rule": rule,
        "sufficient": sufficient,
        "successful_professional_source_ids": sorted(
            source["web_source_id"] for source in successful
        ),
        "tier_one_source_ids": tier_one_ids,
        "independent_provider_ids": provider_ids,
    }


def expected_facts_provenance(
    claim: Dict[str, Any],
    source_index: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    provenance = []
    for evidence, asset in resolve_claim_assets(claim, source_index):
        source = source_index[evidence["web_source_id"]]
        retrieval = source.get("retrieval") or {}
        provenance.append(
            {
                "web_source_id": evidence["web_source_id"],
                "asset_id": asset["asset_id"],
                "asset_type": asset.get("asset_type"),
                "effective_asset_class": effective_source_asset_class(asset),
                "authority_tier": source.get("authority_tier"),
                "provider_id": web_provider_identity(source),
                "canonical_url": source.get("canonical_url"),
                "locator": evidence.get("locator"),
                "retrieval_successful": successful_web_retrieval(source),
                "content_hash": retrieval.get("content_hash"),
                "rights_status": asset.get("rights_status"),
                "research_class": asset.get("research_class"),
            }
        )
    provenance.sort(
        key=lambda item: (
            str(item["web_source_id"]),
            str(item["asset_id"]),
            str(item["locator"]),
        )
    )
    return provenance


def load_facts_only_reviews(dataset_root: str) -> Dict[str, Dict[str, Any]]:
    path = os.path.join(dataset_root, "database", "facts-only-expression-reviews.json")
    if not os.path.exists(path):
        return {}
    reviews = load_json(path)
    if not isinstance(reviews, list):
        raise ValueError("Facts-only reviews must be a JSON list: %s" % path)
    review_index: Dict[str, Dict[str, Any]] = {}
    for review_number, review in enumerate(reviews, start=1):
        if not isinstance(review, dict) or set(review) != FACTS_REVIEW_KEYS:
            raise ValueError(
                "Facts-only review %d has an invalid field set" % review_number
            )
        claim_id = review.get("claim_id")
        if not isinstance(claim_id, str) or not claim_id.strip() or claim_id in review_index:
            raise ValueError("Invalid or duplicate facts-only review: %r" % claim_id)
        for hash_field in ("claim_record_sha256", "evidence_lineage_sha256"):
            digest = review.get(hash_field)
            if (
                not isinstance(digest, str)
                or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
            ):
                raise ValueError(
                    "Invalid %s in facts-only review %s" % (hash_field, claim_id)
                )
        if review.get("status") not in FACTS_REVIEW_STATUSES:
            raise ValueError("Invalid facts-only review status for %s" % claim_id)
        overlap_status = review.get("overlap_review_status")
        if overlap_status not in FACTS_OVERLAP_REVIEW_STATUSES:
            raise ValueError("Invalid overlap review status for %s" % claim_id)
        if (
            review["status"] == "approved"
            and overlap_status != "no_protected_expression_detected"
        ):
            raise ValueError(
                "Approved facts-only review %s must find no protected expression"
                % claim_id
            )
        for field in ("reviewer", "reviewed_at", "method"):
            value = review.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    "Facts-only review %s requires non-empty %s" % (claim_id, field)
                )
        if not isinstance(review.get("notes"), str):
            raise ValueError("Facts-only review %s notes must be a string" % claim_id)
        try:
            reviewed_at = datetime.fromisoformat(
                review["reviewed_at"].replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise ValueError(
                "Facts-only review %s has an invalid reviewed_at" % claim_id
            ) from exc
        if reviewed_at.utcoffset() is None:
            raise ValueError(
                "Facts-only review %s reviewed_at must include a timezone" % claim_id
            )
        review_index[claim_id] = review
    return review_index


def build_facts_only_audit_index(
    dataset_root: str,
    claim_index: Dict[str, Dict[str, Any]],
    source_index: Dict[str, Dict[str, Any]],
) -> Dict[str, str]:
    path = os.path.join(
        dataset_root,
        "database",
        "reports",
        "facts-only-release-audit.json",
    )
    audit = load_json(path)
    audit_records = audit.get("records") if isinstance(audit, dict) else None
    if not isinstance(audit_records, list):
        raise ValueError("Invalid facts-only release audit: %s" % path)
    review_index = load_facts_only_reviews(dataset_root)
    facts_claim_ids = {
        claim_id
        for claim_id, claim in claim_index.items()
        if claim.get("usage_class") == "facts_only"
    }
    if set(review_index) - facts_claim_ids:
        raise ValueError("Facts-only reviews reference non-facts claims")
    audit_index: Dict[str, str] = {}
    for record in audit_records:
        claim_id = record.get("claim_id")
        if claim_id in audit_index:
            raise ValueError("Duplicate claim in facts-only release audit: %s" % claim_id)
        claim = claim_index.get(claim_id)
        if claim is None or claim.get("usage_class") != "facts_only":
            raise ValueError("Facts-only audit references an invalid claim: %r" % claim_id)

        claim_hash = canonical_sha256(claim)
        provenance = expected_facts_provenance(claim, source_index)
        lineage_hash = canonical_sha256(provenance)
        if record.get("claim_record_sha256") != claim_hash:
            raise ValueError("Facts-only audit claim hash mismatch for %s" % claim_id)
        if record.get("evidence_lineage_sha256") != lineage_hash:
            raise ValueError("Facts-only audit lineage hash mismatch for %s" % claim_id)
        if record.get("provenance") != provenance:
            raise ValueError("Facts-only audit provenance mismatch for %s" % claim_id)
        if record.get("piece_id") != claim.get("piece_id"):
            raise ValueError("Facts-only audit piece mismatch for %s" % claim_id)
        expected_claim_text_hash = canonical_sha256(
            {"claim_ko": claim.get("claim_ko"), "value": claim.get("value")}
        )
        if record.get("claim_text_sha256") != expected_claim_text_hash:
            raise ValueError("Facts-only audit claim-text hash mismatch for %s" % claim_id)

        asset_types = sorted(
            {str(item["asset_type"]) for item in provenance if item["asset_type"] is not None}
        )
        potentially_expressive = sorted(set(asset_types) - {"metadata"})
        all_structured_metadata = bool(provenance) and not potentially_expressive
        expression_flags = {
            "declared_contains_source_expression": bool(
                claim.get("contains_source_expression")
            ),
            "claim_original_present": claim.get("claim_original") is not None,
            "evidence_original_text_present": any(
                evidence.get("original_text") is not None
                for evidence in claim.get("evidence", [])
            ),
            "claim_kind_is_factual": claim.get("claim_kind") == "factual",
            "rights_derivation_is_independently_worded_fact": (
                claim.get("rights_derivation") == "independently_worded_fact"
            ),
            "all_evidence_structured_metadata": all_structured_metadata,
            "potentially_expressive_asset_types": potentially_expressive,
        }
        if record.get("expression_flags") != expression_flags:
            raise ValueError("Facts-only audit expression flags mismatch for %s" % claim_id)
        verification = facts_verification_provenance(claim, source_index)
        if record.get("verification") != verification:
            raise ValueError("Facts-only audit verification mismatch for %s" % claim_id)

        review = review_index.get(claim_id)
        if review is None:
            if all_structured_metadata:
                expected_review = {
                    "status": "not_required_structured_metadata",
                    "overlap_review_status": "not_applicable_structured_metadata",
                    "hashes_match": None,
                    "reviewer": None,
                    "reviewed_at": None,
                    "method": "deterministic_structured_metadata_rule",
                    "notes": (
                        "Every referenced asset is metadata; release still depends "
                        "on all other factual, provenance, and expression gates."
                    ),
                }
            else:
                expected_review = {
                    "status": "missing",
                    "overlap_review_status": "not_reviewed",
                    "hashes_match": None,
                    "reviewer": None,
                    "reviewed_at": None,
                    "method": None,
                    "notes": (
                        "Potentially expressive evidence requires a human comparison "
                        "bound to the current claim and evidence hashes."
                    ),
                }
        else:
            hashes_match = (
                review.get("claim_record_sha256") == claim_hash
                and review.get("evidence_lineage_sha256") == lineage_hash
            )
            expected_review = {
                "status": review["status"] if hashes_match else "stale",
                "overlap_review_status": review["overlap_review_status"],
                "hashes_match": hashes_match,
                "reviewer": review["reviewer"],
                "reviewed_at": review["reviewed_at"],
                "method": review["method"],
                "notes": review["notes"],
            }
        if record.get("review") != expected_review:
            raise ValueError("Facts-only audit review mismatch for %s" % claim_id)
        expected_review_status = expected_review["status"]

        reasons = []
        if not expression_flags["claim_kind_is_factual"]:
            reasons.append("claim_kind_not_factual")
        if not expression_flags["rights_derivation_is_independently_worded_fact"]:
            reasons.append("rights_derivation_not_independently_worded_fact")
        if expression_flags["declared_contains_source_expression"]:
            reasons.append("declared_source_expression")
        if expression_flags["claim_original_present"]:
            reasons.append("claim_original_present")
        if expression_flags["evidence_original_text_present"]:
            reasons.append("evidence_original_text_present")
        if not verification["sufficient"]:
            reasons.append("verification_provenance_insufficient")
        if not all_structured_metadata and expected_review_status != "approved":
            reasons.append("expression_review_%s" % expected_review_status)
        if all_structured_metadata and expected_review_status in {"rejected", "stale"}:
            reasons.append("expression_review_%s" % expected_review_status)
        eligible = not reasons
        expected_decision = {
            "eligible": eligible,
            "export_class": "research_open" if eligible else "catalog_only",
            "reasons": reasons,
        }
        if record.get("release_decision") != expected_decision:
            raise ValueError("Facts-only audit decision mismatch for %s" % claim_id)
        audit_index[claim_id] = expected_decision["export_class"]

    if set(audit_index) != facts_claim_ids:
        raise ValueError(
            "Facts-only audit coverage mismatch; missing=%r unexpected=%r"
            % (
                sorted(facts_claim_ids - set(audit_index)),
                sorted(set(audit_index) - facts_claim_ids),
            )
        )
    return audit_index


def build_canonical_web_chunk_index(dataset_root: str) -> Dict[str, Dict[str, Any]]:
    path = os.path.join(dataset_root, "database", "records", "chunks.jsonl")
    chunk_index: Dict[str, Dict[str, Any]] = {}
    for chunk in iter_jsonl(path):
        chunk_id = chunk["chunk_id"]
        if chunk_id in chunk_index:
            raise ValueError("Duplicate canonical web chunk id: %s" % chunk_id)
        chunk_index[chunk_id] = chunk
    return chunk_index


def claim_export_class(
    claim: Dict[str, Any],
    facts_audit_index: Dict[str, str],
) -> str:
    if claim.get("verification_status") not in {
        "verified_primary",
        "verified_multiple",
        "attributed_interpretation",
    }:
        return "catalog_only"
    usage_class = claim.get("usage_class")
    if usage_class == "facts_only":
        return facts_audit_index.get(claim["claim_id"], "catalog_only")
    if usage_class in {"research_open", "research_conditional", "demo_local"}:
        return str(usage_class)
    return "catalog_only"


def expected_inherited_licenses(
    claims: Sequence[Dict[str, Any]],
    source_index: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    values: Dict[str, Dict[str, Any]] = {}
    for claim in claims:
        if claim.get("usage_class") == "facts_only":
            continue
        for evidence, asset in resolve_claim_assets(claim, source_index):
            if effective_source_asset_class(asset) not in {
                "research_open",
                "research_conditional",
                "demo_local",
            }:
                continue
            values[asset["asset_id"]] = {
                "web_source_id": evidence["web_source_id"],
                "asset_id": asset["asset_id"],
                "asset_type": asset["asset_type"],
                "license_id": asset.get("license_id")
                or asset.get("rights_status")
                or "unknown",
                "license_url": asset.get("license_url"),
                "attribution": asset.get("attribution"),
            }
    return [values[asset_id] for asset_id in sorted(values)]


def inherited_license_disclosures(
    inherited_licenses: Sequence[Dict[str, Any]],
    source_index: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Add governing conditions that canonical chunk license summaries omit."""

    disclosures = []
    for inherited in inherited_licenses:
        source = source_index.get(inherited.get("web_source_id"))
        if source is None:
            raise ValueError("Inherited license references an unknown source")
        asset = next(
            (
                candidate
                for candidate in source.get("assets", [])
                if candidate.get("asset_id") == inherited.get("asset_id")
            ),
            None,
        )
        if asset is None:
            raise ValueError("Inherited license references an unknown source asset")
        disclosure = dict(inherited)
        disclosure.update(
            {
                "rights_status": asset.get("rights_status"),
                "terms_url": asset.get("terms_url"),
                "jurisdictions": sorted(str(value) for value in asset.get("jurisdictions", [])),
                "permissions": {
                    key: bool(value)
                    for key, value in sorted((asset.get("permissions") or {}).items())
                },
            }
        )
        disclosures.append(disclosure)
    return disclosures


def asset_license_family(asset: Dict[str, Any]) -> Dict[str, Any]:
    jurisdictions = asset.get("jurisdictions")
    permissions = asset.get("permissions")
    if not isinstance(jurisdictions, list) or not isinstance(permissions, dict):
        raise ValueError("Asset has malformed rights family: %r" % asset.get("asset_id"))
    return {
        "rights_status": asset.get("rights_status"),
        "research_class": asset.get("research_class"),
        "license_id": asset.get("license_id")
        or asset.get("rights_status")
        or "unknown",
        "license_url": asset.get("license_url"),
        "terms_url": asset.get("terms_url"),
        "jurisdictions": sorted(
            " ".join(jurisdiction.split()).casefold()
            for jurisdiction in jurisdictions
        ),
        "permissions": {
            key: permissions[key]
            for key in sorted(permissions)
        },
    }


def conditional_claim_license_family(
    claim: Dict[str, Any],
    source_index: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    families: Dict[str, Dict[str, Any]] = {}
    for _, asset in resolve_claim_assets(claim, source_index):
        if effective_source_asset_class(asset) != "research_conditional":
            continue
        family = asset_license_family(asset)
        key = json.dumps(family, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        families.setdefault(key, family)
    if len(families) != 1:
        raise ValueError(
            "%s conditional claim has %d governing license families"
            % (claim["claim_id"], len(families))
        )
    return next(iter(families.values()))


def conditional_chunk_license_family(
    claims: Sequence[Dict[str, Any]],
    source_index: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    families: Dict[str, Dict[str, Any]] = {}
    for claim in claims:
        family = conditional_claim_license_family(claim, source_index)
        key = json.dumps(family, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        families.setdefault(key, family)
    if len(families) != 1:
        raise ValueError("Conditional chunk mixes incompatible governing license families")
    return next(iter(families.values()))


def expected_original_language_evidence(
    claims: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    values: Dict[tuple[str, str, str, str], Dict[str, Any]] = {}
    for claim in claims:
        for evidence in claim.get("evidence", []):
            original_text = evidence.get("original_text")
            if original_text is None:
                continue
            item = {
                "web_source_id": evidence["web_source_id"],
                "asset_id": evidence["asset_id"],
                "asset_type": evidence["asset_type"],
                "locator": evidence["locator"],
                "text": original_text,
            }
            key = (
                item["web_source_id"],
                item["asset_id"],
                item["locator"],
                item["text"],
            )
            values.setdefault(key, item)
    return [values[key] for key in sorted(values)]


def validate_web_chunk_lineage(
    chunk: Dict[str, Any],
    claim_index: Dict[str, Dict[str, Any]],
    source_index: Dict[str, Dict[str, Any]],
    facts_audit_index: Dict[str, str],
) -> None:
    chunk_id = chunk["chunk_id"]
    claim_ids = chunk.get("claim_ids")
    source_ids = chunk.get("web_source_ids")
    if not isinstance(claim_ids, list) or not claim_ids or claim_ids != sorted(set(claim_ids)):
        raise ValueError("%s has invalid claim_ids" % chunk_id)
    if not isinstance(source_ids, list) or not source_ids or source_ids != sorted(set(source_ids)):
        raise ValueError("%s has invalid web_source_ids" % chunk_id)
    unknown_claims = [claim_id for claim_id in claim_ids if claim_id not in claim_index]
    unknown_sources = [source_id for source_id in source_ids if source_id not in source_index]
    if unknown_claims or unknown_sources:
        raise ValueError(
            "%s has unknown claims=%r or sources=%r"
            % (chunk_id, unknown_claims, unknown_sources)
        )
    linked_claims = [claim_index[claim_id] for claim_id in claim_ids]
    expected_sources = sorted(
        {
            source_id
            for claim in linked_claims
            for source_id in claim["web_source_ids"]
        }
    )
    if source_ids != expected_sources:
        raise ValueError("%s web_source_ids do not equal linked-claim source union" % chunk_id)
    song = chunk.get("piece_id")
    if song not in PIECES:
        raise ValueError("%s has unknown piece_id %r" % (chunk_id, song))
    wrong_piece_sources = [
        source_id for source_id in source_ids if song not in source_index[source_id]["piece_ids"]
    ]
    if wrong_piece_sources:
        raise ValueError(
            "%s links sources outside piece %s: %r"
            % (chunk_id, song, wrong_piece_sources)
        )
    for claim in linked_claims:
        validate_claim_asset_rights(claim, source_index)
    expected_classes = {
        claim_export_class(claim, facts_audit_index) for claim in linked_claims
    }
    if expected_classes != {chunk.get("usage_class")}:
        raise ValueError(
            "%s usage_class does not match linked claims: %r"
            % (chunk_id, sorted(expected_classes))
        )
    chunk_ranges = normalize_ranges(chunk.get("measure_range", []))
    if chunk_ranges != chunk.get("measure_range", []):
        raise ValueError("%s has non-canonical measure ranges" % chunk_id)
    if chunk_ranges and (not chunk.get("edition_id") or not chunk.get("measure_system")):
        raise ValueError(
            "%s has measure ranges without an edition and measure system" % chunk_id
        )
    shared_fields = (
        "piece_id",
        "knowledge_scope",
        "measure_range",
        "measure_system",
        "edition_id",
        "edition_context",
    )
    for claim in linked_claims:
        for field in shared_fields:
            if claim.get(field) != chunk.get(field):
                raise ValueError(
                    "%s linked claim %s has incompatible %s"
                    % (chunk_id, claim["claim_id"], field)
                )
        expected_topic = WEB_CLAIM_TOPIC_TO_CHUNK_TOPIC.get(claim["topic"])
        if expected_topic is None:
            expected_topic = "other_documented_topics"
        if expected_topic != chunk.get("topic"):
            raise ValueError(
                "%s linked claim %s has incompatible topic"
                % (chunk_id, claim["claim_id"])
            )
    expected_subtopics = sorted({claim["topic"] for claim in linked_claims})
    if chunk.get("subtopics") != expected_subtopics:
        raise ValueError("%s subtopics do not match linked claims" % chunk_id)
    expected_text = " ".join(claim["claim_ko"].strip() for claim in linked_claims)
    if chunk.get("text_ko") != expected_text:
        raise ValueError("%s text_ko does not match linked claims" % chunk_id)
    if (
        chunk.get("question_source") == "generated"
        and chunk.get("question") != "%s에 대해 알려 주세요." % chunk.get("topic")
    ):
        raise ValueError("%s generated question is not deterministic" % chunk_id)
    expected_licenses = expected_inherited_licenses(linked_claims, source_index)
    if chunk.get("inherited_licenses") != expected_licenses:
        raise ValueError("%s inherited licenses do not match claim assets" % chunk_id)
    governing_family = None
    if chunk.get("usage_class") == "research_conditional":
        governing_family = conditional_chunk_license_family(
            linked_claims,
            source_index,
        )
    expected_original = expected_original_language_evidence(linked_claims)
    if chunk.get("original_language_evidence") != expected_original:
        raise ValueError("%s original-language evidence does not match claims" % chunk_id)
    if (
        any(claim.get("contains_source_expression") for claim in linked_claims)
        and chunk.get("usage_class") == "research_conditional"
    ):
        assert governing_family is not None
        expected_generated_license = governing_family["license_id"]
    else:
        expected_generated_license = "CC-BY-4.0"
    if chunk.get("generated_text_license") != expected_generated_license:
        raise ValueError("%s generated text license does not match claim rights" % chunk_id)


def build_web_records(
    dataset_root: str,
    export_files: Sequence[str],
    source_index: Dict[str, Dict[str, Any]],
    claim_index: Dict[str, Dict[str, Any]],
    facts_audit_index: Dict[str, str],
    canonical_chunk_index: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    seen_ids = set()
    export_dir = os.path.join(dataset_root, "database", "exports")
    for export_file in export_files:
        if export_file not in WEB_EXPORT_USAGE_CLASSES:
            raise ValueError(
                "Unsupported web export %r; only answer-eligible release partitions are allowed"
                % export_file
            )
        expected_usage_class = WEB_EXPORT_USAGE_CLASSES[export_file]
        path = os.path.join(export_dir, export_file)
        for chunk in iter_jsonl(path):
            chunk_id = chunk["chunk_id"]
            if chunk_id in seen_ids:
                raise ValueError("Duplicate web chunk id across exports: %s" % chunk_id)
            seen_ids.add(chunk_id)
            if canonical_chunk_index.get(chunk_id) != chunk:
                raise ValueError(
                    "%s export record does not exactly match its canonical chunk" % chunk_id
                )
            if chunk.get("usage_class") != expected_usage_class:
                raise ValueError(
                    "%s has usage_class %r in %s"
                    % (chunk_id, chunk.get("usage_class"), export_file)
                )
            validate_web_chunk_lineage(
                chunk,
                claim_index,
                source_index,
                facts_audit_index,
            )
            song = chunk["piece_id"]
            web_source_ids = list(chunk["web_source_ids"])
            measure_range = normalize_ranges(chunk.get("measure_range", []))
            piece = PIECES[song]
            record = {
                "id": chunk_id,
                "evidence_type": "web_database",
                "piece": song,
                "piece_title": piece["title"],
                "work": piece["work"],
                "composer": piece["composer"],
                "genre": piece["genre"],
                "language": piece["language"],
                "topic": chunk["topic"],
                "subtopics": list(chunk.get("subtopics", [])),
                "knowledge_scope": chunk["knowledge_scope"],
                "measure_range": measure_range,
                "measure_scope": default_measure_scope(measure_range),
                "measure_system": chunk.get("measure_system"),
                "edition_id": chunk.get("edition_id"),
                "edition_context": chunk.get("edition_context"),
                "question": chunk.get("question", ""),
                "question_source": chunk.get("question_source", "generated"),
                "answer": chunk["text_ko"],
                "claim_ids": list(chunk["claim_ids"]),
                "web_source_ids": web_source_ids,
                "sources": [
                    compact_web_source(source_index[source_id])
                    for source_id in web_source_ids
                ],
                "usage_class": chunk["usage_class"],
                "generated_text_license": chunk.get("generated_text_license"),
                "inherited_licenses": inherited_license_disclosures(
                    chunk.get("inherited_licenses", []),
                    source_index,
                ),
                "retrieval_eligible": True,
                "retrieval_exclusion_reason": "",
            }
            record["relevance_text"] = semantic_relevance_text(record)
            record["retrieval_text"] = retrieval_text(record)
            records.append(record)
    selected_usage_classes = {
        WEB_EXPORT_USAGE_CLASSES[export_file] for export_file in export_files
    }
    expected_chunk_ids = {
        chunk_id
        for chunk_id, chunk in canonical_chunk_index.items()
        if chunk.get("usage_class") in selected_usage_classes
    }
    if seen_ids != expected_chunk_ids:
        raise ValueError(
            "Selected web exports do not exactly cover canonical answer-eligible chunks; "
            "missing=%r unexpected=%r"
            % (sorted(expected_chunk_ids - seen_ids), sorted(seen_ids - expected_chunk_ids))
        )
    return records


def build_stats(
    records: List[Dict[str, Any]],
    overrides_applied: int,
    web_export_files: Sequence[str],
    dataset_root: str,
    input_fingerprint: str,
    corpus_sha256: str,
) -> Dict[str, Any]:
    excluded = [record for record in records if not record.get("retrieval_eligible", True)]
    expert_records = [
        record for record in records if record["evidence_type"] == "expert_annotation"
    ]
    excluded_expert_records = [
        record for record in expert_records if not record["retrieval_eligible"]
    ]
    warned_expert_records = [
        record
        for record in expert_records
        if record.get("retrieval_review_warning")
    ]
    return {
        "corpus_schema_version": 6,
        "dataset_root": os.path.realpath(dataset_root),
        "input_fingerprint": input_fingerprint,
        "corpus_sha256": corpus_sha256,
        "total_records": len(records),
        "retrievable_records": len(records) - len(excluded),
        "records_by_evidence_type": dict(
            sorted(Counter(record["evidence_type"] for record in records).items())
        ),
        "records_by_piece": dict(sorted(Counter(record["piece"] for record in records).items())),
        "records_by_measure_scope": dict(
            sorted(Counter(record["measure_scope"] for record in records).items())
        ),
        "expert_records_by_question_source": dict(
            sorted(
                Counter(
                    record["question_source"]
                    for record in expert_records
                ).items()
            )
        ),
        "expert_records_by_rewrite_status": dict(
            sorted(Counter(record["rewrite_status"] for record in expert_records).items())
        ),
        "expert_records_by_measure_status": dict(
            sorted(Counter(record["measure_status"] for record in expert_records).items())
        ),
        "expert_records_by_retrieval_exclusion_reason": dict(
            sorted(
                Counter(
                    record["retrieval_exclusion_reason"]
                    for record in excluded_expert_records
                ).items()
            )
        ),
        "expert_records_by_retrieval_review_warning": dict(
            sorted(
                Counter(
                    record["retrieval_review_warning"]
                    for record in warned_expert_records
                ).items()
            )
        ),
        "web_records_by_usage_class": dict(
            sorted(
                Counter(
                    record["usage_class"]
                    for record in records
                    if record["evidence_type"] == "web_database"
                ).items()
            )
        ),
        "excluded_from_retrieval": [
            {
                "id": record["id"],
                "reason": record["retrieval_exclusion_reason"],
                "source_ids": record.get("source_ids", []),
            }
            for record in excluded
        ],
        "expert_retrieval_review_warnings": [
            {
                "id": record["id"],
                "warning": record["retrieval_review_warning"],
                "source_ids": record.get("source_ids", []),
            }
            for record in warned_expert_records
        ],
        "web_export_files": list(web_export_files),
        "overrides_applied": overrides_applied,
    }


def build_corpus(settings: Dict[str, Any]) -> Dict[str, Any]:
    dataset_root = settings["dataset_root"]
    input_fingerprint = corpus_input_fingerprint(settings)
    expert_source_index = build_source_index(dataset_root)
    expert_records = build_expert_records(dataset_root, expert_source_index)
    overrides = load_overrides(settings["feature_overrides_path"])
    overrides_applied = apply_overrides(expert_records, overrides)

    web_export_files = settings.get("web_export_files", ["research-open.jsonl"])
    web_source_index = build_web_source_index(dataset_root)
    web_claim_index = build_web_claim_index(dataset_root, web_source_index)
    facts_audit_index = build_facts_only_audit_index(
        dataset_root,
        web_claim_index,
        web_source_index,
    )
    canonical_chunk_index = build_canonical_web_chunk_index(dataset_root)
    web_records = build_web_records(
        dataset_root,
        web_export_files,
        web_source_index,
        web_claim_index,
        facts_audit_index,
        canonical_chunk_index,
    )
    records = expert_records + web_records
    record_ids = [record["id"] for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("Duplicate IDs in combined corpus")

    if corpus_input_fingerprint(settings) != input_fingerprint:
        raise RuntimeError("Corpus inputs changed while the corpus was being built")
    corpus_payload = serialize_json(records)
    corpus_sha256 = hashlib.sha256(corpus_payload).hexdigest()
    stats = build_stats(
        records,
        overrides_applied,
        web_export_files,
        dataset_root,
        input_fingerprint,
        corpus_sha256,
    )
    atomic_write(settings["corpus_path"], corpus_payload)
    atomic_write(settings["stats_path"], serialize_json(stats))
    return {"records": records, "stats": stats}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", default=None, help="Path to settings JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings(args.settings) if args.settings else load_settings()
    result = build_corpus(settings)
    print("Wrote %d records to %s" % (len(result["records"]), settings["corpus_path"]))
    print("Wrote stats to %s" % settings["stats_path"])


if __name__ == "__main__":
    main()
