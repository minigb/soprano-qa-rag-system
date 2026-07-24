#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ask the local measure-aware Soprano QA system."""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
from typing import Any, Callable, Dict, List

from soprano_qa.corpus import (
    PIECES,
    SONG_ORDER,
    build_corpus,
    corpus_input_fingerprint,
    file_sha256,
)
from soprano_qa.llm import generate
from soprano_qa.retrieval import (
    BM25Index,
    format_measure_range,
    format_result,
    load_corpus,
    parse_measure_ranges,
    validate_question_measure_contract,
)
from soprano_qa.settings import load_settings


SYSTEM_PROMPT = """You are a measure-aware soprano performance QA assistant.

Answer in Korean.
Do not reveal chain-of-thought, hidden reasoning, or <think> blocks. Output the final answer only.
Use only the retrieved evidence below. It has two explicitly labeled lineages:
- expert_annotation: human expert performance commentary.
- web_database: externally collected background or performance evidence with source provenance.
Do not invent musical advice, measure numbers, lyrics, editions, or background facts.
If the evidence does not answer the core question at all, output exactly
<NO_GROUNDED_ANSWER> and nothing else. If only a secondary part is unsupported, answer the
supported core and briefly say what is missing.
Preserve source qualifiers: never turn a fact about one edition, recording, performance, or attributed
commentary into a universal fact about every version of the work.
For a measure-range query, never apply advice from another measure. General evidence may add context,
but web evidence without an edition-qualified measure system cannot support a claim about a specific bar.
When no measure range is supplied, the question concerns the whole song. Prefer general evidence; if a
local example is useful, name its measures and do not generalize it to the entire song.
Keep expert source_ids separate from web_source_ids and claim_ids.
Mention the most relevant measures. Cite only the short evidence labels exactly as shown, such as [E1];
never retype an opaque sqa/webchunk id. The application maps labels to exact record ids after generation.
Do not hide or contradict source attribution and license conditions; the application prints deterministic
evidence notices after the answer for every retrieved web record.
Keep the answer concise and practical for a singer.
"""
INTERNAL_KNOWLEDGE_SYSTEM_PROMPT = """You are a soprano performance QA assistant.

Answer in Korean.
Do not reveal chain-of-thought, hidden reasoning, or <think> blocks. Output the final answer only.
No matching corpus evidence is available for this question. Answer directly from your pretrained
general musical knowledge instead. The user's exact question is authoritative: do not replace it
with a nearby singing topic, and ignore piece or measure metadata when it is not relevant. Never
claim that you inspected the score, annotation database, or selected measures. If the answer
depends on a particular edition, score marking, or exact bar, state that limitation briefly and
provide only safe general guidance. Do not invent exact measure facts, quotations, sources, or
bibliographic details. For a generic technique question, explain the named technique itself rather
than redirecting to the selected piece's diction or interpretation.
Do not emit evidence labels, corpus IDs, or citations such as [E1], [sqa-0001], or [webchunk-*].
Do not respond with a corpus-insufficiency refusal; make a useful best-effort answer while clearly
qualifying uncertainty when needed.
Keep the answer concise and practical for a singer.
"""
OLD_INSUFFICIENT_EVIDENCE_MESSAGE = (
    "현재 답변 가능 자료에서 이 질문을 뒷받침할 근거를 찾지 못했습니다."
)
NO_GROUNDED_ANSWER_SENTINEL = "<NO_GROUNDED_ANSWER>"
NO_CORPUS_EVIDENCE_MESSAGE = "검색된 코퍼스 근거가 없습니다."
INTERNAL_GENERATION_UNAVAILABLE_MESSAGE = (
    "로컬 LLM을 사용할 수 없어 일반 지식 답변을 생성하지 못했습니다."
)
GENERATED_SUMMARY_ATTRIBUTION = "Soprano QA database project"
CC_BY_4_URL = "https://creativecommons.org/licenses/by/4.0/"


def build_context(results: List[Any]) -> str:
    chunks = []
    for idx, result in enumerate(results, start=1):
        record = result.record
        lines = [
            "Evidence %d" % idx,
            "citation_label: E%d" % idx,
            "id: %s" % record["id"],
            "evidence_type: %s" % record.get("evidence_type", "unknown"),
            "piece: %s" % record["piece"],
            "work: %s" % record["work"],
            "topic: %s" % record["topic"],
            "measure_range: %s" % format_measure_range(record.get("measure_range") or []),
            "measure_scope: %s" % record.get("measure_scope", ""),
            "query_scope_relation: %s" % result.scope_match,
        ]
        if record.get("evidence_type") == "expert_annotation":
            lines.extend(
                [
                    "expert_source_ids: %s" % ", ".join(record["source_ids"]),
                    "annotators: %s" % ", ".join(record.get("annotators", [])),
                ]
            )
        elif record.get("evidence_type") == "web_database":
            lines.extend(
                [
                    "knowledge_scope: %s" % record.get("knowledge_scope", ""),
                    "usage_class: %s" % record.get("usage_class", ""),
                    "generated_text_license: %s"
                    % (record.get("generated_text_license") or "(none)"),
                    "claim_ids: %s" % ", ".join(record.get("claim_ids", [])),
                    "web_source_ids: %s" % ", ".join(record.get("web_source_ids", [])),
                    "edition_id: %s" % (record.get("edition_id") or "(none)"),
                    "edition_context: %s"
                    % json.dumps(record.get("edition_context"), ensure_ascii=False, sort_keys=True),
                    "measure_system: %s"
                    % json.dumps(record.get("measure_system"), ensure_ascii=False, sort_keys=True),
                ]
            )
            for source in record.get("sources", []):
                lines.append(
                    "web_source: %s | %s | %s | %s"
                    % (
                        source["web_source_id"],
                        source.get("title", ""),
                        source.get("institution", ""),
                        source.get("url", ""),
                    )
                )
            for inherited in record.get("inherited_licenses", []):
                allowed = ",".join(
                    permission
                    for permission, value in sorted(
                        (inherited.get("permissions") or {}).items()
                    )
                    if value
                )
                lines.append(
                    "license_condition: source=%s; asset=%s (%s); license=%s; "
                    "rights=%s; jurisdictions=%s; attribution=%s; license_url=%s; "
                    "terms_url=%s; allowed=%s"
                    % (
                        inherited.get("web_source_id") or "(unknown)",
                        inherited.get("asset_id") or "(unknown)",
                        inherited.get("asset_type") or "(unknown)",
                        inherited.get("license_id") or "(unknown)",
                        inherited.get("rights_status") or "(unknown)",
                        ",".join(inherited.get("jurisdictions") or []) or "unspecified",
                        inherited.get("attribution") or "(none recorded)",
                        inherited.get("license_url") or "(none)",
                        inherited.get("terms_url") or "(none)",
                        allowed or "none recorded",
                    )
                )
        lines.extend(
            [
                "question: %s" % (record.get("question") or "(standalone tip)"),
                "answer: %s" % record["answer"],
            ]
        )
        chunks.append("\n".join(lines))
    return "\n\n---\n\n".join(chunks) or "(no relevant evidence retrieved)"


def build_messages(piece: str | None, measures: str, question: str, results: List[Any]) -> List[Dict[str, str]]:
    user_prompt = """User query
piece: {piece}
measure_range: {measures}
question: {question}
/no_think

Retrieved evidence
{context}
""".format(
        piece=piece or "(not specified)",
        measures=measures or "(none; whole-song question)",
        question=question,
        context=build_context(results),
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_internal_knowledge_messages(
    piece: str | None,
    measures: str,
    question: str,
) -> List[Dict[str, str]]:
    """Build a separate, explicitly ungrounded prompt for no-hit questions."""

    piece_metadata = PIECES.get(piece or "", {})
    user_prompt = """User context
piece_id: {piece_id}
title: {title}
work: {work}
composer: {composer}
selected_measure_range: {measures}
question: {question}
/no_think

Answer from internal general musical knowledge. The selected range is user context only, not
evidence that you have inspected those measures. Answer the exact question rather than substituting
a related corpus topic.
""".format(
        piece_id=piece or "(not specified)",
        title=piece_metadata.get("title") or "(unknown)",
        work=piece_metadata.get("work") or "(unknown)",
        composer=piece_metadata.get("composer") or "(unknown)",
        measures=measures or "(none; whole-song question)",
        question=question,
    )
    return [
        {"role": "system", "content": INTERNAL_KNOWLEDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def finalize_internal_knowledge_answer(answer: str) -> str:
    """Remove corpus-style provenance claims from an ungrounded model answer."""

    finalized = answer.replace(OLD_INSUFFICIENT_EVIDENCE_MESSAGE, "")
    finalized = finalized.replace(NO_GROUNDED_ANSWER_SENTINEL, "")
    citation_label = (
        r"(?:E(?:vidence)?\s*[0-9]+|sqa-[0-9]+|"
        r"webchunk-[A-Za-z0-9*-]+)"
    )
    finalized = re.sub(
        r"[\[【]\s*"
        r"(?:(?:출처|근거|source|citation)\s*[:：-]?\s*)?"
        + citation_label
        + r"(?:\s*[,;，、]\s*"
        + citation_label
        + r")*\s*[\]】]",
        "",
        finalized,
        flags=re.IGNORECASE,
    )
    finalized = re.sub(
        r"\b(?:sqa-[0-9]+|webchunk-[A-Za-z0-9*-]+)\b",
        "",
        finalized,
        flags=re.IGNORECASE,
    )
    finalized = re.sub(r"[ \t]+\n", "\n", finalized)
    finalized = re.sub(r"\n{3,}", "\n\n", finalized).strip()
    return finalized


def is_grounded_insufficiency_answer(answer: str) -> bool:
    """Recognize the grounded model's no-answer signal and legacy refusal."""

    normalized = re.sub(r"\s+", "", answer)
    if NO_GROUNDED_ANSWER_SENTINEL.lower() in normalized.lower():
        return True
    legacy_refusal = re.sub(r"\s+", "", OLD_INSUFFICIENT_EVIDENCE_MESSAGE)
    return legacy_refusal in normalized


def build_extractive_answer(results: List[Any]) -> str:
    if not results:
        return NO_CORPUS_EVIDENCE_MESSAGE
    return "\n\n".join(
        "%s [%s]" % (result.record["answer"].strip(), result.record["id"])
        for result in results[:3]
    )


def is_context_overflow_error(exc: ValueError) -> bool:
    message = str(exc).lower()
    return "context window" in message and (
        "exceed" in message or "requested tokens" in message
    )


def generate_with_context_retry(
    index: BM25Index,
    *,
    query: str,
    piece: str,
    measure_ranges: List[List[int]],
    measures: str,
    topic: str | None,
    top_k: int,
    generator: Callable[[List[Dict[str, str]]], str],
) -> tuple[str, List[Any], List[Dict[str, str]], bool]:
    """Retry with fewer whole records when llama.cpp reports context overflow."""

    search_limit = top_k
    context_limited = False
    last_results: List[Any] = []
    last_messages = build_messages(piece, measures, query, [])
    while search_limit > 0:
        results = index.search(
            query=query,
            piece=piece,
            measure_ranges=measure_ranges,
            topic=topic,
            top_k=search_limit,
        )
        messages = build_messages(piece, measures, query, results)
        if not results:
            if context_limited and last_results:
                return "", last_results, last_messages, context_limited
            return "", results, messages, context_limited
        last_results = results
        last_messages = messages
        try:
            return generator(messages), results, messages, context_limited
        except ValueError as exc:
            if not is_context_overflow_error(exc):
                raise
            context_limited = True
            search_limit = len(results) - 1
    return "", last_results, last_messages, context_limited


def build_evidence_notices(results: List[Any]) -> List[Dict[str, Any]]:
    notices = []
    seen_ids = set()
    for result in results:
        record = result.record
        if record.get("evidence_type") != "web_database" or record["id"] in seen_ids:
            continue
        seen_ids.add(record["id"])
        notices.append(
            {
                "record_id": record["id"],
                "usage_class": record.get("usage_class"),
                "generated_text_license": record.get("generated_text_license"),
                "generated_text_attribution": GENERATED_SUMMARY_ATTRIBUTION,
                "generated_text_license_url": CC_BY_4_URL,
                "inherited_licenses": list(record.get("inherited_licenses", [])),
                "sources": list(record.get("sources", [])),
            }
        )
    return notices


def format_evidence_notice(notice: Dict[str, Any]) -> str:
    lines = [
        "[%s] usage_class=%s; generated_text_license=%s; "
        "generated_text_attribution=%s; generated_text_license_url=%s"
        % (
            notice["record_id"],
            notice.get("usage_class") or "(none)",
            notice.get("generated_text_license") or "(none)",
            notice.get("generated_text_attribution") or "(none)",
            notice.get("generated_text_license_url") or "(none)",
        )
    ]
    for inherited in notice.get("inherited_licenses", []):
        jurisdictions = ",".join(inherited.get("jurisdictions") or []) or "unspecified"
        permissions = ",".join(
            permission
            for permission, allowed in sorted(
                (inherited.get("permissions") or {}).items()
            )
            if allowed
        ) or "none recorded"
        lines.append(
            "  inherited: source=%s | asset=%s (%s) | %s | "
            "rights_status: %s | jurisdiction: %s | "
            "attribution: %s | license: %s | terms: %s | allowed: %s"
            % (
                inherited.get("web_source_id") or "(unknown)",
                inherited.get("asset_id") or "(unknown)",
                inherited.get("asset_type") or "(unknown)",
                inherited.get("license_id") or "terms recorded by source",
                inherited.get("rights_status") or "unknown",
                jurisdictions,
                inherited.get("attribution") or "(none recorded)",
                inherited.get("license_url") or "(no license URL)",
                inherited.get("terms_url") or "(no terms URL)",
                permissions,
            )
        )
    for source in notice.get("sources", []):
        lines.append(
            "  source: %s | %s | %s"
            % (
                source["web_source_id"],
                source.get("title") or "(untitled)",
                source.get("url") or "(no URL)",
            )
        )
    return "\n".join(lines)


def finalize_answer_citations(answer: str, results: List[Any]) -> str:
    record_ids = [result.record["id"] for result in results]
    if not record_ids:
        return answer
    provenance_ids = {
        provenance_id
        for result in results
        for key in ("source_ids", "web_source_ids", "claim_ids")
        for provenance_id in result.record.get(key, [])
    }

    def label_to_citation(label_number: int) -> str:
        result_index = label_number - 1
        if 0 <= result_index < len(record_ids):
            return "[%s]" % record_ids[result_index]
        return ""

    def replace_short_label_group(match: re.Match[str]) -> str:
        citations = [
            label_to_citation(int(label_number))
            for label_number in re.findall(r"E([0-9]+)", match.group(1), flags=re.IGNORECASE)
        ]
        return " ".join(citation for citation in citations if citation)

    finalized = re.sub(
        r"\[((?:\s*E[0-9]+\s*)(?:[,;]\s*E[0-9]+\s*)+)(?:[,;]\s*)?\]",
        replace_short_label_group,
        answer,
        flags=re.IGNORECASE,
    )

    def replace_short_label(match: re.Match[str]) -> str:
        return label_to_citation(int(match.group(1)))

    finalized = re.sub(
        r"\[\s*E([0-9]+)\s*[,;]?\s*\]",
        replace_short_label,
        finalized,
        flags=re.IGNORECASE,
    )
    valid_ids = set(record_ids)

    def repair_opaque_id(match: re.Match[str]) -> str:
        cited_id = match.group(1)
        if cited_id in valid_ids:
            return match.group(0)
        ranked = sorted(
            (
                (difflib.SequenceMatcher(None, cited_id, record_id).ratio(), record_id)
                for record_id in record_ids
            ),
            reverse=True,
        )
        if ranked and ranked[0][0] >= 0.88:
            if len(ranked) == 1 or ranked[0][0] - ranked[1][0] >= 0.05:
                return "[%s]" % ranked[0][1]
        return ""

    finalized = re.sub(
        r"\[((?:sqa-[0-9]+)|(?:webchunk-[A-Za-z0-9-]+))\]",
        repair_opaque_id,
        finalized,
    )

    def remove_unknown_citation_like_bracket(match: re.Match[str]) -> str:
        content = match.group(1).strip()
        if content in valid_ids:
            return "[%s]" % content
        if any(provenance_id in content for provenance_id in provenance_ids):
            return ""
        if re.fullmatch(r"[0-9]+(?:\s*[,;]\s*[0-9]+)*", content):
            return ""
        if re.match(
            r"^(?:출처|근거|source|citation)\s*(?::|：|-)?",
            content,
            flags=re.IGNORECASE,
        ):
            return ""
        if re.match(
            r"^(?:E\s*[0-9]|Evidence\b|sqa(?:[-_]|[0-9])|"
            r"web(?:chunk|src|source|claim)(?:[-_])|source[_ -]?id\b)",
            content,
            flags=re.IGNORECASE,
        ):
            return ""
        return match.group(0)

    finalized = re.sub(
        r"\[([^\[\]]+)\]",
        remove_unknown_citation_like_bracket,
        finalized,
    )
    if not any("[%s]" % record_id in finalized for record_id in record_ids):
        generated_text = finalized.strip()
        if not generated_text:
            return "생성된 답변에서 유효한 내용을 확인하지 못했습니다."
        # Some local-model generations follow the evidence but omit the label.
        # Preserve that useful answer while adding a deterministic disclosure of
        # every retrieved record supplied to the model. Invalid model-produced
        # labels have already been removed above.
        evidence_footer = "제공된 검색 근거: " + " ".join(
            "[%s]" % record_id for record_id in record_ids
        )
        return generated_text + "\n\n" + evidence_footer
    return finalized.strip()


def ensure_corpus(settings: Dict[str, Any], rebuild: bool) -> None:
    stats_are_current = False
    expected_web_exports = settings.get("web_export_files", ["research-open.jsonl"])
    if os.path.exists(settings["stats_path"]):
        try:
            with open(settings["stats_path"], encoding="utf-8") as f:
                stats = json.load(f)
            stats_are_current = (
                isinstance(stats, dict)
                and stats.get("corpus_schema_version") == 4
                and stats.get("web_export_files") == expected_web_exports
                and stats.get("dataset_root") == os.path.realpath(settings["dataset_root"])
                and stats.get("input_fingerprint") == corpus_input_fingerprint(settings)
                and os.path.isfile(settings["corpus_path"])
                and stats.get("corpus_sha256") == file_sha256(settings["corpus_path"])
            )
        except (OSError, ValueError, TypeError):
            stats_are_current = False
    if rebuild or not os.path.exists(settings["corpus_path"]) or not stats_are_current:
        build_corpus(settings)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", default=None, help="Path to settings JSON")
    parser.add_argument("--piece", required=True, choices=SONG_ORDER, help="Piece slug")
    parser.add_argument(
        "--measures",
        default="",
        help='Optional inclusive measure query like "28-30" or "12, 35"; omit for whole-song questions',
    )
    parser.add_argument("--topic", help="Optional topic substring filter")
    parser.add_argument("--question", required=True, help="Singer/user question")
    parser.add_argument("--top-k", type=positive_int, default=6, help="Number of chunks to retrieve")
    parser.add_argument("--no-generate", action="store_true", help="Only retrieve evidence")
    parser.add_argument("--rebuild-corpus", action="store_true", help="Rebuild corpus before asking")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()
    try:
        args.measure_ranges = parse_measure_ranges(args.measures)
        validate_question_measure_contract(args.question, args.measure_ranges)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main() -> None:
    args = parse_args()
    settings = load_settings(args.settings) if args.settings else load_settings()
    ensure_corpus(settings, args.rebuild_corpus)

    records = load_corpus(settings["corpus_path"])
    index = BM25Index(records)
    generation_mode = "retrieval_only" if args.no_generate else "llm"
    answer_basis = "retrieval_only" if args.no_generate else "retrieved_evidence"
    if args.no_generate:
        results = index.search(
            query=args.question,
            piece=args.piece,
            measure_ranges=args.measure_ranges,
            topic=args.topic,
            top_k=args.top_k,
        )
        messages = build_messages(args.piece, args.measures, args.question, results)
        answer = ""
    else:
        def run_generation(prompt_messages: List[Dict[str, str]]) -> str:
            if not os.path.exists(settings["model_path"]):
                raise SystemExit(
                    "Missing model checkpoint: %s. Run scripts/download_model.py first."
                    % settings["model_path"]
                )
            return generate(settings["model_path"], prompt_messages, settings["llm"])

        raw_answer, results, messages, context_limited = generate_with_context_retry(
            index,
            query=args.question,
            piece=args.piece,
            measure_ranges=args.measure_ranges,
            measures=args.measures,
            topic=args.topic,
            top_k=args.top_k,
            generator=run_generation,
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
            messages = build_internal_knowledge_messages(
                args.piece,
                args.measures,
                args.question,
            )
            answer = finalize_internal_knowledge_answer(
                run_generation(messages)
            )
            results = []
            if answer:
                answer_basis = "internal_knowledge"
            else:
                answer = INTERNAL_GENERATION_UNAVAILABLE_MESSAGE
                answer_basis = "generation_unavailable"
                generation_mode = "unavailable"
        elif context_limited and not raw_answer:
            if results:
                answer = build_extractive_answer(results)
                answer_basis = "retrieval_extractive"
                generation_mode = "extractive"
            else:
                answer = "검색 근거가 모델 컨텍스트 한도를 초과하여 안전하게 답변하지 못했습니다."
                answer_basis = "generation_unavailable"
                generation_mode = "unavailable"
        elif results:
            answer = finalize_answer_citations(raw_answer, results)
        else:
            answer = INTERNAL_GENERATION_UNAVAILABLE_MESSAGE
            answer_basis = "generation_unavailable"
            generation_mode = "unavailable"
    evidence_notices = build_evidence_notices(results)

    if args.json:
        print(
            json.dumps(
                {
                    "query": {
                        "piece": args.piece,
                        "measure_range": args.measure_ranges,
                        "question": args.question,
                        "topic": args.topic,
                        "top_k": args.top_k,
                    },
                    "answer": answer,
                    "generation_mode": generation_mode,
                    "answer_basis": answer_basis,
                    "evidence_notices": evidence_notices,
                    "results": [
                        {
                            "rank": idx,
                            "score": result.score,
                            "text_score": result.text_score,
                            "measure_score": result.measure_score,
                            "piece_score": result.piece_score,
                            "scope_match": result.scope_match,
                            "record": result.record,
                        }
                        for idx, result in enumerate(results, start=1)
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    print("Retrieved chunks")
    print("================")
    for idx, result in enumerate(results, start=1):
        print(format_result(result, idx))
        print()
    if args.no_generate:
        print("Generation skipped by --no-generate.")
    else:
        print("Answer")
        print("======")
        print(answer)
    if evidence_notices:
        print()
        print("Evidence notices")
        print("================")
        for notice in evidence_notices:
            print(format_evidence_notice(notice))
            print()


if __name__ == "__main__":
    main()
