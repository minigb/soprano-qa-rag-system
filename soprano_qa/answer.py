#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ask the local measure-aware Soprano QA system."""

from __future__ import annotations

import argparse
import difflib
from itertools import combinations
import json
import os
import re
import unicodedata
from typing import Any, Callable, Dict, List

from soprano_qa.corpus import (
    PIECES,
    SONG_ORDER,
    build_corpus,
    corpus_input_fingerprint,
    file_sha256,
)
from soprano_qa.dense import (
    DenseRetrievalUnavailable,
    SearchIndex,
    build_retrieval_index,
    retrieval_diagnostics,
    validate_retrieval_requirements,
)
from soprano_qa.llm import generate
from soprano_qa.retrieval import (
    format_measure_range,
    format_result,
    load_corpus,
    parse_measure_ranges,
    ranges_overlap,
    scope_evidence_role,
    validate_question_measure_contract,
)
from soprano_qa.settings import load_settings


SYSTEM_PROMPT = """You are a measure-aware soprano performance QA assistant.

Answer in Korean.
Do not reveal chain-of-thought, hidden reasoning, or <think> blocks. Output the final answer only.
Use only the retrieved evidence below. It has two explicitly labeled lineages:
- expert_annotation: human expert performance commentary.
- web_database: externally collected background or performance evidence with source provenance.
For expert_annotation evidence, verbatim_expert_source_answer preserves the authoritative
human response and answer is its curated knowledge-unit rewrite. Keep the generated answer
semantically faithful to the relevant human response. The curated answer defines the claims
that belong to this record. A verbatim source answer may also contain a separate claim that was
split into another knowledge unit: do not use a source-answer claim absent from this record's
curated answer unless another retrieved, range-applicable record contains that claim. Use the
verbatim answer only to preserve detail, examples, and hedging for claims represented by the
curated answer. Treat the curated answer as the safe answer form: do not restore a detail
omitted from it when rewrite_notes say that detail is unverified. If a review warning exposes
an unresolved rewrite conflict, preserve the source's uncertainty instead of silently choosing
a new claim.
The legacy_measure_range_hints nested in a source answer are non-authoritative review hints.
The knowledge unit's confirmed measure_range remains authoritative. When one unit merges
different source claims for separate confirmed ranges, an overlapping source hint may
disambiguate which claim applies to the selected range. Never cite a legacy hint as a confirmed
location. If the wording of the question projects a different-range behavior onto the selected
range, correct that premise using the applicable expert claim instead of repeating it.
Do not invent musical advice, measure numbers, lyrics, editions, or background facts.
If the evidence does not answer the core question at all, output exactly
<NO_GROUNDED_ANSWER> and nothing else. If only a secondary part is unsupported, answer the
supported core and briefly say what is missing.
Preserve source qualifiers: never turn a fact about one edition, recording, performance, or attributed
commentary into a universal fact about every version of the work.
For a measure-range query, evidence_role=selected_range_support is the local authority for the
selected passage. Evidence_role=other_range_context_only may be retained because it is musically
related, but only its existence and range metadata are exposed as an internal range-mismatch signal.
Its musical claim is withheld: do not infer, cite, paraphrase, or transfer that claim into the final
answer.
Evidence_role=general_context may add background but cannot establish a local score claim.
Evidence_role=unconfirmed_scope_context_only has no confirmed location and likewise cannot establish
what happens in the selected range. Web evidence without an edition-qualified measure system cannot
support a claim about a specific bar.
When no measure range is supplied, no score location has been selected. Prefer whole-piece evidence.
Evidence_role=local_example is different: it has a confirmed local measure_range and may be
cited as a concrete example for a broad question. Every sentence that uses a local_example must
cite that evidence label and name at least one range from that same evidence item's canonical
measure_range. Keep the musical claim explicitly limited to those measures (for example,
"30마디와 33마디에서는... [E2]"). A local example cannot by itself
establish that something happens throughout the work, frequently, or in many places. Do not write
"곡 전반", "자주", or "많이" unless separate whole-piece evidence explicitly supports that
claim in that same sentence. Merely retrieving an unrelated whole-piece record does not broaden a
local claim. Never restore an old source-text measure locator in place of the canonical confirmed
range. A whole-piece or general record has no measure authority, even when its verbatim source text
mentions a measure; do not print that locator in a no-range answer.
Evidence labeled unscoped_pending_review has useful expert content but no confirmed location: answer
the question without inventing a measure or generalizing that content to the whole piece.
If expert evidence has a rewrite or measure review warning, preserve its uncertainty and options,
state the unresolved limitation briefly, and never silently resolve the noted conflict.
Preserve the expert's degree of certainty exactly. In particular, wording such as "볼 수 있다",
"가능하다", "경우가 있다", or a recommendation must not become a definite fact or a claimed
composer intention.
Evidence is ordered by retrieval relevance. Answer the exact question from the smallest sufficient
set of evidence; do not add merely related advice just because another record was retrieved.
Keep expert source_ids separate from web_source_ids and claim_ids.
Mention the most relevant measures. Cite only the short evidence labels exactly as shown, such as [E1];
never retype an opaque knowledge-unit/webchunk id. The application maps labels to exact record ids
after generation.
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
Do not emit evidence labels, corpus IDs, or citations such as [E1],
[die-forelle-ku-001], or [webchunk-*].
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
EXPERT_RECORD_ID_PATTERN = r"[a-z0-9]+(?:-[a-z0-9]+)*-ku-[0-9]{3}"
OPAQUE_CITATION_ID_PATTERN = (
    r"(?:"
    + EXPERT_RECORD_ID_PATTERN
    + r"|sqa-[0-9]+|webchunk-[A-Za-z0-9*-]+)"
)
EVIDENCE_FOOTER_LABEL_RE = re.compile(
    r"제공된[\s*_~`]*검색[\s*_~`]*근거[\s*_~`]*[:：]?"
)


def _remove_evidence_footer(answer: str) -> str:
    """Remove user-facing retrieved-evidence footer variants."""

    cleaned_lines: List[str] = []
    for line in answer.splitlines():
        match = EVIDENCE_FOOTER_LABEL_RE.search(line)
        if match:
            # Preserve substantive prose before an inline footer while
            # dropping Markdown/list decoration that belongs to the label.
            prefix = line[: match.start()].rstrip()
            if (
                not any(character.isalnum() for character in prefix)
                or re.fullmatch(r"[\W_]*\d+[.)][\W_]*", prefix)
            ):
                prefix = ""
            else:
                prefix = re.sub(
                    r"\s+[-+*>#]+$",
                    "",
                    prefix,
                ).rstrip()
                for marker in (
                    "```",
                    "***",
                    "___",
                    "~~",
                    "**",
                    "__",
                    "`",
                    "*",
                    "_",
                ):
                    if not prefix.endswith(marker):
                        continue
                    preceding = prefix[: -len(marker)]
                    if preceding.count(marker) % 2 == 0:
                        prefix = preceding.rstrip()
                    break
            if prefix:
                cleaned_lines.append(prefix)
            # A footer is terminal metadata. Dropping the complete remainder
            # also prevents uncommon citation spellings on continuation lines
            # from becoming orphaned user-facing IDs.
            break
        cleaned_lines.append(line)
    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


SOURCE_MEASURE_LOCATOR_RE = re.compile(
    r"(?<!\d)"
    r"\d+\s*(?:[-–—~]\s*\d+)?"
    r"(?:\s*(?:,|와|과|/|\s)\s*"
    r"\d+\s*(?:[-–—~]\s*\d+)?)*"
    r"\s*마디"
    r"(?P<particle>의|에서|부터|까지|은|는|이|가|을|를|에|와|과|로)?"
)
SOURCE_ORDINAL_MEASURE_LOCATOR_RE = re.compile(
    r"(?:첫\s*(?:번째\s*)?|마지막\s*(?:번째\s*)?|끝\s*)마디"
    r"(?P<particle>의|에서|부터|까지|은|는|이|가|을|를|에|와|과|로)?"
)
SOURCE_ENGLISH_MEASURE_LOCATOR_RE = re.compile(
    r"\b(?:"
    r"m\.\s*\d+(?:\s*[-–—~]\s*\d+)?"
    r"|(?:measures?|bars?)\s+\d+(?:\s*[-–—~]\s*\d+)?"
    r")(?!\d)",
    flags=re.IGNORECASE,
)
LOCAL_SCOPE_GENERALIZATION_RE = re.compile(
    r"(?:"
    r"(?:곡|노래|작품)\s*(?:전체|전반)"
    r"|(?:전체적|전반적)으로"
    r"|항상|언제나|곳곳(?:에서|에)?"
    r"|전\s*(?:곡|노래|작품|구간|마디|부분|대목)"
    r"(?:에\s*걸쳐|을\s*통틀어|에서|에)?"
    r"|(?:모든|대부분(?:의)?)\s*"
    r"(?:구간|마디|부분|대목|곡|노래|작품)(?:에서|에|이|가)?"
    r"|(?:여러|많은)\s*(?:마디|곳|부분|대목)(?:에서|에|이|가)?"
    r"|(?:경우|부분|대목|음|고음|저음|도약음|현상)"
    r"(?:이|가|은|는)?\s*많이\s*(?:나오|등장|발생|나타나|있)"
    r"|(?:경우|부분|대목)(?:이|가|은|는)?\s*많"
    r"|자주\s*(?:나오|등장|발생|나타나|보이|있)"
    r"|많이\s*(?:나오|등장|발생|나타나|있)"
    r"|\b(?:throughout|always|frequently|often|everywhere)\b"
    r"|\b(?:across|all\s+through)\s+(?:the\s+)?"
    r"(?:piece|work|song)\b"
    r"|\bmost\s+(?:bars?|measures?|places?|passages?)\b"
    r"|\bmany\s+(?:bars?|measures?|places?|passages?)\b"
    r")",
    flags=re.IGNORECASE,
)
LOCAL_SCOPE_NEGATION_AFTER_RE = re.compile(
    r"(?:적용|해당|일반화|확인|뒷받침|입증)"
    r".{0,24}(?:않|아니|되지|없)"
    r"|뜻.{0,12}(?:아니|않)"
    r"|것.{0,8}(?:아니|않)"
    r"|^\s*지(?:는)?\s*(?:않|아니)"
    r"|(?:does\s+not|doesn't|cannot|can't|is\s+not)"
    r".{0,24}(?:apply|establish|support|prove)",
    flags=re.IGNORECASE,
)
ANSWER_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
ANSWER_CLAIM_SPLIT_RE = re.compile(
    r"(?<=[.!?。！？])\s+(?![\[【])"
)
ANSWER_EVIDENCE_LABEL_RE = re.compile(
    r"[\[【]([^\]】]+)[\]】]",
    flags=re.IGNORECASE,
)
UNSEEN_CAMEL_CASE_RE = re.compile(
    r"[A-Za-z]*[a-z][A-Z][A-Za-z]*"
)
RANGED_PRIMARY_SCOPE_MATCHES = {
    "overlaps_query_range",
    "global_context",
}
RANGED_CONTEXT_ONLY_SCOPE_MATCHES = {
    "other_range_context",
}
RANGED_NON_GROUNDING_SCOPE_MATCHES = {
    *RANGED_CONTEXT_ONLY_SCOPE_MATCHES,
    "unscoped_pending_context",
    "unspecified_context",
}
NO_RANGE_SCOPE_MATCHES = {
    "general_evidence",
    "local_example",
    "unscoped_pending_review",
    "unspecified_scope",
}


def _location_neutral_source_text(value: str) -> str:
    """Hide non-authoritative source locators from a range-scoped prompt."""

    def replace_locator(match: re.Match[str]) -> str:
        particle = match.group("particle")
        if particle:
            particle = {
                "가": "이",
                "는": "은",
                "를": "을",
                "와": "과",
                "로": "으로",
            }.get(particle, particle)
            return "해당 대목" + particle
        return "해당 대목"

    neutral = SOURCE_MEASURE_LOCATOR_RE.sub(replace_locator, value)
    neutral = SOURCE_ORDINAL_MEASURE_LOCATOR_RE.sub(
        replace_locator,
        neutral,
    )
    neutral = SOURCE_ENGLISH_MEASURE_LOCATOR_RE.sub(
        "해당 대목",
        neutral,
    )
    neutral = neutral.replace("해당 대목의의", "해당 대목의")
    return re.sub(
        r"해당 대목(?:의|과)?\s+해당 대목의",
        "두 부분의",
        neutral,
    )


def _neutralize_unconfirmed_review_locators(value: str) -> str:
    """Preserve a review-note contrast without exposing disputed measures."""

    matches: List[tuple[int, int, str, str]] = []
    for pattern in (
        SOURCE_MEASURE_LOCATOR_RE,
        SOURCE_ORDINAL_MEASURE_LOCATOR_RE,
        SOURCE_ENGLISH_MEASURE_LOCATOR_RE,
    ):
        for match in pattern.finditer(value):
            particle = match.groupdict().get("particle") or ""
            raw_locator = match.group(0)
            locator_key = (
                raw_locator[: -len(particle)]
                if particle
                else raw_locator
            )
            matches.append(
                (
                    match.start(),
                    match.end(),
                    locator_key,
                    particle,
                )
            )
    if not matches:
        return value

    labels: Dict[str, str] = {}
    output: List[str] = []
    cursor = 0
    for start, end, locator_key, particle in sorted(matches):
        if start < cursor:
            continue
        label = labels.get(locator_key)
        if label is None:
            index = len(labels)
            suffix = (
                chr(ord("A") + index)
                if index < 26
                else "A%d" % (index + 1)
            )
            label = "미확인 위치 " + suffix
            labels[locator_key] = label
        output.append(value[cursor:start])
        output.append(label + particle)
        cursor = end
    output.append(value[cursor:])
    return "".join(output)


def local_examples_are_only_scope_authority(results: List[Any]) -> bool:
    """Return whether only local examples have confirmed scope authority."""

    primary = primary_grounding_results(results)
    return any(
        result.scope_match == "local_example"
        for result in primary
    ) and not any(
        result.scope_match == "general_evidence"
        for result in primary
    )


def _answer_claims(answer: str) -> List[str]:
    """Split generated prose into citation-bearing claim-sized lines."""

    claims: List[str] = []
    sanitized_answer = _remove_evidence_footer(answer)
    for line in unicodedata.normalize("NFKC", sanitized_answer).splitlines():
        line = ANSWER_LIST_PREFIX_RE.sub("", line.strip())
        if (
            not line
            or line.startswith("검토 주의:")
            or line.startswith("제공된 검색 근거:")
        ):
            continue
        claims.extend(
            part.strip()
            for part in ANSWER_CLAIM_SPLIT_RE.split(line)
            if part.strip()
        )
    return claims


def _claim_cited_result_indices(
    claim: str,
    results: List[Any],
) -> set[int]:
    """Resolve only unambiguous evidence citations in one generated claim."""

    cited: set[int] = set()
    bracket_contents = ANSWER_EVIDENCE_LABEL_RE.findall(claim)
    for content in bracket_contents:
        cited.update(
            int(label) - 1
            for label in re.findall(
                r"(?<![A-Za-z0-9])E(?:vidence)?\s*([0-9]+)(?![0-9])",
                content,
                flags=re.IGNORECASE,
            )
            if 0 < int(label) <= len(results)
        )
        for index, result in enumerate(results):
            record_id = str(result.record.get("id") or "")
            if record_id and re.search(
                r"(?<![A-Za-z0-9_-])"
                + re.escape(record_id)
                + r"(?![A-Za-z0-9_-])",
                content,
                flags=re.IGNORECASE,
            ):
                cited.add(index)

    outside_brackets = re.sub(
        r"[\[【][^\]】]*[\]】]",
        " ",
        claim,
    )
    for pattern in (
        r"(?<![A-Za-z0-9])Evidence\s*"
        r"(?:(?:no\.?|number)\s*)?(?:[:：#._–—-]\s*)?"
        r"([0-9]+)(?![0-9])",
        r"(?:근거|출처)\s*(?:(?:[:：#._–—-]\s*)"
        r"(?:E(?:vidence)?\s*)?|E(?:vidence)?\s*)"
        r"([0-9]+)(?![0-9])",
    ):
        cited.update(
            int(label) - 1
            for label in re.findall(
                pattern,
                outside_brackets,
                flags=re.IGNORECASE,
            )
            if 0 < int(label) <= len(results)
        )
    return cited


def cited_primary_grounding_results(
    answer: str,
    results: List[Any],
) -> List[Any]:
    """Return primary evidence explicitly cited by substantive answer claims.

    Review warnings are evidence-specific.  They must not be projected from an
    unused retrieval candidate onto an answer grounded in a different record.
    Preserve retrieval order so later citation finalization remains stable.
    """

    primary_ids = {
        id(result)
        for result in primary_grounding_results(results)
    }
    cited_indices: set[int] = set()
    for claim in _answer_claims(answer):
        cited_indices.update(
            _claim_cited_result_indices(claim, results)
        )
    return [
        result
        for index, result in enumerate(results)
        if index in cited_indices and id(result) in primary_ids
    ]


def _claim_names_confirmed_range(
    claim: str,
    measure_ranges: List[List[int]],
) -> bool:
    """Return whether a claim names one canonical inclusive range."""

    normalized = unicodedata.normalize("NFKC", claim)
    dash = r"[-–—~〜]"
    for start, end in measure_ranges:
        if start == end:
            patterns = (
                rf"(?<!\d){start}\s*(?:번째\s*)?마디",
                rf"(?:마디|m\.|measures?|bars?)\s*{start}(?!\d)",
            )
        else:
            patterns = (
                rf"(?<!\d){start}\s*{dash}\s*{end}\s*"
                rf"(?:번째\s*)?마디",
                rf"(?<!\d){start}\s*마디\s*부터\s*{end}\s*마디",
                rf"(?:마디|measures?|bars?)\s*{start}\s*{dash}\s*"
                rf"{end}(?!\d)",
            )
        if any(
            re.search(pattern, normalized, flags=re.IGNORECASE)
            for pattern in patterns
        ):
            return True
    return False


def _claim_measure_locators(claim: str) -> List[List[int]]:
    """Extract explicit score locations from generated answer prose."""

    normalized = unicodedata.normalize("NFKC", claim)
    dash = r"[-–—~〜]"
    patterns = (
        re.compile(
            r"(?<!\d)(\d+)\s*마디\s*부터\s*(\d+)\s*"
            r"(?:번째\s*)?마디"
        ),
        re.compile(
            rf"(?<!\d)(\d+)\s*{dash}\s*(\d+)\s*"
            r"(?:번째\s*)?마디"
        ),
        re.compile(
            rf"(?:마디|m\.|measures?|bars?)\s*(\d+)\s*{dash}\s*"
            r"(\d+)(?!\d)",
            flags=re.IGNORECASE,
        ),
        re.compile(r"(?<!\d)(\d+)\s*(?:번째\s*)?마디"),
        re.compile(
            r"(?:마디|m\.|measures?|bars?)\s*(\d+)(?!\d)",
            flags=re.IGNORECASE,
        ),
    )
    occupied: List[tuple[int, int]] = []
    locators: List[List[int]] = []
    for pattern in patterns:
        for match in pattern.finditer(normalized):
            if any(
                match.start() < end and start < match.end()
                for start, end in occupied
            ):
                continue
            start = int(match.group(1))
            end = int(match.group(2) or start) if match.lastindex == 2 else start
            if start < 1 or end < start:
                continue
            occupied.append((match.start(), match.end()))
            locator = [start, end]
            if locator not in locators:
                locators.append(locator)
    return locators


def _claim_has_unauthorized_measure_locator(
    claim: str,
    results: List[Any],
    cited: set[int],
    selected_measure_ranges: List[List[int]],
) -> bool:
    """Reject score locations not authorized by query or local evidence."""

    locators = _claim_measure_locators(claim)
    if not locators:
        return False
    if selected_measure_ranges:
        return any(
            not any(
                selected_start <= start and end <= selected_end
                for selected_start, selected_end in selected_measure_ranges
            )
            for start, end in locators
        )
    allowed_local_ranges = {
        tuple(measure_range)
        for index in cited
        if results[index].scope_match == "local_example"
        for measure_range in (
            results[index].record.get("measure_range") or []
        )
    }
    return any(tuple(locator) not in allowed_local_ranges for locator in locators)


def _claim_has_piece_wide_generalization(claim: str) -> bool:
    """Detect a positive whole-piece or frequency assertion in one claim."""

    for match in LOCAL_SCOPE_GENERALIZATION_RE.finditer(claim):
        before = claim[max(0, match.start() - 12) : match.start()]
        after = claim[match.end() : match.end() + 56]
        if re.search(r"\bnot\s+$", before, flags=re.IGNORECASE):
            continue
        if LOCAL_SCOPE_NEGATION_AFTER_RE.search(after):
            continue
        return True
    return False


def answer_overgeneralizes_local_examples(
    answer: str,
    results: List[Any],
    selected_measure_ranges: List[List[int]] | None = None,
) -> bool:
    """Reject unbound or overgeneralized local-example claims.

    Local examples are safe factual context for a broad question only when
    the generated claim identifies both the evidence item and one of its
    canonical confirmed ranges. A different whole-piece record can support a
    broad claim only when that record is cited in the same claim.
    """

    if not answer.strip():
        return False
    selected_measure_ranges = selected_measure_ranges or []
    primary = primary_grounding_results(results)
    primary_ids = {id(result) for result in primary}
    local_indices = {
        index
        for index, result in enumerate(results)
        if id(result) in primary_ids
        and result.scope_match == "local_example"
    }
    general_indices = {
        index
        for index, result in enumerate(results)
        if id(result) in primary_ids
        and result.scope_match == "general_evidence"
    }

    body_claims = _answer_claims(answer)
    body_citations: set[int] = set()
    is_no_range_answer = bool(results) and all(
        result.scope_match in NO_RANGE_SCOPE_MATCHES
        for result in results
    )
    for claim in body_claims:
        cited = _claim_cited_result_indices(claim, results)
        body_citations.update(cited)
        if (
            (selected_measure_ranges or is_no_range_answer)
            and _claim_has_unauthorized_measure_locator(
                claim,
                results,
                cited,
                selected_measure_ranges,
            )
        ):
            return True
        cited_local = cited & local_indices
        if not cited_local:
            continue
        for index in cited_local:
            if not _claim_names_confirmed_range(
                claim,
                results[index].record.get("measure_range") or [],
            ):
                return True
        if (
            _claim_has_piece_wide_generalization(claim)
            and not cited & general_indices
        ):
            return True

    # With local examples in the prompt, an unlabeled answer makes it
    # impossible to tell which claim came from which range, so require the
    # model to identify at least one actual evidence item in the answer body.
    if local_indices and body_claims and not body_citations:
        return True
    return False


def suspicious_generation_tokens(
    answer: str,
    messages: List[Dict[str, str]],
) -> List[str]:
    """Find code/UI-like CamelCase tokens not present in the evidence."""

    prompt_text = "\n".join(message["content"] for message in messages)
    return sorted(
        {
            token
            for token in UNSEEN_CAMEL_CASE_RE.findall(answer)
            if token not in prompt_text
        }
    )


def build_generation_repair_messages(
    messages: List[Dict[str, str]],
    answer: str,
    suspicious_tokens: List[str],
) -> List[Dict[str, str]]:
    """Ask the grounded model to repair a visibly corrupted draft."""

    return [
        *messages,
        {"role": "assistant", "content": answer},
        {
            "role": "user",
            "content": (
                "The draft contains corrupted code/UI-like token(s) that do "
                "not occur in the evidence: %s. Rewrite the complete final "
                "answer in natural Korean, replacing the corruption with "
                "the intended evidence-grounded musical term. Preserve all "
                "claims, uncertainty, review disclosures, and valid evidence "
                "labels; add no new advice. Output the corrected answer only."
                "\n/no_think"
            )
            % ", ".join(suspicious_tokens),
        },
    ]


def _range_scoped_source_context(
    record: Dict[str, Any],
    measure_ranges: List[List[int]],
) -> tuple[List[Dict[str, Any]], bool]:
    """Select source subclaims whose review hints overlap this query range."""

    contexts = list(record.get("source_answer_context", []))
    if not measure_ranges:
        return contexts, False
    overlapping = [
        context
        for context in contexts
        if (
            context.get("legacy_measure_range_hints")
            and ranges_overlap(
                context["legacy_measure_range_hints"],
                measure_ranges,
            )
        )
    ]
    nonoverlapping_hinted = [
        context
        for context in contexts
        if (
            context.get("legacy_measure_range_hints")
            and not ranges_overlap(
                context["legacy_measure_range_hints"],
                measure_ranges,
            )
        )
    ]
    if not overlapping or not nonoverlapping_hinted:
        return contexts, False
    unscoped = [
        context
        for context in contexts
        if not context.get("legacy_measure_range_hints")
    ]
    return overlapping + unscoped, True


def _neutralize_source_locators(
    record: Dict[str, Any],
    selected_ranges: List[List[int]],
    *,
    range_partitioned: bool,
) -> bool:
    """Return whether canonical range metadata must replace source locators."""

    # With no selected range, source-level locators are never routing
    # authority. This applies to specific, whole-piece, and unspecified KUs:
    # only the KU's reviewed measure metadata may put a number in the answer.
    return range_partitioned or not selected_ranges


def expert_prompt_factuality_material(
    record: Dict[str, Any],
    measure_ranges: List[List[int]] | None = None,
) -> Dict[str, Any]:
    """Return expert prose that is actually exposed to grounded generation.

    The result intentionally mirrors ``build_context``. In particular, a
    multi-range knowledge-unit rewrite is withheld when source-level review
    hints can isolate the selected range; only the overlapping, locator-
    neutral source Q&A is then returned.
    """

    if record.get("evidence_type") != "expert_annotation":
        raise ValueError("factuality material requires expert evidence")
    selected_ranges = measure_ranges or []
    record_ranges = record.get("measure_range") or []
    if (
        selected_ranges
        and record.get("measure_status") == "specific"
        and record_ranges
        and not ranges_overlap(record_ranges, selected_ranges)
    ):
        # The complete record remains in retrieval/service output, but its
        # claim text is deliberately absent from the answer-model prompt.
        return {
            "range_partitioned": False,
            "answer_texts": [],
            "question_contexts": [],
            "source_ids": [],
        }
    source_answers, range_partitioned = _range_scoped_source_context(
        record,
        selected_ranges,
    )
    neutralize_source_locators = _neutralize_source_locators(
        record,
        selected_ranges,
        range_partitioned=range_partitioned,
    )
    answer_texts: List[str] = []
    question_contexts: List[str] = []
    source_ids: List[str] = []
    for source_answer in source_answers:
        linked_unit_ids = source_answer.get(
            "linked_knowledge_unit_ids",
            [],
        )
        if len(linked_unit_ids) > 1:
            continue
        question = source_answer.get("question", "")
        answer = source_answer.get("answer", "")
        if neutralize_source_locators:
            question = _location_neutral_source_text(question)
            answer = _location_neutral_source_text(answer)
        question = re.sub(r"\s+", " ", question).strip()
        answer = re.sub(r"\s+", " ", answer).strip()
        if question and question not in question_contexts:
            question_contexts.append(question)
        if answer and answer not in answer_texts:
            answer_texts.append(answer)
        source_id = source_answer.get("source_id")
        if (
            isinstance(source_id, str)
            and source_id
            and source_id not in source_ids
        ):
            source_ids.append(source_id)
    if not range_partitioned:
        curated_answer = re.sub(
            r"\s+",
            " ",
            record.get("answer", ""),
        ).strip()
        if curated_answer and curated_answer not in answer_texts:
            answer_texts.append(curated_answer)
    return {
        "range_partitioned": range_partitioned,
        "answer_texts": answer_texts,
        "question_contexts": question_contexts,
        "source_ids": source_ids,
    }


def build_context(
    results: List[Any],
    measure_ranges: List[List[int]] | None = None,
) -> str:
    measure_ranges = measure_ranges or []
    chunks = []
    for idx, result in enumerate(results, start=1):
        record = result.record
        evidence_role = scope_evidence_role(result.scope_match)
        lines = [
            "Evidence %d" % idx,
            (
                "citation_label: (none; context-only evidence is not "
                "citable)"
                if result.scope_match
                in RANGED_CONTEXT_ONLY_SCOPE_MATCHES
                else "citation_label: E%d" % idx
            ),
            "id: %s" % record["id"],
            "evidence_type: %s" % record.get("evidence_type", "unknown"),
            "piece: %s" % record["piece"],
            "work: %s" % record["work"],
            "topic: %s" % record["topic"],
            "measure_range: %s" % format_measure_range(record.get("measure_range") or []),
            "measure_scope: %s" % record.get("measure_scope", ""),
            "query_scope_relation: %s" % result.scope_match,
            "evidence_role: %s" % evidence_role,
        ]
        if result.scope_match == "other_range_context":
            lines.append(
                "usage_constraint: This evidence has a confirmed location "
                "outside the selected range. Only its existence and range "
                "metadata are an internal range-mismatch signal. Its "
                "musical claim is withheld; do not infer, cite, paraphrase, "
                "or transfer that claim into the final answer."
            )
            lines.append(
                "content: (withheld from answer generation; inspect the "
                "returned evidence record separately)"
            )
            chunks.append("\n".join(lines))
            continue
        elif result.scope_match in {
            "unscoped_pending_context",
            "unspecified_context",
        }:
            lines.append(
                "usage_constraint: This evidence has no confirmed location "
                "for the selected range. Preserve it as unconfirmed-scope "
                "context only; never use it to claim what happens in the "
                "selected range."
            )
        elif result.scope_match == "global_context":
            lines.append(
                "usage_constraint: This is general background only. Do not "
                "use it to establish a measure-specific score claim."
            )
        elif result.scope_match == "local_example":
            lines.append(
                "usage_constraint: This is a citable confirmed local "
                "example for a broad, no-range question. Use only the "
                "canonical measure_range above when naming its location, "
                "and explicitly bind the musical claim to those measures. "
                "It does not establish that the claim applies throughout "
                "the work, frequently, or in many places."
            )
        if record.get("evidence_type") == "expert_annotation":
            rewrite_notes = record.get("rewrite_notes") or "(none)"
            measure_notes = record.get("measure_notes") or "(none)"
            retrieval_review_warning = (
                record.get("retrieval_review_warning")
                or "(none)"
            )
            if not measure_ranges:
                rewrite_notes = _neutralize_unconfirmed_review_locators(
                    rewrite_notes
                )
                measure_notes = _neutralize_unconfirmed_review_locators(
                    measure_notes
                )
                retrieval_review_warning = (
                    _neutralize_unconfirmed_review_locators(
                        retrieval_review_warning
                    )
                )
            lines.extend(
                [
                    "expert_source_ids: %s" % ", ".join(record["source_ids"]),
                    "annotators: %s" % ", ".join(record.get("annotators", [])),
                    "rewrite_status: %s" % record.get("rewrite_status", ""),
                    "rewrite_notes: %s" % rewrite_notes,
                    "measure_status: %s" % record.get("measure_status", ""),
                    "measure_notes: %s" % measure_notes,
                    "retrieval_review_warning: %s"
                    % retrieval_review_warning,
                ]
            )
            source_answers, range_partitioned = _range_scoped_source_context(
                record,
                measure_ranges,
            )
            neutralize_source_locators = _neutralize_source_locators(
                record,
                measure_ranges,
                range_partitioned=range_partitioned,
            )
            if range_partitioned:
                lines.extend(
                    [
                        "range_partitioned_source_context: true",
                        "range_specific_instruction: The source answers "
                        "below overlap the selected range. Use them instead "
                        "of other-range clauses in the combined knowledge-"
                        "unit rewrite, and correct a conflicting premise in "
                        "the user question. Do not repeat any measure number "
                        "from the source text in the final answer; refer to "
                        "the selected passage or its relative parts.",
                    ]
                )
            for source_answer in source_answers:
                linked_unit_ids = source_answer.get(
                    "linked_knowledge_unit_ids",
                    [],
                )
                if len(linked_unit_ids) > 1:
                    lines.append(
                        "split_source_context_withheld: %s"
                        % json.dumps(
                            {
                                "source_id": source_answer["source_id"],
                                "linked_knowledge_unit_ids": linked_unit_ids,
                                "reason": (
                                    "full source Q&A contains claims split "
                                    "across knowledge units; use this "
                                    "record's curated answer only"
                                ),
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                    )
                    continue
                prompt_source_answer = source_answer
                source_answer_label = "verbatim_expert_source_answer"
                if neutralize_source_locators:
                    prompt_source_answer = {
                        **source_answer,
                        "question": _location_neutral_source_text(
                            source_answer.get("question", "")
                        ),
                        "answer": _location_neutral_source_text(
                            source_answer.get("answer", "")
                        ),
                        "legacy_measure_range_relation": (
                            (
                                "overlaps_selected_range_review_hint"
                                if range_partitioned
                                else (
                                    "withheld_for_confirmed_local_example; "
                                    "use_canonical_measure_range"
                                    if result.scope_match == "local_example"
                                    else (
                                        "withheld_for_no_range_query; "
                                        "source_locator_has_no_measure_"
                                        "authority"
                                    )
                                )
                            )
                        ),
                    }
                    prompt_source_answer.pop(
                        "legacy_measure_range_hints",
                        None,
                    )
                    source_answer_label = (
                        (
                            "range_scoped_expert_source_answer_"
                            "with_source_locators_neutralized"
                        )
                        if range_partitioned
                        else (
                            (
                                "local_example_expert_source_answer_"
                                if result.scope_match == "local_example"
                                else "no_range_expert_source_answer_"
                            )
                            + "with_source_locators_neutralized"
                        )
                    )
                lines.append(
                    "%s: %s"
                    % (
                        source_answer_label,
                        json.dumps(
                            prompt_source_answer,
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    )
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
        lines.append(
            "question: %s"
            % (record.get("question") or "(standalone tip)")
        )
        if (
            record.get("evidence_type") == "expert_annotation"
            and range_partitioned
        ):
            lines.append(
                "answer: (combined multi-range rewrite omitted here; use "
                "the range-specific verbatim expert source answer above)"
            )
        else:
            lines.append("answer: %s" % record["answer"])
        chunks.append("\n".join(lines))
    return "\n\n---\n\n".join(chunks) or "(no relevant evidence retrieved)"


def primary_grounding_results(results: List[Any]) -> List[Any]:
    """Return records allowed to ground an answer for this query scope."""

    ranged = any(
        result.scope_match
        in RANGED_PRIMARY_SCOPE_MATCHES
        | RANGED_NON_GROUNDING_SCOPE_MATCHES
        for result in results
    )
    if not ranged:
        return list(results)
    return [
        result
        for result in results
        if result.scope_match in RANGED_PRIMARY_SCOPE_MATCHES
    ]


def has_primary_grounding(results: List[Any]) -> bool:
    """Return whether results contain answer-authoritative evidence."""

    return bool(primary_grounding_results(results))


def has_selected_range_grounding(results: List[Any]) -> bool:
    """Return whether confirmed evidence overlaps the selected range."""

    return any(
        result.scope_match == "overlaps_query_range"
        for result in results
    )


def select_generation_evidence(results: List[Any]) -> List[Any]:
    """Prefer human source-question matches over merely related context.

    Retrieval aliases are immutable original annotator questions.  A positive
    alias score therefore identifies expert evidence whose source question
    semantically covers the user query. When such evidence exists, the
    strongest range-applicable alias tier is primary. This preserves several
    units split from one source question while preventing a stronger
    other-range alias from displacing selected-range evidence. Up to two
    semantically gated other-range results may follow as explicitly
    non-citable context.
    """

    authoritative_alias_matches = [
        result
        for result in results
        if (
            result.record.get("evidence_type") == "expert_annotation"
            and result.alias_score > 0
        )
    ]
    ranged_results = any(
        result.scope_match
        in RANGED_PRIMARY_SCOPE_MATCHES
        | RANGED_NON_GROUNDING_SCOPE_MATCHES
        for result in results
    )
    if not authoritative_alias_matches:
        if not ranged_results:
            return results
        return [
            *[
                result
                for result in results
                if result.scope_match in RANGED_PRIMARY_SCOPE_MATCHES
            ],
            *[
                result
                for result in results
                if result.scope_match
                in RANGED_CONTEXT_ONLY_SCOPE_MATCHES
            ][:2],
        ]

    if ranged_results:
        safe_alias_matches = [
            result
            for result in authoritative_alias_matches
            if result.scope_match in RANGED_PRIMARY_SCOPE_MATCHES
        ]
        if safe_alias_matches:
            best_scope_priority = max(
                1
                if result.scope_match == "overlaps_query_range"
                else 0
                for result in safe_alias_matches
            )
            safe_alias_matches = [
                result
                for result in safe_alias_matches
                if (
                    1
                    if result.scope_match == "overlaps_query_range"
                    else 0
                )
                == best_scope_priority
            ]
            strongest_safe_alias = max(
                result.alias_score
                for result in safe_alias_matches
            )
            primary = [
                result
                for result in safe_alias_matches
                if result.alias_score == strongest_safe_alias
            ]
        else:
            # A strong other-range alias must not displace semantically
            # relevant selected-range or general evidence merely because
            # the latter lacks an exact original-question alias.
            primary = [
                result
                for result in results
                if result.scope_match in RANGED_PRIMARY_SCOPE_MATCHES
            ]
        secondary_context = [
            result
            for result in results
            if result.scope_match in RANGED_CONTEXT_ONLY_SCOPE_MATCHES
        ][:2]
        selected = []
        selected_ids = set()
        for result in [*primary, *secondary_context]:
            record_id = result.record["id"]
            if record_id not in selected_ids:
                selected.append(result)
                selected_ids.add(record_id)
        return selected

    strongest_alias_score = max(
        result.alias_score
        for result in authoritative_alias_matches
    )
    return [
        result
        for result in authoritative_alias_matches
        if result.alias_score == strongest_alias_score
    ]


def build_review_disclosure_from_records(
    records: List[Dict[str, Any]],
    *,
    neutralize_specific_measure_locators: bool = False,
) -> str:
    """Build a concise Korean disclosure from provisional expert records."""

    rewrite_notes = []
    measure_review_pending = False
    piece_identity_uncertain = False
    for record in records:
        warning = record.get("retrieval_review_warning") or ""
        if not warning:
            continue
        rewrite_note = (record.get("rewrite_notes") or "").strip()
        if (
            rewrite_note
            and neutralize_specific_measure_locators
            and record.get("measure_status") == "specific"
        ):
            rewrite_note = _neutralize_unconfirmed_review_locators(
                rewrite_note
            )
        if rewrite_note and rewrite_note not in rewrite_notes:
            rewrite_notes.append(rewrite_note)
        if record.get("measure_status") == "waiting_for_review":
            measure_review_pending = True
            measure_note = (record.get("measure_notes") or "").lower()
            piece_identity_uncertain = piece_identity_uncertain or any(
                marker in measure_note
                for marker in (
                    "다른 악보",
                    "다른 곡",
                    "la capinera",
                    "una voce",
                    "였나",
                )
            )

    parts = list(rewrite_notes)
    if measure_review_pending:
        parts.append(
            "이 근거의 정확한 적용 위치와 국소·곡 전체 적용 범위는 "
            "아직 확인되지 않았다."
        )
    if piece_identity_uncertain:
        parts.append(
            "검토 메모가 다른 악보나 곡일 가능성도 제기하므로, 현재 "
            "선택한 곡에 해당하는지 확인이 필요하다."
        )
    if not parts:
        return ""
    parts.append(
        "따라서 아래 내용은 확정된 악보 사실이 아니라 검토 중인 "
        "전문가 주석에 따른 설명이다."
    )
    return "검토 주의: " + " ".join(parts)


def build_review_disclosure(results: List[Any]) -> str:
    """Build a concise Korean disclosure for provisional expert evidence."""

    primary = primary_grounding_results(results)
    return build_review_disclosure_from_records(
        [
            result.record
            for result in primary
        ],
        neutralize_specific_measure_locators=any(
            result.scope_match == "local_example"
            for result in primary
        ),
    )


def _has_exact_leading_prefix_with_boundary(
    value: str,
    prefix: str,
) -> bool:
    """Return whether value begins with prefix followed by whitespace or EOF."""

    if not prefix or not value.startswith(prefix):
        return False
    return len(value) == len(prefix) or value[len(prefix)].isspace()


def ensure_review_disclosure(answer: str, results: List[Any]) -> str:
    """Guarantee that unresolved review state is visible in the answer."""

    disclosure = build_review_disclosure(results)
    if not disclosure or _has_exact_leading_prefix_with_boundary(
        answer,
        disclosure,
    ):
        return answer
    return disclosure + "\n\n" + answer.lstrip()


def reconcile_review_disclosure(
    answer: str,
    supplied_results: List[Any],
    cited_results: List[Any],
) -> str:
    """Keep only the deterministic warning disclosure for cited evidence.

    A model can copy the conditional disclosure for a retrieved record that it
    ultimately does not use.  Remove only disclosures that this application
    itself can reproduce exactly; arbitrary model prose is never deleted.
    """

    required = build_review_disclosure(cited_results)
    warned = [
        result
        for result in primary_grounding_results(supplied_results)
        if result.record.get("retrieval_review_warning")
    ]
    known_disclosures = set()
    for subset_size in range(1, len(warned) + 1):
        for subset in combinations(warned, subset_size):
            disclosure = build_review_disclosure(list(subset))
            if disclosure:
                known_disclosures.add(disclosure)
    for disclosure in sorted(known_disclosures, key=len, reverse=True):
        if disclosure == required:
            continue
        if _has_exact_leading_prefix_with_boundary(answer, disclosure):
            answer = answer[len(disclosure) :].lstrip()
            break
    return ensure_review_disclosure(answer, cited_results)


def build_review_constraints(results: List[Any]) -> str:
    """Render unresolved human-review notes as hard generation constraints."""

    warnings = []
    primary_result_ids = {
        result.record["id"]
        for result in primary_grounding_results(results)
    }
    for idx, result in enumerate(results, start=1):
        record = result.record
        if record["id"] not in primary_result_ids:
            continue
        warning = record.get("retrieval_review_warning") or ""
        if not warning:
            continue
        rewrite_notes = record.get("rewrite_notes") or "(none recorded)"
        measure_notes = record.get("measure_notes") or "(none recorded)"
        if result.scope_match == "local_example":
            warning = _neutralize_unconfirmed_review_locators(warning)
            rewrite_notes = _neutralize_unconfirmed_review_locators(
                rewrite_notes
            )
            measure_notes = _neutralize_unconfirmed_review_locators(
                measure_notes
            )
        required_disclosure_if_used = build_review_disclosure([result])
        warnings.append(
            "\n".join(
                [
                    "Evidence E%d (%s)" % (idx, record["id"]),
                    "review_warning: %s" % warning,
                    "rewrite_status: %s"
                    % (record.get("rewrite_status") or "(unknown)"),
                    "rewrite_notes: %s" % rewrite_notes,
                    "measure_status: %s"
                    % (record.get("measure_status") or "(unknown)"),
                    "measure_notes: %s" % measure_notes,
                    "required_disclosure_if_used: %s"
                    % required_disclosure_if_used,
                ]
            )
        )
    if not warnings:
        return ""
    return """MANDATORY REVIEW CONSTRAINTS
The following evidence is provisional. For each evidence item that the answer actually
uses and cites, its notes are hard safety constraints, not background metadata. Do not
mention a warning or disclosure for an evidence item that the answer omits:
{warnings}

If the answer cites flagged evidence, it must begin with the corresponding exact Korean
required_disclosure_if_used. When several flagged items are cited, combine only their
warnings into one leading disclosure. The application enforces the final combined wording.

Attribute provisional content to the annotation instead of stating it as settled score
fact. A rewrite note about accuracy or source basis means that the flagged detail must be
omitted or explicitly described as unverified. A pending measure review means that its
exact location and local-versus-whole-piece scope are unconfirmed; do not generalize it to
the whole piece. If a note questions whether the material belongs to this piece, explicitly
say that applicability to the selected piece has not been confirmed. Never lead with one
side of a documented conflict as though it were settled. This applies to every sentence,
including the explanatory body after the disclosure: do not repeat a disputed timing,
simultaneity, location, notation, or causal relationship without qualification. State both
recorded alternatives, or state only their shared undisputed conclusion.
""".format(
        warnings="\n\n".join(warnings),
    )


def requires_review_compliance_repair(results: List[Any]) -> bool:
    """Return whether a draft needs a second pass over disputed evidence."""

    return any(
        result.record.get("retrieval_review_warning")
        and result.record.get("rewrite_status") == "needs_review"
        for result in primary_grounding_results(results)
    )


def build_review_compliance_repair_messages(
    messages: List[Dict[str, str]],
    answer: str,
    cited_results: List[Any],
) -> List[Dict[str, str]]:
    """Ask the grounded model to audit a draft against review constraints."""

    required_disclosure = build_review_disclosure(cited_results)

    return [
        *messages,
        {"role": "assistant", "content": answer},
        {
            "role": "user",
            "content": (
                "Audit the complete draft against MANDATORY REVIEW "
                "CONSTRAINTS and rewrite it in natural Korean. A warning "
                "prefix alone is not enough: no later sentence may assert "
                "one disputed timing, simultaneity, location, notation, or "
                "causal relationship as settled. Preserve both recorded "
                "alternatives, or keep only their shared undisputed "
                "conclusion. Preserve this exact required disclosure for the "
                "flagged evidence actually cited in the draft:\n"
                + required_disclosure
                + "\nPreserve all "
                "grounded practical advice, uncertainty, and valid evidence "
                "labels. Add no new claim. Output the corrected answer only."
                "\n/no_think"
            ),
        },
    ]


def build_local_scope_repair_messages(
    messages: List[Dict[str, str]],
    answer: str,
) -> List[Dict[str, str]]:
    """Ask the model to bind local-example claims to confirmed measures."""

    return [
        *messages,
        {"role": "assistant", "content": answer},
        {
            "role": "user",
            "content": (
                "The draft has an unbound local-example claim or an "
                "unsupported whole-piece or frequency claim. Rewrite the "
                "complete answer in "
                "natural Korean. Every sentence that uses a local_example "
                "must cite that item's [E#] label and name at least one exact "
                "range from that same item's canonical measure_range in the "
                "same sentence. If a local example is not used, omit its "
                "label. Remove claims such as 곡 전반, 전체적으로, 자주, "
                "많이 나온다, or 항상 unless a general_support item that "
                "supports that same sentence is cited there. The mere "
                "presence of unrelated whole-piece evidence is not enough. "
                "Every score measure locator must be authorized by the "
                "selected query range or exactly match a canonical range of "
                "a local_example cited in that sentence. A general_support "
                "item never authorizes a measure number restored from its "
                "source text. "
                "Preserve valid practical advice, uncertainty, review "
                "disclosures, and evidence labels. Add no new claim. Output "
                "the corrected answer only.\n/no_think"
            ),
        },
    ]


def build_messages(piece: str | None, measures: str, question: str, results: List[Any]) -> List[Dict[str, str]]:
    measure_ranges = parse_measure_ranges(measures) if measures else []
    review_constraints = build_review_constraints(results)
    user_prompt = """User query
piece: {piece}
measure_range: {measures}
question: {question}
/no_think

Retrieved evidence
{context}

{review_constraints}
""".format(
        piece=piece or "(not specified)",
        measures=measures or "(none selected)",
        question=question,
        context=build_context(results, measure_ranges),
        review_constraints=review_constraints,
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
        measures=measures or "(none selected)",
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
    finalized = _remove_evidence_footer(finalized)
    citation_label = (
        r"(?:E(?:vidence)?\s*[0-9]+|" + OPAQUE_CITATION_ID_PATTERN + r")"
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
        r"\b" + OPAQUE_CITATION_ID_PATTERN + r"\b",
        "",
        finalized,
        flags=re.IGNORECASE,
    )
    finalized = re.sub(r"[ \t]+\n", "\n", finalized)
    finalized = re.sub(r"\n{3,}", "\n\n", finalized).strip()
    return finalized


def is_grounded_insufficiency_answer(answer: str) -> bool:
    """Recognize the grounded model's no-answer signal and legacy refusal."""

    if answer.strip() and not _remove_evidence_footer(answer):
        return True
    normalized = re.sub(r"\s+", "", answer)
    if NO_GROUNDED_ANSWER_SENTINEL.lower() in normalized.lower():
        return True
    legacy_refusal = re.sub(r"\s+", "", OLD_INSUFFICIENT_EVIDENCE_MESSAGE)
    return legacy_refusal in normalized


def select_extractive_fallback_evidence(
    results: List[Any],
    *,
    max_items: int = 3,
    exclude_local_examples: bool = False,
) -> List[Any]:
    """Choose a compact, coherent evidence bundle for safe fallback text.

    Retrieval intentionally keeps diverse top-k candidates for recall.  An
    extractive answer has a different contract: concatenating unrelated
    candidates makes a safe fallback look like a direct multi-part answer.
    Keep split units from one immutable source together and cap unrelated
    expert supplementation at one record. Once a non-local expert anchors the
    answer, web and local-example candidates are not appended merely to
    diversify a dense fallback. A local or web candidate may anchor only when
    it has a direct query signal; rejected local generalizations exclude local
    examples entirely.
    """

    if max_items < 1 or not results:
        return []
    primary = primary_grounding_results(results)
    if not primary:
        # Keep the existing explicit other-range fallback path intact.
        return list(results)
    def has_direct_query_signal(result: Any) -> bool:
        return bool(
            result.alias_score > 0
            or result.answer_relation_score > 0
            or result.text_score > 0
            or (
                result.concept_coverage >= 0.5
                and result.content_concept_coverage >= 0.5
            )
        )

    def directness_anchor(candidates: List[Any]) -> Any:
        return max(
            enumerate(candidates),
            key=lambda item: (
                item[1].alias_score > 0,
                item[1].alias_score,
                item[1].answer_relation_score,
                item[1].concept_coverage,
                item[1].content_concept_coverage,
                -item[0],
            ),
        )[1]

    eligible_primary = [
        result
        for result in primary
        if not (
            exclude_local_examples
            and result.scope_match == "local_example"
        )
    ]
    expert = [
        result
        for result in eligible_primary
        if result.record.get("evidence_type") == "expert_annotation"
    ]
    broad_guidance = any(
        result.semantic_match_type == "broad_guidance"
        for result in expert
    )
    general = [
        result
        for result in expert
        if result.scope_match in {"general_evidence", "global_context"}
    ]
    anchor = None
    if broad_guidance and general:
        pool = general
        anchor = max(
            pool,
            key=lambda result: (
                result.score,
                result.text_score,
                result.dense_score,
            ),
        )
    elif expert:
        pool = expert
        anchor = directness_anchor(pool)
        if anchor.scope_match == "local_example":
            if not has_direct_query_signal(anchor):
                non_local_expert = [
                    result
                    for result in expert
                    if result.scope_match != "local_example"
                ]
                if non_local_expert:
                    pool = non_local_expert
                    anchor = directness_anchor(pool)
                else:
                    anchor = None
            # A genuinely direct local-only result remains useful, but it
            # does not authorize unrelated whole-piece supplementation.
            if anchor is not None and anchor.scope_match == "local_example":
                pool = [
                    result
                    for result in expert
                    if result.scope_match == "local_example"
                ]
        else:
            # Expert whole-piece evidence has higher answer authority than a
            # local example or a web research lead. Preserve those candidates
            # in diagnostics, not in the deterministic answer bundle.
            pool = [
                result
                for result in expert
                if result.scope_match != "local_example"
            ]

    if anchor is None:
        pool = [
            result
            for result in eligible_primary
            if has_direct_query_signal(result)
        ]
        if not pool:
            return []
        anchor = directness_anchor(pool)

    anchor_sources = set(anchor.record.get("source_ids") or [])
    selected = [anchor]
    siblings = [
        result
        for result in pool
        if (
            result is not anchor
            and anchor_sources.intersection(
                result.record.get("source_ids") or []
            )
        )
    ]
    if (
        not broad_guidance
        and anchor is primary[0]
        and anchor.scope_match
        in {"unscoped_pending_review", "unspecified_scope"}
    ):
        # A top-ranked unscoped record can be the exact source-question answer
        # whose location alone awaits curation. Do not dilute it with an
        # unrelated general record merely because both are primary evidence.
        return [anchor, *siblings][:max_items]
    independent = [
        result
        for result in pool
        if result is not anchor and result not in siblings
    ]
    if anchor.scope_match == "local_example":
        selected.extend(siblings)
    elif broad_guidance or anchor.alias_score > 0:
        selected.extend(siblings)
        selected.extend(independent[:1])
    else:
        # Retrieval order breaks ambiguity without turning every diverse
        # candidate into a claim. Same-source siblings may still fill the
        # remaining compact bundle after that single contender.
        selected.extend(independent[:1])
        selected.extend(siblings)
    return selected[:max_items]


def build_extractive_answer(
    results: List[Any],
    *,
    max_items: int | None = None,
) -> str:
    if not results:
        return NO_CORPUS_EVIDENCE_MESSAGE

    primary = primary_grounding_results(results)
    if primary:
        answer_primary = (
            primary[:max_items]
            if max_items is not None
            else primary
        )
        body = "\n\n".join(
            "%s%s [%s]"
            % (
                (
                    "일반 참고 근거: "
                    if result.scope_match == "global_context"
                    else (
                        "확인된 국소 예시(마디 %s): "
                        % format_measure_range(
                            result.record.get("measure_range") or []
                        )
                        if result.scope_match == "local_example"
                        else ""
                    )
                ),
                result.record["answer"].strip(),
                result.record["id"],
            )
            for result in answer_primary
        )
        disclosure = build_review_disclosure(answer_primary)
        scope_notice = (
            "범위 안내: 아래 내용은 표시된 마디에서 확인된 국소 "
            "예시다. 이 근거만으로 곡 전체의 빈도나 모든 구간에 "
            "항상 적용되는 규칙을 뜻하지 않는다."
            if local_examples_are_only_scope_authority(answer_primary)
            else ""
        )
        return "\n\n".join(
            part
            for part in (disclosure, scope_notice, body)
            if part
        )

    secondary = [
        result
        for result in results
        if result.scope_match in RANGED_CONTEXT_ONLY_SCOPE_MATCHES
    ]
    if not secondary:
        return NO_CORPUS_EVIDENCE_MESSAGE
    context_lines = [
        "다른 구간의 관련 주석: %s [%s]"
        % (
            result.record["answer"].strip(),
            result.record["id"],
        )
        for result in secondary[:3]
    ]
    return (
        "선택한 마디 범위를 직접 뒷받침하는 근거는 없습니다.\n\n"
        + "\n\n".join(context_lines)
    )


def is_context_overflow_error(exc: ValueError) -> bool:
    message = str(exc).lower()
    return "context window" in message and (
        "exceed" in message or "requested tokens" in message
    )


def generate_with_context_retry(
    index: SearchIndex,
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
        results = select_generation_evidence(results)
        messages = build_messages(piece, measures, query, results)
        if not results:
            if context_limited and last_results:
                return "", last_results, last_messages, context_limited
            return "", results, messages, context_limited
        last_results = results
        last_messages = messages
        if measure_ranges and not has_primary_grounding(results):
            # Preserve related other-range records for inspection, but do not
            # ask the model to turn them into an answer about the selected
            # range.
            return "", results, messages, context_limited
        try:
            answer = generator(messages)
            for _ in range(2):
                suspicious_tokens = suspicious_generation_tokens(
                    answer,
                    messages,
                )
                if not suspicious_tokens:
                    break
                answer = generator(
                    build_generation_repair_messages(
                        messages,
                        answer,
                        suspicious_tokens,
                    )
                )
            supplied_review_results = [
                result
                for result in primary_grounding_results(results)
                if result.record.get("retrieval_review_warning")
            ]
            cited_results = cited_primary_grounding_results(answer, results)
            if (
                answer.strip()
                and not is_grounded_insufficiency_answer(answer)
                and supplied_review_results
                and not cited_results
            ):
                # With provisional evidence in the prompt, an uncited draft
                # cannot prove which warning contract applies.  Let the caller
                # use the deterministic extractive fallback instead of adding
                # every retrieved warning to an ambiguous answer.
                answer = NO_GROUNDED_ANSWER_SENTINEL
            if (
                answer.strip()
                and not is_grounded_insufficiency_answer(answer)
                and requires_review_compliance_repair(cited_results)
            ):
                answer = generator(
                    build_review_compliance_repair_messages(
                        messages,
                        answer,
                        cited_results,
                    )
                )
                for _ in range(2):
                    suspicious_tokens = suspicious_generation_tokens(
                        answer,
                        messages,
                    )
                    if not suspicious_tokens:
                        break
                    answer = generator(
                        build_generation_repair_messages(
                            messages,
                            answer,
                            suspicious_tokens,
                        )
                    )
            if (
                answer.strip()
                and not is_grounded_insufficiency_answer(answer)
                and answer_overgeneralizes_local_examples(
                    answer,
                    results,
                    measure_ranges,
                )
            ):
                answer = generator(
                    build_local_scope_repair_messages(
                        messages,
                        answer,
                    )
                )
                for _ in range(2):
                    suspicious_tokens = suspicious_generation_tokens(
                        answer,
                        messages,
                    )
                    if not suspicious_tokens:
                        break
                    answer = generator(
                        build_generation_repair_messages(
                            messages,
                            answer,
                            suspicious_tokens,
                        )
                    )
                if measure_ranges and answer_overgeneralizes_local_examples(
                    answer,
                    results,
                    measure_ranges,
                ):
                    # Service consumers do not pass the selected range into
                    # their final generic safety check. Fail closed here if
                    # the repair still names a location outside that range.
                    answer = ""
            if (
                answer.strip()
                and not is_grounded_insufficiency_answer(answer)
            ):
                cited_results = cited_primary_grounding_results(
                    answer,
                    results,
                )
                if supplied_review_results and not cited_results:
                    answer = NO_GROUNDED_ANSWER_SENTINEL
                else:
                    answer = reconcile_review_disclosure(
                        answer,
                        results,
                        cited_results,
                    )
            return answer, results, messages, context_limited
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


def answer_references_secondary_evidence(
    answer: str,
    results: List[Any],
) -> bool:
    """Detect an explicit reference to context that cannot ground the answer."""

    answer = _remove_evidence_footer(answer)
    primary_results = primary_grounding_results(results)
    record_id_by_label = [
        (
            result.record["id"]
            if result in primary_results
            else None
        )
        for result in results
    ]
    secondary_ids = {
        result.record["id"]
        for result in results
        if result not in primary_results
    }
    secondary_label_numbers = {
        index
        for index, record_id in enumerate(
            record_id_by_label,
            start=1,
        )
        if record_id is None
    }
    normalized_answer = unicodedata.normalize("NFKC", answer)
    bracket_contents = re.findall(
        r"[\[【]([^\]】]+)[\]】]",
        normalized_answer,
    )
    cited_label_numbers = {
        int(label_number)
        for content in bracket_contents
        for label_number in re.findall(
            r"(?<![A-Za-z0-9])"
            r"(?:E(?:vidence)?|근거|출처)"
            r"\s*(?:(?:no\.?|number|번호)\s*)?"
            r"(?:[:：#._–—-]\s*)?([0-9]+)(?![0-9])",
            content,
            flags=re.IGNORECASE,
        )
    }
    # Outside brackets, accept only unambiguous citation forms. Bare E2 is a
    # pitch in this domain, and phrases such as "근거 2가지" are ordinary
    # Korean. Require the full "Evidence" label, or explicit punctuation /
    # an E token after a Korean citation noun.
    outside_brackets = re.sub(
        r"[\[【][^\]】]*[\]】]",
        " ",
        normalized_answer,
    )
    for pattern in (
        r"(?<![A-Za-z0-9])Evidence\s*"
        r"(?:(?:no\.?|number)\s*)?(?:[:：#._–—-]\s*)?"
        r"([0-9]+)(?![0-9])",
        r"(?:근거|출처)\s*(?:(?:[:：#._–—-]\s*)"
        r"(?:E(?:vidence)?\s*)?|E(?:vidence)?\s*)"
        r"([0-9]+)(?![0-9])",
    ):
        cited_label_numbers.update(
            int(label_number)
            for label_number in re.findall(
                pattern,
                outside_brackets,
                flags=re.IGNORECASE,
            )
        )
    secondary_id_patterns = [
        re.compile(
            r"(?<![A-Za-z0-9_-])"
            + re.escape(secondary_id)
            + r"(?![A-Za-z0-9_-])",
            flags=re.IGNORECASE,
        )
        for secondary_id in secondary_ids
    ]
    return bool(
        secondary_label_numbers
        & cited_label_numbers
    ) or any(
        pattern.search(normalized_answer)
        for pattern in secondary_id_patterns
    )


def finalize_answer_citations(answer: str, results: List[Any]) -> str:
    answer = _remove_evidence_footer(answer)
    all_record_ids = [result.record["id"] for result in results]
    if not all_record_ids:
        return answer
    primary_results = primary_grounding_results(results)
    record_ids = [result.record["id"] for result in primary_results]
    record_id_by_label = [
        (
            result.record["id"]
            if result in primary_results
            else None
        )
        for result in results
    ]
    if answer_references_secondary_evidence(answer, results):
        # A sentence tied to other-range evidence cannot be made safe merely
        # by deleting its citation. Fall back to the primary extractive answer
        # so the secondary claim is removed together with its attribution.
        return build_extractive_answer(
            select_extractive_fallback_evidence(
                primary_results if primary_results else results
            )
        )
    provenance_ids = {
        provenance_id
        for result in results
        for key in ("source_ids", "web_source_ids", "claim_ids")
        for provenance_id in result.record.get(key, [])
    }

    def label_to_citation(label_number: int) -> str:
        result_index = label_number - 1
        if (
            0 <= result_index < len(record_id_by_label)
            and record_id_by_label[result_index]
        ):
            return "[%s]" % record_id_by_label[result_index]
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
        if cited_id in all_record_ids:
            # The model cited a known context-only record. Remove it rather
            # than fuzzily rewriting it into a primary evidence ID.
            return ""
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
        r"\[(" + OPAQUE_CITATION_ID_PATTERN + r")\]",
        repair_opaque_id,
        finalized,
        flags=re.IGNORECASE,
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
            r"^(?:E\s*[0-9]|Evidence\b|"
            r"(?:[a-z0-9]+-)+ku(?:[-_]|[0-9])|sqa(?:[-_]|[0-9])|"
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
    if not record_ids:
        return finalized.strip()
    if not any("[%s]" % record_id in finalized for record_id in record_ids):
        generated_text = finalized.strip()
        if not generated_text:
            return "생성된 답변에서 유효한 내용을 확인하지 못했습니다."
        # Some local-model generations follow the evidence but omit the label.
        # Preserve that useful answer without adding record IDs to user-facing
        # prose; callers receive the retrieved evidence as separate metadata.
        return generated_text
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
                and stats.get("corpus_schema_version") == 6
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
        help=(
            'Optional inclusive measure query like "28-30" or "12, 35"; '
            "omit when no score range is selected"
        ),
    )
    parser.add_argument("--topic", help="Optional topic substring filter")
    parser.add_argument("--question", required=True, help="Singer/user question")
    parser.add_argument("--top-k", type=positive_int, default=6, help="Number of chunks to retrieve")
    parser.add_argument("--no-generate", action="store_true", help="Only retrieve evidence")
    parser.add_argument(
        "--no-internal-knowledge",
        action="store_true",
        help=(
            "Do not answer from pretrained model knowledge when retrieval "
            "finds no corpus evidence"
        ),
    )
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
    try:
        validate_retrieval_requirements(settings)
    except DenseRetrievalUnavailable as exc:
        raise SystemExit(str(exc)) from exc
    ensure_corpus(settings, args.rebuild_corpus)

    records = load_corpus(settings["corpus_path"])
    index = build_retrieval_index(records, settings)
    generation_mode = "retrieval_only" if args.no_generate else "llm"
    answer_basis = "retrieval_only" if args.no_generate else "retrieved_evidence"
    generation_fallback_reason = None
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
        grounded_answer_unusable = (
            is_grounded_insufficiency_answer(raw_answer)
            or (not context_limited and not raw_answer.strip())
        )
        if not results and not args.no_internal_knowledge:
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
                generation_fallback_reason = (
                    "local model returned no usable answer"
                )
        elif results and grounded_answer_unusable:
            answer = build_extractive_answer(
                select_extractive_fallback_evidence(results)
            )
            answer_basis = (
                "retrieval_extractive"
                if has_primary_grounding(results)
                else "retrieved_secondary_context"
            )
            generation_mode = "extractive"
            generation_fallback_reason = (
                "grounded model returned no usable answer"
                if has_primary_grounding(results)
                else "only other-range context retrieved"
            )
        elif context_limited and not raw_answer:
            if results:
                answer = build_extractive_answer(
                    select_extractive_fallback_evidence(results)
                )
                answer_basis = "retrieval_extractive"
                generation_mode = "extractive"
                generation_fallback_reason = "model context limit exceeded"
            else:
                answer = "검색 근거가 모델 컨텍스트 한도를 초과하여 안전하게 답변하지 못했습니다."
                answer_basis = "generation_unavailable"
                generation_mode = "unavailable"
                generation_fallback_reason = "model context limit exceeded"
        elif results:
            secondary_citation_rejected = (
                answer_references_secondary_evidence(
                    raw_answer,
                    results,
                )
            )
            local_scope_rejected = (
                answer_overgeneralizes_local_examples(
                    raw_answer,
                    results,
                    args.measure_ranges,
                )
            )
            answer = (
                build_extractive_answer(
                    select_extractive_fallback_evidence(
                        results,
                        exclude_local_examples=True,
                    )
                )
                if local_scope_rejected
                else finalize_answer_citations(raw_answer, results)
            )
            if secondary_citation_rejected:
                generation_mode = "extractive"
                answer_basis = "retrieval_extractive"
                generation_fallback_reason = (
                    "secondary evidence citation rejected"
                )
            elif local_scope_rejected:
                generation_mode = "extractive"
                answer_basis = "retrieval_extractive"
                generation_fallback_reason = (
                    "unsupported local-example generalization rejected"
                )
        else:
            answer = NO_CORPUS_EVIDENCE_MESSAGE
            answer_basis = "no_corpus_evidence"
            generation_mode = "unavailable"
            generation_fallback_reason = (
                "internal knowledge fallback disabled"
            )
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
                    "generation_fallback_reason": generation_fallback_reason,
                    "retrieval": retrieval_diagnostics(index),
                    "has_primary_grounding": has_primary_grounding(results),
                    "has_selected_range_grounding": (
                        has_selected_range_grounding(results)
                        if args.measure_ranges
                        else None
                    ),
                    "evidence_notices": evidence_notices,
                    "results": [
                        {
                            "rank": idx,
                            "score": result.score,
                            "text_score": result.text_score,
                            "dense_score": result.dense_score,
                            "dense_content_score": (
                                result.dense_content_score
                            ),
                            "fusion_score": result.fusion_score,
                            "retrieval_mode": result.retrieval_mode,
                            "measure_score": result.measure_score,
                            "piece_score": result.piece_score,
                            "concept_coverage": result.concept_coverage,
                            "content_concept_coverage": (
                                result.content_concept_coverage
                            ),
                            "answer_relation_score": (
                                result.answer_relation_score
                            ),
                            "semantic_match_type": (
                                result.semantic_match_type
                            ),
                            "scope_match": result.scope_match,
                            "generation_role": scope_evidence_role(
                                result.scope_match
                            ),
                            "selected_range_claim_authority": (
                                result.scope_match
                                == "overlaps_query_range"
                                if args.measure_ranges
                                else None
                            ),
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
