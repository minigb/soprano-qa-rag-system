#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ask the local measure-aware Soprano QA system."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
import re
import unicodedata
from typing import Any, Dict, List

from soprano_qa.corpus import (
    CORPUS_SCHEMA_VERSION,
    PIECES,
    SONG_ORDER,
    build_corpus,
    corpus_input_fingerprint,
    file_sha256,
)
from soprano_qa.dense import (
    DenseRetrievalUnavailable,
    SearchIndex,
    SOURCE_QUESTION_LED_MATCH_TYPE,
    SOURCE_QUESTION_LED_MIN_CONTENT_SCORE,
    SOURCE_QUESTION_LED_MIN_OTHER_SOURCE_MARGIN,
    SOURCE_QUESTION_LED_MIN_RELEVANCE_SCORE,
    build_retrieval_index,
    retrieval_abstained_for_ambiguity,
    retrieval_abstained_for_missing_causal_authority,
    retrieval_diagnostics,
    validate_retrieval_requirements,
)
from soprano_qa.llm import (
    GenerationBackendUnavailable,
    generate,
    generate_json,
    validate_generation_requirements,
)
from soprano_qa.retrieval import (
    format_measure_range,
    format_result,
    expert_record_is_runtime_final,
    load_corpus,
    parse_measure_ranges,
    ranges_overlap,
    scope_evidence_role,
    semantic_concepts,
    validate_question_measure_contract,
)
from soprano_qa.settings import load_settings


SYSTEM_PROMPT = """You are a measure-aware soprano performance QA assistant.

Answer in Korean using formal-polite 합니다체 consistently. End complete explanatory
sentences with forms such as -습니다, -ㅂ니다, or -입니다; use formal-polite question or
request endings when needed. Never copy the retrieved evidence's plain -다/-한다 register
into the answer, and never mix plain and polite registers.
Do not reveal chain-of-thought, hidden reasoning, or <think> blocks. Output the final answer only.
Use only the retrieved evidence below. It has two explicitly labeled lineages:
- expert_annotation: human expert performance commentary.
- web_database: externally collected background or performance evidence with source provenance.
For expert_annotation evidence, answer is the reviewed, curated knowledge-unit answer and is
the authoritative claim text. Raw source annotations, legacy range hints, superseded wording,
conflicts, and editorial notes are deliberately withheld from generation. The knowledge unit's
confirmed measure_range may route the record, but it never substitutes raw source prose for the
curated answer. When a curated answer covers several confirmed ranges, keep every claim within
that reviewed answer and limit the response to what it supports about the selected passage. If
the wording of the question projects unsupported behavior onto the selected range, correct the
premise only when the curated answer supports the correction; otherwise return no grounded answer.
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
When no measure range is supplied, no score location has been selected. Rank evidence by how
directly it answers the question, not by whole-piece scope alone. Prefer whole-piece evidence only
when it is at least as semantically relevant as a local example.
Evidence_role=local_example is passage-specific evidence that may directly answer the situation
described by a broad, no-range question. Its canonical location is hidden routing metadata, not a
range selected by the user. Use its musical advice naturally, cite its short [E#] label while
drafting for internal validation, and do not print any canonical measure number, range, scope label,
annotation label, retrieval explanation, or other internal metadata. Refer to the situation itself
when localization is needed (for example, "긴 간주에서는... [E2]"). A local example cannot by
itself establish that something happens throughout the work, frequently, or in many places. Do not
write "곡 전반", "자주", or "많이" unless separate whole-piece evidence explicitly supports
that claim in that same sentence. Merely retrieving an unrelated whole-piece record does not broaden
a local claim. A whole-piece or general record has no measure authority, even when its verbatim
source text mentions a measure; do not print that locator in a no-range answer.
Preserve the expert's degree of certainty exactly. In particular, wording such as "볼 수 있다",
"가능하다", "경우가 있다", or a recommendation must not become a definite fact or a claimed
composer intention.
Evidence is ordered by retrieval relevance. Answer the exact question from the smallest sufficient
set of evidence; do not add merely related advice just because another record was retrieved.
Keep expert source_ids separate from web_source_ids and claim_ids.
Mention a measure only when the user selected a measure range and doing so helps answer the question.
Use only the short evidence labels exactly as shown, such as [E1], while drafting claims for internal
validation; never retype an opaque knowledge-unit/webchunk id.
Do not add a citation list or evidence footer. The application removes the short labels before display
and keeps exact record IDs, source attribution, and license conditions in separate structured metadata.
Never expose prompt field names or pipeline vocabulary such as citation_label, evidence_role,
measure_range, measure_scope, query_scope_relation, usage_constraint, local_example, knowledge unit,
expert annotation, retrieval, corpus, or retrieved evidence in the answer.
Keep the answer concise and practical for a singer.
"""
NO_GROUNDED_ANSWER_SENTINEL = "<NO_GROUNDED_ANSWER>"
NO_CORPUS_EVIDENCE_MESSAGE = (
    "현재 확인된 정보만으로는 이 질문에 정확히 답변하기 어렵습니다."
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


class GeneratedAnswerRejected(RuntimeError):
    """Raised when the sole generated draft fails grounding/safety checks."""


PIPELINE_TOKEN_SEPARATOR_PATTERN = r"[\s*_~`–—-]*"
PIPELINE_FIELD_NAMES = (
    "citation_label",
    "evidence_role",
    "evidence_type",
    "measure_range",
    "measure_scope",
    "query_scope_relation",
    "scope_match",
    "usage_constraint",
    "local_example",
    "source_ids",
    "expert_source_ids",
    "web_source_ids",
    "claim_ids",
    "annotators",
    "rewrite_status",
    "measure_status",
    "retrieval_review_warning",
    "record_id",
    "knowledge_unit_id",
)


def _separator_flexible_pipeline_field(name: str) -> str:
    """Match one internal field despite copied Markdown/separator changes."""

    return PIPELINE_TOKEN_SEPARATOR_PATTERN.join(
        re.escape(part) for part in name.split("_")
    )


PIPELINE_FIELD_TOKEN_PATTERN = (
    r"(?:"
    + "|".join(
        _separator_flexible_pipeline_field(name)
        for name in PIPELINE_FIELD_NAMES
    )
    + r")"
)
PIPELINE_SCOPE_NOTICE_RE = re.compile(
    r"(?im)^[ \t]*(?:[#>*+~-]+[ \t]*)*(?:[*_`~]+[ \t]*)?"
    r"범위[ \t]*안내[ \t]*[:：][^\r\n]*(?:\r?\n)?"
)
PIPELINE_PRESENTATION_PREFIX_RE = re.compile(
    r"(?im)^[ \t]*(?:[#>*+~-]+[ \t]*)*(?:[*_`~]+[ \t]*)?"
    r"(?:"
    r"확인된[ \t]*국소[ \t]*예시[ \t]*\([^\r\n)]*\)"
    r"|일반[ \t]*참고[ \t]*근거"
    r"|다른[ \t]*구간의[ \t]*관련[ \t]*주석"
    r")"
    r"[ \t]*(?:[*_`~]+[ \t]*)?[:：][ \t]*(?:[*_`~]+[ \t]*)?"
)
PIPELINE_METADATA_LINE_RE = re.compile(
    r"(?im)^[ \t]*(?:[#>*+~-]+[ \t]*)*(?:[*_`~]+[ \t]*)?"
    + PIPELINE_FIELD_TOKEN_PATTERN
    + r"[ \t]*(?:[*_`~]+[ \t]*)?[:：][^\r\n]*(?:\r?\n)?",
    flags=re.IGNORECASE | re.MULTILINE,
)
PIPELINE_INTERNAL_LABEL_ANYWHERE_RE = re.compile(
    r"(?:"
    r"범위[\s*_~`]*안내"
    r"|확인된[\s*_~`]*국소[\s*_~`]*예시(?:\s*\([^\r\n)]*\))?"
    r"|일반[\s*_~`]*참고[\s*_~`]*근거"
    r"|다른[\s*_~`]*구간의[\s*_~`]*관련[\s*_~`]*주석"
    r"|"
    + PIPELINE_FIELD_TOKEN_PATTERN
    + r")[\s*_~`]*[:：]",
    flags=re.IGNORECASE,
)
PIPELINE_ANSWER_PREFIX_ANYWHERE_RE = re.compile(
    r"(?:확인된[\s*_~`]*국소[\s*_~`]*예시"
    r"(?:\s*\([^\r\n)]*\))?|일반[\s*_~`]*참고[\s*_~`]*근거|"
    r"다른[\s*_~`]*구간의[\s*_~`]*관련[\s*_~`]*주석)"
    r"[\s*_~`]*[:：]",
    flags=re.IGNORECASE,
)
PIPELINE_INTERNAL_TOKEN_RE = re.compile(
    r"(?:범위[\s*_~`]*안내|확인된[\s*_~`]*국소[\s*_~`]*예시|"
    r"일반[\s*_~`]*참고[\s*_~`]*근거|"
    r"다른[\s*_~`]*구간의[\s*_~`]*관련[\s*_~`]*주석|"
    r"(?<![A-Za-z0-9_])"
    + PIPELINE_FIELD_TOKEN_PATTERN
    + r"(?![A-Za-z0-9_])|"
    + r"(?<![A-Za-z0-9_])knowledge[\s*_~`-]*units?(?![A-Za-z0-9_])|"
    r"(?<![A-Za-z0-9_])expert[\s*_~`-]*annotations?(?![A-Za-z0-9_])|"
    r"(?<![A-Za-z0-9_])retrieved[\s*_~`-]*evidence(?![A-Za-z0-9_])|"
    r"(?<![A-Za-z0-9_])retrieval(?![A-Za-z0-9_])|"
    r"(?<![A-Za-z0-9_])corpus(?![A-Za-z0-9_])|"
    r"지식[\s*_~`-]*단위|전문가[\s*_~`-]*주석|"
    r"검색[\s*_~`-]*(?:파이프라인|결과|근거)|"
    r"(?:말뭉치|코퍼스))",
    flags=re.IGNORECASE,
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


def _remove_pipeline_presentation_scaffolding(answer: str) -> str:
    """Remove known internal renderer/prompt labels from displayed prose.

    Scope, provenance, and routing data remain available in the structured
    response.  This is a final defense for old cached answers and for a model
    that copies a prompt field despite the generation contract.
    """

    safe_lines: List[str] = []
    for line in answer.splitlines():
        answer_prefix = PIPELINE_ANSWER_PREFIX_ANYWHERE_RE.search(line)
        if answer_prefix is not None:
            prefix = line[: answer_prefix.start()].rstrip(
                " \t*_~`#>+-"
            )
            suffix = line[answer_prefix.end() :].strip(
                " \t*_~`#>+-"
            )
            preserved = " ".join(
                part for part in (prefix, suffix) if part
            )
            if any(character.isalnum() for character in preserved):
                safe_lines.append(preserved)
            continue
        match = PIPELINE_INTERNAL_LABEL_ANYWHERE_RE.search(line)
        if match is None:
            safe_lines.append(line)
            continue
        # Internal fields are never user prose. Preserve only substantive
        # text before the first copied field and discard the rest of that
        # line, whose value has no reliable delimiter.
        prefix = line[: match.start()].rstrip(" \t*_~`#>+-")
        if any(character.isalnum() for character in prefix):
            safe_lines.append(prefix)
    cleaned = "\n".join(safe_lines)
    cleaned = PIPELINE_SCOPE_NOTICE_RE.sub("", cleaned)
    cleaned = PIPELINE_PRESENTATION_PREFIX_RE.sub("", cleaned)
    cleaned = PIPELINE_METADATA_LINE_RE.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


SHORT_EVIDENCE_LABEL_RE = re.compile(
    r"(?:"
    r"E(?:vidence)?\s*"
    r"(?:(?:no\.?|number|번호)\s*)?"
    r"(?:[:：#._–—-]\s*)?[0-9]+"
    r"|(?:근거|출처)\s*"
    r"(?:(?:no\.?|number|번호)\s*)?"
    r"(?:[:：#._–—-]\s*)?"
    r"(?:E(?:vidence)?\s*)?[0-9]+"
    r")",
    flags=re.IGNORECASE,
)
OPAQUE_ID_SEPARATOR_PATTERN = r"[\s*_~`–—-]+"
MALFORMED_OPAQUE_CITATION_RE = re.compile(
    r"(?:"
    r"(?:[a-z0-9]+" + OPAQUE_ID_SEPARATOR_PATTERN + r")+ku"
    r"(?:" + OPAQUE_ID_SEPARATOR_PATTERN + r"[A-Za-z0-9*]+)?"
    r"|sqa(?:" + OPAQUE_ID_SEPARATOR_PATTERN + r"[A-Za-z0-9*]+)+"
    r"|web[\s*_~`–—-]*(?:chunk|src|source|claim)"
    r"(?:" + OPAQUE_ID_SEPARATOR_PATTERN + r"[A-Za-z0-9*]+)+"
    r"|source[\s*_~`–—-]*id\b"
    r")",
    flags=re.IGNORECASE,
)
KOREAN_CITATION_SUFFIX_PATTERN = (
    r"(?:"
    r"에[ \t]*(?:따르면|의하면)"
    r"|에서(?:는)?(?:[ \t]+보(?:면|듯이))?"
    r"|은|는|이|가|을|를|의"
    r")"
)


def _known_evidence_ids(results: List[Any] | None) -> set[str]:
    """Collect record and provenance IDs that must never enter answer prose."""

    if not results:
        return set()
    identifiers: set[str] = set()
    for result in results:
        record = result.record
        record_id = str(record.get("id") or "").strip()
        if record_id:
            identifiers.add(record_id)
        for key in ("source_ids", "web_source_ids", "claim_ids"):
            identifiers.update(
                str(value).strip()
                for value in record.get(key, [])
                if str(value).strip()
            )
    return identifiers


def strip_user_visible_evidence_references(
    answer: str,
    results: List[Any] | None = None,
) -> str:
    """Remove internal evidence labels and corpus IDs from displayed prose.

    Citation labels remain available to the grounding and scope checks before
    this presentation-only function runs. Bare pitch names such as ``E2`` are
    intentionally preserved; only bracketed or explicitly citation-shaped
    short labels are removed.
    """

    cleaned = _remove_pipeline_presentation_scaffolding(
        _remove_evidence_footer(answer)
    )
    known_ids = _known_evidence_ids(results)
    def remove_citation_bracket(match: re.Match[str]) -> str:
        content = match.group(1).strip()
        normalized = re.sub(r"[\s*_~`]", "", content).casefold()
        if any(
            identifier.casefold() in content.casefold()
            or re.sub(r"[\s*_~`]", "", identifier).casefold()
            in normalized
            for identifier in known_ids
        ):
            return ""
        if MALFORMED_OPAQUE_CITATION_RE.search(content):
            return ""
        citation_content = re.sub(
            r"^(?:출처|근거|source|citation)\s*[:：-]?\s*",
            "",
            content,
            flags=re.IGNORECASE,
        ).strip()
        citation_content = citation_content.rstrip(".,;:，、 ")
        if re.fullmatch(r"[0-9]+(?:\s*[,;，、]\s*[0-9]+)*", citation_content):
            return ""
        if re.fullmatch(
            SHORT_EVIDENCE_LABEL_RE.pattern
            + r"(?:\s*[,;，、]\s*"
            + SHORT_EVIDENCE_LABEL_RE.pattern
            + r")*",
            citation_content,
            flags=re.IGNORECASE,
        ):
            return ""
        return match.group(0)

    def remove_citation_bracket_and_suffix(match: re.Match[str]) -> str:
        """Drop a citation together with its now-orphaned Korean particle."""

        if remove_citation_bracket(match) == "":
            return ""
        return match.group(0)

    cleaned = re.sub(
        r"[\[【]([^\]】]+)[\]】][ \t]*"
        + KOREAN_CITATION_SUFFIX_PATTERN
        + r"[ \t]*[,，:]?[ \t]*",
        remove_citation_bracket_and_suffix,
        cleaned,
    )
    cleaned = re.sub(
        r"[\[【]([^\]】]+)[\]】]",
        remove_citation_bracket,
        cleaned,
    )
    cleaned = re.sub(
        r"[\(（]([^\)）]+)[\)）][ \t]*"
        + KOREAN_CITATION_SUFFIX_PATTERN
        + r"[ \t]*[,，:]?[ \t]*",
        remove_citation_bracket_and_suffix,
        cleaned,
    )
    cleaned = re.sub(
        r"[\(（]([^\)）]+)[\)）]",
        remove_citation_bracket,
        cleaned,
    )

    for identifier in sorted(known_ids, key=len, reverse=True):
        cleaned = re.sub(
            r"(?<![A-Za-z0-9_-])"
            + re.escape(identifier)
            + r"(?:[ \t]*"
            + KOREAN_CITATION_SUFFIX_PATTERN
            + r"[ \t]*[,，:]?[ \t]*)?"
            + r"(?![A-Za-z0-9_-])",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
    cleaned = re.sub(
        r"(?<![A-Za-z0-9_-])"
        + OPAQUE_CITATION_ID_PATTERN
        + r"(?:[ \t]*"
        + KOREAN_CITATION_SUFFIX_PATTERN
        + r"[ \t]*[,，:]?[ \t]*)?"
        + r"(?![A-Za-z0-9_-])",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"(?<![A-Za-z0-9])Evidence\s*"
        r"(?:(?:no\.?|number)\s*)?(?:[:：#._–—-]\s*)?"
        r"[0-9]+(?:[ \t]*"
        + KOREAN_CITATION_SUFFIX_PATTERN
        + r"[ \t]*[,，:]?[ \t]*)?(?![0-9])",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"(?:근거|출처)\s*[:：#._–—-]\s*"
        r"(?:E(?:vidence)?\s*)?[0-9]+(?:[ \t]*"
        + KOREAN_CITATION_SUFFIX_PATTERN
        + r"[ \t]*[,，:]?[ \t]*)?(?![0-9])",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    # ``finalize_answer_citations`` may have run before this presentation
    # pass. Remove only unmistakable sentence-initial citation remnants so a
    # second pass also repairs older or partially finalized answers.
    orphaned_attribution = (
        r"(?:에[ \t]*(?:따르면|의하면)|에서[ \t]+보(?:면|듯이)|에서는?)"
    )
    cleaned = re.sub(
        r"(^|[.!?。！？])[ \t]+"
        + orphaned_attribution
        + r"[ \t]*[,，:]?[ \t]*",
        r"\1 ",
        cleaned,
        flags=re.MULTILINE,
    )
    cleaned = re.sub(
        r"(?m)^[ \t]*" + orphaned_attribution + r"[ \t]*[,，:]?[ \t]*",
        "",
        cleaned,
    )
    cleaned = re.sub(r"[ \t]+([,.;:!?，。])", r"\1", cleaned)
    cleaned = re.sub(
        r"(?m)^[ \t]*(?:(?:[-+*>•]\s*)|[.,;:]+)[ \t]*$\n?",
        "",
        cleaned,
    )
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


HANGUL_SYLLABLE_BASE = 0xAC00
HANGUL_SYLLABLE_END = 0xD7A3
HANGUL_JONGSEONG_COUNT = 28
HANGUL_JONGSEONG_NIEUN = 4
HANGUL_JONGSEONG_RIEUL = 8
HANGUL_JONGSEONG_BIEUP = 17
PLAIN_DECLARATIVE_ENDING_RE = re.compile(
    r"(?P<word>[가-힣]+다)"
    r"(?=(?:"
    r"[.!?;:。！？；：][\"'”’」』】)]*"
    r"|[\"'”’」』】)]+[.!?;:。！？；：]"
    r"|[\"'”’」』】)]*[ \t]*(?:\n|$)"
    r"))"
)
KOREAN_SENTENCE_FINAL_WORD_RE = re.compile(
    r"(?P<word>[가-힣]+)"
    r"(?=(?:"
    r"[.!?;:。！？；：][\"'”’」』】)]*"
    r"|[\"'”’」』】)]+[.!?;:。！？；：]"
    r"|[\"'”’」』】)]*[ \t]*(?:\n|$)"
    r"))"
)
FORMAL_POLITE_NONDECLARATIVE_SUFFIXES = (
    "습니까",
    "입니까",
    "합니까",
    "십시오",
    "세요",
    "예요",
    "이에요",
    "해요",
    "돼요",
)
FORMAL_POLITE_WORD_OVERRIDES = {
    "근거다": "근거입니다",
    "자료다": "자료입니다",
}


def _has_formal_polite_declarative_ending(word: str) -> bool:
    """Return whether ``word`` ends in the grammatical -ㅂ니다 form.

    A raw ``endswith("니다")`` check is insufficient because plain lexical
    forms such as ``아니다``, ``다니다``, and ``지니다`` have the same final
    two syllables.  In a formal-polite declarative, the syllable immediately
    before ``니다`` carries final bieup (for example ``합``, ``습``, or
    ``입``).
    """

    if not word.endswith("니다") or len(word) < 3:
        return False
    preceding = word[-3]
    codepoint = ord(preceding)
    if not HANGUL_SYLLABLE_BASE <= codepoint <= HANGUL_SYLLABLE_END:
        return False
    jongseong = (codepoint - HANGUL_SYLLABLE_BASE) % HANGUL_JONGSEONG_COUNT
    return jongseong == HANGUL_JONGSEONG_BIEUP


def _has_formal_polite_propositive_ending(word: str) -> bool:
    """Return whether ``word`` ends in the formal-polite -ㅂ시다 form."""

    if not word.endswith("시다") or len(word) < 3:
        return False
    preceding = word[-3]
    codepoint = ord(preceding)
    if not HANGUL_SYLLABLE_BASE <= codepoint <= HANGUL_SYLLABLE_END:
        return False
    jongseong = (codepoint - HANGUL_SYLLABLE_BASE) % HANGUL_JONGSEONG_COUNT
    return jongseong == HANGUL_JONGSEONG_BIEUP


def _formalize_korean_declarative_word(word: str) -> str:
    """Convert one sentence-final Korean plain declarative to 합니다체."""

    if (
        _has_formal_polite_declarative_ending(word)
        or _has_formal_polite_propositive_ending(word)
        or not word.endswith("다")
    ):
        return word
    if word in FORMAL_POLITE_WORD_OVERRIDES:
        return FORMAL_POLITE_WORD_OVERRIDES[word]
    stem = word[:-1]
    if stem.endswith("는"):
        stem = stem[:-1]
    if not stem:
        return word
    final = stem[-1]
    codepoint = ord(final)
    if not HANGUL_SYLLABLE_BASE <= codepoint <= HANGUL_SYLLABLE_END:
        return word
    jongseong = (codepoint - HANGUL_SYLLABLE_BASE) % HANGUL_JONGSEONG_COUNT
    syllable_without_jongseong = codepoint - jongseong
    if jongseong in {
        0,
        HANGUL_JONGSEONG_NIEUN,
        HANGUL_JONGSEONG_RIEUL,
    }:
        formal_stem = (
            stem[:-1]
            + chr(syllable_without_jongseong + HANGUL_JONGSEONG_BIEUP)
        )
        return formal_stem + "니다"
    return stem + "습니다"


def formalize_korean_answer(answer: str) -> str:
    """Deterministically normalize sentence-final plain Korean to 합니다체."""

    formalized = PLAIN_DECLARATIVE_ENDING_RE.sub(
        lambda match: _formalize_korean_declarative_word(match.group("word")),
        answer,
    )
    return re.sub(
        r"([.!?。！？][\"'”’」』】)]*)(?=[가-힣])",
        r"\1 ",
        formalized,
    )


def answer_uses_formal_polite_korean(answer: str) -> bool:
    """Return whether every Korean sentence uses a polite final ending."""

    if not re.search(r"[가-힣]", answer):
        return False
    for match in KOREAN_SENTENCE_FINAL_WORD_RE.finditer(answer):
        word = match.group("word")
        if word.endswith("다"):
            if not (
                _has_formal_polite_declarative_ending(word)
                or _has_formal_polite_propositive_ending(word)
            ):
                return False
        elif not word.endswith(FORMAL_POLITE_NONDECLARATIVE_SUFFIXES):
            return False
    return True


def finalize_user_visible_answer(
    answer: str,
    results: List[Any] | None = None,
) -> str:
    """Apply the invariant for prose returned to a user."""

    cleaned = strip_user_visible_evidence_references(answer, results)
    # Corpus excerpts and generated prose can contain duplicated horizontal
    # whitespace.  Keep paragraph boundaries intact, but never expose those
    # source-formatting artifacts in the final answer.
    cleaned = re.sub(r"[^\S\r\n]+", " ", cleaned)
    finalized = formalize_korean_answer(cleaned).strip()
    if (
        not finalized
        or not answer_uses_formal_polite_korean(finalized)
        or _has_forbidden_user_visible_artifact(finalized, results)
    ):
        return NO_CORPUS_EVIDENCE_MESSAGE
    return finalized


def _has_forbidden_user_visible_artifact(
    answer: str,
    results: List[Any] | None,
) -> bool:
    """Enforce a final fail-closed boundary for internal presentation data."""

    normalized = unicodedata.normalize("NFKC", answer)
    if (
        EVIDENCE_FOOTER_LABEL_RE.search(normalized)
        or PIPELINE_INTERNAL_TOKEN_RE.search(normalized)
        or MALFORMED_OPAQUE_CITATION_RE.search(normalized)
        or re.search(
            r"(?<![A-Za-z0-9_-])"
            + OPAQUE_CITATION_ID_PATTERN
            + r"(?![A-Za-z0-9_-])",
            normalized,
            flags=re.IGNORECASE,
        )
    ):
        return True
    known_ids = _known_evidence_ids(results)
    if any(
        identifier.casefold() in normalized.casefold()
        for identifier in known_ids
    ):
        return True
    for content in re.findall(
        r"[\[【\(（]([^\]】\)）]+)[\]】\)）]",
        normalized,
    ):
        candidate = content.strip().rstrip(".,;:，、 ")
        if re.fullmatch(
            SHORT_EVIDENCE_LABEL_RE.pattern,
            candidate,
            flags=re.IGNORECASE,
        ):
            return True
    return False


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
    "unspecified_context",
}
NO_RANGE_SCOPE_MATCHES = {
    "general_evidence",
    "local_example",
    "unspecified_scope",
}
# Smaller cosine gaps are treated as a semantic near-tie, not evidence that
# a measure-local source should displace the normal no-range bundle.
NO_RANGE_LOCAL_DENSE_DOMINANCE_MARGIN = 0.025
# Match the production hybrid candidate-admission floor. This matters after
# generation selection has already removed competitors: absence of a runner-
# up is not, by itself, proof that a weak local match is safe to extract.
NO_RANGE_LOCAL_DENSE_MIN_RELEVANCE_SCORE = 0.44
NO_RANGE_LOCAL_DENSE_MIN_CONTENT_SCORE = 0.43
NO_RANGE_LOCAL_RELEVANCE_LED_MIN_SCORE = 0.65
NO_RANGE_LOCAL_RELEVANCE_LED_DENSE_ONLY_MIN_SCORE = 0.70
NO_RANGE_LOCAL_RELEVANCE_LED_MARGIN = 0.05
NO_RANGE_LOCAL_RELEVANCE_LED_MAX_CONTENT_DEFICIT = 0.05
NO_RANGE_STABLE_CANDIDATE_MIN = 6
DENSE_SEMANTIC_AUTHORITY_MATCH_TYPE = "dense_semantic_authority"
EXPERT_SELECTOR_CANDIDATE_LIMIT = 12
EXPERT_SELECTOR_MAX_ITEMS = 3
RANGED_DENSE_AUTHORITY_MIN_RELEVANCE_SCORE = 0.60
RANGED_DENSE_AUTHORITY_MIN_CONTENT_SCORE = 0.58
RANGED_DENSE_AUTHORITY_RELEVANCE_MARGIN = 0.05
RANGED_DENSE_AUTHORITY_CONTENT_MARGIN = 0.025
NO_RANGE_GENERAL_DENSE_AUTHORITY_MIN_RELEVANCE_SCORE = 0.70
NO_RANGE_GENERAL_DENSE_AUTHORITY_MIN_CONTENT_SCORE = 0.70
NO_RANGE_GENERAL_DENSE_AUTHORITY_MIN_CONCEPT_COVERAGE = 0.50
NO_RANGE_GENERAL_DENSE_AUTHORITY_MARGIN = 0.075
RANGED_DEICTIC_QUERY_RE = re.compile(
    r"(?:이|해당|선택(?:한|된)?)\s*"
    r"(?:부분|구간|대목|마디)|(?:여기|이곳)(?:에서|은|는|을|를)?"
)
GENERIC_RANGED_QUERY_CONCEPTS = {
    "가창",
    "곡",
    "구간",
    "노래",
    "대목",
    "마디",
    "방법",
    "부분",
    "부르",
    "어떻",
    "연주",
    "음악",
    "작품",
    "표현",
}


def is_underspecified_ranged_query(
    query: str,
    piece: str | None,
    measure_ranges: List[List[int]],
) -> bool:
    """Identify deictic range questions with no musical subject.

    A selected range supplies location, not semantics.  Wording such as
    ``이 부분은 어떻게 노래해야 할까?`` therefore cannot choose among
    unrelated annotations merely because they contain generic singing verbs.
    """

    if not measure_ranges or not RANGED_DEICTIC_QUERY_RE.search(query):
        return False
    concepts = semantic_concepts(query, piece, query=True)
    return not any(
        concept not in GENERIC_RANGED_QUERY_CONCEPTS
        for concept in concepts
    )


def _answer_claims(answer: str) -> List[str]:
    """Split generated prose into citation-bearing claim-sized lines."""

    claims: List[str] = []
    sanitized_answer = _remove_evidence_footer(answer)
    for line in unicodedata.normalize("NFKC", sanitized_answer).splitlines():
        line = ANSWER_LIST_PREFIX_RE.sub("", line.strip())
        if not line or line.startswith("제공된 검색 근거:"):
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
    """Reject score locations not authorized by the user's selected range.

    A canonical local-example range is routing metadata.  It does not become
    user-selected merely because that evidence was retrieved, so no-range
    answers must not volunteer any score locator.
    """

    locators = _claim_measure_locators(claim)
    if not locators:
        return False
    del results, cited
    if selected_measure_ranges:
        return any(
            not any(
                selected_start <= start and end <= selected_end
                for selected_start, selected_end in selected_measure_ranges
            )
            for start, end in locators
        )
    return True


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
    the generated claim identifies the evidence item internally. Their exact
    canonical ranges remain hidden routing metadata. A different whole-piece
    record can support a broad claim only when that record is cited in the
    same claim.
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


def _expert_record_is_finalized(record: Dict[str, Any]) -> bool:
    """Return whether an expert record obeys the release-only runtime schema."""

    return expert_record_is_runtime_final(record)


def expert_prompt_factuality_material(
    record: Dict[str, Any],
    measure_ranges: List[List[int]] | None = None,
) -> Dict[str, Any]:
    """Return expert prose that is actually exposed to grounded generation.

    The result intentionally mirrors ``build_context``: raw source Q&A and
    editorial metadata never become factuality material. Confirmed range
    metadata may exclude a non-overlapping record, but it never replaces the
    reviewed knowledge-unit answer with source prose.
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
    curated_answer = re.sub(
        r"\s+",
        " ",
        record.get("answer", ""),
    ).strip()
    return {
        "range_partitioned": False,
        "answer_texts": [curated_answer] if curated_answer else [],
        "question_contexts": [],
        "source_ids": [],
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
            "evidence_type: %s" % record.get("evidence_type", "unknown"),
            "piece: %s" % record["piece"],
            "work: %s" % record["work"],
        ]
        if record.get("topic"):
            lines.append("topic: %s" % record["topic"])
        hide_unrequested_local_range = (
            not measure_ranges and result.scope_match == "local_example"
        )
        if not hide_unrequested_local_range:
            lines.extend(
                [
                    "measure_range: %s"
                    % format_measure_range(
                        record.get("measure_range") or []
                    ),
                    "measure_scope: %s"
                    % record.get("measure_scope", ""),
                ]
            )
        lines.extend(
            [
                "query_scope_relation: %s" % result.scope_match,
                "evidence_role: %s" % evidence_role,
            ]
        )
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
        elif result.scope_match == "unspecified_context":
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
                "usage_constraint: This is citable passage-specific advice "
                "that directly matches a broad, no-range question. Its "
                "canonical location is hidden presentation metadata. Cite "
                "the [E#] label for internal validation, phrase the advice "
                "naturally around the user's situation, and never print a "
                "measure number, range, local-example label, scope notice, "
                "or retrieval explanation. It does not establish that the "
                "claim applies throughout the work, frequently, or in many "
                "places."
            )
        if record.get("evidence_type") == "web_database":
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
        if record.get("question"):
            lines.append("question: %s" % record["question"])
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


def has_only_web_grounding(results: List[Any]) -> bool:
    """Return whether every answer-authoritative record is web evidence."""

    primary = primary_grounding_results(results)
    return bool(primary) and all(
        result.record.get("evidence_type") == "web_database"
        for result in primary
    )


def has_selected_range_grounding(results: List[Any]) -> bool:
    """Return whether confirmed evidence overlaps the selected range."""

    return any(
        result.scope_match == "overlaps_query_range"
        for result in results
    )


def semantically_dominant_no_range_local_expert_bundle(
    results: List[Any],
) -> List[Any]:
    """Focus generation on a clearly dominant local expert source bundle.

    A no-range request does not make measure-local knowledge irrelevant; it
    means only that the generated claim must remain explicitly local. Hybrid
    retrieval embeds both each record's relevance text (including immutable
    source-question aliases) and its finalized curated answer. A local expert
    source is focused when both dense views agree, or when relevance clearly
    dominates while the curated-answer view remains within a narrow deficit
    and lexical/fusion evidence (or an unusually high dense score) supplies
    independent corroboration. These gates prevent measure metadata or a
    one-view near-match from overpromoting an unrelated detail.

    Knowledge units split from the winning immutable source answer remain
    together. A dense semantic winner may sit below fusion-ranked results only
    when all preceding results are clean, finalized siblings from that same
    source; the semantic winner is still returned first. The complete
    retrieval list itself is not changed, so callers retain broader
    diagnostics even when generation receives this compact bundle.
    """

    if not results or any(
        result.scope_match
        in RANGED_PRIMARY_SCOPE_MATCHES
        | RANGED_NON_GROUNDING_SCOPE_MATCHES
        for result in results
    ):
        return []
    dense_results = [
        result
        for result in results
        if (
            result.retrieval_mode in {"dense", "hybrid"}
            and result.dense_score > 0
            and result.dense_content_score > 0
        )
    ]
    if not dense_results:
        return []
    source_question_led_results = [
        result
        for result in dense_results
        if result.semantic_match_type
        == SOURCE_QUESTION_LED_MATCH_TYPE
    ]
    if len(source_question_led_results) > 1:
        return []
    relevance_winner = (
        source_question_led_results[0]
        if source_question_led_results
        else max(
            dense_results,
            key=lambda result: result.dense_score,
        )
    )
    source_question_led = bool(source_question_led_results)

    def clean_finalized_expert(result: Any) -> bool:
        return _expert_record_is_finalized(result.record)

    winner_source_ids = set(
        relevance_winner.record.get("source_ids") or []
    )
    winner_position = next(
        (
            index
            for index, result in enumerate(results)
            if result is relevance_winner
        ),
        None,
    )
    preceding_results = (
        results[:winner_position]
        if winner_position is not None
        else []
    )
    preceding_results_are_clean_same_source_siblings = bool(
        winner_source_ids
    ) and all(
        clean_finalized_expert(result)
        and bool(
            winner_source_ids.intersection(
                result.record.get("source_ids") or []
            )
        )
        for result in preceding_results
    )
    if (
        winner_position is None
        or (
            winner_position > 0
            and not preceding_results_are_clean_same_source_siblings
        )
        or relevance_winner.scope_match != "local_example"
        or not clean_finalized_expert(relevance_winner)
        or relevance_winner.record.get("measure_status") != "specific"
        or not (relevance_winner.record.get("measure_range") or [])
        or (
            source_question_led
            and (
                not bool(
                    relevance_winner.record.get("retrieval_aliases") or []
                )
                or not winner_source_ids
                or relevance_winner.dense_score
                < SOURCE_QUESTION_LED_MIN_RELEVANCE_SCORE
                or relevance_winner.dense_content_score
                < SOURCE_QUESTION_LED_MIN_CONTENT_SCORE
            )
        )
    ):
        return []

    siblings = [
        result
        for result in results
        if (
            result is not relevance_winner
            and winner_source_ids
            and clean_finalized_expert(result)
            and winner_source_ids.intersection(
                result.record.get("source_ids") or []
            )
        )
    ]
    # The semantic winner is always Evidence 1. Same-source split KUs retain
    # retrieval order after it, so context truncation drops siblings before
    # it can ever displace the direct answer.
    bundle = [relevance_winner, *siblings]
    bundle_ids = {id(result) for result in bundle}
    dense_bundle = [
        result for result in dense_results if id(result) in bundle_ids
    ]
    best_relevance_score = max(
        result.dense_score for result in dense_bundle
    )
    best_content_score = max(
        result.dense_content_score for result in dense_bundle
    )
    if not source_question_led and (
        best_relevance_score
        < NO_RANGE_LOCAL_DENSE_MIN_RELEVANCE_SCORE
        or best_content_score
        < NO_RANGE_LOCAL_DENSE_MIN_CONTENT_SCORE
    ):
        return []

    competitors = [
        result for result in dense_results if id(result) not in bundle_ids
    ]
    if not competitors:
        return bundle
    relevance_margin = best_relevance_score - max(
        result.dense_score for result in competitors
    )
    content_margin = best_content_score - max(
        result.dense_content_score for result in competitors
    )
    if source_question_led:
        source_question_relevance_margin = (
            relevance_winner.dense_score
            - max(result.dense_score for result in competitors)
        )
        if (
            source_question_relevance_margin
            >= SOURCE_QUESTION_LED_MIN_OTHER_SOURCE_MARGIN
        ):
            return bundle
        return []
    two_view_consensus = (
        relevance_margin >= NO_RANGE_LOCAL_DENSE_DOMINANCE_MARGIN
        and content_margin >= NO_RANGE_LOCAL_DENSE_DOMINANCE_MARGIN
    )
    relevance_led = (
        best_relevance_score >= NO_RANGE_LOCAL_RELEVANCE_LED_MIN_SCORE
        and relevance_margin >= NO_RANGE_LOCAL_RELEVANCE_LED_MARGIN
        and content_margin
        >= -NO_RANGE_LOCAL_RELEVANCE_LED_MAX_CONTENT_DEFICIT
        and (
            (
                relevance_winner.retrieval_mode == "hybrid"
                and relevance_winner.text_score > 0
            )
            or best_relevance_score
            >= NO_RANGE_LOCAL_RELEVANCE_LED_DENSE_ONLY_MIN_SCORE
        )
    )
    if not two_view_consensus and not relevance_led:
        return []
    return bundle


def semantically_dominant_ranged_expert_bundle(
    results: List[Any],
) -> List[Any]:
    """Return one unambiguous, range-applicable finalized expert family.

    A selected score range is useful routing evidence, but overlap alone does
    not prove that a dense hit answers the question.  This gate admits a sole
    strong expert hit, or a winner that independently leads every unrelated
    overlapping expert in both retrieval and curated-answer embedding views.
    Exact-source global siblings may accompany it when they also have direct
    lexical or source-question support.  Ambiguous overlapping families stay
    unavailable instead of being handed to the answer model.
    """

    if not results or not any(
        result.scope_match
        in RANGED_PRIMARY_SCOPE_MATCHES | RANGED_NON_GROUNDING_SCOPE_MATCHES
        for result in results
    ):
        return []
    if any(result.alias_score > 0 for result in results):
        # Positive immutable source-question aliases already have their own
        # stronger authority path.
        return []

    overlapping = [
        result
        for result in results
        if (
            result.scope_match == "overlaps_query_range"
            and result.record.get("evidence_type") == "expert_annotation"
            and _expert_record_is_finalized(result.record)
            and result.retrieval_mode in {"dense", "hybrid"}
            and result.dense_score
            >= RANGED_DENSE_AUTHORITY_MIN_RELEVANCE_SCORE
            and result.dense_content_score
            >= RANGED_DENSE_AUTHORITY_MIN_CONTENT_SCORE
        )
    ]
    if not overlapping:
        return []
    anchor = overlapping[0]
    anchor_sources = set(anchor.record.get("source_ids") or [])
    if not anchor_sources:
        return []

    competing_families = [
        result
        for result in results
        if (
            result.scope_match == "overlaps_query_range"
            and result.record.get("evidence_type") == "expert_annotation"
            and _expert_record_is_finalized(result.record)
            and not anchor_sources.intersection(
                result.record.get("source_ids") or []
            )
            and result.dense_score > 0
            and result.dense_content_score > 0
        )
    ]
    if competing_families:
        if (
            anchor.dense_score
            - max(result.dense_score for result in competing_families)
            < RANGED_DENSE_AUTHORITY_RELEVANCE_MARGIN
            or anchor.dense_content_score
            - max(
                result.dense_content_score
                for result in competing_families
            )
            < RANGED_DENSE_AUTHORITY_CONTENT_MARGIN
        ):
            return []

    siblings = [
        result
        for result in results
        if (
            result is not anchor
            and result.scope_match == "global_context"
            and result.record.get("evidence_type") == "expert_annotation"
            and _expert_record_is_finalized(result.record)
            and set(result.record.get("source_ids") or []) == anchor_sources
            and (
                result.alias_score > 0
                or result.semantic_match_type
                == SOURCE_QUESTION_LED_MATCH_TYPE
                or (
                    result.semantic_match_type == "strict"
                    and result.text_score > 0
                    and _has_answer_side_subject_support(result)
                )
            )
        )
    ]
    return [anchor, *siblings]


def semantically_dominant_no_range_general_expert_bundle(
    results: List[Any],
) -> List[Any]:
    """Return one exceptionally strong whole/general expert answer.

    This path is deliberately narrower than ordinary dense retrieval.  Both
    the source-question/relevance view and the finalized-answer view must be
    strong, explicit query concepts must occur in the answer, and the anchor
    must lead every unrelated expert family by a material margin.  It recovers
    self-contained definition-and-application answers without allowing a
    small embedding lead to rewrite a nearby but different expert meaning.
    """

    if not results or any(
        result.scope_match
        in RANGED_PRIMARY_SCOPE_MATCHES | RANGED_NON_GROUNDING_SCOPE_MATCHES
        for result in results
    ):
        return []
    anchor = results[0]
    if (
        anchor.scope_match not in {
            "general_evidence",
            "unspecified_scope",
        }
        or anchor.record.get("evidence_type") != "expert_annotation"
        or not _expert_record_is_finalized(anchor.record)
        or anchor.retrieval_mode not in {"dense", "hybrid"}
        or anchor.dense_score
        < NO_RANGE_GENERAL_DENSE_AUTHORITY_MIN_RELEVANCE_SCORE
        or anchor.dense_content_score
        < NO_RANGE_GENERAL_DENSE_AUTHORITY_MIN_CONTENT_SCORE
        or anchor.concept_coverage
        < NO_RANGE_GENERAL_DENSE_AUTHORITY_MIN_CONCEPT_COVERAGE
        or anchor.content_concept_coverage
        < NO_RANGE_GENERAL_DENSE_AUTHORITY_MIN_CONCEPT_COVERAGE
    ):
        return []
    anchor_sources = set(anchor.record.get("source_ids") or [])
    if not anchor_sources:
        return []
    competitors = [
        result
        for result in results[1:]
        if (
            result.record.get("evidence_type") == "expert_annotation"
            and _expert_record_is_finalized(result.record)
            and not anchor_sources.intersection(
                result.record.get("source_ids") or []
            )
            and result.dense_score > 0
            and result.dense_content_score > 0
        )
    ]
    if competitors and (
        anchor.dense_score
        - max(result.dense_score for result in competitors)
        < NO_RANGE_GENERAL_DENSE_AUTHORITY_MARGIN
        or anchor.dense_content_score
        - max(result.dense_content_score for result in competitors)
        < NO_RANGE_GENERAL_DENSE_AUTHORITY_MARGIN
    ):
        return []
    return [anchor]


def select_generation_evidence(
    results: List[Any],
    *,
    query: str | None = None,
    piece: str | None = None,
    measure_ranges: List[List[int]] | None = None,
) -> List[Any]:
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

    selected_ranges = measure_ranges or []
    if (
        query is not None
        and is_underspecified_ranged_query(query, piece, selected_ranges)
    ):
        # A deictic range plus a generic verb does not identify any global
        # topic.  The range itself may qualify a finalized annotation only
        # when the surviving local evidence is independently answer-supported
        # and belongs to one immutable source bundle.  Otherwise abstain.
        overlapping = [
            result
            for result in results
            if (
                result.scope_match == "overlaps_query_range"
                and result.record.get("evidence_type")
                == "expert_annotation"
                and _expert_record_is_finalized(result.record)
                and _has_direct_query_signal(result)
                and _has_answer_side_subject_support(result)
            )
        ]
        source_sets = {
            tuple(sorted(result.record.get("source_ids") or []))
            for result in overlapping
            if result.record.get("source_ids")
        }
        if len(source_sets) != 1 or any(
            not result.record.get("source_ids")
            for result in overlapping
        ):
            return []
        return overlapping

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
            semantic_focus = (
                semantically_dominant_no_range_local_expert_bundle(
                    results
                )
            )
            if semantic_focus:
                return semantic_focus
            if (
                results
                and results[0].record.get("evidence_type")
                == "web_database"
                and _has_direct_query_signal(results[0])
            ):
                # A direct web result must not carry lower, merely related
                # expert records into generation. Otherwise the model could
                # ignore ranking and turn optional biography/background into
                # the answer to a date, rights, or catalog question.
                return [
                    result
                    for result in results
                    if result.record.get("evidence_type")
                    == "web_database"
                ]
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


def build_messages(piece: str | None, measures: str, question: str, results: List[Any]) -> List[Dict[str, str]]:
    measure_ranges = parse_measure_ranges(measures) if measures else []
    user_prompt = """User query
piece: {piece}
measure_range: {measures}
question: {question}
/no_think

Retrieved evidence
{context}
""".format(
        piece=piece or "(not specified)",
        measures=measures or "(none selected)",
        question=question,
        context=build_context(results, measure_ranges),
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def is_grounded_insufficiency_answer(answer: str) -> bool:
    """Recognize the grounded model's exact no-answer signal."""

    if answer.strip() and not _remove_evidence_footer(answer):
        return True
    normalized = re.sub(r"\s+", "", answer)
    return NO_GROUNDED_ANSWER_SENTINEL.lower() in normalized.lower()


def _has_direct_query_signal(result: Any) -> bool:
    """Return whether one ranked record independently grounds the query."""

    answer_side_support = _has_answer_side_subject_support(result)
    return bool(
        (result.alias_score > 0 and answer_side_support)
        or result.semantic_match_type
        in {
            SOURCE_QUESTION_LED_MATCH_TYPE,
            DENSE_SEMANTIC_AUTHORITY_MATCH_TYPE,
        }
        or (
            result.semantic_match_type == "strict"
            and result.text_score > 0
            and answer_side_support
        )
    )


def _has_answer_side_subject_support(result: Any) -> bool:
    """Require evidence content, not just its question/metadata, to match."""

    return bool(
        result.answer_relation_score > 0
        or result.content_concept_coverage >= 0.5
        or result.dense_content_score
        >= NO_RANGE_LOCAL_DENSE_MIN_CONTENT_SCORE
    )


def has_grounded_generation_authority(results: List[Any]) -> bool:
    """Allow model synthesis only for a directly supported leading web hit.

    Finalized expert evidence is rendered only through
    ``select_authoritative_expert_bundle``. Ordinary dense similarity cannot
    authorize an expert claim or send it through the LLM, because a nearby
    passage can preserve nearly all wording while changing the intended
    musical meaning.
    """

    primary = primary_grounding_results(results)
    return bool(
        primary
        and primary[0].record.get("evidence_type") == "web_database"
        and _has_direct_query_signal(primary[0])
    )


def select_authoritative_expert_bundle(
    results: List[Any],
    *,
    max_items: int = 3,
) -> List[Any]:
    """Select complete reviewed expert units that can answer this query.

    Retrieval and generation selection establish candidate order and range
    applicability, but a lower expert candidate is not automatically direct
    authority. The leading primary result must itself be a finalized expert
    with an explicit lexical, alias, relation, or calibrated semantic signal.
    Thus a direct web result remains authoritative over a merely related
    expert result farther down the ranking.

    The smallest sufficient unit is the direct anchor. Additional units are
    admitted only when they have the same complete immutable source set and
    the same positive source-question alias tier as the anchor, and each has
    its own direct semantic support. Mere source-ID overlap is insufficient:
    one broad annotation can be split into an exact answer and unrelated
    background. Duplicate curated answers are removed without merging units.
    """

    if max_items < 1 or not results:
        return []
    primary = primary_grounding_results(results)
    if not primary:
        return []

    def is_direct_expert(result: Any) -> bool:
        record = result.record
        if (
            record.get("evidence_type") != "expert_annotation"
            or not _expert_record_is_finalized(record)
            or not str(record.get("answer") or "").strip()
            or result.semantic_match_type == "selector_candidate"
        ):
            return False
        return _has_direct_query_signal(result)

    anchor = primary[0]
    if not is_direct_expert(anchor):
        return []
    anchor_sources = set(anchor.record.get("source_ids") or [])
    candidates = [anchor]
    if (
        anchor.semantic_match_type == "broad_guidance"
        and anchor.alias_score <= 0
    ):
        # The broad-guidance retriever facet-scores every returned unit
        # independently and intentionally interleaves complementary facets.
        # Preserve that compact bundle, but do not admit ordinary dense or
        # lexical records that merely happened to follow it in the ranking.
        candidates.extend(
            result
            for result in primary[1:]
            if (
                result.semantic_match_type == "broad_guidance"
                and is_direct_expert(result)
            )
        )
    elif (
        anchor.semantic_match_type == SOURCE_QUESTION_LED_MATCH_TYPE
        and anchor_sources
    ):
        candidates.extend(
            result
            for result in primary[1:]
            if (
                result.semantic_match_type
                == SOURCE_QUESTION_LED_MATCH_TYPE
                and bool(
                    anchor_sources.intersection(
                        result.record.get("source_ids") or []
                    )
                )
                and is_direct_expert(result)
            )
        )
    elif (
        anchor.semantic_match_type
        == DENSE_SEMANTIC_AUTHORITY_MATCH_TYPE
        and anchor_sources
    ):
        candidates.extend(
            result
            for result in primary[1:]
            if (
                result.semantic_match_type
                == DENSE_SEMANTIC_AUTHORITY_MATCH_TYPE
                and set(result.record.get("source_ids") or [])
                == anchor_sources
                and is_direct_expert(result)
            )
        )
    elif anchor.alias_score > 0 and anchor_sources:
        candidates.extend(
            result
            for result in primary[1:]
            if (
                result.alias_score == anchor.alias_score
                and set(result.record.get("source_ids") or [])
                == anchor_sources
                and is_direct_expert(result)
            )
        )

    selected: List[Any] = []
    seen_answers: set[str] = set()
    for result in candidates:
        record = result.record
        answer = re.sub(r"\s+", " ", str(record.get("answer") or "")).strip()
        if not answer or answer in seen_answers:
            continue
        selected.append(result)
        seen_answers.add(answer)
        if len(selected) >= max_items:
            break
    return selected


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
            result.record["answer"].strip()
            for result in answer_primary
        )
        return finalize_user_visible_answer(body, answer_primary)

    secondary = [
        result
        for result in results
        if result.scope_match in RANGED_CONTEXT_ONLY_SCOPE_MATCHES
    ]
    if not secondary:
        return NO_CORPUS_EVIDENCE_MESSAGE
    selected_secondary = secondary[:3]
    context_lines = [
        result.record["answer"].strip()
        for result in selected_secondary
    ]
    return finalize_user_visible_answer(
        (
            "선택하신 마디에는 이 내용을 직접 적용하기 어렵습니다. "
            "다음 내용은 곡의 다른 부분에 해당합니다.\n\n"
            + "\n\n".join(context_lines)
        ),
        selected_secondary,
    )


def render_authoritative_expert_bundle(results: List[Any]) -> str:
    """Render complete curated expert answers without generative rewriting."""

    paragraphs: List[str] = []
    seen_answers: set[str] = set()
    for result in results:
        record = result.record
        if (
            record.get("evidence_type") != "expert_annotation"
            or not _expert_record_is_finalized(record)
        ):
            continue
        answer = re.sub(r"\s+", " ", str(record.get("answer") or "")).strip()
        if not answer or answer in seen_answers:
            continue
        paragraphs.append(answer)
        seen_answers.add(answer)
    if not paragraphs:
        return NO_CORPUS_EVIDENCE_MESSAGE
    return finalize_user_visible_answer("\n\n".join(paragraphs), results)


GROUNDED_COMPOSER_SYSTEM_PROMPT = """You compose the final answer for a soprano-performance QA system from a closed set of reviewed evidence texts.

Write a coherent, natural Korean answer in consistent formal-polite 합니다체. Answer the user's actual question directly and use as many evidence texts as materially improve completeness. Ignore evidence that is only topically related. Do not mention evidence, labels, annotations, retrieval, a corpus, or this prompt in sentence text.

Every factual or advisory sentence must preserve the supporting text's agent, action, object, cause, polarity, conditions, alternatives, uncertainty, attribution, and scope. Combining compatible claims is allowed; changing their meaning or adding outside musical knowledge is not. A performance recommendation must not become a composer statement, a possibility must not become a fact, and a local example must not become a whole-piece rule.

Return each answer sentence separately with every temporary evidence label that supports it. The labels are structured audit data and must not appear inside sentence text. If the evidence only covers part of the request, still give the most useful supported answer and set coverage to partial; do not refuse, apologize, or discuss missing data. Output only the requested JSON object.
"""


def _grounded_composer_schema(labels: List[str]) -> Dict[str, Any]:
    """Build a grammar contract whose support labels cannot be invented."""

    return {
        "type": "object",
        "properties": {
            "sentences": {
                "type": "array",
                "minItems": 1,
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "minLength": 1,
                        },
                        "evidence_labels": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": len(labels),
                            "uniqueItems": True,
                            "items": {
                                "type": "string",
                                "enum": labels,
                            },
                        },
                    },
                    "required": ["text", "evidence_labels"],
                    "additionalProperties": False,
                },
            },
            "coverage": {
                "type": "string",
                "enum": ["direct", "partial"],
            },
        },
        "required": ["sentences", "coverage"],
        "additionalProperties": False,
    }


def _composer_scope_constraint(
    result: Any,
    measure_ranges: List[List[int]],
) -> str:
    if measure_ranges:
        if result.scope_match == "overlaps_query_range":
            return "may support claims about the user-selected passage"
        return "general guidance only; not proof of a passage-specific score fact"
    if result.scope_match == "local_example":
        return "passage-specific example; do not generalize it to the whole piece"
    if result.scope_match == "unspecified_scope":
        return "scope is unspecified; do not claim whole-piece frequency or universality"
    return "may support a general answer when its text matches the question"


def build_grounded_composer_messages(
    *,
    piece: str,
    measure_ranges: List[List[int]],
    question: str,
    results: List[Any],
) -> List[Dict[str, str]]:
    """Expose complete reviewed texts, but no opaque IDs or editorial data."""

    evidence = []
    for index, result in enumerate(results, start=1):
        record = result.record
        evidence.append(
            {
                "label": f"E{index}",
                "scope_constraint": _composer_scope_constraint(
                    result,
                    measure_ranges,
                ),
                "source_question_hints": list(
                    record.get("retrieval_aliases")
                    or ([record["question"]] if record.get("question") else [])
                ),
                "reviewed_answer_text": re.sub(
                    r"\s+",
                    " ",
                    str(record.get("answer") or ""),
                ).strip(),
            }
        )
    payload = {
        "piece": piece,
        "user_selected_a_measure_range": bool(measure_ranges),
        "question": question,
        "evidence": evidence,
    }
    return [
        {"role": "system", "content": GROUNDED_COMPOSER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Compose the grounded answer from this closed evidence set.\n"
                + json.dumps(
                    payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n/no_think"
            ),
        },
    ]


def compose_grounded_answer_with_model(
    model_path: str,
    *,
    piece: str,
    measure_ranges: List[List[int]],
    question: str,
    results: List[Any],
    llm_settings: Dict[str, Any],
) -> tuple[str, List[Any], str]:
    """Compose once from full reviewed texts and return only cited evidence.

    The model never receives opaque knowledge-unit IDs. Temporary E-labels are
    grammar constrained, validated structurally, and removed before display.
    There is no alternate model, internal-knowledge answer, or prose fallback.
    """

    candidates = [
        result
        for result in primary_grounding_results(results)
        if str(result.record.get("answer") or "").strip()
    ]
    if not candidates:
        raise GeneratedAnswerRejected(
            "No answer-authoritative reviewed text is available for composition"
        )
    labels = [f"E{index}" for index in range(1, len(candidates) + 1)]
    messages = build_grounded_composer_messages(
        piece=piece,
        measure_ranges=measure_ranges,
        question=question,
        results=candidates,
    )
    composition = generate_json(
        model_path,
        messages,
        llm_settings,
        schema=_grounded_composer_schema(labels),
        max_tokens=768,
    )
    sentences = composition.get("sentences")
    coverage = composition.get("coverage")
    if (
        not isinstance(sentences, list)
        or not sentences
        or len(sentences) > 8
        or coverage not in {"direct", "partial"}
    ):
        raise GeneratedAnswerRejected(
            "Grounded composer violated its response contract"
        )

    label_to_result = dict(zip(labels, candidates))
    sentence_texts: List[str] = []
    cited_answer_parts: List[str] = []
    used_labels: set[str] = set()
    for sentence in sentences:
        if not isinstance(sentence, dict):
            raise GeneratedAnswerRejected(
                "Grounded composer returned a malformed sentence"
            )
        text = re.sub(r"\s+", " ", str(sentence.get("text") or "")).strip()
        evidence_labels = sentence.get("evidence_labels")
        if (
            not text
            or not isinstance(evidence_labels, list)
            or not evidence_labels
            or len(set(evidence_labels)) != len(evidence_labels)
            or any(label not in label_to_result for label in evidence_labels)
        ):
            raise GeneratedAnswerRejected(
                "Grounded composer returned invalid sentence support"
            )
        sentence_texts.append(text)
        used_labels.update(evidence_labels)
        cited_answer_parts.append(
            f"{text} [{', '.join(evidence_labels)}]"
        )

    raw_answer = " ".join(sentence_texts)
    cited_answer = " ".join(cited_answer_parts)
    suspicious = suspicious_generation_tokens(raw_answer, messages)
    if suspicious:
        raise GeneratedAnswerRejected(
            "Generated answer contains unsupported tokens: %s"
            % ", ".join(suspicious)
        )
    if answer_overgeneralizes_local_examples(
        cited_answer,
        candidates,
        measure_ranges,
    ):
        raise GeneratedAnswerRejected(
            "Generated answer overgeneralizes local evidence"
        )
    used_results = [
        result
        for label, result in label_to_result.items()
        if label in used_labels
    ]
    finalized = finalize_user_visible_answer(raw_answer, used_results)
    if finalized == NO_CORPUS_EVIDENCE_MESSAGE:
        raise GeneratedAnswerRejected(
            "Generated answer contains forbidden internal metadata"
        )
    return finalized, used_results, coverage


EXPERT_SELECTOR_SYSTEM_PROMPT = """You are a closed-world evidence reranker for a soprano-performance QA system.

Do not answer the user and do not use outside musical knowledge. The finalized answer is the sole claim authority; aliases are retrieval hints, not additional facts.

First identify the core relation requested: identification or meaning, cause or intent, effect or interpretation, performance procedure, placement or scope, comparison or choice, or broad performance guidance. Then choose one anchor whose finalized answer most directly states that relation for the named musical subject. A how-to statement does not establish a cause; pronunciation or transition advice does not establish placement; shared musical vocabulary does not establish relevance.

After choosing the anchor, add a complementary candidate only if removing it would leave a requested part or a genuinely useful facet of an explicitly broad request materially unanswered. A complement must state its own directly responsive claim. Do not add background, a weaker alternative, a conflicting recommendation, a duplicate, or merely related advice. If the anchor fully answers the request, return no complements.

For a comparison or choice, preserve supported alternatives and uncertainty; do not combine a permissive answer with a categorical instruction that changes it. Scope eligibility has already been enforced, so semantic fit—not location alone—decides. Never infer a cause, intention, rule, frequency, location, certainty, or connection across candidates.

If nothing fully answers, choose the safest anchor that covers the largest supported part, keep complements minimal, and report partial coverage. Output only the requested JSON object.
"""


def _expert_selector_schema(candidate_ids: List[str]) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "request_relation": {
                "type": "string",
                "enum": [
                    "identification_meaning",
                    "cause_intent",
                    "effect_interpretation",
                    "performance_procedure",
                    "placement_scope",
                    "comparison_choice",
                    "broad_performance_guidance",
                ],
            },
            "anchor_id": {
                "type": "string",
                "enum": candidate_ids,
            },
            "complementary_ids": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": candidate_ids,
                },
                "maxItems": EXPERT_SELECTOR_MAX_ITEMS - 1,
                "uniqueItems": True,
            },
            "coverage": {
                "type": "string",
                "enum": ["direct", "partial"],
            },
        },
        "required": [
            "request_relation",
            "anchor_id",
            "complementary_ids",
            "coverage",
        ],
        "additionalProperties": False,
    }


def _expert_selector_candidates(results: List[Any]) -> List[Any]:
    """Keep only release-ready expert units that can authorize prose."""

    selected: List[Any] = []
    seen_ids: set[str] = set()
    for result in primary_grounding_results(results):
        record = result.record
        record_id = str(record.get("id") or "")
        if (
            not record_id
            or record_id in seen_ids
            or record.get("evidence_type") != "expert_annotation"
            or not _expert_record_is_finalized(record)
            or not str(record.get("answer") or "").strip()
        ):
            continue
        selected.append(result)
        seen_ids.add(record_id)
    return selected[:EXPERT_SELECTOR_CANDIDATE_LIMIT]


def build_expert_selector_messages(
    *,
    piece: str,
    measure_ranges: List[List[int]],
    question: str,
    candidates: List[Any],
) -> List[Dict[str, str]]:
    """Build the closed-world candidate-selection prompt."""

    payload = []
    for result in candidates:
        record = result.record
        payload.append(
            {
                "id": record["id"],
                "applicability": (
                    "selected_range_support"
                    if result.scope_match == "overlaps_query_range"
                    else (
                        "whole_piece_support"
                        if result.scope_match
                        in {"global_context", "general_evidence"}
                        else "local_example"
                    )
                ),
                "source_question_aliases": list(
                    record.get("retrieval_aliases") or []
                ),
                "finalized_answer": re.sub(
                    r"\s+",
                    " ",
                    str(record.get("answer") or ""),
                ).strip(),
            }
        )
    user_prompt = (
        "Select finalized expert evidence for this request.\n"
        + json.dumps(
            {
                "piece": piece,
                "selected_measure_ranges": measure_ranges,
                "question": question,
                "candidates": payload,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n/no_think"
    )
    return [
        {"role": "system", "content": EXPERT_SELECTOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def select_expert_bundle_with_model(
    model_path: str,
    *,
    piece: str,
    measure_ranges: List[List[int]],
    question: str,
    results: List[Any],
    llm_settings: Dict[str, Any],
) -> List[Any]:
    """Choose supplied finalized KUs with one constrained local-model call.

    The call selects IDs only.  User-visible prose is subsequently rendered
    from the immutable curated answers, so the selector cannot paraphrase,
    merge, or invent a musical claim.  Invalid output raises immediately;
    there is no rank-one, internal-knowledge, or alternate-model fallback.
    """

    candidates = _expert_selector_candidates(results)
    if not candidates:
        raise GeneratedAnswerRejected(
            "No scope-applicable finalized expert candidates are available"
        )
    candidate_by_id = {
        result.record["id"]: result for result in candidates
    }
    selection = generate_json(
        model_path,
        build_expert_selector_messages(
            piece=piece,
            measure_ranges=measure_ranges,
            question=question,
            candidates=candidates,
        ),
        llm_settings,
        schema=_expert_selector_schema(list(candidate_by_id)),
    )
    anchor_id = selection.get("anchor_id")
    complementary_ids = selection.get("complementary_ids")
    request_relation = selection.get("request_relation")
    coverage = selection.get("coverage")
    if (
        not isinstance(anchor_id, str)
        or not isinstance(complementary_ids, list)
        or len(complementary_ids) >= EXPERT_SELECTOR_MAX_ITEMS
        or any(not isinstance(item, str) for item in complementary_ids)
        or len(set(complementary_ids)) != len(complementary_ids)
        or anchor_id in complementary_ids
        or coverage not in {"direct", "partial"}
        or request_relation not in {
            "identification_meaning",
            "cause_intent",
            "effect_interpretation",
            "performance_procedure",
            "placement_scope",
            "comparison_choice",
            "broad_performance_guidance",
        }
    ):
        raise GeneratedAnswerRejected(
            "Grounded expert selector violated its response contract"
        )
    selected_ids = [anchor_id, *complementary_ids]
    unknown = [
        record_id
        for record_id in selected_ids
        if record_id not in candidate_by_id
    ]
    if unknown:
        raise GeneratedAnswerRejected(
            "Grounded expert selector returned IDs outside its candidates: "
            + ", ".join(unknown)
        )
    return [candidate_by_id[record_id] for record_id in selected_ids]


def _merge_unique_results(*groups: List[Any]) -> List[Any]:
    merged: List[Any] = []
    seen_ids: set[str] = set()
    for group in groups:
        for result in group:
            record_id = str(result.record.get("id") or "")
            if not record_id or record_id in seen_ids:
                continue
            merged.append(result)
            seen_ids.add(record_id)
    return merged


def retrieve_generation_evidence(
    index: SearchIndex,
    *,
    query: str,
    piece: str,
    measure_ranges: List[List[int]],
    measures: str,
    topic: str | None,
    top_k: int,
) -> tuple[List[Any], List[Dict[str, str]]]:
    """Retrieve and freeze the one evidence bundle used for generation.

    The function performs no model call, repair call, context retry, or
    alternate retrieval. Callers either render a complete authoritative
    expert unit or make exactly one grounded generation call with the
    returned messages.
    """

    if top_k < 1:
        return [], build_messages(piece, measures, query, [])
    candidate_limit = (
        max(top_k, NO_RANGE_STABLE_CANDIDATE_MIN)
        if not measure_ranges
        else top_k
    )
    stable_candidates = index.search(
        query=query,
        piece=piece,
        measure_ranges=measure_ranges,
        topic=topic,
        top_k=candidate_limit,
    )
    rejected_candidates = list(
        getattr(index, "last_answer_candidates", []) or []
    )
    semantic_authority = (
        (
            semantically_dominant_no_range_local_expert_bundle(
                stable_candidates
            )
            or semantically_dominant_no_range_general_expert_bundle(
                stable_candidates
            )
        )
        if not measure_ranges
        else semantically_dominant_ranged_expert_bundle(
            stable_candidates
        )
    )
    selected_results = select_generation_evidence(
        stable_candidates,
        query=query,
        piece=piece,
        measure_ranges=measure_ranges,
    )
    if semantic_authority:
        authority_ids = {
            result.record["id"] for result in semantic_authority
        }
        if not measure_ranges:
            selected_results = list(semantic_authority)
        else:
            selected_results = [
                *semantic_authority,
                *(
                    result
                    for result in selected_results
                    if (
                        result.record["id"] not in authority_ids
                        and result.scope_match
                        in RANGED_CONTEXT_ONLY_SCOPE_MATCHES
                    )
                ),
            ]
        selected_results = [
            replace(
                result,
                semantic_match_type=(
                    DENSE_SEMANTIC_AUTHORITY_MATCH_TYPE
                    if result.record["id"] in authority_ids
                    else result.semantic_match_type
                ),
            )
            for result in selected_results
        ]
    needs_expert_selector = (
        not select_authoritative_expert_bundle(selected_results)
        and not has_grounded_generation_authority(selected_results)
    )
    if needs_expert_selector:
        semantic_search = getattr(index, "semantic_candidates", None)
        broad_candidates = (
            semantic_search(
                query=query,
                piece=piece,
                measure_ranges=measure_ranges,
                topic=topic,
                top_k=EXPERT_SELECTOR_CANDIDATE_LIMIT,
            )
            if callable(semantic_search)
            else []
        )
        selected_results = _merge_unique_results(
            rejected_candidates,
            selected_results,
            broad_candidates,
        )[:EXPERT_SELECTOR_CANDIDATE_LIMIT]
        selected_results = [
            replace(
                result,
                semantic_match_type="selector_candidate",
            )
            for result in selected_results
        ]
    result_limit = (
        EXPERT_SELECTOR_CANDIDATE_LIMIT
        if needs_expert_selector
        else top_k
    )
    results = selected_results[: min(result_limit, len(selected_results))]
    return results, build_messages(piece, measures, query, results)


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
    """Validate internal citations, then remove them from displayed prose."""

    answer = _remove_evidence_footer(answer)
    all_record_ids = [result.record["id"] for result in results]
    if not all_record_ids:
        return finalize_user_visible_answer(answer, results)
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
        raise GeneratedAnswerRejected(
            "Generated answer cites context-only evidence"
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
        raise GeneratedAnswerRejected(
            "Generated answer contains an unknown or context-only evidence ID"
        )

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
            raise GeneratedAnswerRejected(
                "Generated answer exposes provenance metadata"
            )
        if re.fullmatch(r"[0-9]+(?:\s*[,;]\s*[0-9]+)*", content):
            raise GeneratedAnswerRejected(
                "Generated answer contains an unresolvable citation"
            )
        if re.match(
            r"^(?:출처|근거|source|citation)\s*(?::|：|-)?",
            content,
            flags=re.IGNORECASE,
        ):
            raise GeneratedAnswerRejected(
                "Generated answer contains an unresolvable citation"
            )
        if re.match(
            r"^(?:E\s*[0-9]|Evidence\b|"
            r"(?:[a-z0-9]+-)+ku(?:[-_]|[0-9])|sqa(?:[-_]|[0-9])|"
            r"web(?:chunk|src|source|claim)(?:[-_])|source[_ -]?id\b)",
            content,
            flags=re.IGNORECASE,
        ):
            raise GeneratedAnswerRejected(
                "Generated answer contains an unresolvable citation"
            )
        return match.group(0)

    finalized = re.sub(
        r"\[([^\[\]]+)\]",
        remove_unknown_citation_like_bracket,
        finalized,
    )
    if not record_ids:
        return finalize_user_visible_answer(finalized, results)
    if not any("[%s]" % record_id in finalized for record_id in record_ids):
        raise GeneratedAnswerRejected(
            "Generated answer does not cite primary retrieved evidence"
        )
    return finalize_user_visible_answer(finalized, results)


def finalize_grounded_generated_answer(
    answer: str,
    results: List[Any],
    messages: List[Dict[str, str]],
    measure_ranges: List[List[int]],
) -> str:
    """Validate the sole model draft without repair or substitution."""

    if not answer.strip() or is_grounded_insufficiency_answer(answer):
        raise GeneratedAnswerRejected(
            "Grounded model returned no usable answer"
        )
    suspicious = suspicious_generation_tokens(answer, messages)
    if suspicious:
        raise GeneratedAnswerRejected(
            "Generated answer contains unsupported tokens: %s"
            % ", ".join(suspicious)
        )
    if answer_references_secondary_evidence(answer, results):
        raise GeneratedAnswerRejected(
            "Generated answer cites context-only evidence"
        )
    if answer_overgeneralizes_local_examples(
        answer,
        results,
        measure_ranges,
    ):
        raise GeneratedAnswerRejected(
            "Generated answer overgeneralizes local evidence"
        )
    finalized = finalize_answer_citations(answer, results)
    if finalized == NO_CORPUS_EVIDENCE_MESSAGE:
        raise GeneratedAnswerRejected(
            "Generated answer contains forbidden internal metadata"
        )
    return finalized


def ensure_corpus(settings: Dict[str, Any], rebuild: bool) -> None:
    stats_are_current = False
    expected_web_exports = settings.get("web_export_files", ["research-open.jsonl"])
    if os.path.exists(settings["stats_path"]):
        try:
            with open(settings["stats_path"], encoding="utf-8") as f:
                stats = json.load(f)
            stats_are_current = (
                isinstance(stats, dict)
                and stats.get("corpus_schema_version")
                == CORPUS_SCHEMA_VERSION
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
    parser.add_argument("--rebuild-corpus", action="store_true", help="Rebuild corpus before asking")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()
    try:
        selected_measure_ranges = parse_measure_ranges(args.measures)
        question_measure_ranges = validate_question_measure_contract(
            args.question,
            selected_measure_ranges,
        )
        args.selected_measure_ranges = selected_measure_ranges
        args.measure_ranges = (
            question_measure_ranges or selected_measure_ranges
        )
        args.measures = (
            format_measure_range(args.measure_ranges)
            if args.measure_ranges
            else ""
        )
    except ValueError as exc:
        parser.error(str(exc))
    return args


def _retrieval_unavailable_reason(
    index: SearchIndex,
    *,
    question: str,
    piece: str,
    measure_ranges: List[List[int]],
    results: List[Any],
) -> str:
    if retrieval_abstained_for_ambiguity(index):
        return "ambiguous_dense_grounding"
    if retrieval_abstained_for_missing_causal_authority(index):
        return "missing_scoped_causal_authority"
    if (
        not results
        and is_underspecified_ranged_query(
            question,
            piece,
            measure_ranges,
        )
    ):
        return "underspecified_ranged_question"
    return "no_corpus_evidence"


def main() -> None:
    args = parse_args()
    settings = load_settings(
        args.settings,
        use_legacy_dataset_env=False,
    ) if args.settings else load_settings(use_legacy_dataset_env=False)
    try:
        validate_retrieval_requirements(settings)
        if not args.no_generate:
            validate_generation_requirements(
                settings["model_path"],
                settings["llm"],
            )
    except (DenseRetrievalUnavailable, GenerationBackendUnavailable) as exc:
        raise SystemExit(str(exc)) from exc

    ensure_corpus(settings, args.rebuild_corpus)
    records = load_corpus(settings["corpus_path"])
    index = build_retrieval_index(records, settings)
    unavailable_reason: str | None = None

    if args.no_generate:
        results = index.search(
            query=args.question,
            piece=args.piece,
            measure_ranges=args.measure_ranges,
            topic=args.topic,
            top_k=args.top_k,
        )
        answer = ""
        generation_mode = (
            "retrieval_only" if results else "unavailable"
        )
        answer_basis = (
            "retrieved_evidence" if results else "no_corpus_evidence"
        )
        if not results:
            unavailable_reason = _retrieval_unavailable_reason(
                index,
                question=args.question,
                piece=args.piece,
                measure_ranges=args.measure_ranges,
                results=results,
            )
    else:
        results, messages = retrieve_generation_evidence(
            index,
            query=args.question,
            piece=args.piece,
            measure_ranges=args.measure_ranges,
            measures=args.measures,
            topic=args.topic,
            top_k=args.top_k,
        )
        if not results or (
            args.measure_ranges and not has_primary_grounding(results)
        ):
            answer = NO_CORPUS_EVIDENCE_MESSAGE
            generation_mode = "unavailable"
            answer_basis = "no_corpus_evidence"
            unavailable_reason = _retrieval_unavailable_reason(
                index,
                question=args.question,
                piece=args.piece,
                measure_ranges=args.measure_ranges,
                results=results,
            )
        else:
            authoritative_expert = select_authoritative_expert_bundle(
                results
            )
            if authoritative_expert:
                answer = render_authoritative_expert_bundle(
                    authoritative_expert
                )
                generation_mode = "expert_verbatim"
                answer_basis = "authoritative_expert"
            elif has_grounded_generation_authority(results):
                raw_answer = generate(
                    settings["model_path"],
                    messages,
                    settings["llm"],
                )
                answer = finalize_grounded_generated_answer(
                    raw_answer,
                    results,
                    messages,
                    args.measure_ranges,
                )
                generation_mode = "llm"
                answer_basis = "retrieved_evidence"
            else:
                selected_expert = select_expert_bundle_with_model(
                    settings["model_path"],
                    piece=args.piece,
                    measure_ranges=args.measure_ranges,
                    question=args.question,
                    results=results,
                    llm_settings=settings["llm"],
                )
                answer = render_authoritative_expert_bundle(
                    selected_expert
                )
                results = selected_expert
                generation_mode = "expert_verbatim"
                answer_basis = "authoritative_expert"

    answer = finalize_user_visible_answer(answer, results)
    if (
        answer == NO_CORPUS_EVIDENCE_MESSAGE
        and answer_basis not in {
            "no_corpus_evidence",
            "retrieved_secondary_context",
        }
    ):
        generation_mode = "unavailable"
        answer_basis = "no_corpus_evidence"
        unavailable_reason = "forbidden_internal_metadata"
    evidence_notices = build_evidence_notices(results)

    if args.json:
        print(
            json.dumps(
                {
                    "query": {
                        "piece": args.piece,
                        "measure_range": getattr(
                            args,
                            "selected_measure_ranges",
                            args.measure_ranges,
                        ),
                        "effective_measure_range": args.measure_ranges,
                        "question": args.question,
                        "topic": args.topic,
                        "top_k": args.top_k,
                    },
                    "answer": answer,
                    "generation_mode": generation_mode,
                    "answer_basis": answer_basis,
                    "unavailable_reason": unavailable_reason,
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
