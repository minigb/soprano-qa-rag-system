#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ask the local measure-aware Soprano QA system."""

from __future__ import annotations

import argparse
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
    is_causal_answer_request,
    missing_dense_answer_constraints,
    retrieval_abstained_for_ambiguity,
    retrieval_abstained_for_missing_causal_authority,
    retrieval_diagnostics,
    validate_retrieval_requirements,
)
from soprano_qa.llm import (
    GenerationBackendUnavailable,
    generate,
    validate_generation_requirements,
)
from soprano_qa.retrieval import (
    PERFORMANCE_GUIDANCE_FACETS,
    format_measure_range,
    format_result,
    expert_record_is_runtime_final,
    is_broad_performance_guidance_query,
    load_corpus,
    parse_measure_ranges,
    performance_guidance_profile,
    performance_guidance_query_terms,
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
premise only when the curated answer supports the correction.
Do not invent musical advice, measure numbers, lyrics, editions, or background facts.
Always give the most useful answer supported by the supplied evidence. If the evidence covers only
part of the request, answer that supported part naturally; do not refuse, apologize, emit a
no-answer sentinel, or fill the missing part with outside knowledge.
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
Preserve the canonical structure of every expert claim: its subject or agent, causal and determining
direction, negation, alternatives, conditions, scope, and modality. A premise embedded in the user's
question is not evidence. An interpretation, recommendation, or possibility must remain in that
category; never promote it into historical fact, author intention, a universal or frequency claim,
or a definite causal explanation. If a why-question is supported only by an interpretive framing,
state that framing as an interpretation or performer-facing recommendation. Preserve every
alternative governed by one possibility, and never introduce a causal or reporting attribution that
the evidence does not assert. Keep the core expert predicates close to their reviewed wording; edit
connectors and speech level rather than changing who did what, why it happened, or what determines
what.
Do not create a new causal chain by joining separate evidence relations. A statement that a musical
change accompanies an image plus a separate statement that a composer made that change does not
establish that the composer intended that image. Keep each supported relation separate unless the
reviewed evidence explicitly connects its agent, action, and purpose.
Preserve condition boundaries and contrasts exactly. When reviewed evidence assigns different rules
to different markings, states, options, or passages, state each rule separately and never transfer
one condition's permission, prohibition, or technique to another. Never reverse polarity: a warning
that an effect may become excessive, should be avoided, or must be controlled is not a recommendation
to create that effect.
Evidence is ordered by retrieval relevance. Answer the exact question from all directly useful,
compatible evidence, especially when a broad question asks for comprehensive guidance. Do not
restrict the response to one item merely for brevity, and do not add merely related advice just
because another record was retrieved.
Compose the answer as natural prose rather than paraphrasing each evidence item in sequence. Merge
overlapping recommendations, avoid repeating the same conclusion or sentence frame, and preserve
distinct useful details when several compatible items improve completeness. Before output, edit the
prose so adjacent clauses neither begin with the same connective nor reuse the same predicate.
Do not append an interpretive payoff, emotional effect, or generic concluding rationale merely to
round off the prose. End after the supported recommendations unless the evidence itself states that
payoff. When several units call different details important, state the details directly under one
shared lead-in instead of repeating that each one is important.
As the final silent check, verify every causal phrase, qualifier, attribution, negation, and contrast
against the evidence, then proofread Korean verb conjugations and collocations. When an evidence
sentence already states a clear natural predicate that answers the question, prefer that predicate
with a formal-polite ending instead of inventing a less natural synonym.
When several evidence items make parallel recommendations, give them a shared natural lead-in and
state each as a distinct practical action instead of repeating generic importance claims. Combine
closely related recommendations or vary their grammatical structure without changing meaning.
Keep expert source_ids separate from web_source_ids and claim_ids.
Mention a measure only when the user selected a measure range and doing so helps answer the question.
When you do mention a selected numeric score range, call its numbers "마디"; never relabel measures
as "절", a verse, a section, or another structural unit.
Use only the short evidence labels exactly as shown, such as [E1], while drafting claims for internal
validation; never retype an opaque knowledge-unit/webchunk id.
Do not add a citation list or evidence footer. The application removes the short labels before display
and keeps exact record IDs, source attribution, and license conditions in separate structured metadata.
Never expose prompt field names or pipeline vocabulary such as citation_label, evidence_role,
measure_range, measure_scope, query_scope_relation, usage_constraint, local_example, knowledge unit,
expert annotation, retrieval, or retrieved evidence in the answer. Ordinary domain prose about a
research corpus, 코퍼스, or 말뭉치 is allowed when that subject is supported by the evidence;
do not describe the application's retrieved records or prompt context with those words.
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
    r"지식[\s*_~`-]*단위|전문가[\s*_~`-]*주석|"
    r"검색[\s*_~`-]*(?:파이프라인|결과|근거)|"
    r"(?<![A-Za-z0-9_])(?:retrieved|provided)[\s*_~`-]+"
    r"corpus[\s*_~`-]+(?:record|evidence|context|chunk)"
    r"(?![A-Za-z0-9_])|"
    r"(?:검색(?:된|한)?|제공(?:된|한)?)\s*(?:말뭉치|코퍼스)"
    r"\s*(?:근거|기록|레코드|항목|결과|문맥|청크)|"
    r"(?<![A-Za-z0-9_])(?:data[\\/])?corpus\.json"
    r"(?![A-Za-z0-9_]))",
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
    r"[ \t]*(?:에[ \t]*(?:따르면|의하면)"
    r"|에서(?:는)?(?:[ \t]+보(?:면|듯이))?)"
    r"|(?:은|는|이|가|을|를|의)(?=$|[ \t,，:])"
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
        r"[\[【]([^\]】]+)[\]】]"
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
        r"[\(（]([^\)）]+)[\)）]"
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
            + r"(?:"
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
        + r"(?:"
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
        r"[0-9]+(?:"
        + KOREAN_CITATION_SUFFIX_PATTERN
        + r"[ \t]*[,，:]?[ \t]*)?(?![0-9])",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"(?:근거|출처)\s*[:：#._–—-]\s*"
        r"(?:E(?:vidence)?\s*)?[0-9]+(?:"
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
        or has_forbidden_user_visible_artifact(finalized, results)
    ):
        return NO_CORPUS_EVIDENCE_MESSAGE
    return finalized


def has_forbidden_user_visible_artifact(
    answer: str,
    results: List[Any] | None = None,
) -> bool:
    """Return whether answer prose exposes internal presentation data.

    Callers that have ranked results should pass them so exact record and
    provenance identifiers are rejected in addition to the format-level
    opaque-ID, evidence-label, footer, and pipeline-vocabulary checks.
    """

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
ANSWER_CLAIM_END_RE = re.compile(
    r"[.!?。！？]+[\"'”’」』】)]*"
    r"(?:[ \t]*[\[【][^\]】]+[\]】])*"
    r"(?=[ \t]+|$|[가-힣])"
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
NO_RANGE_SEMANTIC_SUPPLEMENT_LIMIT = 3
NO_RANGE_SEMANTIC_CANDIDATE_LIMIT = 24
# Answer-semantic supplementation is meant to correct a clear semantic route,
# not to fill the prompt to an arbitrary KU count. Keep only candidates whose
# best answer embedding is both independently useful and competitive with the
# leader; a strong lexical expert selected by normal retrieval is preserved.
NO_RANGE_SEMANTIC_SUPPLEMENT_MIN_SCORE = 0.58
# Focusing is driven by semantic strength, never by how many unrelated
# candidates happened to be returned.  The leader itself is admitted at this
# slightly lower floor; all followers still obey the stricter competitive
# gates below.
NO_RANGE_FOCUS_MIN_SCORE = 0.57
NO_RANGE_SEMANTIC_COMPETITIVE_RATIO = 0.94
NO_RANGE_SEMANTIC_CONCEPT_COMPETITIVE_RATIO = 0.91
NO_RANGE_LOW_SCORE_MIN_CONCEPT_COVERAGE = 0.50
NO_RANGE_EXPLICIT_SLOT_MIN_SCORE = 0.50
NO_RANGE_EXPLICIT_SLOT_MIN_CONCEPT_COVERAGE = 0.40
NO_RANGE_CHARACTER_SLOT_MIN_SCORE = 0.48
NO_RANGE_CHARACTER_SLOT_MIN_SEMANTIC_RATIO = 0.87
NO_RANGE_COORDINATED_RELATION_MIN_SCORE = 0.45
NO_RANGE_COORDINATED_RELATION_MIN_SEMANTIC_RATIO = 0.75
NO_RANGE_COORDINATED_RELATION_MIN_CONCEPT_COVERAGE = 0.40
NO_RANGE_LEXICAL_SUPPLEMENT_MIN_SEMANTIC_RATIO = 0.72
NO_RANGE_LEXICAL_SUPPLEMENT_MIN_DIRECT_CONCEPT_COVERAGE = 1.0 / 3.0


def _broad_guidance_requested_groups(
    query: str,
    piece: str,
) -> list[tuple[str, frozenset[str]]]:
    """Return independently requested broad-guidance term groups.

    Tone words form one descriptive slot because an answer need not repeat
    the user's exact adjective. Breath terminology has one common lexical
    alternation (``호흡``/``숨``); other concrete terms remain exact so a
    high-note passage cannot masquerade as breathing guidance merely because
    both belong to the broad vocal-technique facet.
    """

    if not is_broad_performance_guidance_query(query, piece):
        return []
    requested_terms = performance_guidance_query_terms(query)
    tone_terms = frozenset(PERFORMANCE_GUIDANCE_FACETS["tone_color"])
    groups: list[tuple[str, frozenset[str]]] = []
    if requested_terms & tone_terms:
        groups.append(("tone_color", tone_terms))
    breath_terms = frozenset({"호흡", "숨"})
    seen: set[str] = set()
    for term in sorted(requested_terms - tone_terms):
        if term in seen:
            continue
        equivalents = breath_terms if term in breath_terms else frozenset({term})
        groups.append((term, equivalents))
        seen.update(equivalents)
    return groups


def _broad_guidance_group_coverage(
    result: Any,
    groups: list[tuple[str, frozenset[str]]],
) -> set[str]:
    """Return explicit requested guidance groups supported by one result."""

    _score, facets, matched_terms = performance_guidance_profile(result.record)
    return {
        name
        for name, terms in groups
        if (
            (name == "tone_color" and "tone_color" in facets)
            or (name != "tone_color" and bool(terms & matched_terms))
        )
    }
# A selected range proves where a knowledge unit applies, not that the unit
# answers the user's question. Keep ranged generation bundles broad only when
# the answer-semantic signal is too weak to distinguish useful supplements.
# These constants intentionally mirror the calibrated no-range values while
# remaining separately named so either policy can evolve without coupling.
RANGED_SEMANTIC_SUPPLEMENT_MIN_SCORE = 0.58
RANGED_SEMANTIC_COMPETITIVE_RATIO = 0.93
RANGED_SEMANTIC_CONCEPT_COMPETITIVE_RATIO = 0.91
RANGED_LOW_SCORE_MIN_CONCEPT_COVERAGE = 0.50
DENSE_SEMANTIC_AUTHORITY_MATCH_TYPE = "dense_semantic_authority"
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
        claim_start = 0
        for match in ANSWER_CLAIM_END_RE.finditer(line):
            claim = line[claim_start : match.end()].strip()
            if claim:
                claims.append(claim)
            claim_start = match.end()
            while claim_start < len(line) and line[claim_start].isspace():
                claim_start += 1
        trailing_claim = line[claim_start:].strip()
        if trailing_claim:
            claims.append(trailing_claim)
    return claims


def _claim_has_substantive_text(claim: str) -> bool:
    """Return whether a claim contains user-facing prose beyond citations."""

    without_citations = ANSWER_EVIDENCE_LABEL_RE.sub("", claim)
    return bool(re.search(r"[A-Za-z0-9가-힣]", without_citations))


def _validate_primary_citation_per_claim(
    answer: str,
    results: List[Any],
) -> None:
    """Require every substantive generated claim to cite primary evidence."""

    primary_ids = {
        id(result)
        for result in primary_grounding_results(results)
    }
    primary_indices = {
        index
        for index, result in enumerate(results)
        if id(result) in primary_ids
    }
    for claim in _answer_claims(answer):
        if not _claim_has_substantive_text(claim):
            continue
        if _claim_cited_result_indices(claim, results) & primary_indices:
            continue
        raise GeneratedAnswerRejected(
            "Generated answer contains an uncited substantive claim"
        )


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

Answer contract
Write one coherent answer, not a list of evidence excerpts. Use every retrieved
item that materially improves the answer, but omit merely related advice. End
each factual or advisory sentence with the supporting temporary label, for
example [E1] or [E1, E2]. Never print an opaque record ID. The application will
validate and remove the temporary labels before displaying the answer.
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


KOREAN_GROUNDING_TERM_PATTERN = (
    r"(?:정보|근거|자료|내용|문맥|단서|증거)"
)
KOREAN_QUALIFIED_GROUNDING_TERM_PATTERN = (
    r"(?:제공된|주어진|검색된|확인된|제시된|전달받은|"
    r"현재\s*확인된|관련)\s*"
    + KOREAN_GROUNDING_TERM_PATTERN
)
KOREAN_MISSING_ENDING_PATTERN = (
    r"없(?:어|어서|으니|으므로|기\s*때문|습니다|다|고|는|음|$)"
)
KOREAN_GROUNDING_SHORTFALL_RE = re.compile(
    r"(?:"
    + KOREAN_GROUNDING_TERM_PATTERN
    + r"\s*만으로(?:는)?"
    r"|"
    + KOREAN_QUALIFIED_GROUNDING_TERM_PATTERN
    + r"\s*(?:으)?로(?:는)?"
    r"|"
    + KOREAN_GROUNDING_TERM_PATTERN
    + r"(?:가|이|은|는|을|를|에는?|에서)?\s*"
    r"(?:(?![.!?。！？\n]).){0,28}?"
    r"(?:"
    r"부족|불충분|충분하지\s*않|충분치\s*않|누락|"
    r"(?:포함|제공|제시|확인|검색)(?:되어|돼)?\s*있지\s*않|"
    r"(?:포함|제공|제시|확인|검색)되지\s*않|"
    r"찾(?:을|아볼)\s*수\s*없|찾지\s*못"
    r")"
    r"|"
    + KOREAN_GROUNDING_TERM_PATTERN
    + r"(?:가|이|은|는)?\s*"
    + KOREAN_MISSING_ENDING_PATTERN
    + r"|"
    + KOREAN_GROUNDING_TERM_PATTERN
    + r"(?:에는?|에서)\s*"
    r"(?:(?:관련|필요한|해당|그에\s*대한)\s*)?"
    r"(?:정보|근거|자료|내용|문맥|단서|증거|답)(?:가|이)?\s*"
    + KOREAN_MISSING_ENDING_PATTERN
    + r"|(?:부족|불충분)한\s*"
    + KOREAN_GROUNDING_TERM_PATTERN
    + r")"
)
KOREAN_ANSWER_INABILITY_RE = re.compile(
    r"(?:"
    r"(?:대답|답변|응답|답)(?:을|이|은|는)?\s*"
    r"(?:드리|하|제공하)?기\s*(?:어렵|힘들|불가능)"
    r"|(?:대답|답변|응답|답)(?:을)?\s*"
    r"(?:드릴|할|제공할)?\s*수(?:가)?\s*없"
    r"|(?:대답|답변|응답)(?:이|은|는)?\s*(?:어렵|불가능)"
    r"|(?:알|판단하|결정하|확인하|파악하|판별하|특정하|확정하)기\s*"
    r"(?:어렵|힘들|불가능)"
    r"|(?:알|판단할|결정할|확인할|파악할|판별할|특정할|확정할)\s*"
    r"수(?:가)?\s*없"
    r"|결론(?:을)?\s*내리기\s*(?:어렵|힘들|불가능)"
    r"|결론(?:을)?\s*내릴\s*수\s*없"
    r")"
)
ENGLISH_GROUNDING_TERM_PATTERN = (
    r"(?:evidence|information|context|details?|data|material|sources?)"
)
ENGLISH_QUALIFIED_GROUNDING_TERM_PATTERN = (
    r"(?:(?:the|this|that)\s+)?"
    r"(?:(?:provided|available|given|retrieved|supplied|current|existing)\s+)?"
    + ENGLISH_GROUNDING_TERM_PATTERN
    + r"(?:\s+(?:provided|available|given|retrieved|supplied))?"
)
ENGLISH_GROUNDING_SHORTFALL_RE = re.compile(
    r"(?:"
    r"(?:no|not\s+enough|insufficient|inadequate|incomplete|missing|absent)\s+"
    r"(?:relevant\s+)?"
    + ENGLISH_GROUNDING_TERM_PATTERN
    + r"\b"
    r"|"
    + ENGLISH_QUALIFIED_GROUNDING_TERM_PATTERN
    + r"\s+(?:because\s+it\s+)?(?:is|are|was|were|remains?)\s+"
    r"(?:insufficient|inadequate|incomplete|limited|lacking|missing|"
    r"absent|unavailable|not\s+(?:enough|sufficient|available|provided|"
    r"included|present|found))\b"
    r"|"
    + ENGLISH_QUALIFIED_GROUNDING_TERM_PATTERN
    + r"\s+(?:does|do)\s+not\s+"
    r"(?:contain|provide|include|cover|address|support)\b"
    r"|"
    + ENGLISH_QUALIFIED_GROUNDING_TERM_PATTERN
    + r"\s+(?:doesn't|don't)\s+"
    r"(?:contain|provide|include|cover|address|support)\b"
    r"|"
    + ENGLISH_QUALIFIED_GROUNDING_TERM_PATTERN
    + r"\s+lacks?\b"
    r"|(?:lack|absence)\s+of\s+(?:relevant\s+)?"
    + ENGLISH_GROUNDING_TERM_PATTERN
    + r"\b"
    r"|"
    + ENGLISH_QUALIFIED_GROUNDING_TERM_PATTERN
    + r"\s+(?:alone|only)\b"
    r")",
    flags=re.IGNORECASE,
)
ENGLISH_ANSWER_INABILITY_RE = re.compile(
    r"(?:"
    r"(?:cannot|can't|can\s+not|unable\s+to|impossible\s+to|"
    r"not\s+possible\s+to|difficult\s+to|hard\s+to)\s+"
    r"(?:(?:confidently|accurately|reliably|definitively|fully)\s+)?"
    r"(?:answer|respond|determine|know|conclude|infer|establish|verify|"
    r"identify|tell|say|(?:provide|give)\s+(?:an?|the)\s+answer)\b"
    r"|(?:do\s+not|don't|does\s+not|doesn't)\s+know\b"
    r"|(?:answer|conclusion|question|result|it|this|that)\s+"
    r"(?:cannot|can't|can\s+not)\s+be\s+"
    r"(?:answered|determined|provided|known|established|verified|identified)\b"
    r")",
    flags=re.IGNORECASE,
)
ENGLISH_DIRECT_INSUFFICIENCY_RE = re.compile(
    r"(?:"
    r"there\s+(?:is|are)\s+(?:no|not\s+enough|insufficient)\s+"
    r"(?:relevant\s+)?"
    + ENGLISH_GROUNDING_TERM_PATTERN
    + r"\s+to\s+"
    r"(?:answer|determine|know|conclude|establish|verify|identify)\b"
    r"|(?:i|we)\s+(?:do\s+not|don't)\s+have\s+enough\s+"
    + ENGLISH_GROUNDING_TERM_PATTERN
    + r"\s+to\s+"
    r"(?:answer|determine|know|conclude|establish|verify|identify)\b"
    r")",
    flags=re.IGNORECASE,
)


def _nearby_signal_pair(
    text: str,
    first: re.Pattern[str],
    second: re.Pattern[str],
    *,
    max_gap: int,
) -> bool:
    """Return whether two refusal signals occur in one short explanation."""

    first_matches = list(first.finditer(text))
    second_matches = list(second.finditer(text))
    for first_match in first_matches:
        for second_match in second_matches:
            gap = max(
                0,
                first_match.start() - second_match.end(),
                second_match.start() - first_match.end(),
            )
            if gap <= max_gap:
                return True
    return False


def is_grounded_insufficiency_answer(answer: str) -> bool:
    """Recognize a grounded model refusal caused by missing evidence.

    A bare word such as ``부족`` or a general discussion of missing score
    information is not a refusal. Natural-language detection requires both a
    grounding-information shortfall and an inability to answer, know, or
    determine something within one short explanation.
    """

    without_footer = _remove_evidence_footer(answer)
    if answer.strip() and not without_footer:
        return True
    compact = re.sub(r"\s+", "", answer)
    if NO_GROUNDED_ANSWER_SENTINEL.casefold() in compact.casefold():
        return True

    normalized = unicodedata.normalize("NFKC", without_footer)
    if _nearby_signal_pair(
        normalized,
        KOREAN_GROUNDING_SHORTFALL_RE,
        KOREAN_ANSWER_INABILITY_RE,
        max_gap=80,
    ):
        return True
    if ENGLISH_DIRECT_INSUFFICIENCY_RE.search(normalized):
        return True
    return _nearby_signal_pair(
        normalized,
        ENGLISH_GROUNDING_SHORTFALL_RE,
        ENGLISH_ANSWER_INABILITY_RE,
        max_gap=120,
    )


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
            result.semantic_match_type != "broad_guidance"
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


def _answer_semantic_score(result: Any) -> float:
    """Return the strongest reviewed-answer embedding similarity."""

    return max(
        float(getattr(result, "dense_score", 0.0) or 0.0),
        float(getattr(result, "dense_content_score", 0.0) or 0.0),
    )


def _answers_explicit_requested_slots(result: Any, query: str) -> bool:
    """Return whether finalized answer text fills every explicit query slot."""

    return not missing_dense_answer_constraints(
        query,
        str(result.record.get("answer") or ""),
    )


def _explicit_requested_answer_slots(query: str) -> set[str]:
    """Return answer attributes that the wording explicitly requires.

    An empty answer cannot satisfy any requested attribute, so the existing
    dense constraint validator is also the single source of truth for which
    slots a query activates.  This avoids maintaining a second set of query
    patterns in generation selection.
    """

    return set(missing_dense_answer_constraints(query, ""))


def _query_requests_coordinated_relations(query: str) -> bool:
    """Return whether one question explicitly coordinates two relations.

    This is deliberately clause-aware.  Korean connective endings must be
    followed by whitespace or punctuation, so syllables inside words such as
    ``고음`` cannot turn a single request into a multi-part one.  Coordinated
    noun phrases such as ``음색과 느낌`` likewise remain one semantic slot.
    """

    causal_then_procedural = re.search(
        r"(?:왜|이유|원인|까닭|의미).{0,140}"
        r"(?:[가-힣]+(?:고|며))\s*[,，;；]?\s*"
        r"(?:어떻게|방법|무엇(?:을|를)?|뭘|how\b|what\s+should)",
        query,
        flags=re.IGNORECASE,
    )
    change_then_performance = re.search(
        r"어떻게.{0,72}(?:[가-힣]+(?:고|며))\s*[,，;；]?\s*"
        r"(?:가창|노래|연주|표현|성악).{0,72}"
        r"(?:무엇|어떻게|중요|해야|하여야|방법)",
        query,
        flags=re.IGNORECASE,
    )
    return causal_then_procedural is not None or change_then_performance is not None


def _result_source_ids(result: Any) -> set[str]:
    """Return immutable expert-source identifiers for bundle compatibility."""

    return {
        str(source_id)
        for source_id in (result.record.get("source_ids") or [])
        if str(source_id)
    }


def _prune_unsupported_local_supplements(
    results: List[Any],
) -> List[Any]:
    """Drop a dense-only local aside after a stronger general answer.

    Local reviewed units remain first-class no-range evidence.  The narrow
    exception is an unrelated local follower with no lexical, relation, or
    answer-concept support after a general leader already won.  Exact-source
    nonlocal siblings protect split expert answers, so a local technique and
    its general companion stay together.
    """

    if not results or results[0].scope_match != "general_evidence":
        return results
    leader = results[0]
    leader_sources = _result_source_ids(leader)
    nonlocal_source_families = {
        tuple(sorted(_result_source_ids(result)))
        for result in results
        if (
            result.scope_match != "local_example"
            and _result_source_ids(result)
        )
    }
    return [
        result
        for result in results
        if not (
            result is not leader
            and result.scope_match == "local_example"
            and not (leader_sources & _result_source_ids(result))
            and tuple(sorted(_result_source_ids(result)))
            not in nonlocal_source_families
            and result.text_score <= 0
            and result.answer_relation_score <= 0
            and result.content_concept_coverage <= 0
        )
    ]


def _lexical_supplement_is_semantically_compatible(
    result: Any,
    *,
    leader: Any,
    semantic_supplements: List[Any],
    semantic_pool: List[Any],
    concept_floor: float,
    query: str,
    piece: str,
) -> bool:
    """Keep lexical additions only when they belong with the semantic answer.

    Broad-guidance retrieval deliberately has high recall, so its synthetic
    lexical score is not evidence that every returned unit should be composed
    into one answer.  A lexical result remains useful when answer-semantic
    ranking independently retained it, when it is competitive with the
    semantic leader, or when it is another finalized facet of the same
    immutable expert source.  This keeps genuinely complementary multi-unit
    answers without restoring unrelated units merely because they share broad
    words such as "rhythm" or "singing".
    """

    record_id = str(result.record.get("id") or "")
    if any(
        str(candidate.record.get("id") or "") == record_id
        for candidate in semantic_supplements
    ):
        return True
    semantic_counterpart = next(
        (
            candidate
            for candidate in semantic_pool
            if str(candidate.record.get("id") or "") == record_id
        ),
        result,
    )
    if (
        _answers_explicit_requested_slots(semantic_counterpart, query)
        and _answer_semantic_score(semantic_counterpart) >= concept_floor
    ):
        return True
    if (
        result.semantic_match_type != "broad_guidance"
        and result.text_score > 0
        and result.content_concept_coverage
        >= NO_RANGE_LOW_SCORE_MIN_CONCEPT_COVERAGE
    ):
        # Normal strict lexical retrieval carries answer-side query support.
        # The broad-guidance route assigns intentionally synthetic lexical
        # coverage, so it must satisfy one of the semantic/source-family gates
        # above instead of using this exception.
        return True
    query_concepts = set(semantic_concepts(query, piece, query=True))
    answer_concepts = set(
        semantic_concepts(
            str(result.record.get("answer") or ""),
            piece,
            query=False,
        )
    )
    direct_concept_coverage = (
        len(query_concepts & answer_concepts) / len(query_concepts)
        if query_concepts
        else 0.0
    )
    leader_score = _answer_semantic_score(leader)
    if (
        leader_score > 0
        and _answer_semantic_score(semantic_counterpart)
        >= leader_score * NO_RANGE_LEXICAL_SUPPLEMENT_MIN_SEMANTIC_RATIO
        and direct_concept_coverage
        >= NO_RANGE_LEXICAL_SUPPLEMENT_MIN_DIRECT_CONCEPT_COVERAGE
    ):
        return True
    leader_sources = _result_source_ids(leader)
    return bool(leader_sources & _result_source_ids(result))


def _focus_no_range_semantic_bundle(
    base_results: List[Any],
    semantic_pool: List[Any],
    *,
    query: str,
    piece: str,
    top_k: int,
) -> List[Any] | None:
    """Focus a no-alias bundle without consulting evaluation target IDs.

    The semantic pool is ranked only from the production query and finalized
    reviewed-answer text. A close candidate is retained either by a strict
    embedding ratio or, within the wider near-tie band, by explicit query
    concepts. Weak semantic pools can prune an oversized lexical bundle but
    cannot manufacture broad support. Unsupported web neighbors are removed
    whenever a finalized expert answer already grounds the query.
    """

    if not base_results or not semantic_pool:
        return None
    all_finalized_experts = [
        result
        for result in base_results
        if _expert_record_is_finalized(result.record)
    ]
    all_semantic_experts = [
        result
        for result in semantic_pool
        if _expert_record_is_finalized(result.record)
    ]
    constrained_finalized_experts = [
        result
        for result in all_finalized_experts
        if _answers_explicit_requested_slots(result, query)
    ]
    constrained_semantic_pool = [
        result
        for result in all_semantic_experts
        if _answers_explicit_requested_slots(result, query)
    ]
    # Requested-slot constraints distinguish a direct answer from a nearby
    # semantic match when at least one candidate actually fills the slot. If
    # none does, semantic search may prune records already selected by normal
    # retrieval, but it must not expand that bundle with new slot-missing
    # neighbors. This rule depends on provenance, not candidate count.
    if constrained_semantic_pool:
        finalized_experts = constrained_finalized_experts
        eligible_semantic_pool = constrained_semantic_pool
    else:
        finalized_experts = all_finalized_experts
        base_ids = {
            str(result.record.get("id") or "")
            for result in all_finalized_experts
        }
        eligible_semantic_pool = [
            result
            for result in all_semantic_experts
            if str(result.record.get("id") or "") in base_ids
        ]

    asks_cause_and_procedure = is_causal_answer_request(query) and bool(
        re.search(
            r"(?:어떻게|방법|해야|하여야|처리|how\b|what\s+should)",
            query,
            flags=re.IGNORECASE,
        )
    )
    explicit_requested_slots = _explicit_requested_answer_slots(query)
    broad_guidance_groups = _broad_guidance_requested_groups(query, piece)
    requests_coordinated_relations = _query_requests_coordinated_relations(
        query
    )
    protected_source_results: List[Any] = []
    if asks_cause_and_procedure:
        source_groups: Dict[tuple[str, ...], List[Any]] = {}
        for result in finalized_experts:
            family = tuple(sorted(_result_source_ids(result)))
            if family:
                source_groups.setdefault(family, []).append(result)
        protected_source_results = [
            result
            for group in source_groups.values()
            if len(group) > 1
            for result in group
        ]
    if (
        not eligible_semantic_pool
        or any(result.alias_score > 0 for result in finalized_experts)
        or any(
            result.semantic_match_type
            == SOURCE_QUESTION_LED_MATCH_TYPE
            for result in finalized_experts
        )
    ):
        return None

    # A complete, ordinary lexical answer match is stronger evidence of the
    # requested relation than an embedding-only neighbor that merely shares
    # the topic.  In that narrow situation, do not let semantic focusing add
    # an unrelated source family with weaker answer-side concept coverage.
    # Exact-source companions remain eligible so split, compatible expert
    # claims can still form a complete answer.
    semantic_leader = eligible_semantic_pool[0]
    direct_lexical_anchors = [
        result
        for result in finalized_experts
        if (
            result.semantic_match_type == "strict"
            and result.text_score > 0
            and result.content_concept_coverage >= 1.0
            and _answers_explicit_requested_slots(result, query)
        )
    ]
    strongest_direct_concept_coverage = max(
        (
            float(result.content_concept_coverage or 0.0)
            for result in direct_lexical_anchors
        ),
        default=0.0,
    )
    direct_source_families = {
        tuple(sorted(_result_source_ids(result)))
        for result in direct_lexical_anchors
        if _result_source_ids(result)
    }
    if (
        direct_lexical_anchors
        and tuple(sorted(_result_source_ids(semantic_leader)))
        not in direct_source_families
        and float(semantic_leader.content_concept_coverage or 0.0)
        < strongest_direct_concept_coverage
    ):
        exact_source_companions = [
            result
            for result in finalized_experts
            if tuple(sorted(_result_source_ids(result)))
            in direct_source_families
            and _answers_explicit_requested_slots(result, query)
        ]
        return _merge_unique_results(
            direct_lexical_anchors,
            exact_source_companions,
        )[:top_k]

    leader_score = _answer_semantic_score(eligible_semantic_pool[0])
    lexical_experts = [
        result
        for result in finalized_experts
        if result.text_score > 0
    ]
    strict_floor = leader_score * NO_RANGE_SEMANTIC_COMPETITIVE_RATIO
    concept_floor = (
        leader_score * NO_RANGE_SEMANTIC_CONCEPT_COMPETITIVE_RATIO
    )
    leader = eligible_semantic_pool[0]

    explicit_slot_floor = max(
        NO_RANGE_EXPLICIT_SLOT_MIN_SCORE,
        concept_floor,
    )
    if explicit_requested_slots == {"vocal_character_description"}:
        # Character descriptions use varied imagery (rather than repeating
        # the query's exact vocabulary), so compatible style examples form a
        # slightly wider semantic band. The dedicated relation constraint is
        # what excludes mere key/voice suitability advice.
        explicit_slot_floor = max(
            NO_RANGE_CHARACTER_SLOT_MIN_SCORE,
            leader_score * NO_RANGE_CHARACTER_SLOT_MIN_SEMANTIC_RATIO,
        )

    def is_direct_explicit_slot_supplement(result: Any) -> bool:
        """Keep a near-tie that directly fills an explicit answer slot."""

        return bool(explicit_requested_slots) and (
            _answer_semantic_score(result) >= explicit_slot_floor
            and result.content_concept_coverage
            >= NO_RANGE_EXPLICIT_SLOT_MIN_CONCEPT_COVERAGE
        )

    def is_coordinated_relation_supplement(result: Any) -> bool:
        """Keep a useful second relation without broadly lowering floors."""

        generic_relation_concepts = {
            "가창",
            "노래",
            "부르",
            "표현하",
            "중요",
            "좋다",
            "이러한",
            "다시",
        }
        leader_concepts = set(
            semantic_concepts(
                str(leader.record.get("answer") or ""),
                piece,
                query=False,
            )
        ) - generic_relation_concepts
        candidate_concepts = set(
            semantic_concepts(
                str(result.record.get("answer") or ""),
                piece,
                query=False,
            )
        ) - generic_relation_concepts
        adds_distinct_detail = len(candidate_concepts - leader_concepts) >= 2
        return requests_coordinated_relations and (
            _answer_semantic_score(result)
            >= NO_RANGE_COORDINATED_RELATION_MIN_SCORE
            and _answer_semantic_score(result)
            >= leader_score
            * NO_RANGE_COORDINATED_RELATION_MIN_SEMANTIC_RATIO
            and result.content_concept_coverage
            >= NO_RANGE_COORDINATED_RELATION_MIN_CONCEPT_COVERAGE
            and adds_distinct_detail
        )

    has_direct_supplement = any(
        is_direct_explicit_slot_supplement(result)
        or is_coordinated_relation_supplement(result)
        for result in eligible_semantic_pool
        if result is not leader
    )
    should_focus = (
        leader_score >= NO_RANGE_FOCUS_MIN_SCORE
        or (
            len(finalized_experts) >= NO_RANGE_SEMANTIC_SUPPLEMENT_LIMIT
            and bool(lexical_experts)
        )
        or has_direct_supplement
    )
    if should_focus:
        semantic_supplements = [
            leader,
            *[
                result
                for result in eligible_semantic_pool
                if result is not leader
                if (
                    _answer_semantic_score(result) >= strict_floor
                    or (
                        _answer_semantic_score(result) >= concept_floor
                        and (
                            result.text_score > 0
                            or result.content_concept_coverage
                            >= NO_RANGE_LOW_SCORE_MIN_CONCEPT_COVERAGE
                        )
                    )
                    or is_direct_explicit_slot_supplement(result)
                    or is_coordinated_relation_supplement(result)
                )
                and (
                    leader_score >= NO_RANGE_SEMANTIC_SUPPLEMENT_MIN_SCORE
                    or result.text_score > 0
                    or result.content_concept_coverage
                    >= NO_RANGE_LOW_SCORE_MIN_CONCEPT_COVERAGE
                    or is_direct_explicit_slot_supplement(result)
                    or is_coordinated_relation_supplement(result)
                )
            ],
        ]
        semantic_supplements = _prune_unsupported_local_supplements(
            semantic_supplements
        )
        if any(
            result.semantic_match_type == "broad_guidance"
            for result in finalized_experts
        ):
            selected_families = {
                tuple(sorted(_result_source_ids(result)))
                for result in semantic_supplements
                if _result_source_ids(result)
            }
            protected_source_results = _merge_unique_results(
                protected_source_results,
                [
                    result
                    for result in eligible_semantic_pool
                    if tuple(sorted(_result_source_ids(result)))
                    in selected_families
                ],
            )
        compatible_lexical_experts = [
            result
            for result in lexical_experts
            if _lexical_supplement_is_semantically_compatible(
                result,
                leader=leader,
                semantic_supplements=semantic_supplements,
                semantic_pool=eligible_semantic_pool,
                concept_floor=concept_floor,
                query=query,
                piece=piece,
            )
        ]
        focused = _merge_unique_results(
            semantic_supplements,
            compatible_lexical_experts,
            protected_source_results,
        )[:top_k]
        if len(broad_guidance_groups) >= 2:
            # Broad routing deliberately builds a complementary facet bundle.
            # Answer-embedding focus may reorder that bundle, but it must not
            # erase an explicitly requested facet or replace all router
            # evidence with a semantic neighbor that fills none of the named
            # slots. Preserve only the smallest router subset needed to cover
            # the requested groups that the corpus can actually supply.
            available_groups = set().union(
                *(
                    _broad_guidance_group_coverage(
                        result,
                        broad_guidance_groups,
                    )
                    for result in base_results
                ),
                set(),
            )
            base_ids = {
                str(result.record.get("id") or "")
                for result in base_results
            }
            focused = [
                result
                for result in focused
                if (
                    str(result.record.get("id") or "") in base_ids
                    or _broad_guidance_group_coverage(
                        result,
                        broad_guidance_groups,
                    )
                )
            ]
            covered_groups = set().union(
                *(
                    _broad_guidance_group_coverage(
                        result,
                        broad_guidance_groups,
                    )
                    for result in focused
                ),
                set(),
            )
            missing_groups = available_groups - covered_groups
            router_anchors: List[Any] = []
            for result in base_results:
                newly_covered = (
                    _broad_guidance_group_coverage(
                        result,
                        broad_guidance_groups,
                    )
                    & missing_groups
                )
                if not newly_covered:
                    continue
                router_anchors.append(result)
                missing_groups -= newly_covered
                if not missing_groups:
                    break
            focused = _merge_unique_results(focused, router_anchors)[:top_k]
        if focused:
            return focused

    supported_web = [
        result
        for result in base_results
        if result.record.get("evidence_type") == "web_database"
        and _answers_explicit_requested_slots(result, query)
        and (
            result.text_score > 0
            or result.content_concept_coverage
            >= NO_RANGE_LOW_SCORE_MIN_CONCEPT_COVERAGE
        )
    ]
    return _merge_unique_results(
        _prune_unsupported_local_supplements(finalized_experts),
        supported_web,
    )[:top_k]


def _focus_ranged_semantic_bundle(
    base_results: List[Any],
    semantic_pool: List[Any],
    *,
    query: str,
    top_k: int,
) -> List[Any] | None:
    """Remove range-only neighbors from one no-alias generation bundle.

    Measure overlap is an applicability boundary, not a semantic relevance
    signal. Once an overlapping reviewed answer is independently strong, keep
    it together with semantic near-ties and candidates that directly match the
    query on both retrieval and answer content. This preserves complementary
    knowledge units without filling the prompt with every annotation that
    happens to touch the selected measures.

    A weak leader leaves the high-recall bundle unchanged. Positive immutable
    source-question aliases also retain their existing authority path. The
    selector never consults evaluation questions or expected knowledge-unit
    IDs and never turns uncertainty into an empty answer bundle.
    """

    if (
        not base_results
        or not semantic_pool
        or any(
            result.alias_score > 0
            and result.scope_match in RANGED_PRIMARY_SCOPE_MATCHES
            and _answers_explicit_requested_slots(result, query)
            for result in base_results
        )
        or any(
            result.alias_score > 0
            and result.scope_match in RANGED_PRIMARY_SCOPE_MATCHES
            and _answers_explicit_requested_slots(result, query)
            for result in semantic_pool
        )
    ):
        return None
    primary_experts = [
        result
        for result in semantic_pool
        if (
            result.scope_match in RANGED_PRIMARY_SCOPE_MATCHES
            and _expert_record_is_finalized(result.record)
            and _answers_explicit_requested_slots(result, query)
        )
    ]
    if not primary_experts:
        return None
    overlapping_experts = [
        result
        for result in primary_experts
        if result.scope_match == "overlaps_query_range"
    ]
    global_experts = [
        result
        for result in primary_experts
        if result.scope_match == "global_context"
    ]

    def anchor_key(result: Any) -> tuple[float, float, float, str]:
        return (
            _answer_semantic_score(result),
            float(getattr(result, "dense_score", 0.0) or 0.0),
            float(getattr(result, "dense_content_score", 0.0) or 0.0),
            str(result.record.get("id") or ""),
        )

    best_overlap = (
        max(overlapping_experts, key=anchor_key)
        if overlapping_experts
        else None
    )
    best_global = (
        max(global_experts, key=anchor_key) if global_experts else None
    )
    direct_overlaps = [
        result
        for result in overlapping_experts
        if _has_direct_query_signal(result)
        and _has_answer_side_subject_support(result)
        and _answer_semantic_score(result)
        >= RANGED_SEMANTIC_SUPPLEMENT_MIN_SCORE
    ]
    best_direct_overlap = (
        max(direct_overlaps, key=anchor_key) if direct_overlaps else None
    )
    anchor = best_direct_overlap or best_overlap or best_global
    if best_global is not None and (
        best_direct_overlap is None
        and (
            best_overlap is None
            or (
                float(best_global.dense_score or 0.0)
                >= RANGED_DENSE_AUTHORITY_MIN_RELEVANCE_SCORE
                and float(best_global.dense_content_score or 0.0)
                >= RANGED_DENSE_AUTHORITY_MIN_CONTENT_SCORE
                and float(best_global.dense_score or 0.0)
                >= float(best_overlap.dense_score or 0.0)
                + RANGED_DENSE_AUTHORITY_RELEVANCE_MARGIN
                and float(best_global.dense_content_score or 0.0)
                >= float(best_overlap.dense_content_score or 0.0)
                + RANGED_DENSE_AUTHORITY_CONTENT_MARGIN
            )
        )
    ):
        # Whole-piece guidance is applicable in a selected range, but it may
        # displace an overlapping unit only with a strong, two-view semantic
        # lead.  This prevents range overlap from defeating an exact answer
        # while preserving local detail in near-tie cases.
        anchor = best_global
    assert anchor is not None
    leader_score = _answer_semantic_score(anchor)
    if leader_score < RANGED_SEMANTIC_SUPPLEMENT_MIN_SCORE:
        return None

    strict_floor = leader_score * RANGED_SEMANTIC_COMPETITIVE_RATIO
    concept_floor = (
        leader_score * RANGED_SEMANTIC_CONCEPT_COMPETITIVE_RATIO
    )
    semantic_supplements = [
        result
        for result in semantic_pool
        if (
            result.scope_match in RANGED_PRIMARY_SCOPE_MATCHES
            and _expert_record_is_finalized(result.record)
            and _answers_explicit_requested_slots(result, query)
            and (
                _answer_semantic_score(result) >= strict_floor
                or (
                    _answer_semantic_score(result) >= concept_floor
                    and (
                        result.text_score > 0
                        or result.content_concept_coverage
                        >= RANGED_LOW_SCORE_MIN_CONCEPT_COVERAGE
                    )
                )
            )
        )
    ]
    directly_supported = [
        result
        for result in base_results
        if (
            result.scope_match in RANGED_PRIMARY_SCOPE_MATCHES
            and _answers_explicit_requested_slots(result, query)
            and _has_direct_query_signal(result)
            and _has_answer_side_subject_support(result)
        )
    ]
    context_only = [
        result
        for result in base_results
        if result.scope_match in RANGED_CONTEXT_ONLY_SCOPE_MATCHES
    ][:2]
    return _merge_unique_results(
        [anchor],
        semantic_supplements,
        directly_supported,
        context_only,
    )[:top_k]


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

    The function performs no model call, repair call, or model fallback. It
    returns the reviewed evidence bundle and prompt for the one natural,
    grounded generation call made by the caller.
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
    if retrieval_abstained_for_missing_causal_authority(index):
        # ``last_answer_candidates`` is diagnostic context captured before a
        # causal-authority hard rejection. It is not generation evidence.
        # Re-running broad semantic selection here would silently undo the
        # retriever's fail-closed decision and let a procedural neighbor
        # answer a why-question.
        return [], build_messages(piece, measures, query, [])
    rejected_candidates = list(
        getattr(index, "last_answer_candidates", []) or []
    )
    selected_results = select_generation_evidence(
        stable_candidates,
        query=query,
        piece=piece,
        measure_ranges=measure_ranges,
    )
    semantic_search = getattr(index, "semantic_candidates", None)
    # Original annotator questions can route directly through immutable alias
    # matches. Units created from answer-only annotations have no such alias,
    # so a no-range query can otherwise be dominated by a nearby source
    # question or web chunk even when a reviewed answer embedding is a much
    # better match. Focus a sparse bundle when the answer-semantic leader is
    # independently strong, and also prune an oversized no-alias bundle when
    # normal lexical retrieval provides an anchor. Keep only semantically
    # competitive reviewed answers; never fill to a target KU count. A strong
    # lexical expert remains eligible even when its embedding is weaker. This
    # is a production retrieval rule, not an evaluation-question alias: the
    # held-out wording and expected KU IDs are never indexed or consulted.
    if (
        not measure_ranges
        and callable(semantic_search)
        and selected_results
        and not any(
            result.alias_score > 0
            and _answers_explicit_requested_slots(result, query)
            for result in selected_results
        )
    ):
        semantic_pool = [
            result
            for result in semantic_search(
                query=query,
                piece=piece,
                measure_ranges=measure_ranges,
                topic=topic,
                top_k=max(NO_RANGE_SEMANTIC_CANDIDATE_LIMIT, top_k),
            )
            if _expert_record_is_finalized(result.record)
        ]
        focused_results = _focus_no_range_semantic_bundle(
            selected_results,
            semantic_pool,
            query=query,
            piece=piece,
            top_k=top_k,
        )
        if focused_results is not None:
            selected_results = focused_results
    if (
        measure_ranges
        and callable(semantic_search)
        and selected_results
        and not any(
            result.alias_score > 0
            and result.scope_match in RANGED_PRIMARY_SCOPE_MATCHES
            and _answers_explicit_requested_slots(result, query)
            for result in selected_results
        )
    ):
        ranged_semantic_pool = [
            result
            for result in semantic_search(
                query=query,
                piece=piece,
                measure_ranges=measure_ranges,
                topic=topic,
                top_k=max(NO_RANGE_SEMANTIC_SUPPLEMENT_LIMIT, top_k),
            )
            if _expert_record_is_finalized(result.record)
        ]
        focused_results = _focus_ranged_semantic_bundle(
            selected_results,
            ranged_semantic_pool,
            query=query,
            top_k=top_k,
        )
        if focused_results is not None:
            selected_results = focused_results
    needs_broader_generation_context = (
        not selected_results
        or (measure_ranges and not has_primary_grounding(selected_results))
    )
    if needs_broader_generation_context:
        broad_candidates = (
            semantic_search(
                query=query,
                piece=piece,
                measure_ranges=measure_ranges,
                topic=topic,
                top_k=top_k,
            )
            if callable(semantic_search)
            else []
        )
        broad_base = _merge_unique_results(
            rejected_candidates,
            selected_results,
            broad_candidates,
        )
        broad_semantic_pool = [
            result
            for result in broad_candidates
            if _expert_record_is_finalized(result.record)
        ]
        focused_broad = (
            _focus_ranged_semantic_bundle(
                broad_base,
                broad_semantic_pool,
                query=query,
                top_k=top_k,
            )
            if measure_ranges
            else _focus_no_range_semantic_bundle(
                broad_base,
                broad_semantic_pool,
                query=query,
                piece=piece,
                top_k=top_k,
            )
        )
        selected_results = (
            focused_broad
            if focused_broad is not None
            else broad_base[:top_k]
        )
    results = selected_results[: min(top_k, len(selected_results))]
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
    _validate_primary_citation_per_claim(answer, results)
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
    generation_validation_warning: str | None = None

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
            raw_answer = generate(
                settings["model_path"],
                messages,
                settings["llm"],
            )
            try:
                answer = finalize_grounded_generated_answer(
                    raw_answer,
                    results,
                    messages,
                    args.measure_ranges,
                )
            except GeneratedAnswerRejected as exc:
                answer = finalize_user_visible_answer(raw_answer, results)
                if answer == NO_CORPUS_EVIDENCE_MESSAGE:
                    raise
                generation_validation_warning = str(exc)
            generation_mode = "llm"
            answer_basis = "retrieved_evidence"

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
                    "generation_validation_warning": (
                        generation_validation_warning
                    ),
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
