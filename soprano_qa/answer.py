#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ask the local measure-aware Soprano QA system."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import unicodedata
from typing import Any, Dict, List


from soprano_qa.corpus import (
    CORPUS_SCHEMA_VERSION,
    SONG_ORDER,
    build_corpus,
    corpus_input_fingerprint,
    file_sha256,
    is_web_record,
)
from soprano_qa.dense import (
    SearchIndex,
    SOURCE_QUESTION_LED_MATCH_TYPE,
    SOURCE_QUESTION_LED_MIN_CONTENT_SCORE,
    SOURCE_QUESTION_LED_MIN_OTHER_SOURCE_MARGIN,
    SOURCE_QUESTION_LED_MIN_RELEVANCE_SCORE,
    is_causal_answer_request,
    missing_dense_answer_constraints,
    retrieval_abstained_for_missing_causal_authority,
)
from soprano_qa.retrieval import (
    PERFORMANCE_GUIDANCE_FACETS,
    format_measure_range,
    expert_record_is_runtime_final,
    is_broad_performance_guidance_query,
    parse_measure_ranges,
    performance_guidance_profile,
    performance_guidance_query_terms,
    scope_evidence_role,
    semantic_concepts,
)
from soprano_qa.settings import load_settings
from soprano_qa.structured_answer import PreservedSourceAnswer
from soprano_qa.security import screen_evidence_record, screen_text


NO_CORPUS_EVIDENCE_MESSAGE = "현재 확인된 정보만으로는 이 질문에 정확히 답변하기 어렵습니다."

NO_CORPUS_EVIDENCE_MESSAGE_EN = (
    "I could not find grounded evidence in the reviewed materials to "
    "answer this question."
)
# Above this share of Hangul among Hangul and Latin letters, an English
# answer is treated as a Korean draft rather than English prose that quotes
# a Korean lyric word.
ENGLISH_ANSWER_MAX_HANGUL_RATIO = 0.3


def no_corpus_evidence_message(language: str = "ko") -> str:
    """Return the user-visible no-evidence message for one answer language."""

    if language == "ko":
        return NO_CORPUS_EVIDENCE_MESSAGE
    if language == "en":
        return NO_CORPUS_EVIDENCE_MESSAGE_EN
    raise ValueError("Unsupported answer language: %r" % (language,))


def answer_is_mostly_hangul(answer: str) -> bool:
    """Return whether an intended English answer is predominantly Korean."""

    hangul = len(re.findall(r"[가-힣ᄀ-ᇿㄱ-ㆎ]", answer))
    latin = len(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]", answer))
    if latin == 0:
        return True
    return hangul / (hangul + latin) > ENGLISH_ANSWER_MAX_HANGUL_RATIO
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
    r"[.!?;。！？；][\"'”’」』】)]*"
    r"|[\"'”’」』】)]+[.!?;。！？；]"
    r"|[\"'”’」』】)]*[ \t]*(?:\n|$)"
    r"))"
)
KOREAN_SENTENCE_FINAL_WORD_RE = re.compile(
    r"(?P<word>[가-힣]+)"
    r"(?=(?:"
    r"[.!?;。！？；][\"'”’」』】)]*"
    r"|[\"'”’」』】)]+[.!?;。！？；]"
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

    # These two common polite endings have unambiguous formal equivalents.
    # Avoid general stemming of -아요/-어요, whose irregular verbs can change
    # meaning or become ungrammatical under a mechanical replacement.
    answer = re.sub(
        r"(있|없)어요(?=[.!?。！？]|\s*$)", r"\1습니다", answer,
        flags=re.MULTILINE,
    )
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
    *,
    language: str = "ko",
) -> str:
    """Apply the invariant for prose returned to a user.

    Korean answers are normalized to and validated as 합니다체. English
    answers skip the Korean register pass and are instead rejected when the
    prose is predominantly Hangul. Both share the artifact checks.
    """

    if isinstance(answer, PreservedSourceAnswer):
        # Only the identity-checked quotation path creates this internal type.
        # Rewriting a quoted author's register would stop preserving the
        # source. Generated additions were already finalized independently.
        if language != "ko" or not answer.strip() or has_forbidden_user_visible_artifact(answer, results):
            return no_corpus_evidence_message(language)
        return PreservedSourceAnswer(str(answer).strip())
    cleaned = strip_user_visible_evidence_references(answer, results)
    # Corpus excerpts and generated prose can contain duplicated horizontal
    # whitespace.  Keep paragraph boundaries intact, but never expose those
    # source-formatting artifacts in the final answer.
    cleaned = re.sub(r"[^\S\r\n]+", " ", cleaned)
    if language == "en":
        finalized = cleaned.strip()
        if (
            not finalized
            or answer_is_mostly_hangul(finalized)
            or has_forbidden_user_visible_artifact(finalized, results)
        ):
            return NO_CORPUS_EVIDENCE_MESSAGE_EN
        return finalized
    if language != "ko":
        raise ValueError("Unsupported answer language: %r" % (language,))
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


def _expert_record_is_finalized(record: Dict[str, Any]) -> bool:
    """Return whether an expert record obeys the release-only runtime schema."""

    return expert_record_is_runtime_final(record)


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
                and is_web_record(results[0].record)
                and _has_direct_query_signal(results[0])
            ):
                # A direct web result must not carry lower, merely related
                # expert records into generation. Otherwise the model could
                # ignore ranking and turn optional biography/background into
                # the answer to a date, rights, or catalog question.
                return [
                    result
                    for result in results
                    if is_web_record(result.record)
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


def sanitize_generation_results(results: List[Any]) -> List[Any]:
    """Keep safe source bodies when only optional category metadata was unsafe."""
    cleaned = []
    for result in results:
        record = screen_evidence_record(result.record)
        if record is not result.record:
            result = copy.copy(result)
            result.record = record
        cleaned.append(result)
    return cleaned


def generation_evidence_payload(
    results: List[Any], measure_ranges: List[List[int]],
) -> List[Dict[str, Any]]:
    """Preserve per-record structure; source strings cannot forge new fields.

    Select the same curated claims and scope as the legacy display formatter,
    while keeping every source value inside its own JSON field. Raw source
    annotations and editorial history remain excluded.
    """
    evidence = []
    primary = primary_grounding_results(results)
    for idx, result in enumerate(results, start=1):
        record = screen_evidence_record(result.record)
        item = {
            "citation_label": "E%d" % idx if result in primary else None,
            "evidence_type": record.get("evidence_type", "unknown"),
            "piece": record["piece"],
            "work": record["work"],
            "query_scope_relation": result.scope_match,
            "evidence_role": scope_evidence_role(result.scope_match),
        }
        # This is a retriever-computed relevance signal, never source text or
        # safety approval. Keep a directly matching reviewed question's answer
        # available even when the user's wording asks for a general definition.
        if result in primary and getattr(result, "alias_score", 0.0) >= 1.0:
            item["question_alias_match"] = True
        if measure_ranges or result.scope_match != "local_example":
            item["measure_range"] = record.get("measure_range") or []
            item["measure_scope"] = record.get("measure_scope", "")
        if result.scope_match == "other_range_context":
            item["content_withheld"] = True
        else:
            for field in (
                "topic", "question", "answer", "piece_title", "knowledge_scope", "usage_class",
                "lyric_anchor", "web_coaching_source", "sumijo_source", "edition_id",
                "edition_context", "measure_system", "sources",
                "generated_text_license", "inherited_licenses",
            ):
                if field in record:
                    item[field] = record[field]
            # Finalized KUs store their original teaching questions in retrieval
            # aliases, often with no question field. Supply that real context
            # instead of asking the selector to interpret an empty question.
            if not item.get("question"):
                aliases = record.get("retrieval_aliases") or []
                if isinstance(aliases, list):
                    questions = [q for q in aliases if isinstance(q, str) and q.strip()]
                    if questions:
                        item["question"] = "\n".join(dict.fromkeys(questions))
        evidence.append(item)
    return evidence


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
    for result in results:
        screen_text(json.dumps(result.record, ensure_ascii=False), source="evidence")
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
        if is_web_record(result.record)
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
    web_coaching_supplement: int = 0,
    web_coaching_supplement_gate: str = "always",
    web_coaching_supplement_margin: float = 0.0,
    score_context: str = "",
    message_builder=None,
) -> tuple[List[Any], List[Dict[str, str]]]:
    """Retrieve and freeze the one evidence bundle used for generation.

    ``web_coaching_supplement`` appends up to that many of the best-ranked
    web_coaching candidates (no-range queries only) after the expert-led
    selection, which would otherwise replace them with expert-only bundles.
    With ``web_coaching_supplement_gate="dense_margin"`` this happens only
    when the best web_coaching claim is the semantically closer match.

    The function performs no model call, repair call, or model fallback. It
    returns the reviewed evidence bundle and prompt for the one natural,
    grounded generation call made by the caller.
    """

    from soprano_qa.source_priority import SourcePriorityIndex, merge_evidence
    if message_builder is None:
        from soprano_qa.single_pass import build_messages as message_builder
    make_messages = message_builder
    priority_index = index if isinstance(index, SourcePriorityIndex) else None
    if priority_index is not None:
        index = priority_index.base_index

    if top_k < 1:
        return [], make_messages(piece, measures, query, [], score_context=score_context)
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
        return [], make_messages(piece, measures, query, [], score_context=score_context)
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
    coaching_index = getattr(index, "coaching_index", None)
    coaching_candidates = (
        coaching_index.search(query, piece, top_k=web_coaching_supplement)
        if coaching_index is not None
        and web_coaching_supplement > 0
        and not measure_ranges
        else []
    )
    results = supplement_web_coaching_evidence(
        results,
        coaching_candidates,
        measure_ranges=measure_ranges,
        limit=web_coaching_supplement,
        gate=web_coaching_supplement_gate,
        reference_candidates=stable_candidates,
        margin=web_coaching_supplement_margin,
    )
    if priority_index is not None:
        direct = priority_index.sumijo_index.search(
            query=query, piece=piece, measure_ranges=measure_ranges,
            topic=topic, top_k=top_k,
        )
        results = merge_evidence(direct, results, top_k)
    return results, make_messages(
        piece, measures, query, results, score_context=score_context
    )


def supplement_web_coaching_evidence(
    results: List[Any],
    candidates: List[Any],
    *,
    measure_ranges: List[List[int]],
    limit: int,
    gate: str = "always",
    reference_candidates: List[Any] = (),
    margin: float = 0.0,
) -> List[Any]:
    """Append the best-ranked web_coaching candidates after expert evidence.

    ``candidates`` come from the separate web_coaching index. Expert-led
    focusing stays authoritative for which expert records are used; this only
    adds web_coaching evidence. Ranged queries are left unchanged because
    web_coaching claims carry no confirmed score location. With
    ``gate="dense_margin"``, claims are added only when the best one is more
    similar to the question (dense cosine) than every main-index candidate in
    ``reference_candidates`` by at least ``margin``: when curated evidence is
    the closer match, adding web claims invites the model to blend in material
    the expert did not state.
    """

    if gate not in {"always", "dense_margin"}:
        raise ValueError("Unsupported web_coaching supplement gate %r" % gate)
    if limit <= 0 or measure_ranges or not candidates:
        return results
    if gate == "dense_margin":
        best_reference = max(
            (float(c.dense_score) for c in reference_candidates),
            default=0.0,
        )
        best_coaching = max(float(c.dense_score) for c in candidates)
        if best_coaching < best_reference + margin:
            return results
    selected_ids = {result.record["id"] for result in results}
    supplement = []
    for candidate in candidates:
        if len(supplement) >= limit:
            break
        record = candidate.record
        if (
            record.get("evidence_type") == "web_coaching"
            and record["id"] not in selected_ids
        ):
            supplement.append(candidate)
            selected_ids.add(record["id"])
    return [*results, *supplement]


def build_evidence_notices(results: List[Any]) -> List[Dict[str, Any]]:
    notices = []
    seen_ids = set()
    for result in results:
        record = result.record
        if not is_web_record(record) or record["id"] in seen_ids:
            continue
        seen_ids.add(record["id"])
        if record.get("evidence_type") == "web_coaching":
            source = record.get("web_coaching_source") or {}
            notices.append(
                {
                    "record_id": record["id"],
                    "usage_class": None,
                    "generated_text_license": None,
                    "generated_text_attribution": None,
                    "generated_text_license_url": None,
                    "inherited_licenses": [],
                    "sources": [
                        {
                            "web_coaching_source_id": source.get("source_id"),
                            "kind": source.get("kind"),
                            "title": source.get("title"),
                            "author": source.get("author"),
                            "url": source.get("url"),
                            "locator": source.get("locator"),
                        }
                    ],
                }
            )
            continue
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



# Checks used by the evaluation runners (evaluation/run_qualitative.py and
# others) to flag answers that only state missing evidence.
NO_GROUNDED_ANSWER_SENTINEL = "<NO_GROUNDED_ANSWER>"

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
    """Recognize a wholly insufficient answer, preserving supported partial prose.

    A bare word such as ``부족`` or a general discussion of missing score
    information is not a refusal. Natural-language detection requires both a
    grounding-information shortfall and an inability to answer, know, or
    determine something within one short explanation. A substantive answer can
    coexist with a limitation, including in a contrasting clause of the same
    sentence. Its support is checked by the output guard and citation/scope
    validators, not by this text detector. Clarification requests alone do not
    supply an answer.
    """

    without_footer = _remove_evidence_footer(answer)
    if answer.strip() and not without_footer:
        return True
    compact = re.sub(r"\s+", "", answer)
    if NO_GROUNDED_ANSWER_SENTINEL.casefold() in compact.casefold():
        return True

    normalized = unicodedata.normalize("NFKC", without_footer)
    has_refusal = _nearby_signal_pair(
        normalized,
        KOREAN_GROUNDING_SHORTFALL_RE,
        KOREAN_ANSWER_INABILITY_RE,
        max_gap=80,
    ) or bool(ENGLISH_DIRECT_INSUFFICIENCY_RE.search(normalized)) or _nearby_signal_pair(
        normalized,
        ENGLISH_GROUNDING_SHORTFALL_RE,
        ENGLISH_ANSWER_INABILITY_RE,
        max_gap=120,
    )
    if not has_refusal:
        return False
    signals = (
        KOREAN_GROUNDING_SHORTFALL_RE, KOREAN_ANSWER_INABILITY_RE,
        ENGLISH_GROUNDING_SHORTFALL_RE, ENGLISH_ANSWER_INABILITY_RE,
        ENGLISH_DIRECT_INSUFFICIENCY_RE,
    )
    boilerplate = re.compile(
        r"(?:sorry|i(?:['’]m| am) sorry|thank you|thanks|죄송합니다|감사합니다)",
        re.IGNORECASE,
    )
    clarification = re.compile(
        r"(?:"
        r"(?:(?:could|can|would)\s+you\s+)?(?:please\s+)?"
        r"(?:(?:provide|share|supply|give|add|send)\s+(?:me\s+)?"
        r"(?:(?:some|more|additional|further|relevant)\s+)*"
        r"(?:details?|information|context|evidence)"
        r"(?:\s+(?:about|on|for)\s+(?:(?:your|the|this)\s+)?"
        r"(?:question|request|passage|issue))?"
        r"|(?:clarify|rephrase|specify)"
        r"(?:\s+(?:(?:your|the|this)\s+)?(?:question|request))?"
        r"|elaborate(?:\s+on\s+(?:(?:your|the|this)\s+)?(?:question|request))?)"
        r"(?:,?\s+please)?"
        r"|(?:(?:좀\s*더|더)\s*)?"
        r"(?:(?:자세한|구체적인|추가적인|추가|관련)\s*)?"
        r"(?:내용|정보|자료|문맥|근거|질문|상황)(?:을|를)?\s*"
        r"(?:(?:좀\s*더|더)\s*)?(?:구체적으로\s*)?"
        r"(?:알려|제공해|설명해|구체화해)\s*(?:주세요|주십시오|주시겠습니까)"
        r")",
        re.IGNORECASE,
    )
    # Split explicit contrasts, not every comma: a parenthetical fragment in
    # a refusal is not itself a supported answer. Keep Korean negative stems
    # intact so the limitation side still matches the existing signals.
    contrast = re.compile(
        r"\s*;\s*|\s+(?:but|however|nevertheless|yet)\b[,，]?\s*"
        r"|(?<=없)(?:지만|으나)[,，]?\s*"
        r"|(?<=어렵)지만[,，]?\s*|(?<=힘들)지만[,，]?\s*"
        r"|\s+(?:하지만|그러나|다만)[,，]?\s*",
        re.IGNORECASE,
    )
    for claim in _answer_claims(normalized):
        for clause in contrast.split(ANSWER_EVIDENCE_LABEL_RE.sub("", claim)):
            prose = clause.strip(" \t\r\n.!?。！？,，;")
            if (
                not _claim_has_substantive_text(prose)
                or boilerplate.fullmatch(prose)
                or clarification.fullmatch(prose)
            ):
                continue
            if not any(pattern.search(prose) for pattern in signals):
                return False
    return True


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
            'Optional inclusive measure range like "28-30" or "12"; '
            "omit when no score range is selected"
        ),
    )
    parser.add_argument("--question", required=True, help="Singer/user question")
    parser.add_argument("--language", default="ko", choices=("ko", "en"), help="Answer language")
    parser.add_argument("--top-k", type=positive_int, default=6, help="Number of chunks to retrieve")
    parser.add_argument("--no-generate", action="store_true", help="Only retrieve evidence")
    parser.add_argument("--rebuild-corpus", action="store_true", help="Rebuild corpus before asking")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()
    try:
        ranges = parse_measure_ranges(args.measures)
    except ValueError as exc:
        parser.error(str(exc))
    if len(ranges) > 1:
        parser.error("--measures takes one range, like 28-30")
    args.measure_range = tuple(ranges[0]) if ranges else None
    return args


def main() -> None:
    """Ask one question through soprano_qa.service and print the result."""
    args = parse_args()
    from soprano_qa import service

    if args.settings:
        service.SETTINGS = load_settings(args.settings, use_legacy_dataset_env=False)
    if args.rebuild_corpus:
        ensure_corpus(service.SETTINGS, rebuild=True)
    result = service.ask(
        piece_id=args.piece,
        question=args.question,
        measure_range=args.measure_range,
        generate=not args.no_generate,
        top_k=args.top_k,
        language=args.language,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print("Retrieved evidence")
    print("==================")
    for idx, item in enumerate(result["evidence"], 1):
        ranges = format_measure_range(item["measure_ranges"]) if item["measure_ranges"] else "whole piece"
        print(f"[{idx}] {item['kind']} {item['id']} ({ranges})")
        print(item["text"])
        print()
    if args.no_generate:
        print("Generation skipped by --no-generate.")
    print("Answer")
    print("======")
    print(result["answer"])
    if result["evidence_notices"]:
        print()
        print("Evidence notices")
        print("================")
        for notice in result["evidence_notices"]:
            print(format_evidence_notice(notice))


if __name__ == "__main__":
    main()
