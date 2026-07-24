#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dependency-free BM25 retrieval with piece and measure-aware boosts."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence


TOKEN_RE = re.compile(r"[0-9]+|[A-Za-zÀ-ÖØ-öø-ÿ]+|[가-힣]+")
RANGE_DASHES = str.maketrans({"–": "-", "—": "-", "−": "-", "‒": "-"})
PIECE_TITLE_ALIASES = {
    "die-forelle": [
        "die forelle",
        "die-forelle",
        "디 포렐레",
        "송어",
        "the trout",
        "forelle",
        "포렐레",
        "d. 550",
        "d 550",
    ],
    "in-flowery-clouds": [
        "in flowery clouds",
        "in-flowery-clouds",
        "꽃구름 속에",
        "꽃 구름 속에",
        "꽃구름",
    ],
    "la-capinera": [
        "la capinera",
        "la-capinera",
        "라 카피네라",
        "capinera",
        "카피네라",
        "the wren",
    ],
    "nella-fantasia": [
        "nella fantasia",
        "nella-fantasia",
        "넬라 판타지아",
        "fantasia",
        "판타지아",
    ],
    "una-voce-poco-fa": [
        "una voce poco fa",
        "una-voce-poco-fa",
        "우나 보체 포코 파",
        "una voce",
        "우나 보체",
    ],
}
PIECE_CREATOR_ALIASES = {
    "die-forelle": ["franz schubert", "프란츠 슈베르트", "schubert", "슈베르트"],
    "in-flowery-clouds": ["lee heung-ryul", "lee heung ryul", "이흥렬"],
    "la-capinera": [
        "julius benedict",
        "줄리어스 베네딕트",
        "benedict",
        "베네딕트",
    ],
    "nella-fantasia": [
        "ennio morricone",
        "엔니오 모리코네",
        "morricone",
        "모리코네",
    ],
    "una-voce-poco-fa": [
        "gioachino rossini",
        "조아키노 로시니",
        "rossini",
        "로시니",
    ],
}
PIECE_QUERY_ALIASES = {
    piece: PIECE_TITLE_ALIASES[piece] + PIECE_CREATOR_ALIASES[piece]
    for piece in PIECE_TITLE_ALIASES
}
CREATOR_INTENT_TERMS = (
    "누가",
    "누구",
    "생애",
    "약력",
    "배경",
    "작곡가",
    "창작자",
    "출생",
    "사망",
    "인물",
    "who",
    "biography",
    "composer",
    "composed",
    "작곡",
    "어떤 사람",
    "무슨 사람",
    "어떤 인물",
    "무슨 인물",
)
CREATOR_ACTION_TERMS = (
    "작곡",
    "작곡가",
    "작곡자",
    "composer",
    "composed",
    "wrote",
)
TITLE_IDENTITY_INTENT_TERMS = (
    "무엇",
    "어떤 곡",
    "무슨 곡",
    "어떤 작품",
    "무슨 작품",
    "어떤 노래",
    "무슨 노래",
    "정식 표제",
    "제목",
    "작품 번호",
    "작품번호",
    "원제",
    "장르",
    "what is",
    "title",
    "catalog",
    "genre",
)
EXPLICIT_TITLE_IDENTITY_TERMS = (
    "정식 표제",
    "제목",
    "작품 번호",
    "작품번호",
    "원제",
    "title",
    "catalog",
)
GENERIC_QUERY_PHRASES = [
    "어떤 노래",
    "무슨 노래",
    "어떤 작품",
    "무슨 작품",
    "어떤 곡",
    "무슨 곡",
    "어디인가요",
    "어디인가",
    "어디인지",
    "무엇인가요",
    "무엇인가",
    "무엇인지",
    "해당 마디",
    "이 마디",
    "해야 하나요",
    "알려 주세요",
    "알려주세요",
    "설명해 주세요",
    "설명해주세요",
    "말해 주세요",
    "어떻게",
    "하나요",
    "인가요",
    "할까요",
    "여기",
    "이 곡",
    "이 노래",
    "이 작품",
    "하면 좋을까요",
    "에 대해",
    "대해",
]
QUERY_SCAFFOLD_WORDS = {
    "어떤",
    "어느",
    "무슨",
    "뭐",
    "뭐예요",
    "뭔가요",
    "무엇",
    "누구",
    "누가",
    "누구예요",
    "누구인가요",
    "왜",
    "언제",
    "어디",
    "몇",
    "얼마나",
    "되나요",
    "될까요",
    "됩니까",
    "되는지",
    "궁금해요",
    "궁금합니다",
    "하면",
    "좋을까요",
    "좋나요",
    "알려줘",
    "알려줘요",
    "말해줘",
    "말해주세요",
    "말해",
    "주세요",
    "부탁해요",
    "어떤가요",
    "에게서",
    "으로는",
    "에서는",
    "부터는",
    "까지는",
    "으로",
    "에서",
    "에게",
    "부터",
    "까지",
    "부분",
    "방법",
    "정확하게",
    "자연스럽게",
    "부르는",
    "부를",
    "부르",
    "구성되어",
    "및",
    "관해",
    "같이",
    "함께",
    "그리고",
    "작곡자",
    "작곡가",
    "작곡한",
    "사람",
    "누구죠",
}
ENGLISH_QUERY_SCAFFOLD_WORDS = {
    "about",
    "and",
    "are",
    "at",
    "be",
    "been",
    "being",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "explain",
    "for",
    "from",
    "how",
    "in",
    "is",
    "me",
    "of",
    "on",
    "or",
    "please",
    "should",
    "tell",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "why",
    "with",
    "would",
    "you",
    "your",
    "wikimedia",
    "commons",
    "bibliothèque",
    "bnf",
}
QUERY_SCAFFOLD_SUFFIXES = (
    "나요",
    "까요",
    "예요",
    "에요",
    "습니까",
    "는지",
    "인지",
    "었는지",
    "였는지",
)
QUERY_TOKEN_NORMALIZATIONS = {
    "딕션": "발음",
    "관할권별": "관할권",
    # The one-syllable stem in "박의" is intentionally outside the generic
    # particle stripper. Canonicalize this exact morphology without relaxing
    # the full-concept coverage gate.
    "박의": "박자",
    # The annotation uses the inflection "나타낸다"; its indexed variants
    # include "나타낸", which is a safe common form for this paraphrase.
    "나타내는": "나타낸",
    "바뀌": "변화",
    "바뀌는": "변화",
    "바뀌나요": "변화",
    "달라지": "변화",
    "달라지는지": "변화",
    "달라지나요": "변화",
}
QUERY_TOKEN_STEM_RES = (
    re.compile(r"^([가-힣]{2,}?)(?:이에요|예요|에요|인가요|입니까|일까요)$"),
    re.compile(r"^([가-힣]{2,})하려면$"),
    re.compile(r"^([가-힣]{2,})할지$"),
    re.compile(r"^([가-힣]{2,})해야$"),
    re.compile(r"^([가-힣]{2,})하$"),
)
MEASURE_MENTION_RE = re.compile(
    r"[0-9]+\s*(?:[-–—−‒~～]\s*[0-9]+\s*)?(?:번째\s*)?마디"
    r"(?:부터|까지|에서|에는|의|는|를)?"
)
QUESTION_SPAN_RANGE_RES = (
    re.compile(
        r"(?:제\s*)?([0-9]+)\s*(?:번째\s*)?마디\s*"
        r"[-–—−‒~～]\s*(?:제\s*)?([0-9]+)\s*(?:번째\s*)?마디"
    ),
    re.compile(
        r"(?:제\s*)?([0-9]+)\s*(?:번째\s*)?마디\s*(?:부터|에서)\s*([0-9]+)\s*"
        r"(?:(?:번째\s*)?마디(?:까지)?|까지)"
    ),
    re.compile(
        r"(?:제\s*)?([0-9]+)\s*(?:부터|에서)\s*([0-9]+)\s*"
        r"(?:번째\s*)?마디(?:까지)?"
    ),
    re.compile(
        r"\b(?:bars?|measures?)\s*([0-9]+)\s*"
        r"(?:[-–—−‒~～]|to|through)\s*([0-9]+)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\bm\.\s*([0-9]+)\s*(?:[-–—−‒~～]|to|through)\s*"
        r"(?:m\.\s*)?([0-9]+)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b([0-9]+)\s*(?:[-–—−‒~～]|to|through)\s*([0-9]+)\s*"
        r"(?:bars?|measures?)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b([0-9]+)\s*(?:bars?|measures?)\s*"
        r"(?:[-–—−‒~～]|to|through)\s*([0-9]+)\s*(?:bars?|measures?)\b",
        flags=re.IGNORECASE,
    ),
)
QUESTION_COMPACT_RANGE_RE = re.compile(
    r"([0-9]+)\s*[-–—−‒~～]\s*([0-9]+)\s*(?:번째\s*)?마디"
)
QUESTION_ABBREVIATED_MEASURE_LIST_RE = re.compile(
    r"((?:[0-9]+\s*(?:번\s*)?(?:[,，]|과|와|및)\s*)+)"
    r"([0-9]+)\s*(?:번\s*)?(?:번째\s*)?마디"
)
QUESTION_ENGLISH_MEASURE_LIST_RE = re.compile(
    r"\b(?:bars?|measures?)\s+"
    r"((?:[0-9]+\s*(?:(?:[,，]\s*(?:and\s+)?)|(?:and\s+)))+[0-9]+)\b",
    flags=re.IGNORECASE,
)
QUESTION_KOREAN_MIXED_MEASURE_LIST_RE = re.compile(
    r"("
    r"[0-9]+\s*(?:번\s*)?(?:[-–—−‒~～]\s*[0-9]+\s*(?:번\s*)?)?"
    r"(?:\s*(?:[,，]|과|와|및)\s*"
    r"[0-9]+\s*(?:번\s*)?(?:[-–—−‒~～]\s*[0-9]+\s*(?:번\s*)?)?)+"
    r")\s*(?:번째\s*)?마디"
)
QUESTION_ENGLISH_MIXED_MEASURE_LIST_RES = (
    re.compile(
        r"\b(?:bars?|measures?)\s+("
        r"[0-9]+(?:\s*[-–—−‒~～]\s*[0-9]+)?"
        r"(?:(?:\s*[,，]\s*(?:and\s+)?|\s+and\s+)"
        r"[0-9]+(?:\s*[-–—−‒~～]\s*[0-9]+)?)+)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b("
        r"[0-9]+(?:\s*[-–—−‒~～]\s*[0-9]+)?"
        r"(?:(?:\s*[,，]\s*(?:and\s+)?|\s+and\s+)"
        r"[0-9]+(?:\s*[-–—−‒~～]\s*[0-9]+)?)+)\s*"
        r"(?:bars?|measures?)\b",
        flags=re.IGNORECASE,
    ),
)
QUESTION_OPENING_MEASURE_COUNT_RE = re.compile(
    r"(?:첫|처음|앞)\s*([0-9]+)\s*마디"
    r"|\b(?:first|opening)\s+([0-9]+)\s*(?:bars?|measures?)\b",
    flags=re.IGNORECASE,
)
QUESTION_CARDINAL_MEASURE_RE = re.compile(
    r"[0-9]+\s*마디(?:"
    r"\s*(?:의\s*)?(?:프레이즈|구절|악절|길이|단위|동안|분량|규모|구조|패턴|형식|주기)"
    r"|씩|마다|짜리|\s*에\s*걸쳐|\s*로\s*(?:된|구성|이루어진)"
    r")"
)
QUESTION_TOTAL_MEASURE_COUNT_RES = (
    re.compile(r"(?:총|전체(?:가|는|은)?)\s*[0-9]+\s*(?:개\s*)?마디"),
    re.compile(
        r"(?:(?:총|전체)\s*)?마디\s*(?:수|개수)\s*(?:은|는|이|가)?\s*"
        r"[0-9]+\s*(?:개\s*)?마디"
    ),
    re.compile(
        r"(?:이\s*)?(?:곡|노래|작품)\s*(?:은|는|이|가)\s*(?:모두\s*)?"
        r"[0-9]+\s*(?:개\s*)?마디"
        r"(?=\s*(?:인가요|인가|이에요|예요|에요|입니까|맞나요|맞습니까|죠|이지요|\?|$))"
    ),
    re.compile(
        r"(?:이\s*)?(?:곡|노래|작품)\s*(?:의\s*)?(?:전체\s*)?"
        r"길이\s*(?:은|는|이|가)?\s*[0-9]+\s*(?:개\s*)?마디"
        r"(?=\s*(?:인가요|인가|이에요|예요|에요|입니까|맞나요|맞습니까|죠|이지요|\?|$))"
    ),
)
QUESTION_LOCATIVE_MEASURE_RES = (
    re.compile(r"제\s*([0-9]+)\s*마디"),
    re.compile(r"([0-9]+)\s*번째\s*마디"),
    re.compile(r"([0-9]+)\s*번\s*(?:째\s*)?마디"),
    re.compile(r"([0-9]+)\s*마디"),
    re.compile(r"\b(?:bar|measure)\s*(?:no\.?\s*)?([0-9]+)\b", re.IGNORECASE),
    re.compile(r"\bm\.\s*([0-9]+)\b", re.IGNORECASE),
)
QUESTION_FIRST_MEASURE_RE = re.compile(
    r"첫\s*(?:번째\s*)?마디|\bfirst\s+(?:bar|measure)\b",
    flags=re.IGNORECASE,
)
QUESTION_UNRESOLVED_MEASURE_RE = re.compile(
    r"(?:"
    r"(?:마지막|끝)\s*(?:(?:[0-9]+|한|두|세|네)\s*)?(?:번째\s*)?마디"
    r"|\b(?:last|final|end)\s+(?:[0-9]+\s+)?(?:bars?|measures?)\b"
    r")",
    flags=re.IGNORECASE,
)
KOREAN_PARTICLE_RE = re.compile(
    r"([가-힣]{2,}?)(?:에게서|으로는|에서는|부터는|까지는|은요|는요|이요|가요|도요|이랑|하고|으로|에서|에게|부터|까지|은|는|이|가|을|를|의|와|과|랑|에|도|만|로)\b"
)
GENERIC_PIECE_MENTION_RE = re.compile(
    r"(?<![가-힣])(?:이\s*)?(?:곡|노래|작품)"
    r"(?:에게서|으로는|에서는|부터는|까지는|으로|에서|에게|부터|까지|은|는|이|가|을|를|의|와|과|에|도|만|로)?"
    r"(?![가-힣])"
)


def token_variants(token: str) -> List[str]:
    variants = [token]
    if re.fullmatch(r"[가-힣]+", token):
        for n in (2, 3):
            if len(token) > n:
                variants.extend(token[i : i + n] for i in range(len(token) - n + 1))
    return list(dict.fromkeys(variants))


def is_query_scaffolding(token: str) -> bool:
    return token in QUERY_SCAFFOLD_WORDS or token in ENGLISH_QUERY_SCAFFOLD_WORDS or (
        len(token) > 2 and token.endswith(QUERY_SCAFFOLD_SUFFIXES)
    )


def token_groups(text: str) -> List[List[str]]:
    groups: List[List[str]] = []
    seen_tokens = set()
    for match in TOKEN_RE.finditer(text.lower()):
        token = match.group(0)
        token = QUERY_TOKEN_NORMALIZATIONS.get(token, token)
        for stem_pattern in QUERY_TOKEN_STEM_RES:
            stem_match = stem_pattern.fullmatch(token)
            if stem_match:
                token = stem_match.group(1)
                break
        if len(token) == 1 and not token.isdigit():
            continue
        if token in seen_tokens or is_query_scaffolding(token):
            continue
        seen_tokens.add(token)
        groups.append(token_variants(token))
    return groups


def tokenize(text: str) -> List[str]:
    tokens = []
    normalized_text = KOREAN_PARTICLE_RE.sub(r"\1", text.lower())
    for match in TOKEN_RE.finditer(normalized_text):
        token = match.group(0)
        if len(token) == 1 and not token.isdigit():
            continue
        tokens.extend(token_variants(token))
    return tokens


def load_corpus(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def strip_question_measure_mentions(text: str) -> str:
    """Remove score-location syntax without leaving range endpoints as concepts."""

    stripped = text
    for total_count_pattern in QUESTION_TOTAL_MEASURE_COUNT_RES:
        stripped = total_count_pattern.sub(" ", stripped)
    stripped = QUESTION_OPENING_MEASURE_COUNT_RE.sub(" ", stripped)
    stripped = QUESTION_UNRESOLVED_MEASURE_RE.sub(" ", stripped)
    stripped = QUESTION_KOREAN_MIXED_MEASURE_LIST_RE.sub(" ", stripped)
    for mixed_list_pattern in QUESTION_ENGLISH_MIXED_MEASURE_LIST_RES:
        stripped = mixed_list_pattern.sub(" ", stripped)
    stripped = QUESTION_ABBREVIATED_MEASURE_LIST_RE.sub(" ", stripped)
    stripped = QUESTION_ENGLISH_MEASURE_LIST_RE.sub(" ", stripped)
    for span_pattern in QUESTION_SPAN_RANGE_RES:
        stripped = span_pattern.sub(" ", stripped)
    stripped = QUESTION_COMPACT_RANGE_RE.sub(" ", stripped)
    stripped = QUESTION_FIRST_MEASURE_RE.sub(" ", stripped)
    for locative_pattern in QUESTION_LOCATIVE_MEASURE_RES:
        stripped = locative_pattern.sub(" ", stripped)
    return MEASURE_MENTION_RE.sub(" ", stripped)


def query_components(query: str, piece: str | None) -> tuple[str, List[str]]:
    """Return real query text separately from synthetic identity intent anchors."""

    stripped = query
    lowered_query = query.lower()
    creator_intent = False
    creator_mentioned = False
    creator_subject_mentioned = False
    title_identity_requested = False
    explicit_title_identity_requested = False
    if piece:
        generic_piece_mentioned = bool(GENERIC_PIECE_MENTION_RE.search(lowered_query))
        creator_mentioned = any(
            alias.lower() in lowered_query for alias in PIECE_CREATOR_ALIASES.get(piece, [])
        )
        title_mentioned = any(
            alias.lower() in lowered_query for alias in PIECE_TITLE_ALIASES.get(piece, [])
        )
        identity_subject_mentioned = title_mentioned or generic_piece_mentioned
        creator_subject_mentioned = creator_mentioned or identity_subject_mentioned
        creator_action_requested = any(
            term in lowered_query for term in CREATOR_ACTION_TERMS
        )
        creator_intent = (
            creator_subject_mentioned
            and any(term in lowered_query for term in CREATOR_INTENT_TERMS)
        ) or creator_action_requested
        title_identity_requested = identity_subject_mentioned and any(
            term in lowered_query for term in TITLE_IDENTITY_INTENT_TERMS
        )
        explicit_title_identity_requested = identity_subject_mentioned and any(
            term in lowered_query for term in EXPLICIT_TITLE_IDENTITY_TERMS
        )
        aliases = sorted(PIECE_QUERY_ALIASES.get(piece, []), key=len, reverse=True)
        for alias in aliases:
            stripped = re.sub(re.escape(alias), " ", stripped, flags=re.IGNORECASE)
    stripped = strip_question_measure_mentions(stripped)
    if piece:
        stripped = GENERIC_PIECE_MENTION_RE.sub(" ", stripped)
    for phrase in GENERIC_QUERY_PHRASES:
        stripped = stripped.replace(phrase, " ")
    stripped = KOREAN_PARTICLE_RE.sub(r"\1", stripped)
    intent_anchors = []
    if creator_intent and (creator_subject_mentioned or not token_groups(stripped)):
        intent_anchors.append("약력")
    if title_identity_requested and (
        explicit_title_identity_requested or not token_groups(stripped)
    ):
        intent_anchors.append("별칭")
    return stripped, intent_anchors


def strip_piece_aliases(query: str, piece: str | None) -> str:
    stripped, intent_anchors = query_components(query, piece)
    if intent_anchors:
        stripped += " " + " ".join(intent_anchors)
    return stripped


def parse_measure_ranges(text: str) -> List[List[int]]:
    ranges: List[List[int]] = []
    normalized = text.strip().translate(RANGE_DASHES)
    if not normalized:
        return ranges
    for part in re.split(r"[,，;]", normalized):
        part = part.strip()
        if not part:
            raise ValueError("Empty item in measure range: %r" % text)
        match = re.fullmatch(r"([0-9]+)(?:\s*-\s*([0-9]+))?", part)
        if not match:
            raise ValueError("Invalid measure range %r; use forms like 28-30 or 12, 35" % part)
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if start < 1:
            raise ValueError("Measure numbers must be positive: %r" % part)
        if end < start:
            raise ValueError("Measure range must be ascending: %r" % part)
        ranges.append([start, end])
    return ranges


def extract_question_measure_ranges(text: str) -> List[List[int]]:
    """Extract explicit Arabic-number measure mentions from natural-language text."""

    ranges: List[List[int]] = []

    def capture_range(match: re.Match[str]) -> str:
        start = int(match.group(1))
        end = int(match.group(2))
        if start < 1 or end < start:
            raise ValueError("Invalid measure range in question: %r" % match.group(0))
        ranges.append([start, end])
        return " " * len(match.group(0))

    remaining = text
    for total_count_pattern in QUESTION_TOTAL_MEASURE_COUNT_RES:
        remaining = total_count_pattern.sub(
            lambda match: " " * len(match.group(0)),
            remaining,
        )

    def capture_opening_count(match: re.Match[str]) -> str:
        end = int(next(value for value in match.groups() if value is not None))
        if end < 1:
            raise ValueError("Opening measure count must be positive")
        ranges.append([1, end])
        return " " * len(match.group(0))

    remaining = QUESTION_OPENING_MEASURE_COUNT_RE.sub(capture_opening_count, remaining)
    remaining = QUESTION_UNRESOLVED_MEASURE_RE.sub(
        lambda match: " " * len(match.group(0)),
        remaining,
    )

    def capture_mixed_measure_list(match: re.Match[str]) -> str:
        normalized_items = re.sub(
            r"(?:과|와|및|\band\b)",
            ",",
            match.group(1),
            flags=re.IGNORECASE,
        )
        for raw_item in re.split(r"[,，]", normalized_items):
            item = re.sub(r"번", "", raw_item).strip()
            item_match = re.fullmatch(
                r"([0-9]+)(?:\s*[-–—−‒~～]\s*([0-9]+))?",
                item,
            )
            if item_match is None:
                raise ValueError("Invalid mixed measure list item: %r" % raw_item)
            start = int(item_match.group(1))
            end = int(item_match.group(2) or start)
            if start < 1 or end < start:
                raise ValueError("Invalid measure range in question: %r" % raw_item)
            ranges.append([start, end])
        return " " * len(match.group(0))

    remaining = QUESTION_KOREAN_MIXED_MEASURE_LIST_RE.sub(
        capture_mixed_measure_list,
        remaining,
    )
    for mixed_list_pattern in QUESTION_ENGLISH_MIXED_MEASURE_LIST_RES:
        remaining = mixed_list_pattern.sub(capture_mixed_measure_list, remaining)

    def capture_measure_list(match: re.Match[str]) -> str:
        numbers = [int(value) for value in re.findall(r"[0-9]+", match.group(0))]
        if any(number < 1 for number in numbers):
            raise ValueError("Measure numbers in the question must be positive")
        ranges.extend([number, number] for number in numbers)
        return " " * len(match.group(0))

    remaining = QUESTION_ABBREVIATED_MEASURE_LIST_RE.sub(
        capture_measure_list,
        remaining,
    )
    remaining = QUESTION_ENGLISH_MEASURE_LIST_RE.sub(
        capture_measure_list,
        remaining,
    )
    for span_pattern in QUESTION_SPAN_RANGE_RES:
        remaining = span_pattern.sub(capture_range, remaining)
    remaining = QUESTION_COMPACT_RANGE_RE.sub(capture_range, remaining)
    # A bare count can describe phrase length rather than a score location.
    # Remove common cardinal constructions before treating remaining N마디
    # forms as locative by default.
    remaining = QUESTION_CARDINAL_MEASURE_RE.sub(
        lambda match: " " * len(match.group(0)),
        remaining,
    )

    def capture_first(match: re.Match[str]) -> str:
        ranges.append([1, 1])
        return " " * len(match.group(0))

    remaining = QUESTION_FIRST_MEASURE_RE.sub(capture_first, remaining)

    def capture_single(match: re.Match[str]) -> str:
        measure = int(match.group(1))
        if measure < 1:
            raise ValueError("Measure numbers in the question must be positive")
        ranges.append([measure, measure])
        return " " * len(match.group(0))

    for locative_pattern in QUESTION_LOCATIVE_MEASURE_RES:
        remaining = locative_pattern.sub(capture_single, remaining)
    return ranges


def ranges_cover(
    covering_ranges: Sequence[Sequence[int]],
    requested_ranges: Sequence[Sequence[int]],
) -> bool:
    """Return whether every requested integer measure is covered by the union."""

    covering = sorted((int(start), int(end)) for start, end in covering_ranges)
    for requested_start, requested_end in requested_ranges:
        cursor = int(requested_start)
        for covering_start, covering_end in covering:
            if covering_end < cursor:
                continue
            if covering_start > cursor:
                break
            cursor = max(cursor, covering_end + 1)
            if cursor > requested_end:
                break
        if cursor <= requested_end:
            return False
    return True


def validate_question_measure_contract(
    question: str,
    measure_ranges: Sequence[Sequence[int]],
) -> List[List[int]]:
    """Reject ambiguous or contradictory CLI/question measure scopes."""

    unresolved = QUESTION_UNRESOLVED_MEASURE_RE.search(question)
    if unresolved and not measure_ranges:
        raise ValueError(
            "Question uses %r without a numeric measure; pass an explicit numeric "
            "--measures range" % unresolved.group(0)
        )
    mentioned_ranges = extract_question_measure_ranges(question)
    if mentioned_ranges and not measure_ranges:
        raise ValueError(
            "Question mentions measures %r; pass a matching --measures value"
            % mentioned_ranges
        )
    if mentioned_ranges and not ranges_cover(measure_ranges, mentioned_ranges):
        raise ValueError(
            "Question measures %r are outside --measures %r"
            % (mentioned_ranges, list(measure_ranges))
        )
    return mentioned_ranges


def ranges_overlap(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]]) -> bool:
    for a_start, a_end in a:
        for b_start, b_end in b:
            if a_start <= b_end and b_start <= a_end:
                return True
    return False


def range_distance(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]]) -> int | None:
    if not a or not b:
        return None
    best = None
    for a_start, a_end in a:
        for b_start, b_end in b:
            if a_start <= b_end and b_start <= a_end:
                return 0
            dist = b_start - a_end if a_end < b_start else a_start - b_end
            best = dist if best is None else min(best, dist)
    return best


@dataclass
class SearchResult:
    record: Dict[str, Any]
    score: float
    text_score: float
    measure_score: float
    piece_score: float
    scope_match: str


class BM25Index:
    def __init__(self, records: List[Dict[str, Any]], k1: float = 1.4, b: float = 0.75) -> None:
        self.records = records
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(record.get("retrieval_text") or record["answer"]) for record in records]
        self.relevance_tokens = [
            tokenize(
                record.get("relevance_text")
                or record.get("retrieval_text")
                or record["answer"]
            )
            for record in records
        ]
        self.doc_lens = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_lens) / max(1, len(self.doc_lens))
        self.term_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        self.relevance_term_freqs = [Counter(tokens) for tokens in self.relevance_tokens]
        doc_freq: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            doc_freq.update(set(tokens))
        self.idf = {
            term: math.log(1 + (len(records) - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }

    def text_score(self, query: str, idx: int) -> float:
        query_terms = list(dict.fromkeys(tokenize(query)))
        return self.text_score_terms(query_terms, idx)

    def text_score_terms(self, query_terms: Sequence[str], idx: int) -> float:
        if not query_terms:
            return 0.0
        freqs = self.term_freqs[idx]
        doc_len = self.doc_lens[idx]
        score = 0.0
        for term in query_terms:
            tf = freqs.get(term, 0)
            if not tf:
                continue
            denom = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avgdl, 1e-9))
            score += self.idf.get(term, 0.0) * (tf * (self.k1 + 1) / denom)
        return score

    def search(
        self,
        query: str,
        piece: str | None = None,
        measure_ranges: List[List[int]] | None = None,
        topic: str | None = None,
        top_k: int = 8,
    ) -> List[SearchResult]:
        measure_ranges = measure_ranges or []
        real_query, intent_anchors = query_components(query, piece)
        real_query_groups = token_groups(real_query)
        intent_query_groups = [token_variants(anchor) for anchor in intent_anchors]
        query_groups = real_query_groups + intent_query_groups
        query_terms = list(
            dict.fromkeys(variant for group in query_groups for variant in group)
        )
        if not query_groups:
            return []
        candidates: List[tuple[SearchResult, set[int], set[int]]] = []
        for idx, record in enumerate(self.records):
            if not record.get("retrieval_eligible", True):
                continue
            if piece and record["piece"] != piece:
                continue
            if topic and topic not in record["topic"]:
                continue
            scope_match = measure_scope_match(record, measure_ranges)
            if measure_ranges and scope_match == "outside_query_range":
                continue
            matched_real_groups = {
                group_index
                for group_index, group in enumerate(real_query_groups)
                if group[0] in self.relevance_term_freqs[idx]
            }
            matched_intent_groups = {
                group_index
                for group_index, group in enumerate(intent_query_groups)
                if group[0] in self.relevance_term_freqs[idx]
            }
            if not matched_real_groups and not matched_intent_groups:
                continue
            text = self.text_score_terms(query_terms, idx)
            measure = measure_boost(record, measure_ranges)
            piece_boost = 1.0 if piece and record["piece"] == piece else 0.0
            total = text + measure
            candidates.append(
                (
                    SearchResult(record, total, text, measure, piece_boost, scope_match),
                    matched_real_groups,
                    matched_intent_groups,
                )
            )
        covered_real_groups = {
            group_index
            for _, matched_groups, _ in candidates
            for group_index in matched_groups
        }
        if covered_real_groups != set(range(len(real_query_groups))):
            return []
        covered_intent_groups = {
            group_index
            for _, _, matched_groups in candidates
            for group_index in matched_groups
        }
        if covered_intent_groups != set(range(len(intent_query_groups))):
            return []
        candidates.sort(
            key=lambda candidate: (
                scope_priority(candidate[0].scope_match, bool(measure_ranges)),
                candidate[0].score,
                candidate[0].text_score,
                candidate[0].record["id"],
            ),
            reverse=True,
        )
        if top_k < 1:
            return []
        real_group_count = len(real_query_groups)
        full_coverage_mask = (1 << (real_group_count + len(intent_query_groups))) - 1
        candidate_masks = []
        for _, matched_real, matched_intent in candidates:
            mask = sum(1 << group_index for group_index in matched_real)
            mask |= sum(
                1 << (real_group_count + group_index)
                for group_index in matched_intent
            )
            candidate_masks.append(mask)

        def subset_quality(indices: tuple[int, ...]) -> tuple[Any, ...]:
            selected_results = [candidates[index][0] for index in indices]
            return (
                max(
                    (
                        scope_priority(result.scope_match, bool(measure_ranges))
                        for result in selected_results
                    ),
                    default=0,
                ),
                -len(indices),
                sum(
                    scope_priority(result.scope_match, bool(measure_ranges))
                    for result in selected_results
                ),
                sum(result.score for result in selected_results),
                sum(result.text_score for result in selected_results),
                tuple(-index for index in indices),
            )

        coverage_states: Dict[int, tuple[int, ...]] = {0: ()}
        for candidate_index, candidate_mask in enumerate(candidate_masks):
            updated_states = dict(coverage_states)
            for covered_mask, selected_indices in coverage_states.items():
                if len(selected_indices) >= top_k:
                    continue
                next_mask = covered_mask | candidate_mask
                next_indices = selected_indices + (candidate_index,)
                existing = updated_states.get(next_mask)
                if existing is None or subset_quality(next_indices) > subset_quality(existing):
                    updated_states[next_mask] = next_indices
            coverage_states = updated_states
        selected_indices = coverage_states.get(full_coverage_mask)
        if selected_indices is None:
            return []
        selected = [candidates[index][0] for index in selected_indices]
        preferred_scope = (
            "overlaps_query_range" if measure_ranges else "general_evidence"
        )
        if any(
            result.scope_match == preferred_scope for result, _, _ in candidates
        ) and not any(result.scope_match == preferred_scope for result in selected):
            return []
        selected_ids = {result.record["id"] for result in selected}
        for result, _, _ in candidates:
            if len(selected) >= top_k:
                break
            if result.record["id"] not in selected_ids:
                selected.append(result)
                selected_ids.add(result.record["id"])
        return selected


def scope_priority(scope_match: str, has_query_range: bool) -> int:
    if has_query_range:
        return 2 if scope_match == "overlaps_query_range" else 1
    return 1 if scope_match == "general_evidence" else 0


def measure_scope_match(record: Dict[str, Any], query_ranges: List[List[int]]) -> str:
    record_ranges = record.get("measure_range") or []
    if not query_ranges:
        return "general_evidence" if not record_ranges else "local_example"
    if not record_ranges:
        return "global_context"
    if ranges_overlap(record_ranges, query_ranges):
        return "overlaps_query_range"
    return "outside_query_range"


def measure_boost(record: Dict[str, Any], query_ranges: List[List[int]]) -> float:
    if not query_ranges:
        return 1.5 if not (record.get("measure_range") or []) else -0.25
    record_ranges = record.get("measure_range") or []
    if not record_ranges:
        return 0.25
    if ranges_overlap(record_ranges, query_ranges):
        return 6.0 if record.get("measure_scope") == "recurring" else 5.0
    return float("-inf")


def format_measure_range(ranges: Sequence[Sequence[int]]) -> str:
    if not ranges:
        return "general / no specific measure"
    parts = []
    for start, end in ranges:
        parts.append(str(start) if start == end else "%s-%s" % (start, end))
    return ", ".join(parts)


def format_result(result: SearchResult, rank: int) -> str:
    record = result.record
    return (
        "[%d] %s | %s | %s | %s | score %.2f "
        "(text %.2f, scope %.2f)\nQ: %s\nA: %s%s"
        % (
            rank,
            record["id"],
            record.get("evidence_type", "unknown"),
            record["piece"],
            format_measure_range(record.get("measure_range") or []),
            result.score,
            result.text_score,
            result.measure_score,
            record.get("question") or "(standalone tip)",
            record["answer"],
            format_web_sources(record),
        )
    )


def format_web_sources(record: Dict[str, Any]) -> str:
    sources = record.get("sources") or []
    if not sources:
        return ""
    lines = []
    for source in sources:
        lines.append(
            "%s (%s)"
            % (
                source.get("title") or source["web_source_id"],
                source.get("url") or "no URL",
            )
        )
    return "\nSources: " + "; ".join(lines)
