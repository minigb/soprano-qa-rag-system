#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measure-aware lexical retrieval primitives and evidence formatting."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence


TOKEN_RE = re.compile(r"[0-9]+|[A-Za-zÀ-ÖØ-öø-ÿ]+|[가-힣]+")
MUSICAL_SINGLE_LETTER_TOKENS = {"p", "f"}
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
    # In this vocal-score corpus, "피아노 파트" refers to the accompaniment
    # material indexed by the expert answers as "반주" or "반주부".
    "파트": "반주",
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
    re.compile(r"^([가-힣]{2,})할까$"),
    re.compile(r"^([가-힣]{2,})해야$"),
    re.compile(r"^([가-힣]{2,})하$"),
)
CONCEPT_STOP_WORDS = {
    "해야",
    "할까",
    "좋을까",
    "무엇일까",
    "무엇이며",
    "무엇",
    "것이",
    "정도",
    "있을까",
    "경우",
    "혹은",
    "수가",
    "하는",
    "하면",
    "한가",
    "될지",
    "더욱",
    "위해",
    "통해",
    "이유",
    "왜",
    # Question framing and relational words do not identify musical evidence.
    # Keeping them in the full-concept gate makes a natural paraphrase fail
    # merely because an expert used a different interrogative construction.
    "같은",
    "서로",
    "대신",
    "방식",
    "적절",
    "처리",
    "기준",
    "익히",
    "익히려면",
    "곳에",
}
CONCEPT_WORD_NORMALIZATIONS: Dict[str, str] = {}


def register_concept_forms(forms: str, concept: str) -> None:
    for form in forms.split():
        CONCEPT_WORD_NORMALIZATIONS[form] = concept


register_concept_forms(
    "악보 악보마다 스코어 표기 표시 표시된 기보 기보된 유무",
    "악보기보",
)
register_concept_forms("다르면 다를 다를까 다른 다르다 달라 다르게", "차이")
register_concept_forms("어려울 어려운 어렵다 쉽지 쉬운 쉽다", "난도")
register_concept_forms("들리 들릴 들린다 들리는 들리지", "소리")
register_concept_forms("소리", "소리")
register_concept_forms("않은 않은데 않으면 않는 않다 않았지 않았", "않")
register_concept_forms("높으면 높은데 높은 높지", "높")
register_concept_forms("없다면 없는데 없다 없는 없을", "없")
register_concept_forms(
    "부르려면 부르기 부르는 부른다 부른 부를 부를까 불러 불러도 불러야",
    "부르",
)
register_concept_forms("바뀔 바뀌는 바뀐 바뀌", "변화")
register_concept_forms("따라야 따라가야 따라가야할까", "따라")
register_concept_forms("조정하거나 조정해 조정해도", "조정")
register_concept_forms("편곡해 편곡해도", "편곡")
register_concept_forms("될까 되지 잘되지", "되")
register_concept_forms("끊겨 끊기 끊기는 끊어지게 끊어질", "끊")
register_concept_forms("나올까 나오는", "나오")
register_concept_forms("느리게 느린", "느리")
register_concept_forms(
    "자연스럽게 자연스러울까 자연스럽다",
    "자연스럽",
)
register_concept_forms("그어진 그어져있을때 그어져", "그어")
register_concept_forms("맞춰 맞춰서", "맞추")
register_concept_forms(
    "붙여야 붙여 붙이는 붙일지 붙일 붙일까 붙여부를까 위치 두고 배치",
    "배치",
)
register_concept_forms("노래해야", "노래")
register_concept_forms("끌어야", "끌")
register_concept_forms("뒤의", "이후")
register_concept_forms("빠르게", "빠르")
register_concept_forms("나타나는 나타낸다 나타낸 나타내는 나타내", "나타내")
register_concept_forms("뜻일까", "뜻")
register_concept_forms("많지 많은데 많지만", "많")
register_concept_forms("음의 음이 음이나 음은 음을", "음표")
register_concept_forms("길이 길게 끄는", "지속")
register_concept_forms("달라질까 달라지면 달라지는", "변화")
register_concept_forms("이전 처음", "처음")
register_concept_forms("들어갈 들어가면 들어가는 들어가", "들어가")
register_concept_forms("곳에서 곳에서는 곳은 대목", "구간")
register_concept_forms("잡아야 잡아가나요 잡아가 잡는", "박자잡기")
register_concept_forms("성별이나 성별", "성별")
register_concept_forms("제한 제한이 구분", "구분")
register_concept_forms("선율 멜로디", "멜로디")
register_concept_forms("나누는 나누기 나누어 나누", "구분")
register_concept_forms(
    "올라갈 올라가야 올라가야하 올라가는 도약 도약음",
    "도약",
)
register_concept_forms("부족 부족할 모자라 모자라면", "모자라")
register_concept_forms("겹부점 겹점", "겹점")
register_concept_forms("사용한 사용했 사용", "사용")
register_concept_forms("앞에 처음에 처음", "처음")
register_concept_forms("적힌 써있는 써있 기재된", "기재")
register_concept_forms("글은 글", "글")
register_concept_forms("내용일까 내용 이야기 이야기인", "이야기")
register_concept_forms("셋잇단음표 3연음보 연음보 3연음", "연음")
register_concept_forms("빠른 빠르게 빠르", "빠르")
register_concept_forms("박이 박자", "박자")
register_concept_forms("구간 대목", "구간")
register_concept_forms("전반 전체적 전체", "전체")
register_concept_forms("테크닉이나 테크닉", "테크닉")
register_concept_forms("해석 표현", "표현")
register_concept_forms("의미 뭘까 뜻", "의미")
register_concept_forms("바꿀까 바꿔야 바꾸 바꾸어 변환 전환 변화", "변화")

CONCEPT_PARTICLE_RE = re.compile(
    r"([가-힣]{2,}?)(?:에게서|으로는|에서는|부터는|까지는|마다|"
    r"은요|는요|이요|가요|도요|이랑|하고|으로|에서|에게|부터|"
    r"까지|은|는|이|가|을|를|의|와|과|랑|에|도|만|로)\b"
)
TERMINAL_MEANING_REQUEST_RE = re.compile(
    r"(?:무엇|무얼|뭘|뭐)\s*(?:을|를)?\s*"
    r"(?:의미|뜻)(?:하|합|할)"
    r"[가-힣]*\s*[?？.]?\s*$"
)
TERMINAL_REPRESENTATION_REQUEST_RE = re.compile(
    r"(?:무엇|무얼|뭘|뭐)\s*(?:을|를)?\s*"
    r"나타(?:내|냅|낼)"
    r"[가-힣]*\s*[?？.]?\s*$"
)
KOREAN_ANSWER_RELATION_REQUEST_RE = re.compile(
    r"(?:"
    r"(?:무엇|무얼|뭘|뭐)\s*(?:을|를)?\s*"
    r"(?:"
    r"(?:의미|뜻|표현|묘사|상징|시사|암시|반영|전달|말|설명)"
    r"(?:하|한|합|할|해|했)"
    r"|나타(?:내|낸|냅|낼)"
    r"|가리키|보여주|드러내"
    r")"
    r"[가-힣]*"
    r"|(?:무슨|어떤)\s*(?:뜻|의미)"
    r"(?:인가요|인가|인지|일까요|일까|입니까|예요|에요|이죠)?"
    r"(?=\s|[?？.]|$)"
    r"|(?:뜻|의미)(?:은|는|이|가)?\s*(?:무엇|뭐)[가-힣]*"
    r")"
)
KOREAN_ANSWER_RELATION_ROOTS = (
    "의미",
    "뜻",
    "나타내",
    "나타낸",
    "표현",
    "묘사",
    "상징",
    "가리키",
    "보여주",
    "드러내",
    "시사",
    "암시",
    "반영",
    "전달",
    "말하",
    "설명",
)
KOREAN_ANSWER_RELATION_STATEMENT_RE = re.compile(
    r"(?:"
    r"(?:의미|뜻|표현|묘사|상징|시사|암시|반영|전달|말|설명)"
    r"(?:하|한|합|할|해|했|이|인)"
    r"|나타(?:내|낸|냅|낼)"
    r"|가리키|보여주|드러내"
    r")[가-힣]*"
)
CONCEPT_PRODUCTIVE_SUFFIXES = (
    "하려면",
    "하면서",
    "하거나",
    "해서",
    "해도",
    "해야",
    "하게",
    "하는",
    "한다",
    "하다",
    "할까",
    "할지",
)
CONCEPT_PREDICATE_SUFFIXES = (
    "었을까",
    "았을까",
    "을까요",
    "을까",
    "나요",
    "는지",
    "다면",
    "으면",
    "은데",
    "는데",
    "거나",
    "도록",
    "어서",
    "아서",
    "하게",
    "하는",
    "한다",
    "하다",
    "해",
    "게",
    "는",
    "지",
)
ALIAS_SCORE_WEIGHT = 20.0
CONCEPT_COVERAGE_SCORE_WEIGHT = 4.0
MIN_FALLBACK_CONTENT_COVERAGE = 2 / 3
BROAD_PERFORMANCE_SUBJECT_TERMS = (
    "가창",
    "가창자",
    "성악가",
    "노래",
    "부르",
    "불러",
    "singer",
    "sing",
    "perform",
)
BROAD_PERFORMANCE_ADVICE_TERMS = (
    "유의",
    "주의",
    "중요",
    "잘 부르",
    "잘 부를",
    "잘 불러",
    "잘 노래",
    "제대로 소화",
    "무엇을 해야",
    "어떻게 해야",
    "어떻게 부르",
    "어떻게 불러",
    "어떻게 노래",
    "어떤 점",
    "점",
    "점은",
    "점에",
    "신경",
    "포인트",
    "팁",
    "사항",
    "조언",
    "법",
    "알려",
    "focus",
    "give me",
    "advice",
    "keep in mind",
    "consider",
    "tip",
    "what should",
    "how should",
)
BROAD_PERFORMANCE_GENERIC_CONCEPT_PREFIXES = (
    "가창",
    "성악가",
    "노래",
    "부르",
    "입장",
    "관점",
    "유의",
    "주의",
    "중요",
    "가장",
    "전체",
    "전반",
    "신경",
    "써",
    "포인트",
    "팁",
    "사항",
    "조언",
    "법",
    "알려",
    "줘",
    "위한",
    "필요",
    "잘",
    "점",
    "singer",
    "sing",
    "perform",
    "performance",
    "keep",
    "mind",
    "focus",
    "give",
    "work",
    "consider",
    "advice",
    "tip",
    "what",
    "how",
    "main",
    "overall",
    "general",
    "should",
    "this",
    "piece",
)
BROAD_PERFORMANCE_EXPLICIT_PIECE_SCOPE_TERMS = (
    "이 곡",
    "이 노래",
    "이 작품",
    "곡 전체",
    "곡 전반",
    "노래 전체",
    "노래 전반",
    "작품 전체",
    "작품 전반",
    "this piece",
    "this work",
    "this song",
    "the piece",
    "the work",
    "the song",
    "overall",
    "general",
)
PERFORMANCE_GUIDANCE_FACETS = {
    "tone_color": (
        "음색",
        "탄력",
        "밝고",
        "밝은",
        "깔끔",
        "가볍",
    ),
    "vocal_technique": (
        "고음",
        "높은음",
        "저음",
        "낮은음",
        "발성",
        "호흡",
        "숨",
        "공명",
        "음역",
        "포지션",
        "레가토",
        "프레이즈",
        "도약",
        "음정",
        "소리",
        "음색",
        "기교",
        "트릴",
        "콜로라투라",
    ),
    "diction": (
        "발음",
        "딕션",
        "가사",
        "단어",
        "강세",
        "악센트",
        "모음",
        "자음",
        "연음",
        "텍스트",
    ),
    "rhythm_and_timing": (
        "리듬",
        "박자",
        "강박",
        "약박",
        "악센트",
        "템포",
        "진입",
        "페르마타",
        "부점",
        "셋잇단",
        "쉼표",
    ),
    "interpretation": (
        "음악",
        "음악적",
        "해석",
        "표현",
        "분위기",
        "다이나믹",
        "크레센도",
        "디미뉴엔도",
        "피아노",
        "반주",
        "선율",
        "멜로디",
        "프레이즈",
        "가사",
        "이야기",
        "장면",
        "조성",
        "형식",
        "캐릭터",
        "색채",
    ),
}
PERFORMANCE_GUIDANCE_ACTION_TERMS = (
    "주의",
    "유의",
    "중요",
    "연습",
    "집중",
    "고려",
    "생각하며",
    "생각하고",
    "도움",
    "좋다",
    "좋겠다",
    "해야",
    "하도록",
    "유지",
    "살려",
    "정확",
    "명료",
    "노래한다",
    "부른다",
    "부르는 것이",
    "표현한다",
    "표현해야",
    "나타내어",
    "드러내어",
)
PERFORMANCE_GUIDANCE_STRONG_ACTION_TERMS = (
    "주의",
    "유의",
    "중요",
)
PERFORMANCE_GUIDANCE_NEGATIVE_TERMS = (
    "출생",
    "사망",
    "생애",
    "약력",
    "음악사",
    "가곡의 왕",
    "불린다",
    "작곡 연도",
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
        if (
            len(token) == 1
            and not token.isdigit()
            and token not in MUSICAL_SINGLE_LETTER_TOKENS
        ):
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
        token = QUERY_TOKEN_NORMALIZATIONS.get(token, token)
        if (
            len(token) == 1
            and not token.isdigit()
            and token not in MUSICAL_SINGLE_LETTER_TOKENS
        ):
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


def canonical_concept_word(word: str) -> str:
    def finish(value: str) -> str:
        value = QUERY_TOKEN_NORMALIZATIONS.get(value, value)
        value = CONCEPT_WORD_NORMALIZATIONS.get(value, value)
        return "" if value in CONCEPT_STOP_WORDS else value

    word = finish(word)
    if not word:
        return ""
    for suffix in CONCEPT_PRODUCTIVE_SUFFIXES:
        if word.endswith(suffix) and len(word) > len(suffix):
            return finish(word[: -len(suffix)])
    for suffix in CONCEPT_PREDICATE_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 2:
            return finish(word[: -len(suffix)])
    return finish(word)


def semantic_concepts(
    text: str,
    piece: str | None,
    *,
    query: bool,
) -> List[str]:
    """Extract stable concepts for candidate-local semantic coverage."""

    answer_relation_request = bool(
        query and KOREAN_ANSWER_RELATION_REQUEST_RE.search(text.lower())
    )
    if query:
        # Canonicalize common terminal relations before general cleanup removes
        # endings such as "하나요". Relation wording may be soft at retrieval;
        # the remaining musical concepts still use strict coverage.
        text = TERMINAL_MEANING_REQUEST_RE.sub(" 의미 ", text)
        text = TERMINAL_REPRESENTATION_REQUEST_RE.sub(" 나타내 ", text)
        text, _ = query_components(text, piece)
    normalized = CONCEPT_PARTICLE_RE.sub(r"\1", text.lower())
    output = []
    for match in TOKEN_RE.finditer(normalized):
        surface = match.group(0)
        concept = canonical_concept_word(surface)
        # Some surface forms (for example "부를" and "자연스럽게")
        # appear in the broad query-scaffolding allowlist but also have a
        # deliberate domain concept mapping. Preserve those mapped concepts;
        # discard only scaffolding that remained semantically unchanged.
        if (
            query
            and is_query_scaffolding(surface)
            and surface not in QUERY_TOKEN_NORMALIZATIONS
            and surface not in CONCEPT_WORD_NORMALIZATIONS
            and not (
                answer_relation_request
                and is_answer_relation_concept(concept)
            )
        ):
            continue
        if not concept or (
            len(concept) == 1
            and not concept.isdigit()
            and concept not in MUSICAL_SINGLE_LETTER_TOKENS
        ):
            continue
        if concept not in output:
            output.append(concept)
    # A named rhythmic figure already carries the generic "rhythm" concept.
    # Requiring both words would reject an expert answer that says "겹점"
    # without redundantly repeating "리듬".
    if query and "리듬" in output and {"부점", "겹점"}.intersection(output):
        output.remove("리듬")
    return output


def concept_matches(
    query_concept: str,
    document_concepts: set[str],
) -> bool:
    if query_concept in document_concepts:
        return True
    minimum = 2 if re.fullmatch(r"[가-힣]+", query_concept) else 4
    maximum_extension = 2 if re.fullmatch(r"[가-힣]+", query_concept) else 3
    return len(query_concept) >= minimum and any(
        len(document_concept) >= minimum
        and abs(len(query_concept) - len(document_concept)) <= maximum_extension
        and (
            query_concept.startswith(document_concept)
            or document_concept.startswith(query_concept)
        )
        for document_concept in document_concepts
    )


def concepts_are_covered(
    query_concepts: Sequence[str],
    document_concepts: set[str],
) -> bool:
    return bool(query_concepts) and all(
        concept_matches(concept, document_concepts)
        for concept in query_concepts
    )


def concept_coverage(
    query_concepts: Sequence[str],
    document_concepts: set[str],
) -> float:
    """Return the fraction of query concepts supported by one record."""

    if not query_concepts:
        return 0.0
    matched = sum(
        concept_matches(concept, document_concepts)
        for concept in query_concepts
    )
    return matched / len(query_concepts)


def is_answer_relation_concept(concept: str) -> bool:
    """Return whether a normalized concept expresses an answer relation."""

    return any(
        concept.startswith(root) or root.startswith(concept)
        for root in KOREAN_ANSWER_RELATION_ROOTS
    )


def answer_relation_query_concepts(
    query: str,
    query_concepts: Sequence[str],
) -> set[str]:
    """Return Korean answer-relation predicates that may be soft evidence.

    In questions such as ``악센트는 무엇을 상징하는가?``, ``상징하다``
    expresses the requested answer relation rather than the musical subject.
    It should help ranking when present in a record, but a synonymous answer
    need not repeat that exact verb. The interrogative shape keeps noun uses
    such as ``가사의 의미`` on the normal strict path.
    """

    if not KOREAN_ANSWER_RELATION_REQUEST_RE.search(query.lower()):
        return set()
    relation_concepts = [
        concept
        for concept in query_concepts
        if is_answer_relation_concept(concept)
    ]
    # Only the final relation concept belongs to the interrogative predicate.
    # Earlier words such as "표현주의" may be genuine content concepts.
    return {relation_concepts[-1]} if relation_concepts else set()


def has_answer_relation_statement(
    answer: str,
    document_concepts: set[str],
    required_query_concepts: Sequence[str] = (),
) -> bool:
    """Return whether answer content adds a distinct relation concept."""

    return bool(
        KOREAN_ANSWER_RELATION_STATEMENT_RE.search(answer.lower())
    ) and any(
        is_answer_relation_concept(concept)
        for concept in document_concepts
        if not any(
            concept_matches(required, {concept})
            for required in required_query_concepts
        )
    )


def is_broad_performance_guidance_query(
    query: str,
    piece: str | None,
) -> bool:
    """Recognize an intentionally broad request for singer guidance.

    This is deliberately narrower than a generic "performance" keyword
    fallback. A concrete musical concept such as ``고음`` or ``발음`` keeps
    the query on the normal full-concept retrieval path.
    """

    if not piece:
        return False
    lowered = query.lower()
    piece_performance_idiom = "제대로 소화" in lowered
    concepts = semantic_concepts(query, piece, query=True)
    subject_concept_prefixes = (
        "가창",
        "성악가",
        "노래",
        "부르",
        "singer",
        "sing",
        "perform",
    )
    if not (
        any(term in lowered for term in BROAD_PERFORMANCE_SUBJECT_TERMS)
        or piece_performance_idiom
        or any(
            concept.startswith(prefix)
            for concept in concepts
            for prefix in subject_concept_prefixes
        )
    ):
        return False
    if not any(term in lowered for term in BROAD_PERFORMANCE_ADVICE_TERMS):
        return False
    if not concepts:
        return False

    def is_generic(concept: str) -> bool:
        if piece_performance_idiom and concept.startswith(("제대", "소화")):
            return True
        return any(
            concept.startswith(prefix)
            or prefix.startswith(concept)
            for prefix in BROAD_PERFORMANCE_GENERIC_CONCEPT_PREFIXES
        )

    def is_guidance_facet(concept: str) -> bool:
        return any(
            concept.startswith(term)
            or term.startswith(concept)
            for terms in PERFORMANCE_GUIDANCE_FACETS.values()
            for term in terms
        )

    if all(is_generic(concept) for concept in concepts):
        return True
    explicitly_piece_wide = any(
        term in lowered
        for term in BROAD_PERFORMANCE_EXPLICIT_PIECE_SCOPE_TERMS
    ) or any(
        alias.lower() in lowered
        for alias in PIECE_TITLE_ALIASES.get(piece, [])
    )
    return explicitly_piece_wide and all(
        is_generic(concept) or is_guidance_facet(concept)
        for concept in concepts
    )


def performance_guidance_query_facets(query: str) -> set[str]:
    """Return optional musical facets requested by a broad guidance query."""

    lowered = query.lower()
    return {
        facet
        for facet, terms in PERFORMANCE_GUIDANCE_FACETS.items()
        if any(term in lowered for term in terms)
    }


def performance_guidance_query_terms(query: str) -> set[str]:
    """Return explicit facet terms that must remain relevant to the answer."""

    lowered = query.lower()
    return {
        term
        for terms in PERFORMANCE_GUIDANCE_FACETS.values()
        for term in terms
        if term in lowered
    }


def performance_guidance_profile(
    record: Dict[str, Any],
) -> tuple[float, set[str]]:
    """Score practical singer guidance and return its musical facets."""

    text = str(record.get("answer") or "").lower()
    facets = {
        facet
        for facet, terms in PERFORMANCE_GUIDANCE_FACETS.items()
        if any(term in text for term in terms)
    }
    action_terms = {
        term
        for term in PERFORMANCE_GUIDANCE_ACTION_TERMS
        if term in text
    }
    if not facets or not action_terms:
        return 0.0, set()
    strong_action_score = 3.0 * sum(
        term in text
        for term in PERFORMANCE_GUIDANCE_STRONG_ACTION_TERMS
    )
    repeated_range_score = min(
        3.0,
        max(0, len(record.get("measure_range") or []) - 1),
    )
    negative_score = 6.0 * sum(
        term in text
        for term in PERFORMANCE_GUIDANCE_NEGATIVE_TERMS
    )
    score = (
        4.0 * len(facets)
        + 2.0 * min(4, len(action_terms))
        + strong_action_score
        + repeated_range_score
        - negative_score
    )
    return max(0.0, score), facets


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
    alias_score: float = 0.0
    concept_coverage: float = 0.0
    content_concept_coverage: float = 0.0
    answer_relation_score: float = 0.0
    semantic_match_type: str = "strict"
    dense_score: float = 0.0
    dense_content_score: float = 0.0
    fusion_score: float = 0.0
    retrieval_mode: str = "lexical"


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
        self.document_concepts = [
            set(
                semantic_concepts(
                    record.get("relevance_text")
                    or record.get("retrieval_text")
                    or record["answer"],
                    record.get("piece"),
                    query=False,
                )
            )
            for record in records
        ]
        self.content_concepts = [
            set(
                semantic_concepts(
                    record["answer"],
                    record.get("piece"),
                    query=False,
                )
            )
            for record in records
        ]
        self.alias_concepts = [
            [
                set(
                    semantic_concepts(
                        str(alias),
                        record.get("piece"),
                        # Retrieval aliases are authoritative source questions,
                        # so normalize them with the same question scaffolding
                        # and score-location removal used for user queries.
                        # Measure applicability remains a separate hard scope
                        # check and must not dilute semantic alias specificity.
                        query=True,
                    )
                )
                for alias in record.get("retrieval_aliases", [])
                if str(alias).strip()
            ]
            for record in records
        ]
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

    def alias_score(
        self,
        query_concepts: Sequence[str],
        idx: int,
    ) -> float:
        return max(
            (
                min(1.0, len(query_concepts) / max(1, len(alias)))
                for alias in self.alias_concepts[idx]
                if concepts_are_covered(query_concepts, alias)
            ),
            default=0.0,
        )

    @staticmethod
    def _diverse_guidance_order(
        candidates: List[tuple[SearchResult, set[str]]],
    ) -> List[SearchResult]:
        """Prefer candidates that add a new singing-performance facet."""

        remaining = list(candidates)
        selected: List[SearchResult] = []
        covered_facets: set[str] = set()
        while remaining:
            best_index = max(
                range(len(remaining)),
                key=lambda index: (
                    len(remaining[index][1] - covered_facets),
                    remaining[index][0].score,
                    remaining[index][0].text_score,
                    remaining[index][0].record["id"],
                ),
            )
            result, facets = remaining.pop(best_index)
            selected.append(result)
            covered_facets.update(facets)
        return selected

    @staticmethod
    def _expand_guidance_source_siblings(
        selected: List[SearchResult],
        ranked_candidates: List[SearchResult],
        *,
        top_k: int,
    ) -> List[SearchResult]:
        """Keep complementary KUs split from a selected expert source.

        Expert curation may split one broad annotator answer into several
        independent knowledge units (for example tone colour and diction).
        Once one such unit is selected on musical relevance, its same-scope
        siblings remain legitimate complementary evidence.  Expanding only
        along shared immutable source IDs avoids combining arbitrary records
        merely to cover a vague broad question.
        """

        output: List[SearchResult] = []
        seen_ids: set[str] = set()
        for result in selected:
            record_id = result.record["id"]
            if record_id not in seen_ids:
                output.append(result)
                seen_ids.add(record_id)
            source_ids = set(result.record.get("source_ids") or [])
            if source_ids:
                siblings = [
                    sibling
                    for sibling in ranked_candidates
                    if sibling.record["id"] not in seen_ids
                    and sibling.scope_match == result.scope_match
                    and source_ids.intersection(
                        sibling.record.get("source_ids") or []
                    )
                ]
                siblings.sort(
                    key=lambda sibling: (
                        set(sibling.record.get("source_ids") or [])
                        == source_ids,
                        -len(sibling.record.get("source_ids") or []),
                    ),
                    reverse=True,
                )
                for sibling in siblings:
                    sibling_id = sibling.record["id"]
                    output.append(sibling)
                    seen_ids.add(sibling_id)
                    if len(output) >= top_k:
                        return output
            if len(output) >= top_k:
                return output
        return output[:top_k]

    def _search_broad_performance_guidance(
        self,
        *,
        query: str,
        piece: str,
        topic: str | None,
        top_k: int,
    ) -> List[SearchResult]:
        """Return a scope-balanced set of expert singer guidance.

        Whole-piece records establish general claims. Confirmed local records
        are interleaved as measure-bound examples. Pending or unspecified
        records are used only when no confirmed guidance exists.
        """

        if top_k < 1:
            return []
        requested_facets = performance_guidance_query_facets(query)
        requested_terms = performance_guidance_query_terms(query)
        pools: Dict[str, List[tuple[SearchResult, set[str]]]] = {
            "whole": [],
            "local": [],
            "provisional": [],
        }
        for record in self.records:
            if not record.get("retrieval_eligible", True):
                continue
            if record.get("piece") != piece:
                continue
            if record.get("evidence_type") != "expert_annotation":
                continue
            if topic and topic not in record.get("topic", ""):
                continue
            guidance_score, facets = performance_guidance_profile(record)
            if guidance_score <= 0:
                continue
            answer_text = str(record.get("answer") or "").lower()
            matched_requested_terms = {
                term
                for term in requested_terms
                if term in answer_text
            }
            if requested_terms and not matched_requested_terms:
                continue
            matched_requested_facets = facets & requested_facets
            guidance_score += (
                6.0 * len(matched_requested_facets)
                + 4.0 * len(matched_requested_terms)
            )
            scope_match = measure_scope_match(record, [])
            if scope_match == "general_evidence":
                pool = "whole"
            elif scope_match == "local_example":
                pool = "local"
            elif scope_match in {
                "unscoped_pending_review",
                "unspecified_scope",
            }:
                pool = "provisional"
            else:
                continue
            measure = measure_boost(record, [])
            result = SearchResult(
                record=record,
                score=guidance_score + measure,
                text_score=guidance_score,
                measure_score=measure,
                piece_score=1.0,
                scope_match=scope_match,
                # Broad wording is not an authoritative source-question
                # alias. Keep this zero so generation retains a diverse set.
                alias_score=0.0,
                concept_coverage=1.0,
                content_concept_coverage=1.0,
                semantic_match_type="broad_guidance",
            )
            pools[pool].append((result, facets))

        whole = self._diverse_guidance_order(pools["whole"])
        local = self._diverse_guidance_order(pools["local"])
        provisional = self._diverse_guidance_order(
            pools["provisional"]
        )
        confirmed: List[SearchResult] = []
        if whole and local:
            local_limit = min(3, (top_k + 1) // 2)
            local = local[:local_limit]
            whole_index = 0
            local_index = 0
            while len(confirmed) < top_k and (
                whole_index < len(whole)
                or local_index < len(local)
            ):
                if whole_index < len(whole):
                    confirmed.append(whole[whole_index])
                    whole_index += 1
                    if len(confirmed) >= top_k:
                        break
                if local_index < len(local):
                    confirmed.append(local[local_index])
                    local_index += 1
        elif whole:
            confirmed = whole
        elif local:
            confirmed = local[: min(3, top_k)]
        if confirmed:
            return self._expand_guidance_source_siblings(
                confirmed,
                [*whole, *local],
                top_k=top_k,
            )
        return provisional[:top_k]

    def search(
        self,
        query: str,
        piece: str | None = None,
        measure_ranges: List[List[int]] | None = None,
        topic: str | None = None,
        top_k: int = 8,
    ) -> List[SearchResult]:
        measure_ranges = measure_ranges or []
        if (
            not measure_ranges
            and piece
            and is_broad_performance_guidance_query(query, piece)
        ):
            return self._search_broad_performance_guidance(
                query=query,
                piece=piece,
                topic=topic,
                top_k=top_k,
            )
        real_query, intent_anchors = query_components(query, piece)
        real_query_groups = token_groups(real_query)
        query_concepts = semantic_concepts(query, piece, query=True)
        soft_relation_concepts = answer_relation_query_concepts(
            query,
            query_concepts,
        )
        required_query_concepts = [
            concept
            for concept in query_concepts
            if concept not in soft_relation_concepts
        ]
        intent_query_groups = [token_variants(anchor) for anchor in intent_anchors]
        query_groups = real_query_groups + intent_query_groups
        query_terms = list(
            dict.fromkeys(variant for group in query_groups for variant in group)
        )
        if not query_groups:
            return []
        strict_candidates: List[
            tuple[SearchResult, set[int], set[int]]
        ] = []
        relation_fallback_candidates: List[
            tuple[SearchResult, set[int], set[int]]
        ] = []
        for idx, record in enumerate(self.records):
            if not record.get("retrieval_eligible", True):
                continue
            if piece and record["piece"] != piece:
                continue
            if topic and topic not in record["topic"]:
                continue
            scope_match = measure_scope_match(record, measure_ranges)
            if scope_match in {
                "unscoped_pending_context",
                "unspecified_context",
            }:
                # These records remain searchable without a selected range,
                # but they have no confirmed location that can be compared
                # with a range-selected request.
                continue
            candidate_covers_query = concepts_are_covered(
                query_concepts,
                self.document_concepts[idx],
            )
            coverage = concept_coverage(
                query_concepts,
                self.document_concepts[idx],
            )
            content_coverage = concept_coverage(
                required_query_concepts,
                self.content_concepts[idx],
            )
            relation_support = (
                1.0
                if soft_relation_concepts
                and has_answer_relation_statement(
                    record["answer"],
                    self.content_concepts[idx],
                    required_query_concepts,
                )
                else 0.0
            )
            relation_fallback = (
                bool(soft_relation_concepts)
                and bool(required_query_concepts)
                and not candidate_covers_query
                and relation_support > 0
                and content_coverage >= MIN_FALLBACK_CONTENT_COVERAGE
                and concepts_are_covered(
                    required_query_concepts,
                    self.document_concepts[idx],
                )
                and (
                    scope_match == "overlaps_query_range"
                    or len(required_query_concepts) >= 2
                )
            )
            alias = self.alias_score(query_concepts, idx)
            matched_real_groups = {
                group_index
                for group_index, group in enumerate(real_query_groups)
                if any(
                    variant in self.relevance_term_freqs[idx]
                    for variant in group
                )
            }
            matched_intent_groups = {
                group_index
                for group_index, group in enumerate(intent_query_groups)
                if any(
                    variant in self.relevance_term_freqs[idx]
                    for variant in group
                )
            }
            if query_concepts:
                if not candidate_covers_query and not relation_fallback:
                    continue
                matched_real_groups = set(range(len(real_query_groups)))
            if not matched_real_groups and not matched_intent_groups:
                continue
            if relation_fallback and len(matched_intent_groups) != len(
                intent_query_groups
            ):
                continue
            text = self.text_score_terms(query_terms, idx)
            measure = measure_boost(record, measure_ranges)
            piece_boost = 1.0 if piece and record["piece"] == piece else 0.0
            if relation_fallback:
                # A partial source-question alias is never authoritative.
                alias = 0.0
            total = (
                text
                + measure
                + alias * ALIAS_SCORE_WEIGHT
                + (
                    coverage * CONCEPT_COVERAGE_SCORE_WEIGHT
                    if relation_fallback
                    else 0.0
                )
            )
            target_candidates = (
                relation_fallback_candidates
                if relation_fallback
                else strict_candidates
            )
            target_candidates.append(
                (
                    SearchResult(
                        record=record,
                        score=total,
                        text_score=text,
                        measure_score=measure,
                        piece_score=piece_boost,
                        scope_match=scope_match,
                        alias_score=alias,
                        concept_coverage=coverage,
                        content_concept_coverage=content_coverage,
                        answer_relation_score=relation_support,
                        semantic_match_type=(
                            "answer_relation_fallback"
                            if relation_fallback
                            else "strict"
                        ),
                    ),
                    matched_real_groups,
                    matched_intent_groups,
                )
            )
        strict_overlap_exists = any(
            result.scope_match == "overlaps_query_range"
            for result, _, _ in strict_candidates
        )
        fallback_overlap = [
            candidate
            for candidate in relation_fallback_candidates
            if (
                candidate[0].scope_match == "overlaps_query_range"
                and candidate[0].answer_relation_score > 0
            )
        ]
        using_relation_fallback = False
        if measure_ranges and fallback_overlap and not strict_overlap_exists:
            # A confirmed, strongly anchored local paraphrase is more useful
            # than an exact surface-form hit that is only global context.
            candidates = fallback_overlap
            using_relation_fallback = True
        elif strict_candidates:
            candidates = strict_candidates
        elif relation_fallback_candidates:
            best_scope = max(
                scope_priority(candidate[0].scope_match, bool(measure_ranges))
                for candidate in relation_fallback_candidates
            )
            candidates = [
                candidate
                for candidate in relation_fallback_candidates
                if scope_priority(
                    candidate[0].scope_match,
                    bool(measure_ranges),
                )
                == best_scope
            ]
            using_relation_fallback = True
        else:
            candidates = []
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
                candidate[0].alias_score > 0,
                candidate[0].alias_score,
                (
                    candidate[0].answer_relation_score
                    if using_relation_fallback
                    else 0.0
                ),
                (
                    candidate[0].content_concept_coverage
                    if using_relation_fallback
                    else 0.0
                ),
                candidate[0].concept_coverage,
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
                sum(result.alias_score for result in selected_results),
                sum(
                    result.answer_relation_score
                    for result in selected_results
                )
                if using_relation_fallback
                else 0.0,
                sum(
                    result.content_concept_coverage
                    for result in selected_results
                )
                if using_relation_fallback
                else 0.0,
                sum(result.concept_coverage for result in selected_results),
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
        if measure_ranges:
            preferred_scope = "overlaps_query_range"
            if any(
                result.scope_match == preferred_scope
                for result, _, _ in candidates
            ) and not any(
                result.scope_match == preferred_scope
                for result in selected
            ):
                return []
        if using_relation_fallback:
            # Search results are already generation evidence in the current
            # pipeline. Keep the fallback narrow until a separate semantic
            # reranker is introduced.
            return selected[:1]
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
        return {
            "overlaps_query_range": 3,
            "global_context": 2,
            "other_range_context": 1,
            "unscoped_pending_context": 0,
            "unspecified_context": 0,
        }.get(scope_match, 0)
    return {
        # Without a selected range, semantic/source-question specificity
        # remains the primary authority. Scope confidence is expressed by
        # ``measure_boost`` rather than allowing a generic whole-piece record
        # to displace an exact pending annotator-question match.
        "general_evidence": 1,
        "local_example": 1,
        "unscoped_pending_review": 1,
        "unspecified_scope": 1,
    }.get(scope_match, 0)


def scope_evidence_role(scope_match: str) -> str:
    """Return the prompt/output role implied by a scope relationship."""

    return {
        "overlaps_query_range": "selected_range_support",
        "global_context": "general_context",
        "other_range_context": "other_range_context_only",
        "unscoped_pending_context": "unconfirmed_scope_context_only",
        "unspecified_context": "unconfirmed_scope_context_only",
        "general_evidence": "general_support",
        "unscoped_pending_review": "unconfirmed_scope_support",
        "unspecified_scope": "unconfirmed_scope_support",
        "local_example": "local_example",
    }.get(scope_match, "context")


def measure_scope_match(record: Dict[str, Any], query_ranges: List[List[int]]) -> str:
    record_ranges = record.get("measure_range") or []
    measure_status = record.get("measure_status")
    if not query_ranges:
        if measure_status == "waiting_for_review":
            return "unscoped_pending_review"
        if measure_status == "unspecified":
            return "unspecified_scope"
        return "general_evidence" if not record_ranges else "local_example"
    if measure_status == "waiting_for_review":
        return "unscoped_pending_context"
    if measure_status == "unspecified":
        return "unspecified_context"
    if not record_ranges:
        return "global_context"
    if ranges_overlap(record_ranges, query_ranges):
        return "overlaps_query_range"
    return "other_range_context"


def measure_boost(record: Dict[str, Any], query_ranges: List[List[int]]) -> float:
    if not query_ranges:
        if record.get("measure_status") in {
            "waiting_for_review",
            "unspecified",
        }:
            return 0.0
        return 1.5 if not (record.get("measure_range") or []) else 0.75
    if record.get("measure_status") in {
        "waiting_for_review",
        "unspecified",
    }:
        return -1.25
    record_ranges = record.get("measure_range") or []
    if not record_ranges:
        return 0.25
    if ranges_overlap(record_ranges, query_ranges):
        return 6.0 if record.get("measure_scope") == "recurring" else 5.0
    return -1.0


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
        "(text %.2f, dense %.2f, dense-content %.2f, fusion %.2f, scope %.2f, "
        "concepts %.2f, content %.2f, relation %.2f, %s/%s)"
        "\nQ: %s\nA: %s%s"
        % (
            rank,
            record["id"],
            record.get("evidence_type", "unknown"),
            record["piece"],
            format_measure_range(record.get("measure_range") or []),
            result.score,
            result.text_score,
            result.dense_score,
            result.dense_content_score,
            result.fusion_score,
            result.measure_score,
            result.concept_coverage,
            result.content_concept_coverage,
            result.answer_relation_score,
            result.retrieval_mode,
            result.semantic_match_type,
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
