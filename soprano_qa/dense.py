"""Local dense embeddings and hybrid retrieval for the Soprano QA corpus."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import threading
from collections import OrderedDict
from dataclasses import replace
from typing import Any, Dict, List, Protocol, Sequence

from soprano_qa.retrieval import (
    BM25Index,
    PIECE_QUERY_ALIASES,
    SearchResult,
    concept_matches,
    concept_coverage,
    expert_record_is_runtime_final,
    is_broad_performance_guidance_query,
    measure_boost,
    measure_scope_match,
    query_components,
    ranges_overlap,
    scope_priority,
    semantic_concepts,
    strip_question_measure_mentions,
    token_groups,
)


DEFAULT_QUERY_INSTRUCTION = (
    "Given a Korean question about soprano singing and musical scores, "
    "retrieve the relevant evidence passage that answers the question."
)

DENSE_DOMAIN_ANCHOR_RE = re.compile(
    r"(?:"
    r"악보|음악|작품|곡|노래|성악|가창|소프라노|피아노|건반|반주|"
    r"오케스트라|선율|멜로디|리듬|박자|강박|약박|음표|음정|음형|"
    r"화음|화성|조성|악센트|강세|스타카토|트레몰로|페르마타|"
    r"셋잇단음|꾸밈음|보표|마디|가사|발음|모음|호흡|음역|고음|"
    r"작곡|형식|프레이즈|템포|다이나믹|세뇨|코다|카덴차|지시어|"
    r"주법|기호|옥타브|도약|연주|아리아|간주|"
    r"\b(?:coda|segno|dolce|amorosa|ossia|fermata|staccato|"
    r"tremolo|piano|forte|crescendo|decrescendo)\b"
    r")",
    flags=re.IGNORECASE,
)
DENSE_COORDINATED_RELATION_RE = re.compile(
    r"(?P<left>[^?？.,，]{1,40}?)\s*(?:와|과|및|,|，)\s*"
    r"(?P<right>[^?？.,，]{1,40}?)"
    r"(?:"
    r"(?:은|는|이|가)\s*(?:(?:어떤|어떻게)\s*)?"
    r"(?:관련|관계|차이|같|연결|연관)|"
    r"\s*사이(?:의|에)?\s*(?:관련|관계|차이|연결|연관)"
    r")",
    flags=re.IGNORECASE,
)


SOURCE_QUESTION_LED_MATCH_TYPE = "dense_source_question_led"
SOURCE_QUESTION_LED_MIN_RELEVANCE_SCORE = 0.64
SOURCE_QUESTION_LED_MIN_CONTENT_SCORE = 0.40
SOURCE_QUESTION_LED_MIN_OTHER_SOURCE_MARGIN = 0.10
# A source-question embedding identifies an immutable annotation family, but
# it does not by itself identify which split knowledge unit answers the user's
# exact intent.  Let an independently strict sibling displace that rescue
# anchor only when its finalized answer covers materially more of the query
# and its lexical evidence is also stronger.  This keeps the dense rescue for
# paraphrases that have no direct answer-side sibling while preventing a
# procedural split unit from outranking the causal/direct split unit merely
# because both inherited the same source question.
SOURCE_FAMILY_DIRECT_CONTENT_MIN = 0.75
SOURCE_FAMILY_DIRECT_CONTENT_MARGIN = 0.25
SOURCE_FAMILY_DIRECT_CONCEPT_MIN = 0.80
SOURCE_FAMILY_CAUSAL_CONTENT_MARGIN = 0.20
CAUSAL_ANSWER_REQUEST_RE = re.compile(
    r"(?:"
    # Standalone interrogatives only. In particular, do not interpret the
    # first syllable of words such as \"왜곡\" as a why-question.
    r"(?<![가-힣])(?:왜|어째서)(?![가-힣])|"
    r"(?:무슨|어떤|어느)(?:\s+[가-힣A-Za-z]+){0,2}\s*"
    r"(?:이유|원인|까닭|의도|목적|효과|의미|배경|계기|역할|기능)|"
    r"(?:이유|원인|까닭|의도|목적|효과|의미|배경|계기|역할|기능)"
    r"\s*(?:은|는|이|가|을|를|인지|인가|일까|입니까)?\s*"
    r"(?:[,，;；]\s*)?"
    r"(?:무엇|뭐|뭔(?:가|지)?|어떤|어느|궁금|"
    r"있(?:나|어|습니까|을까|는가)|알고\s*싶|알려|"
    r"말해|설명|[?？]|$)|"
    r"(?:무엇|뭐|뭘)\s*때문(?:에|인지|일까|입니까)?|"
    r"(?:(?:무엇|뭐)(?:을|를)|뭘)\s*"
    r"(?:노(?:리|린|렸|리는)|의도)|"
    r"(?:(?:무엇|뭐)(?:을|를)|뭘)\s*위해|"
    r"(?:어디|무엇|뭐).{0,24}(?:에서|로부터).{0,24}"
    r"(?:비롯|유래|기원|가져오|가져온|옮겨오|옮겨온|따오|따온)|"
    r"(?:이유|원인|까닭|의도|목적|효과|의미|배경|계기|역할|기능)"
    r"\s*(?:와|과|및)(?!\s*(?:상관|무관|관계|관련))|"
    r"(?:(?:무엇|뭐)(?:을|를)|뭘|"
    r"(?:무슨(?:\s+\S+){1,3}|어떤(?:\s+\S+){1,3})(?:을|를))\s*"
    r"(?:나타내|표현(?!력)|상징|묘사|암시|그리)|"
    r"\bwhy\b|"
    r"\b(?:what|which)\b.{0,24}"
    r"\b(?:reason|cause|purpose|intent|effect|meaning)\b|"
    r"\b(?:reason|cause|purpose|intent|effect|meaning)\b.{0,24}"
    r"\b(?:what|which|why)\b"
    r")",
    flags=re.IGNORECASE,
)
CAUSAL_REQUEST_IRRELEVANCE_RE = re.compile(
    r"(?:(?:무슨|어떤)\s*)?"
    r"(?:이유|원인|까닭|의도|목적|효과|의미|배경|계기|역할|기능)"
    r"(?:은|는|이|가|을|를|인지|인지는)?\s*"
    r"(?:몰라도|모른\s*채|상관없이|무관하게|관계없이|관련\s*없이)",
    flags=re.IGNORECASE,
)
CAUSAL_ANSWER_SUPPORT_RE = re.compile(
    r"(?:"
    # Grammatical causal relations, rather than bare substrings such as
    # \"이유\" or \"의도\" that may be incidental or negated.
    r"(?:때문에|때문이다|때문입니다|덕분에)|"
    r"[가-힣]+(?:으므로|이므로|므로)|"
    r"[가-힣]{2,}(?:기|도록)\s*(?:위해|위하여|위한|때문)|"
    r"(?:이유|원인|까닭|목적|의도)\s*"
    r"(?:은|는|이|가)\s*.{1,140}"
    r"(?:때문|위해|위한|나타내|표현|만들|살리|부각|고조|"
    r"중요(?:하게)?\s*여(?:기|겼|기는|긴)|중요시|이다|입니다)|"
    r"모습(?:을)?\s*그려.{0,100}"
    r"(?:전환|긴장감|분위기).{0,40}"
    r"(?:만드|만든|만들|나타내)|"
    r"(?:박자|조성|리듬|음형|선율|화성).{0,100}"
    r"(?:바꾸|변화|전환).{0,100}"
    r"(?:나타내|표현|전환|만드|만든|만들|살리|부각|고조)|"
    r"(?:이는|이것은|그것은).{0,120}"
    r"(?:표현|의미|효과).{0,40}"
    r"(?:극대화|나타내|만드|만든|만들|드러내|살리|부각|고조)|"
    r"(?:리듬|박자|조성|음형|선율|화성|악센트|트레몰로|지시|표시)"
    r"(?:은|는|이|가|을|를)?.{0,120}"
    r"(?:효과|의미|모습|성격|분위기|이미지).{0,80}"
    r"(?:만들|나타내|표현|연상|드러내|살리|부각|고조)|"
    r"(?:리듬|박자|조성|음형|선율|화성|악센트|트레몰로|지시|표시)"
    r"(?:은|는|이|가|을|를)?.{0,120}"
    r"(?:나타낸다|나타냅니다|표현한다|표현합니다|상징한다|상징합니다|"
    r"묘사한다|묘사합니다|암시한다|암시합니다)|"
    r"(?:나타낸|표현한|묘사한|상징한)\s*"
    r"(?:것|장치)(?:으로)?\s*(?:볼|해석|이해)|"
    r"(?:의도한|의도된)\s*것(?:으로)?\s*(?:볼|해석|이해)|"
    r"(?:의미로\s*해석|의도일\s*수)|"
    r"(?:것은|것이).{0,120}(?:바꾸|차용|가능성)|"
    r"(?:데에는|데에는|데에|데는).{0,180}"
    r"(?:관련(?:되어|이\s*있)|한몫|기여)|"
    r"(?:꾸밈음|기교|테크닉|음형|선율|리듬|박자|조성).{0,100}"
    r"(?:더\s*잘\s*)?(?:표현|드러내|부각)하는\s*(?:테크닉|장치)|"
    r"\b(?:because|so\s+that|in\s+order\s+to)\b"
    r")",
    flags=re.IGNORECASE | re.DOTALL,
)
PROCEDURAL_ANSWER_REQUEST_RE = re.compile(
    r"(?:어떻게|방법|해야|처리|바로잡|고쳐|고치|어디서|언제|"
    r"[가-힣]+(?:어야|아야|여야)|"
    r"\bhow\b|\bwhat\s+should\b)",
    flags=re.IGNORECASE,
)
MIXED_PROCEDURAL_INTERROGATIVE_RE = re.compile(
    r"(?:"
    r"어떻게|"
    r"방법(?![가-힣])|"
    r"어디서(?![가-힣])|"
    r"언제(?![가-힣])|"
    r"해야|"
    r"[가-힣]+(?:어야|아야|여야)(?![가-힣])|"
    r"\bhow\b|\bwhat\s+should\b"
    r")",
    flags=re.IGNORECASE,
)
MIXED_REQUEST_CLAUSE_BOUNDARY_RE = re.compile(
    r"(?:"
    r"(?:이유|원인|까닭|의도|목적|효과|의미|배경|계기|역할|기능)"
    r"\s*(?:와|과)|"
    r"(?:무엇|뭐)(?:이고|이며)|"
    r"(?:그리고|및)(?=\s)|"
    # Recognize common predicate endings, not a bare ``고`` or ``며``.
    # This deliberately cannot split nouns such as ``사고`` or match a
    # syllable inside ``고음``/``효과``.
    r"(?:했|됐|였|썼|랐|렀|았|었|웠|냈|인|하|되|쓰|넣|놓|꾸)"
    r"(?:고|며|으며)(?=\s|[,，;；])[,，;；]?|"
    # Sentence punctuation is a boundary only after a recognizable terminal
    # predicate. A comma immediately after ``왜`` or ``이유는`` is not one.
    r"[가-힣]{2,}(?:다|까|나요|습니까|했나요|인가요|일까요)"
    r"\s*[?？.!。！]"
    r")\s*",
    flags=re.IGNORECASE,
)
MIXED_NOMINAL_REQUEST_RE = re.compile(
    r"(?:"
    r"(?:이유|원인|까닭|의도|목적|효과|의미|배경|계기|역할|기능)"
    r"\s*(?:와|과|및|그리고|,|，).{0,100}방법|"
    r"방법\s*(?:와|과|및|그리고|,|，).{0,100}"
    r"(?:이유|원인|까닭|의도|목적|효과|의미|배경|계기|역할|기능)"
    r")",
    flags=re.IGNORECASE | re.DOTALL,
)
PROCEDURAL_ANSWER_SUPPORT_RE = re.compile(
    r"(?:해야|하는\s*것이\s*(?:좋|중요)|부르는\s*것이\s*(?:좋|중요)|"
    r"놓아\s*부르|유의|확인해|연주하|노래하|연습하|추천)",
    flags=re.IGNORECASE,
)
DESCRIPTIVE_HOW_REQUEST_RE = re.compile(
    r"어떻게\s*(?:달라|다르|변|바뀌|변경|나타|들리|보이|표현되)",
    flags=re.IGNORECASE,
)
# Keep this aligned with the calibrated semantic-dominance margin used by
# no-range authoritative selection.  A smaller lead is a near-tie, not
# evidence that one unrelated source is safer than the other.
DENSE_ONLY_AMBIGUITY_MARGIN = 0.025
AMBIGUOUS_DENSE_ONLY_RANGE_REASON = "ambiguous_dense_only_range"
SOURCE_FAMILY_MISSING_CAUSAL_AUTHORITY_REASON = (
    "source_family_missing_causal_authority"
)
CAUSAL_ANSWER_NEGATION_RE = re.compile(
    r"(?:"
    r"(?:이유|원인|까닭|의도|목적|효과|의미)"
    r"(?:은|는|이|가|을|를)?\s*"
    r"(?:없|모르|알\s*수\s*없|확인되지\s*않|불분명)|"
    r"(?:이유|의미)\s*없이|의도치\s*않|"
    r"관련(?:되어|돼|이)?\s*있지(?:는)?\s*않|"
    r"관련(?:되어|돼|이)?\s*있(?:는지|다고)?\s*"
    r"(?:알|확인(?:할)?)\s*수\s*없|"
    r"관련(?:되어|돼|이)?\s*있다고\s*"
    r"(?:(?:할|볼|보기|말하)\s*(?:수\s*)?(?:없|어렵|힘들)|"
    r"(?:보|말하)기\s*어렵)|"
    r"관련(?:이|은|는)?\s*없|"
    r"한몫(?:을\s*)?하지\s*(?:않|못)|"
    r"한몫(?:을\s*)?(?:한다고|했다고)\s*"
    r"(?:보|말하)기\s*어렵|"
    r"한몫(?:이|은|는)?\s*아니|"
    r"기여(?:를\s*)?하지\s*(?:않|못)|"
    r"기여(?:를\s*)?(?:한다고|했다고)\s*"
    r"(?:보|말하)기\s*어렵|"
    r"기여(?:가|는)?\s*없|"
    r"가능성(?:이|은|는|도)?\s*(?:전혀\s*|거의\s*)?"
    r"(?:없|낮|희박)|"
    r"가능성(?:을)?\s*배제(?:한다|합니다|했)|"
    r"가능성(?:이|은|는|도)?\s*있다고\s*"
    r"(?:보|말하)기\s*어렵|"
    r"(?:의도적.{0,40}바꾸|차용(?:한|했)).{0,40}"
    r"것(?:은|이)?\s*아니"
    r")",
    flags=re.IGNORECASE,
)
COMPOSITIONAL_CHOICE_REQUEST_RE = re.compile(
    r"(?:쓴|썼|쓰인|쓰였|사용|넣|선택|배치|표시|바꾸|바뀌|차용|"
    r"놓인|놓였|놓은|작곡)",
    flags=re.IGNORECASE,
)
COMPOSITIONAL_CAUSAL_SUPPORT_RE = re.compile(
    r"(?:"
    r"(?:작곡가|로시니|슈베르트|composer).{0,160}"
    r"(?:의도|목적|나타내|표현|상징|묘사|암시|전환|효과)|"
    r"(?:의도|목적|이유|까닭)(?:은|는|이|가|을|를)?.{0,160}"
    r"(?:나타내|표현|상징|묘사|암시|전환|효과|중요(?:하게)?\s*여|중요시)|"
    r"(?:리듬|박자|조성|음형|선율|화성|악센트|트레몰로|지시|표시|"
    r"꾸밈음|부점).{0,160}"
    r"(?:나타내|표현|상징|묘사|암시|분위기.{0,24}전환|"
    r"(?:긴장감|극적).{0,24}(?:만들|높이|고조)|효과.{0,32}(?:만들|내))|"
    r"(?:나타내|표현|상징|묘사|암시).{0,24}기\s*위(?:해|한).{0,80}"
    r"(?:리듬|박자|조성|음형|선율|화성|악센트|트레몰로|지시|표시|"
    r"꾸밈음|부점).{0,40}(?:사용|쓰|넣|배치|표시)|"
    r"(?:표현|효과|분위기|이미지).{0,32}"
    r"(?:만들|살리|부각|고조)기\s*위(?:해|한).{0,100}"
    r"(?:사용|쓰|넣|배치|표시)|"
    r"(?:시대|당시).{0,160}"
    r"(?:중요(?:하게)?\s*여|중요시|유행|관습)|"
    r"(?:이는|이것은|그것은).{0,120}(?:표현|의미|효과).{0,40}"
    r"(?:극대화|나타내|만드|드러내|살리|부각|고조)|"
    r"(?:것은|것이).{0,160}"
    r"(?:의도적.{0,40}바꾸|차용했.{0,40}가능성)"
    r")",
    flags=re.IGNORECASE | re.DOTALL,
)
COMPOSER_AGENT_REQUEST_RE = re.compile(
    r"(?:작곡가|작곡자|composer|로시니|rossini|슈베르트|schubert|"
    r"모리코네|morricone|윤학준|benedict|베네딕트)",
    flags=re.IGNORECASE,
)
COMPOSER_AGENT_SUPPORT_RE = re.compile(
    r"(?:"
    r"(?:작곡가|작곡자|composer|로시니|rossini|슈베르트|schubert|"
    r"모리코네|morricone|윤학준|benedict|베네딕트)(?:은|는|이|가)"
    r".{0,100}(?:사용|쓰|넣|선택|배치|표시|바꾸|차용|놓)"
    r".{0,140}(?:의도|목적|위해|나타내|표현|상징|묘사|암시|전환|효과)|"
    r"(?:작곡가|작곡자|composer|로시니|rossini|슈베르트|schubert|"
    r"모리코네|morricone|윤학준|benedict|베네딕트)의\s*.{0,120}"
    r"(?:사용한\s*것|쓴\s*것|선택한\s*것|배치한\s*것|표시한\s*것)"
    r".{0,140}(?:의도|목적|위해|나타내|표현|상징|묘사|암시|전환|효과)"
    r")",
    flags=re.IGNORECASE,
)
COMPOSER_AGENT_NONAUTHORITY_RE = re.compile(
    r"(?:의도대로.{0,20}(?:않|아니)|않았다면|아니라면|"
    r"의도가.{0,30}(?:있다고\s*생각|있다고\s*가정))",
    flags=re.IGNORECASE,
)
DEFINITIONAL_MEANING_REQUEST_RE = re.compile(
    r"(?:의미|뜻)",
    flags=re.IGNORECASE,
)
DEFINITIONAL_MEANING_SUPPORT_RE = re.compile(
    r"(?:뜻(?:이며|이다|입니다|으로)|의미(?:이며|이다|입니다|로|는|가)|"
    r"(?:사선|표시|기호).{0,100}(?:있으면|없으면)|"
    r"(?:라고|으로)\s*(?:해석|이해))",
    flags=re.IGNORECASE | re.DOTALL,
)
MARKING_INTERPRETATION_REQUEST_RE = re.compile(
    r"(?:"
    r"(?:왜|이유|의미|뜻).{0,100}(?:사선|표시|기호)|"
    r"(?:사선|표시|기호).{0,100}(?:왜|이유|의미|뜻)"
    r")",
    flags=re.IGNORECASE | re.DOTALL,
)
CAUSAL_ANSWER_ADJACENT_EXPLANATION_RE = re.compile(
    r"(?:쉽지\s*않|쉬운\s*곡.{0,20}아니|어렵|난도).{0,220}"
    r"(?:템포가?\s*빠르|빠르게\s*발음|발음해야\s*하는\s*단어가\s*많|"
    r"음역이?\s*높|기교가?\s*많)",
    flags=re.IGNORECASE | re.DOTALL,
)
CAUSAL_PROVENANCE_SUPPORT_RE = re.compile(
    r"(?:"
    # A reviewed explanation may identify where a marking, melody, or other
    # musical material came from without using an explicit 'because'. Keep
    # this to completed provenance predicates so procedural advice such as
    # 'consult the full score' does not become causal authority.
    r"(?:옮겨\s*적(?:은|힌)|옮겨\s*온|옮긴|가져온|따온|본뜬|"
    r"차용(?:한|된)|전사(?:한|된)|유래(?:한|된)|비롯(?:한|된)|"
    r"기원(?:한|한\s*것)|편곡(?:한|된)|변형(?:한|된))"
    r"[^.!?。！？\n]{0,56}"
    r"(?:것(?:으로)?\s*(?:보|볼|본|해석|이해)|것이다|것입니다|"
    r"결과|때문)|"
    r"(?:에서|으로부터)\s*(?:비롯|유래|기원)"
    r")",
    flags=re.IGNORECASE,
)


def _is_finalized_expert_record(record: Dict[str, Any]) -> bool:
    """Return whether a record can answer authoritatively from expert review."""

    return expert_record_is_runtime_final(record)


def _is_finalized_expert_answer(result: SearchResult) -> bool:
    return _is_finalized_expert_record(result.record)


def _source_family_key(record: Dict[str, Any]) -> tuple[str, ...]:
    return tuple(sorted(set(record.get("source_ids") or [])))


def is_causal_answer_request(query: str) -> bool:
    """Return whether the question explicitly asks for explanation/meaning."""

    relevant_query = CAUSAL_REQUEST_IRRELEVANCE_RE.sub(" ", query)
    return bool(CAUSAL_ANSWER_REQUEST_RE.search(relevant_query))


def _record_has_causal_answer_authority(record: Dict[str, Any]) -> bool:
    if not _is_finalized_expert_record(record):
        return False
    answer = str(record.get("answer") or "")
    if CAUSAL_ANSWER_NEGATION_RE.search(answer):
        return False
    if CAUSAL_ANSWER_ADJACENT_EXPLANATION_RE.search(answer):
        return True
    sentences = re.split(r"(?<=[.!?。！？])\s+|[\r\n]+", answer)
    return any(
        (
            CAUSAL_ANSWER_SUPPORT_RE.search(sentence)
            or CAUSAL_PROVENANCE_SUPPORT_RE.search(sentence)
        )
        for sentence in sentences
        if sentence.strip()
    )


def _record_has_composer_agent_authority(record: Dict[str, Any]) -> bool:
    if not _is_finalized_expert_record(record):
        return False
    answer = str(record.get("answer") or "")
    for sentence in re.split(r"(?<=[.!?。！？])\s+|[\r\n]+", answer):
        if (
            sentence.strip()
            and not COMPOSER_AGENT_NONAUTHORITY_RE.search(sentence)
            and COMPOSER_AGENT_SUPPORT_RE.search(sentence)
        ):
            return True
    return False


def _meaning_answer_authority(
    result: SearchResult,
    query: str,
) -> bool:
    answer = str(result.record.get("answer") or "")
    if not DEFINITIONAL_MEANING_SUPPORT_RE.search(answer):
        return False
    piece = str(result.record.get("piece") or "") or None
    meaning_end = max(
        (
            match.end()
            for match in re.finditer(r"(?:의미|뜻)", query)
        ),
        default=len(query),
    )
    meaning_start = 0
    for connector in re.finditer(
        r"(?:와|과|및|그리고|,|，)\s*",
        query[:meaning_end],
    ):
        meaning_start = connector.end()
    focused_query = query[meaning_start:meaning_end]
    query_concepts = [
        concept
        for concept in semantic_concepts(
            focused_query,
            piece,
            query=True,
        )
        if concept not in {
            "의미",
            "뜻",
            "어떻게",
            "방법",
            "노래",
            "성악",
            "표현하",
        }
    ]
    if not query_concepts:
        return True
    answer_concepts = set(semantic_concepts(answer, piece, query=False))
    return concept_coverage(query_concepts, answer_concepts) >= 0.5


def _procedural_request_match(query: str) -> re.Match[str] | None:
    """Find an instructional request, excluding descriptive ``how`` uses."""

    instructional_view = DESCRIPTIVE_HOW_REQUEST_RE.sub(
        lambda match: " " * len(match.group(0)),
        query,
    )
    marker = PROCEDURAL_ANSWER_REQUEST_RE.search(instructional_view)
    if marker is None or not is_causal_answer_request(query):
        return marker
    # In a question such as "왜 ... 공부해야 할까?", the necessity form is
    # the predicate whose reason is requested, not a second request for a
    # procedure. A mixed request must contain a distinct instructional
    # interrogative separated from a causal clause. Do not use bare Korean
    # syllables as coordinators: ``고`` in ``고음`` and ``과`` in ``효과`` are not
    # clause boundaries.
    if MIXED_NOMINAL_REQUEST_RE.search(instructional_view):
        return marker
    causal_matches = list(CAUSAL_ANSWER_REQUEST_RE.finditer(query))
    for procedural_match in MIXED_PROCEDURAL_INTERROGATIVE_RE.finditer(
        instructional_view
    ):
        for causal_match in causal_matches:
            if procedural_match.start() >= causal_match.end():
                between = query[
                    causal_match.start() : procedural_match.start()
                ]
            elif causal_match.start() >= procedural_match.end():
                between = query[
                    procedural_match.end() : causal_match.end()
                ]
            else:
                continue
            if MIXED_REQUEST_CLAUSE_BOUNDARY_RE.search(between):
                return procedural_match
    return None


def _procedural_request_clause(query: str) -> str:
    marker = _procedural_request_match(query)
    if marker is None:
        return ""
    clause_start = 0
    for connector in MIXED_REQUEST_CLAUSE_BOUNDARY_RE.finditer(query):
        if connector.end() <= marker.start():
            clause_start = connector.end()
    return query[clause_start:]


def _procedural_answer_authority(
    result: SearchResult,
    query: str,
) -> bool:
    answer = str(result.record.get("answer") or "")
    if not PROCEDURAL_ANSWER_SUPPORT_RE.search(answer):
        return False
    clause = _procedural_request_clause(query)
    if not clause:
        return False
    clause = re.sub(r"숨(?:을|이|은|에서)?", "호흡", clause)
    piece = str(result.record.get("piece") or "") or None
    generic_concepts = {
        "어떻게",
        "방법",
        "해야",
        "처리",
        "적용",
        "바로잡",
        "고치",
        "어디서",
        "언제",
        "쉬어야",
        "성악",
        "노래",
        "부르",
        "이를",
        "이것",
        "한다",
        "좋다",
        "표현하",
    }
    requested_concepts = [
        concept
        for concept in semantic_concepts(clause, piece, query=True)
        if concept not in generic_concepts
    ]
    if not requested_concepts:
        return True
    answer_concepts = set(semantic_concepts(answer, piece, query=False))
    # A mixed question asks for every named procedural facet.  Partial
    # overlap (for example, an answer mentioning the lyrics but saying
    # nothing about pronunciation) cannot authorize the missing instruction.
    return concept_coverage(requested_concepts, answer_concepts) >= 1.0


def has_causal_answer_authority(
    result: SearchResult,
    query: str = "",
) -> bool:
    """Return whether a finalized expert answer states a causal relation."""

    if (
        query
        and is_causal_answer_request(query)
    ):
        record = result.record
        if not _is_finalized_expert_record(record):
            return False
        answer = str(record.get("answer") or "")
        if CAUSAL_ANSWER_NEGATION_RE.search(answer):
            return False
        if COMPOSER_AGENT_REQUEST_RE.search(query):
            return _record_has_composer_agent_authority(record)
        if (
            DEFINITIONAL_MEANING_REQUEST_RE.search(query)
            and _meaning_answer_authority(result, query)
        ):
            return True
        if COMPOSITIONAL_CHOICE_REQUEST_RE.search(query):
            return bool(
                COMPOSITIONAL_CAUSAL_SUPPORT_RE.search(answer)
                or CAUSAL_PROVENANCE_SUPPORT_RE.search(answer)
            )
    return _record_has_causal_answer_authority(result.record)


def _authorities_cover_measure_ranges(
    authorities: Sequence[SearchResult],
    measure_ranges: Sequence[Sequence[int]],
) -> bool:
    return all(
        any(
            not (authority.record.get("measure_range") or [])
            or ranges_overlap(
                authority.record.get("measure_range") or [],
                [requested_range],
            )
            for authority in authorities
        )
        for requested_range in measure_ranges
    )


def _filter_causal_answer_authority(
    query: str,
    candidates: Sequence[SearchResult],
    *,
    measure_ranges: Sequence[Sequence[int]] = (),
) -> tuple[List[SearchResult], bool]:
    """Remove intent-incompatible expert answers from every retrieval route.

    A lexical or ordinary dense hit is relevance evidence, not permission to
    reinterpret procedural advice as a composer's reason or an effect.  When
    the only retrieved evidence is finalized expert text that does not state
    the requested causal relation, signal a fail-closed abstention.
    """

    results = list(candidates)
    if not is_causal_answer_request(query):
        return results, False
    procedural_request = _procedural_request_match(query) is not None
    finalized_experts = [
        result for result in results if _is_finalized_expert_answer(result)
    ]
    if not finalized_experts:
        return results, False
    ranged = bool(measure_ranges)
    scoped_experts = (
        [
            result
            for result in finalized_experts
            if result.scope_match == "overlaps_query_range"
        ]
        if ranged
        else finalized_experts
    )
    if not scoped_experts:
        return results, False
    anchor = scoped_experts[0]
    anchor_sources = set(anchor.record.get("source_ids") or [])

    def belongs_to_anchor_family(result: SearchResult) -> bool:
        result_sources = set(result.record.get("source_ids") or [])
        return result is anchor or bool(anchor_sources.intersection(result_sources))

    causal_authorities = [
        result
        for result in scoped_experts
        if belongs_to_anchor_family(result)
        and has_causal_answer_authority(result, query)
    ]
    if not causal_authorities:
        # Never replace the leading answer family with an unrelated lower
        # causal-looking hit. An implicit or technique-only answer cannot
        # authorize a causal claim; abstention is safer than reinterpretation.
        return [], True
    if ranged and not _authorities_cover_measure_ranges(
        causal_authorities,
        measure_ranges,
    ):
        # ``overlaps_query_range`` means "overlaps at least one range".  It
        # cannot authorize an answer about multiple named score locations
        # when one or more of those locations have no causal evidence.
        return [], True
    meaning_authorities = [
        result
        for result in scoped_experts
        if belongs_to_anchor_family(result)
        and (
            DEFINITIONAL_MEANING_REQUEST_RE.search(query)
            or MARKING_INTERPRETATION_REQUEST_RE.search(query)
        )
        and _meaning_answer_authority(result, query)
    ]
    if procedural_request:
        # A mixed why-and-how question may retain both reviewed facets, but
        # only after proving causal authority in the same effective scope and
        # source family. Preserve the compatible procedural siblings too.
        family_experts = [
            result
            for result in scoped_experts
            if belongs_to_anchor_family(result)
        ]
        procedural_authorities = [
            result
            for result in family_experts
            if _procedural_answer_authority(result, query)
        ]
        if not procedural_authorities or (
            ranged
            and not _authorities_cover_measure_ranges(
                procedural_authorities,
                measure_ranges,
            )
        ):
            return [], True
        ordered_experts = [
            *causal_authorities,
            *(
                result for result in meaning_authorities
                if result not in causal_authorities
            ),
            *(
                result for result in procedural_authorities
                if result not in causal_authorities
                and result not in meaning_authorities
            ),
        ]
    else:
        ordered_experts = [
            *causal_authorities,
            *(
                result for result in meaning_authorities
                if result not in causal_authorities
            ),
        ]
    non_experts = [
        result
        for result in results
        if not _is_finalized_expert_answer(result)
    ]
    return [*ordered_experts, *non_experts], False


def _source_family_has_same_scope_causal_authority(
    anchor: SearchResult,
    candidates: Sequence[SearchResult],
    query: str,
) -> bool:
    """Require causal support from the exact finalized family and scope."""

    anchor_sources = set(anchor.record.get("source_ids") or [])
    return bool(anchor_sources) and any(
        result.scope_match == anchor.scope_match
        and set(result.record.get("source_ids") or []) == anchor_sources
        and _is_finalized_expert_answer(result)
        and has_causal_answer_authority(result, query)
        for result in candidates
    )


def source_family_answer_authority(
    anchor: SearchResult,
    candidates: Sequence[SearchResult],
    *,
    query: str = "",
    known_causal_family: bool = False,
) -> SearchResult | None:
    """Choose a materially stronger direct answer within one source family.

    ``dense_source_question_led`` is deliberately a recall/rescue signal.  A
    strict sibling from the exact same immutable source set and scope has
    stronger answer authority when both its answer-side concept coverage and
    lexical score materially beat the rescue anchor.  Unrelated records,
    partial source overlaps, and near ties cannot take over. Dense-only
    promotion is limited to an unambiguous causal-purpose sibling for an
    explicitly causal request.
    """

    anchor_sources = set(anchor.record.get("source_ids") or [])
    if not anchor_sources:
        return anchor
    causal_request = is_causal_answer_request(query)
    family_has_causal_answer = known_causal_family or any(
        set(result.record.get("source_ids") or []) == anchor_sources
        and has_causal_answer_authority(result, query)
        for result in candidates
    )
    if (
        causal_request
        and family_has_causal_answer
        and not _source_family_has_same_scope_causal_authority(
            anchor,
            candidates,
            query,
        )
    ):
        # A source-question alias establishes annotation-family recall, not
        # answer intent.  If the selected scope contains only procedural or
        # otherwise non-causal split units, emitting one of them as the reason
        # would change the curated meaning.  Abstain instead of borrowing a
        # causal sibling whose finalized measure scope does not apply.
        return None
    if causal_request and not family_has_causal_answer:
        # Do not reject a finalized review merely because its explanation is
        # implicit. The strict relation guard is activated only when this
        # immutable family contains a contrasting unit with explicit causal
        # authority.
        return anchor
    if (
        query
        and not causal_request
        and _procedural_request_match(query)
        and PROCEDURAL_ANSWER_SUPPORT_RE.search(
            str(anchor.record.get("answer") or "")
        )
    ):
        return anchor
    strict_eligible = [
        result
        for result in candidates
        if (
            result is not anchor
            and result.semantic_match_type == "strict"
            and result.scope_match == anchor.scope_match
            and set(result.record.get("source_ids") or [])
            == anchor_sources
            and result.concept_coverage
            >= SOURCE_FAMILY_DIRECT_CONCEPT_MIN
            and result.content_concept_coverage
            >= SOURCE_FAMILY_DIRECT_CONTENT_MIN
            and result.content_concept_coverage
            >= (
                anchor.content_concept_coverage
                + SOURCE_FAMILY_DIRECT_CONTENT_MARGIN
            )
            and result.text_score > anchor.text_score
            and (
                not causal_request
                or has_causal_answer_authority(result, query)
            )
        )
    ]
    if strict_eligible:
        return max(
            strict_eligible,
            key=lambda result: (
                result.content_concept_coverage,
                result.concept_coverage,
                result.text_score,
                result.dense_content_score,
                result.dense_score,
                result.record["id"],
            ),
        )

    # Dense paraphrases may leave every split unit below the lexical strict
    # gate.  For an explicit causal/purpose request, the same immutable family
    # can still resolve answer intent safely when exactly one sibling contains
    # a direct causal-purpose statement and the rescue anchor does not.  This
    # is not a general dense rerank: answer-side coverage must improve by the
    # same material margin and curated-answer similarity must remain adequate.
    if (
        not query
        or not causal_request
        or has_causal_answer_authority(anchor, query)
    ):
        return anchor
    causal_eligible = [
        result
        for result in candidates
        if (
            result is not anchor
            and result.scope_match == anchor.scope_match
            and set(result.record.get("source_ids") or [])
            == anchor_sources
            and has_causal_answer_authority(result, query)
            and result.content_concept_coverage
            >= (
                anchor.content_concept_coverage
                + SOURCE_FAMILY_CAUSAL_CONTENT_MARGIN
            )
            and result.dense_content_score
            >= SOURCE_QUESTION_LED_MIN_CONTENT_SCORE
        )
    ]
    if len(causal_eligible) != 1:
        # The anchor has already been shown not to answer the causal intent.
        # A weak or ambiguous causal sibling cannot safely authorize it.
        return None
    return max(
        causal_eligible,
        key=lambda result: (
            result.content_concept_coverage,
            result.concept_coverage,
            result.text_score,
            result.dense_content_score,
            result.dense_score,
            result.record["id"],
        ),
    )


def _constraint(
    name: str,
    request: str,
    support: str,
) -> tuple[str, re.Pattern[str], re.Pattern[str]]:
    return (
        name,
        re.compile(request, flags=re.IGNORECASE | re.DOTALL),
        re.compile(support, flags=re.IGNORECASE | re.DOTALL),
    )


_SUNG_TEXT_TARGET = r"(?:모음|가사|텍스트|음절|글자|발음|단어)"
_SUNG_TEXT_REQUESTED_SLOT = (
    r"(?:"
    r"(?:어느|어떤|무슨)\s*(?:음(?:표|정)?|박(?:자)?)|"
    r"몇\s*(?:(?:번|번째)\s*)?(?:음(?:표|정)?|박(?:자)?)|"
    r"(?:음(?:표|정)?|박(?:자)?).{0,10}"
    r"(?:어디|어느\s*곳|어떤\s*위치)|"
    r"(?:어디|어느\s*곳|어떤\s*위치|무슨\s*위치)"
    r")"
)
_SUNG_TEXT_PLACEMENT_ACTION = (
    r"(?:놓|두|붙|배치|얹|싣|실|시작|위치|들어가|노래하|노래해|부르|불러)"
)
SUNG_TEXT_PLACEMENT_REQUEST = (
    rf"(?:{_SUNG_TEXT_TARGET}).{{0,48}}"
    rf"(?:{_SUNG_TEXT_REQUESTED_SLOT}).{{0,32}}"
    rf"(?:{_SUNG_TEXT_PLACEMENT_ACTION})|"
    rf"(?:{_SUNG_TEXT_REQUESTED_SLOT}).{{0,48}}"
    rf"(?:{_SUNG_TEXT_TARGET}).{{0,32}}"
    rf"(?:{_SUNG_TEXT_PLACEMENT_ACTION})|"
    rf"(?:{_SUNG_TEXT_TARGET}).{{0,14}}(?:의\s*)?위치"
    r".{0,24}(?:어떻|어디|어느|어떤|정해|되어야|될까|인가)"
)
_SUNG_TEXT_NAMED_SLOT = (
    r"(?:"
    r"첫\s*머리|첫머리|처음|시작(?:점|부분)?|"
    r"(?:(?:첫|두|둘째|세|셋째|네|넷째|다섯|여섯|일곱|여덟|아홉|"
    r"[1-9])\s*(?:(?:번|번째)\s*)?"
    r"(?:음(?:표|정)?|박(?:자)?))|"
    r"제\s*[1-9]\s*(?:음(?:표|정)?|박(?:자)?)|"
    r"(?:꾸밈음|장식음)|"
    r"(?:가온\s*)?(?:도|레|미|파|솔|라|시)\s*(?:음|음표)?|"
    r"(?<![A-Za-z])[A-G][#♯b♭]?[0-8]?\s*(?:음|음표)?"
    r")"
)
_SUNG_TEXT_NAMED_SLOT_LOCATION = (
    rf"(?:{_SUNG_TEXT_NAMED_SLOT})"
    r"(?:\s*\([^)]{1,30}\))?\s*(?:에|에서|부터|로|위에)"
)
_SUNG_TEXT_PLACEMENT_SUPPORT_ACTION = (
    r"(?:놓|두|붙|배치|얹|싣|실|시작|발음|노래하|부르)"
)
SUNG_TEXT_PLACEMENT_SUPPORT = (
    rf"(?:{_SUNG_TEXT_NAMED_SLOT_LOCATION}).{{0,32}}"
    rf"(?:{_SUNG_TEXT_TARGET}).{{0,20}}"
    rf"(?:{_SUNG_TEXT_PLACEMENT_SUPPORT_ACTION})|"
    rf"(?:{_SUNG_TEXT_TARGET}).{{0,32}}"
    rf"(?:{_SUNG_TEXT_NAMED_SLOT_LOCATION}).{{0,20}}"
    rf"(?:{_SUNG_TEXT_PLACEMENT_SUPPORT_ACTION})|"
    rf"(?:{_SUNG_TEXT_TARGET}).{{0,20}}"
    rf"(?:{_SUNG_TEXT_PLACEMENT_SUPPORT_ACTION}).{{0,32}}"
    rf"(?:{_SUNG_TEXT_NAMED_SLOT})(?:이|가)?\s*"
    r"(?:맞|적절|좋|이다|입니다)"
)


def _has_sung_text_placement_relation(answer: str) -> bool:
    """Require the answer to place sung text, not merely name a note slot.

    A short answer may omit an already-understood subject (for example,
    ``첫 번째 음표에서 시작합니다.``).  Limit that ellipsis to one concise
    placement clause so a longer passage about an unrelated action at a named
    note cannot accidentally satisfy the requested text-to-note relation.
    """

    if re.search(
        SUNG_TEXT_PLACEMENT_SUPPORT,
        answer,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        return True
    concise_answer = answer.strip()
    if len(concise_answer) > 40 or "\n" in concise_answer:
        return False
    return bool(
        re.fullmatch(
            rf"\s*(?:{_SUNG_TEXT_NAMED_SLOT_LOCATION})"
            r"[^.!?。！？\n]{0,18}"
            r"(?:놓|두|붙|배치|얹|싣|실|시작)"
            r"[가-힣\s]{0,16}[.!?。！？]?\s*",
            concise_answer,
            flags=re.IGNORECASE,
        )
    )


DENSE_REQUESTED_CONSTRAINTS = (
    _constraint(
        "sung_text_placement",
        SUNG_TEXT_PLACEMENT_REQUEST,
        SUNG_TEXT_PLACEMENT_SUPPORT,
    ),
    _constraint(
        "recording_score_mismatch",
        r"(?=.*(?:음원|녹음(?:본|물)?|음반|오디오|recording|audio))"
        r"(?=.*(?:악보|총보|보표|score))"
        r"(?=.*(?:다르|달라|차이|불일치|일치하지|맞지\s*않|"
        r"같지\s*않|어긋|엇갈|"
        r"(?:있|표시되|들리).{0,100}(?:없|표시되지|들리지)|"
        r"(?:없|표시되지|들리지).{0,100}(?:있|표시되|들리)))",
        r"(?:"
        r"(?=.*(?:음원|녹음(?:본|물)?|음반|오디오|recording|audio))"
        r"(?=.*(?:악보|총보|보표|score|판본|버전|편곡))"
        r"(?=.*(?:따르|기준|우선|선택|정하|결정|확인|계획|"
        r"생략|반복|허용|가능|어떤\s*형태|목적에\s*따라))|"
        r"(?=.*(?:판본|버전|편곡|version|arrangement))"
        r"(?=.*(?:차이|다르|달라|유무|있(?:는|고)?.{0,40}없|"
        r"없(?:는|고)?.{0,40}있))"
        r"(?=.*(?:허용|가능|선택|따라|정하|결정|확인|계획))"
        r")",
    ),
    _constraint(
        "video_score_mismatch",
        r"(?=.*(?:영상|유튜브|비디오|video|youtube))"
        r"(?=.*(?:악보|총보|보표|score))"
        r"(?=.*(?:다르|다를|달라|차이|불일치|맞지\s*않|기준|"
        r"무엇을\s*따라|어떤\s*것을\s*따라))",
        r"(?=.*(?:영상|유튜브|비디오|video|youtube))"
        r"(?:"
        r"(?=.*(?:하나의\s*정답|정답))(?=.*(?:없|아니))|"
        r"(?=.*(?:악보|총보|보표|score|판본|버전|가창자|연주자|변주))"
        r"(?=.*(?:따르|기준|우선|선택|정하|결정|판단|해석|허용|가능))"
        r")",
    ),
    _constraint(
        "score_front_matter",
        r"(?=.*(?:악보|스코어|총보|score))"
        r"(?=.*(?:맨\s*앞|가장\s*(?:먼저|처음(?:에)?)|앞(?:부분|쪽|에|서)|"
        r"본문(?:보다|의)?\s*앞|처음|첫\s*(?:장|페이지|머리)|"
        r"시작하기\s*전|서두|서문|머리말|표제부|front\s*matter))"
        r"(?=.*(?:글|문구|문장|텍스트|내용|설명|안내))",
        r"(?:"
        r"(?:(?:악보|스코어|score)[^.!?\n]{0,32}"
        r"(?:맨\s*앞|가장\s*(?:먼저|처음(?:에)?)|앞부분|앞쪽|본문\s*앞|처음|"
        r"첫\s*(?:장|페이지|머리)|서두|서문|머리말|표제부)|"
        r"front\s*matter|지문|scene|서문|머리말|표제부)"
        r"[^.!?\n]{0,96}"
        r"(?:제목|곡명|작품명|장면|무대|배경|상황|등장인물|인물|"
        r"줄거리|헌사|작곡가|작사가|대본|시대|장소|시간|맥락|"
        r"집|창문|편지)|"
        r"(?:제목|곡명|작품명|장면|무대|배경|상황|등장인물|인물|"
        r"줄거리|헌사|작곡가|작사가|대본|시대|장소|시간|맥락|"
        r"집|창문|편지)"
        r"[^.!?\n]{0,96}"
        r"(?:(?:악보|스코어|score)[^.!?\n]{0,32}"
        r"(?:맨\s*앞|가장\s*(?:먼저|처음(?:에)?)|앞부분|앞쪽|본문\s*앞|처음|"
        r"첫\s*(?:장|페이지|머리)|서두|서문|머리말|표제부)|"
        r"front\s*matter|지문|scene|서문|머리말|표제부)"
        r")",
    ),
    _constraint(
        "stage_atmosphere",
        r"(?:어떤|무슨|어떠한)\s*분위기의\s*(?:무대|장소)|"
        r"(?:무대|장소)(?:의)?\s*분위기.{0,24}(?:어떻|어떤|무슨)",
        r"(?:무대\s*구성|무대\s*배경|지문|scene|"
        r"배경.{0,24}(?:집|창문|장소|무대)|"
        r"(?:집|창문|장소).{0,40}(?:닫|열|편지|인물))",
    ),
    _constraint(
        "sung_text_technique_attachment",
        r"(?=.*(?:가사|텍스트|음절|단어|모음))"
        r"(?=.*(?:기교|테크닉|스케일|트릴|콜로라투라))"
        r"(?=.*(?:붙|배치|연결|얹|싣|실어|이어))",
        r"(?=.*(?:가사|텍스트|음절|단어|모음))"
        r"(?=.*(?:기교|테크닉|스케일|트릴|콜로라투라|"
        r"3\s*연음(?:보)?|셋잇단(?:음표)?|패시지|음형))"
        r"(?=.*(?:붙|배치|연결|얹|싣|실어|이어|끝맺|프레이즈|"
        r"첫머리|모음.{0,24}유지))",
    ),
    _constraint(
        "vocal_character_description",
        r"(?=.*(?:부르|불러|부를|노래|표현|가창|소리))"
        r"(?=.*(?:"
        r"(?:어떤|무슨|어떠한|어느)\s*"
        r"(?:음색|톤|느낌|분위기|성격|캐릭터|색채)(?!\s*의)|"
        r"(?:음색|톤|느낌|분위기|성격|캐릭터|색채)"
        r"(?:은|는|이|가|으로|로)?\s*"
        r"(?:무엇|어떻|어떤|무슨)"
        r"))",
        r"(?:음색|느낌|분위기|성격|캐릭터|색채|부르|노래|표현)",
    ),
    _constraint(
        "ornament_execution",
        r"(?=.*(?:꾸밈음|장식음|grace\s*note|ornament))"
        r"(?:"
        r"(?=.*(?:처리|다루|연주법|창법|부르는\s*법|노래하는\s*법))|"
        r"(?=.*어떻게)(?=.*(?:부르|노래|연주|해야|할까))|"
        r"(?=.*(?:부를\s*때|노래할\s*때|연주할\s*때))"
        r")",
        r"(?=.*(?:꾸밈음|장식음|grace\s*note|ornament))"
        r"(?=.*(?:"
        r"사선.{0,40}(?:원음|앞|미리|있|없)|"
        r"(?:꾸밈음|장식음).{0,48}(?:원음|균등|박자.{0,20}나누)|"
        r"원음.{0,32}(?:앞|미리|나누|균등)|"
        r"(?:꾸밈음|장식음|grace\s*note|ornament)"
        r"[^.!?\n]{0,48}"
        r"(?:가볍|빠르|빠르게|짧|길게|부드럽|또렷|정확|균등|"
        r"박자|리듬|원음|앞|뒤)"
        r"[^.!?\n]{0,32}"
        r"(?:노래|부르|연주|처리|나누|붙|이어|시작)"
        r"))",
    ),
    _constraint(
        "accent_placement",
        r"(?:악센트|강세).{0,30}"
        r"(?:첫\s*(?:번째\s*)?박|다른\s*박|박자|위치|곳|놓|표시)|"
        r"(?:첫\s*(?:번째\s*)?박|다른\s*박|박자|위치|곳)"
        r".{0,30}(?:악센트|강세)",
        r"(?:강박|약박)|"
        r"(?:첫|두|세|네|다섯|여섯|일곱|여덟|아홉|[1-9])\s*"
        r"(?:(?:번|번째)\s*)?박(?:자)?",
    ),
    _constraint(
        "fingering",
        r"(?:손가락\s*(?:번호|순서)|운지(?:법|번호)?|핑거링).{0,30}"
        r"(?:무엇|어떻|어느|몇|추천|권장|알려|인가)|"
        r"(?:추천|권장|어느|어떤).{0,20}"
        r"(?:손가락\s*(?:번호|순서)?|운지(?:법)?|핑거링)",
        r"(?:[1-5]\s*번|(?:첫|두|세|네|다섯)\s*번째)\s*손가락|"
        r"손가락(?:\s*번호)?(?:은|는|이|가)?\s*[1-5]\s*번|"
        r"엄지|검지|중지|약지|소지|새끼손가락",
    ),
    _constraint(
        "handedness",
        r"(?:어느|어떤|무슨)\s*손(?:으로|이|인가|을)|"
        r"(?:왼손|오른손)\s*(?:인가요|인가|입니까|일까요|일까)|"
        r"(?:왼손|오른손)\s*(?:으로|로)\s*(?:쳐|치|연주|해야|하)"
        r"[가-힣\s]{0,10}(?:나요|습니까|건가요|맞나요|될까요)"
        r"\s*[?？.]?\s*$",
        r"(?:왼손|오른손).{0,12}"
        r"(?:연주|치|맡|사용|파트|성부|입니다|이다)",
    ),
    _constraint(
        "roman_harmony",
        r"로마\s*(?:숫자|기호).{0,24}(?:화성|화음|코드|진행|분석)|"
        r"(?:화성|화음|코드|진행|분석).{0,24}로마\s*(?:숫자|기호)",
        r"(?<![a-z])(?:vii|iii|vi|iv|ii|v|i)"
        r"(?:[°ø+]?(?:6|64|7|65|43|42|9)?)?(?![a-z])"
        r"(?:\s*(?:[-–—→>]|에서|to)\s*|\s+)"
        r"(?<![a-z])(?:vii|iii|vi|iv|ii|v|i)"
        r"(?:[°ø+]?(?:6|64|7|65|43|42|9)?)?(?![a-z])",
    ),
    _constraint(
        "chord_identity",
        r"(?:무슨|어떤)\s*(?:화음|코드)(?:으로|로)?\s*"
        r"(?:구성|이루|되어|쓰|사용|인가|일까|입니까)|"
        r"(?:화음|코드).{0,6}(?:무엇|뭐)",
        r"(?<![a-z])[a-g][#♯b♭]?(?:m|maj|min|dim|aug|sus|add|\d)+(?![a-z])|"
        r"(?:장|단|증|감)\s*(?:3|7)?\s*화음|"
        r"(?:으뜸|딸림|버금딸림)\s*화음",
    ),
    _constraint(
        "exact_count",
        r"(?:정확히|정확한)?\s*몇\s*(?:개|개의|음|음표|박|마디|번)",
        r"(?:\d+|한|두|세|네|다섯|여섯|일곱|여덟|아홉|열)\s*"
        r"개(?:의)?\s*(?:음표|음정|음(?!악)|노트)|"
        r"(?:음표|음정|노트).{0,5}"
        r"(?:\d+|한|두|세|네|다섯|여섯|일곱|여덟|아홉|열)\s*개|"
        r"(?:\d+|한|두|세|네|다섯|여섯|일곱|여덟|아홉|열)\s*"
        r"(?:개(?:의)?\s*)?(?:음표|음(?!악)|노트)\s*(?:으로|로)\s*"
        r"(?:구성|이루)",
    ),
    _constraint(
        "metronome_value",
        r"(?:몇|정확한|정확히|수치).{0,25}(?:bpm|메트로놈)|"
        r"(?:bpm|메트로놈).{0,25}(?:몇|정확|수치)",
        r"\d{2,3}\s*(?:bpm|박\s*/?\s*분)|"
        r"(?:m\.?m\.?|메트로놈|♩|♪)\s*(?:은|는)?\s*=?\s*\d{2,3}",
    ),
    _constraint(
        "tongue_muscle",
        r"혀.{0,8}(?:어느|어떤|무슨).{0,4}근육|"
        r"(?:어느|어떤|무슨).{0,8}혀.{0,4}근육",
        r"이설근|설골설근|경상설근|구개설근|상종설근|하종설근|"
        r"종근|횡근|수직근|내재근|외재근",
    ),
    _constraint(
        "decibels",
        r"(?:db|데시벨)",
        r"\d+(?:\.\d+)?\s*(?:db|데시벨)",
    ),
    _constraint(
        "frequency",
        r"(?:hz|헤르츠|주파수|초당\s*몇\s*번\s*진동)",
        r"\d+(?:\.\d+)?\s*(?:hz|헤르츠)|"
        r"(?:초당|1\s*초(?:에|당)?)\s*\d+(?:\.\d+)?\s*(?:번|회)",
    ),
    _constraint(
        "page_location",
        # Bare ``어느 쪽`` normally means "which option/side", not a page.
        # Treat only an explicit page noun or an unambiguous page counter as
        # a requested answer slot.
        r"(?:몇\s*(?:페이지|쪽)|어느\s*페이지)|"
        r"(?:페이지|쪽).{0,20}어디",
        r"p{1,2}\.?\s*\d+|\d+\s*(?:쪽|페이지)",
    ),
    _constraint(
        "notation_page_subject",
        r"(?:d\.?\s*s\.?|dal\s+segno|세뇨|코다|coda).{0,40}"
        r"(?:페이지|쪽)|(?:페이지|쪽).{0,40}"
        r"(?:d\.?\s*s\.?|dal\s+segno|세뇨|코다|coda)",
        r"(?:d\.?\s*s\.?|dal\s+segno|세뇨|코다|coda).{0,40}"
        r"(?:p{1,2}\.?\s*\d+|\d+\s*(?:쪽|페이지))|"
        r"(?:p{1,2}\.?\s*\d+|\d+\s*(?:쪽|페이지)).{0,40}"
        r"(?:d\.?\s*s\.?|dal\s+segno|세뇨|코다|coda)",
    ),
    _constraint(
        "biographical_date",
        r"출생(?:한)?\s*(?:연도|년도|해|날짜)|"
        r"언제\s*(?:태어|출생)",
        r"(?:출생|태어).{0,24}(?<![0-9])"
        r"(?:1[0-9]{3}|20[0-9]{2})(?![0-9])|"
        r"(?<![0-9])(?:1[0-9]{3}|20[0-9]{2})(?![0-9])"
        r".{0,24}(?:출생|태어)",
    ),
    _constraint(
        "physical_dimension",
        r"(?:가로|세로|너비|폭|길이).{0,30}"
        r"(?:몇|cm|mm|센티미터|밀리미터)|"
        r"(?:몇).{0,20}(?:cm|mm|센티미터|밀리미터)",
        r"(?:\d+(?:\.\d+)?\s*(?:cm|mm)|"
        r"\d+(?:\.\d+)?\s*(?:센티미터|밀리미터))",
    ),
    _constraint(
        "interval_name",
        r"(?:정확한\s*)?음정.{0,12}(?:이름|명칭|종류)|"
        r"(?:무슨|어떤|어느)\s*음정\s*"
        r"(?:인지|인가|일까|입니까|인가요|이야|이죠|[?？]|$)|"
        r"(?:몇|무슨|어떤)\s*도\s*음정",
        r"(?:완전|장|단|증|감)\s*"
        r"(?:\d+|일|이|삼|사|오|육|칠|팔)\s*도|"
        r"(?:\d+|일|이|삼|사|오|육|칠|팔)\s*도\s*음정|"
        r"유니즌|옥타브",
    ),
    _constraint(
        "pitch_name",
        r"(?:정확한\s*)?(?:음고|계이름).{0,12}(?:이름|명칭)|"
        r"(?:무슨|어떤|어느)\s*(?:음고|계이름)|"
        r"(?:무슨|어떤|어느)\s*음(?!악|색|형|절|표|정|역)"
        r"(?:을|를|이|가|은|는)?[^.!?\n]{0,16}(?:부르|불러|내야|낼|소리)",
        r"(?<![a-z])[a-g][#♯b♭]?[0-8](?![a-z0-9])|"
        r"(?:가온|높은|낮은|[1-8]\s*옥타브)\s*"
        r"(?:도|레|미|파|솔|라|시)(?:\s*음)?|"
        r"(?:도|레|미|파|솔|라|시)\s*음(?:을|이|입니다|이다|으로|로)",
    ),
    _constraint(
        "score_conversion",
        r"(?:기타\s*(?:타브|코드)|타브\s*악보).{0,30}"
        r"(?:변환|바꿔|편곡)|"
        r"(?:변환|바꿔|편곡).{0,30}"
        r"(?:기타\s*(?:타브|코드)|타브\s*악보)",
        r"(?:기타|guitar).{0,20}(?:타브|탭|tab|코드|chord)",
    ),
)


_ANSWER_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+|[\r\n]+")
_ANSWER_CLAUSE_SPLIT_RE = re.compile(r"[.!?。！？;,，\r\n]+")
_RECORDING_SOURCE_RE = re.compile(
    r"(?:음원|녹음(?:본|물)?|음반|오디오|recording|audio)",
    flags=re.IGNORECASE,
)
_VIDEO_SOURCE_RE = re.compile(
    r"(?:영상|유튜브|비디오|video|youtube)",
    flags=re.IGNORECASE,
)
_SCORE_SOURCE_RE = re.compile(
    r"(?:악보|총보|보표|score|판본|버전|편곡)",
    flags=re.IGNORECASE,
)
_SOURCE_MISMATCH_RE = re.compile(
    r"(?:다르|달라|차이|불일치|일치하지|맞지\s*않|같지\s*않|"
    r"어긋|엇갈|유무|대신|서로|중\s*(?:하나|어느|어떤)|"
    r"있(?:는|고)?.{0,40}없|없(?:는|고)?.{0,40}있|"
    r"하나의\s*정답|정답.{0,12}(?:없|아니))",
    flags=re.IGNORECASE,
)
_SOURCE_RESOLUTION_RE = re.compile(
    r"(?:따르|기준|우선|선택|정하|결정|판단|확인|계획|참고|"
    r"생략|반복|허용|가능|정답.{0,12}(?:없|아니))",
    flags=re.IGNORECASE,
)
_EDITION_VARIANT_RE = re.compile(
    r"(?:판본|버전|편곡|형태|arrangement|version)",
    flags=re.IGNORECASE,
)


def _sentence_windows(answer: str) -> List[str]:
    """Return one- and two-sentence windows for local relation checks."""

    sentences = [
        sentence.strip()
        for sentence in _ANSWER_SENTENCE_SPLIT_RE.split(answer)
        if sentence.strip()
    ]
    return [
        *sentences,
        *(
            f"{sentences[index]} {sentences[index + 1]}"
            for index in range(len(sentences) - 1)
        ),
    ]


def _has_source_mismatch_resolution(
    answer: str,
    source_pattern: re.Pattern[str],
) -> bool:
    """Require compared sources and their resolution in one local claim."""

    windows = _sentence_windows(answer)
    for window in windows:
        if not source_pattern.search(window) or not _SCORE_SOURCE_RE.search(window):
            continue
        if (
            _SOURCE_MISMATCH_RE.search(window)
            and _SOURCE_RESOLUTION_RE.search(window)
        ):
            return True

    # A reviewed answer may resolve a recording mismatch as an edition or
    # arrangement difference without repeating the recording noun. Both the
    # variant relation and the actionable resolution must still be local.
    if source_pattern is _RECORDING_SOURCE_RE:
        return any(
            _EDITION_VARIANT_RE.search(window)
            and _SOURCE_MISMATCH_RE.search(window)
            and _SOURCE_RESOLUTION_RE.search(window)
            for window in windows
        )
    return False


_ACCENT_RE = re.compile(r"(?:악센트|강세)", flags=re.IGNORECASE)
_NAMED_BEAT_RE = re.compile(
    r"(?:강박|약박|"
    r"(?:첫|두|둘째|세|셋째|네|넷째|다섯|여섯|일곱|여덟|아홉|[1-9])"
    r"\s*(?:(?:번|번째)\s*)?박(?:자)?)",
    flags=re.IGNORECASE,
)
_ACCENT_TO_BEAT_RE = re.compile(
    r"(?:"
    r"(?:악센트|강세)(?:은|는|이|가|을|를)?[^,.;!?。！？\n]{0,24}"
    r"(?:강박|약박|(?:첫|두|둘째|세|셋째|네|넷째|[1-9])\s*"
    r"(?:(?:번|번째)\s*)?박(?:자)?)\s*"
    r"(?:에|에서|으로|로|위에|마다|이다|입니다|이\s*아닌)"
    r"[^,.;!?。！？\n]{0,16}"
    r"(?:놓|두|주|표시|배치|실리|나타|오|있|이다|입니다)?|"
    r"(?:강박|약박|(?:첫|두|둘째|세|셋째|네|넷째|[1-9])\s*"
    r"(?:(?:번|번째)\s*)?박(?:자)?)(?:의|에|에서|으로|로|위에|마다)"
    r"[^,.;!?。！？\n]{0,20}(?:악센트|강세)"
    r")",
    flags=re.IGNORECASE,
)


def _has_accent_to_beat_relation(answer: str) -> bool:
    return any(
        _ACCENT_RE.search(clause)
        and _NAMED_BEAT_RE.search(clause)
        and _ACCENT_TO_BEAT_RE.search(clause)
        for clause in _ANSWER_CLAUSE_SPLIT_RE.split(answer)
        if clause.strip()
    )


_ORNAMENT_RE = re.compile(
    r"(?:꾸밈음|장식음|grace\s*note|ornament)",
    flags=re.IGNORECASE,
)
_ORNAMENT_RULE_RE = re.compile(
    r"(?:사선|원음|균등|박자|리듬|가볍|빠르|짧|길게|부드럽|"
    r"또렷|정확|앞|뒤)",
    flags=re.IGNORECASE,
)
_ORNAMENT_ACTION_RE = re.compile(
    r"(?:노래|부르|불러|연주|처리|다루|나누|붙|이어|시작)",
    flags=re.IGNORECASE,
)


def _has_ornament_execution_relation(answer: str) -> bool:
    return any(
        _ORNAMENT_RE.search(clause)
        and _ORNAMENT_RULE_RE.search(clause)
        and _ORNAMENT_ACTION_RE.search(clause)
        for clause in _ANSWER_CLAUSE_SPLIT_RE.split(answer)
        if clause.strip()
    )


_FRONT_MATTER_MARKER_RE = re.compile(
    r"(?:(?:악보|스코어|총보|score).{0,32}(?:맨\s*앞|가장\s*먼저|"
    r"앞부분|앞쪽|본문\s*앞|처음|첫\s*(?:장|페이지|머리)|서두|"
    r"서문|머리말|표제부)|front\s*matter|지문|scene|서문|머리말|표제부)",
    flags=re.IGNORECASE,
)
_FRONT_MATTER_CONTENT_RE = re.compile(
    r"(?:제목|곡명|작품명|장면|무대|배경|상황|등장인물|인물|줄거리|"
    r"헌사|작곡가|작사가|대본|시대|장소|시간|맥락|집|창문|편지)",
    flags=re.IGNORECASE,
)


def _has_front_matter_content_relation(answer: str) -> bool:
    return any(
        _FRONT_MATTER_MARKER_RE.search(window)
        and _FRONT_MATTER_CONTENT_RE.search(window)
        for window in _sentence_windows(answer)
    )


_SUNG_TEXT_TECHNIQUE_RE = re.compile(
    r"(?:기교|테크닉|스케일|트릴|콜로라투라|"
    r"(?:3|6)\s*연음(?:보)?|셋잇단(?:음표)?|여섯잇단(?:음표)?|"
    r"꾸밈음|장식음|패시지|음형)",
    flags=re.IGNORECASE,
)
_SUNG_TEXT_TARGET_RE = re.compile(
    r"(?:가사|텍스트|음절|단어|모음)",
    flags=re.IGNORECASE,
)
_SUNG_TEXT_DIRECT_ATTACHMENT_RE = re.compile(
    r"(?:가사|텍스트|음절|단어|모음)"
    r"[^.!?。！？\n]{0,64}"
    r"(?:붙|배치|연결|얹|싣|실어|이어|끝맺|유지|길게)|"
    r"(?:첫머리|처음|시작)"
    r"[^.!?。！？\n]{0,32}"
    r"(?:가사|텍스트|음절|단어|모음)"
    r"[^.!?。！？\n]{0,20}"
    r"(?:두|놓|붙|배치)",
    flags=re.IGNORECASE,
)


def _has_sung_text_technique_attachment_relation(answer: str) -> bool:
    """Require a named technique and an actual lyric-placement relation."""

    return any(
        _SUNG_TEXT_TECHNIQUE_RE.search(window)
        and _SUNG_TEXT_TARGET_RE.search(window)
        and _SUNG_TEXT_DIRECT_ATTACHMENT_RE.search(window)
        for window in _sentence_windows(answer)
    )


_VOCAL_CHARACTER_QUALIFIER_RE = re.compile(
    r"(?:가볍|깔끔|밝|어둡|경쾌|부드럽|유연|서정|희망|따뜻|차갑|"
    r"강렬|탄력|편안|맑|무겁|생기|우아|화려|담백|자연스럽|"
    r"아카데믹|활기|날카롭|둥글|섬세|차분|격정|장난스럽|쓸쓸|"
    r"앞쪽|뒤쪽|열린|덮인|두껍|얇|꽃가루|바람|흩뿌리|"
    r"스타카토|레가토|고난|시련|겨울)",
    flags=re.IGNORECASE,
)
_VOCAL_CHARACTER_ACTION_RE = re.compile(
    r"(?:부르|부른|불러|부를|부릅|노래|가창|표현|"
    r"소리.{0,20}(?:내|만들)|"
    r"음색.{0,20}(?:내|만들|유지|표현|살리))",
    flags=re.IGNORECASE,
)
_VOCAL_CHARACTER_RELATION_RE = re.compile(
    r"(?:느낌|분위기|이미지|성격|캐릭터|색채)"
    r"[^.!?。！？\n]{0,120}"
    r"(?:부르|부른|불러|부를|노래|가창|표현|유지|살리|드러내|나타내)|"
    r"(?:부르|부른|불러|부를|노래|가창|표현|유지|살리|드러내|나타내)"
    r"[^.!?。！？\n]{0,120}"
    r"(?:느낌|분위기|이미지|성격|캐릭터|색채)|"
    r"음색(?:을|은|이|으로)"
    r"[^.!?。！？\n]{0,48}"
    r"(?:내|만들|유지|표현|살리)",
    flags=re.IGNORECASE,
)


def _has_vocal_character_description_relation(answer: str) -> bool:
    """Distinguish a singing character from key/voice suitability advice."""

    return any(
        _VOCAL_CHARACTER_QUALIFIER_RE.search(window)
        and (
            _VOCAL_CHARACTER_ACTION_RE.search(window)
            or _VOCAL_CHARACTER_RELATION_RE.search(window)
        )
        for window in _sentence_windows(answer)
    )


_SPECIALIZED_CONSTRAINT_SUPPORT = {
    "sung_text_placement": _has_sung_text_placement_relation,
    "recording_score_mismatch": lambda answer: _has_source_mismatch_resolution(
        answer,
        _RECORDING_SOURCE_RE,
    ),
    "video_score_mismatch": lambda answer: _has_source_mismatch_resolution(
        answer,
        _VIDEO_SOURCE_RE,
    ),
    "accent_placement": _has_accent_to_beat_relation,
    "ornament_execution": _has_ornament_execution_relation,
    "score_front_matter": _has_front_matter_content_relation,
    "sung_text_technique_attachment": (
        _has_sung_text_technique_attachment_relation
    ),
    "vocal_character_description": (
        _has_vocal_character_description_relation
    ),
}


def missing_dense_answer_constraints(query: str, answer: str) -> List[str]:
    """Return explicit requested attributes absent from an answer passage."""

    missing = []
    sung_text_placement_requested = bool(
        re.search(
            SUNG_TEXT_PLACEMENT_REQUEST,
            query,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )
    for name, request_pattern, support_pattern in DENSE_REQUESTED_CONSTRAINTS:
        if (
            sung_text_placement_requested
            and name in {"pitch_name", "interval_name"}
        ):
            # ``어느 음에서 가사를 시작`` asks for a placement slot, not
            # the pitch class or interval name of that note.
            continue
        if not request_pattern.search(query):
            continue
        specialized_support = _SPECIALIZED_CONSTRAINT_SUPPORT.get(name)
        supported = (
            specialized_support(answer)
            if specialized_support is not None
            else bool(support_pattern.search(answer))
        )
        if not supported:
            missing.append(name)
    named_rhythm_patterns = {
        "dotted_rhythm": re.compile(r"(?:겹)?부점|(?:겹)?점음표"),
        "triplet": re.compile(r"셋잇단(?:음표)?|3\s*연음"),
        "sextuplet": re.compile(r"여섯잇단(?:음표)?|6\s*연음"),
    }
    requested_rhythms = {
        name
        for name, pattern in named_rhythm_patterns.items()
        if pattern.search(query)
    }
    if (
        len(requested_rhythms) >= 2
        and any(
            not named_rhythm_patterns[name].search(answer)
            for name in requested_rhythms
        )
    ):
        missing.append("named_rhythm_set")
    return missing


def _source_question_led_answers_requested_constraints(
    query: str,
    record: Dict[str, Any],
) -> bool:
    """Allow only a narrow, authoritative ellipsis in reviewed Q/A pairs.

    Human answers commonly omit a subject that is explicit in their source
    question. For a causal accent-placement question, the reviewed answer may
    therefore explain the purpose while naming the relevant beat but not
    repeat ``악센트``. The source-question-led route can restore that local
    context; arbitrary dense matches and non-causal requested slots cannot.
    """

    answer = str(record.get("answer") or "")
    missing = missing_dense_answer_constraints(query, answer)
    if not missing:
        return True
    return (
        missing == ["accent_placement"]
        and is_causal_answer_request(query)
        and _NAMED_BEAT_RE.search(answer) is not None
        and _record_has_causal_answer_authority(record)
    )


class Embedder(Protocol):
    """Minimal embedding contract used by the in-memory dense index."""

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        ...

    def embed_query(self, text: str) -> List[float]:
        ...


class SearchIndex(Protocol):
    """Search contract shared by lexical and hybrid indexes."""

    records: List[Dict[str, Any]]

    def search(
        self,
        query: str,
        piece: str | None = None,
        measure_ranges: List[List[int]] | None = None,
        topic: str | None = None,
        top_k: int = 8,
    ) -> List[SearchResult]:
        ...

    def semantic_candidates(
        self,
        query: str,
        piece: str | None = None,
        measure_ranges: List[List[int]] | None = None,
        topic: str | None = None,
        top_k: int = 12,
    ) -> List[SearchResult]:
        ...


class DenseRetrievalUnavailable(RuntimeError):
    """Raised when the configured local embedding backend cannot start."""


def _configured_retrieval_mode(settings: Dict[str, Any]) -> str:
    retrieval = settings.get("retrieval")
    if not isinstance(retrieval, dict):
        raise ValueError("settings.retrieval must be an object")
    raw_mode = retrieval.get("mode")
    if not isinstance(raw_mode, str) or not raw_mode.strip():
        raise ValueError("settings.retrieval.mode must be explicitly configured")
    mode = raw_mode.strip().lower()
    if mode not in {"lexical", "hybrid"}:
        raise ValueError("Unsupported retrieval mode: %s" % mode)
    if "fallback_to_lexical" in retrieval:
        raise ValueError(
            "retrieval.fallback_to_lexical is obsolete and unsupported; "
            "configure the single intended retrieval.mode"
        )
    return mode


def _require_llama_cpp_embedding_backend() -> Any:
    """Import a llama.cpp runtime that exposes sequence embeddings."""

    try:
        import llama_cpp
    except (ImportError, OSError) as exc:
        raise DenseRetrievalUnavailable(
            "Required hybrid-retrieval backend `llama-cpp-python` is "
            "unavailable. Activate the `soprano-qa` Conda environment and "
            "install requirements.txt before starting Soprano QA."
        ) from exc
    if not hasattr(llama_cpp.Llama, "embed") or not hasattr(
        llama_cpp,
        "LLAMA_POOLING_TYPE_LAST",
    ):
        raise DenseRetrievalUnavailable(
            "Installed llama-cpp-python lacks sequence embedding support; "
            "install the version required by requirements.txt."
        )
    return llama_cpp


def validate_retrieval_requirements(settings: Dict[str, Any]) -> None:
    """Fail before corpus work when configured hybrid assets are unavailable."""

    retrieval = settings["retrieval"]
    mode = _configured_retrieval_mode(settings)
    if mode == "lexical":
        return
    model_path = str(settings.get("embedding_model_path") or "").strip()
    if not model_path or not os.path.isfile(model_path):
        display_path = model_path or "(embedding_model_path is not configured)"
        raise DenseRetrievalUnavailable(
            "Required hybrid-retrieval embedding checkpoint not found: %s. "
            "Run `python scripts/download_embedding_model.py` before starting "
            "Soprano QA. To run without embeddings intentionally, configure "
            "retrieval.mode as `lexical`; hybrid mode never falls back silently."
            % display_path
        )
    _require_llama_cpp_embedding_backend()


class LlamaCppQwen3Embedder:
    """Qwen3 GGUF embedding backend using the existing llama.cpp runtime."""

    def __init__(
        self,
        model_path: str,
        *,
        query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
        n_ctx: int = 2048,
        n_batch: int = 2048,
        batch_size: int = 8,
        n_gpu_layers: int = -1,
    ) -> None:
        if not os.path.isfile(model_path):
            raise DenseRetrievalUnavailable(
                "Embedding checkpoint not found: %s" % model_path
            )
        llama_cpp = _require_llama_cpp_embedding_backend()

        self.model_path = os.path.abspath(model_path)
        self.query_instruction = query_instruction.strip()
        self.batch_size = max(1, int(batch_size))
        self.n_ctx = max(512, int(n_ctx))
        self.n_batch = max(512, int(n_batch))
        self._lock = threading.Lock()
        try:
            self._model = llama_cpp.Llama(
                model_path=self.model_path,
                embedding=True,
                pooling_type=llama_cpp.LLAMA_POOLING_TYPE_LAST,
                n_ctx=self.n_ctx,
                n_batch=self.n_batch,
                n_gpu_layers=int(n_gpu_layers),
                verbose=False,
            )
        except Exception as exc:
            raise DenseRetrievalUnavailable(
                "Could not load embedding checkpoint: %s" % exc
            ) from exc

        stat = os.stat(self.model_path)
        self.cache_identity = json.dumps(
            {
                "backend": "llama-cpp-python",
                "backend_version": getattr(llama_cpp, "__version__", None),
                "model_path": self.model_path,
                "model_size": stat.st_size,
                "model_mtime_ns": stat.st_mtime_ns,
                "n_ctx": self.n_ctx,
                "n_batch": self.n_batch,
                "pooling": "last",
                "normalize": True,
                "truncate": True,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def _coerce_vectors(
        encoded: Any,
        expected_count: int,
    ) -> List[List[float]]:
        if expected_count == 1 and encoded and isinstance(encoded[0], (int, float)):
            encoded = [encoded]
        if not isinstance(encoded, list) or len(encoded) != expected_count:
            raise DenseRetrievalUnavailable(
                "Embedding backend returned an unexpected batch shape"
            )
        vectors = []
        for vector in encoded:
            if not isinstance(vector, list) or not vector:
                raise DenseRetrievalUnavailable(
                    "Embedding backend returned an empty vector"
                )
            values = [float(value) for value in vector]
            if not all(math.isfinite(value) for value in values):
                raise DenseRetrievalUnavailable(
                    "Embedding backend returned a non-finite vector"
                )
            vectors.append(values)
        return vectors

    def _embed_many(self, texts: Sequence[str]) -> List[List[float]]:
        output: List[List[float]] = []
        with self._lock:
            for start in range(0, len(texts), self.batch_size):
                batch = list(texts[start : start + self.batch_size])
                if not batch:
                    continue
                try:
                    encoded = self._model.embed(
                        batch,
                        normalize=True,
                        truncate=True,
                    )
                    output.extend(self._coerce_vectors(encoded, len(batch)))
                except Exception as exc:
                    raise DenseRetrievalUnavailable(
                        "Embedding inference failed: %s" % exc
                    ) from exc
        return output

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return self._embed_many(texts)

    def embed_query(self, text: str) -> List[float]:
        prompt = "Instruct: %s\nQuery: %s" % (
            self.query_instruction,
            text.strip(),
        )
        return self._embed_many([prompt])[0]


def dense_document_text(record: Dict[str, Any]) -> str:
    """Return semantic unit text without provenance-only retrieval metadata."""

    return str(record.get("relevance_text") or record.get("answer") or "").strip()


def dense_query_text(query: str, piece: str | None) -> str:
    """Remove deterministic routing syntax before embedding a natural query."""

    stripped = query
    if piece:
        aliases = sorted(
            PIECE_QUERY_ALIASES.get(piece, []),
            key=len,
            reverse=True,
        )
        for alias in aliases:
            stripped = re.sub(
                re.escape(alias),
                " ",
                stripped,
                flags=re.IGNORECASE,
            )
    stripped = strip_question_measure_mentions(stripped)
    return " ".join(stripped.split()) or query.strip()


def _normalized_vector(vector: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in vector)
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("Embeddings must contain finite numeric values")
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 0:
        raise ValueError("Embeddings must not be zero vectors")
    return tuple(value / norm for value in values)


def _embedding_fingerprint(
    records: Sequence[Dict[str, Any]],
    relevance_texts: Sequence[str],
    content_texts: Sequence[str],
    embedder: Embedder,
) -> str:
    payload = {
        "version": 1,
        "embedder": getattr(embedder, "cache_identity", None)
        or "%s:%d" % (type(embedder).__qualname__, id(embedder)),
        "documents": [
            {
                "id": record.get("id"),
                "relevance_text": relevance_text,
                "content_text": content_text,
            }
            for record, relevance_text, content_text in zip(
                records,
                relevance_texts,
                content_texts,
            )
        ],
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _load_cached_embeddings(
    cache_path: str,
    fingerprint: str,
) -> List[tuple[float, ...]] | None:
    try:
        with open(cache_path, encoding="utf-8") as file:
            payload = json.load(file)
        if not isinstance(payload, dict):
            return None
        if payload.get("version") != 1 or payload.get("fingerprint") != fingerprint:
            return None
        raw_vectors = payload.get("vectors", [])
        if not isinstance(raw_vectors, list) or not raw_vectors:
            return None
        vectors = [
            _normalized_vector(vector)
            for vector in raw_vectors
        ]
        if len({len(vector) for vector in vectors}) != 1:
            return None
        return vectors
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _write_cached_embeddings(
    cache_path: str,
    fingerprint: str,
    vectors: Sequence[Sequence[float]],
) -> None:
    directory = os.path.dirname(os.path.abspath(cache_path))
    try:
        os.makedirs(directory, exist_ok=True)
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=".dense-embeddings-",
            suffix=".json",
            dir=directory,
            text=True,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "version": 1,
                        "fingerprint": fingerprint,
                        "vectors": vectors,
                    },
                    file,
                    separators=(",", ":"),
                )
            os.replace(temporary_path, cache_path)
        except Exception:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass
            raise
    except OSError:
        # A read-only deployment can still use the in-memory vectors.
        return


class HybridIndex:
    """Fuse BM25 and dense ranks while retaining deterministic scope authority."""

    retrieval_mode = "hybrid"
    configured_retrieval_mode = "hybrid"

    def __init__(
        self,
        records: List[Dict[str, Any]],
        *,
        embedder: Embedder,
        dense_min_score: float = 0.44,
        dense_min_content_score: float = 0.43,
        dense_max_alias_gap: float = 0.20,
        dense_only_ambiguity_margin: float = DENSE_ONLY_AMBIGUITY_MARGIN,
        dense_candidate_k: int = 24,
        lexical_weight: float = 0.35,
        dense_weight: float = 0.65,
        rrf_k: int = 60,
        query_cache_size: int = 128,
        cache_path: str | None = None,
    ) -> None:
        if not -1.0 <= dense_min_score <= 1.0:
            raise ValueError("dense_min_score must be between -1 and 1")
        if not -1.0 <= dense_min_content_score <= 1.0:
            raise ValueError("dense_min_content_score must be between -1 and 1")
        if not 0.0 <= dense_max_alias_gap <= 2.0:
            raise ValueError("dense_max_alias_gap must be between 0 and 2")
        if not 0.0 <= dense_only_ambiguity_margin <= 2.0:
            raise ValueError(
                "dense_only_ambiguity_margin must be between 0 and 2"
            )
        if dense_candidate_k < 1:
            raise ValueError("dense_candidate_k must be positive")
        if lexical_weight <= 0 or dense_weight <= 0:
            raise ValueError("Hybrid weights must both be positive")
        if rrf_k < 1:
            raise ValueError("rrf_k must be positive")
        if query_cache_size < 1:
            raise ValueError("query_cache_size must be positive")
        self.records = records
        self._causal_answer_families = {
            family
            for record in records
            if _record_has_causal_answer_authority(record)
            and (family := _source_family_key(record))
        }
        self.lexical = BM25Index(records)
        self.embedder = embedder
        self.dense_min_score = float(dense_min_score)
        self.dense_min_content_score = float(dense_min_content_score)
        self.dense_max_alias_gap = float(dense_max_alias_gap)
        self.dense_only_ambiguity_margin = float(
            dense_only_ambiguity_margin
        )
        self.dense_candidate_k = int(dense_candidate_k)
        self.lexical_weight = float(lexical_weight)
        self.dense_weight = float(dense_weight)
        self.rrf_k = int(rrf_k)
        self.query_cache_size = int(query_cache_size)
        self._search_state = threading.local()
        self._set_search_state(
            "not_searched",
            reason="index_initialized",
        )
        self.embedding_model = getattr(
            embedder,
            "model_path",
            type(embedder).__qualname__,
        )
        self._record_indices = {
            record["id"]: index for index, record in enumerate(records)
        }
        self._piece_concepts: Dict[str, set[str]] = {}
        for index, record in enumerate(records):
            self._piece_concepts.setdefault(record.get("piece", ""), set()).update(
                self.lexical.document_concepts[index]
            )
        self._query_cache: OrderedDict[str, tuple[float, ...]] = OrderedDict()
        self._query_cache_lock = threading.Lock()

        relevance_texts = [dense_document_text(record) for record in records]
        content_texts = [
            str(record.get("answer") or "").strip()
            for record in records
        ]
        fingerprint = _embedding_fingerprint(
            records,
            relevance_texts,
            content_texts,
            embedder,
        )
        cached = (
            _load_cached_embeddings(cache_path, fingerprint)
            if cache_path
            else None
        )
        if cached is not None and len(cached) == 2 * len(records):
            vectors = cached
        else:
            unique_texts = list(dict.fromkeys([*relevance_texts, *content_texts]))
            try:
                unique_vectors = embedder.embed_documents(unique_texts)
            except DenseRetrievalUnavailable:
                raise
            except Exception as exc:
                raise DenseRetrievalUnavailable(
                    "Document embedding inference failed: %s" % exc
                ) from exc
            if len(unique_vectors) != len(unique_texts):
                raise DenseRetrievalUnavailable(
                    "Embedder returned %d document vectors for %d records"
                    % (len(unique_vectors), len(unique_texts))
                )
            try:
                vector_by_text = {
                    text: _normalized_vector(vector)
                    for text, vector in zip(unique_texts, unique_vectors)
                }
            except (TypeError, ValueError) as exc:
                raise DenseRetrievalUnavailable(
                    "Document embedding normalization failed: %s" % exc
                ) from exc
            vectors = [
                *(vector_by_text[text] for text in relevance_texts),
                *(vector_by_text[text] for text in content_texts),
            ]
            if cache_path:
                _write_cached_embeddings(
                    cache_path,
                    fingerprint,
                    vectors,
                )
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) != 1:
            raise ValueError("Document embeddings have inconsistent dimensions")
        self.document_embeddings = vectors[: len(records)]
        self.content_embeddings = vectors[len(records) :]
        self.embedding_dimension = next(iter(dimensions), 0)

    @property
    def last_dense_error(self) -> str | None:
        return getattr(self._search_state, "last_dense_error", None)

    @last_dense_error.setter
    def last_dense_error(self, value: str | None) -> None:
        self._search_state.last_dense_error = value

    @property
    def last_search_mode(self) -> str:
        return getattr(self._search_state, "last_search_mode", "not_searched")

    @last_search_mode.setter
    def last_search_mode(self, value: str) -> None:
        self._search_state.last_search_mode = value

    @property
    def last_route_reason(self) -> str | None:
        return getattr(self._search_state, "last_route_reason", None)

    @last_route_reason.setter
    def last_route_reason(self, value: str | None) -> None:
        self._search_state.last_route_reason = value

    @property
    def dense_attempted(self) -> bool:
        return bool(getattr(self._search_state, "dense_attempted", False))

    @dense_attempted.setter
    def dense_attempted(self, value: bool) -> None:
        self._search_state.dense_attempted = bool(value)

    @property
    def dense_contributed(self) -> bool:
        return bool(getattr(self._search_state, "dense_contributed", False))

    @dense_contributed.setter
    def dense_contributed(self, value: bool) -> None:
        self._search_state.dense_contributed = bool(value)

    @property
    def last_ambiguous_candidates(self) -> List[Dict[str, Any]]:
        return list(
            getattr(
                self._search_state,
                "last_ambiguous_candidates",
                [],
            )
        )

    @property
    def last_answer_candidates(self) -> List[SearchResult]:
        """Return candidates retained before conservative hard rejection."""

        return list(
            getattr(
                self._search_state,
                "last_answer_candidates",
                [],
            )
        )

    def _set_search_state(
        self,
        mode: str,
        *,
        reason: str,
        dense_attempted: bool = False,
        dense_contributed: bool = False,
        dense_error: str | None = None,
        ambiguous_candidates: List[Dict[str, Any]] | None = None,
        answer_candidates: Sequence[SearchResult] | None = None,
    ) -> None:
        """Record one query's route without conflating routing and failure."""

        self.last_search_mode = mode
        self.last_route_reason = reason
        self.dense_attempted = dense_attempted
        self.dense_contributed = dense_contributed
        self.last_dense_error = dense_error
        self._search_state.last_ambiguous_candidates = list(
            ambiguous_candidates or []
        )
        self._search_state.last_answer_candidates = list(
            answer_candidates or []
        )

    def _query_embedding(self, query: str, piece: str | None) -> tuple[float, ...]:
        prepared_query = dense_query_text(query, piece)
        cache_key = hashlib.sha256(prepared_query.encode("utf-8")).hexdigest()
        with self._query_cache_lock:
            cached = self._query_cache.pop(cache_key, None)
            if cached is not None:
                self._query_cache[cache_key] = cached
                return cached
            try:
                query_vector = _normalized_vector(
                    self.embedder.embed_query(prepared_query)
                )
            except DenseRetrievalUnavailable:
                raise
            except Exception as exc:
                raise DenseRetrievalUnavailable(
                    "Query embedding inference failed: %s" % exc
                ) from exc
            if len(query_vector) != self.embedding_dimension:
                raise DenseRetrievalUnavailable(
                    "Query embedding dimension %d does not match document "
                    "dimension %d"
                    % (len(query_vector), self.embedding_dimension)
                )
            self._query_cache[cache_key] = query_vector
            while len(self._query_cache) > self.query_cache_size:
                self._query_cache.popitem(last=False)
            return query_vector

    def _dense_candidates(
        self,
        *,
        query: str,
        piece: str | None,
        measure_ranges: List[List[int]],
        topic: str | None,
    ) -> List[tuple[int, float, float, str, bool]]:
        query_vector = self._query_embedding(query, piece)

        scored_records: List[tuple[int, float, float, str]] = []
        for index, record in enumerate(self.records):
            if not record.get("retrieval_eligible", True):
                continue
            if piece and record.get("piece") != piece:
                continue
            if topic and topic not in record.get("topic", ""):
                continue
            scope_match = measure_scope_match(record, measure_ranges)
            if scope_match == "unspecified_context":
                continue
            similarity = sum(
                query_value * document_value
                for query_value, document_value in zip(
                    query_vector,
                    self.document_embeddings[index],
                )
            )
            content_similarity = sum(
                query_value * document_value
                for query_value, document_value in zip(
                    query_vector,
                    self.content_embeddings[index],
                )
            )
            scored_records.append(
                (index, similarity, content_similarity, scope_match)
            )

        source_question_led_index: int | None = None
        direct_scope = (
            "overlaps_query_range" if measure_ranges else "local_example"
        )
        local_expert_candidates = [
            item
            for item in scored_records
            if (
                    item[3] == direct_scope
                    and expert_record_is_runtime_final(
                        self.records[item[0]]
                    )
                    and self.records[item[0]].get("measure_status")
                    == "specific"
                    and bool(
                        self.records[item[0]].get("measure_range") or []
                    )
                    and bool(
                        self.records[item[0]].get("retrieval_aliases")
                        or []
                    )
                    and bool(
                        self.records[item[0]].get("source_ids") or []
                    )
                )
        ]
        if local_expert_candidates:
            local_winner = max(
                local_expert_candidates,
                key=lambda item: (
                    item[1],
                    item[2],
                    self.records[item[0]]["id"],
                ),
            )
            winner_record = self.records[local_winner[0]]
            winner_source_ids = set(
                winner_record.get("source_ids") or []
            )
            other_source_scores = [
                item[1]
                for item in scored_records
                if not winner_source_ids.intersection(
                    self.records[item[0]].get("source_ids") or []
                )
            ]
            other_source_margin = local_winner[1] - max(
                other_source_scores,
                default=-1.0,
            )
            if (
                local_winner[1]
                >= SOURCE_QUESTION_LED_MIN_RELEVANCE_SCORE
                and local_winner[2]
                >= SOURCE_QUESTION_LED_MIN_CONTENT_SCORE
                and other_source_margin
                >= SOURCE_QUESTION_LED_MIN_OTHER_SOURCE_MARGIN
                and _source_question_led_answers_requested_constraints(
                    query,
                    winner_record,
                )
            ):
                source_question_led_index = local_winner[0]

        candidates: List[tuple[int, float, float, str, bool]] = []
        for index, similarity, content_similarity, scope_match in (
            scored_records
        ):
            source_question_led = index == source_question_led_index
            if (
                not source_question_led
                and (
                    similarity < self.dense_min_score
                    or content_similarity < self.dense_min_content_score
                )
            ):
                continue
            candidates.append(
                (
                    index,
                    similarity,
                    content_similarity,
                    scope_match,
                    source_question_led,
                )
            )

        candidates.sort(
            key=lambda item: (
                scope_priority(item[3], bool(measure_ranges)),
                item[1],
                item[2],
                self.records[item[0]]["id"],
            ),
            reverse=True,
        )
        return candidates[: self.dense_candidate_k]

    def semantic_candidates(
        self,
        query: str,
        piece: str | None = None,
        measure_ranges: List[List[int]] | None = None,
        topic: str | None = None,
        top_k: int = 12,
    ) -> List[SearchResult]:
        """Return a broad, scope-applicable expert pool for semantic choice.

        Normal retrieval thresholds are precision gates, not a reason to hide
        finalized expert candidates from the constrained selector.  This
        method therefore scores every release-ready expert unit in the piece,
        while preserving the hard piece, topic, and selected-range scope
        boundaries.  It does not choose an answer and is never itself an
        authority signal.
        """

        selected_ranges = measure_ranges or []
        if top_k < 1 or not query.strip():
            return []
        query_vector = self._query_embedding(query, piece)
        lexical_results = self.lexical.search(
            query=query,
            piece=piece,
            measure_ranges=selected_ranges,
            topic=topic,
            top_k=len(self.records),
        )
        lexical_by_id = {
            result.record["id"]: result for result in lexical_results
        }
        candidates: List[SearchResult] = []
        for index, record in enumerate(self.records):
            if (
                not record.get("retrieval_eligible", True)
                or record.get("evidence_type") != "expert_annotation"
                or not expert_record_is_runtime_final(record)
                or (piece and record.get("piece") != piece)
                or (topic and topic not in record.get("topic", ""))
            ):
                continue
            scope_match = measure_scope_match(record, selected_ranges)
            if selected_ranges:
                if scope_match not in {
                    "overlaps_query_range",
                    "global_context",
                }:
                    continue
            elif scope_match not in {
                "general_evidence",
                "local_example",
                "unspecified_scope",
            }:
                continue
            dense_score = sum(
                query_value * document_value
                for query_value, document_value in zip(
                    query_vector,
                    self.document_embeddings[index],
                )
            )
            dense_content_score = sum(
                query_value * document_value
                for query_value, document_value in zip(
                    query_vector,
                    self.content_embeddings[index],
                )
            )
            lexical = lexical_by_id.get(record["id"])
            if lexical is not None:
                candidate = replace(
                    lexical,
                    dense_score=dense_score,
                    dense_content_score=dense_content_score,
                    fusion_score=0.0,
                    retrieval_mode="hybrid",
                    semantic_match_type="selector_candidate",
                )
            else:
                candidate = SearchResult(
                    record=record,
                    score=dense_score,
                    text_score=0.0,
                    measure_score=measure_boost(
                        record,
                        selected_ranges,
                    ),
                    piece_score=(
                        1.0
                        if piece and record.get("piece") == piece
                        else 0.0
                    ),
                    scope_match=scope_match,
                    concept_coverage=concept_coverage(
                        semantic_concepts(query, piece, query=True),
                        self.lexical.document_concepts[index],
                    ),
                    content_concept_coverage=concept_coverage(
                        semantic_concepts(query, piece, query=True),
                        self.lexical.content_concepts[index],
                    ),
                    semantic_match_type="selector_candidate",
                    dense_score=dense_score,
                    dense_content_score=dense_content_score,
                    fusion_score=0.0,
                    retrieval_mode="dense",
                )
            candidates.append(candidate)
        candidates.sort(
            key=lambda result: (
                scope_priority(result.scope_match, bool(selected_ranges)),
                max(result.dense_score, result.dense_content_score),
                (result.dense_score + result.dense_content_score) / 2,
                result.alias_score,
                result.text_score,
                result.record["id"],
            ),
            reverse=True,
        )
        return candidates[:top_k]

    def _has_dense_domain_anchor(
        self,
        query: str,
        query_concepts: Sequence[str],
        piece: str | None,
    ) -> bool:
        if DENSE_DOMAIN_ANCHOR_RE.search(query):
            return True
        if piece:
            document_concepts = self._piece_concepts.get(piece, set())
        else:
            document_concepts = set().union(*self._piece_concepts.values())
        return any(
            concept_matches(concept, document_concepts)
            for concept in query_concepts
        )

    def _has_unsupported_coordinated_subject(
        self,
        query: str,
        piece: str | None,
    ) -> bool:
        """Reject relation questions that introduce an unknown second domain."""

        if piece:
            document_concepts = self._piece_concepts.get(piece, set())
        else:
            document_concepts = set().union(*self._piece_concepts.values())
        for match in DENSE_COORDINATED_RELATION_RE.finditer(query):
            for subject in (match.group("left"), match.group("right")):
                subject_concepts = semantic_concepts(
                    subject,
                    piece,
                    query=True,
                )
                if any(
                    not concept_matches(concept, document_concepts)
                    for concept in subject_concepts
                ):
                    return True
        return False

    def search(
        self,
        query: str,
        piece: str | None = None,
        measure_ranges: List[List[int]] | None = None,
        topic: str | None = None,
        top_k: int = 8,
    ) -> List[SearchResult]:
        measure_ranges = measure_ranges or []
        if top_k < 1 or not query.strip():
            self._set_search_state(
                "none",
                reason="blank_query_or_nonpositive_top_k",
            )
            return []
        lexical_results = self.lexical.search(
            query=query,
            piece=piece,
            measure_ranges=measure_ranges,
            topic=topic,
            top_k=max(top_k, self.dense_candidate_k),
        )
        if (
            not measure_ranges
            and piece
            and is_broad_performance_guidance_query(query, piece)
        ):
            routed, missing_causal_authority = (
                _filter_causal_answer_authority(
                    query,
                    lexical_results[:top_k],
                    measure_ranges=measure_ranges,
                )
            )
            self._set_search_state(
                "lexical_route",
                reason=(
                    SOURCE_FAMILY_MISSING_CAUSAL_AUTHORITY_REASON
                    if missing_causal_authority
                    else "broad_performance_guidance_router"
                ),
            )
            return routed
        real_query, intent_anchors = query_components(query, piece)
        if not token_groups(real_query) and not intent_anchors:
            routed, missing_causal_authority = (
                _filter_causal_answer_authority(
                    query,
                    lexical_results[:top_k],
                    measure_ranges=measure_ranges,
                )
            )
            self._set_search_state(
                "lexical_route",
                reason=(
                    SOURCE_FAMILY_MISSING_CAUSAL_AUTHORITY_REASON
                    if missing_causal_authority
                    else "no_dense_query_content"
                ),
            )
            return routed
        query_concepts = semantic_concepts(query, piece, query=True)
        has_dense_domain_anchor = self._has_dense_domain_anchor(
            query,
            query_concepts,
            piece,
        ) and not self._has_unsupported_coordinated_subject(query, piece)
        if not has_dense_domain_anchor and not lexical_results:
            self._set_search_state(
                "lexical_route",
                reason="no_domain_anchor_or_lexical_match",
            )
            return []

        try:
            dense_candidates = self._dense_candidates(
                query=query,
                piece=piece,
                measure_ranges=measure_ranges,
                topic=topic,
            )
        except DenseRetrievalUnavailable as exc:
            dense_error = "%s: %s" % (type(exc).__name__, exc)
            self._set_search_state(
                "hybrid_error",
                reason="dense_query_error",
                dense_attempted=True,
                dense_error=dense_error,
            )
            raise

        lexical_ranks = {
            result.record["id"]: rank
            for rank, result in enumerate(lexical_results, start=1)
        }
        lexical_by_id = {
            result.record["id"]: result for result in lexical_results
        }
        dense_by_id = {}
        for rank, (
            index,
            similarity,
            content_similarity,
            scope_match,
            source_question_led,
        ) in enumerate(dense_candidates, start=1):
            record_id = self.records[index]["id"]
            if source_question_led and not has_dense_domain_anchor:
                # A lexical overlap must not turn an out-of-domain query into
                # a privileged source-question-led dense match.
                continue
            if record_id not in lexical_by_id:
                if not has_dense_domain_anchor:
                    continue
                if (
                    not source_question_led
                    and similarity - content_similarity
                    > self.dense_max_alias_gap
                ):
                    continue
                if (
                    not source_question_led
                    and missing_dense_answer_constraints(
                        query,
                        str(self.records[index].get("answer") or ""),
                    )
                ):
                    continue
            dense_by_id[record_id] = (
                rank,
                similarity,
                content_similarity,
                scope_match,
                source_question_led,
            )
        candidate_ids = list(lexical_by_id)
        lexical_alias_is_authoritative = any(
            result.alias_score > 0
            for result in lexical_results
        )
        confident_other_range_subject = bool(measure_ranges) and any(
            result.scope_match == "other_range_context"
            and result.concept_coverage >= 0.8
            for result in lexical_results
        )
        allow_dense_only = (
            not lexical_results
            or (
                not lexical_alias_is_authoritative
                and not confident_other_range_subject
            )
        )
        if allow_dense_only:
            dense_only_ids = [
                record_id
                for record_id in dense_by_id
                if record_id not in lexical_by_id
            ]
            if measure_ranges:
                dense_only_ids = [
                    record_id
                    for record_id in dense_only_ids
                    if dense_by_id[record_id][3] == "overlaps_query_range"
                ]
            candidate_ids.extend(
                dense_only_ids
            )
        if not candidate_ids:
            self._set_search_state(
                "hybrid_no_dense_match",
                reason="no_retrieval_candidates",
                dense_attempted=True,
            )
            return []

        normalizer = (
            (self.lexical_weight + self.dense_weight) / (self.rrf_k + 1)
        )
        merged: List[SearchResult] = []
        for record_id in candidate_ids:
            lexical_result = lexical_by_id.get(record_id)
            (
                dense_rank,
                dense_score,
                dense_content_score,
                dense_scope,
                source_question_led,
            ) = dense_by_id.get(
                record_id,
                (None, 0.0, 0.0, None, False),
            )
            lexical_rank = lexical_ranks.get(record_id)
            fusion_score = 0.0
            if lexical_rank is not None:
                fusion_score += self.lexical_weight / (
                    self.rrf_k + lexical_rank
                )
            if dense_rank is not None:
                fusion_score += self.dense_weight / (
                    self.rrf_k + dense_rank
                )
            fusion_score /= normalizer

            if lexical_result is not None:
                result = replace(
                    lexical_result,
                    dense_score=dense_score,
                    dense_content_score=dense_content_score,
                    fusion_score=fusion_score,
                    retrieval_mode=(
                        "hybrid" if dense_rank is not None else "lexical"
                    ),
                    semantic_match_type=(
                        SOURCE_QUESTION_LED_MATCH_TYPE
                        if source_question_led
                        else lexical_result.semantic_match_type
                    ),
                )
            else:
                index = self._record_indices[record_id]
                record = self.records[index]
                result = SearchResult(
                    record=record,
                    score=dense_score,
                    text_score=0.0,
                    measure_score=measure_boost(record, measure_ranges),
                    piece_score=(
                        1.0 if piece and record.get("piece") == piece else 0.0
                    ),
                    scope_match=str(dense_scope),
                    concept_coverage=concept_coverage(
                        query_concepts,
                        self.lexical.document_concepts[index],
                    ),
                    content_concept_coverage=concept_coverage(
                        query_concepts,
                        self.lexical.content_concepts[index],
                    ),
                    semantic_match_type=(
                        SOURCE_QUESTION_LED_MATCH_TYPE
                        if source_question_led
                        else "dense"
                    ),
                    dense_score=dense_score,
                    dense_content_score=dense_content_score,
                    fusion_score=fusion_score,
                    retrieval_mode="dense",
                )
            merged.append(result)

        merged.sort(
            key=lambda result: (
                scope_priority(result.scope_match, bool(measure_ranges)),
                result.alias_score > 0,
                result.alias_score,
                result.fusion_score,
                result.dense_score,
                result.dense_content_score,
                result.measure_score,
                result.text_score,
                result.record["id"],
            ),
            reverse=True,
        )
        source_question_led_results = [
            result
            for result in merged
            if result.semantic_match_type
            == SOURCE_QUESTION_LED_MATCH_TYPE
        ]
        if source_question_led_results:
            # The calibrated winner has a clear lead in the immutable
            # source-question view and independently adequate curated-answer
            # support. Do not retain unrelated overlapping dense passages
            # merely because their canonical ranges are broad. Exact
            # complete-source siblings may remain when curation split one
            # directly matched annotation into complementary units.
            anchor = source_question_led_results[0]
            direct_authority = source_family_answer_authority(
                anchor,
                merged,
                query=query,
                known_causal_family=(
                    _source_family_key(anchor.record)
                    in self._causal_answer_families
                ),
            )
            if direct_authority is None:
                self._set_search_state(
                    "hybrid_no_dense_match",
                    reason=SOURCE_FAMILY_MISSING_CAUSAL_AUTHORITY_REASON,
                    dense_attempted=True,
                    answer_candidates=merged[:top_k],
                )
                return []
            if direct_authority is not anchor:
                anchor_id = anchor.record["id"]
                authority_id = direct_authority.record["id"]
                promoted: List[SearchResult] = []
                for result in merged:
                    if result.record["id"] == authority_id:
                        result = replace(
                            result,
                            semantic_match_type=(
                                SOURCE_QUESTION_LED_MATCH_TYPE
                            ),
                        )
                    elif result.record["id"] == anchor_id:
                        result = replace(
                            result,
                            semantic_match_type=(
                                "strict"
                                if result.text_score > 0
                                else "dense"
                            ),
                        )
                    promoted.append(result)
                merged = promoted
                anchor = next(
                    result
                    for result in merged
                    if result.record["id"] == authority_id
                )
            anchor_sources = set(anchor.record.get("source_ids") or [])
            mixed_causal_procedural_request = bool(
                is_causal_answer_request(query)
                and _procedural_request_match(query)
            )
            merged = [
                result
                for result in merged
                if (
                    result is anchor
                    or (
                        result.record.get("evidence_type")
                        == "expert_annotation"
                        and bool(
                            anchor_sources.intersection(
                                result.record.get("source_ids") or []
                            )
                        )
                        and (
                            result.semantic_match_type
                            == SOURCE_QUESTION_LED_MATCH_TYPE
                            or result.alias_score > 0
                            or (
                                mixed_causal_procedural_request
                                and set(
                                    result.record.get("source_ids") or []
                                )
                                == anchor_sources
                                and _procedural_answer_authority(
                                    result,
                                    query,
                                )
                            )
                        )
                        and result.scope_match == anchor.scope_match
                    )
                )
            ]
            # The source-family authority must lead the compact family.  The
            # remaining siblings preserve their existing diagnostic order.
            merged = [
                anchor,
                *(result for result in merged if result is not anchor),
            ]
        pre_causal_candidates = list(merged)
        merged, missing_causal_authority = (
            _filter_causal_answer_authority(
                query,
                merged,
                measure_ranges=measure_ranges,
            )
        )
        if missing_causal_authority:
            self._set_search_state(
                "hybrid_no_dense_match",
                reason=SOURCE_FAMILY_MISSING_CAUSAL_AUTHORITY_REASON,
                dense_attempted=True,
                answer_candidates=pre_causal_candidates[:top_k],
            )
            return []
        ambiguous_dense_only_range = False
        ambiguous_candidates: List[Dict[str, Any]] = []
        if measure_ranges and len(merged) >= 2:
            pre_ambiguity_candidates = list(merged)
            first, second = merged[:2]
            first_sources = set(first.record.get("source_ids") or [])
            second_sources = set(second.record.get("source_ids") or [])

            def is_uncorroborated_final_expert_dense(
                result: SearchResult,
            ) -> bool:
                record = result.record
                return (
                    result.retrieval_mode == "dense"
                    and result.semantic_match_type == "dense"
                    and result.scope_match == "overlaps_query_range"
                    and result.text_score <= 0
                    and result.alias_score <= 0
                    and result.answer_relation_score <= 0
                    and record.get("evidence_type")
                    == "expert_annotation"
                    and expert_record_is_runtime_final(record)
                    and record.get("measure_status") == "specific"
                )

            ambiguous_dense_only_range = (
                is_uncorroborated_final_expert_dense(first)
                and is_uncorroborated_final_expert_dense(second)
                and bool(first_sources)
                and bool(second_sources)
                and not first_sources.intersection(second_sources)
                and first.dense_score - second.dense_score
                < self.dense_only_ambiguity_margin
            )
            if ambiguous_dense_only_range:
                # Range overlap proves only that a passage can apply at the
                # selected location.  When two unrelated, otherwise
                # unsupported dense passages are effectively tied, neither
                # has enough semantic authority to generate an answer.  Keep
                # independently lexical/hybrid evidence if present; otherwise
                # abstain instead of turning embedding noise into a claim.
                ambiguous_candidates = [
                    {
                        "id": result.record["id"],
                        "source_ids": sorted(
                            result.record.get("source_ids") or []
                        ),
                        "scope_match": result.scope_match,
                        "dense_score": round(result.dense_score, 6),
                        "dense_content_score": round(
                            result.dense_content_score,
                            6,
                        ),
                    }
                    for result in (first, second)
                ]
                merged = [
                    result
                    for result in merged
                    if result.retrieval_mode != "dense"
                ]
        if not merged:
            self._set_search_state(
                "hybrid_no_dense_match",
                reason=(
                    AMBIGUOUS_DENSE_ONLY_RANGE_REASON
                    if ambiguous_dense_only_range
                    else "no_retrieval_candidates"
                ),
                dense_attempted=True,
                ambiguous_candidates=ambiguous_candidates,
                answer_candidates=(
                    pre_ambiguity_candidates[:top_k]
                    if ambiguous_dense_only_range
                    else []
                ),
            )
            return []
        returned = merged[:top_k]
        dense_contributed = any(
            result.retrieval_mode in {"hybrid", "dense"}
            for result in returned
        )
        self._set_search_state(
            "hybrid" if dense_contributed else "hybrid_no_dense_match",
            reason=(
                "ambiguous_dense_only_range_filtered"
                if ambiguous_dense_only_range
                else (
                    "dense_result_returned"
                    if dense_contributed
                    else "no_dense_result_returned"
                )
            ),
            dense_attempted=True,
            dense_contributed=dense_contributed,
            ambiguous_candidates=ambiguous_candidates,
            answer_candidates=returned,
        )
        return returned


def _build_lexical_index(records: List[Dict[str, Any]]) -> BM25Index:
    """Build the explicitly configured model-free retrieval mode."""

    index = BM25Index(records)
    index.retrieval_mode = "lexical"
    index.configured_retrieval_mode = "lexical"
    index.last_search_mode = "lexical"
    index.last_route_reason = "configured_lexical"
    index.dense_attempted = False
    index.dense_contributed = False
    return index


def build_retrieval_index(
    records: List[Dict[str, Any]],
    settings: Dict[str, Any],
    *,
    embedder: Embedder | None = None,
) -> SearchIndex:
    """Build the configured local index without downloading models at runtime."""

    retrieval = settings["retrieval"]
    mode = _configured_retrieval_mode(settings)
    if mode == "lexical":
        return _build_lexical_index(records)
    if embedder is None:
        validate_retrieval_requirements(settings)

    try:
        active_embedder = embedder or LlamaCppQwen3Embedder(
            settings["embedding_model_path"],
            query_instruction=str(
                retrieval.get("query_instruction")
                or DEFAULT_QUERY_INSTRUCTION
            ),
            n_ctx=int(retrieval.get("n_ctx", 2048)),
            n_batch=int(retrieval.get("n_batch", 2048)),
            batch_size=int(retrieval.get("embedding_batch_size", 8)),
            n_gpu_layers=int(retrieval.get("n_gpu_layers", -1)),
        )
        index = HybridIndex(
            records,
            embedder=active_embedder,
            dense_min_score=float(retrieval.get("dense_min_score", 0.44)),
            dense_min_content_score=float(
                retrieval.get("dense_min_content_score", 0.43)
            ),
            dense_max_alias_gap=float(
                retrieval.get("dense_max_alias_gap", 0.20)
            ),
            dense_only_ambiguity_margin=float(
                retrieval.get(
                    "dense_only_ambiguity_margin",
                    DENSE_ONLY_AMBIGUITY_MARGIN,
                )
            ),
            dense_candidate_k=int(retrieval.get("dense_candidate_k", 24)),
            lexical_weight=float(retrieval.get("lexical_weight", 0.35)),
            dense_weight=float(retrieval.get("dense_weight", 0.65)),
            rrf_k=int(retrieval.get("rrf_k", 60)),
            query_cache_size=int(retrieval.get("query_cache_size", 128)),
            cache_path=settings.get("embedding_cache_path"),
        )
        index.configured_retrieval_mode = mode
        return index
    except DenseRetrievalUnavailable:
        raise


def retrieval_diagnostics(index: SearchIndex) -> Dict[str, Any]:
    """Return stable diagnostics for API and CLI consumers."""

    last_search_mode = getattr(index, "last_search_mode", "not_searched")
    return {
        "configured_mode": getattr(
            index,
            "configured_retrieval_mode",
            "lexical",
        ),
        "active_mode": getattr(index, "retrieval_mode", "lexical"),
        "dense_available": getattr(index, "retrieval_mode", "lexical")
        == "hybrid",
        "embedding_model": getattr(index, "embedding_model", None),
        "embedding_dimension": getattr(index, "embedding_dimension", None),
        "last_dense_error": getattr(index, "last_dense_error", None),
        "last_search_mode": last_search_mode,
        "query_route": (
            "hybrid"
            if last_search_mode in {"hybrid", "hybrid_no_dense_match"}
            else (
                "lexical"
                if last_search_mode
                in {"lexical", "lexical_route"}
                else "none"
            )
        ),
        "route_reason": getattr(index, "last_route_reason", None),
        "ambiguous_candidates": getattr(
            index,
            "last_ambiguous_candidates",
            [],
        ),
        "dense_attempted": bool(getattr(index, "dense_attempted", False)),
        "dense_contributed": bool(
            getattr(index, "dense_contributed", False)
        ),
    }


def retrieval_abstained_for_ambiguity(index: SearchIndex) -> bool:
    """Return whether the latest search rejected a cross-source dense tie."""

    return (
        getattr(index, "last_route_reason", None)
        == AMBIGUOUS_DENSE_ONLY_RANGE_REASON
    )


def retrieval_abstained_for_missing_causal_authority(
    index: SearchIndex,
) -> bool:
    """Return whether scoped expert evidence could not answer causal intent."""

    return (
        getattr(index, "last_route_reason", None)
        == SOURCE_FAMILY_MISSING_CAUSAL_AUTHORITY_REASON
    )
