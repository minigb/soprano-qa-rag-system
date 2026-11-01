"""Conservative recognition of explicitly general concept questions.

This is a composition choice after security admission, not an authorization
decision. Ambiguous short questions retain the selected work's evidence.
"""
from __future__ import annotations

import re
from typing import Any


_LOCAL_BASE = (
    r"마디|악보|이\s*곡|이\s*부분|여기|해당|작곡가|작곡자|음원|판본|반주|프레이즈|"
    r"왜|이유|의도|녹음|연주자|성악가|선생|전문가|자료|위의|앞의|"
    r"\b(?:bar|bars|measure|measures|score|piece|passage|edition|recording|"
    r"composer|accompaniment|accompanist|phrase|singer|performer|expert|"
    r"source|above|previous|why)\b|\b(?:this|these|that|those)\b"
)
_LOCAL = re.compile(_LOCAL_BASE + r"|가사|\blyrics?\b", re.IGNORECASE)
_PRACTICE_LOCAL = re.compile(
    _LOCAL_BASE + r"|(?:이|그|저|이번|본)\s*(?:가사|노래|곡|선율|멜로디|대목|부분)",
    re.IGNORECASE,
)
_PRACTICE = re.compile(
    r"(?:연습|훈련|학습|익히|배우)[가-힣]*\s*때.*(?:어떤|무엇|어떻게)|"
    r"\b(?:when|while)\s+(?:practi[cs]ing|learning|studying|rehearsing)\b",
    re.IGNORECASE,
)
_UNIVERSAL = re.compile(r"항상|언제나|\balways\b", re.IGNORECASE)
_GENERAL = re.compile(
    r"정의|무슨\s*뜻|어떤\s*뜻|차이|비교|일반적|"
    r"\b(?:define|definition|difference|compare|compared|versus)\b|"
    r"\bin general\b|\bwhat (?:is|are)\b|\bwhat does\b.*\bmean\b",
    re.IGNORECASE,
)


def explicit_general_intent(payload: dict[str, Any]) -> str | None:
    """Require explicit general intent and no local or causal reference.

    No hardcoded piece names, music-term allowlists or case IDs are used.
    A selected measure range always keeps the evidence path. Unrecognized
    wording also keeps it; the passage selector can still reject irrelevant KUs.
    """
    question = payload.get("question")
    selected_range = payload.get("measure_range")
    if not isinstance(question, str) or (selected_range and selected_range != "(none selected)"):
        return None
    # General wording does not make a directly answering KU irrelevant. This
    # flag comes from the retriever's complete source-question concept match;
    # it changes only routing, never admission, claim authority or output review.
    if any(isinstance(r, dict) and r.get("citation_label") and r.get("question_alias_match") is True
           for r in payload.get("Retrieved evidence", [])):
        return None
    # A named-work comparison remains situated even if it says "always" or
    # "difference". Check titles before every general branch, not just practice.
    compact = lambda text: re.sub(r"[\W_]+", "", text).casefold()
    names = [payload.get("piece")]
    names.extend(r.get(key) for r in payload.get("Retrieved evidence", []) if isinstance(r, dict)
                 for key in ("work", "piece_title"))
    if any(isinstance(name, str) and len(compact(name)) >= 3 and compact(name) in compact(question)
           for name in names):
        return None
    if not _LOCAL.search(question):
        if _UNIVERSAL.search(question):
            return "universal"
        if _GENERAL.search(question):
            # A definition may be directly answered by a retrieved source.
            # The caller must assess its actual text, even for one source.
            return "definition"
    # An explicit practice scenario can mention lyrics in general, without
    # referring to the selected score's words. Keep local/causal questions and
    # directly matching source questions on the evidence path. Work titles are
    # read from this bundle, not a maintained list of benchmark pieces.
    if not _PRACTICE.search(question) or _PRACTICE_LOCAL.search(question):
        return None
    return "practice"


def is_explicit_general_question(payload: dict[str, Any]) -> bool:
    return explicit_general_intent(payload) is not None
