"""Conservative recognition of explicitly general concept questions.

This is a composition choice after security admission, not an authorization
decision. Ambiguous short questions retain the selected work's evidence.
"""
from __future__ import annotations

import re
from typing import Any


_LOCAL = re.compile(
    r"마디|악보|이\s*곡|이\s*부분|여기|해당|작곡가|작곡자|음원|판본|반주|프레이즈|가사|"
    r"왜|이유|의도|녹음|연주자|성악가|선생|전문가|자료|위의|앞의|"
    r"\b(?:bar|bars|measure|measures|score|piece|passage|edition|recording|"
    r"composer|accompaniment|accompanist|phrase|lyrics?|singer|performer|expert|"
    r"source|above|previous|why)\b|\b(?:this|these|that|those)\b",
    re.IGNORECASE,
)
_GENERAL = re.compile(
    r"정의|무슨\s*뜻|어떤\s*뜻|차이|비교|항상|언제나|일반적|"
    r"\b(?:define|definition|difference|compare|compared|versus|always)\b|"
    r"\bin general\b|\bwhat (?:is|are)\b|\bwhat does\b.*\bmean\b",
    re.IGNORECASE,
)


def is_explicit_general_question(payload: dict[str, Any]) -> bool:
    """Require explicit general intent and no local or causal reference.

    No piece names, music-term allowlists, or benchmark question IDs are used.
    A selected measure range always keeps the evidence path. Unrecognized
    wording also keeps it; the passage selector can still reject irrelevant KUs.
    """
    question = payload.get("question")
    selected_range = payload.get("measure_range")
    if not isinstance(question, str) or (selected_range and selected_range != "(none selected)"):
        return False
    # General wording does not make a directly answering KU irrelevant. This
    # flag comes from the retriever's complete source-question concept match;
    # it changes only routing, never admission, claim authority or output review.
    if any(isinstance(r, dict) and r.get("citation_label") and r.get("question_alias_match") is True
           for r in payload.get("Retrieved evidence", [])):
        return False
    return not _LOCAL.search(question) and bool(_GENERAL.search(question))
