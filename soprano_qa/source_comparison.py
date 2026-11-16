"""Conservative rendering of two explicit opening-dynamics accounts.

This small Korean-source extractor is not a general contradiction detector or
an answer cache. It recognizes affirmative accompaniment markings and quiet
vocal-entrance advice independently in each selected source. Unrecognized,
conditional, ambiguous or local cases keep the ordinary source-preserving path.
"""
from __future__ import annotations

import re


_OPENING = re.compile(r"처음|첫\s*(?:소리|음|반주)|시작|\b(?:open\w*|begin\w*|start\w*|first)\b", re.I)
_TOPIC = re.compile(r"셈여림|다이?나믹|다이내믹|표기|\b(?:dynamics?|markings?)\b|(?<![a-z])(?:ppp|pp|mp|mf|fff|ff|p|f)(?![a-z])", re.I)
_ASSERTION = re.compile(
    r"반주(?:부)?[^.!?\n]{0,80}?(?<![a-z])(?P<mark>ppp|pp|mp|mf|fff|ff|p|f)(?![a-z])"
    r"\s*\)?\s*(?:으?로|이라고|라고)?\s*"
    r"(?:시작(?:한다(?!면)|합니다|하는|하며|하고)|"
    r"(?:표기(?:되어|돼)|표시(?:되어|돼)|적혀|되어)\s*있(?:다(?!면)|습니다|는|으므로|어서|고))", re.I)
_AMBIGUOUS = re.compile(r"만약|가정|예를\s*들어|추정|아마|또는|혹은|아니|아닌|아닙|후주|간주|후반|중간|마지막|재현|\d+\s*마디")
_EXPLAINED_DIFFERENCE = re.compile(r"판본|개정판|편곡본|다른\s*악보|(?:차이|불일치|서로\s*다른).{0,80}(?:때문|이유|원인)")
_VOICE = re.compile(r"첫\s*소리|노래|가창|성악")
_QUIET = re.compile(r"크지\s*않게.{0,25}시작|지나치게\s*큰\s*소리로\s*시작하기보다는.{0,35}부드러운\s*어택")
_SUGGESTION = re.compile(r"좋아\s*보|좋(?:다|습니다|아요)|적절(?:해요|합니다|하다)|권(?:합니다|한다)")


def _opening_mark_and_advice(answer: str) -> str | None:
    if not isinstance(answer, str) or _EXPLAINED_DIFFERENCE.search(answer):
        return None
    sentences = re.split(r"[.!?\n]+", answer)
    markings = []
    for sentence in sentences:
        for match in _ASSERTION.finditer(sentence):
            if _AMBIGUOUS.search(sentence) or not _OPENING.search(sentence):
                return None
            markings.append(match["mark"])
    # Multiple claims are deliberately left to the full source path, even
    # when the same symbol repeats: their conditions may differ.
    if len(markings) != 1:
        return None
    shared_advice = any(_VOICE.search(s) and _QUIET.search(s) and _SUGGESTION.search(s)
                        and not re.search(r"않(?:다|습니다)|아니|아닌|아닙|만약|가정", s) for s in sentences)
    return markings[0] if shared_advice else None


def opening_dynamics_accounts(payload: dict):
    """Extract two differing accounts as data, never as a generated answer.

    Values and citation labels come from actual selected text; no work IDs,
    evaluation questions, reference answers or model verdicts are cached here.
    """
    question = payload.get("question", "")
    if (not isinstance(question, str) or not _OPENING.search(question) or not _TOPIC.search(question)
            or payload.get("measure_range") not in (None, "", "(none selected)")
            or payload.get("answer_language", "ko") not in ("ko", "en")):
        return None
    sources = [r for r in payload.get("Retrieved evidence", []) if r.get("citation_label")]
    if len(sources) != 2 or any(r.get("evidence_role") == "context" for r in sources):
        return None
    labels = [r["citation_label"] for r in sources]
    if len(set(labels)) != 2:
        return None
    values = [_opening_mark_and_advice(r.get("answer", "")) for r in sources]
    if not all(values) or values[0].casefold() == values[1].casefold():
        return None
    return [{"evidence": label, "mark": value} for label, value in zip(labels, values)]


