"""Application boundaries for direct and retrieved prompt injection.

The cheap screen catches recognizable control requests, including quoted ones,
without calling a model. It is deliberately not a general domain classifier and
not a proof of prompt-injection resistance; the writing prompt handles scope.
"""
from __future__ import annotations

import json
import re
import unicodedata
from typing import Any


MAX_QUESTION_CHARS = 4000
_REJECTION_REASONS = frozenset({
    "security_policy_violation",
    "invalid_question",
    "question_too_long",
    "prompt_injection_detected",
    "unsafe_retrieved_evidence",
    "unsafe_score_context",
    "unsafe_translation",
})


class SecurityViolation(ValueError):
    """A closed boundary with a fixed, non-sensitive public reason code."""

    def __init__(self, reason: str = "security_policy_violation") -> None:
        self.reason = (
            reason if reason in _REJECTION_REASONS else "security_policy_violation"
        )
        super().__init__(self.reason)


def _normalized_screen_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    # Drop invisible formatting and non-whitespace controls rather than insert
    # separators that would let e.g. ig\u200bnore evade the same screen.
    normalized = "".join(
        char for char in normalized
        if unicodedata.category(char) not in {"Cf", "Cc"} or char.isspace()
    )
    return re.sub(r"\s+", " ", normalized).strip().casefold()


_CONTROL_PATTERNS = tuple(re.compile(pattern, re.IGNORECASE) for pattern in (
    r"\b(?:ignore|disregard|forget|override|bypass|discard)\b.{0,90}"
    r"\b(?:(?:system|developer|previous|prior|earlier|above|all)\s+)?"
    r"(?:instructions?|prompts?|rules?|polic(?:y|ies)|guardrails?)\b",
    r"\b(?:ignore|disregard|forget|discard)\s+(?:absolutely\s+)?everything\b",
    r"\b(?:ignore|disregard|forget|discard)\b.{0,45}"
    r"\b(?:previous|prior|earlier|above)\s+(?:messages?|context|conversation)\b",
    r"\b(?:reveal|print|show|repeat|return|expose|leak|display)\b.{0,70}"
    r"\b(?:system|developer|hidden|initial|original)\s+(?:prompt|instructions?)\b",
    r"\b(?:you\s+are\s+now|your\s+new\s+role\s+is)\b",
    r"\b(?:act|behave)\s+as\s+(?:an?\s+)?"
    r"(?:unrestricted|unfiltered|jailbroken|system|developer|chatgpt|chef|assistant)\b",
    r"\b(?:do\s+not|don't|never|stop)\s+(?:use|follow|obey|cite)\b.{0,45}"
    r"\b(?:evidence|sources?|system|instructions?|polic(?:y|ies)|rules?)\b",
    r"<\|(?:im_start|im_end|start_header_id|end_header_id|eot_id|"
    r"start|end|message|endoftext)\|>|\[/?inst\]|<<\s*/?sys\s*>>",
    r"<\s*/?\s*(?:system|developer|assistant)(?:\s[^>]*|\s*)>",
    r"\b(?:system|developer|assistant)\s*:",
    r"\[\s*(?:system|developer|assistant)\s*\]",
    r"(?:이전|앞선|앞의|기존|위의|모든|시스템|개발자).{0,35}"
    r"(?:지시|명령|규칙|정책|프롬프트).{0,35}"
    r"(?:무시|잊|폐기|취소|우회|덮어|따르지)",
    r"(?:이전|앞선|앞의|기존|위의).{0,20}(?:내용|대화).{0,20}"
    r"(?:무시|잊|폐기)",
    r"(?:시스템|개발자|숨겨진|비공개).{0,12}(?:프롬프트|지시문|명령어)"
    r".{0,25}(?:알려|보여|출력|공개|반복|노출)",
    r"(?:너|당신)(?:는|의).{0,12}(?:이제|지금부터|새로운\s*역할)",
    r"\b(?:first|last)\s+(?:sentence|line|word)\s+of\s+(?:your|the(?:\s+final)?)\s+"
    r"(?:answer|response|reply)\b.{0,65}\b(?:must|shall|has\s+to)\s+be\b.{0,40}"
    r"(?:[a-z][a-z0-9]*[-_]){2,}[a-z0-9]*\d[a-z0-9]*",
    r"(?:최종\s*)?(?:답변|응답)(?:의|에서)?\s*(?:첫|마지막)\s*(?:문장|줄|단어)"
    r".{0,60}(?:[a-z][a-z0-9]*[-_]){2,}[a-z0-9]*\d[a-z0-9]*"
    r".{0,20}(?:이어야|이여야|출력|추가|시작|끝내)",
))

# A document may describe musical performance, but cannot configure the QA
# response screen or its answer template. These patterns apply to source/output
# content, not ordinary user requests for a concise format. They do not depend
# on a benchmark marker, source ID or musical vocabulary allowlist.
_SOURCE_CONTROL_PATTERNS = tuple(re.compile(pattern, re.IGNORECASE) for pattern in (
    r"\b(?:answer|response|reply|output)\s+(?:format|template|screen|marker|terminator)\b"
    r".{0,180}\b(?:append|prepend|insert|emit|print|output|include)\b",
    r"(?:답변|응답|출력)(?:\s*(?:의|용))?\s*(?:형식|화면|템플릿|마커|종료\s*표시)"
    r".{0,180}(?:붙이|붙인|붙여|덧붙|추가|삽입|출력|내보내|표시하)",
    # Decoding musical notation is subject matter. A source directing the QA
    # system to turn a decoded request into its answer is a control instruction,
    # even when the payload itself is never decoded or executed.
    r"\b(?:decode|decipher|decrypt)\b.{0,160}\b(?:final|actual)\s+"
    r"(?:answer|response|reply)\b",
    r"(?:해독|디코딩|복호화).{0,160}(?:최종|실제)\s*(?:답변|응답)",
))


def screen_text(text: str, *, source: str = "question") -> None:
    """Reject known control instructions without trusting their claimed role.

    ``source`` determines only a fixed reason code, never policy precedence.
    Quoted attack examples are also rejected in this narrowly scoped QA app.
    """
    if not isinstance(text, str):
        raise SecurityViolation()
    candidates = [text]
    if source in {"evidence", "score_context"}:
        # Callers screen serialized records. Decode their string values first:
        # JSON's escaped newline or Unicode sequences must not hide a role
        # delimiter or a control request that the model sees as normal text.
        try:
            pending = [json.loads(text)]
        except (ValueError, RecursionError):
            pending = []
        if pending:
            candidates = []
            while pending:
                value = pending.pop()
                if isinstance(value, str):
                    candidates.append(value)
                elif isinstance(value, dict):
                    pending.extend(value.keys())
                    pending.extend(value.values())
                elif isinstance(value, list):
                    pending.extend(value)
    patterns = _CONTROL_PATTERNS
    if source in {"evidence", "score_context", "answer"}:
        patterns += _SOURCE_CONTROL_PATTERNS
    if any(
        pattern.search(normalized)
        for normalized in map(_normalized_screen_text, candidates)
        for pattern in patterns
    ):
        reason = {
            "evidence": "unsafe_retrieved_evidence",
            "score_context": "unsafe_score_context",
            "translation": "unsafe_translation",
        }.get(source, "prompt_injection_detected")
        raise SecurityViolation(reason)


def validate_question_input(question: str) -> None:
    """Validate transport/input shape without classifying the user's task."""
    if not isinstance(question, str):
        raise SecurityViolation("invalid_question")
    if len(question) > MAX_QUESTION_CHARS:
        raise SecurityViolation("question_too_long")
    if not _normalized_screen_text(question):
        raise SecurityViolation("invalid_question")


def screen_question(question: str) -> None:
    """Apply the historical rule-based task screen before model inference."""
    validate_question_input(question)
    screen_text(question)


def screen_evidence_record(record: dict) -> dict:
    """Keep classification metadata out of the answer-generation boundary.

    ``topic`` belongs to retrieval and is not needed to establish a musical claim.
    Omit the complete field regardless of its wording, including semantic attacks
    that have no known control phrase. Answer text, source
    questions, attribution and scope are never trimmed to make a record pass.
    Validate the complete remaining record, and never mutate the corpus object.
    """
    cleaned = {key: value for key, value in record.items() if key != 'topic'} if 'topic' in record else record
    screen_text(json.dumps(cleaned, ensure_ascii=False), source='evidence')
    return cleaned


def serialize_untrusted(payload: dict[str, Any]) -> str:
    """Encode fields as JSON data and neutralize literal markup/chat tokens.

    This encoding prevents literal delimiter collisions, not semantic attacks.
    Korean and normal music notation remain readable to the local model.
    """
    serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    return serialized.replace("<", "\\u003c").replace(
        ">", "\\u003e"
    ).replace("[INST]", "\\u005bINST\\u005d").replace(
        "[/INST]", "\\u005b/INST\\u005d"
    )


