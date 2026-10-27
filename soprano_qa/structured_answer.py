"""Constrained answer structure with application-rendered model citations.

The schema guarantees neither truth nor entailment. A semantic guard must check
this module's rendered output, followed by the normal citation/scope validators.
The model selects evidence labels; the renderer never invents an attribution.
"""
from __future__ import annotations

import json
import re
from typing import Any

from soprano_qa.security import SecurityViolation


MAX_SENTENCES = 16
MAX_SENTENCE_CHARS = 2000

STRUCTURED_ANSWER_POLICY = """Mandatory structured answer format
Return exactly one JSON object of this form, with no markdown or extra keys:
{"sentences":[{"text":"One supported sentence.","evidence":["E1"]}]}
Write each sentence's prose in text and select all evidence labels that actually
support that sentence in evidence. Use only non-null citation_label values from
Retrieved evidence. Every item requires at least one such label when citable
evidence is supplied. Do not put inline citation labels or record IDs in text:
the application will render the labels you select. Prefer one sentence per item.
Use at most 16 items, keeping each text nonempty and at most 2000 characters.
If no citable evidence exists and score context is permitted, evidence must be
an empty array for each item; only identify words sung in the selected bars.
Preserve the required answer language, factual support, scope, and all security
rules. This format changes only how you express supported prose and citations;
it does not permit new facts or new musical advice. Never invent support to
satisfy the format. Do not include a verdict, commentary, or additional fields.
/no_think
"""

_LABEL_RE = re.compile(r"E[1-9][0-9]*\Z")
_INLINE_CITATION_RE = re.compile(
    r"[\[【]\s*(?:E\s*[0-9]|Evidence\b|근거\s*[0-9])",
    flags=re.IGNORECASE,
)
_REDUNDANT_CITATION_RE = re.compile(r"\[(E[1-9][0-9]*(?:\s*,\s*E[1-9][0-9]*)*)\]")
# Match the downstream claim validator's punctuation boundaries. Decimal dots
# and abbreviation-internal dots have no following whitespace and stay intact.
# An abbreviation ending before whitespace is conservatively cited as well;
# its original characters are preserved and display-time stripping restores it.
_SENTENCE_END_RE = re.compile(r"[.!?。！？]+[\"'”’」』】)]*(?=\s|$|[가-힣])")


def _allowed_labels(payload: dict[str, Any]) -> list[str]:
    evidence = payload.get("Retrieved evidence", [])
    if not isinstance(evidence, list):
        raise SecurityViolation()
    labels: list[str] = []
    for record in evidence:
        if not isinstance(record, dict):
            raise SecurityViolation()
        label = record.get("citation_label")
        if label is None:
            continue
        if not isinstance(label, str) or not _LABEL_RE.fullmatch(label):
            raise SecurityViolation()
        if label not in labels:
            labels.append(label)
    return labels


def answer_schema(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON schema constrained to this bundle's primary labels."""
    labels = _allowed_labels(payload)
    evidence_schema: dict[str, Any] = {
        "type": "array",
        "items": {"type": "string", **({"enum": labels} if labels else {})},
        "minItems": 1 if labels else 0,
        "maxItems": len(labels),
        "uniqueItems": True,
    }
    return {
        "type": "object",
        "properties": {
            "sentences": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_SENTENCES,
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "minLength": 1,
                            # llama.cpp expands bounded string repetition;
                            # a 2000-character schema bound exceeds its grammar
                            # complexity limit. Enforce that cap in the parser;
                            # generation remains bounded by max_tokens.
                        },
                        "evidence": evidence_schema,
                    },
                    "required": ["text", "evidence"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["sentences"],
        "additionalProperties": False,
    }


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate JSON key")
        value[key] = item
    return value


def _cite_text(text: str, labels: list[str]) -> str:
    if not labels:
        return text
    suffix = " [" + ", ".join(labels) + "]"
    cited_lines = []
    for line in text.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        newline = line[len(content):]
        segments: list[str] = []
        start = 0
        for match in _SENTENCE_END_RE.finditer(content):
            segment = content[start:match.end()]
            segments.append(segment + suffix if segment.strip() else segment)
            start = match.end()
        trailing = content[start:]
        # Keep all existing whitespace: insert labels immediately after prose.
        if trailing.strip():
            prose = trailing.rstrip()
            segments.append(prose + suffix + trailing[len(prose):])
        else:
            segments.append(trailing)
        cited_lines.append("".join(segments) + newline)
    return "".join(cited_lines)


def render_structured_answer(raw: str, payload: dict[str, Any]) -> str:
    """Validate exact structure and render only the model-selected citations.

    Each item's labels apply to every sentence fragment in its text. This also
    covers a model that puts multiple sentences in one item; the subsequent
    semantic guard must verify support for every rendered claim and citation.
    """
    try:
        parsed = json.loads(raw, object_pairs_hook=_unique_object)
    except (TypeError, ValueError, RecursionError):
        raise SecurityViolation() from None
    if not isinstance(parsed, dict) or set(parsed) != {"sentences"}:
        raise SecurityViolation()
    sentences = parsed["sentences"]
    if not isinstance(sentences, list) or not 1 <= len(sentences) <= MAX_SENTENCES:
        raise SecurityViolation()
    allowed_labels = set(_allowed_labels(payload))
    rendered = []
    for item in sentences:
        if not isinstance(item, dict) or set(item) != {"text", "evidence"}:
            raise SecurityViolation()
        text, labels = item["text"], item["evidence"]
        if (
            not isinstance(text, str)
            or not text.strip()
            or len(text) > MAX_SENTENCE_CHARS
            or not isinstance(labels, list)
            or any(not isinstance(label, str) for label in labels)
        ):
            raise SecurityViolation()
        if (
            len(labels) != len(set(labels))
            or not set(labels) <= allowed_labels
            or (allowed_labels and not labels)
        ):
            raise SecurityViolation()
        # Some models repeat a selected label in prose despite the schema
        # instruction. Remove only redundant labels already selected for this
        # item; an unknown, context-only or different label still fails closed.
        def remove_redundant(match: re.Match[str]) -> str:
            if not set(re.findall(r"E[1-9][0-9]*", match.group(1))) <= set(labels):
                raise SecurityViolation()
            return ""

        text = _REDUNDANT_CITATION_RE.sub(remove_redundant, text)
        if not text.strip() or _INLINE_CITATION_RE.search(text):
            raise SecurityViolation()
        rendered.append(_cite_text(text, labels))
    return "\n".join(rendered)
