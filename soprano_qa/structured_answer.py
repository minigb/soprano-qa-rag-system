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
from soprano_qa.music_terms import MUSIC_TERMINOLOGY


MAX_SENTENCES = 6
MAX_SENTENCE_CHARS = 2000
# Keep this low-level module independent of answer.py, which imports llm.py.
NO_GROUNDED_ANSWER_SENTINEL = "<NO_GROUNDED_ANSWER>"

STRUCTURED_ANSWER_POLICY = """Return one JSON object, no markdown or extra keys:
{"sentences":[{"kind":"evidence","evidence":["E1"],"text":"Source-based answer."}]}
Choose each item's kind and evidence before writing its text:
- evidence: a faithful claim from the supplied reviewed answers; select its non-null
  citation_label values. At least one supporting label is required.
- background: general music knowledge, an explanation, or an explicitly optional
  practice suggestion. evidence must be []. It must not assert unverified facts
  about this piece/bar, quote an expert, explain an unestablished cause, or contradict
  the supplied advice. The application identifies it as general explanation.
- score_lyrics: only words sung in the supplied score context; evidence must be [].
Keep these kinds in separate items. Never attach a source citation to background.
Use 1-4 items, up to 6 if needed. Each text must be nonempty and at most 2000
characters. Do not pad, repeat, print labels/IDs, or write the kind name in prose.
If the source answers the question, do not add unnecessary background. If evidence
is missing, a useful general explanation is allowed, with an explicit limitation
when the question asks for a specific score fact. If no useful authorized answer
is possible, return {"sentences":[]}.
/no_think
"""


SOURCE_PRESERVATION_POLICY = """Return one JSON object with the existing sentences/kind/evidence/text fields.
Select only passages that directly answer the actual question, preserving useful
complementary advice. An evidence item must select exactly ONE citation label;
never select the same label twice. Select complete reviewed answers, not isolated
clauses whose conditions could be lost. Do not merge conflicting passages into
one new claim or invent a reason for their disagreement.
Omit a second source if it only repeats advice already fully covered; retain
distinct qualifications, alternatives and genuine disagreements. A shared
musical subject does not make a source an answer to the requested task.

For Korean answers, every evidence item's text MUST be the empty string "".
Example: {"sentences":[{"kind":"evidence","evidence":["E1"],"text":""}]}
The application will quote that selected source's complete answer verbatim,
including its conditions, uncertainty, lyric referents and original register.
Do not rewrite it or add an explanation inside an evidence item.
For English answers, an evidence item's text MUST faithfully translate the
COMPLETE answer of its one selected source. Preserve every condition, actor,
negation, qualification, alternative and lyric referent. Do not summarize,
combine sources, strengthen an interpretation into composer intention, or add
a practice claim. Keep quoted sung words and vowel/consonant symbols unchanged.
Do not add romanization, transliteration, or a new phonetic spelling of sung
words. Retain their original script exactly. Explain a pronunciation spelling
only when the source itself supplies it. Preserve every term in a source's
list of techniques; do not substitute different familiar examples.

Background items require evidence=[] and nonempty text. They may give accurate
general music knowledge or explicitly optional suggestions, clearly separate
from the selected source. Add them only when the source alone leaves a useful
part of the question unanswered. Do not repeat, contradict, or invent a cause
for the source. Unknown local facts must remain unknown. score_lyrics items
remain limited to supplied sung words and require evidence=[]. All application
scope, safety and output-review requirements still apply to every item.
Use as few items as needed, at most six. Return {"sentences":[]} if no useful
authorized answer is possible. Never print labels or headings in item text.
/no_think
"""

SOURCE_PRESERVATION_POLICY += "\n" + MUSIC_TERMINOLOGY

SOURCE_SELECTION_POLICY = """Return one JSON object with sentences/kind/evidence/text fields.
Select complete reviewed answers that directly help answer the question.
For EVERY evidence item choose exactly one supplied citation label and set
text to the EMPTY STRING. Never select a label twice. Do not translate or
paraphrase a source here: the application translates selected sources separately.
Example: {"sentences":[{"kind":"evidence","evidence":["E1"],"text":""}]}
Preserve useful complementary sources without selecting unrelated passages.
Omit a second source if it only repeats the same advice already fully covered;
keep distinct qualifications, alternatives and genuine source disagreements.
Background items may provide accurate general music knowledge or explicitly
optional practice suggestions in English, only when they answer a useful aspect
left open by the sources. Their evidence must be [], and text must be nonempty.
Never invent a local fact, cause or expert attribution, or contradict supplied
advice. score_lyrics items are limited to supplied sung words, with evidence=[].
Use as few items as needed, at most six. Return {"sentences":[]} if no useful
authorized answer is possible. All safety and output-review requirements apply.
/no_think
""" + MUSIC_TERMINOLOGY


class PreservedSourceAnswer(str):
    """Visible source quotations already identity-checked by the finalizer.

    This internal type preserves a quoted author's register through the final
    display pass. Plain model prose and a prose heading cannot create it.
    """


GENERAL_MUSIC_SYSTEM_PROMPT = """You answer music-theory and singing-practice questions.
No directly answering reviewed source was selected for this request. Use your
general musical knowledge to answer the actual question helpfully and directly.
Definitions, comparisons, note values, rhythm, terminology and optional general
practice suggestions do not require a source passage. Do not refuse these just
because no retrieved source was selected. The selected piece name is context,
not an instruction to discuss that piece in a general concept answer.

Use kind=background and evidence=[] for general explanations. Do not claim that
an expert said them. Do not invent facts about the selected piece, measure,
edition, recording or composer's particular intention. If asked for an unknown
score fact, briefly state what cannot be established; offer a general explanation
only if it helps. Never infer a specific pitch, lyric, notation or source-conflict
cause from general knowledge. Practice suggestions must be optional, not personal
diagnoses or guaranteed outcomes. When score context is present, kind=score_lyrics
may identify only the supplied sung words, without making performance claims.

Give one concise, coherent paragraph that directly answers the question.
Include only details needed to answer the question. Avoid assumed tempos,
numerical demonstrations or extra illustrations unless the question requires
them. Check units and proportions before stating a quantitative relationship.
Do not assume a BPM beat is a quarter note: a seconds-based example must state
which note value is the beat unit, or omit the unsupported numerical example.
Use the requested answer_language. For Korean use formal-polite 합니다체;
for English use natural English and preserve quoted lyric words as supplied.
The application labels background; do not add a heading yourself. If a useful
general answer is impossible, state the missing information briefly. A statement
of that limitation is a valid background item; never invent a fact to fill one.
/no_think
"""

GENERAL_MUSIC_FORMAT_POLICY = """Return one JSON object, no markdown or extra keys:
{"sentences":[{"kind":"background","evidence":[],"text":"Your answer in the requested language."}]}
Write one concise background item for general musical knowledge
or an explicit statement of what cannot be established. evidence must be [].
When actual score context is supplied, score_lyrics may identify only its words.
Only that mixed lyric/explanation case may need two items; keep kinds separate.
Do not print labels, IDs, headings or kind names in prose.
Each text must be nonempty and at most 2000 characters. Do not pad or repeat.
For an unanswerable local fact, state the limitation; do not guess it. A
required text item is not a requirement to make a factual claim. Task, source
and security boundaries still apply to every item.
/no_think
"""


GENERAL_MUSIC_FORMAT_POLICY += "\n" + MUSIC_TERMINOLOGY


class RenderedAnswer(str):
    """Carry validated claim provenance until public answer finalization.

    A prose prefix alone never grants an exemption from citation validation.
    Only parts validated by render_structured_answer receive this metadata.
    """

    def __new__(cls, parts: list[dict[str, Any]], language: str = "ko"):
        grounded = "\n".join(_cite_text(p["text"], p["evidence"])
                             for p in parts if p["kind"] != "background")
        background = "\n".join(p["text"] for p in parts if p["kind"] == "background")
        prefix = "General explanation: " if language == "en" else "일반적인 음악 설명입니다.\n"
        visible = "\n\n".join(x for x in (grounded, prefix + background if background else "") if x)
        if any(p.get("source_quotation") for p in parts):
            visible = "참고 자료의 원문입니다.\n" + visible
        elif any(p.get("source_translation") for p in parts):
            visible = "Translated reference text:\n" + visible
        value = super().__new__(cls, visible)
        value.parts = tuple(parts)
        value.grounded_text = grounded
        value.background_text = background
        kinds = {p["kind"] for p in parts}
        value.answer_basis = (
            "retrieved_evidence_with_background" if kinds >= {"evidence", "background"}
            else "retrieved_evidence" if "evidence" in kinds
            else "score_context_with_background" if kinds >= {"score_lyrics", "background"}
            else "general_music_knowledge" if "background" in kinds
            else "score_context"
        )
        return value

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
# The downstream validator applies NFKC, which turns … into ... and fullwidth
# punctuation into ASCII. Cite these boundaries too, while preserving the
# original lyric typography in displayed text.
_SENTENCE_END_RE = re.compile(r"[.!?。！？…﹒．｡]+[\"'”’」』】)）]*(?=\s|$|[가-힣])")


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
        "minItems": 0,
        "maxItems": len(labels),
        "uniqueItems": True,
    }
    schema = {
        "type": "object",
        "properties": {
            "sentences": {
                "type": "array",
                # Empty-bundle generations otherwise collapsed to [] even for
                # elementary theory questions. Require an explanation OR an
                # explicit limitation; the semantic guard still checks both.
                "minItems": 0 if labels else 1,
                "maxItems": MAX_SENTENCES if labels else (2 if payload.get('Score context') else 1),
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "enum":
                                 (["evidence"] if labels else []) + ["background"] +
                                 (["score_lyrics"] if payload.get("Score context") else [])},
                        "evidence": evidence_schema,
                        "text": {
                            "type": "string",
                            "minLength": 0 if labels and payload.get("source_preservation") and (payload.get("answer_language", "ko") == "ko" or payload.get("separate_source_translation")) else 1,
                            # llama.cpp expands bounded string repetition;
                            # a 2000-character schema bound exceeds its grammar
                            # complexity limit. Enforce that cap in the parser;
                            # generation remains bounded by max_tokens.
                        },
                    },
                    "required": ["kind", "evidence", "text"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["sentences"],
        "additionalProperties": False,
    }
    if payload.get("source_preservation") is True and labels:
        native = payload.get("answer_language", "ko") == "ko" or payload.get("separate_source_translation") is True
        def item(kind, source=False):
            return {
                "type": "object",
                "properties": {
                    "kind": {"const": kind},
                    "evidence": {"type": "array", "items": {"type": "string", "enum": labels},
                                 "minItems": 1 if source else 0, "maxItems": 1 if source else 0},
                    "text": {"const": ""} if source and native else {"type": "string", "minLength": 1},
                },
                "required": ["kind", "evidence", "text"], "additionalProperties": False,
            }
        alternatives = [item("evidence", True), item("background")]
        if payload.get("Score context"):
            alternatives.append(item("score_lyrics"))
        schema["properties"]["sentences"]["items"] = {"oneOf": alternatives}
    return schema


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
    An empty array represents an unsupported answer, never a successful response.
    """
    try:
        parsed = json.loads(raw, object_pairs_hook=_unique_object)
    except (TypeError, ValueError, RecursionError):
        raise SecurityViolation() from None
    if not isinstance(parsed, dict) or set(parsed) != {"sentences"}:
        raise SecurityViolation()
    sentences = parsed["sentences"]
    if not isinstance(sentences, list) or len(sentences) > MAX_SENTENCES:
        raise SecurityViolation()
    allowed_labels = set(_allowed_labels(payload))
    if not sentences:
        return NO_GROUNDED_ANSWER_SENTINEL
    rendered = []
    selected_quotes: set[str] = set()
    preserve = payload.get("source_preservation") is True
    source_by_label = {r.get("citation_label"): r for r in payload.get("Retrieved evidence", [])}
    for item in sentences:
        if not isinstance(item, dict) or set(item) not in ({"text", "evidence"}, {"kind", "text", "evidence"}):
            raise SecurityViolation()
        text, labels = item["text"], item["evidence"]
        # Legacy source-only drafts remain readable; new model output always
        # carries kind under the constrained schema.
        kind = item.get("kind", "evidence" if allowed_labels else "score_lyrics")
        if kind not in {"evidence", "background", "score_lyrics"}:
            raise SecurityViolation()
        native_quote = preserve and kind == "evidence" and (
            payload.get("answer_language", "ko") == "ko" or payload.get("separate_source_translation") is True)
        if (
            not isinstance(text, str)
            or (not text.strip() and not native_quote)
            or len(text) > MAX_SENTENCE_CHARS
            or not isinstance(labels, list)
            or any(not isinstance(label, str) for label in labels)
        ):
            raise SecurityViolation()
        if (
            len(labels) != len(set(labels))
            or not set(labels) <= allowed_labels
            or (kind == "evidence" and not labels)
            or (kind != "evidence" and labels)
            or (kind == "score_lyrics" and not payload.get("Score context"))
        ):
            raise SecurityViolation()
        if preserve and kind == "evidence":
            if len(labels) != 1 or labels[0] in selected_quotes:
                raise SecurityViolation()
            selected_quotes.add(labels[0])
        if native_quote:
            if text != "":
                raise SecurityViolation()
            source = source_by_label[labels[0]].get("answer")
            if (not isinstance(source, str) or not source.strip()
                    or len(source) > MAX_SENTENCE_CHARS or _INLINE_CITATION_RE.search(source)):
                raise SecurityViolation()
            # Text comes from the frozen primary record, never a model quote.
            # Existing security screening and semantic output review still
            # inspect this complete rendered text before it can be released.
            rendered.append({"kind": kind, "text": source, "evidence": labels,
                             "source_quotation": True})
            continue
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
        rendered.append({"kind": kind, "text": text, "evidence": labels})
    return RenderedAnswer(rendered, payload.get("answer_language", "ko"))
