"""Translate an English question into Korean for the Korean-only retriever.

The reviewed corpus, lexical concepts, aliases, and dense-retrieval
calibration are Korean. An English question is therefore translated once,
with the same local model and settings used for answer generation, and only
the translation is used for retrieval and Korean-oriented gating. The
original English question remains authoritative for measure-contract
validation and for the generation prompt.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List

from soprano_qa.llm import generate


SUPPORTED_LANGUAGES = ("ko", "en")

QUERY_TRANSLATION_SYSTEM_PROMPT = """You translate a singer's English question about a soprano piece into Korean.
The Korean translation is used only to search Korean expert commentary.
Rules:
- Translate faithfully and completely. Do not answer, explain, expand, shorten, or correct the question.
- Keep piece titles, lyric words, quoted text, person names, and foreign musical terms
  (Italian, German, French, Latin, and so on) exactly as written in the question.
- Keep every bar or measure number exactly as written, for example "bar 12" as "12마디" and
  "bars 2-5" as "2-5마디". Never add a number or placeholder that the question does not contain;
  "these bars" is "이 마디들".
- Use standard Korean music terms: a verse or stanza of the poem is "절" (for example, "the third
  verse" is "3절"); introduction or prelude "전주"; interlude "간주"; grace note or ornament
  "꾸밈음"; dotted rhythm "부점 리듬"; double-dotted "겹부점"; meter or time signature "박자";
  the sung text or lyrics "가사"; breath "호흡".
- Output only the Korean translation as plain text: no quotation marks, notes, labels,
  alternatives, or romanization.
"""


class QueryTranslationFailed(RuntimeError):
    """Raised when the model returns no usable Korean retrieval query."""


def validate_language(language: str) -> str:
    """Return a supported answer language or raise ``ValueError``."""

    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            "Unsupported language %r; expected one of %s"
            % (language, ", ".join(SUPPORTED_LANGUAGES))
        )
    return language


def build_translation_messages(question: str) -> List[Dict[str, str]]:
    """Return the strict chat prompt for one English-to-Korean translation."""

    return [
        {"role": "system", "content": QUERY_TRANSLATION_SYSTEM_PROMPT},
        {"role": "user", "content": "%s\n/no_think" % question.strip()},
    ]


def _clean_translation(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(
        r"^(?:korean(?:\s+translation)?|translation|번역)\s*[:：]\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    # Keep only the first non-empty line: the contract asks for one line,
    # and a trailing note must never enter retrieval.
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    cleaned = lines[0] if lines else ""
    pairs = (('"', '"'), ("“", "”"), ("'", "'"), ("‘", "’"), ("「", "」"))
    for opening, closing in pairs:
        if (
            len(cleaned) >= 2
            and cleaned.startswith(opening)
            and cleaned.endswith(closing)
        ):
            cleaned = cleaned[1:-1].strip()
            break
    return cleaned


def translate_question_to_korean(
    question: str,
    *,
    model_path: str,
    llm_settings: Dict[str, Any],
    generate_fn: Callable[..., str] = generate,
) -> str:
    """Translate ``question`` with exactly one local-model call.

    The call uses the configured generation settings (temperature 0 in the
    production configuration). A translation without Hangul is rejected
    rather than silently retrieving with English text that the Korean
    calibration was never tuned for.
    """

    if not question.strip():
        raise QueryTranslationFailed("Cannot translate an empty question")
    raw = generate_fn(
        model_path,
        build_translation_messages(question),
        llm_settings,
    )
    translation = _clean_translation(raw)
    if not translation or not re.search(r"[가-힣]", translation):
        raise QueryTranslationFailed(
            "Question translation returned no Korean text: %r" % raw
        )
    return translation
