"""Source-locked rendering for finalized expert knowledge units.

The model performs speech-level conversion on one complete finalized unit at
a time.  Retrieval selection, source/result mapping, citations, and multi-unit
presentation remain deterministic application responsibilities.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
import json
import re
import unicodedata
from typing import Any, Callable, Dict, Mapping, Sequence


SOURCE_LOCKED_RENDERER_SYSTEM_PROMPT = """당신은 검토 완료된 한국어 성악 자료 원문의 문체만 바꾸는 교정기입니다. 질문에 답하거나 내용을 요약하는 역할이 아닙니다.

입력에는 하나의 finalized source unit만 문장별 source_sentence로 제공됩니다. 다른 자료 단위나 사용자 질문은 제공되지 않습니다.

반드시 지킬 것:
1. source_sentence 하나를 output sentence 하나로 정확히 한 번 옮기고, 입력 순서와 문장 수를 유지하십시오. 문장을 합치거나 나누지 마십시오.
2. exact_text 전체를 먼저 그대로 복사한 뒤, 문장 맨 끝의 평서형 종결만 합니다체 존댓말로 바꾸십시오. 예: '있다.'→'있습니다.', '좋다.'→'좋습니다.', 권고의 '좋겠다.'→'좋습니다.', '한다.'→'합니다.'입니다.
3. 종결 표현 이외에는 명백한 띄어쓰기와 문장부호만 최소한으로 교정할 수 있습니다. 따옴표 안의 인용문이나 무대 지문은 종결 표현까지 원문 그대로 두고, 바깥 문장의 마지막 종결만 바꾸십시오. 단어ㆍ구ㆍ절을 동의어로 바꾸거나 요약하지 마십시오.
4. 주체, 행위자, 대상, 부정, 조건, 예외, 대안, 인과ㆍ목적 방향, 범위, 가능성ㆍ경향ㆍ허용ㆍ권고ㆍ일반적 해석의 강도를 바꾸지 마십시오.
5. 원문이 어색하거나 관계가 모호해 보여도 새로운 원인ㆍ목적ㆍ효과ㆍ작곡가 의도를 만들어 설명하지 마십시오.
6. 문장 사이에 연결어, 결론, 도입문을 새로 넣지 마십시오. sentence_id, 근거 설명, 마디 범위 안내, 각주를 text에 쓰지 마십시오.

지정된 JSON 형식만 출력하십시오."""

TERMINAL_PUNCTUATION = frozenset(".!?。！？")
TRAILING_SENTENCE_CLOSERS = frozenset("\"'”’」』】)]}")
INTERNAL_LABEL_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:S|U|E)[0-9]+(?![A-Za-z0-9_])",
    flags=re.IGNORECASE,
)
SCHEMA_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:"
    r"finalized_source_unit|source_sentences?|sentence_id|exact_text"
    r")"
    r"(?![A-Za-z0-9_])",
    flags=re.IGNORECASE,
)
PROMPT_CONTROL_TOKEN_RE = re.compile(r"/no_think", flags=re.IGNORECASE)
NUMBER_RE = re.compile(
    r"(?<![0-9A-Za-z])\d+(?:[.,]\d+)*(?![0-9A-Za-z])"
)
FINAL_KOREAN_WORD_RE = re.compile(
    r"(?P<prefix>.*?)(?P<word>[가-힣]+)(?P<punct>[.!?。！？]+[\"'”’」』】)]*)$",
    flags=re.DOTALL,
)
PROTECTED_SEMANTIC_MARKERS = (
    "않",
    "없",
    "아니",
    "못",
    "가능",
    "수 있",
    "수 없",
    "경우",
    "일반적",
    "대체로",
    "권장",
    "권하",
    "추천",
    "허용",
    "금지",
    "피하",
    "주의",
    "반드시",
    "항상",
    "자주",
    "전체",
    "전반",
    "일부",
    "국소",
    "따라",
    "때문",
    "위해",
    "의도",
    "작곡가",
    "말하",
    "언급",
    "해석",
    "해야",
    "좋",
    "중요",
    "과도",
    "모두",
    "또는",
    "혹은",
)


class SourceLockedProtocolError(RuntimeError):
    """Raised when the sole schema-constrained rendering call is invalid."""


@dataclass(frozen=True)
class SourceSentence:
    sentence_id: str
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class SourceLockedUnit:
    unit_id: str
    result_index: int
    evidence_label: str
    source_text: str
    sentences: tuple[SourceSentence, ...]


@dataclass(frozen=True)
class RenderedSentence:
    sentence_id: str
    text: str
    expected_text: str | None = None


@dataclass(frozen=True)
class RenderedSourceLockedUnit:
    unit_id: str
    result_index: int
    evidence_label: str
    source_text: str
    source_sentences: tuple[SourceSentence, ...]
    rendered_sentences: tuple[RenderedSentence, ...]

    @property
    def text(self) -> str:
        return " ".join(item.text for item in self.rendered_sentences)


@dataclass(frozen=True)
class SourceLockedWarning:
    stage: str
    code: str
    message: str
    unit_index: int | None = None
    sentence_index: int | None = None

    def as_dict(self) -> Dict[str, Any]:
        value: Dict[str, Any] = {
            "stage": self.stage,
            "code": self.code,
            "message": self.message,
        }
        if self.unit_index is not None:
            value["unit_index"] = self.unit_index
        if self.sentence_index is not None:
            value["sentence_index"] = self.sentence_index
        return value


def normalize_source_text(text: Any) -> str:
    """Normalize layout once while preserving finalized source wording."""

    collapsed = re.sub(r"\s+", " ", str(text or "")).strip()
    protected_periods = protected_period_positions(collapsed)
    protected_delimited_punctuation = (
        protected_delimited_punctuation_positions(collapsed)
    )
    output: list[str] = []
    index = 0
    while index < len(collapsed):
        character = collapsed[index]
        output.append(character)
        if (
            character in TERMINAL_PUNCTUATION
            and index not in protected_delimited_punctuation
            and not (character == "." and index in protected_periods)
        ):
            next_index = index + 1
            while (
                next_index < len(collapsed)
                and collapsed[next_index] in TRAILING_SENTENCE_CLOSERS
            ):
                output.append(collapsed[next_index])
                next_index += 1
            if (
                next_index < len(collapsed)
                and _is_korean_syllable(collapsed[next_index])
            ):
                output.append(" ")
            if next_index > index + 1:
                index = next_index - 1
        index += 1
    return "".join(output)


def protected_period_positions(text: str) -> set[int]:
    """Return periods belonging to decimals, initials, or abbreviations."""

    protected: set[int] = set()

    def protect_matches(pattern: str, flags: int = 0) -> None:
        for match in re.finditer(pattern, text, flags=flags):
            protected.update(
                index
                for index in range(match.start(), match.end())
                if text[index] == "."
            )

    protect_matches(r"(?<=\d)\.(?=\d)")
    protect_matches(r"(?<=[A-Za-z0-9])\.(?=[A-Za-z0-9])")
    protect_matches(r"(?<![A-Za-z])(?:[A-Za-z]\.){2,}")
    protect_matches(
        r"(?<![A-Za-z])(?:[A-Z]\.\s*){2,}(?=[A-ZÀ-ÖØ-Þ][a-zÀ-ÖØ-öø-ÿ])"
    )
    protect_matches(
        r"\b[A-Z]\.(?=\s+[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'’\-]+)"
    )
    protect_matches(
        r"\b(?:Op|No|Nr|m|mm)\.(?=\s*(?:\d|[IVXLCDM]+\b))",
        flags=re.IGNORECASE,
    )
    protect_matches(
        r"\b(?:p|pp|f|ff)\.(?=\s*(?:[:=]\s*)?\d)",
        flags=re.IGNORECASE,
    )
    protect_matches(
        r"\b(?:dim|cresc|decresc|rit|rall|accel|sfz|smorz)\."
        r"(?=[가-힣])",
        flags=re.IGNORECASE,
    )
    return protected


def protected_delimited_punctuation_positions(text: str) -> set[int]:
    """Protect punctuation inside dates, quotations, and score delimiters.

    This keeps uncertain dates such as ``[185-?]`` intact and, importantly,
    prevents a quoted stage direction followed by ``라고`` from being split
    into separate renderer sentences or having a space inserted before the
    reporting particle.
    """

    protected: set[int] = set()
    stack: list[str] = []
    matching = {
        "[": "]",
        "【": "】",
        "“": "”",
        "「": "」",
        "『": "』",
    }
    for index, character in enumerate(text):
        if character in matching:
            stack.append(matching[character])
            continue
        if stack and character == stack[-1]:
            stack.pop()
            continue
        if stack and character in TERMINAL_PUNCTUATION:
            protected.add(index)
    return protected


def _is_korean_syllable(character: str) -> bool:
    return "가" <= character <= "힣"


def segment_source_sentences(text: Any) -> tuple[SourceSentence, ...]:
    """Split normalized source prose without breaking score abbreviations."""

    normalized = normalize_source_text(text)
    if not normalized:
        return ()
    protected_periods = protected_period_positions(normalized)
    protected_delimited_punctuation = protected_delimited_punctuation_positions(
        normalized
    )
    spans: list[tuple[int, int]] = []
    start = 0
    index = 0
    length = len(normalized)
    while index < length:
        character = normalized[index]
        if (
            character not in TERMINAL_PUNCTUATION
            or index in protected_delimited_punctuation
            or (character == "." and index in protected_periods)
        ):
            index += 1
            continue

        end = index + 1
        while end < length and normalized[end] in TERMINAL_PUNCTUATION:
            if (
                end not in protected_delimited_punctuation
                and (
                    normalized[end] != "."
                    or end not in protected_periods
                )
            ):
                end += 1
            else:
                break
        while end < length and normalized[end] in TRAILING_SENTENCE_CLOSERS:
            end += 1

        next_index = end
        while next_index < length and normalized[next_index].isspace():
            next_index += 1
        is_boundary = (
            next_index >= length
            or next_index > end
            or _is_korean_syllable(normalized[next_index])
        )
        if is_boundary:
            span_start = start
            while span_start < end and normalized[span_start].isspace():
                span_start += 1
            span_end = end
            while span_end > span_start and normalized[span_end - 1].isspace():
                span_end -= 1
            if span_start < span_end:
                spans.append((span_start, span_end))
            start = next_index
            index = next_index
            continue
        index = end

    if start < length:
        span_start = start
        while span_start < length and normalized[span_start].isspace():
            span_start += 1
        if span_start < length:
            spans.append((span_start, length))

    return tuple(
        SourceSentence(
            sentence_id="S%d" % (sentence_index + 1),
            start=span_start,
            end=span_end,
            text=normalized[span_start:span_end],
        )
        for sentence_index, (span_start, span_end) in enumerate(spans)
    )


def build_source_locked_units(
    source_items: Sequence[tuple[int, Any]],
) -> tuple[SourceLockedUnit, ...]:
    """Build prompt-local units from ``(result_index, finalized_answer)``."""

    units: list[SourceLockedUnit] = []
    for result_index, source in source_items:
        normalized = normalize_source_text(source)
        if not normalized:
            continue
        sentences = segment_source_sentences(normalized)
        if not sentences:
            continue
        units.append(
            SourceLockedUnit(
                unit_id="U%d" % (len(units) + 1),
                result_index=result_index,
                evidence_label="E%d" % (result_index + 1),
                source_text=normalized,
                sentences=sentences,
            )
        )
    return tuple(units)


def source_locked_response_format(unit: SourceLockedUnit) -> Dict[str, Any]:
    """Return a strict one-output-sentence-per-source-sentence schema."""

    sentence_ids = [item.sentence_id for item in unit.sentences]
    return {
        "type": "json_object",
        "schema": {
            "type": "object",
            "properties": {
                "sentences": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "sentence_id": {
                                "type": "string",
                                "enum": sentence_ids,
                            },
                            "text": {
                                "type": "string",
                                "minLength": 1,
                            },
                        },
                        "required": ["sentence_id", "text"],
                        "additionalProperties": False,
                    },
                    "minItems": len(sentence_ids),
                    "maxItems": len(sentence_ids),
                }
            },
            "required": ["sentences"],
            "additionalProperties": False,
        },
    }


def build_source_locked_messages(unit: SourceLockedUnit) -> list[Dict[str, str]]:
    """Build an isolated style-transfer prompt with no question or peer KU."""

    payload = {
        "finalized_source_unit": {
            "source_sentences": [
                {
                    "sentence_id": sentence.sentence_id,
                    "exact_text": sentence.text,
                }
                for sentence in unit.sentences
            ]
        }
    }
    return [
        {"role": "system", "content": SOURCE_LOCKED_RENDERER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, indent=2)
            + "\n/no_think",
        },
    ]


def parse_source_locked_render(
    raw: str,
    unit: SourceLockedUnit,
    *,
    formalize_fn: Callable[[str], str],
    polite_check_fn: Callable[[str], bool],
) -> RenderedSourceLockedUnit:
    """Parse, formalize, and structurally validate one rendered full KU."""

    try:
        payload = json.loads(raw.strip())
    except (AttributeError, json.JSONDecodeError) as exc:
        raise SourceLockedProtocolError(
            "Source-locked renderer returned invalid JSON"
        ) from exc
    if not isinstance(payload, dict) or set(payload) != {"sentences"}:
        raise SourceLockedProtocolError(
            "Source-locked renderer returned an invalid top-level object"
        )
    items = payload["sentences"]
    if not isinstance(items, list):
        raise SourceLockedProtocolError(
            "Source-locked renderer sentences must be a list"
        )
    expected_ids = [sentence.sentence_id for sentence in unit.sentences]
    actual_ids = [
        item.get("sentence_id")
        for item in items
        if isinstance(item, dict)
    ]
    if len(actual_ids) != len(items) or actual_ids != expected_ids:
        raise SourceLockedProtocolError(
            "Source-locked renderer must cover every source sentence once "
            "and in order"
        )

    rendered: list[RenderedSentence] = []
    for item_index, item in enumerate(items):
        if set(item) != {"sentence_id", "text"}:
            raise SourceLockedProtocolError(
                "Source-locked renderer sentence has unexpected fields"
            )
        text = item["text"]
        if not isinstance(text, str) or not text.strip():
            raise SourceLockedProtocolError(
                "Source-locked renderer returned an empty sentence"
            )
        finalized = formalize_fn(text).strip()
        source_text = unit.sentences[item_index].text
        if re.search(r"좋겠다([.!?。！？]+[\"'”’」』】)]*)$", source_text):
            finalized = re.sub(
                r"좋겠습니다([.!?。！？]+[\"'”’」』】)]*)$",
                r"좋습니다\1",
                finalized,
            )
        expected_finalized = formalize_fn(source_text).strip()
        if len(segment_source_sentences(finalized)) != 1:
            raise SourceLockedProtocolError(
                "Source-locked renderer split or merged a source sentence"
            )
        if not polite_check_fn(finalized):
            raise SourceLockedProtocolError(
                "Source-locked renderer sentence is not formal-polite Korean"
            )
        if (
            _introduced_protocol_tokens(
                INTERNAL_LABEL_RE,
                source_text,
                finalized,
            )
            or _introduced_protocol_tokens(
                SCHEMA_TOKEN_RE,
                source_text,
                finalized,
            )
            or _introduced_protocol_tokens(
                PROMPT_CONTROL_TOKEN_RE,
                source_text,
                finalized,
            )
        ):
            raise SourceLockedProtocolError(
                "Source-locked renderer exposed an internal protocol label"
            )
        rendered.append(
            RenderedSentence(
                sentence_id=item["sentence_id"],
                text=finalized,
                expected_text=expected_finalized,
            )
        )
    return RenderedSourceLockedUnit(
        unit_id=unit.unit_id,
        result_index=unit.result_index,
        evidence_label=unit.evidence_label,
        source_text=unit.source_text,
        source_sentences=unit.sentences,
        rendered_sentences=tuple(rendered),
    )


def _normalized_fidelity_text(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def _introduced_protocol_tokens(
    pattern: re.Pattern[str],
    source: str,
    rendered: str,
) -> bool:
    """Return whether rendering added a token absent from source prose.

    Catalog identifiers such as ``ML30.8b.S35`` legitimately contain strings
    that resemble temporary sentence labels.  Reject only newly introduced
    protocol tokens instead of rejecting an unchanged finalized source token.
    """

    source_tokens = Counter(
        match.group(0).casefold() for match in pattern.finditer(source)
    )
    rendered_tokens = Counter(
        match.group(0).casefold() for match in pattern.finditer(rendered)
    )
    return any(
        count > source_tokens.get(token, 0)
        for token, count in rendered_tokens.items()
    )


def _content_changed_before_final_word(source: str, rendered: str) -> bool:
    source_match = FINAL_KOREAN_WORD_RE.fullmatch(source.strip())
    rendered_match = FINAL_KOREAN_WORD_RE.fullmatch(rendered.strip())
    if source_match is None or rendered_match is None:
        return _normalized_fidelity_text(source) != _normalized_fidelity_text(rendered)
    source_prefix = _normalized_fidelity_text(source_match.group("prefix"))
    rendered_prefix = _normalized_fidelity_text(rendered_match.group("prefix"))
    if source_prefix != rendered_prefix:
        return True
    source_word = source_match.group("word")
    rendered_word = rendered_match.group("word")
    if source_word == "좋겠다" and rendered_word == "좋습니다":
        return False
    common = 0
    for source_character, rendered_character in zip(source_word, rendered_word):
        if source_character != rendered_character:
            break
        common += 1
    return common < 1


def source_locked_fidelity_warnings(
    unit: RenderedSourceLockedUnit,
) -> list[SourceLockedWarning]:
    """Return non-blocking fidelity warnings for one complete rendered KU."""

    warnings: list[SourceLockedWarning] = []
    for sentence_index, (source, rendered) in enumerate(
        zip(unit.source_sentences, unit.rendered_sentences),
        start=1,
    ):
        normalized_source = unicodedata.normalize("NFKC", source.text)
        normalized_rendered = unicodedata.normalize("NFKC", rendered.text)
        normalized_expected = unicodedata.normalize(
            "NFKC",
            rendered.expected_text or source.text,
        )
        if rendered.expected_text is not None:
            content_changed = (
                _normalized_fidelity_text(normalized_expected)
                != _normalized_fidelity_text(normalized_rendered)
            )
        else:
            content_changed = _content_changed_before_final_word(
                normalized_source,
                normalized_rendered,
            )
        if content_changed:
            warnings.append(
                SourceLockedWarning(
                    stage="source_fidelity",
                    code="source_text_changed",
                    message=(
                        "Rendered sentence changes source wording outside the "
                        "sentence-final polite conversion"
                    ),
                    unit_index=int(unit.unit_id[1:]),
                    sentence_index=sentence_index,
                )
            )
        ratio = SequenceMatcher(
            None,
            _normalized_fidelity_text(normalized_expected),
            _normalized_fidelity_text(normalized_rendered),
        ).ratio()
        if content_changed and ratio < 0.90:
            warnings.append(
                SourceLockedWarning(
                    stage="source_fidelity",
                    code="source_fidelity_low",
                    message="Rendered sentence has low lexical fidelity (%.3f)" % ratio,
                    unit_index=int(unit.unit_id[1:]),
                    sentence_index=sentence_index,
                )
            )
        source_numbers = NUMBER_RE.findall(normalized_expected)
        rendered_numbers = NUMBER_RE.findall(normalized_rendered)
        if source_numbers != rendered_numbers:
            warnings.append(
                SourceLockedWarning(
                    stage="source_fidelity",
                    code="numeric_content_changed",
                    message="Rendered sentence changes numeric source content",
                    unit_index=int(unit.unit_id[1:]),
                    sentence_index=sentence_index,
                )
            )
        changed_markers = [
            marker
            for marker in PROTECTED_SEMANTIC_MARKERS
            if normalized_expected.count(marker)
            != normalized_rendered.count(marker)
        ]
        if changed_markers:
            warnings.append(
                SourceLockedWarning(
                    stage="source_fidelity",
                    code="protected_marker_changed",
                    message=(
                        "Rendered sentence changes protected semantic markers: "
                        + ", ".join(changed_markers)
                    ),
                    unit_index=int(unit.unit_id[1:]),
                    sentence_index=sentence_index,
                )
            )
    return warnings


PIECE_TITLE_PREFIX_PARTICLES = r"(?:에서는|에서|은|는|의|을|를|이|가)?"


def _first_piece_title(text: str) -> str | None:
    match = re.match(
        r"^(《[^》]+》)%s\s+" % PIECE_TITLE_PREFIX_PARTICLES,
        text,
    )
    return match.group(1) if match else None


def _strip_exact_piece_title_topic_prefix(text: str, title: str) -> str:
    return re.sub(
        r"^%s%s\s+"
        % (re.escape(title), PIECE_TITLE_PREFIX_PARTICLES),
        "",
        text,
        count=1,
    )


def format_source_locked_answer(
    units: Sequence[RenderedSourceLockedUnit],
) -> str:
    """Format retained units without a model-driven cross-unit rewrite."""

    if not units:
        return ""
    if len(units) < 2:
        return " ".join(unit.text for unit in units)
    lead_in = "다음 사항을 함께 참고하시면 좋습니다."
    retained_title: str | None = None
    bullets: list[str] = []
    for unit in units:
        text = unit.text
        title = _first_piece_title(text)
        if title is not None:
            if retained_title == title:
                text = _strip_exact_piece_title_topic_prefix(text, title)
            elif retained_title is None:
                retained_title = title
        bullets.append("- " + text)
    return lead_in + "\n\n" + "\n".join(bullets)


def warning_dicts(
    warnings: Sequence[SourceLockedWarning],
) -> list[Dict[str, Any]]:
    return [warning.as_dict() for warning in warnings]


def legacy_warning_text(
    warnings: Sequence[SourceLockedWarning],
) -> str | None:
    if not warnings:
        return None
    return "; ".join(
        "%s: %s" % (warning.code, warning.message)
        for warning in warnings
    )


def copy_llm_settings(settings: Mapping[str, Any]) -> Dict[str, Any]:
    """Use deterministic generation and enough space for a complete full KU."""

    value = dict(settings)
    value["temperature"] = 0.0
    value["max_tokens"] = max(1024, int(value.get("max_tokens", 768)))
    return value
