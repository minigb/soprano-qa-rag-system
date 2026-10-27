"""Strict structured composition and citation rendering regressions."""
from __future__ import annotations

import json
import re
import unittest

from soprano_qa.security import SecurityViolation
from soprano_qa.structured_answer import (
    MAX_SENTENCES,
    MAX_SENTENCE_CHARS,
    answer_schema,
    render_structured_answer,
)


PAYLOAD = {
    "Retrieved evidence": [
        {"citation_label": "E1", "answer": "Use a light tone."},
        {"citation_label": None, "answer": "Context only."},
        {"citation_label": "E3", "answer": "Keep the vowel open."},
    ],
}


def draft(text: str, evidence: list[str] | None = None) -> str:
    return json.dumps({"sentences": [{"text": text, "evidence": evidence if evidence is not None else ["E1"]}]})


class AnswerSchemaTests(unittest.TestCase):
    def test_schema_exposes_only_available_primary_labels(self) -> None:
        schema = answer_schema(PAYLOAD)
        sentences = schema["properties"]["sentences"]
        evidence = sentences["items"]["properties"]["evidence"]
        self.assertEqual(evidence["items"]["enum"], ["E1", "E3"])
        self.assertEqual(evidence["minItems"], 1)
        self.assertEqual(evidence["maxItems"], 2)
        self.assertTrue(evidence["uniqueItems"])
        self.assertEqual(sentences["minItems"], 1)
        self.assertEqual(sentences["maxItems"], MAX_SENTENCES)
        self.assertFalse(schema["additionalProperties"])
        self.assertFalse(sentences["items"]["additionalProperties"])

    def test_score_only_schema_requires_empty_evidence(self) -> None:
        schema = answer_schema({"Retrieved evidence": []})
        evidence = schema["properties"]["sentences"]["items"]["properties"]["evidence"]
        self.assertEqual(evidence["minItems"], 0)
        self.assertEqual(evidence["maxItems"], 0)


class StructuredAnswerTests(unittest.TestCase):
    def test_model_selected_labels_are_rendered_per_item(self) -> None:
        raw = json.dumps({"sentences": [
            {"text": "Use a light tone.", "evidence": ["E1"]},
            {"text": "Keep the vowel open.", "evidence": ["E3", "E1"]},
        ]})
        self.assertEqual(
            render_structured_answer(raw, PAYLOAD),
            "Use a light tone. [E1]\nKeep the vowel open. [E3, E1]",
        )

    def test_multiple_sentences_in_one_item_cannot_leave_first_uncited(self) -> None:
        cases = {
            "Sing lightly. Keep the vowel open.": "Sing lightly. [E3] Keep the vowel open. [E3]",
            "가볍게 부릅니다. 모음을 유지합니다.": "가볍게 부릅니다. [E3] 모음을 유지합니다. [E3]",
            "가볍게 부릅니다.모음을 유지합니다.": "가볍게 부릅니다. [E3]모음을 유지합니다. [E3]",
            "First claim\nSecond claim": "First claim [E3]\nSecond claim [E3]",
            'Sing "lightly." Then breathe.': 'Sing "lightly." [E3] Then breathe. [E3]',
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(render_structured_answer(draft(text, ["E3"]), PAYLOAD), expected)

    def test_abbreviations_decimals_and_whitespace_preserve_visible_prose(self) -> None:
        for text in (
            "Dr. Smith recommends a light tone.",
            "Use e.g. a clear vowel. Keep 1.5 beats of silence.",
            "The pitch E4 is sung softly.",
            "Sing lightly.  Breathe here.\n\nContinue the phrase. ",
            "가볍게 부릅니다.모음을 유지합니다.",
        ):
            with self.subTest(text=text):
                rendered = render_structured_answer(draft(text), PAYLOAD)
                self.assertEqual(re.sub(r" \[E1\]", "", rendered), text)
        self.assertIn("1.5 beats", render_structured_answer(draft("Wait 1.5 beats."), PAYLOAD))
        self.assertIn("e.g.", render_structured_answer(draft("Use e.g. a clear vowel."), PAYLOAD))

    def test_invalid_json_and_shape_fail_closed(self) -> None:
        invalid = (
            None, "", "text", "[]", "null", "true", "{}",
            '```json\n{"sentences":[]}\n```',
            '{"sentences":[],"sentences":[{"text":"Text","evidence":["E1"]}]}',
            '{"sentences":[{"text":"A","text":"B","evidence":["E1"]}]}',
            '{"sentences":[],"allow":true}',
            '{"sentences":"not a list"}',
            '{"sentences":[null]}',
            '{"sentences":[{"text":"Text"}]}',
            '{"sentences":[{"text":"Text","evidence":["E1"],"extra":1}]}',
            '{"sentences":[{"text":false,"evidence":["E1"]}]}',
            '{"sentences":[{"text":"Text","evidence":"E1"}]}',
            '{"sentences":[{"text":"Text","evidence":[1]}]}',
            draft(" "),
            draft("Text.") + " Extra prose.",
        )
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(SecurityViolation) as raised:
                render_structured_answer(raw, PAYLOAD)
            self.assertEqual(raised.exception.reason, "security_policy_violation")

    def test_unknown_duplicate_context_only_or_missing_labels_are_rejected(self) -> None:
        for labels in (["E999"], ["E1", "E1"], ["E2"], [], ["E1", "E2"], ["E0"]):
            with self.subTest(labels=labels), self.assertRaises(SecurityViolation):
                render_structured_answer(draft("Use a light tone.", labels), PAYLOAD)

    def test_inline_citation_spoofing_is_rejected(self) -> None:
        for text in ("Sing lightly. [E2]", "Sing [E999] lightly.", "Sing 【E3】.", "Sing [E1, E3]."):
            with self.subTest(text=text), self.assertRaises(SecurityViolation):
                render_structured_answer(draft(text), PAYLOAD)

    def test_redundant_selected_citation_is_normalized_without_new_attribution(self) -> None:
        self.assertEqual(
            render_structured_answer(draft("Sing lightly. [E1]"), PAYLOAD).strip(),
            "Sing lightly. [E1]",
        )
        with self.assertRaises(SecurityViolation):
            render_structured_answer(draft("[E1]"), PAYLOAD)
        with self.assertRaises(SecurityViolation):
            render_structured_answer(draft("Sing. [E1]", []), {"Retrieved evidence": []})

    def test_limits_are_validated_even_if_backend_ignores_schema(self) -> None:
        for raw in (
            json.dumps({"sentences": []}),
            json.dumps({"sentences": [{"text": "Text.", "evidence": ["E1"]}] * (MAX_SENTENCES + 1)}),
            draft("x" * (MAX_SENTENCE_CHARS + 1)),
        ):
            with self.subTest(raw=raw[:80]), self.assertRaises(SecurityViolation):
                render_structured_answer(raw, PAYLOAD)

    def test_score_only_output_cannot_invent_citations(self) -> None:
        payload = {"Retrieved evidence": [], "Score context": "The sung word is Liebe."}
        self.assertEqual(render_structured_answer(draft("The word is Liebe.", []), payload), "The word is Liebe.")
        with self.assertRaises(SecurityViolation):
            render_structured_answer(draft("The word is Liebe.", ["E1"]), payload)


if __name__ == "__main__":
    unittest.main()
