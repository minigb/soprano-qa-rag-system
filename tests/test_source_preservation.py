"""Source quotations preserve conditions and referents without bypassing review."""
import json
import unittest
from unittest.mock import Mock, patch

from soprano_qa import answer, llm
from soprano_qa.security import SecurityViolation
from soprano_qa.structured_answer import (
    RenderedAnswer, PreservedSourceAnswer, render_structured_answer, answer_schema,
)
from tests.test_security_integration import evidence, completion


def selection(label="E1", text="", kind="evidence"):
    return {"kind": kind, "evidence": [label] if label else [], "text": text}


class SourcePreservationTests(unittest.TestCase):
    def prepare(self, text, language="ko"):
        self.results = [evidence(text)]
        self.messages = answer.build_messages("die-forelle", "", "이 부분은 어떻게 부르나요?",
                                              self.results, language=language)
        self.payload = {**json.loads(self.messages[1]["content"]), "source_preservation": True}
        self.payload["answer_language"] = language
        return self.payload

    def render(self, *items):
        return render_structured_answer(json.dumps({"sentences": list(items)}, ensure_ascii=False), self.payload)

    def finalize(self, rendered):
        result = answer.finalize_grounded_generated_answer(rendered, self.results, self.messages, [])
        return answer.finalize_user_visible_answer(result, self.results)

    def test_complete_source_keeps_conditions_interpretation_and_lyric_vowels(self):
        for text in (
            "곡 첫머리의 페르마타가 있는 경우와 없는 경우 모두 허용된다.",
            "로지나의 발랄한 성격을 표현한 것이라고 생각하고 노래하는 것이 좋겠다.",
            "앞 단어의 끝 모음과 뒤 단어의 첫 모음이 모두 e이므로 연음한다. ed의 d는 i에 가깝게 한다.",
            "사선 없는 꾸밈음이 4분음표와 함께 나오면 원음을 16분음표 네 개로 나눈다.",
            "마이크 없이 불러야 할 때는 성악적인 발성이 요구돼요. 상황을 고려하는 것이 중요해요.",
        ):
            with self.subTest(text=text):
                self.prepare(text)
                rendered = self.render(selection())
                self.assertEqual(rendered.parts[0]["text"], text)
                visible = self.finalize(rendered)
                self.assertIsInstance(visible, PreservedSourceAnswer)
                self.assertEqual(visible, "참고 자료의 원문입니다.\n" + text)
                self.assertNotIn("[E1]", visible)

    def test_model_cannot_rewrite_quote_or_claim_quotation_metadata(self):
        self.prepare("조건이 맞을 때만 가볍게 부른다.")
        invalid = [selection(text="항상 가볍게 부른다."), selection("E2"),
                   selection(None), {**selection(), "source_quotation": True}]
        for item in invalid:
            with self.subTest(item=item), self.assertRaises(SecurityViolation):
                self.render(item)
        with self.assertRaises(SecurityViolation):
            self.render(selection(), selection())

    def test_finalizer_checks_identity_against_actual_primary_record(self):
        self.prepare("조건이 맞을 때만 가볍게 부른다.")
        fabricated = RenderedAnswer([{"kind": "evidence", "evidence": ["E1"],
                                    "text": "항상 가볍게 부른다.", "source_quotation": True}])
        with self.assertRaises(answer.GeneratedAnswerRejected):
            self.finalize(fabricated)

    def test_background_stays_generated_and_separately_labelled(self):
        text = "마이크 없이 부를 때는 성악적인 발성이 필요해요."
        self.prepare(text)
        visible = self.finalize(self.render(selection(), selection(None, "연습 속도는 편안하게 조절할 수 있습니다.", "background")))
        self.assertIn(text, visible)
        self.assertIn("일반적인 음악 설명입니다.\n연습 속도는 편안하게 조절할 수 있습니다.", visible)
        with self.assertRaises(SecurityViolation):
            self.render(selection(None, "", "background"))

    def test_english_requires_a_translation_with_one_source_and_still_gets_review(self):
        self.prepare("호흡을 길게 유지한다.", "en")
        with self.assertRaises(SecurityViolation):
            self.render(selection())
        translated = self.render(selection(text="Maintain a long breath."))
        self.assertFalse(translated.parts[0].get("source_quotation"))
        backend = Mock()
        backend.create_chat_completion.side_effect = [completion(x) for x in (
            '{"allow":true}',
            json.dumps({"sentences": [selection(text="Maintain a long breath.")]}),
            '{"allow":true}',
        )]
        with patch.object(llm, "load_llama", return_value=backend):
            result = llm.generate("fake", self.messages, {"source_preservation": "verbatim_or_translation"},
                                  security_context={k: v for k, v in self.payload.items() if k != "source_preservation"})
        self.assertIn("Maintain a long breath.", result)
        calls = backend.create_chat_completion.call_args_list
        self.assertIn("COMPLETE answer", calls[1].kwargs["messages"][0]["content"])
        self.assertIn("draft", json.loads(calls[2].kwargs["messages"][1]["content"]))

    def test_quoted_source_cannot_skip_negative_output_review(self):
        self.prepare("조건이 맞을 때 가볍게 부른다.")
        backend = Mock()
        backend.create_chat_completion.side_effect = [completion(x) for x in (
            '{"allow":true}', json.dumps({"sentences": [selection()]}), '{"allow":false}',
            json.dumps({"sentences": [selection()]}), '{"allow":false}',
        )]
        with patch.object(llm, "load_llama", return_value=backend), self.assertRaises(SecurityViolation) as error:
            llm.generate("fake", self.messages, {"source_preservation": "verbatim_or_translation"},
                         security_context=self.payload)
        self.assertEqual(error.exception.reason, "security_output_rejected")
        self.assertEqual(backend.create_chat_completion.call_count, 5)

    def test_empty_general_bundle_still_requires_nonempty_text(self):
        schema = answer_schema({"source_preservation": True, "answer_language": "ko", "Retrieved evidence": []})
        self.assertEqual(schema["properties"]["sentences"]["items"]["properties"]["text"]["minLength"], 1)


if __name__ == "__main__":
    unittest.main()
