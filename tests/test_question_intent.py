import json
import unittest
from unittest.mock import Mock, patch

from soprano_qa import llm
from soprano_qa.question_intent import is_explicit_general_question
from soprano_qa.security import SecurityViolation
from tests.test_security_integration import completion


class GeneralIntentTests(unittest.TestCase):
    def test_direct_question_match_preserves_an_available_definition(self):
        payload = {"question": "What is syncopation?", "Retrieved evidence": [
            {"citation_label": "E2", "question_alias_match": True, "answer": "A definition."}]}
        self.assertFalse(is_explicit_general_question(payload))
        payload["Retrieved evidence"][0]["question_alias_match"] = False
        self.assertTrue(is_explicit_general_question(payload))
        payload["Retrieved evidence"][0].update(citation_label=None, question_alias_match=True)
        self.assertTrue(is_explicit_general_question(payload))

    def test_definitions_and_universal_rules_without_local_referents(self):
        for question in (
            "레가토와 포르타멘토는 어떤 차이가 있나요?",
            "페르마타는 언제나 정확히 세 배 길이인가요?",
            "이조의 정의를 알려 주세요.",
            "What is syncopation?", "Define counterpoint.",
            "Must a fermata always last three times the written duration?",
        ):
            with self.subTest(question=question):
                self.assertTrue(is_explicit_general_question({"question": question}))
                self.assertTrue(is_explicit_general_question({"question": question, "measure_range": "(none selected)"}))
                self.assertFalse(is_explicit_general_question({"question": question, "measure_range": "12-13"}))

    def test_local_causal_and_ambiguous_short_questions_keep_sources(self):
        for question in (
            "작은 음표들은 무엇일까?", "사선 없는 꾸밈음은 어떻게 부르나요?",
            "이 곡의 간주는 어떤 뜻인가요?", "악보와 음원의 차이는 왜 생기나요?",
            "What is the difference between these two notations?",
            "What is the purpose of the interlude in this piece?",
            "Why does the composer use an accent?",
            "How should I practice the interlude?",
        ):
            with self.subTest(question=question):
                self.assertFalse(is_explicit_general_question({"question": question}))

    def test_general_routing_retains_admission_and_output_guards(self):
        payload = {"question": "Define syncopation.", "answer_language": "en",
                   "Retrieved evidence": [{"citation_label": "E1", "answer": "Unrelated advice."}]}
        messages = [{"role": "system", "content": "Music."},
                    {"role": "user", "content": json.dumps(payload)}]
        backend = Mock()
        draft = json.dumps({"sentences": [{"kind": "background", "evidence": [],
                                           "text": "Syncopation displaces expected rhythmic accents."}]})
        backend.create_chat_completion.side_effect = [completion(x) for x in
            ('{"allow":true}', draft, '{"allow":false}', draft, '{"allow":false}')]
        with patch.object(llm, "load_llama", return_value=backend), \
             patch.object(llm, "select_question_evidence") as selector, \
             self.assertRaises(SecurityViolation):
            llm.generate("fake", messages, {"source_preservation": "verbatim_or_translation"},
                         security_context=payload)
        selector.assert_not_called()
        self.assertEqual(backend.create_chat_completion.call_count, 5)
        composed = backend.create_chat_completion.call_args_list[1].kwargs["messages"]
        self.assertEqual(json.loads(composed[1]["content"])["Retrieved evidence"], [])
        self.assertEqual(len(payload["Retrieved evidence"]), 1)


if __name__ == "__main__":
    unittest.main()
