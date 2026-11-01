import json
import unittest
from unittest.mock import Mock, patch

from soprano_qa import llm
from soprano_qa.question_intent import explicit_general_intent, is_explicit_general_question
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
            "Does rubato mean abandoning the pulse?",
            "Is transposition the same as octave displacement?",
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
             patch.object(llm, "select_question_evidence", return_value={**payload, "Retrieved evidence": []}) as selector, \
             self.assertRaises(SecurityViolation):
            llm.generate("fake", messages, {"source_preservation": "verbatim_or_translation"},
                         security_context=payload)
        selector.assert_called_once()
        self.assertTrue(selector.call_args.kwargs["assess_single"])
        self.assertEqual(backend.create_chat_completion.call_count, 5)
        composed = backend.create_chat_completion.call_args_list[1].kwargs["messages"]
        self.assertEqual(json.loads(composed[1]["content"])["Retrieved evidence"], [])
        self.assertEqual(len(payload["Retrieved evidence"]), 1)

    def test_definition_selects_actual_single_source_before_composition(self):
        for labels in (["E1"], []):
            with self.subTest(labels=labels):
                source = {"citation_label": "E1", "answer": "당김음은 예상되는 박의 강세를 옮기는 리듬입니다."}
                payload = {"question": "당김음의 정의를 알려 주세요.", "answer_language": "ko",
                           "Retrieved evidence": [source]}
                messages = [{"role": "system", "content": "Music."},
                            {"role": "user", "content": json.dumps(payload)}]
                draft = {"kind": "evidence", "evidence": labels, "text": ""} if labels else {
                    "kind": "background", "evidence": [], "text": "당김음은 예상되는 강세를 옮깁니다."}
                backend = Mock()
                backend.create_chat_completion.side_effect = [completion(x) for x in (
                    '{"allow":true}', json.dumps({"labels": labels}),
                    json.dumps({"sentences": [draft]}), '{"allow":true}')]
                with patch.object(llm, "load_llama", return_value=backend):
                    result = llm.generate("fake", messages, {"source_preservation": "verbatim_or_translation"},
                                          security_context=payload)
                calls = backend.create_chat_completion.call_args_list
                selection_input = json.loads(calls[1].kwargs["messages"][1]["content"])
                self.assertEqual(selection_input["passages"][0]["answer"], source["answer"])
                composed = json.loads(calls[2].kwargs["messages"][1]["content"])
                self.assertEqual(composed["Retrieved evidence"], [source] if labels else [])
                self.assertIn(source["answer"] if labels else draft["text"], result)
                self.assertEqual(payload["Retrieved evidence"], [source])

    def test_named_comparison_precedes_definition_and_universal_branches(self):
        for question in ("새로운 노래에서는 두 연주 방식의 차이가 무엇인가요?",
                         "When singing My New Song, should I always sing the same way? Compare the settings."):
            with self.subTest(question=question):
                self.assertIsNone(explicit_general_intent({
                    "question": question, "piece": "my-new-song",
                    "Retrieved evidence": [{"piece_title": "새로운 노래"}]}))
        self.assertEqual(explicit_general_intent({"question": "Must I always use the same approach?"}), "universal")
        self.assertEqual(explicit_general_intent({"question": "Define syncopation."}), "definition")

    def test_rejected_definition_source_uses_one_guarded_general_retry(self):
        payload = {"question": "Define syncopation.", "answer_language": "en",
                   "Retrieved evidence": [{"citation_label": "E1", "answer": "An unrelated practice suggestion."}]}
        messages = [{"role": "system", "content": "Music."},
                    {"role": "user", "content": json.dumps(payload)}]
        source = json.dumps({"sentences": [{"kind": "evidence", "evidence": ["E1"],
                                          "text": "An unrelated practice suggestion."}]})
        background = json.dumps({"sentences": [{"kind": "background", "evidence": [],
                                               "text": "Syncopation displaces expected rhythmic accents."}]})
        for final_allow in (True, False):
            with self.subTest(final_allow=final_allow):
                backend = Mock()
                backend.create_chat_completion.side_effect = [completion(x) for x in (
                    '{"allow":true}', '{"labels":["E1"]}', source, '{"allow":false}',
                    background, json.dumps({"allow": final_allow}))]
                with patch.object(llm, "load_llama", return_value=backend):
                    if final_allow:
                        result = llm.generate("fake", messages, {"source_preservation": "verbatim_or_translation"},
                                              security_context=payload)
                        self.assertIn("Syncopation displaces", result)
                        self.assertNotIn("unrelated", result)
                    else:
                        with self.assertRaises(SecurityViolation):
                            llm.generate("fake", messages, {"source_preservation": "verbatim_or_translation"},
                                         security_context=payload)
                calls = backend.create_chat_completion.call_args_list
                self.assertEqual(len(calls), 6)
                self.assertEqual(json.loads(calls[4].kwargs['messages'][1]['content'])['Retrieved evidence'], [])
                self.assertIn('draft', json.loads(calls[5].kwargs['messages'][1]['content']))
                self.assertEqual(len(payload['Retrieved evidence']), 1)

    def test_explicit_practice_scenarios_without_local_referents(self):
        for question in (
            "연습할 때 소리를 어떻게 점검하면 좋을까요?",
            "발성을 훈련할 때 어떤 점에 집중하면 좋나요?",
            "When learning a melody, what should I check?",
            "What should I notice while practising a new scale?",
        ):
            with self.subTest(question=question):
                payload = {"question": question}
                self.assertTrue(is_explicit_general_question(payload))
                self.assertFalse(is_explicit_general_question({**payload, "measure_range": "9-12"}))
                self.assertFalse(is_explicit_general_question({**payload, "Retrieved evidence": [
                    {"citation_label": "E1", "question_alias_match": True}]}))

    def test_local_practice_and_named_work_keep_their_sources(self):
        for question in (
            "이 악보를 연습할 때 무엇을 확인하나요?", "이 가사를 연습할 때 어떤 점이 중요한가요?",
            "When practising this aria, what should I check?",
            "What should I notice when learning these lyrics?",
        ):
            with self.subTest(question=question):
                self.assertFalse(is_explicit_general_question({"question": question}))
        self.assertFalse(is_explicit_general_question({
            "question": "When learning My New Song, what should I check?", "piece": "my-new-song"}))
        self.assertFalse(is_explicit_general_question({
            "question": "새로운 노래를 연습할 때 무엇을 확인하나요?", "piece": "new-song",
            "Retrieved evidence": [{"work": "A Composer - A New Song", "piece_title": "새로운 노래", "citation_label": "E1"}]}))


if __name__ == "__main__":
    unittest.main()
