"""Default RAG answers must be authored prose, with all output checks intact."""
import json
import unittest
from unittest.mock import Mock, patch

from soprano_qa import answer, llm
from soprano_qa.security import SecurityViolation
from soprano_qa.security import select_question_evidence
from soprano_qa.settings import load_settings
from soprano_qa.structured_answer import synthesis_evidence_view, render_structured_answer
from tests.test_security_integration import completion, evidence


class SemanticSynthesisTests(unittest.TestCase):
    def test_format_placeholder_is_not_released_as_a_music_answer(self):
        from soprano_qa.response_policy import compose_music_answer
        messages = answer.build_messages('die-forelle', '', 'How can I practise phrasing?', [], language='en')
        generator = Mock(side_effect=['General explanation: Your answer in the requested language.',
            'You can first speak the text in phrases, then sing it while keeping those connections.'])
        result = compose_music_answer('fake', messages, {}, [], [], language='en',
            generate_fn=generator, finalizer_fn=lambda draft, *args, **kwargs: str(draft))
        self.assertEqual(result.generation_mode, 'llm_general_fallback')
        self.assertNotIn('Your answer in the requested language', result)
        self.assertIn('speak the text', result)
        self.assertTrue(generator.call_args.kwargs['general_answer_fallback'])

    def test_general_fallback_policy_reaches_the_actual_writer_after_routing(self):
        self.prepare()
        messages = answer.build_messages('die-forelle', '', '편하게 호흡하려면 어떻게 연습하나요?', [])
        payload = json.loads(messages[1]['content'])
        model = Mock()
        model.create_chat_completion.side_effect = [completion('{"task":"music_question"}'),
            completion(json.dumps({'sentences': [{'kind': 'background', 'evidence': [],
                'text': '편안한 음역에서 짧게 노래하며 호흡이 고르게 이어지는지 살펴볼 수 있습니다.'}]})),
            completion('{"allow":true}')]
        with patch.object(llm, 'load_llama', return_value=model):
            result = llm.generate('fake', messages, self.settings, security_context=payload,
                                  general_answer_fallback=True)
        self.assertIn('호흡이 고르게', result)
        actual_prompt = model.create_chat_completion.call_args_list[1].kwargs['messages'][0]['content']
        self.assertIn('Do not stop at saying information is missing', actual_prompt)
        self.assertIn('Do not invent a local score fact', actual_prompt)

    def test_quality_denial_can_return_safety_checked_prose_after_one_rewrite(self):
        self.prepare()
        prose = "반주가 없는 구간에서는 자신의 음정을 주의 깊게 듣는 것이 좋습니다."
        draft = completion(json.dumps({"sentences": [{"kind": "evidence", "evidence": ["E1"], "text": prose}]}))
        model = Mock()
        model.create_chat_completion.side_effect = [completion('{"task":"music_question"}'),
            draft, completion('{"allow":false}'), draft, completion('{"allow":false}'),
            completion('{"allow":true}')]
        relaxed = Mock(return_value=prose + " 일부 세부 내용은 추가 확인이 필요합니다.")
        with patch.object(llm, "load_llama", return_value=model):
            result = llm.generate("fake", self.messages, self.settings, security_context=self.payload,
                                  warning_finalizer=relaxed)
        self.assertEqual(result.generation_mode, "llm_with_warning")
        self.assertEqual(result.generation_validation_warning, "output_quality_review")
        self.assertIn(prose, result.validated_text)
        self.assertEqual(model.create_chat_completion.call_count, 6)
        self.assertIn("Review ONLY", model.create_chat_completion.call_args.kwargs['messages'][0]['content'])

    def test_repeated_scope_failure_keeps_safe_answer_and_warning(self):
        self.prepare()
        prose = "조건에 따라 반주의 흐름에 맞춰 마지막 음을 마무리합니다."
        draft = completion(json.dumps({"sentences": [{"kind": "evidence", "evidence": ["E1"], "text": prose}]}))
        model = Mock()
        model.create_chat_completion.side_effect = [completion('{"task":"music_question"}'),
            draft, completion('{"allow":true}'), draft, completion('{"allow":true}'), completion('{"allow":true}')]
        finalizer = Mock(side_effect=answer.GeneratedAnswerRejected("Generated answer overgeneralizes local evidence"))
        with patch.object(llm, "load_llama", return_value=model):
            result = llm.generate("fake", self.messages, self.settings, security_context=self.payload,
                                  finalizer=finalizer, warning_finalizer=lambda draft: prose + " 일부 세부 내용은 추가 확인이 필요합니다.")
        self.assertEqual(finalizer.call_count, 2)
        self.assertEqual(result.generation_validation_warning, "answer_validation_warning")
        self.assertIn(prose, result.validated_text)

    def test_native_reasoning_is_writer_only_and_prefilled_thoughts_stay_private(self):
        self.prepare()
        self.settings = {**self.settings, "chat_format": "native_controlled", "synthesis_reasoning": True}
        model = Mock()
        model._soprano_native_chat_handlers = {False: "direct", True: "reasoned"}
        prose = "반주가 없는 구간에서는 자신의 음정을 주의 깊게 듣는 것이 좋습니다."
        draft = 'PRIVATE_REASONING_SENTINEL</think>\n' + json.dumps({"sentences": [
            {"kind": "evidence", "evidence": ["E1"], "text": prose}]})
        replies = iter([completion('{"task":"music_question"}'), completion(draft), completion('{"allow":true}')])
        handlers = []
        def complete(**kwargs):
            handlers.append(model.chat_handler)
            return next(replies)
        model.create_chat_completion.side_effect = complete
        with patch.object(llm, "load_llama", return_value=model):
            generated = llm.generate("fake", self.messages, self.settings, security_context=self.payload)
        self.assertEqual(handlers, ["direct", "reasoned", "direct"])
        self.assertNotIn("PRIVATE_REASONING_SENTINEL", str(generated))
        self.assertNotIn("PRIVATE_REASONING_SENTINEL", str(model.create_chat_completion.call_args))
        self.assertEqual(answer.finalize_grounded_generated_answer(generated, self.results, self.messages, []), prose)
        for call in model.create_chat_completion.call_args_list:
            self.assertNotRegex(call.kwargs["messages"][0]["content"], r"(?m)^/(?:no_)?think\s*$")

    def test_native_reasoning_without_an_end_marker_cannot_be_delivered(self):
        self.prepare()
        self.settings = {**self.settings, "chat_format": "native_controlled", "synthesis_reasoning": True}
        model = Mock()
        model._soprano_native_chat_handlers = {False: "direct", True: "reasoned"}
        model.create_chat_completion.side_effect = [completion('{"task":"music_question"}'),
                                                    completion('PRIVATE_UNFINISHED_REASONING')]
        with patch.object(llm, "load_llama", return_value=model), self.assertRaises(SecurityViolation):
            llm.generate("fake", self.messages, self.settings, security_context=self.payload)
        self.assertEqual(model.create_chat_completion.call_count, 2)

    def test_reasoned_synthesis_still_validates_only_the_final_composition(self):
        self.prepare()
        self.settings = {**self.settings, "synthesis_reasoning": True}
        self.settings.pop("chat_format", None)
        model = Mock()
        prose = "반주가 없는 구간에서는 자신의 음정을 주의 깊게 듣는 것이 좋습니다."
        draft = '<think>PRIVATE_REASONING_SENTINEL</think>```json\n' + json.dumps({"sentences": [
            {"kind": "evidence", "evidence": ["E1"], "text": prose}]}) + '\n```'
        model.create_chat_completion.side_effect = [completion('{"task":"music_question"}'),
            completion(draft), completion('{"allow":true}')]
        with patch.object(llm, "load_llama", return_value=model):
            generated = llm.generate("fake", self.messages, self.settings, security_context=self.payload)
        calls = model.create_chat_completion.call_args_list
        self.assertNotIn("response_format", calls[1].kwargs)
        self.assertEqual(calls[1].kwargs["temperature"], 0.6)
        for call in (calls[0], calls[2]):
            self.assertEqual(call.kwargs["temperature"], 0.0)
            self.assertEqual(call.kwargs["max_tokens"], 64)
        self.assertNotIn("PRIVATE_REASONING_SENTINEL", str(generated))
        self.assertNotIn("PRIVATE_REASONING_SENTINEL", str(calls[2]))
        self.assertEqual(answer.finalize_grounded_generated_answer(
            generated, self.results, self.messages, []), prose)

    def test_reviewed_score_lyrics_do_not_need_a_fabricated_ku_citation(self):
        self.prepare()
        payload = {**self.payload, "Score context": {"measures": [{"measure": 8, "lyric_lines": ["la"]}]}}
        messages = [self.messages[0], {"role": "user", "content": json.dumps(payload)}]
        prose = "반주가 없는 부분에서는 자신의 음정을 주의 깊게 듣는 것이 좋습니다."
        parts = [{"kind": "evidence", "evidence": ["E1"], "text": prose},
                 {"kind": "score_lyrics", "evidence": [], "text": "8마디의 가사는 'la'입니다."}]
        generated = render_structured_answer(json.dumps({"sentences": parts}), payload)
        result = answer.finalize_grounded_generated_answer(generated, self.results, messages, [[8, 8]])
        self.assertEqual(result, prose + "\n\n8마디의 가사는 'la'입니다.")
        with self.assertRaises(answer.GeneratedAnswerRejected):
            answer.finalize_grounded_generated_answer(generated, self.results, messages, [[9, 9]])
        with self.assertRaises(answer.GeneratedAnswerRejected):
            answer.finalize_grounded_generated_answer(generated, self.results, self.messages, [[8, 8]])

    def test_dense_note_passage_is_not_a_claim_about_many_passages(self):
        self.assertFalse(answer._claim_has_piece_wide_generalization(
            "짧은 음표가 많은 부분은 천천히 연습하면 도움이 됩니다."))
        self.assertTrue(answer._claim_has_piece_wide_generalization(
            "많은 부분에서 짧은 음표가 나오므로 천천히 연습합니다."))
        self.assertTrue(answer._claim_has_piece_wide_generalization(
            "음표가 많은 부분에서는 곡 전체에 걸쳐 같은 연습법을 사용합니다."))

    def test_finalization_failure_uses_one_reviewed_recomposition_without_echoing_error_text(self):
        self.prepare()
        text = "반주가 없는 부분에서는 자신의 음정을 주의 깊게 듣는 것이 좋습니다."
        draft = completion(json.dumps({"sentences": [{"kind": "evidence", "evidence": ["E1"], "text": text}]}))
        model = Mock()
        model.create_chat_completion.side_effect = [completion('{"task":"music_question"}'),
            draft, completion('{"allow":true}'), draft, completion('{"allow":true}')]
        finalizer = Mock(side_effect=[answer.GeneratedAnswerRejected(
            "Generated answer overgeneralizes local evidence. PRIVATE_DIAGNOSTIC_SENTINEL"), "Validated answer"])
        with patch.object(llm, "load_llama", return_value=model):
            result = llm.generate("fake", self.messages, self.settings, security_context=self.payload,
                                  finalizer=finalizer)
        self.assertEqual(result.validated_text, "Validated answer")
        self.assertEqual(finalizer.call_count, 2)
        self.assertEqual(model.create_chat_completion.call_count, 5)
        retry = model.create_chat_completion.call_args_list[3].kwargs["messages"]
        self.assertIn("exceeded its cited scope", retry[0]["content"])
        self.assertNotIn("PRIVATE_DIAGNOSTIC_SENTINEL", json.dumps(retry))
        first = model.create_chat_completion.call_args_list[1].kwargs["messages"]
        self.assertEqual(first[1], retry[1])

    def test_semantic_and_finalization_failures_share_a_single_retry_budget(self):
        self.prepare()
        draft = completion('{"sentences":[{"kind":"evidence","evidence":["E1"],"text":"출처에 근거한 답변입니다."}]}')
        model = Mock()
        model.create_chat_completion.side_effect = [completion('{"task":"music_question"}'),
            draft, completion('{"allow":false}'), draft, completion('{"allow":true}')]
        finalizer = Mock(side_effect=answer.GeneratedAnswerRejected("Generated answer overgeneralizes local evidence"))
        with patch.object(llm, "load_llama", return_value=model), self.assertRaises(answer.GeneratedAnswerRejected):
            llm.generate("fake", self.messages, self.settings, security_context=self.payload, finalizer=finalizer)
        self.assertEqual(finalizer.call_count, 1)
        self.assertEqual(model.create_chat_completion.call_count, 5)

    def test_security_failure_in_finalization_never_triggers_recomposition(self):
        self.prepare()
        draft = completion('{"sentences":[{"kind":"evidence","evidence":["E1"],"text":"출처에 근거한 답변입니다."}]}')
        model = Mock()
        model.create_chat_completion.side_effect = [completion('{"task":"music_question"}'),
            draft, completion('{"allow":true}')]
        with patch.object(llm, "load_llama", return_value=model), self.assertRaises(SecurityViolation):
            llm.generate("fake", self.messages, self.settings, security_context=self.payload,
                         finalizer=Mock(side_effect=SecurityViolation()))
        self.assertEqual(model.create_chat_completion.call_count, 3)

    def test_relevant_disagreeing_accounts_survive_selection_without_bypassing_verdict_validation(self):
        from tests.test_source_synthesis import context
        payload = context()
        selected = select_question_evidence("fake", {"source_preservation": "semantic"}, payload,
                                            generate_fn=Mock(return_value='{"labels":["E2"]}'))
        self.assertEqual(selected["Retrieved evidence"], payload["Retrieved evidence"])
        with self.assertRaises(SecurityViolation):
            select_question_evidence("fake", {"source_preservation": "semantic"}, payload,
                                     generate_fn=Mock(return_value='{"labels":["E999"]}'))
        # Empty relevance decisions and explicit general definitions retain
        # their normal route; source conflicts never invent label authority.
        for decision, definition, expected in [('[]', False, []), ('["E2"]', True, [payload["Retrieved evidence"][0]])]:
            selected = select_question_evidence("fake", {"source_preservation": "semantic"}, payload,
                        assess_single=definition, generate_fn=Mock(return_value='{"labels":'+decision+'}'))
            self.assertEqual(selected["Retrieved evidence"], expected)

    def test_alternative_conditions_and_disagreement_are_data_not_a_canned_answer(self):
        source = "혼자 연습하거나 반주가 없는 경우에는 녹음으로 확인할 수 있다."
        payload = {"question": "첫 반주의 다이나믹은?", "answer_language": "ko",
                   "Retrieved evidence": [
                       {"citation_label": "E7", "answer": "시작 반주가 mp로 되어 있습니다. 첫소리를 크지 않게 시작하는 것이 좋습니다."},
                       {"citation_label": "E9", "answer": "반주는 p로 시작합니다. 첫소리를 크지 않게 시작하는 것이 좋습니다."}]}
        with patch("soprano_qa.source_comparison.opening_dynamics_comparison") as renderer:
            view = synthesis_evidence_view(payload)
            renderer.assert_not_called()
        self.assertEqual(view["source_disagreements"][0]["accounts"],
                         [{"evidence": "E7", "mark": "mp"}, {"evidence": "E9", "mark": "p"}])
        self.assertNotIn("source_disagreements", payload)
        view = synthesis_evidence_view({"Retrieved evidence": [{"answer": source}]})
        hint = view["Retrieved evidence"][0]["meaning_constraints"][0]
        self.assertEqual(hint["logic"], "OR")
        for name in ("source_excerpt", "condition_fragment_1", "condition_fragment_2"):
            self.assertIn(hint[name], source)

    def test_source_operator_hints_preserve_literal_text_and_do_not_copy_question_premises(self):
        source = "어깨를 올리기보다는 편하게 내리는 것이 좋다. 긴장을 표현한 것으로 볼 수 있다. 쉬거나 구절을 짧게 나눌 수 있다."
        payload = {"Retrieved evidence": [{"question": "무조건 어깨를 올려야 하나요?", "answer": source, "citation_label": "E1"}]}
        result = synthesis_evidence_view(payload)
        record = result["Retrieved evidence"][0]
        self.assertNotIn("question", record)
        self.assertEqual(record["answer"], source)
        hints = record["meaning_constraints"]
        self.assertEqual(hints[0]["preferred_clause"], "편하게 내리는 것이 좋다")
        self.assertTrue(all(h["source_excerpt"] in source for h in hints))
        self.assertEqual(hints[1]["relation"], "interpretation_not_established_intention")
        self.assertEqual(hints[2]["relation"], "alternatives_not_joint_requirements")
        self.assertIn("question", payload["Retrieved evidence"][0])

    def prepare(self, language="ko"):
        self.source = "반주가 없는 부분에서는 피아노에 의존하기 어려우므로 자신의 음정을 주의 깊게 듣는 것이 좋다."
        self.results = [evidence(self.source)]
        self.messages = answer.build_messages("die-forelle", "", "반주가 없는 부분은 어떻게 부르나요?",
                                              self.results, language=language)
        self.payload = json.loads(self.messages[1]["content"])
        self.settings = load_settings()["llm"]

    def run_model(self, text, verdicts=(True,)):
        model = Mock()
        calls = [completion('{"task":"music_question"}')]
        for allow in verdicts:
            calls.extend([completion(json.dumps({"sentences": [{"kind": "evidence", "evidence": ["E1"], "text": text}]})),
                          completion(json.dumps({"allow": allow}))])
        model.create_chat_completion.side_effect = calls
        with patch.object(llm, "load_llama", return_value=model), \
             patch.object(llm, "translate_source_parts") as translate, \
             patch.object(llm, "opening_dynamics_comparison") as canned:
            result = llm.generate("fake", self.messages, self.settings, security_context=self.payload)
            translate.assert_not_called()
            canned.assert_not_called()
        return result, model

    def test_default_returns_the_reviewed_model_answer_not_source_replacement(self):
        self.prepare()
        authored = "반주가 없는 구간에서는 자신의 음정을 주의 깊게 들어보는 것이 좋습니다. 피아노에 의존하기 어렵기 때문입니다."
        generated, model = self.run_model(authored)
        self.assertFalse(any(p.get("source_quotation") or p.get("source_translation") for p in generated.parts))
        visible = answer.finalize_grounded_generated_answer(generated, self.results, self.messages, [])
        self.assertEqual(visible, authored)
        self.assertNotEqual(visible, self.source)
        self.assertNotIn("참고 자료의 원문", visible)
        self.assertEqual(model.create_chat_completion.call_count, 3)
        checked = json.loads(model.create_chat_completion.call_args.kwargs["messages"][1]["content"])
        self.assertEqual(checked["composition"][0]["text"], authored)

    def test_empty_model_text_cannot_silently_become_a_source_paragraph(self):
        self.prepare()
        with self.assertRaises(SecurityViolation):
            self.run_model("")

    def test_denied_generated_claims_do_not_fall_back_to_quotations(self):
        self.prepare()
        with self.assertRaises(SecurityViolation) as error:
            self.run_model("모든 구간에서 피아노를 무시하고 불러야 합니다.", verdicts=(False, False))
        self.assertEqual(error.exception.reason, "security_output_rejected")

    def test_english_answer_is_generated_and_reviewed_without_catalog_substitution(self):
        self.prepare("en")
        authored = "Listen carefully to your own intonation in unaccompanied passages, where you cannot rely on the piano."
        generated, _ = self.run_model(authored)
        visible = answer.finalize_grounded_generated_answer(generated, self.results, self.messages, [], language="en")
        self.assertEqual(visible, authored)
        self.assertNotIn("Translated reference", visible)


if __name__ == "__main__":
    unittest.main()
