"""Relevance filtering cannot create facts, citations or source authority."""
import unittest
from unittest.mock import Mock

from soprano_qa.security import SecurityViolation, select_question_evidence, question_uses_general_music


class QuestionRoutingTests(unittest.TestCase):
    def test_route_uses_only_question_and_selected_range(self):
        payload = {'question': 'Is a fermata always three times as long?', 'measure_range': '(none selected)',
                   'piece': 'selected work', 'Retrieved evidence': [{'answer': 'Unrelated triplet advice.'}],
                   'Score context': 'private lyric context'}
        generate = Mock(return_value='{"mode":"general"}')
        self.assertTrue(question_uses_general_music('model', {}, payload, generate_fn=generate))
        import json
        actual = json.loads(generate.call_args.args[1][1]['content'])
        self.assertEqual(actual, {k: payload[k] for k in ('question', 'measure_range')})

    def test_evidence_route_does_not_authorize_or_rewrite_any_source(self):
        generate = Mock(return_value='{"mode":"evidence"}')
        self.assertFalse(question_uses_general_music('model', {}, {'question': 'How should I sing this phrase?'},
                                                    generate_fn=generate))

    def test_failed_or_ambiguous_route_is_not_a_general_knowledge_fallback(self):
        for raw in ('{}', '{"mode":"general","allow":true}', '{"mode":false}',
                    '{"mode":"general","mode":"evidence"}', '[]', 'null', 'general'):
            with self.subTest(raw=raw), self.assertRaises(SecurityViolation):
                question_uses_general_music('model', {}, {}, generate_fn=Mock(return_value=raw))
        with self.assertRaises(SecurityViolation) as error:
            question_uses_general_music('model', {}, {}, generate_fn=Mock(side_effect=RuntimeError('private')))
        self.assertEqual(error.exception.reason, 'security_guard_unavailable')


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.payload = {'question': 'How should I pronounce this word?',
                        'Retrieved evidence': [
                            {'citation_label': 'E1', 'answer': 'Diction advice.'},
                            {'citation_label': None, 'answer': 'Other measure context.'},
                            {'citation_label': 'E3', 'answer': 'Breathing advice.'}]}

    def select(self, raw):
        return select_question_evidence('model', {}, self.payload, generate_fn=Mock(return_value=raw))

    def test_selection_preserves_label_identity_order_and_original_bundle(self):
        selected = self.select('{"labels":["E3"]}')
        self.assertEqual(selected['Retrieved evidence'], [self.payload['Retrieved evidence'][2]])
        self.assertEqual(len(self.payload['Retrieved evidence']), 3)
        selected = self.select('{"labels":["E3","E1"]}')
        self.assertEqual([r['citation_label'] for r in selected['Retrieved evidence']], ['E1', 'E3'])

    def test_no_direct_support_leaves_general_knowledge_available(self):
        self.assertEqual(self.select('{"labels":[]}')['Retrieved evidence'], [])

    def test_context_only_invented_duplicate_and_malformed_labels_fail_closed(self):
        for raw in ('{"labels":["E2"]}', '{"labels":["E4"]}', '{"labels":["E1","E1"]}',
                    '{"labels":[null]}', '{"labels":"E1"}', '{"labels":[],"allow":true}',
                    '{"labels":[],"labels":["E1"]}', '[]', 'null', 'Relevant: E1'):
            with self.subTest(raw=raw), self.assertRaises(SecurityViolation):
                self.select(raw)

    def test_backend_failure_never_becomes_an_approved_selection(self):
        with self.assertRaises(SecurityViolation) as raised:
            select_question_evidence('model', {}, self.payload,
                                     generate_fn=Mock(side_effect=RuntimeError('private input')))
        self.assertEqual(raised.exception.reason, 'security_guard_unavailable')

    def test_single_primary_passage_does_not_need_another_call(self):
        self.payload['Retrieved evidence'] = self.payload['Retrieved evidence'][:1]
        generate = Mock()
        self.assertIs(select_question_evidence('model', {}, self.payload, generate_fn=generate), self.payload)
        generate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
