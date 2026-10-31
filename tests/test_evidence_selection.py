"""Relevance filtering cannot create facts, citations or source authority."""
import unittest
from unittest.mock import Mock

from soprano_qa.security import SecurityViolation, select_question_evidence


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
        selected = self.select('{"mode":"evidence","labels":["E3"]}')
        self.assertEqual(selected['Retrieved evidence'], [self.payload['Retrieved evidence'][2]])
        self.assertEqual(len(self.payload['Retrieved evidence']), 3)
        selected = self.select('{"mode":"evidence","labels":["E3","E1"]}')
        self.assertEqual([r['citation_label'] for r in selected['Retrieved evidence']], ['E1', 'E3'])

    def test_no_direct_support_leaves_general_knowledge_available(self):
        self.assertEqual(self.select('{"mode":"evidence","labels":[]}')['Retrieved evidence'], [])
        self.assertEqual(self.select('{"mode":"general_definition","labels":[]}')['Retrieved evidence'], [])

    def test_definition_mode_cannot_smuggle_a_piece_passage_into_its_prompt(self):
        for raw in ('{"mode":"general_definition","labels":["E1"]}',
                    '{"mode":"other","labels":[]}', '{"labels":[]}'):
            with self.subTest(raw=raw), self.assertRaises(SecurityViolation):
                self.select(raw)

    def test_empty_primary_bundle_does_not_need_relevance_review(self):
        self.payload['Retrieved evidence'] = []
        generate = Mock()
        self.assertIs(select_question_evidence('model', {}, self.payload, generate_fn=generate), self.payload)
        generate.assert_not_called()

    def test_context_only_invented_duplicate_and_malformed_labels_fail_closed(self):
        for raw in ('{"mode":"evidence","labels":["E2"]}', '{"mode":"evidence","labels":["E4"]}', '{"mode":"evidence","labels":["E1","E1"]}',
                    '{"mode":"evidence","labels":[null]}', '{"mode":"evidence","labels":"E1"}', '{"mode":"evidence","labels":[],"allow":true}',
                    '{"mode":"evidence","labels":[],"labels":["E1"]}', '[]', 'null', 'Relevant: E1'):
            with self.subTest(raw=raw), self.assertRaises(SecurityViolation):
                self.select(raw)

    def test_backend_failure_never_becomes_an_approved_selection(self):
        with self.assertRaises(SecurityViolation) as raised:
            select_question_evidence('model', {}, self.payload,
                                     generate_fn=Mock(side_effect=RuntimeError('private input')))
        self.assertEqual(raised.exception.reason, 'security_guard_unavailable')

    def test_even_one_passage_needs_question_intent_and_relevance_review(self):
        self.payload['Retrieved evidence'] = self.payload['Retrieved evidence'][:1]
        generate = Mock(return_value='{"mode":"general_definition","labels":[]}')
        selected = select_question_evidence('model', {}, self.payload, generate_fn=generate)
        self.assertEqual(selected['Retrieved evidence'], [])
        generate.assert_called_once()


if __name__ == '__main__':
    unittest.main()
