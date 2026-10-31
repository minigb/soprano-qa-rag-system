"""General explanation must not acquire a source's citation or scope authority."""
import json
import unittest
from types import SimpleNamespace

from soprano_qa.answer import (
    GeneratedAnswerRejected, finalize_grounded_generated_answer,
    _copies_scoped_source_sentence,
)
from soprano_qa.security import SecurityViolation
from soprano_qa.structured_answer import RenderedAnswer, render_structured_answer


class BackgroundProvenanceTests(unittest.TestCase):
    payload = {"answer_language": "en", "Retrieved evidence": [
        {"citation_label": "E1", "answer": "Practice without consonants."},
    ]}

    def render(self, kind, labels, text='A half note is twice a quarter note.'):
        return render_structured_answer(json.dumps({'sentences': [
            {'kind': kind, 'evidence': labels, 'text': text},
        ]}), self.payload)

    def test_background_is_identified_and_never_receives_a_citation(self):
        result = self.render('background', [])
        self.assertIsInstance(result, RenderedAnswer)
        self.assertEqual(result, 'General explanation: A half note is twice a quarter note.')
        self.assertEqual(result.answer_basis, 'general_music_knowledge')
        self.assertEqual(result.grounded_text, '')

    def test_background_cannot_borrow_source_labels(self):
        with self.assertRaises(SecurityViolation):
            self.render('background', ['E1'])

    def test_evidence_kind_cannot_drop_required_citation(self):
        with self.assertRaises(SecurityViolation):
            self.render('evidence', [])

    def test_score_lyrics_requires_actual_score_context(self):
        with self.assertRaises(SecurityViolation):
            self.render('score_lyrics', [])

    def test_unknown_kind_cannot_bypass_validation(self):
        with self.assertRaises(SecurityViolation):
            self.render('trusted_background', [])

    def test_background_inline_citations_are_rejected(self):
        with self.assertRaises(SecurityViolation):
            self.render('background', [], 'Use this practice. [E1]')

    def test_general_answer_can_be_finalized_without_a_ku(self):
        result = self.render('background', [])
        self.assertEqual(finalize_grounded_generated_answer(
            result, [], [{'role': 'system', 'content': 'Music explanation.'}], [], language='en',
        ), str(result))

    def test_mixed_answer_keeps_source_and_background_metadata(self):
        result = render_structured_answer(json.dumps({'sentences': [
            {'kind': 'evidence', 'evidence': ['E1'], 'text': 'Practice without consonants.'},
            {'kind': 'background', 'evidence': [], 'text': 'Legato means smoothly connected.'},
        ]}), self.payload)
        self.assertEqual(result.answer_basis, 'retrieved_evidence_with_background')
        self.assertEqual(result.grounded_text, 'Practice without consonants. [E1]')
        self.assertNotIn('[E1]', result.background_text)

    def test_frequency_screen_preserves_complete_scoped_source_sentences(self):
        sentence = '꾸밈음이 자주 나타나는 프레이즈에서는 가벼운 소리를 유지한다.'
        results = [SimpleNamespace(record={'answer': sentence})]
        self.assertTrue(_copies_scoped_source_sentence(
            '꾸밈음이 자주 나타나는 프레이즈에서는 가벼운 소리를 유지합니다. [E1]', results, {0}))
        self.assertFalse(_copies_scoped_source_sentence(
            '이 곡에는 꾸밈음이 자주 나타납니다. [E1]', results, {0}))
        self.assertFalse(_copies_scoped_source_sentence(
            '꾸밈음이 자주 나타나는 프레이즈에서는 가벼운 소리를 유지합니다. [E2]', results, set()))
        results[0].record['answer'] = '곡 전체에서 항상 가볍게 부른다.'
        self.assertFalse(_copies_scoped_source_sentence(
            '곡 전체에서 항상 가볍게 부릅니다. [E1]', results, {0}))


if __name__ == '__main__':
    unittest.main()
