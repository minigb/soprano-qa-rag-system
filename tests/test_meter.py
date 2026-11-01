"""Check meter arithmetic, limits and the retained security boundaries."""
from fractions import Fraction
import json
import unittest
from unittest.mock import Mock, patch

from soprano_qa import llm
from soprano_qa.meter import meter_facts, basic_meter_explanation
from soprano_qa.security import SecurityViolation
from tests.test_security_integration import completion


class MeterFactsTests(unittest.TestCase):
    def test_note_lengths_reconstruct_every_supported_measure(self):
        for n in (2, 3, 4, 6, 9, 12):
            for d in (2, 4, 8, 16, 32):
                with self.subTest(n=n, d=d):
                    f = meter_facts(n, d)
                    beat = Fraction(1, f['beat_note_denominator']) * (Fraction(3, 2) if f['compound'] else 1)
                    self.assertEqual(beat * f['beats'], Fraction(n, d))
                    self.assertEqual(Fraction(f['divisions'], f['division_note_denominator']), beat)

    def test_compound_and_simple_units_and_false_equal_duration_premise(self):
        payload = {'question': 'Compare the beat units in 3/2 and 9/4 time. Is the duration the same?',
                   'answer_language': 'en'}
        text = basic_meter_explanation(payload)
        self.assertIn('3 half note beats, each divided into 2 quarter notes', text)
        self.assertIn('3 dotted half note beats, each divided into 3 quarter notes', text)
        self.assertIn('duration per bar is also different', text)
        text = basic_meter_explanation({'question': '12/16박자의 큰 박과 분할을 설명해 주세요.'})
        self.assertIn('점8분음표 박이 4개', text)
        self.assertIn('16분음표 3개', text)

    def test_equal_written_duration_does_not_merge_the_groupings(self):
        text = basic_meter_explanation({'question': '6/8박자와 3/4박자의 차이는 무엇인가요?'})
        self.assertIn('점4분음표 박이 2개', text)
        self.assertIn('4분음표 박이 3개', text)
        self.assertIn('한 박은 8분음표 2개', text)
        self.assertIn('합은 같지만 박의 묶음은 다릅니다', text)

    def test_ambiguous_meters_and_performance_or_local_facts_are_not_calculated(self):
        for question in ('Explain the beats in 7/8 time.', 'Explain 6/8 and 8/8 beat groupings.',
                         'Why does the composer use 9/8 time here?',
                         'How should I conduct the beats in 6/8?', '6/8박자에서 고음을 어떻게 부르나요?'):
            with self.subTest(question=question):
                self.assertIsNone(basic_meter_explanation({'question': question}))
        for extra in ({'measure_range': '2-4'}, {'Score context': 'lyrics'},
                      {'Retrieved evidence': [{'question_alias_match': True}]},
                      {'piece': 'new-song', 'question': 'Explain the beats in 6/8 in New Song.'}):
            self.assertIsNone(basic_meter_explanation({'question': 'Explain 6/8 beats.', **extra}))

    def test_calculated_answer_requires_admission_and_output_approval(self):
        payload = {'question': 'Explain the beat unit in 9/8 time.', 'answer_language': 'en',
                   'Retrieved evidence': []}
        messages = [{'role': 'system', 'content': 'Music.'},
                    {'role': 'user', 'content': json.dumps(payload)}]
        for admission, output in ((True, True), (False, True), (True, False)):
            with self.subTest(admission=admission, output=output):
                backend = Mock()
                backend.create_chat_completion.side_effect = [completion(json.dumps({'allow': allow}))
                                                              for allow in (admission, output)]
                with patch.object(llm, 'load_llama', return_value=backend):
                    if admission and output:
                        answer = llm.generate('fake', messages, {'source_preservation': 'verbatim_or_translation'},
                                              security_context=payload)
                        self.assertEqual(answer.generation_mode, 'llm_guarded_calculation')
                        self.assertEqual(answer.answer_basis, 'calculated_music_knowledge')
                        self.assertIn('3 dotted quarter note beats', answer)
                    else:
                        with self.assertRaises(SecurityViolation):
                            llm.generate('fake', messages, {'source_preservation': 'verbatim_or_translation'},
                                         security_context=payload)
                self.assertEqual(backend.create_chat_completion.call_count, 2 if admission else 1)


if __name__ == '__main__':
    unittest.main()
