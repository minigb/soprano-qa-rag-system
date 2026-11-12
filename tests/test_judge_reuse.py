"""A cached judgment cannot cover a changed meaning, context or rubric."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from evaluation.run_reused_claude_judge import partition, completed_review, joint, judge


class ExactReviewReuseTests(unittest.TestCase):
    def setUp(self):
        self.packet = {'id': 'opaque', 'question': 'What changes?', 'answer': 'Do not shorten the note.',
                       'evidence': [{'text': 'Do not shorten the note.'}],
                       'reference_answers': ['Retain the note length.'], 'piece': 'work-a',
                       'measure_range': [3, 4], 'score_context': None, 'task': 'normal'}
        self.row = {'id': 'opaque', 'assessment': {'source_fidelity': 'material_drift'},
                    'contract_id': 'original-contract', 'packet_hash': 'original-hash'}

    def test_identical_packet_reuses_even_an_unfavorable_assessment_unchanged(self):
        reused, pending = partition([deepcopy(self.packet)], [self.packet], [self.row])
        self.assertEqual(pending, [])
        self.assertIs(reused[0], self.row)

    def test_each_changed_semantic_input_requires_a_new_review(self):
        for key, value in {'question': 'Why?', 'answer': 'Shorten the note.',
                           'evidence': [{'text': 'Shorten it.'}], 'reference_answers': ['Shorten it.'],
                           'piece': 'work-b', 'measure_range': [4, 5],
                           'score_context': {'lyrics': 'different'}, 'task': 'attack'}.items():
            with self.subTest(key=key):
                changed = {**self.packet, key: value}
                reused, pending = partition([changed], [self.packet], [self.row])
                self.assertEqual(reused, [])
                self.assertEqual(pending, [changed])

    def fixture(self, directory):
        shard = directory/'shard-00'
        shard.mkdir()
        for d in (directory, shard):
            joint.frozen_write(d/'packets.json', [self.packet])
        manifest = {'model': 'test-model', 'runner_sha256': joint.digest(Path(judge.__file__)),
                    'rubric_sha256': hashlib.sha256(judge.SYSTEM.encode()).hexdigest(),
                    'schema_sha256': judge.sha(judge.schema([])),
                    'acceptance': 'first valid batch; never retry based on a score',
                    'packet_file_sha256': joint.digest(shard/'packets.json')}
        assessment = {'id': 'opaque', 'source_fidelity': 'faithful', 'factual_correctness': 4,
                      'attribution': 2, 'relevance': 2, 'completeness': 2, 'language': 1,
                      'refusal': False, 'attack_success': None, 'answer_excerpt': '',
                      'source_excerpt': '', 'rationale': 'Retains the prohibition.'}
        row = {'id': 'opaque', 'status': 'ok', 'model': 'test-model', 'assessment': assessment,
               'packet_hash': judge.sha(self.packet), 'contract_id': judge.sha(manifest)}
        for d in (directory, shard):
            (d/'assessments.jsonl').write_text(json.dumps(row)+'\n')
        joint.frozen_write(shard/'assessments.jsonl.manifest.json', manifest)
        contract = {'model': 'test-model', 'packets_sha256': joint.digest(directory/'packets.json'),
                    'judge_runner_sha256': joint.digest(Path(judge.__file__)),
                    'runner_sha256': joint.digest(Path(joint.__file__))}
        joint.frozen_write(directory/'joint-contract.json', contract)
        joint.frozen_write(directory/'aggregation.json', {
            'contract': contract, 'assessment_sha256': joint.digest(directory/'assessments.jsonl'),
            'shards': {'shard-00': {'assessment_sha256': joint.digest(shard/'assessments.jsonl'),
                                  'manifest_sha256': joint.digest(shard/'assessments.jsonl.manifest.json')}}})

    def test_completed_review_checks_full_provenance_and_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.fixture(directory)
            packets, assessed = completed_review(directory, 'test-model')
            self.assertEqual(packets, [self.packet])
            self.assertEqual(len(assessed), 1)
            with self.assertRaises(ValueError):
                completed_review(directory, 'different-model')

    def test_changed_results_or_rubric_cannot_be_reused(self):
        for target in ('packets.json', 'assessments.jsonl', 'shard-00/assessments.jsonl.manifest.json'):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                self.fixture(directory)
                path = directory/target
                path.write_text(path.read_text()+' ')
                with self.assertRaises(ValueError):
                    completed_review(directory, 'test-model')

    def test_partial_review_is_never_treated_as_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.fixture(directory)
            (directory/'aggregation.json').unlink()
            with self.assertRaises(FileNotFoundError):
                completed_review(directory, 'test-model')


if __name__ == '__main__':
    unittest.main()
