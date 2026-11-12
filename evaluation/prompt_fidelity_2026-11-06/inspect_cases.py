"""Print saved answers and actual retrieved texts for diagnostic human review."""
from pathlib import Path
import argparse
import json

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('run')
parser.add_argument('ids', nargs='*')
args = parser.parse_args()
run = ROOT / args.run
cases = {c['id']: c for c in json.loads((run / 'cases.json').read_text())['cases']}
assessments = {}
for batch in sorted((run / 'judge-batches').glob('*')):
    if not (batch / 'assessments.jsonl').exists():
        continue
    mapping = {m['review_id']: m['case_id'] for m in json.loads((batch / 'mapping.json').read_text())}
    for row in map(json.loads, (batch / 'assessments.jsonl').read_text().splitlines()):
        assessments[mapping[row['id']]] = row['assessment']
for row in map(json.loads, (run / 'current.jsonl').read_text().splitlines()):
    cid = row['case_id']
    assessment = assessments.get(cid, {})
    if args.ids and cid not in args.ids:
        continue
    if not args.ids and not (assessment.get('source_fidelity') == 'material_drift' or assessment.get('attack_success') or assessment.get('refusal')):
        continue
    response = row['response']
    print(json.dumps({'id': cid, 'question': cases[cid]['question'], 'answer': response['answer'],
                      'evidence': [{'id': e['id'], 'text': e['text']} for e in response['evidence']],
                      'assessment': assessment}, ensure_ascii=False))
