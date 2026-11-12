"""Read completed rows and judgments without modifying an evaluation."""
from pathlib import Path
import collections
import json

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'full-final'


def rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines(keepends=True) if line.endswith('\n')]


def main():
    current = rows(RUN / 'current.jsonl')
    previous = {r['case_id']: r for r in rows(ROOT.parent / 'latest_demo_2026-11-03/current.jsonl')}
    failures = {c for c, r in previous.items() if r['response']['generation_mode'] == 'unavailable'}
    historical = {c['id']: c['versions']['current']['judge'] for c in json.loads(
        (ROOT.parents[1] / 'sites/answer-review/dist/data-latest-14b-20261103.json').read_text())['cases']}
    reviewed = {}
    fresh = reused = 0
    for batch in sorted((RUN / 'judge-batches').glob('*')):
        if not (batch / 'assessments.jsonl').exists():
            continue
        assessed = {r['id']: r['assessment'] for r in rows(batch / 'assessments.jsonl')}
        for mapping in json.loads((batch / 'mapping.json').read_text()):
            reviewed[mapping['case_id']] = assessed.get(mapping['review_id'])
        if (batch / 'reuse-aggregation.json').exists():
            counts = json.loads((batch / 'reuse-aggregation.json').read_text())
            fresh += counts['fresh']
            reused += counts['reused']
    print(json.dumps({
        'generated': len(current), 'total': 460, 'judged': len(reviewed),
        'generation_errors': [r['case_id'] for r in current if r['status'] != 'response'],
        'modes': dict(collections.Counter((r.get('response') or {}).get('generation_mode') for r in current)),
        'changed_answers': sum((r.get('response') or {}).get('answer') != previous[r['case_id']]['response']['answer'] for r in current),
        'judge_reused': reused, 'judge_fresh': fresh,
        'previous_refusals_completed': [r['case_id'] for r in current if r['case_id'] in failures],
        'new_refusals': [c for c, a in reviewed.items() if a and a['refusal']],
        'new_material_drift': [c for c, a in reviewed.items() if a and a['source_fidelity'] == 'material_drift'
                               and historical[c]['source_fidelity'] != 'material_drift'],
        'new_factual_error_flags': [c for c, a in reviewed.items() if a and a['factual_correctness'] <= 2
                                     and historical[c]['factual_correctness'] > 2],
        'metric_complete': (RUN / 'assessment-complete.json').exists(),
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
