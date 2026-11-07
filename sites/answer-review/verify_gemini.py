"""Check displayed Gemini pairs against saved outputs, without any model calls."""
from pathlib import Path
import hashlib
import json
import math
import statistics

SITE = Path(__file__).resolve().parent
REPO = SITE.parents[1]
RUN = REPO / 'evaluation/gemini_flash_recovery_2026-11-07'
PRIMARY = REPO / 'evaluation/gemini_flash_writer_2026-11-07'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    catalog = read(SITE / 'dist/evaluations.json')
    assert catalog['default'] == 'gemini-20261107'
    entry = next(e for e in catalog['evaluations'] if e['id'] == catalog['default'])
    data = read(SITE / 'dist' / entry['data'])
    original = {c['id']: c for c in read(SITE / 'dist/data.json')['cases']}
    baseline = {c['id']: c for c in read(REPO / 'evaluation/expanded_460_2026-11-02/cases.json')['cases']}
    cases = {c['id']: c for c in read(PRIMARY / 'expert/cases.json')['cases']}
    rows = {r['case_id']: r for r in jsonl(RUN / 'combined/expert.jsonl')}
    first = {r['case_id']: r for r in jsonl(PRIMARY / 'expert/current.jsonl')}
    judges = read(RUN / 'combined/expert-judgments.json')
    scores = {r['id']: r for r in read(RUN / 'combined/expert-bertscore.json')['cases']}
    assert len(data['cases']) == entry['caseCount'] == 109
    assert {c['id'] for c in data['cases']} == set(rows) == set(cases) == set(judges) == set(scores)
    assert len({e['storageKey'] for e in catalog['evaluations']}) == len(catalog['evaluations'])
    assert entry['latencyComparable'] is False
    retries = []
    for c in data['cases']:
        cid = c['id']; old = original[cid]; v = c['versions']['current']; r = rows[cid]
        response = r['response']; metric = scores[cid]
        assert {k: value for k, value in c.items() if k not in ['versions', 'evaluation_references']} == {k: value for k, value in old.items() if k != 'versions'}
        assert c['versions']['original'] == old['versions']['original']
        assert all(c[k] == cases[cid][k] for k in ['question', 'piece_id', 'measure_range', 'family'])
        assert v['answer'] == response['answer'] and v['answer'].strip()
        assert v['latency_seconds'] == r['elapsed_seconds']
        assert v['status'] == r['status'] == 'response'
        assert v['generation_mode'] == response['generation_mode']
        assert response['generation_metadata']['model_version'] == 'gemini-3.8-flash'
        assert v['evidence'] == [{k: e.get(k) for k in ['id', 'text', 'measure_ranges', 'evidence_type', 'generation_role']} for e in response['evidence']]
        assert v['judge']['available'] is True
        assert {k: value for k, value in v['judge'].items() if k != 'available'} == {k: judges[cid][k] for k in old['versions']['original']['judge']}
        assert v['bertscore'] == {'scored': metric['scored'], 'f1': metric['f1'] if metric['scored'] else None, 'reason': metric['unscorable_reason'], 'candidate_tokens': metric['candidate_tokens']}
        assert v['technical_retry'] == (r.get('attempt_number') == 2) == v['first_attempt_error'] == (first[cid]['status'] == 'exception')
        if v['technical_retry']: retries.append(cid)
        ref = c['evaluation_references']
        assert ref['original']['sources'] == old['references']
        assert ref['original']['bertscore_text'] == baseline[cid]['reference_answers'][0]
        assert ref['current']['bertscore_text'] == cases[cid]['reference_answers'][0]
        assert ref['current']['sources'] == [{k: item.get(k) for k in ['knowledge_unit_id', 'text', 'measure_ranges', 'reference_role']} for item in cases[cid]['reference_authority']['items']]
    assert len(retries) == 6
    expert = [c for c in data['cases'] if c['family'] == 'original']
    assert len(expert) == 105
    for version in ['original', 'current']:
        for scope, selected in [('all', data['cases']), ('original', expert)]:
            values = [c['versions'][version] for c in selected]
            summary = data['summary'][version][scope]
            assert summary['cases'] == len(values)
            assert summary['material_drifts'] == sum(v['judge']['source_fidelity'] == 'material_drift' for v in values)
            assert math.isclose(summary['naturalness_mean_effective_out_of_4'], sum(v['judge']['naturalness'] if not v['judge']['refusal'] else 0 for v in values) / len(values))
            assert math.isclose(summary['bertscore_effective_f1'], sum(v['bertscore']['f1'] if v['bertscore']['scored'] else 0 for v in values) / len(values))
            assert summary['latency_all']['median_seconds'] == statistics.median(v['latency_seconds'] for v in values)
    assert data['summary']['original']['original']['material_drifts'] == 10
    assert data['summary']['current']['original']['material_drifts'] == 0
    assert math.isclose(data['summary']['current']['original']['bertscore_effective_f1'], 0.9509922112737383)
    assert data['naturalness_comparison']['losses'] == sum(c['versions']['current']['judge']['naturalness'] < c['versions']['original']['judge']['naturalness'] for c in data['cases'])
    for rel, digest in data['source_hashes'].items():
        assert sha(REPO / rel) == digest
    before = read(REPO / 'evaluation/gemini_demo_2026-11-07/before.json')
    for name, digest in before['worktree_assets'].items():
        if name.startswith('data'):
            assert sha(SITE / 'dist' / name) == digest, name
    for historical in catalog['evaluations'][1:]:
        assert len(read(SITE / 'dist' / historical['data'])['cases']) == 460
    serialized = (SITE / 'dist' / entry['data']).read_text()
    assert not any(s in serialized for s in ['/home/minhee/', 'private_trace_attempt_id', 'batch_usage', 'model_path', 'access_token', 'refresh_token', 'client_secret'])
    print(json.dumps({'verified': True, 'cases': 109, 'full_answers_verified': 218,
                      'original': 'unchanged pre-security outputs, not candidate J',
                      'current': 'exact saved Gemini outputs, evidence, judgments and scores',
                      'retry_cases': retries, 'historical_datasets': 'unchanged, 460 cases each',
                      'evaluation_references': 'original and current verified separately',
                      'latency': 'separate scopes labeled, no direct speed claim',
                      'api_calls': 0}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
