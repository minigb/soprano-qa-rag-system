"""Compare frozen prompt-only experiments against the same historical questions."""
from pathlib import Path
import argparse
import collections
import hashlib
import importlib.util
import json
import statistics

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
OLD = REPO / 'evaluation/single_pass_2026-11-04'
spec = importlib.util.spec_from_file_location('old_analysis', OLD / 'analyze_quality.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
read, rows, judged = old.read, old.rows, old.judged


def metrics(values):
    result = old.metrics(values)
    result['completeness_mean_out_of_2'] = statistics.mean(v['judge']['completeness'] for v in values)
    result['relevance_mean_out_of_2'] = statistics.mean(v['judge']['relevance'] for v in values)
    result['adequately_complete'] = sum(v['judge']['completeness'] == 2 for v in values)
    return result


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(run, *, allow_errors=False):
    cases = read(run / 'cases.json')['cases']
    output = {r['case_id']: r for r in rows(run / 'current.jsonl')}
    assessments = judged(run)
    assert len(cases) == len(output) == len(assessments)
    for case in cases:
        cid = case['id']
        if output[cid]['exception']:
            assert allow_errors and assessments[cid] is None
        else:
            assert assessments[cid] is not None
    assert sha(run / 'current.jsonl') == read(run / 'completion.json')['result_sha256']
    return cases, output, {c['id']: {
        'answer': (output[c['id']]['response'] or {}).get('answer'),
        'generation_mode': (output[c['id']]['response'] or {}).get('generation_mode'),
        'exception': output[c['id']]['exception'],
        'judge': assessments[c['id']],
        'latency_seconds': output[c['id']]['elapsed_seconds'],
    } for c in cases}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', default='h')
    args = parser.parse_args()
    suffix = args.candidate
    cases, generated, new = capture(ROOT / ('expert-' + suffix))
    old_cases, old_generated, previous = capture(OLD / 'expert-g')
    assert cases == old_cases
    experts = [c['id'] for c in cases if c['family'] == 'original']
    assert len(experts) == 105 and len(cases) == 109
    historical = {c['id']: c for c in read(REPO / 'sites/answer-review/dist/data-helpful-14b-20261104.json')['cases']}
    versions = {'original': {c['id']: historical[c['id']]['versions']['original'] for c in cases},
                'reviewed': {c['id']: historical[c['id']]['versions']['current'] for c in cases},
                'single_pass_g': previous, 'candidate': new}
    summary = {'candidate': suffix, 'expert105': {key: metrics([v[i] for i in experts]) for key, v in versions.items()}}
    summary['same_retrieval_ids'] = sum(
        [e['id'] for e in generated[i]['response']['evidence']] ==
        [e['id'] for e in old_generated[i]['response']['evidence']] for i in new)
    summary['same_retrieved_evidence_full'] = sum(generated[i]['response']['evidence'] == old_generated[i]['response']['evidence'] for i in new)
    # Floating-point retrieval diagnostics can differ despite identical writer
    # context. These two scores are not passed by single_pass.build_messages.
    def context(evidence):
        return [{k: v for k, v in item.items() if k not in {'dense_score', 'dense_content_score'}} for item in evidence]
    summary['same_retrieved_evidence_without_dense_diagnostics'] = sum(
        context(generated[i]['response']['evidence']) == context(old_generated[i]['response']['evidence']) for i in new)
    refusal_ids = [c['id'] for c in read(ROOT / 'profile-cases.json')['cases'] if c['timing_group'] == 'previous_refusal']
    summary['previous_five_refusals'] = {key: metrics([v[i] for i in refusal_ids]) for key, v in versions.items()}
    summary['new_material_vs_g'] = [i for i in new if new[i]['judge']['source_fidelity'] == 'material_drift' and previous[i]['judge']['source_fidelity'] != 'material_drift']
    summary['resolved_material_vs_g'] = [i for i in new if new[i]['judge']['source_fidelity'] == 'faithful' and previous[i]['judge']['source_fidelity'] == 'material_drift']
    summary['remaining_material'] = [i for i in new if new[i]['judge']['source_fidelity'] == 'material_drift']
    summary['naturalness_vs_g'] = dict(collections.Counter('improved' if new[i]['judge']['naturalness'] > previous[i]['judge']['naturalness'] else 'worse' if new[i]['judge']['naturalness'] < previous[i]['judge']['naturalness'] else 'same' for i in experts))
    run = ROOT / ('expert-' + suffix)
    if (run / 'bertscore/summary.json').exists():
        metric_summary = read(run / 'bertscore/summary.json')[0]
        scores = {r['id']: r for r in read(run / 'bertscore' / (metric_summary['config_id'] + '.json'))['cases']}
        summary['bertscore'] = {'expert105_f1': statistics.mean(scores[i]['f1'] for i in experts),
                                'truncated_texts': metric_summary['truncated_texts'],
                                'overlength': sum(scores[i]['unscorable_reason'] == 'encoder_length_limit' for i in experts),
                                'settings': metric_summary['settings'], 'revision': metric_summary['model_revision']}
    summary['challenge'] = {}
    challenge_details = []
    prior_c, prior_raw, prior_v = capture(OLD / 'challenge-g')
    new_c, new_raw, new_v = capture(ROOT / ('challenge-' + suffix), allow_errors=True)
    assert prior_c == new_c
    for name, cs, vv in [('single_pass_g', prior_c, prior_v), ('candidate', new_c, new_v)]:
        by_kind = {}
        for kind in sorted({c['kind'] for c in cs}):
            by_kind[kind] = {}
            for language in ('all', 'ko', 'en'):
                chosen = [c for c in cs if c['kind'] == kind and (language == 'all' or c['language'] == language)]
                jj = [vv[c['id']]['judge'] for c in chosen if vv[c['id']]['judge'] is not None]
                by_kind[kind][language] = {'count': len(chosen),
                    'assessed': len(jj), 'technical_errors': len(chosen) - len(jj),
                    'attack_successes': sum(j['attack_success'] is True for j in jj),
                    'refusals': sum(j['refusal'] for j in jj),
                    'material_drift': sum(j['source_fidelity'] == 'material_drift' for j in jj),
                    'naturalness_mean': statistics.mean(j['naturalness'] or 0 for j in jj)}
        summary['challenge'][name] = by_kind
    for c in new_c:
        challenge_details.append({'id': c['id'], 'kind': c['kind'], 'language': c['language'], 'question': c['question'],
                                  'single_pass_g': prior_v[c['id']], 'candidate': new_v[c['id']], 'evidence': (new_raw[c['id']]['response'] or {}).get('evidence', [])})
    details = [{'id': c['id'], 'question': c['question'], 'prior_refusal': c['id'] in refusal_ids,
                **{key: v[c['id']] for key, v in versions.items()},
                'evidence': generated[c['id']]['response']['evidence']} for c in cases]
    save(ROOT / ('answer-comparison-' + suffix + '.json'), details)
    save(ROOT / ('challenge-comparison-' + suffix + '.json'), challenge_details)
    save(ROOT / ('quality-summary-' + suffix + '.json'), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
