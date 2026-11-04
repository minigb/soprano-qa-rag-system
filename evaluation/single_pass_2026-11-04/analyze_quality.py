"""Compare the unchanged expert/regression cohort and bilingual challenge sets."""
from pathlib import Path
import collections
import json
import statistics

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PREVIOUS = REPO / 'evaluation/helpful_responses_2026-11-04'


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def judged(run):
    output = {}
    for batch in sorted((run / 'judge-batches').iterdir()):
        assessments = {r['id']: r['assessment'] for r in rows(batch / 'assessments.jsonl')}
        for m in read(batch / 'mapping.json'):
            output[m['case_id']] = assessments.get(m['review_id'])
    return output


def metrics(values):
    return {'count': len(values), 'refusals': sum(v['judge']['refusal'] for v in values),
            'source_fidelity': dict(collections.Counter(v['judge']['source_fidelity'] for v in values)),
            'factual_error_flags': sum(v['judge']['factual_correctness'] is not None and v['judge']['factual_correctness'] <= 2 and not v['judge']['refusal'] for v in values),
            'naturalness_effective_mean': sum((v['judge']['naturalness'] or 0) if not v['judge']['refusal'] else 0 for v in values) / len(values),
            'latency_mean_seconds': statistics.mean(v['latency_seconds'] for v in values)}


def main():
    run = ROOT / 'expert-g'
    cases = read(run / 'cases.json')['cases']
    experts = sorted(c['id'] for c in cases if c['family'] == 'original')
    assert len(experts) == 105 and len(cases) == 109
    data = {r['case_id']: r for r in rows(run / 'current.jsonl')}
    judge = judged(run)
    assert len(data) == len(judge) == 109 and all(judge.values())
    old = {c['id']: c for c in read(REPO / 'sites/answer-review/dist/data-helpful-14b-20261104.json')['cases']}
    versions = {key: {c['id']: old[c['id']]['versions'][key] for c in cases} for key in ('original', 'current')}
    new = {c['id']: {'answer': data[c['id']]['response']['answer'], 'judge': judge[c['id']],
                     'latency_seconds': data[c['id']]['elapsed_seconds'],
                     'generation_mode': data[c['id']]['response']['generation_mode']} for c in cases}
    summary = {'scope': 'Existing expert 105 + four prior-refusal paraphrases. Development regression, not a fresh holdout; does not claim a new 460-case evaluation.',
               'original_expert105': metrics([versions['original'][c] for c in experts]),
               'previous_expert105': metrics([versions['current'][c] for c in experts]),
               'single_pass_expert105': metrics([new[c] for c in experts])}
    metric_summary = read(run / 'bertscore/summary.json')[0]
    metric = {r['id']: r for r in read(run / 'bertscore' / (metric_summary['config_id'] + '.json'))['cases']}
    summary['bertscore'] = {'expert105_effective_f1': statistics.mean(metric[c]['f1'] for c in experts),
                           'previous_expert105_effective_f1': statistics.mean(versions['current'][c]['bertscore']['f1'] or 0 for c in experts),
                           'truncated_texts': metric_summary['truncated_texts'],
                           'unscorable_overlength_cases': sum(metric[c]['unscorable_reason'] == 'encoder_length_limit' for c in experts),
                           'settings': metric_summary['settings'], 'model_revision': metric_summary['model_revision']}
    refused = sorted(c['id'] for c in read(ROOT / 'profile-cases.json')['cases'] if c['timing_group'] == 'previous_refusal')
    summary['previous_five_refusals'] = metrics([new[c] for c in refused])
    summary['new_material_drift_ids'] = [c for c in new if judge[c]['source_fidelity'] == 'material_drift' and versions['current'][c]['judge']['source_fidelity'] != 'material_drift']
    summary['resolved_material_drift_ids'] = [c for c in new if judge[c]['source_fidelity'] == 'faithful' and versions['current'][c]['judge']['source_fidelity'] == 'material_drift']
    summary['lower_naturalness_ids'] = [c for c in new if judge[c]['naturalness'] < versions['current'][c]['judge']['naturalness']]
    summary['challenge'] = {}
    for label, directory in (('previous', PREVIOUS / 'security-hardened'), ('single_pass', ROOT / 'challenge-g')):
        challenge_cases = read(directory / 'cases.json')['cases']; challenge_judge = judged(directory)
        assert len(challenge_cases) == len(challenge_judge) == 74
        by_kind = {}
        for kind in sorted({c['kind'] for c in challenge_cases}):
            chosen = [c for c in challenge_cases if c['kind'] == kind]
            by_kind[kind] = {'count': len(chosen), 'attack_successes': sum(challenge_judge[c['id']]['attack_success'] is True for c in chosen),
                             'refusals': sum(challenge_judge[c['id']]['refusal'] for c in chosen),
                             'material_drift': sum(challenge_judge[c['id']]['source_fidelity'] == 'material_drift' for c in chosen),
                             'factual_error_flags': sum((challenge_judge[c['id']]['factual_correctness'] or 0) <= 2 and not challenge_judge[c['id']]['refusal'] for c in chosen)}
        summary['challenge'][label] = by_kind
    details = [{'id': c['id'], 'question': c['question'], 'prior_refusal': c['id'] in refused,
                'original': versions['original'][c['id']], 'previous': versions['current'][c['id']],
                'single_pass': new[c['id']], 'evidence': data[c['id']]['response']['evidence']} for c in cases]
    (ROOT / 'answer-comparison.json').write_text(json.dumps(details, ensure_ascii=False, indent=2) + '\n')
    (ROOT / 'quality-summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
