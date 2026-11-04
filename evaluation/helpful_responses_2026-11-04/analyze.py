"""Summarize complete captures, keeping quality and response coverage separate."""
from pathlib import Path
import collections
import hashlib
import json
import statistics

ROOT = Path(__file__).resolve().parent
SITE = ROOT.parents[1] / 'sites/answer-review/dist'


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def judgments(run):
    output = {}
    for batch in sorted((run / 'judge-batches').iterdir()):
        assessed = {r['id']: r['assessment'] for r in rows(batch / 'assessments.jsonl')}
        for mapping in read(batch / 'mapping.json'):
            output[mapping['case_id']] = assessed.get(mapping['review_id'])
    return output


def summary(versions, expert_ids):
    values = list(versions.values())
    return {
        'count': len(values),
        'unanswered': sum(v.get('status', 'response') != 'response'
                          or v.get('generation_mode') in {'unavailable', 'response_error', 'domain_guidance'}
                          or v['judge']['refusal'] or not v['answer'].strip() for v in values),
        'source_fidelity': dict(collections.Counter(v['judge']['source_fidelity'] for v in values)),
        'factual_error_flags': sum(v['judge']['factual_correctness'] is not None
                                  and v['judge']['factual_correctness'] <= 2
                                  and not v['judge']['refusal'] for v in values),
        'naturalness_effective_mean': sum(v['judge']['naturalness'] or 0 for v in values
                                          if not v['judge']['refusal']) / len(values),
        'expert_105_bertscore': sum(versions[c]['bertscore']['f1'] or 0 for c in expert_ids) / len(expert_ids),
        'latency_mean_seconds': statistics.mean(v['latency_seconds'] for v in values),
        'latency_median_seconds': statistics.median(v['latency_seconds'] for v in values),
        'generation_modes': dict(collections.Counter(v.get('generation_mode') for v in values)),
    }


def main():
    current = read(SITE / 'data-helpful-14b-20261104.json')
    previous = read(SITE / 'data-latest-14b-20261103.json')
    assert current['assessment_status'] == 'complete'
    now = {c['id']: c['versions']['current'] for c in current['cases']}
    old = {c['id']: c['versions']['current'] for c in previous['cases']}
    original = {c['id']: c['versions']['original'] for c in current['cases']}
    experts = {c['id'] for c in current['cases'] if c['family'] == 'original'}
    assert len(now) == len(old) == len(original) == 460 and len(experts) == 105
    changed = [c for c in now if now[c]['answer'] != old[c]['answer']]
    refused = [c for c in old if old[c]['generation_mode'] == 'unavailable']
    five = [{'id': c, 'question': next(x['question'] for x in current['cases'] if x['id'] == c),
             'original': original[c], 'previous': old[c], 'current': now[c]} for c in refused]
    new_material = [c for c in now if now[c]['judge']['source_fidelity'] == 'material_drift'
                    and old[c]['judge']['source_fidelity'] != 'material_drift']
    resolved_material = [c for c in now if old[c]['judge']['source_fidelity'] == 'material_drift'
                         and now[c]['judge']['source_fidelity'] != 'material_drift'
                         and not now[c]['judge']['refusal']]
    new_lower = [c for c in now if (now[c]['judge']['naturalness'] or 0) < (old[c]['judge']['naturalness'] or 0)
                 and not old[c]['judge']['refusal']]
    paired_answered = [c for c in now if not old[c]['judge']['refusal'] and not now[c]['judge']['refusal']
                       and old[c].get('generation_mode') not in {'unavailable', 'response_error', 'domain_guidance'}
                       and now[c].get('generation_mode') not in {'unavailable', 'response_error', 'domain_guidance'}]
    paired_naturalness = {
        'count': len(paired_answered),
        'november_3_mean': statistics.mean(old[c]['judge']['naturalness'] for c in paired_answered),
        'current_mean': statistics.mean(now[c]['judge']['naturalness'] for c in paired_answered),
        'improved': sum(now[c]['judge']['naturalness'] > old[c]['judge']['naturalness'] for c in paired_answered),
        'equal': sum(now[c]['judge']['naturalness'] == old[c]['judge']['naturalness'] for c in paired_answered),
        'declined': sum(now[c]['judge']['naturalness'] < old[c]['judge']['naturalness'] for c in paired_answered),
    }
    security_run = ROOT / 'security-hardened'
    security_cases = {c['id']: c for c in read(security_run / 'cases.json')['cases']}
    security_rows = rows(security_run / 'current.jsonl')
    security_judge = judgments(security_run)
    assert len(security_rows) == len(security_judge) == len(security_cases) == 74
    security = {}
    for kind in sorted({c['kind'] for c in security_cases.values()}):
        group = [r for r in security_rows if security_cases[r['case_id']]['kind'] == kind]
        security[kind] = {
            'count': len(group), 'exceptions': sum(r['status'] != 'response' for r in group),
            'attack_successes': sum((security_judge[r['case_id']] or {}).get('attack_success') is True for r in group),
            'refusals': sum(bool((security_judge[r['case_id']] or {}).get('refusal')) for r in group),
            'modes': dict(collections.Counter((r.get('response') or {}).get('generation_mode') for r in group)),
        }
    reused = sum(read(p)['reused'] for p in (ROOT / 'full-final/judge-batches').glob('*/reuse-aggregation.json'))
    fresh = sum(read(p)['fresh'] for p in (ROOT / 'full-final/judge-batches').glob('*/reuse-aggregation.json'))
    output = {
        'model': 'Qwen3-14B-Q6_K', 'generator_commit': read(ROOT / 'full-final/provenance.json')['source_head'],
        'original': summary(original, experts), 'november_3': summary(old, experts), 'current': summary(now, experts),
        'changed_answers_from_november_3': len(changed), 'changed_ids': changed,
        'previous_five_refusals': five, 'new_material_drift_ids': new_material,
        'resolved_material_drift_ids': resolved_material, 'lower_naturalness_ids': new_lower,
        'lower_attribution_ids': [c for c in now if not old[c]['judge']['refusal']
                                  and now[c]['judge']['attribution'] < old[c]['judge']['attribution']],
        'paired_answered_naturalness_vs_november_3': paired_naturalness,
        'security': security, 'claude': {'model': 'claude-opus-5-5', 'exact_packets_reused': reused, 'fresh_packets': fresh},
        'metric_policy': 'Unchanged xlm-roberta-large layer 18, no IDF, raw F1; full displayed answers. No tuning or clipping.',
        'security_scope': 'Frozen retrieval intervention: 30 direct attacks, 20 indirect attacks, 12 out-of-domain requests, 12 normal controls; Korean/English. Not a comprehensive security guarantee.',
        'latency_scope': 'Separate single-run captures; no claim of controlled speed comparison.',
        'files': {str(p.relative_to(ROOT)): sha(p) for p in [ROOT/'full-final/current.jsonl', ROOT/'full-final/assessment-complete.json', security_run/'current.jsonl']},
    }
    (ROOT / 'comparison.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: output[k] for k in ('original', 'november_3', 'current', 'changed_answers_from_november_3', 'new_material_drift_ids', 'security', 'claude')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
