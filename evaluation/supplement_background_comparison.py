"""Add language-by-task counts and paired descriptive latency to a complete review.

This never judges an answer or changes the frozen primary score. Errors remain
visible, and matched-response latency is separated from overall availability.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics

from summarize_background_eval import summarize


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def delivered(item):
    return item['delivered'] and item['judge'] is not None and not item['judge']['refusal']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = read(args.config)
    directory = Path(config['output'])
    cases = {c['id']: c for c in read(Path(config['cases']))['cases']}
    primary = read(directory/'review/summary.json')
    mapping = {m['key']: m for m in read(directory/'judge/mapping.json')}
    assessed = rows(directory/'judge/assessments.jsonl')
    judgments = {a['id']: a['assessment'] for a in assessed if a['status'] == 'ok'}
    if len(judgments) != len(assessed):
        raise ValueError('Incomplete or duplicate independent judgments')
    variants = {}
    for system in config['systems']:
        name = system['name']
        path = directory/(name+'.jsonl')
        expected_hashes = primary['contract']['results_sha256']
        recorded_hash = expected_hashes.get(str(path.resolve()), expected_hashes.get(str(path)))
        if digest(path) != recorded_hash:
            raise ValueError('Responses differ from the completed primary review')
        data = rows(path)
        if len(data) != len(cases) or {r['case_id'] for r in data} != set(cases):
            raise ValueError('Each system must contain the complete cohort')
        items = {}
        for row in data:
            case_id = row['case_id']
            review_id = mapping[name+':'+case_id]['review_id']
            if review_id and review_id not in judgments:
                raise ValueError('Missing review')
            response = row.get('response') or {}
            items[case_id] = {
                'case': cases[case_id], 'row': row, 'judge': judgments.get(review_id),
                'delivered': row['status'] == 'response' and response.get('generation_mode') != 'unavailable'
                             and bool(response.get('answer', '').strip()),
            }
        variants[name] = items
    by_language = {}
    for name, indexed in variants.items():
        groups = {}
        for language in ('ko', 'en'):
            for kind in ('benign', 'direct_attack', 'indirect_attack', 'out_of_scope'):
                group = [x for x in indexed.values() if x['case']['language'] == language and x['case']['kind'] == kind]
                stats = summarize(group)
                lengths = [len(x['row']['response']['answer']) for x in group if delivered(x)]
                stats['delivered_answer_median_characters'] = statistics.median(lengths) if lengths else None
                groups[language+'/'+kind] = stats
        by_language[name] = groups
    paired = {}
    candidate = variants['candidate']
    for name, baseline in variants.items():
        if name == 'candidate':
            continue
        eligible = [key for key, c in cases.items() if c['kind'] == 'benign'
                    and c.get('answerability') != 'unknown_local_fact'
                    and candidate[key]['row'].get('latency_scope') != 'fixed_evidence_generation'
                    and baseline[key]['row'].get('latency_scope') != 'fixed_evidence_generation']
        common = [key for key in eligible if delivered(candidate[key]) and delivered(baseline[key])]
        a = [candidate[key]['row']['elapsed_seconds'] for key in common]
        b = [baseline[key]['row']['elapsed_seconds'] for key in common]
        paired[name] = {
            'eligible_answerable_service_requests': len(eligible),
            'both_delivered_n': len(common),
            'candidate_only_delivered_n': sum(delivered(candidate[k]) and not delivered(baseline[k]) for k in eligible),
            'baseline_only_delivered_n': sum(delivered(baseline[k]) and not delivered(candidate[k]) for k in eligible),
            'neither_delivered_n': sum(not delivered(candidate[k]) and not delivered(baseline[k]) for k in eligible),
            'candidate_median_seconds_on_common': statistics.median(a) if a else None,
            'baseline_median_seconds_on_common': statistics.median(b) if b else None,
            'median_paired_difference_seconds': statistics.median(x-y for x,y in zip(a,b)) if a else None,
            'common_case_ids': common,
        }
    result = {
        'contract': {'config_sha256': digest(args.config), 'primary_summary_sha256': digest(directory/'review/summary.json'),
                     'script_sha256': digest(Path(__file__))},
        'notes': [
            'Attack success counts use the independent judge. Report attempted and judged denominators plus execution errors; errors are not evidence of a successful defense.',
            'Unknown local facts are excluded from the false-refusal denominator. Execution errors are separate and remain zero in effective quality.',
            'Quality is a rubric composite, not factual accuracy. Source fidelity does not establish source truth.',
            'Paired latency uses the same answerable service cases delivered by both systems. It does not hide availability differences, which are listed separately.',
            'Single measured run per system, sequential GPU use, authored/reused cases: descriptive comparison, not a statistical speedup or generalization claim.',
        ],
        'language_by_task': by_language,
        'paired_service_latency': paired,
    }
    target = directory/'review/supplement.json'
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(target)


if __name__ == '__main__':
    main()
