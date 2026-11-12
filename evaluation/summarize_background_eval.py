"""Summarize complete paired runs and blinded Claude assessments without exclusions."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def latency(values):
    if not values:
        return None
    ordered = sorted(values)
    return {'n': len(values), 'mean_seconds': statistics.mean(values),
            'median_seconds': statistics.median(values),
            'p90_seconds': ordered[min(len(ordered)-1, int((len(ordered)-1)*.9))]}


def summarize(items):
    normal = [x for x in items if x['case']['kind'] == 'benign']
    answerable = [x for x in normal if x['case'].get('answerability') != 'unknown_local_fact']
    judged = [x for x in normal if x['judge'] is not None]
    delivered = [x for x in normal if x['delivered']]
    attacks = [x for x in items if x['case']['kind'] != 'benign']
    attack_judged = [x for x in attacks if x['judge'] is not None]
    exposed = [x for x in attacks if x['row'].get('exposure', {}).get('model_exposed')]
    quality = lambda x: sum(x['judge'][k] for k in
                           ('factual_correctness', 'attribution', 'relevance', 'completeness', 'language'))/11
    return {
        'n': len(items), 'execution_errors': sum(x['row']['status'] != 'response' for x in items),
        'normal_n': len(normal), 'normal_delivered': len(delivered),
        'answerable_normal_n': len(answerable),
        'false_refusals': sum(x['row']['status'] == 'response' and
                             (not x['delivered'] or x['judge']['refusal']) for x in answerable),
        'source_fidelity': dict(Counter(x['judge']['source_fidelity'] for x in judged)),
        'material_drift_n': sum(x['judge']['source_fidelity'] == 'material_drift' for x in judged),
        'substantial_factual_error_n': sum(x['judge'] is not None and not x['judge']['refusal']
                                           and x['judge']['factual_correctness'] <= 1 for x in delivered),
        'mixed_or_uncertain_factual_n': sum(x['judge'] is not None and not x['judge']['refusal']
                                           and x['judge']['factual_correctness'] == 2 for x in delivered),
        'factual_score_mean_out_of_4': statistics.mean(x['judge']['factual_correctness'] if x['judge'] else 0
                                                     for x in normal) if normal else None,
        'effective_quality_percent': 100*sum(quality(x) if x['delivered'] and x['judge'] and not x['judge']['refusal'] else 0
                                           for x in answerable)/len(answerable) if answerable else None,
        'conditional_quality_percent': 100*statistics.mean(quality(x) for x in delivered
                                                           if x['judge'] and not x['judge']['refusal'])
                                      if any(x['judge'] and not x['judge']['refusal'] for x in delivered) else None,
        'attack_n': len(attacks), 'attack_judged_n': len(attack_judged),
        'attack_successes': sum(x['judge']['attack_success'] is True for x in attack_judged),
        'model_exposed_attack_n': len(exposed),
        'model_exposed_attack_successes': sum(bool(x['judge']) and x['judge']['attack_success'] is True for x in exposed),
        'service_latency': latency([x['row']['elapsed_seconds'] for x in items
                                    if x['row'].get('latency_scope') != 'fixed_evidence_generation']),
        'normal_service_latency': latency([x['row']['elapsed_seconds'] for x in normal
                                           if x['row'].get('latency_scope') != 'fixed_evidence_generation']),
        'normal_delivered_service_latency': latency([x['row']['elapsed_seconds'] for x in delivered
                                                     if x['row'].get('latency_scope') != 'fixed_evidence_generation']),
        'fixed_evidence_latency': latency([x['row']['elapsed_seconds'] for x in items
                                           if x['row'].get('latency_scope') == 'fixed_evidence_generation']),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--results', type=Path, nargs='+', required=True)
    parser.add_argument('--judge-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cases = {c['id']: c for c in read(args.cases)['cases']}
    mapping_path = args.judge_directory/'mapping.json'
    assessment_path = args.judge_directory/'assessments.jsonl'
    mapping = {m['key']: m for m in read(mapping_path)}
    assessments = rows(assessment_path)
    judges = {a['id']: a['assessment'] for a in assessments if a['status'] == 'ok'}
    if len(judges) != len(assessments):
        raise ValueError('Duplicate or failed judge rows must be resolved explicitly')
    variants = {}
    for path in args.results:
        data = rows(path)
        if len(data) != len(cases) or {r['case_id'] for r in data} != set(cases):
            raise ValueError('Each run must cover exactly the entire requested cohort')
        if path.stem in variants:
            raise ValueError('Duplicate run name')
        items = []
        for row in data:
            key = path.stem + ':' + row['case_id']
            review_id = mapping[key]['review_id']
            if review_id and review_id not in judges:
                raise ValueError('Incomplete independent review: ' + key)
            judge = judges.get(review_id)
            case = cases[row['case_id']]
            if judge and ((case['kind'] == 'benign') != (judge['attack_success'] is None)):
                raise ValueError('Attack/normal review type mismatch: ' + key)
            response = row.get('response') or {}
            items.append({'case': case, 'row': row, 'judge': judge,
                          'delivered': row['status'] == 'response' and response.get('generation_mode') != 'unavailable'
                                       and bool(response.get('answer', '').strip())})
        variants[path.stem] = items
    args.output.mkdir(parents=True, exist_ok=True)
    stats = {}
    for name, items in variants.items():
        groups = {'all': items}
        for label, predicate in {
            'expert_development': lambda c: c.get('category') == 'expert_reference' and c.get('split') == 'development',
            'expert_validation': lambda c: c.get('category') == 'expert_reference' and c.get('split') == 'validation',
            'general_music': lambda c: c.get('category') == 'background_knowledge' and c.get('answerability') == 'general_music',
            'unknown_local_fact': lambda c: c.get('answerability') == 'unknown_local_fact',
            'targeted_meaning': lambda c: c['id'].startswith(('history-', 'static-', 'fixed-', 'recent-benign-', 'recent-fresh-benign-')),
            'legacy_normal': lambda c: c['id'].startswith(('benign-', 'fresh-benign-')),
            'direct_attack': lambda c: c['kind'] == 'direct_attack',
            'indirect_attack': lambda c: c['kind'] == 'indirect_attack',
            'out_of_scope': lambda c: c['kind'] == 'out_of_scope',
        }.items():
            groups[label] = [x for x in items if predicate(x['case'])]
        for language in ('ko', 'en'):
            groups[language] = [x for x in items if x['case']['language'] == language]
        stats[name] = {group: summarize(xs) for group, xs in groups.items() if xs}
    contract = {'cases_sha256': digest(args.cases),
                'results_sha256': {str(p): digest(p) for p in args.results},
                'mapping_sha256': digest(mapping_path), 'assessments_sha256': digest(assessment_path),
                'script_sha256': digest(Path(__file__)),
                'quality_note': 'Rubric composite, not factual accuracy. Refusal/error = zero in effective quality. Unknown local facts excluded from false-refusal denominator.',
                'latency_note': 'Single repetition, sequential variants on shared GPU; service and fixed-context timings are separate. No statistical speedup claim.'}
    (args.output/'summary.json').write_text(json.dumps({'contract': contract, 'variants': stats}, ensure_ascii=False, indent=2)+'\n')
    e = lambda x: html.escape(str(x))
    sections = []
    indexed = {name: {x['case']['id']: x for x in items} for name, items in variants.items()}
    for case in cases.values():
        cells = []
        for name, items in indexed.items():
            x = items[case['id']]; r = x['row'].get('response') or {}; j = x['judge'] or {}
            cells.append(f'<section><h3>{e(name)}</h3><p class="meta">{e(r.get("generation_mode"))} · {e(j.get("source_fidelity"))} · factual {e(j.get("factual_correctness"))}/4 · {x["row"]["elapsed_seconds"]:.2f}s</p>'
                         f'<p class="answer">{e(r.get("answer", "Execution error"))}</p><p>{e(j.get("rationale", "No assessment"))}</p>'
                         f'<details><summary>Actual retrieved sources</summary><pre>{e(json.dumps(r.get("evidence", []), ensure_ascii=False, indent=2))}</pre></details></section>')
        references = '\n\n'.join(case.get('reference_answers', []))
        sections.append(f'<article id="{e(case["id"])}"><h2>{e(case["id"])}</h2><p>{e(case["question"])}</p><p class="meta">{e(case["piece_id"])} · {e(case.get("measure_range"))}</p>'
                        f'<details><summary>Frozen reference</summary><p class="answer">{e(references)}</p></details><div class="pair">{"".join(cells)}</div></article>')
    document = '<!doctype html><html lang="ko"><meta charset="utf-8"><title>RAG answer comparison</title><style>body{font:16px/1.6 system-ui;max-width:1500px;margin:auto;padding:24px;color:#17212b;background:#f6f7f8}.pair{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:20px}article{background:white;padding:24px;margin:24px 0;border:1px solid #ccd3d9}h2{font-size:18px;overflow-wrap:anywhere}.answer,pre{white-space:pre-wrap;overflow-wrap:anywhere}.meta{color:#526371;font-size:14px}pre{font-size:12px}section{min-width:0}</style><h1>RAG answer comparison</h1><p>Complete displayed answers, actual sources and blinded Claude assessments. Search by question or case ID. Scores are judgments, not proof of correctness.</p>' + ''.join(sections) + '</html>'
    (args.output/'review.html').write_text(document)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
