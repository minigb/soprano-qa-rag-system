"""Bind direct semantic reviews to exact answers and render a portable audit.

This tool never infers correctness from citation IDs, similarity or a guard
verdict. Delivered answers require an explicit reviewer assessment. Unavailable
responses are counted separately and never treated as faithful answers.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
VARIANTS = ('original', 'hardened')
LABELS = {'original': 'Before refinement (29fe3e2)', 'hardened': 'Current refinement (79cbd7b)'}


def read(path):
    return json.loads(path.read_text())


def write(name, data):
    (ROOT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def packets():
    cases = {c['id']: c for c in read(ROOT / 'cases.json')['cases']}
    result = []
    seen = set()
    for variant, filename in [('original', 'baseline.jsonl'), ('hardened', 'refined.jsonl')]:
        path = ROOT / filename
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            row = json.loads(line)
            case = cases[row['case_id']]
            key = variant + ':' + row['case_id']
            if key in seen or row['variant'] != variant:
                raise ValueError('Duplicate or mismatched response identity')
            seen.add(key)
            response = row.get('response') or {}
            outcome = 'execution_error' if row['status'] != 'response' else (
                'abstained' if response.get('generation_mode') == 'unavailable'
                or not response.get('answer', '').strip() else 'answer'
            )
            packet = {
                'id': key, 'case_id': case['id'], 'variant': variant,
                'question': case['question'], 'piece_id': case['piece_id'],
                'language': case['language'], 'measure_range': case.get('measure_range'),
                'meaning_constraints': case['meaning_constraints'],
                'reference_answers': case.get('reference_answers', []),
                'outcome': outcome, 'answer': response.get('answer'),
                'evidence': response.get('evidence', []),
                'score_context': response.get('score_context'),
                'unavailable_reason': response.get('unavailable_reason'),
                'exception': row.get('exception'),
                'latency_scope': row.get('latency_scope'),
            }
            packet['review_hash'] = hashlib.sha256(
                json.dumps(packet, ensure_ascii=False, sort_keys=True).encode()
            ).hexdigest()
            result.append(packet)
    write('review_packets.json', result)
    return cases, result


def summarize(cases, data):
    protocol = read(ROOT / 'protocol.json')
    if hashlib.sha256((ROOT / 'cases.json').read_bytes()).hexdigest() != protocol['cases_sha256']:
        raise ValueError('Frozen cases changed after inference')
    runs = {}
    for variant, name in [('original', 'baseline'), ('hardened', 'refined')]:
        manifest = read(ROOT / (name + '.jsonl.manifest.json'))
        if (manifest['contract']['cases_sha256'] != protocol['cases_sha256']
                or manifest['contract']['variant'] != variant
                or manifest['sessions'][-1]['status'] != 'complete'):
            raise ValueError('Run manifest is incomplete or mismatched')
        runs[variant] = [json.loads(line) for line in (ROOT / (name + '.jsonl')).read_text().splitlines()]
        if any(row['contract_id'] != manifest['contract_id'] or row['repeat'] != 0
               for row in runs[variant]):
            raise ValueError('Mixed run contract or unexpected repetition')
    expected = {v + ':' + c for c in cases for v in VARIANTS}
    if {p['id'] for p in data} != expected:
        raise ValueError('Both complete runs are required')
    reviews = read(ROOT / 'assessments.json')
    indexed = {r['id']: r for r in reviews}
    if len(indexed) != len(reviews):
        raise ValueError('Duplicate assessment')
    needed = {p['id'] for p in data if p['outcome'] == 'answer'}
    if set(indexed) != needed:
        raise ValueError(f'Review coverage mismatch: missing={sorted(needed-set(indexed))}, extra={sorted(set(indexed)-needed)}')
    for packet in data:
        if packet['outcome'] != 'answer':
            packet['assessment'] = {'meaning': packet['outcome'], 'coverage': 'no_answer', 'rationale': packet['unavailable_reason'] or str(packet['exception'])}
            continue
        review = indexed[packet['id']]
        if review.get('review_hash') != packet['review_hash']:
            raise ValueError('Stale review: ' + packet['id'])
        if review['meaning'] not in {'faithful', 'minor_drift', 'material_drift'} or review['coverage'] not in {'adequate', 'partial'}:
            raise ValueError('Invalid meaning/coverage verdict')
        if not review.get('rationale'):
            raise ValueError('A rationale is required for every reviewed answer')
        if review['meaning'] != 'faithful' and not all(review.get(k) for k in ('answer_excerpt', 'source_excerpt', 'faithful_alternative')):
            raise ValueError('Drift findings require actual excerpts and an editorial faithful alternative')
        if review.get('answer_excerpt') and review['answer_excerpt'] not in packet['answer']:
            raise ValueError('Answer excerpt does not match delivered prose')
        if review.get('source_excerpt'):
            sources = [e.get('text', '') for e in packet['evidence']]
            if not any(review['source_excerpt'] in source for source in sources):
                raise ValueError('Source excerpt does not match the returned evidence')
        packet['assessment'] = review

    result = {'requests': len(data), 'cases_per_variant': len(cases),
              'method': 'Direct Codex review, targeted authored diagnostics; not independent musical expert review or a general accuracy estimate.',
              'variants': {}, 'paired_transitions': {}, 'material_regressions': [],
              'material_regression_inputs': []}
    for variant in VARIANTS:
        subset = [p for p in data if p['variant'] == variant]
        result['variants'][variant] = {
            'meaning': dict(Counter(p['assessment']['meaning'] for p in subset)),
            'coverage': dict(Counter(p['assessment']['coverage'] for p in subset)),
            'by_language': {lang: dict(Counter(p['assessment']['meaning'] for p in subset if p['language'] == lang)) for lang in ('ko', 'en')},
            'by_path': {scope: dict(Counter(p['assessment']['meaning'] for p in subset if p['latency_scope'] == scope)) for scope in sorted({p['latency_scope'] for p in subset})},
            'latency_seconds_by_path': {
                scope: {
                    'count': len(values),
                    'median': statistics.median(values),
                    'mean': statistics.mean(values),
                }
                for scope in sorted({p['latency_scope'] for p in subset})
                for values in [[r['elapsed_seconds'] for r in runs[variant] if r['latency_scope'] == scope]]
            },
        }
    by_id = {p['id']: p for p in data}
    transitions = Counter()
    for case in cases:
        before = by_id['original:' + case]['assessment']['meaning']
        after = by_id['hardened:' + case]['assessment']['meaning']
        transitions[before + ' -> ' + after] += 1
        if before in {'faithful', 'minor_drift'} and after == 'material_drift':
            result['material_regressions'].append(case)
            def support_signature(packet):
                return [(e.get('id'), e.get('text'), e.get('generation_role'),
                         e.get('measure_ranges'), e.get('scope_match'))
                        for e in packet['evidence']]
            left = support_signature(by_id['original:' + case])
            right = support_signature(by_id['hardened:' + case])
            result['material_regression_inputs'].append({
                'case_id': case,
                'same_evidence_text_order_and_scope': left == right,
                'same_evidence_text_set_and_scope': sorted(left) == sorted(right),
            })
    result['paired_transitions'] = dict(transitions)
    write('summary.json', result)
    write('reviewed_answers.json', data)
    render(cases, by_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def render(cases, by_id):
    esc = lambda value: html.escape(str(value or ''))
    sections = []
    for cid, case in cases.items():
        pair = [by_id[v + ':' + cid] for v in VARIANTS]
        current = pair[1]['assessment']['meaning']
        columns = []
        for packet in pair:
            review = packet['assessment']
            evidence = ''.join('<li><strong>' + esc(e.get('id')) + '</strong><p>' + esc(e.get('text')) + '</p></li>' for e in packet['evidence'])
            alternative = '<h4>Reviewer’s faithful wording (not model output)</h4><p>' + esc(review['faithful_alternative']) + '</p>' if review.get('faithful_alternative') else ''
            columns.append('<div><h3>' + esc(LABELS[packet['variant']]) + '</h3><b>' + esc(review['meaning']) + ' · ' + esc(review['coverage']) + '</b><blockquote>' + esc(packet['answer']) + '</blockquote><p>' + esc(review['rationale']) + '</p>' + alternative + '<details><summary>Evidence supplied with this answer</summary><ul>' + evidence + '</ul></details></div>')
        sections.append('<section data-current="' + current + '" data-language="' + case['language'] + '"><h2>' + esc(case['question']) + '</h2><p class="meta">' + esc(cid) + ' · ' + esc(case.get('measure_range') or 'No selected range') + '</p><p><b>Meaning to preserve:</b> ' + esc(case['meaning_constraints']) + '</p><div class="pair">' + ''.join(columns) + '</div></section>')
    page = '''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Meaning fidelity audit</title>
<style>body{font:16px/1.6 system-ui,sans-serif;max-width:1450px;margin:auto;padding:2rem;color:#17212b;background:#f6f8fa}h1,h2,h3{line-height:1.3}h2{font-size:1.2rem}.pair{display:grid;grid-template-columns:1fr 1fr;gap:2rem}section{background:white;border:1px solid #d0d7de;border-radius:8px;margin:1.5rem 0;padding:1.5rem}p,blockquote{white-space:pre-wrap}blockquote{border-left:3px solid #8796a5;margin:1rem 0;padding-left:1rem}.meta{font-size:.85rem;color:#57606a}select,input{font:inherit;padding:.4rem;margin-right:1rem}summary{cursor:pointer}li{margin:1rem 0}h4{margin-bottom:0}header{position:sticky;top:0;background:#f6f8fa;padding:1rem 0;border-bottom:1px solid #d0d7de}@media(max-width:850px){.pair{grid-template-columns:1fr}body{padding:1rem}}</style>
<h1>Do the answers preserve their sources’ meanings?</h1><p>41 targeted cases, two systems. Direct Codex review; not independent expert validation or a production accuracy estimate. Editorial alternatives are explicitly separate from actual generated answers. No displayed demo answer was changed.</p>
<header><label>Current result <select id="status"><option value="">All</option><option>material_drift</option><option>minor_drift</option><option>faithful</option><option>abstained</option></select></label><label>Language <select id="language"><option value="">All</option><option>ko</option><option>en</option></select></label><input id="search" placeholder="Find a question or phrase" aria-label="Find a question or phrase"><span id="count"></span></header>
''' + ''.join(sections) + '''<script>function filter(){let count=0;for(const item of document.querySelectorAll('section')){const visible=(!statusBox.value||item.dataset.current===statusBox.value)&&(!languageBox.value||item.dataset.language===languageBox.value)&&item.textContent.toLowerCase().includes(searchBox.value.toLowerCase());item.hidden=!visible;if(visible)count++;}document.querySelector('#count').textContent=count+' cases';}const statusBox=document.querySelector('#status'),languageBox=document.querySelector('#language'),searchBox=document.querySelector('#search');for(const control of [statusBox,languageBox,searchBox])control.addEventListener('input',filter);filter();</script>'''
    (ROOT / 'review.html').write_text(page, encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['packets', 'summarize'])
    args = parser.parse_args()
    cases, data = packets()
    if args.action == 'summarize':
        summarize(cases, data)
    else:
        print('Prepared', len(data), 'response packets')
