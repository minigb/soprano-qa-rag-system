"""Summarize all460 requests, paired outcomes and explicit naturalness judgments."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import html
import json
from pathlib import Path
import random
import statistics

from run_joint_expanded_judge import validate_coverage

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'evaluation/expanded_460_2026-11-02'


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(values):
    return statistics.mean(values) if values else None


def latency(values):
    ordered = sorted(values)
    return {'count': len(values), 'mean_seconds': mean(values),
            'median_seconds': statistics.median(values) if values else None,
            'p90_seconds': ordered[int((len(ordered)-1)*.9)] if values else None}


def effective(item, metric):
    return item['judge'][metric] if item['useful'] else 0


def summarize(items):
    useful = [i for i in items if i['useful']]
    assessed = [i for i in items if i['judge'] is not None]
    return {
        'cases': len(items), 'useful_answers': len(useful),
        'execution_errors': sum(i['row']['status'] != 'response' for i in items),
        'false_refusals': sum(i['row']['status'] == 'response' and not i['useful'] for i in items),
        'source_fidelity': dict(Counter(i['judge']['source_fidelity'] for i in assessed)),
        'material_drifts': sum(i['judge']['source_fidelity'] == 'material_drift' for i in assessed),
        'material_drift_rate_all_requests': sum(i['judge']['source_fidelity'] == 'material_drift' for i in assessed)/len(items),
        'substantial_factual_errors': sum(i['judge']['factual_correctness'] <= 1 for i in useful),
        'mixed_or_uncertain_facts': sum(i['judge']['factual_correctness'] == 2 for i in useful),
        'naturalness_distribution_useful': dict(Counter(i['judge']['naturalness'] for i in useful)),
        'naturalness_mean_useful_out_of_4': mean([i['judge']['naturalness'] for i in useful]),
        'naturalness_mean_effective_out_of_4': mean([effective(i, 'naturalness') for i in items]),
        'factual_mean_effective_out_of_4': mean([effective(i, 'factual_correctness') for i in items]),
        'completeness_mean_effective_out_of_2': mean([effective(i, 'completeness') for i in items]),
        'bertscore_effective_f1': mean([i['bert']['f1'] for i in items]),
        'bertscore_judge_adjusted_effective_f1': mean([i['bert']['f1'] if i['useful'] else 0 for i in items]),
        'bertscore_delivered': sum(i['bert']['delivered'] for i in items),
        'latency_all': latency([i['row']['elapsed_seconds'] for i in items]),
        'generation_modes': dict(Counter((i['row'].get('response') or {}).get('generation_mode') for i in items)),
    }


def paired_cluster_interval(pairs, metric, *, iterations=2000):
    """Request-weighted paired bootstrap, resampling source-question groups."""
    groups = defaultdict(list)
    for old, new in pairs:
        groups[new['case']['source_group_id']].append(effective(new, metric)-effective(old, metric))
    totals = [(sum(xs), len(xs)) for xs in groups.values()]
    rng = random.Random(20261102)
    samples = []
    for _ in range(iterations):
        chosen = rng.choices(totals, k=len(totals))
        samples.append(sum(x[0] for x in chosen)/sum(x[1] for x in chosen))
    samples.sort()
    return {'mean_difference': sum(x[0] for x in totals)/sum(x[1] for x in totals),
            'percentile_95_interval': [samples[int(.025*iterations)], samples[int(.975*iterations)]],
            'source_groups': len(totals), 'iterations': iterations, 'seed': 20261102,
            'scope': 'Case-sampling uncertainty only; not judge/model/run variability.'}


def main():
    config = read(OUT/'config.json')
    cases = {c['id']: c for c in read(OUT/'cases.json')['cases']}
    assert len(cases) == 460
    packets = read(OUT/'judge/packets.json')
    assessments = rows(OUT/'judge/assessments.jsonl')
    validate_coverage(packets, assessments, 'claude-opus-5-5')
    judged = {r['id']: r['assessment'] for r in assessments}
    mapping_rows = read(OUT/'judge/mapping.json')
    mapping = {r['key']: r for r in mapping_rows}
    assert len(mapping) == len(mapping_rows) == 920
    variants = {}
    metric_files = {}
    for system in config['systems']:
        name = system['name']
        data = rows(OUT/(name+'.jsonl'))
        assert len(data) == 460 and {r['case_id'] for r in data} == set(cases)
        summaries = read(OUT/('bertscore-'+name)/'summary.json')
        assert len(summaries) == 1 and summaries[0]['count'] == 460
        assert not summaries[0]['truncated_texts']
        score_file = OUT/('bertscore-'+name)/(summaries[0]['config_id']+'.json')
        bert = {r['id']: r for r in read(score_file)['cases']}
        assert set(bert) == set(cases)
        metric_files[name] = str(score_file.relative_to(OUT))
        items = {}
        for row in data:
            case = cases[row['case_id']]
            m = mapping[name+':'+case['id']]
            judge = judged.get(m['review_id'])
            response = row.get('response') or {}
            delivered = row['status'] == 'response' and response.get('generation_mode') != 'unavailable' and bool(response.get('answer', '').strip())
            assert (judge is not None) == (row['status'] == 'response')
            if judge:
                assert judge['attack_success'] is None
            items[case['id']] = {'case': case, 'row': row, 'judge': judge, 'bert': bert[case['id']],
                                'delivered': delivered, 'useful': bool(delivered and judge and not judge['refusal'])}
        variants[name] = items
    stats = {}
    for name, indexed in variants.items():
        items = list(indexed.values())
        groups = {'all': items}
        for family in ['original', 'paraphrase', 'ku_derived']:
            groups[family] = [i for i in items if i['case']['family'] == family]
        for split in ['development', 'validation']:
            groups['original_'+split] = [i for i in groups['original'] if i['case']['split'] == split]
        for piece in sorted({c['piece_id'] for c in cases.values()}):
            groups['piece_'+piece] = [i for i in items if i['case']['piece_id'] == piece]
        stats[name] = {label: summarize(xs) for label, xs in groups.items()}
    old, new = variants['pre-security'], variants['candidate-e37']
    pairs = [(old[k], new[k]) for k in cases]
    both_useful = [(a, b) for a, b in pairs if a['useful'] and b['useful']]
    material = lambda i: bool(i['judge'] and i['judge']['source_fidelity'] == 'material_drift')
    paired = {
        'both_useful': len(both_useful),
        'new_material_drift_ids': [b['case']['id'] for a,b in pairs if material(b) and not material(a)],
        'resolved_material_drift_ids': [b['case']['id'] for a,b in pairs if material(a) and not material(b) and b['useful']],
        'old_material_but_current_no_useful_answer_ids': [b['case']['id'] for a,b in pairs if material(a) and not b['useful']],
        'current_no_useful_answer_ids': [b['case']['id'] for a,b in pairs if not b['useful']],
        'naturalness_both_useful': {
            'original_mean': mean([a['judge']['naturalness'] for a,b in both_useful]),
            'current_mean': mean([b['judge']['naturalness'] for a,b in both_useful]),
            'wins': sum(b['judge']['naturalness'] > a['judge']['naturalness'] for a,b in both_useful),
            'ties': sum(b['judge']['naturalness'] == a['judge']['naturalness'] for a,b in both_useful),
            'losses': sum(b['judge']['naturalness'] < a['judge']['naturalness'] for a,b in both_useful),
        },
        'naturalness_effective_cluster_bootstrap': paired_cluster_interval(pairs, 'naturalness'),
        'factual_effective_cluster_bootstrap': paired_cluster_interval(pairs, 'factual_correctness'),
        'latency_difference_both_useful': latency([b['row']['elapsed_seconds']-a['row']['elapsed_seconds'] for a,b in both_useful]),
    }
    contract = {'cases_sha256': sha(OUT/'cases.json'),
                'results_sha256': {name: sha(OUT/(name+'.jsonl')) for name in variants},
                'assessments_sha256': sha(OUT/'judge/assessments.jsonl'),
                'mapping_sha256': sha(OUT/'judge/mapping.json'),
                'metric_file_sha256': {name: sha(OUT/p) for name,p in metric_files.items()},
                'summary_script_sha256': sha(Path(__file__)),
                'failure_policy': 'All460 requests retained. Effective naturalness counts errors and pipeline/judge refusals as zero. Frozen BERTScore counts pipeline-declared unavailable/errors as zero; judge-adjusted BERTScore additionally zeroes judge-only refusals. Useful-answer naturalness and error counts are separate.',
                'bert_policy': 'Full displayed text and frozen references/settings. Original43/62 split retained; expanded families reported separately. No threshold-based filtering.',
                'latency_policy': 'One serial run per version; descriptive timings, no statistical speedup claim.',
                'corpus_policy': config['protocol']['corpus_policy']}
    review = OUT/'review'; review.mkdir(exist_ok=True)
    (review/'summary.json').write_text(json.dumps({'contract': contract, 'variants': stats, 'paired': paired}, ensure_ascii=False, indent=2)+'\n')
    worklist = []
    for a,b in pairs:
        j=b['judge']
        flags=[]
        if not b['useful']: flags.append('current_no_useful_answer')
        if material(b): flags.append('current_material_drift')
        if b['useful'] and j['factual_correctness'] <= 2: flags.append('current_fact_score_at_most2')
        if b['useful'] and j['naturalness'] <= 2: flags.append('current_naturalness_at_most2')
        if a['useful'] and b['useful'] and j['naturalness'] < a['judge']['naturalness']: flags.append('naturalness_declined')
        if flags:worklist.append({'id': b['case']['id'], 'family': b['case']['family'], 'flags': flags})
    (review/'manual-review-worklist.json').write_text(json.dumps(worklist,ensure_ascii=False,indent=2)+'\n')
    e=lambda value:html.escape(str(value))
    sections=[]
    for case in cases.values():
        cells=[]
        for name in ['pre-security','candidate-e37']:
            i=variants[name][case['id']];j=i['judge'] or {};r=i['row'].get('response') or {}
            cells.append(f'<section><h3>{e(name)}</h3><p class="meta">{e(j.get("source_fidelity"))} · factual {e(j.get("factual_correctness"))}/4 · naturalness {e(j.get("naturalness"))}/4 · BERT F1 {i["bert"]["f1"]:.4f} · {i["row"]["elapsed_seconds"]:.2f}s</p><p class="answer">{e(r.get("answer", "Execution error"))}</p><p>{e(j.get("rationale", i["row"].get("exception")))}</p><p>{e(j.get("naturalness_rationale", ""))}</p><details><summary>Retrieved evidence</summary><pre>{e(json.dumps(r.get("evidence",[]),ensure_ascii=False,indent=2))}</pre></details></section>')
        sections.append(f'<article data-family="{e(case["family"])}"><h2>{e(case["id"])}</h2><p>{e(case["question"])} · {e(case["measure_range"])}</p><div class="pair">'+''.join(cells)+f'</div><details><summary>Frozen expert reference</summary><p class="answer">{e(case["reference_answers"][0])}</p></details></article>')
    doc='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Expanded460: original vs E37</title><style>body{font:16px/1.65 system-ui;margin:30px auto;max-width:1500px;padding:0 20px;color:#20252b}article{border-top:2px solid #b9c2cc;margin:35px 0;padding:20px 0}.pair{display:grid;grid-template-columns:1fr 1fr;gap:24px}section{background:#f3f5f7;padding:20px;border-radius:8px}.answer,pre{white-space:pre-wrap;overflow-wrap:anywhere}pre{font-size:12px}.meta{color:#506070}input,select{font:inherit;padding:8px}header{position:sticky;top:0;background:white;padding:15px 0;border-bottom:1px solid #ddd}h2{font-size:18px}@media(max-width:850px){.pair{grid-template-columns:1fr}}</style><header><strong>원본 시스템 ↔ E37 ·460 cases</strong><br><input id="q" placeholder="질문·ID·평가 내용 검색"><select id="family"><option value="">전체</option><option>original</option><option>paraphrase</option><option>ku_derived</option></select> <span id="n"></span></header><p>Fresh blinded Claude Opus5.5 judgments. Naturalness is separate from factual/source fidelity. Both versions retain errors/refusals. Native corpus difference:946 shared records; E37 adds7 performer-feedback records. Reused internal evaluation.</p>'''+''.join(sections)+'''<script>const q=document.querySelector('#q'),f=document.querySelector('#family'),articles=[...document.querySelectorAll('article')];function filter(){let n=0;for(const a of articles){const show=(!f.value||a.dataset.family===f.value)&&a.textContent.toLowerCase().includes(q.value.toLowerCase());a.hidden=!show;n+=show;}document.querySelector('#n').textContent=n+' /460';}q.oninput=filter;f.onchange=filter;filter();</script></html>'''
    (review/'review.html').write_text(doc)
    print(json.dumps({'cases':len(cases),'systems':{name:stat['all'] for name,stat in stats.items()},'manual_review_cases':len(worklist)},ensure_ascii=False))


if __name__ == '__main__':
    main()
