"""Verify all current answers and the unchanged historical left column independently."""
from pathlib import Path
import hashlib,json,subprocess
SITE=Path(__file__).resolve().parent
RUN=SITE.parents[1]/'evaluation/helpful_responses_2026-11-04/full-final'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    catalog=read(SITE/'dist/evaluations.json');active=next(e for e in catalog['evaluations'] if e['id']=='helpful-14b-20261104')
    data=read(SITE/'dist'/active['data']);old=read(SITE/'dist/data.json')
    raw={r['case_id']:r for r in map(json.loads,(RUN/'current.jsonl').read_text().splitlines())}
    cases={c['id']:c for c in read(RUN/'cases.json')['cases']}
    original={c['id']:c for c in old['cases']}
    assert len(data['cases'])==len(raw)==len(cases)==460
    assert data['source_hashes']['current_results']==sha(RUN/'current.jsonl')
    assert data['source_hashes']['historical_demo']==sha(SITE/'dist/data.json')
    evaluated=data.get('assessment_status')=='complete'
    judged={};metrics={}
    if evaluated:
        for batch in sorted((RUN/'judge-batches').iterdir()):
            assessments={r['id']:r['assessment'] for r in map(json.loads,(batch/'assessments.jsonl').read_text().splitlines())}
            for mapping in read(batch/'mapping.json'):judged[mapping['case_id']]=assessments.get(mapping['review_id'])
        metric=read(RUN/'bertscore/summary.json')[0]
        metrics={r['id']:r for r in read(RUN/'bertscore'/(metric['config_id']+'.json'))['cases']}
        contract=read(RUN/'bertscore/contract.json')
        assert contract['results_sha256']==sha(RUN/'metric-input.jsonl')
        assert read(RUN/'assessment-complete.json')['results_sha256']==sha(RUN/'current.jsonl')
        assert len(judged)==len(metrics)==460
    for c in data['cases']:
        assert {k:v for k,v in c.items() if k!='versions'}=={k:v for k,v in original[c['id']].items() if k!='versions'}
        assert c['question']==cases[c['id']]['question']
        assert c['versions']['original']==original[c['id']]['versions']['original']
        v=c['versions']['current'];r=raw[c['id']];response=r.get('response') or {}
        assert v['answer']==response.get('answer','')
        assert v['latency_seconds']==r['elapsed_seconds'] and v['status']==r['status']
        assert v['generation_mode']==response.get('generation_mode')
        assert v['generation_validation_warning']==response.get('generation_validation_warning')
        assert v['answer_basis']==response.get('answer_basis')
        assert [e['text'] for e in v['evidence']]==[e['text'] for e in response.get('evidence',[])]
        if evaluated:
            a=judged[c['id']];m=metrics[c['id']]
            if a:assert all(value==a[k] for k,value in v['judge'].items() if k!='available')
            else:assert v['judge']['available'] is False
            assert v['bertscore']['f1']==(m['f1'] if m['scored'] else None)
            assert v['bertscore']['reason']==m['unscorable_reason']
        else:
            assert v['judge']['available'] is False and v['judge']['naturalness'] is None
            assert v['bertscore']['f1'] is None and v['bertscore']['reason']=='not_evaluated'
    assert data['summary']['original']==old['summary']['original']
    previous=read(SITE/'dist/data-latest-14b-20261103.json')
    refused={c['id'] for c in previous['cases'] if c['versions']['current']['generation_mode']=='unavailable'}
    assert len(refused)==5
    assert {c['id'] for c in data['cases'] if c['versions']['current']['previous_refusal']}==refused
    if evaluated:
        assert data['summary']['current']['all']['material_drifts']==sum(a['source_fidelity']=='material_drift' for a in judged.values() if a)
        expert=[c['id'] for c in data['cases'] if c['family']=='original']
        assert data['summary']['current']['original']['bertscore_effective_f1']==sum(metrics[c]['f1'] for c in expert)/len(expert)
    else:
        assert data['summary']['current']['all']['material_drifts'] is None
        assert data['summary']['current']['original']['bertscore_effective_f1'] is None
    serialized=(SITE/'dist'/active['data']).read_text()
    assert not any(x in serialized for x in ['/home/minhee/','private_trace_attempt_id','batch_usage','model_path'])
    assert len({e['storageKey'] for e in catalog['evaluations']})==len(catalog['evaluations'])
    provenance=read(RUN/'provenance.json')
    for rel,digest in provenance['source_files'].items():
        # This is now a historical capture; later branch changes are expected.
        captured=subprocess.check_output(['git','show',f"{provenance['source_head']}:{rel}"],cwd=SITE.parents[1])
        assert hashlib.sha256(captured).hexdigest()==digest
    print(json.dumps({'cases':460,'full_answers_verified':920,'left':'unchanged','right':'exact current 14B service outputs',
                      'implementation':'capture commit verified against frozen source hashes','stale_scores':'excluded','notes':'separate version keys'}))
if __name__=='__main__':main()
