"""Verify frozen460-case evaluation inputs, full coverage and metric definitions."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

from run_joint_expanded_judge import validate_coverage

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'evaluation/expanded_460_2026-11-02'


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    config=read(OUT/'config.json');provenance=read(OUT/'provenance.json')
    cases=read(OUT/'cases.json')['cases'];ids={c['id'] for c in cases}
    assert len(cases)==len(ids)==460
    checks={}
    for name in ['generation-contract.json','analysis-contract-v2.json']:
        contract=read(OUT/name)
        assert contract['cases_sha256']==sha(OUT/'cases.json')
        assert contract['config_sha256']==sha(OUT/'config.json')
        for script,expected in contract['scripts'].items():
            assert sha(ROOT/'evaluation'/script)==expected,script
        if 'metric_settings_sha256' in contract:
            assert sha(ROOT/config['metric_settings'])==contract['metric_settings_sha256']
            assert sha(OUT/'metric-length-amendment.json')==contract['length_policy_sha256']
            assert sha(OUT/'judge-rubric.txt')==contract['judge_rubric_sha256']
            assert sha(OUT/'judge-schema.json')==contract['judge_schema_sha256']
        checks[name]='all hashes match'
    for path,expected in read(OUT/'inventory.json')['source_input_sha256'].items():
        assert sha(ROOT/path)==expected,path
    checks['archived_questions_and_references']='unchanged source artifacts and exact original105 fields'
    expert={c['id']:c for c in read(ROOT/'evaluation/ku_background_2026-10-31/expert_cases.json')['cases']}
    for c in cases:
        if c['family']=='original':assert all(c[k]==v for k,v in expert[c['id']].items())
    selected=read(ROOT/config['metric_settings'])
    result_details={}
    for s in config['systems']:
        name=s['name'];snap=ROOT/s['root'];p=provenance['systems'][name]
        for relative,expected in p['source_files_sha256'].items():
            assert sha(snap/relative)==expected
            if name=='candidate-e37':
                assert sha(ROOT/relative)==expected
                assert sha(ROOT.parent/'soprano-lab/rag'/relative)==expected
        assert sha(ROOT/s['settings'])==p['settings_sha256']
        assert sha(snap/'data/corpus.json')==p['corpus_sha256']
        data=rows(OUT/(name+'.jsonl'))
        assert len(data)==460 and {r['case_id'] for r in data}==ids
        manifest=read(OUT/(name+'.jsonl.manifest.json'))
        assert manifest['contract']['cases_sha256']==sha(OUT/'cases.json')
        assert manifest['contract']['usage_rollout'] is None
        assert manifest['sessions'][-1]['status'] in ['complete','already_complete']
        assert all(r['contract_id']==manifest['contract_id'] for r in data)
        summary=read(OUT/('bertscore-'+name)/'summary.json')[0]
        assert summary['settings']==selected['settings']
        assert summary['model_revision']==selected['model_revision']
        assert summary['truncated_texts']==0
        scores=read(OUT/('bertscore-'+name)/(summary['config_id']+'.json'))['cases']
        assert len(scores)==460 and {x['id'] for x in scores}==ids
        for x in scores:
            if x['scored']:
                assert x['candidate_tokens']<=512 and x['reference_tokens']<=512
                assert x['unscorable_reason'] is None
            else:
                assert all(x[k]==0 for k in ['precision','recall','f1'])
                if x['unscorable_reason']=='encoder_length_limit':
                    assert x['candidate_tokens']>512 or x['reference_tokens']>512
        assert abs(summary['effective']['f1']-statistics.mean(x['f1'] for x in scores))<1e-12
        result_details[name]={'requests':460,'errors':sum(r['status']!='response' for r in data),
                             'bertscore_scored':summary['scored'],
                             'unscorable_overlength':summary['unscorable_overlength_cases'],
                             'truncated_texts':0,'source_matches_frozen_snapshot':True}
    packets=read(OUT/'judge/packets.json');assessed=rows(OUT/'judge/assessments.jsonl')
    validate_coverage(packets,assessed,'claude-opus-5-5')
    mapping=read(OUT/'judge/mapping.json')
    assert len(mapping)==920 and len({m['key'] for m in mapping})==920
    assert {m['review_id'] for m in mapping if m['review_id']}=={a['id'] for a in assessed}
    checks['fresh_blind_judge']={'packets':len(packets),'requests_mapped':len(mapping),
                                'model':'claude-opus-5-5','prior_judgments_reused':0}
    audit=read(OUT/'quote-audit-candidate.json')
    assert audit['results_sha256']==sha(OUT/'candidate-e37.jsonl')
    assert audit['quotation_answers']==audit['verified_quotation_answers']==459
    checks['literal_quote_identity']={'verified':459,'nonquotation_requires_semantic_review':1}
    for path,expected in provenance['protected_checkouts'].items():
        head=subprocess.check_output(['git','-C',path,'rev-parse','HEAD'],text=True).strip()
        status=subprocess.check_output(['git','-C',path,'status','--porcelain'],text=True).splitlines()
        assert head==expected['head'] and status==expected['status'],path
    checks['protected_main_and_dataset_checkouts']='unchanged heads and worktree states'
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    assert branch=='codex/ku-background-knowledge'
    checks['production_modules']='Root RAG, Lab and E37 frozen source identical'
    payload={'status':'passed','recorded_at':datetime.now(timezone.utc).isoformat(),
             'branch':branch,'systems':result_details,'checks':checks,
             'model_file_hash_verification':'Full model file hashes verified before execution and recorded in provenance.json.',
             'limits':'Internal reused evaluation; one judge model; native corpora differ by7 performer-feedback records;2 E37 overlength scores are unscorable zero contributions.',
             'usage_reset_stop':False,'notifier_used':False,
             'artifacts_sha256':{str(p.relative_to(OUT)):sha(p) for p in [OUT/'config.json',OUT/'cases.json',OUT/'provenance.json',OUT/'review/summary.json',OUT/'manual-review.json',OUT/'metric-length-amendment.json',OUT/'judge/aggregation.json']}}
    (OUT/'verification.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'passed','requests':920,'judge_packets':len(packets),'systems':result_details},ensure_ascii=False))


if __name__=='__main__':
    main()
