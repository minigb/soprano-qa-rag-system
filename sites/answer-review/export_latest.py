"""Keep the historical left column and replace the right with the actual latest run."""
from pathlib import Path
import copy, hashlib, json, statistics

SITE=Path(__file__).resolve().parent
RUN=SITE.parents[1]/'evaluation/latest_demo_2026-11-03'
ID='latest-14b-20261103'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n');temp.replace(p)
def main():
    history=read(SITE/'dist/data.json');cases=read(RUN/'cases.json')['cases']
    results=[json.loads(x) for x in (RUN/'current.jsonl').read_text().splitlines()]
    by_id={r['case_id']:r for r in results};provenance=read(RUN/'provenance.json');complete=read(RUN/'completion.json')
    assert len(results)==len(by_id)==len(cases)==len(history['cases'])==460
    assert set(by_id)=={c['id'] for c in cases}=={c['id'] for c in history['cases']}
    assert complete['result_sha256']==sha(RUN/'current.jsonl')
    assert complete['source_contract_sha256']==sha(RUN/'provenance.json')
    data=copy.deepcopy(history)
    for c in data['cases']:
        row=by_id[c['id']];response=row.get('response') or {}
        c['versions']['current']={
            'answer':response.get('answer',''),'status':row['status'],'generation_mode':response.get('generation_mode'),
            'latency_seconds':row['elapsed_seconds'],
            'evidence':[{key:e.get(key) for key in ['id','text','measure_ranges','evidence_type','generation_role']} for e in response.get('evidence',[])],
            'bertscore':{'scored':False,'f1':None,'reason':'not_evaluated','candidate_tokens':None},
            'judge':{'available':False,**{key:None for key in ['source_fidelity','factual_correctness','naturalness','completeness','relevance','attribution','refusal']},
                     'naturalness_rationale':'','rationale':''},'secondary_note':None}
    data.update(evaluation_id=ID,evaluation_date=complete['completed_at'][:10],judge_model=None,
                naturalness_comparison=None,assessment_status='not_evaluated')
    data['summary']['current']={'all':{'material_drifts':None,'naturalness_mean_effective_out_of_4':None,
        'latency_all':{'median_seconds':statistics.median(r['elapsed_seconds'] for r in results)}},
        'original':{'bertscore_effective_f1':None}}
    data['source_hashes']={'historical_demo':sha(SITE/'dist/data.json'),'current_results':sha(RUN/'current.jsonl'),
                           'current_provenance':sha(RUN/'provenance.json'),'cases':sha(RUN/'cases.json')}
    data['generation']={'model':provenance['model'],'source_head':provenance['source_head'],
                        'source_state':'uncommitted semantic-generation WIP','count':len(results),
                        'completed_at':complete['completed_at']}
    evaluated=(RUN/'assessment-complete.json').exists()
    if evaluated:
        assert read(RUN/'assessment-complete.json')['results_sha256']==sha(RUN/'current.jsonl')
        judged={}
        for batch in sorted((RUN/'judge-batches').iterdir()):
            scores={r['id']:r['assessment'] for r in map(json.loads,(batch/'assessments.jsonl').read_text().splitlines())}
            for m in read(batch/'mapping.json'):judged[m['case_id']]=scores.get(m['review_id'])
        metric_summary=read(RUN/'bertscore/summary.json')[0]
        metrics={m['id']:m for m in read(RUN/'bertscore'/(metric_summary['config_id']+'.json'))['cases']}
        assert len(judged)==len(metrics)==460
        for c in data['cases']:
            v=c['versions']['current'];a=judged[c['id']];m=metrics[c['id']]
            if a:v['judge']={'available':True,**{key:a[key] for key in v['judge'] if key!='available'}}
            v['bertscore']={'scored':m['scored'],'f1':m['f1'] if m['scored'] else None,
                'reason':m['unscorable_reason'],'candidate_tokens':m['candidate_tokens']}
        available=[a for a in judged.values() if a]
        data['summary']['current']['all'].update(material_drifts=sum(a['source_fidelity']=='material_drift' for a in available),
            naturalness_mean_effective_out_of_4=sum(a['naturalness'] for a in available if not a['refusal'])/460)
        expert_ids=[c['id'] for c in data['cases'] if c['family']=='original'];assert len(expert_ids)==105
        data['summary']['current']['original']['bertscore_effective_f1']=sum(metrics[c]['f1'] for c in expert_ids)/105
        useful=[c for c in data['cases'] if c['versions']['current']['judge']['available'] and not c['versions']['current']['judge']['refusal']
                and c['versions']['current']['status']=='response' and c['versions']['current']['generation_mode']!='unavailable'
                and not c['versions']['original']['judge']['refusal']]
        data['naturalness_comparison']={'losses':sum(c['versions']['current']['judge']['naturalness']<c['versions']['original']['judge']['naturalness'] for c in useful)}
        data['assessment_status']='complete';data['judge_model']='Claude Opus 5.5'
    save(SITE/'dist'/('data-'+ID+'.json'),data)
    catalog=read(SITE/'dist/evaluations.json')
    entry={'id':ID,'label':'최신 14B · 현재 작업 버전','shortLabel':'최신 14B','answerLabel':'최신 시스템 · 14B',
        'answerSubtitle':'현재 작업 코드의 생성 답변','data':'data-'+ID+'.json',
        'storageKey':'soprano-answer-review:'+ID+':v1','date':complete['completed_at'][:10].replace('-','.'),
        'metricNote':'최신 답변은 아직 BERTScore 미평가',
        'note':'오른쪽은 현재 14B 작업 코드로 460문항을 새로 실행한 실제 답변입니다. 왼쪽 초기 답변은 유지했습니다. 최신 답변의 Claude·BERTScore 평가는 아직 실시하지 않았으며 이전 점수를 가져오지 않았습니다.',
        'methodIntro':'같은 460문항의 초기 답변과 현재 14B 작업 코드의 새 출력을 비교합니다. 답변과 실제 검색 KU는 원문 그대로 표시하며 실패한 요청도 포함합니다. 전체 검색·생성 경로를 실행한 결과입니다. 생성 오류와 Claude의 거절 판정은 상단 자연스러움 평균에서 0점 기여로 포함합니다.',
        'methodBert':'왼쪽 점수는 기존 초기 답변에 대한 평가입니다. 오른쪽 최신 답변은 미평가이며 0점이나 계산 실패가 아닙니다. 과거 E37 또는 구성요소 분리 실험의 점수는 최신 답변에 적용하지 않습니다.',
        'methodScope':'현재 작업 코드의 생성형 경로를 고정한 14B 실행 결과입니다. 직전 구성요소 분리 실험의 조합이나 32B 후보가 아닙니다. 초기 시스템과 검색·코퍼스·프롬프트가 달라질 수 있는 전체 시스템 비교입니다. 양쪽 지연은 서로 다른 시점의 1회 실행 기록이며 통제된 속도 실험이 아닙니다.'}
    if evaluated:
        entry.update(metricNote='105문항 평균 · 기존 BERTScore 설정 유지',
            note='오른쪽은 현재 14B 작업 코드로 460문항을 새로 실행한 실제 답변입니다. 왼쪽 초기 답변은 유지했습니다. 최신 답변은 같은 기준으로 Claude와 BERTScore를 새로 평가했으며, 거절·오류도 포함합니다. 품질 개선을 확정한 배포 버전은 아닙니다.',
            methodBert='새 답변 전문을 기존 xlm-roberta-large 18층, IDF 없음, raw F1 설정으로 평가했습니다. 점수를 위한 답변 수정이나 설정 탐색은 없습니다. 거절·실행 실패와 길이 초과는 유효 평균에 0으로 포함하며 길이 초과를 계산된 점수로 표시하지 않습니다. 높은 유사도가 정확성이나 자연스러움을 보장하지 않습니다.')
    catalog['evaluations']=[entry]+[e for e in catalog['evaluations'] if e['id']!=ID];catalog['default']=ID
    save(SITE/'dist/evaluations.json',catalog)
    print(json.dumps({'cases':460,'original_answers':'unchanged','current_answers':'exact service output','default':ID,'new_scores':'complete' if evaluated else 'not_evaluated'}))
if __name__=='__main__':main()
