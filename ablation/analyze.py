"""Summarize all paired conditions, retaining refusals, errors and unfavorable judgments."""
from collections import Counter
from pathlib import Path
import copy, hashlib, html, json, re, statistics

ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.read_text().splitlines()]
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def mean(xs):return statistics.mean(xs) if xs else None
def escaped_answer(text):
    # Preserve displayed text, including generated Markdown hard-break spaces,
    # without storing trailing whitespace in the HTML source.
    return re.sub(r'[ \t]+(?=\n|$)',lambda m:''.join('&#32;' if x==' ' else '&#9;' for x in m[0]),html.escape(text))
LABELS={'g0s0q0':'원본 작성·경고 허용','g1s0q0':'보안·품질 차단만 추가','g0s1q0':'추가 KU 선택만 추가',
        'g0s0q1':'원문 대입만 추가','g1s1q0':'검사+KU 선택','g1s0q1':'검사+원문 대입',
        'g0s1q1':'KU 선택+원문 대입','g1s1q1':'세 변경 모두','warning':'품질 경고와 답변 제공','helpful':'경고+일반 지식 보완',
        'security_only':'현재 보안 검사만 적용','validation_only':'인용·범위 검증 실패만 차단'}

def main():
    cases={c['id']:c for c in read(ROOT/'cases.json')['cases']};completed=rows(ROOT/'results.jsonl')
    assert len(completed)==len(cases)==233 and {x['case_id'] for x in completed}==set(cases),'Incomplete generation'
    outputs={(r['variant'],r['case_id']):r for batch in completed for r in batch['outputs']}
    assert len(outputs)==2330 and all((v,c) in outputs for v in LABELS if v not in ['security_only','validation_only'] for c in cases)
    assessments={};judgment_by_packet={};unique_rows={}
    for batch in sorted((ROOT/'judge-batches').iterdir()):
        packet_index={p['id']:p for p in read(batch/'packets.json')}
        rr={r['id']:r for r in rows(batch/'assessments.jsonl')}
        assert set(rr)==set(packet_index),'Incomplete judge batch'
        for key,r in rr.items():
            if key in judgment_by_packet:assert judgment_by_packet[key]==r['assessment'],'Duplicate packet was rejudged'
            judgment_by_packet[key]=r['assessment'];unique_rows[key]=r
        for m in read(batch/'mapping.json'):
            v=m['key'].split(':',1)[0]; key=(v,m['case_id'])
            assessments[key]=rr[m['review_id']]['assessment']
    assert set(assessments)==set(outputs)
    static_audit=read(ROOT/'static-output-audit.json')
    assert not static_audit['additional_combined_withholding'], 'Static output screen changes the original G decisions; report separately before analysis.'
    extra_static={r['case_id']:r['reason'] for r in static_audit['additional_security_only_withholding']}
    for cid in cases:
        reasons=list(outputs['g1s0q0',cid]['diagnostic']['rejections'])
        if cid in extra_static:reasons.append(extra_static[cid])
        for v,blocked in [('security_only',any(r!='answer_validation_failed' for r in reasons)),
                          ('validation_only','answer_validation_failed' in reasons)]:
            source='g1s0q0' if blocked else 'g0s0q0'
            row=copy.deepcopy(outputs[source,cid]);row['variant']=v
            retained=[r for r in reasons if (r=='answer_validation_failed')==(v=='validation_only')]
            row['diagnostic']={**row['diagnostic'],'derived_from_exact_response':source,'derivation':'ANALYSIS-AMENDMENT.md',
                               'recorded_combined_gate_rejections':outputs['g1s0q0',cid]['diagnostic']['rejections'],
                               'audited_static_output_reason':extra_static.get(cid),'rejections':retained}
            outputs[v,cid]=row;assessments[v,cid]=copy.deepcopy(assessments[source,cid])
    normal=[c for c,x in cases.items() if x['kind']=='benign']
    attacks=[c for c,x in cases.items() if x['kind'] in ['direct_attack','indirect_attack']]
    unrelated=[c for c,x in cases.items() if x['kind']=='out_of_scope']
    def metrics(v,ids):
        valid=[c for c in ids if not assessments[v,c]['refusal']]
        return {'n':len(ids),'false_refusals':sum(assessments[v,c]['refusal'] for c in ids),
                'material_drift':sum(assessments[v,c]['source_fidelity']=='material_drift' for c in ids),
                'fidelity':dict(Counter(assessments[v,c]['source_fidelity'] for c in ids)),
                'naturalness_effective':mean([0 if assessments[v,c]['refusal'] else assessments[v,c]['naturalness'] for c in ids]),
                'naturalness_answered_only':mean([assessments[v,c]['naturalness'] for c in valid]),
                'factuality_effective':mean([0 if assessments[v,c]['refusal'] else assessments[v,c]['factual_correctness'] for c in ids]),
                'completeness_effective':mean([0 if assessments[v,c]['refusal'] else assessments[v,c]['completeness'] for c in ids]),
                'substantial_factual_errors':sum(assessments[v,c]['factual_correctness']<=1 and not assessments[v,c]['refusal'] for c in ids),
                'pipeline_unavailable':sum(outputs[v,c]['response']['generation_mode']=='unavailable' for c in ids)}
    summary={v:{'normal':metrics(v,normal),
                'languages':{l:metrics(v,[c for c in normal if cases[c]['language']==l]) for l in ['ko','en']},
                'cohorts':{group:metrics(v,[c for c in normal if cases[c]['cohort']==group]) for group in sorted({cases[c]['cohort'] for c in normal})},
                'attack_successes':sum(assessments[v,c]['attack_success'] is True for c in attacks),
                'attacks_n':len(attacks),'unassessed_attacks':sum(assessments[v,c]['attack_success'] is None for c in attacks),
                'attack_breakdown':{group:{'successes':sum(assessments[v,c]['attack_success'] is True for c in attacks if cases[c]['kind']==group),
                                            'n':sum(cases[c]['kind']==group for c in attacks)} for group in ['direct_attack','indirect_attack']},
                'unrelated_compliance':sum(assessments[v,c]['attack_success'] is True for c in unrelated),
                'unrelated_n':len(unrelated),'visible_warnings':sum(bool(outputs[v,c]['diagnostic'].get('visible_warning')) for c in normal)} for v in LABELS}
    def compare(a,b,ids):
        return {'from':a,'to':b,'n':len(ids),
                'new_refusals':[c for c in ids if not assessments[a,c]['refusal'] and assessments[b,c]['refusal']],
                'resolved_refusals':[c for c in ids if assessments[a,c]['refusal'] and not assessments[b,c]['refusal']],
                'new_material_drift':[c for c in ids if assessments[a,c]['source_fidelity']!='material_drift' and assessments[b,c]['source_fidelity']=='material_drift'],
                'material_removed_with_answer':[c for c in ids if assessments[a,c]['source_fidelity']=='material_drift' and assessments[b,c]['source_fidelity']!='material_drift' and not assessments[b,c]['refusal']],
                'material_hidden_by_refusal':[c for c in ids if assessments[a,c]['source_fidelity']=='material_drift' and assessments[b,c]['refusal']],
                'naturalness_effective_delta':mean([(0 if assessments[b,c]['refusal'] else assessments[b,c]['naturalness'])-(0 if assessments[a,c]['refusal'] else assessments[a,c]['naturalness']) for c in ids]),
                'completeness_delta':mean([assessments[b,c]['completeness']-assessments[a,c]['completeness'] for c in ids]),
                'both_answered_naturalness_delta':mean([assessments[b,c]['naturalness']-assessments[a,c]['naturalness'] for c in ids if not assessments[a,c]['refusal'] and not assessments[b,c]['refusal']])}
    effects=[]
    for component,pos in [('combined_gate',1),('selection',3),('quotation',5)]:
        for a in sorted(v for v in LABELS if v.startswith('g') and v[pos]=='0'):
            b=a[:pos]+'1'+a[pos+1:]
            effect=compare(a,b,normal);effect['component']=component;effects.append(effect)
    effects += [dict(compare('g1s0q0','warning',normal),component='warning_policy'),dict(compare('warning','helpful',normal),component='general_knowledge_fallback')]
    effects += [dict(compare('g0s0q0',v,normal),component=v) for v in ['security_only','validation_only']]
    quote_applicable=[c for c in normal if outputs['g0s0q1',c]['diagnostic']['quote_applied']]
    conditional_quote=dict(compare('g0s0q0','g0s0q1',quote_applicable),component='quotation_when_applicable')
    gate_counts={v:dict(Counter(reason for c in normal for reason in outputs[v,c]['diagnostic'].get('rejections',[]))) for v in LABELS}
    # Verify the interventions share the exact draft and prompt; no inferred equivalence.
    for c in cases:
        for s in [0,1]:
            hashes={(outputs[f'g{g}s{s}q{q}',c]['diagnostic']['shared_draft_hash'],outputs[f'g{g}s{s}q{q}',c]['diagnostic']['writer_prompt_hash']) for g in [0,1] for q in [0,1]}
            assert len(hashes)==1
    protected=read(ROOT/'protected-sources.json');protection={}
    for name,source in protected.items():
        changed=[p for p,sha in source['files'].items() if hashlib.sha256((Path(source['root'])/p).read_bytes()).hexdigest()!=sha]
        assert not changed,'Protected checkout was modified: '+str(changed)
        protection[name]={'unchanged':True,'verified_files':len(source['files'])}
    contract={'model':read(ROOT/'manifest.json')['generation_model'],'questions':len(cases),'normal':len(normal),'attacks':len(attacks),'unrelated':len(unrelated),
              'outputs':len(outputs),'direct_conditions':10,'derived_decision_conditions':2,'unique_claude_packets':len(unique_rows),'protected_sources':protection,
              'frozen_context_status':dict(Counter(c.get('context_status') for c in cases.values())),
              'attacks_without_baseline_draft':[c for c in attacks if outputs['g0s0q0',c]['diagnostic']['shared_draft_hash']==hashlib.sha256(json.dumps('',ensure_ascii=False,sort_keys=True).encode()).hexdigest()],
              'result_sha256':hashlib.sha256((ROOT/'results.jsonl').read_bytes()).hexdigest(),
              'scope':'Frozen-context component intervention, not full-service replay; shadow/cached calls exclude retrieval and cannot be treated as measured per-arm service latency.',
              'limitations':['Reused diagnostic inventory, not a fresh holdout.','G bundles current gates with strict handling of baseline finalizer failures; JSON drafting is held out.',
                             'S tests the extra selector; full current intent routing and source-priority retrieval are held out.',
                             'Q uses shared draft citations, not the E37 selection-only model call. English includes source translation.',
                             'Nine unavailable historical contexts remain empty and are retained in outcomes.','Claude judgments are model assessments, not human musical adjudication.']}
    measured_calls={};cached_uses=Counter()
    for batch in completed:
        for call in batch['outputs'][0]['calls_for_case']:
            if call['cached']:cached_uses[call['kind']]+=1
            else:measured_calls.setdefault(call['key'],call)
    timing={stage:{'unique_measured_calls':len(items),'mean_seconds':mean([x['seconds'] for x in items]),
                   'median_seconds':statistics.median(x['seconds'] for x in items),
                   'cached_uses':cached_uses[stage]} for stage in sorted({x['kind'] for x in measured_calls.values()})
            if (items:=[x for x in measured_calls.values() if x['kind']==stage])}
    execution={'calls_by_stage':timing,'total_measured_compute_seconds':sum(x['seconds'] for x in measured_calls.values()),
               'limitation':'Per-call component timings with shared caches and offline shadow drafts. Not per-condition service latency; retrieval is excluded.',
               'quotation_applied_normal_cases':len(quote_applicable),'quotation_conditional_effect':conditional_quote}
    execution['static_output_audit']={'additional_combined':len(static_audit['additional_combined_withholding']),
        'additional_security_only':list(extra_static),
        'policy':'Security-only derivation includes the post-run deterministic output screen. Original 10 generated conditions remain unchanged.'}
    # The legacy judge stamps each row separately. Every row repeats the whole
    # batch usage/cost, so divide by batch_size instead of summing row charges.
    execution['judge_batches']={'accepted_calls':round(sum(1/r['batch_size'] for r in unique_rows.values())),
        'invalid_attempts':round(sum(len(r['batch_attempts']['failures'])/r['batch_size'] for r in unique_rows.values())),
        'retry_policy':'Only invalid schema or nonexact quotations retry; first valid verdict retained regardless of quality.',
        'accepted_calls_reported_cost_usd':sum((r.get('batch_cost_usd',0) or 0)/r['batch_size'] for r in unique_rows.values()),
        'cost_note':'Reported accepted CLI responses only; invalid-attempt cost is not included. Not a billing statement.'}
    private_calls=ROOT/'calls.private.jsonl'
    if private_calls.exists():
        execution['generation_finish_reasons']=dict(Counter(r['finish_reason'] for r in rows(private_calls)))
    save(ROOT/'summary.json',{'contract':contract,'variants':summary,'paired_effects':effects,'rejection_reasons':gate_counts,'execution':execution})
    save(ROOT/'assessments-by-variant.json', {v:{c:assessments[v,c] for c in cases} for v in LABELS})
    # Every actual answer is exported, not just selected favorable examples.
    for v in LABELS:
        (ROOT/(v+'.jsonl')).write_text(''.join(json.dumps(outputs[v,c],ensure_ascii=False)+'\n' for c in cases))
    examples=[]
    for e in effects:
        for reason in ['new_refusals','new_material_drift','resolved_refusals','material_removed_with_answer']:
            for cid in e[reason]:
                a,b=e['from'],e['to']
                examples.append({'component':e['component'],'change':reason,'case':cases[cid],
                    'before':outputs[a,cid],'before_assessment':assessments[a,cid],
                    'after':outputs[b,cid],'after_assessment':assessments[b,cid]})
    save(ROOT/'changed-cases.json',examples)
    report=['# 14B 고정: RAG 구성요소 분리 비교','',
            '원본과 현재 production 코드는 수정하지 않았다. 같은 질문·검색 결과·14B 모델을 고정한 별도 실험이다.',
            f"전체 {len(cases)}문항 × (10조건 + 동일 초안에서 파생한 2검사 조건) = {len(outputs)}개 결과, Claude Opus5.5 {len(unique_rows)}개 고유 packet 평가.",'',
            '| 조건 | 정상 오거절 /171 | 중대한 의미 변경 | 자연스러움 유효평균 /4 | 응답한 경우 /4 | 사실성 /4 | 공격 성공 /50 | 비음악 수행 /12 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
    for v,label in LABELS.items():
        z=summary[v];n=z['normal'];report.append(f"| {label} | {n['false_refusals']} | {n['material_drift']} | {n['naturalness_effective']:.3f} | {n['naturalness_answered_only']:.3f} | {n['factuality_effective']:.3f} | {z['attack_successes']} | {z['unrelated_compliance']} |")
    report+=['','거절은 유효 품질 평균에 0점으로 포함했다. 비음악 요청에 서비스 범위를 설명한 것은 정상 음악 질문의 오거절로 세지 않는다.',
             '검사로 잘못된 답변이 숨겨진 경우와, 답변을 제공하면서 의미가 개선된 경우를 분리했다.',
             '현재 보안 검사는 지시 위반뿐 아니라 근거·사실성 판단도 포함한다. 별도의 인용·범위 검증과 구분해서 표시했다.',
             '이 수치는 원본 자유 서술 초안에 각 결정을 적용한 결과다. 인용을 자동 생성하는 현재 구조화 작성 경로 전체의 오거절률은 아니다.','',
             '## 한 요소씩 추가한 결과','', '| 변경 | 새 거절 | 거절 해소 | 새 중대 오류 판정 | 답변 유지하며 중대 판정 해소 | 오류를 거절로 숨김 | 자연스러움 변화 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    chosen=[e for e in effects if e['from']=='g0s0q0' or e['component'] in ['warning_policy','general_knowledge_fallback']]
    for e in chosen:
        report.append(f"| {LABELS[e['from']]} → {LABELS[e['to']]} | {len(e['new_refusals'])} | {len(e['resolved_refusals'])} | {len(e['new_material_drift'])} | {len(e['material_removed_with_answer'])} | {len(e['material_hidden_by_refusal'])} | {e['naturalness_effective_delta']:+.3f} |")
    report+=['','## 구성요소 호출 시간','',
             '캐시를 사용하지 않은 실제 호출만 집계했다. 검색은 제외했으며, 같은 초안을 여러 조건이 공유하고 차단될 요청의 비교용 초안도 생성했으므로 조건별 서비스 응답 시간으로 해석할 수 없다.',
             '', '| 호출 | 고유 호출 수 | 평균 초 | 중앙값 초 |', '|---|---:|---:|---:|']
    for stage,t in timing.items():
        report.append(f"| {stage} | {t['unique_measured_calls']} | {t['mean_seconds']:.3f} | {t['median_seconds']:.3f} |")
    report+=['','중대 오류 판정의 전환이 곧 새로운 오류의 발생을 입증하지는 않는다. 실제 문장에 같은 오류가 이미 있었는데 경미/중대 등급만 달라진 사례는 examples.md에 별도로 기록했다.',
             '','## 읽을 자료','', '- [실제 답변 전문과 직접 검토한 사례](examples.md)','- [전체 수치·언어별 결과·상호작용](summary.json)','- [변경된 모든 사례의 실제 답변과 Claude 판정](changed-cases.json)',
             '- [12조건 전체 답변 비교](review.html)','- [사전 실험 계약](PROTOCOL.md)','- [검사 분리 집계 보완](ANALYSIS-AMENDMENT.md)','- [고정 입력과 모델·출처 해시](manifest.json)','',
             '## 해석 제한','']+['- '+x for x in contract['limitations']]
    (ROOT/'report.md').write_text('\n'.join(report)+'\n')
    # A standalone, local-only inspection artifact. It starts no server.
    body=[]
    for cid,c in cases.items():
        blocks=[]
        for v,label in LABELS.items():
            a=assessments[v,cid];o=outputs[v,cid]
            blocks.append('<article><h3>'+html.escape(label)+'</h3><p class="score">'+html.escape(f"의미: {a['source_fidelity']} · 자연스러움: {a['naturalness']}/4 · 거절: {a['refusal']}")+'</p><pre>'+escaped_answer(o['response']['answer'])+'</pre><p>'+html.escape(a['rationale'])+'</p></article>')
        body.append('<details id="'+html.escape(cid)+'"><summary>'+html.escape(c['question']+' ['+cid+']')+'</summary><div class="grid">'+''.join(blocks)+'</div></details>')
    (ROOT/'review.html').write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>14B 구성요소 비교</title><style>body{font:16px system-ui;max-width:1500px;margin:30px auto;padding:16px}summary{cursor:pointer;padding:16px;background:#eef3f8;margin-top:10px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:12px}article{padding:16px;border:1px solid #ccd;border-radius:8px}pre{white-space:pre-wrap;font:inherit;line-height:1.6}.score{color:#456}</style><h1>14B 고정 구성요소 비교</h1><p>233문항 · 원본 작성 경로 기반 별도 실험. production 변경 없음.</p>'+''.join(body))
    print(json.dumps({'complete':True,**contract},ensure_ascii=False))
if __name__=='__main__':main()
