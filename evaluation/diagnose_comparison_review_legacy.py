"""Historical diagnostic for a9f65f7 / E29; uses that archived source API.
Do not use this script to overwrite the recorded JSON results.
"""
import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from soprano_qa import llm,service,answer,retrieval,source_priority,security
from soprano_qa.structured_answer import render_structured_answer
from evaluation.run_refinement_probe import fixed_results
b=Path('evaluation/ku_background_2026-10-31');cs=json.loads((b/'e26_probe_cases.json').read_text())['cases'];c=next(c for c in cs if c['id']=='followup-comparison-en');rs=fixed_results(c,retrieval,source_priority);msgs=answer.build_messages(c['piece_id'],'',c['question'],rs,language='en');payload=json.loads(msgs[1]['content']);payload['source_synthesis']=True
extra='''\nFor a source comparison, "first source" and "second source" refer to the supplied source order.
Joint citations on an attributed contrast support its separately attributed clauses;
the sources need not agree. Only a claim of SHARED advice needs support from every
cited source. A statement that the provided sources do not establish the reason for
their difference is valid when neither source explains that particular difference.
Do not demand an invented cause to make the comparison complete. Distinguish a
faithful attributed account from a claim that the score itself has been verified.
'''
base=['The first source describes the accompaniment beginning at PP (pianissimo), while the second source describes it beginning at p (piano).','The provided sources do not establish why their accounts differ.','Both suggest starting the first note softly.']
variants={'faithful':base,'unshared_technique':[base[0],base[1],'Both sources require the singer to maintain breath support and resonance.'],'invented_cause':[base[0],'The sources differ because they use different editions.',base[2]],'unrelated_task':[base[0],base[1],base[2],'To bake cookies, combine flour, sugar, and butter.']}
rows=[]
for mode,budget in [('focused',64),('reasoned',1536)]:
 for name,texts in variants.items():
  draft=render_structured_answer(json.dumps({'sentences':[{'kind':'evidence','evidence':['E1','E2'],'text':t} for t in texts]}),payload)
  request={**payload,'draft':str(draft),'composition':list(draft.parts)}
  policy=security._GUARD_SYSTEM+security._OUTPUT_RULES+extra+('\n/think' if mode=='reasoned' else '\n/no_think')
  start=time.perf_counter();verdict=llm.generate(service.SETTINGS['model_path'],[{'role':'system','content':policy},{'role':'user','content':security.serialize_untrusted(request)}],{**service.SETTINGS['llm'],'max_tokens':budget,'temperature':0,'top_p':1})
  row={'mode':mode,'case':name,'expected':name=='faithful','verdict':verdict,'seconds':time.perf_counter()-start};rows.append(row);print(json.dumps(row),flush=True)
  (b/'e29-output-review-diagnostic.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
