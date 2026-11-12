"""Historical diagnostic for b017c15 / E28; uses that archived source API.
Do not use this script to overwrite the recorded JSON results.
"""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from soprano_qa import llm,service,answer,source_comparison,retrieval,source_priority
from soprano_qa.security import serialize_untrusted,semantic_guard,SECURITY_SYSTEM_POLICY,screen_generated_answer
from evaluation.run_refinement_probe import fixed_results
b=Path('evaluation/ku_background_2026-10-31');cs=json.loads((b/'e26_probe_cases.json').read_text())['cases'];ref=next(c for c in cs if c['id']=='fixed-observed-p-pp-1')
sources=[{'citation_label':f'E{i+1}','answer':x['record']['answer']} for i,x in enumerate(ref['selected_pool'])]
results=[]
raw=llm.generate(service.SETTINGS['model_path'],[{'role':'system','content':source_comparison.RELATIONSHIP_POLICY.replace('/no_think','/think')},{'role':'user','content':serialize_untrusted({'sources':sources})}],{**service.SETTINGS['llm'],'max_tokens':1536})
print('relationship',raw,flush=True);results.append({'stage':'relationship','final':raw})
for c in cs:
 if c['id'] not in {'fixed-observed-p-pp-1','fixed-observed-p-pp-2','followup-comparison-en'}:continue
 rs=fixed_results(c,retrieval,source_priority);msgs=answer.build_messages(c['piece_id'],'',c['question'],rs,language=c['language']);payload=json.loads(msgs[1]['content'])
 raw=llm.generate(service.SETTINGS['model_path'],[{'role':'system','content':source_comparison.COMPARISON_POLICY.replace('/no_think','')+'\n'+SECURITY_SYSTEM_POLICY+'\n/think'},msgs[1]],{**service.SETTINGS['llm'],'max_tokens':2560})
 row={'case_id':c['id'],'final':raw}
 try:
  out=source_comparison.render_comparison(raw,payload);screen_generated_answer(out,payload)
  semantic_guard(service.SETTINGS['model_path'],service.SETTINGS['llm'],{**payload,'draft':str(out),'composition':list(out.parts)},stage='output',generate_fn=llm.generate)
  vis=answer.finalize_grounded_generated_answer(out,rs,msgs,[],language=c['language']);row['answer']=answer.finalize_user_visible_answer(vis,rs,language=c['language'])
 except Exception as e:row['error']=type(e).__name__;row['reason']=getattr(e,'reason',str(e))
 results.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
 (b/'e28-reasoning-diagnostic.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
