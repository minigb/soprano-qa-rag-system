"""Historical E36 diagnostic: requires source fbae849 / runtime/probe-e36.
The textbook prompt tested here was rejected and removed from production.
Compare lean general-theory prompts; diagnostic only, all decisions retained."""
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from soprano_qa import llm, service
from soprano_qa.music_terms import PITCH_CHANGE_TERMINOLOGY

b=Path('evaluation/ku_background_2026-10-31')
p=b/'e36-pitch-prompt-diagnostic.json'
if p.exists(): raise ValueError('Do not overwrite an experiment')
cases=json.loads((b/'e26_probe_cases.json').read_text())['cases']
cs=[c for c in cases if c['id'] in ('background-transpose-en','followup-transpose-ko','followup-transpose-paraphrase')]
out=[]
for version,extra in [('lean',''),('lean_with_definitions',PITCH_CHANGE_TERMINOLOGY)]:
 for c in cs:
  policy=('Explain the requested music theory accurately and briefly. Answer in '+c['language']+'. '
          'Distinguish what changes and what remains the same. No unrequested examples.\n'+extra+'\n/think')
  start=time.perf_counter()
  result=llm.generate(service.SETTINGS['model_path'],[{'role':'system','content':policy},{'role':'user','content':c['question']}],
                      {**service.SETTINGS['llm'],'max_tokens':1536})
  row={'version':version,'case_id':c['id'],'policy':policy,'answer':result,'seconds':time.perf_counter()-start}
  out.append(row);p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
  print(json.dumps(row,ensure_ascii=False),flush=True)
