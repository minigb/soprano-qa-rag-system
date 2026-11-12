"""Historical E36 prompt diagnostic; no security or final-quality claim.
All six visible outputs remain recorded, including incorrect concept descriptions.
Controlled prompt simplification for pitch concepts, all six outputs retained."""
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from soprano_qa import llm, service
from soprano_qa.security import SECURITY_SYSTEM_POLICY
from soprano_qa.structured_answer import GENERAL_MUSIC_FORMAT_POLICY
b=Path('evaluation/ku_background_2026-10-31'); p=b/'e36-concise-pitch-diagnostic.json'
if p.exists(): raise ValueError('Do not overwrite an experiment')
cs=[c for c in json.loads((b/'e26_probe_cases.json').read_text())['cases'] if c['id'] in ('background-transpose-en','followup-transpose-ko','followup-transpose-paraphrase')]
brief_facts='''Music-theory distinctions: chromatic transposition shifts every pitch by one fixed interval; the tonal centre shifts with the notes. Selective octave displacement retains the selected notes' pitch classes, with possible changes to melodic intervals and voicing. Enharmonic respelling changes notation for an equivalent sounding pitch in equal temperament. Answer only the requested distinction.'''
out=[]
for version,extra in [('concise',''),('concise_facts',brief_facts)]:
 for c in cs:
  policy=(GENERAL_MUSIC_FORMAT_POLICY+'\nExplain the requested music concept accurately in TWO short sentences. '+
          'Answer in '+c['language']+'. No examples, headings, lists or extra judgments.\n'+extra+'\n'+SECURITY_SYSTEM_POLICY+'\n/think')
  start=time.perf_counter();result=llm.generate(service.SETTINGS['model_path'],[{'role':'system','content':policy},{'role':'user','content':json.dumps({'question':c['question'],'answer_language':c['language']},ensure_ascii=False)}],{**service.SETTINGS['llm'],'max_tokens':1536})
  row={'version':version,'case_id':c['id'],'policy':policy,'answer':result,'seconds':time.perf_counter()-start}
  out.append(row);p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(row,ensure_ascii=False),flush=True)
