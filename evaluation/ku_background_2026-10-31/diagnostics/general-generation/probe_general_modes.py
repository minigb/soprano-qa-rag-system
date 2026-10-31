import hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from soprano_qa.llm import load_llama,strip_thinking
from soprano_qa.security import SECURITY_SYSTEM_POLICY,serialize_untrusted
from soprano_qa.structured_answer import GENERAL_MUSIC_SYSTEM_PROMPT,GENERAL_MUSIC_FORMAT_POLICY,answer_schema,render_structured_answer
root=Path('evaluation/ku_background_2026-10-31/runtime')
cases=json.loads((root/'general-diagnostic-cases.json').read_text())
model=load_llama(model_path='models/Qwen3-14B-Q6_K.gguf',n_ctx=8192,n_gpu_layers=-1)
policy=GENERAL_MUSIC_FORMAT_POLICY+'\n'+GENERAL_MUSIC_SYSTEM_PROMPT+'\n'+SECURITY_SYSTEM_POLICY
for case in cases:
 payload={'question':case['question'],'answer_language':case['language'],'piece':case['piece_id'],'measure_range':'(none selected)','Retrieved evidence':[],'Score context':None}
 for mode in ('greedy_constrained','sampled_constrained','thinking_parsed'):
  system=policy if mode!='thinking_parsed' else policy.replace('/no_think','')+'\n/think'
  opts={'temperature':0.,'top_p':.8,'max_tokens':768,'response_format':{'type':'json_object','schema':answer_schema(payload)}}
  if mode=='sampled_constrained':opts.update(temperature=.7,top_k=20,min_p=0,seed=20261031)
  if mode=='thinking_parsed':opts={'temperature':.6,'top_p':.95,'top_k':20,'min_p':0,'max_tokens':2048,'seed':20261031}
  started=time.perf_counter()
  response=model.create_chat_completion(messages=[{'role':'system','content':system},{'role':'user','content':serialize_untrusted(payload)}],**opts)
  raw=response['choices'][0]['message']['content'];final=strip_thinking(raw)
  error=None
  try:visible=str(render_structured_answer(final,payload))
  except Exception as exc:visible=None;error=type(exc).__name__
  print(json.dumps({'case_id':case['id'],'mode':mode,'policy_sha256':hashlib.sha256(system.encode()).hexdigest(),'options':opts,'seconds':time.perf_counter()-started,'usage':response.get('usage'),'finish_reason':response['choices'][0]['finish_reason'],'final':final,'visible':visible,'parse_error':error,'raw_sha256':hashlib.sha256(raw.encode()).hexdigest()},ensure_ascii=False),flush=True)
