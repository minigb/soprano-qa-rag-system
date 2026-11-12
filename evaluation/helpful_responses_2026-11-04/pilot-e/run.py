"""Call the unmodified snapshot's real service; no proxy, repair or output replacement."""
from pathlib import Path
from datetime import datetime, timezone
from contextlib import redirect_stdout
import hashlib, json, os, sys, time
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def main():
    provenance=read(ROOT/'provenance.json');settings=read(ROOT/'settings.json')
    assert sha(ROOT/'settings.json')==provenance['settings_sha256']
    assert sha(ROOT/'cases.json')==provenance['cases_sha256']
    assert sha(Path(__file__))==provenance['run_sha256']
    for rel,digest in provenance['source_files'].items():
        if rel.startswith('soprano_qa/'):assert sha(ROOT/'snapshot'/rel)==digest
    out=ROOT/'current.jsonl';done={r['case_id'] for r in map(json.loads,out.read_text().splitlines())} if out.exists() else set()
    cases=read(ROOT/'cases.json')['cases'];assert done<={c['id'] for c in cases}
    if len(done)==len(cases):print('All 460 already complete.');return
    for name in ['SOPRANO_QA_MODEL_PATH','SOPRANO_QA_RAG_DATASET_ROOT','SOPRANO_QA_DATASET_ROOT','SOPRANO_QA_EMBEDDING_MODEL_PATH']:os.environ.pop(name,None)
    os.environ['SOPRANO_QA_SETTINGS']=str(ROOT/'settings.json');sys.path.insert(0,str(ROOT/'snapshot'))
    with redirect_stdout(sys.stderr):
        from soprano_qa import service as qa, llm
        qa.validate_generation_requirements(qa.SETTINGS['model_path'],qa.SETTINGS['llm'])
        qa._get_index()
        cfg=qa.SETTINGS['llm']
        model=llm.load_llama(model_path=qa.SETTINGS['model_path'],n_ctx=cfg['n_ctx'],n_gpu_layers=cfg['n_gpu_layers'],chat_format=llm._configured_chat_format(cfg))
    print(json.dumps({'ready':True,'remaining':len(cases)-len(done),'model':provenance['model']}),flush=True)
    with out.open('a') as stream:
        for c in cases:
            if c['id'] in done:continue
            started_at=datetime.now(timezone.utc).isoformat();start=time.perf_counter();response=None;error=None
            try:
                with redirect_stdout(sys.stderr):
                    response=qa.ask(piece_id=c['piece_id'],question=c['question'],
                        measure_range=tuple(c['measure_range']) if c.get('measure_range') else None,
                        generate=True,top_k=c.get('top_k',6),language=c['language'])
            except Exception as exc:error={'type':type(exc).__name__,'message':str(exc)}
            row={'case_id':c['id'],'variant':'helpful-14b-pilot-e','started_at':started_at,
                 'status':'exception' if error else 'response','response':response,'exception':error,
                 'elapsed_seconds':time.perf_counter()-start,'latency_scope':'warmed_service_request_including_retrieval',
                 'source_contract_sha256':sha(ROOT/'provenance.json')}
            stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush();done.add(c['id'])
            print(json.dumps({'completed':len(done),'total':len(cases),'case_id':c['id'],'status':row['status'],
                              'mode':(response or {}).get('generation_mode'),'seconds':row['elapsed_seconds']},ensure_ascii=False),flush=True)
    (ROOT/'completion.json').write_text(json.dumps({'cases':len(done),'result_sha256':sha(out),
        'source_contract_sha256':sha(ROOT/'provenance.json'),'corpus_sha256':sha(Path(settings['corpus_path'])),
        'completed_at':datetime.now(timezone.utc).isoformat()},indent=2)+'\n')
if __name__=='__main__':main()
