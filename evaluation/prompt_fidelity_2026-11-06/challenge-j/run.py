"""Exercise the actual service boundary with the frozen bilingual attack corpus."""
from pathlib import Path
from unittest.mock import patch
import copy, hashlib, json, os, sys, time
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    contract=read(ROOT/'provenance.json')
    assert sha(ROOT/'settings.json')==contract['settings_sha256']
    assert sha(ROOT/'cases.json')==contract['cases_sha256']
    assert sha(Path(__file__))==contract['run_sha256']
    for rel,digest in contract['source_files'].items():
        if rel.startswith('soprano_qa/'):assert sha(ROOT/'snapshot'/rel)==digest
    for key in ['SOPRANO_QA_MODEL_PATH','SOPRANO_QA_RAG_DATASET_ROOT','SOPRANO_QA_DATASET_ROOT','SOPRANO_QA_EMBEDDING_MODEL_PATH']:os.environ.pop(key,None)
    os.environ['SOPRANO_QA_SETTINGS']=str(ROOT/'settings.json');sys.path.insert(0,str(ROOT/'snapshot'))
    from soprano_qa import service as qa,llm
    from soprano_qa.retrieval import SearchResult
    cases=read(ROOT/'cases.json')['cases'];out=ROOT/'current.jsonl'
    done={r['case_id'] for r in map(json.loads,out.read_text().splitlines())} if out.exists() else set()
    qa._get_index();cfg=qa.SETTINGS['llm']
    llm.load_llama(model_path=qa.SETTINGS['model_path'],n_ctx=cfg['n_ctx'],n_gpu_layers=cfg['n_gpu_layers'],chat_format=llm._configured_chat_format(cfg))
    with out.open('a') as stream:
        for c in cases:
            if c['id'] in done:continue
            visited=[]
            def frozen_retrieve(index,**kw):
                visited.append(True)
                results=[SearchResult(copy.deepcopy(p['record']),p.get('score',p.get('relevance',1)),0,0,1,
                    p.get('scope',p.get('scope_match','general_evidence')),alias_score=p.get('alias_score',0)) for p in c['fixed_evidence']]
                return results,kw.get('message_builder',qa.build_messages)(kw['piece'],kw['measures'],kw['query'],results,score_context=kw['score_context'])
            started=time.perf_counter();response=None;error=None
            try:
                with patch.object(qa,'retrieve_generation_evidence',side_effect=frozen_retrieve), \
                     patch.object(qa,'selected_measure_lyrics',return_value=copy.deepcopy(c.get('score_context'))):
                    response=qa.ask(piece_id=c['piece_id'],question=c['question'],measure_range=tuple(c['measure_range']) if c.get('measure_range') else None,
                                    generate=True,language=c['language'])
            except Exception as exc:error={'type':type(exc).__name__,'message':str(exc)}
            row={'case_id':c['id'],'variant':'single-pass-challenge-14b','status':'exception' if error else 'response',
                 'response':response,'exception':error,'elapsed_seconds':time.perf_counter()-started,
                 'frozen_retrieval_visited':bool(visited),'source_contract_sha256':sha(ROOT/'provenance.json')}
            stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush();done.add(c['id'])
            print(json.dumps({'completed':len(done),'total':len(cases),'id':c['id'],'mode':(response or {}).get('generation_mode')},ensure_ascii=False),flush=True)
    (ROOT/'completion.json').write_text(json.dumps({'cases':len(cases),'result_sha256':sha(out),'source_contract_sha256':sha(ROOT/'provenance.json')},indent=2)+'\n')
if __name__=='__main__':main()
