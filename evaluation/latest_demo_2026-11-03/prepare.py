"""Freeze the actual current 14B service for the existing 460-question demo."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, shutil, subprocess

ROOT=Path(__file__).resolve().parent
SOURCE=Path('/home/minhee/.codex/worktrees/rag-background-knowledge/soprano-qa-rag-system')
def sha(p):
    d=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):d.update(b)
    return d.hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
    if (ROOT/'provenance.json').exists():raise RuntimeError('Snapshot exists; preserve and resume it.')
    snapshot=ROOT/'snapshot';snapshot.mkdir()
    shutil.copytree(SOURCE/'soprano_qa',snapshot/'soprano_qa',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    runtime=ROOT/'runtime';runtime.mkdir()
    shutil.copytree(SOURCE/'data',runtime/'data')
    shutil.copyfile(SOURCE/'config/feature_overrides.json',runtime/'feature_overrides.json')
    for name in ['soprano-corpus-qwen3-embedding-cache.json','soprano-corpus-qwen3-embedding-cache-web-coaching.json']:
        shutil.copyfile(SOURCE/'models'/name,runtime/name)
    settings=json.loads((SOURCE/'config/settings.json').read_text())
    for name in ['dataset_root','model_path','embedding_model_path']:
        settings[name]=str((SOURCE/'config'/settings[name]).resolve())
    for name in ['corpus_path','stats_path']:settings[name]=str(runtime/'data'/Path(settings[name]).name)
    settings['embedding_cache_path']=str(runtime/'soprano-corpus-qwen3-embedding-cache.json')
    settings['feature_overrides_path']=str(runtime/'feature_overrides.json')
    assert settings['model_filename']=='Qwen3-14B-Q6_K.gguf'
    assert settings['llm']['source_preservation']=='semantic'
    save(ROOT/'settings.json',settings)
    shutil.copyfile(SOURCE/'evaluation/expanded_460_2026-11-02/cases.json',ROOT/'cases.json')
    assert len(json.loads((ROOT/'cases.json').read_text())['cases'])==460
    paths=list((SOURCE/'soprano_qa').glob('*.py'))+[SOURCE/'config/settings.json',SOURCE/'config/feature_overrides.json']
    model_hash=sha(Path(settings['model_path']))
    assert model_hash=='ec1f1d1421d7636a23e17645cac6fda39ee4b5f1f11e42eb09238089705ef699'
    save(ROOT/'provenance.json',{'created_at':datetime.now(timezone.utc).isoformat(),'source_root':str(SOURCE),
        'source_branch':subprocess.check_output(['git','branch','--show-current'],cwd=SOURCE,text=True).strip(),
        'source_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip(),
        'source_state':'Current uncommitted 14B semantic-generation WIP; not a new production adoption.',
        'source_files':{str(p.relative_to(SOURCE)):sha(p) for p in paths},
        'model':'Qwen3-14B-Q6_K','model_sha256':model_hash,'settings_sha256':sha(ROOT/'settings.json'),
        'cases_sha256':sha(ROOT/'cases.json'),'run_sha256':sha(ROOT/'run.py'),
        'comparison':'Retain historical original answers on the left; run the complete current service on all same 460 questions on the right.',
        'evaluation_policy':'No historical Claude verdict or BERTScore is transferred to changed answers. New outputs are initially not evaluated.',
        'no_usage_stop':True,'no_notifier':True})
    print('Frozen current 14B source, settings, caches and 460 questions.',flush=True)
if __name__=='__main__':main()
