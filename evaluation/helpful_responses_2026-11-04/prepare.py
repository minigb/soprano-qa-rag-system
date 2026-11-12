"""Freeze a pilot or complete cohort with unchanged 14B weights and retrieval data."""
from pathlib import Path
import argparse, hashlib, json, shutil, subprocess

BASE=Path(__file__).resolve().parent
SOURCE=BASE.parents[1]
OLD=SOURCE/'evaluation/latest_demo_2026-11-03'
OLD_RUNTIME=Path('/home/minhee/.codex/worktrees/rag-latest-answer-demo/soprano-qa-rag-system/evaluation/latest_demo_2026-11-03/runtime')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
    parser=argparse.ArgumentParser();parser.add_argument('name');parser.add_argument('--pilot',action='store_true');parser.add_argument('--security',action='store_true');parser.add_argument('--case-id',action='append');args=parser.parse_args()
    root=BASE/args.name
    if root.exists():raise RuntimeError('Preserve existing run; choose a new candidate directory')
    root.mkdir()
    for n in ['run.py','watch_judge.py','prepare_expanded_judge_packets.py','run_joint_expanded_judge.py',
              'run_expanded_judge.py','run_reused_expanded_judge.py','score_expanded_bertscore_bounded.py',
              'score.py','selected-metric-settings.json','metric-length-amendment.json']:
        shutil.copyfile(OLD/n,root/n)
    watcher=(root/'watch_judge.py').read_text()
    watcher=watcher.replace("lines=(ROOT/'current.jsonl').read_text().splitlines(keepends=True)", "lines=((ROOT/'current.jsonl').read_text() if (ROOT/'current.jsonl').exists() else '').splitlines(keepends=True)")
    watcher=watcher.replace("'--workers','3','--model'", "'--workers',str(min(3,len(picked))),'--model'")
    reused=(root/'run_reused_expanded_judge.py').read_text()
    reused=reused.replace("'--workers', str(args.workers)], check=True)", "'--workers', str(min(args.workers, len(pending)))], check=True)")
    (root/'run_reused_expanded_judge.py').write_text(reused)
    if not args.pilot and not args.security:
        # Reuse only identical full packets under the unchanged rubric/model.
        # The reuse runner checks every input/assessment contract and retains
        # the first judgment regardless of its score.
        previous=[]
        for batch in sorted((OLD/'judge-batches').iterdir()):
            for directory in [batch,batch/'fresh']:
                if (directory/'aggregation.json').exists():previous.append(str(directory))
        watcher=watcher.replace('previous=[]', 'previous='+repr(previous))
    (root/'watch_judge.py').write_text(watcher)
    runner=((BASE/'run_security.py').read_text() if args.security else (root/'run.py').read_text()).replace('current-wip-14b-20261103','helpful-14b-'+args.name)
    (root/'run.py').write_text(runner)
    shutil.copytree(SOURCE/'soprano_qa',root/'snapshot/soprano_qa',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(OLD_RUNTIME,root/'runtime')
    settings=json.loads((OLD/'settings.json').read_text())
    settings['llm']=json.loads((SOURCE/'config/settings.json').read_text())['llm']
    for key in ['corpus_path','stats_path','embedding_cache_path','feature_overrides_path']:
        old=Path(settings[key]);settings[key]=str(root/'runtime'/old.relative_to(OLD_RUNTIME))
    assert Path(settings['model_path']).name=='Qwen3-14B-Q6_K.gguf'
    assert settings['llm']['source_preservation']=='semantic'
    save(root/'settings.json',settings)
    all_cases=json.loads((OLD/'cases.json').read_text())['cases']
    selected=all_cases
    if args.pilot:
        ids={r['case_id'] for r in map(json.loads,(OLD/'current.jsonl').read_text().splitlines()) if r['response']['generation_mode']=='unavailable'}
        selected=[c for c in all_cases if c['id'] in ids];assert len(selected)==5
    if args.security:
        inventory=json.loads(Path('/home/minhee/.codex/worktrees/rag-component-ablation/soprano-qa-rag-system/ablation/cases.json').read_text())['cases']
        selected=[c for c in inventory if c['kind']!='benign']+[c for c in inventory if c['kind']=='benign'][:12]
        assert len(selected)==74
    if args.case_id:
        selected=[c for c in selected if c['id'] in args.case_id]
        assert {c['id'] for c in selected}==set(args.case_id)
    save(root/'cases.json',{'cases':selected,'scope':'explicit regression subset: '+','.join(args.case_id) if args.case_id else '50 attacks, 12 domain requests, 12 normal controls' if args.security else 'previous five refusals' if args.pilot else 'all existing 460 requests'})
    paths=list((SOURCE/'soprano_qa').glob('*.py'))+[SOURCE/'config/settings.json',SOURCE/'config/feature_overrides.json']
    old=json.loads((OLD/'provenance.json').read_text())
    save(root/'provenance.json',{'source_root':str(SOURCE),'source_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip(),
        'source_branch':'codex/helpful-rag-responses','source_files':{str(p.relative_to(SOURCE)):sha(p) for p in paths},
        'settings_sha256':sha(root/'settings.json'),'cases_sha256':sha(root/'cases.json'),'run_sha256':sha(root/'run.py'),
        'model':old['model'],'model_sha256':old['model_sha256'],'model_path':settings['model_path'],
        'policy':'Quality failures may return safety-reviewed prose with a warning. Same model, retrieval corpus, and questions as prior capture.',
        'baseline_results_sha256':sha(OLD/'current.jsonl')})
    (root/'.gitignore').write_text('runtime/\n*.log\n*.private-traces.jsonl\n*.private-failure.json\n__pycache__/\n')
    print(json.dumps({'run':str(root),'cases':len(selected)}))
if __name__=='__main__':main()
