"""Freeze an expert/previous-refusal regression run or the bilingual challenge set."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PREVIOUS = REPO / 'evaluation/helpful_responses_2026-11-04'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('--challenge', action='store_true')
    args = parser.parse_args()
    run = ROOT / args.name
    run.mkdir(exist_ok=False)
    old = PREVIOUS / 'full-final'
    for name in ('prepare_expanded_judge_packets.py', 'run_expanded_judge.py',
                 'run_joint_expanded_judge.py', 'run_reused_expanded_judge.py',
                 'score_expanded_bertscore_bounded.py', 'selected-metric-settings.json',
                 'metric-length-amendment.json'):
        shutil.copyfile(old / name, run / name)
    watcher = (old / 'watch_judge.py').read_text()
    # No score-dependent repeat: first valid assessment, identical rubric/model.
    watcher = '\n'.join('            previous=[]' if line.startswith('            previous=') else line
                        for line in watcher.split('\n'))
    (run / 'watch_judge.py').write_text(watcher)
    score = (old / 'score.py').read_text()
    score = score.replace('==460', "==len(read(ROOT/'cases.json')['cases'])")
    score = score.replace("'cases':460", "'cases':len(raw)")
    score = score.replace("bool(judgment and judgment['refusal'])",
                          "bool(judgment and judgment['refusal']) or (row.get('response') or {}).get('generation_mode') in {'response_error', 'domain_guidance', 'unavailable'}")
    (run / 'score.py').write_text(score)
    if args.challenge:
        cases = read(PREVIOUS / 'security-hardened/cases.json')
        runner = (PREVIOUS / 'run_security.py').read_text()
        # Inject only the retrieval result, using the actual configured prompt builder.
        runner = runner.replace('return results,qa.build_messages(', "return results,kw.get('message_builder',qa.build_messages)(")
        runner = runner.replace('helpful-security-14b', 'single-pass-challenge-14b')
    else:
        originals = read(old / 'cases.json')['cases']
        refused = {c['id'] for c in read(ROOT / 'profile-cases.json')['cases']
                   if c['timing_group'] == 'previous_refusal'}
        chosen = [c for c in originals if c['family'] == 'original' or c['id'] in refused]
        assert len(chosen) == 109
        cases = {'cases': chosen, 'scope': 'All existing 105 expert original questions plus four additional prior-refusal paraphrases; fifth prior refusal is already an expert original. This is an existing regression set, not a new holdout.'}
        runner = (old / 'run.py').read_text().replace('helpful-14b-full-final', 'single-pass-expert-regressions-14b')
        runner = runner.replace('All 460 already complete.', 'All cases already complete.')
    (run / 'run.py').write_text(runner)
    save(run / 'cases.json', cases)
    shutil.copytree(REPO / 'soprano_qa', run / 'snapshot/soprano_qa', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copytree(old / 'runtime', run / 'runtime')
    settings = read(old / 'settings.json')
    settings['llm'] = read(REPO / 'config/settings.single-pass.json')['llm']
    for key in ('corpus_path', 'stats_path', 'embedding_cache_path', 'feature_overrides_path'):
        settings[key] = str(run / 'runtime' / Path(settings[key]).relative_to(old / 'runtime'))
    save(run / 'settings.json', settings)
    files = list((REPO / 'soprano_qa').glob('*.py')) + [REPO / 'config/settings.single-pass.json']
    save(run / 'provenance.json', {
        'source_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        'source_branch': 'codex/helpful-rag-responses',
        'source_files': {str(p.relative_to(REPO)): sha(p) for p in files},
        'settings_sha256': sha(run / 'settings.json'), 'cases_sha256': sha(run / 'cases.json'),
        'run_sha256': sha(run / 'run.py'), 'model': 'Qwen3-14B-Q6_K',
        'model_sha256': read(old / 'provenance.json')['model_sha256'],
        'corpus_sha256': sha(Path(settings['corpus_path'])),
        'policy': 'Real single-pass writer, no online answer assessment or retries. Frozen reference, model and metric settings. Claude assessment is an offline development check only.'})
    (run / '.gitignore').write_text('runtime/\n*.log\n*.private-traces.jsonl\n*.private-failure.json\n__pycache__/\n')
    print(json.dumps({'run': str(run), 'cases': len(cases['cases'])}))


if __name__ == '__main__':
    main()
