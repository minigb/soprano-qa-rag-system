"""Check that the evaluated writer is the checkout default and originals survived."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PREVIOUS = REPO / 'evaluation/helpful_responses_2026-11-04'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def git(path, *args):
    return subprocess.check_output(['git', '-C', str(path), *args], text=True)


def main():
    contract = read(ROOT / 'expert-g/provenance.json')
    for rel, expected in contract['source_files'].items():
        assert sha(REPO / ('config/settings.single-pass.json' if rel == 'config/settings.json' else rel)) == expected, rel
    for name in ('single-pass-g-clean', 'challenge-g'):
        assert read(ROOT / name / 'provenance.json')['source_files'] == contract['source_files']
    settings = read(ROOT / 'expert-g/settings.json')
    old_settings = read(PREVIOUS / 'full-final/settings.json')
    for field in ('retrieval', 'web_coaching', 'sumijo', 'score_context', 'model_path', 'embedding_model_path'):
        assert settings[field] == old_settings[field], field
    model_sha = sha(Path(settings['model_path']))
    assert model_sha == contract['model_sha256']
    assert settings['llm']['pipeline'] == 'single_pass'
    assert Path(settings['model_path']).name == 'Qwen3-14B-Q6_K.gguf'
    assert sha(Path(settings['corpus_path'])) == sha(Path(old_settings['corpus_path']))
    assert (REPO / 'config/settings.json').read_bytes() == subprocess.check_output(['git', '-C', str(REPO), 'show', 'f71d5fe:config/settings.json'])
    protected = {}
    for path, expected in read(PREVIOUS / 'protected-checkouts-before.json').items():
        path = Path(path)
        assert git(path, 'rev-parse', 'HEAD').strip() == expected['head']
        assert git(path, 'status', '--porcelain') == expected['status']
        for rel, digest in expected['files'].items():
            assert sha(path / rel) == digest, (path, rel)
        protected[str(path)] = {'head': expected['head'], 'files_checked': len(expected['files']), 'unchanged': True}
    background = Path('/home/minhee/.codex/worktrees/rag-background-knowledge/soprano-qa-rag-system')
    backend_files = read(PREVIOUS / 'verification.json')['previous_backend_preserved_files']
    for rel in backend_files:
        assert (background / rel).read_bytes() == subprocess.check_output(['git', '-C', str(REPO), 'show', '13f8b5b:' + rel])
    mirrors = read(PREVIOUS / 'demo-mirror-complete.json')
    for root in [REPO / 'sites/answer-review/dist', *map(Path, mirrors['destinations'])]:
        for rel, digest in mirrors['files'].items():
            assert sha(root / rel) == digest, (root, rel)
    unit = read(ROOT / 'unit-validation.json')
    assert unit['tests'] == 684 and unit['result'] == 'passed'
    assert sha(ROOT / unit['log']) == unit['log_sha256']
    assert unit['source_contract_sha256'] == sha(ROOT / 'expert-g/provenance.json')
    assert sha(ROOT / 'expert-g/selected-metric-settings.json') == sha(PREVIOUS / 'full-final/selected-metric-settings.json')
    for name in ('expert-g', 'challenge-g'):
        run = ROOT / name
        assert sha(run / 'current.jsonl') == read(run / 'completion.json')['result_sha256']
    profile = read(ROOT / 'profile-summary.json')['runs']['single-pass-g-clean']
    assert profile['one_writer_no_model_review_verified']
    result = {'branch': git(REPO, 'branch', '--show-current').strip(),
              'candidate_source_matches_all_final_captures': True, 'validated_default_preserved': True, 'model_sha256': model_sha,
              'retrieval_settings_and_corpus_unchanged': True, 'protected': protected,
              'one_writer_no_online_model_assessment': True,
              'previous_backend_files_unchanged': len(backend_files),
              'previous_demo_and_all_mirrors_unchanged': True,
              'unit_tests_passed': unit['tests'], 'metric_settings_unchanged': True,
              'demo': 'Preserved previous complete 460-case dataset. This task did not overwrite it with a partial or unjudged new dataset.',
              'quality_summary_sha256': sha(ROOT / 'quality-summary.json'),
              'profile_summary_sha256': sha(ROOT / 'profile-summary.json')}
    (ROOT / 'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
