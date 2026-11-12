"""Verify the evaluated implementation and preserved checkouts after demo export."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True)


def main():
    assert git(REPO, 'branch', '--show-current').strip() == 'codex/helpful-rag-responses'
    protected = read(HERE / 'protected-checkouts-before.json')
    for root, before in protected.items():
        assert git(root, 'rev-parse', 'HEAD').strip() == before['head'], root
        assert git(root, 'status', '--porcelain') == before['status'], root
        for rel, digest in before['files'].items():
            assert sha(Path(root) / rel) == digest, (root, rel)

    # This commit captured the previous WIP backend before any implementation
    # changes; refreshing its static demo must not switch its service code.
    background = Path('/home/minhee/.codex/worktrees/rag-background-knowledge/soprano-qa-rag-system')
    prior_files = git(REPO, 'ls-tree', '-r', '--name-only', '13f8b5b', 'soprano_qa', 'config').splitlines()
    checked_prior = []
    for rel in prior_files:
        if not (rel.endswith('.py') or rel in {'config/settings.json', 'config/feature_overrides.json'}):
            continue
        frozen = subprocess.check_output(['git', '-C', str(REPO), 'show', '13f8b5b:' + rel])
        assert (background / rel).read_bytes() == frozen, rel
        checked_prior.append(rel)

    run = HERE / 'full-final'
    provenance = read(run / 'provenance.json')
    unit = read(HERE / 'unit-validation-final.json')
    assert unit['tests'] == 674 and unit['result'] == 'passed'
    assert sha(HERE / 'unit-tests-final.txt') == unit['log_sha256']
    assert unit['source_files'] == provenance['source_files']
    for rel, digest in provenance['source_files'].items():
        assert sha(REPO / rel) == digest, rel
    assert provenance['model'] == 'Qwen3-14B-Q6_K'
    assert sha(Path(provenance['model_path'])) == provenance['model_sha256']
    settings = read(run / 'settings.json')
    assert settings['llm']['source_preservation'] == 'semantic'
    corpus_sha = sha(Path(settings['corpus_path']))
    assert corpus_sha == read(REPO / 'evaluation/latest_demo_2026-11-03/completion.json')['corpus_sha256']
    assert corpus_sha == sha(REPO / 'data/corpus.json')
    assert read(run / 'completion.json')['result_sha256'] == sha(run / 'current.jsonl')
    assert read(run / 'assessment-complete.json')['results_sha256'] == sha(run / 'current.jsonl')
    rows = [json.loads(line) for line in (run / 'current.jsonl').read_text().splitlines()]
    assert len(rows) == len({row['case_id'] for row in rows}) == 460
    assert all(row['status'] == 'response' and row['response']['answer'].strip() for row in rows)
    assert not any('참고 자료의 원문입니다.' in row['response']['answer']
                   or row['response']['answer'].startswith('Translated reference text:')
                   for row in rows)
    assert sha(run / 'selected-metric-settings.json') == sha(REPO / 'evaluation/latest_demo_2026-11-03/selected-metric-settings.json')

    comparison = read(HERE / 'comparison.json')
    assert comparison['current']['count'] == 460
    assert comparison['current']['expert_105_bertscore'] >= 0.9
    assert comparison['current']['unanswered'] == 0
    assert len(comparison['previous_five_refusals']) == 5
    assert all(not row['current']['judge']['refusal'] for row in comparison['previous_five_refusals'])

    mirror = read(HERE / 'demo-mirror-complete.json')
    dist = REPO / 'sites/answer-review/dist'
    for root in mirror['destinations']:
        for rel, digest in mirror['files'].items():
            assert sha(Path(root) / rel) == digest == sha(dist / rel), (root, rel)
    before = read(HERE / 'demo-mirror-before.json')
    for files in before.values():
        for rel in ('data.json', 'data-latest-14b-20261103.json'):
            assert files[rel] == sha(dist / rel), rel

    output = {
        'verified_at': datetime.now(timezone.utc).isoformat(),
        'branch': 'codex/helpful-rag-responses',
        'implementation_commit': provenance['source_head'],
        'protected_checkouts': list(protected),
        'previous_backend_preserved_files': checked_prior,
        'implementation_matches_unit_tests_and_evaluation': True,
        'unit_tests_passed': unit['tests'],
        'model': provenance['model'], 'model_sha256': provenance['model_sha256'],
        'corpus_sha256': corpus_sha, 'corpus_matches_november_3': True,
        'metric_settings_unchanged': True,
        'semantic_generation_enabled': True, 'source_substitution_headers': 0,
        'expert_105_bertscore': comparison['current']['expert_105_bertscore'],
        'useful_responses': 460, 'previous_five_refusals_answered': 5,
        'static_mirrors_match': mirror['destinations'],
        'initial_and_november_3_demo_data_unchanged': True,
        'artifact_hashes': {name: sha(HERE / name) for name in (
            'full-final/current.jsonl', 'full-final/assessment-complete.json',
            'security-hardened/current.jsonl', 'comparison.json', 'manual-review.json',
            'report.md', 'demo-mirror-complete.json', 'browser-validation.json')},
    }
    (HERE / 'verification.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
