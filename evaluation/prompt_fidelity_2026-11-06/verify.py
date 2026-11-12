"""Verify frozen inputs, model/retrieval invariants and one-call capture coverage."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
OLD = REPO / 'evaluation/single_pass_2026-11-04'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def source_files(contract):
    # G was frozen before its identical configuration moved to the opt-in name.
    return {('config/settings.single-pass.json' if key == 'config/settings.json' else key): digest
            for key, digest in contract['source_files'].items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', default='j')
    args = parser.parse_args()
    suffix = args.candidate
    reference = read(OLD / 'expert-g/settings.json')
    prior = read(OLD / 'expert-g/provenance.json')
    new = read(ROOT / ('expert-' + suffix) / 'provenance.json')
    allowed = ({'soprano_qa/single_pass_prompt.py'} if suffix == 'l' else
               {'soprano_qa/single_pass.py', 'soprano_qa/single_pass_prompt.py'})
    assert source_files(prior).keys() == source_files(new).keys()
    assert {k for k in source_files(prior) if source_files(prior)[k] != source_files(new)[k]} == allowed
    for rel, digest in new['source_files'].items():
        assert sha(REPO / rel) == digest, rel
    captures = {}
    for name, prior_name, count, output_name in [
        ('expert-' + suffix, 'expert-g', 109, 'current.jsonl'),
        ('challenge-' + suffix, 'challenge-g', 74, 'current.jsonl'),
        ('timing-' + suffix, 'single-pass-g-clean', 40, 'timings.jsonl'),
        ('timing-g', 'single-pass-g-clean', 40, 'timings.jsonl'),
    ]:
        run = ROOT / name
        contract = read(run / 'provenance.json')
        settings = read(run / 'settings.json')
        assert sha(run / 'settings.json') == contract['settings_sha256']
        assert sha(run / 'cases.json') == sha(OLD / prior_name / 'cases.json') == contract['cases_sha256']
        expected_source = source_files(prior) if name == 'timing-g' else source_files(new)
        assert source_files(contract) == expected_source
        for rel, digest in contract['source_files'].items():
            if rel.startswith('soprano_qa/'):
                assert sha(run / 'snapshot' / rel) == digest
        for key in ('model_path', 'embedding_model_path', 'retrieval', 'llm', 'web_coaching', 'sumijo', 'score_context'):
            assert settings[key] == reference[key], key
        assert sha(Path(settings['corpus_path'])) == prior['corpus_sha256']
        data = rows(run / output_name)
        key = 'id' if name.startswith('timing-') else 'case_id'
        assert len(data) == len({r[key] for r in data}) == count
        failures = [r[key] for r in data if r['exception']]
        assert not failures or name == 'challenge-j' and failures == ['out-of-scope-01-en']
        completion = read(run / 'completion.json')
        assert sha(run / output_name) == completion['results_sha256' if name.startswith('timing-') else 'result_sha256']
        captures[name] = {'count': count, 'sha256': sha(run / output_name), 'technical_errors': failures}
        if name.startswith('timing-'):
            assert all(set(r['llm_seconds_by_stage']) <= {'answer_generation', 'question_translation'} for r in data)
            assert all(r['llm_calls'] == (1 if r['language'] == 'ko' else 2) for r in data)
    assert Path(reference['model_path']).name == 'Qwen3-14B-Q6_K.gguf'
    assert sha(Path(reference['model_path'])) == prior['model_sha256']
    assert sha(ROOT / ('expert-' + suffix) / 'selected-metric-settings.json') == sha(OLD / 'expert-g/selected-metric-settings.json')
    assert sha(ROOT / ('expert-' + suffix) / 'run_expanded_judge.py') == sha(OLD / 'expert-g/run_expanded_judge.py')
    assert (REPO / 'config/settings.json').read_bytes() == subprocess.check_output(
        ['git', '-C', str(REPO), 'show', 'f71d5fe:config/settings.json'])
    all_completed = {}
    for run in sorted(ROOT.iterdir()):
        if not run.is_dir() or not (run / 'completion.json').exists():
            continue
        contract = read(run / 'provenance.json')
        assert sha(run / 'settings.json') == contract['settings_sha256']
        assert sha(run / 'cases.json') == contract['cases_sha256']
        for rel, digest in contract['source_files'].items():
            if rel.startswith('soprano_qa/'):
                assert sha(run / 'snapshot' / rel) == digest
        timing = (run / 'timings.jsonl').exists()
        output = run / ('timings.jsonl' if timing else 'current.jsonl')
        assert sha(output) == read(run / 'completion.json')['results_sha256' if timing else 'result_sha256']
        all_completed[run.name] = {'rows': len(rows(output)), 'sha256': sha(output)}
    preserved = {}
    protected = read(REPO / 'evaluation/helpful_responses_2026-11-04/protected-checkouts-before.json')
    for raw_path, expected in protected.items():
        path = Path(raw_path)
        for rel, digest in expected['files'].items():
            assert sha(path / rel) == digest, str(path / rel)
        assert subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip() == expected['head']
        assert subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain'], text=True) == expected['status']
        preserved[raw_path] = True
    mirrors = read(REPO / 'evaluation/helpful_responses_2026-11-04/demo-mirror-complete.json')
    for root in [REPO / 'sites/answer-review/dist', *map(Path, mirrors['destinations'])]:
        for rel, digest in mirrors['files'].items():
            assert sha(root / rel) == digest
    log = (ROOT / 'unit-tests.log').read_text()
    assert 'Ran 684 tests' in log and '\nOK\n' in log
    unit = read(ROOT / 'unit-validation.json')
    assert unit['log_sha256'] == sha(ROOT / 'unit-tests.log')
    assert unit['source_contract_sha256'] == sha(ROOT / ('expert-' + suffix) / 'provenance.json')
    metric = read(ROOT / ('quality-summary-' + suffix + '.json'))['bertscore']
    assert metric['expert105_f1'] >= 0.90
    assert metric['truncated_texts'] == 0 and metric['overlength'] == 0
    result = {'candidate': suffix, 'only_generation_prompt_sources_changed_vs_g': True,
              'same_cases_model_hybrid_corpus_decoding': True,
              'same_judge_and_metric': True, 'no_online_assessment_or_retry': True,
              'captures': captures, 'all_completed_frozen_captures': all_completed,
              'validated_default_preserved': True, 'protected_checkouts': preserved, 'demo_mirrors_preserved': True,
              'tests': 684, 'unit_log_sha256': sha(ROOT / 'unit-tests.log'),
              'frozen_expert_bertscore': metric}
    (ROOT / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
