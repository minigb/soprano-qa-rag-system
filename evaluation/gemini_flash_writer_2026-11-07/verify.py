"""Verify preserved baselines, real provider identity and complete capture counts."""
from pathlib import Path
import ast
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(l) for l in path.read_text().splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    contract = read(ROOT / 'contract.json')
    assert sha(ROOT / 'writer-settings.json') == contract['writer_settings_sha256']
    assert read(REPO / 'config/settings.gemini.json')['llm'] == read(ROOT / 'writer-settings.json')
    assert read(ROOT / 'expert/selected-metric-settings.json') == read(
        REPO / 'evaluation/prompt_fidelity_2026-11-06/expert-j/selected-metric-settings.json')
    for rel, digest in contract['source_files'].items():
        assert sha(ROOT / 'snapshot' / rel) == digest
        assert sha(REPO / rel) == digest, 'Live implementation differs: ' + rel
    preserved = {}
    for rel in ['config/settings.json', 'config/settings.single-pass.json',
                'soprano_qa/single_pass_prompt.py', 'soprano_qa/dense.py',
                'soprano_qa/retrieval.py', 'soprano_qa/translation.py']:
        before = subprocess.check_output(['git', 'show', '38dd4e7:' + rel], cwd=REPO)
        digest = hashlib.sha256(before).hexdigest()
        assert sha(REPO / rel) == digest
        preserved[rel] = digest
    # The only single_pass module change routes the writer. Its prompt builder
    # and service-boundary policy are the same as candidate J.
    old = ast.parse(subprocess.check_output(['git', 'show', '38dd4e7:soprano_qa/single_pass.py'], cwd=REPO, text=True))
    now = ast.parse((REPO / 'soprano_qa/single_pass.py').read_text())
    for name in ['build_messages', 'enabled']:
        assert ast.dump(next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == name)) == ast.dump(next(n for n in now.body if isinstance(n, ast.FunctionDef) and n.name == name))
    checks = {}
    for name, count in [('expert', 109), ('challenge', 74)]:
        run = ROOT / name; result = rows(run / 'current.jsonl'); cases = read(run / 'cases.json')['cases']
        assert len(result) == len(cases) == count
        assert len({r['case_id'] for r in result}) == count
        assert {r['case_id'] for r in result} == {c['id'] for c in cases}
        assert sha(run / 'current.jsonl') == read(run / 'completion.json')['result_sha256']
        assert sha(run / 'writer-inputs.json') == contract['cohorts'][name]['inputs_sha256']
        inputs = {r['case_id']: r for r in read(run / 'writer-inputs.json')}
        for row in result:
            if row['status'] == 'excluded': continue
            packet = inputs[row['case_id']]
            assert hashlib.sha256(json.dumps(packet['messages'], ensure_ascii=False,
                                             sort_keys=True).encode()).hexdigest() == packet['messages_sha256']
            assert row['messages_sha256'] == packet['messages_sha256']
            if row['status'] != 'response': continue
            response = row['response']; md = response['generation_metadata']
            assert md['provider'] == 'vertex_gemini'
            assert md['requested_model'] == md['model_version'] == 'gemini-3.8-flash'
            assert md['finish_reason'] == 'STOP' and response['answer'].strip()
            assert response['evidence'] == packet['baseline_response']['evidence']
            assert response['score_context'] == packet['baseline_response']['score_context']
        assert not list((run / 'request-journal').glob('*.pending'))
        checks[name] = {'count': count, 'successful': sum(r['status'] == 'response' for r in result),
                        'technical_errors': sum(r['status'] == 'exception' for r in result),
                        'excluded': sum(r['status'] == 'excluded' for r in result),
                        'result_sha256': sha(run / 'current.jsonl')}
    assert read(ROOT / 'tests.json')['tests'] == 693
    assert read(ROOT / 'service-smoke.json')['local_writer_loaded'] == 0
    assert read(ROOT / 'service-smoke-en.json')['local_calls'] == ['local_query_translation']
    assert read(ROOT / 'summary.json')['bertscore']['truncated_texts'] == 0
    assert len(rows(ROOT / 'timing-sequential.jsonl')) == 10
    for name in ['main', 'codex/pre-injection-baseline']:
        expected = {'main': '29fe3e2', 'codex/pre-injection-baseline': 'f307470'}[name]
        head = subprocess.check_output(['git', 'rev-parse', '--short=7', name], cwd=REPO, text=True).strip()
        assert head == expected
    subprocess.run(['git', 'diff', '--exit-code', '38dd4e7', '--', 'sites/answer-review', 'data/corpus.json'], cwd=REPO, check=True, stdout=subprocess.DEVNULL)
    result = {'verified': True, 'captures': checks, 'preserved_files': preserved,
              'snapshot_matches_live_code': True, 'provider_model_verified_in_responses': 'gemini-3.8-flash',
              'same_prompt_builder_and_writing_policy_as_j': True, 'tests': 693,
              'main_original_corpus_and_460demo_preserved': True,
              'api_failures_retained_separately': True,
              'metric_hyperparameters_unchanged': read(ROOT / 'summary.json')['bertscore']['settings']}
    (ROOT / 'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
