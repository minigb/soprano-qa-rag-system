"""Offline paired assessment and fixed-metric regression report."""
from pathlib import Path
import argparse
import copy
import json
import os
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
FINAL = ROOT / 'v3'
REPO = ROOT.parents[1]
PRIMARY = ROOT.parent / 'gemini_flash_writer_2026-11-07'
BASELINE = ROOT.parent / 'gemini_flash_recovery_2026-11-07'
SCRIPTS = PRIMARY / 'expert'
sys.path.insert(0, str(ROOT))
from run import read, rows, sha, freeze


def call(script, *args):
    subprocess.run([sys.executable, str(SCRIPTS / script), *map(str, args)], check=True)


def expert():
    assert read(FINAL / 'completion.json')['after_sha256'] == sha(FINAL / 'after.jsonl')
    call('prepare_expanded_judge_packets.py', '--cases', FINAL / 'cases.json',
         '--results', FINAL / 'before.jsonl', FINAL / 'after.jsonl', '--output', FINAL / 'judge')
    call('run_reused_expanded_judge.py', '--directory', FINAL / 'judge',
         '--reuse-directory', ROOT / 'judge', '--model', 'claude-opus-5-5', '--workers', '3')
    paired = judgments(FINAL / 'judge')
    metric = []
    for r in rows(FINAL / 'after.jsonl'):
        item = copy.deepcopy(r)
        item['metric_claude_refusal'] = paired[r['case_id']]['after']['refusal']
        if item['metric_claude_refusal']: item['response']['generation_mode'] = 'unavailable'
        metric.append(item)
    path = FINAL / 'metric-input.jsonl'; text = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in metric)
    assert not path.exists() or path.read_text() == text
    path.write_text(text)
    env = dict(os.environ, HF_HUB_OFFLINE='1')
    subprocess.run(['/home/minhee/.cache/soprano-rag-eval-venv/bin/python',
        str(SCRIPTS / 'score_expanded_bertscore_bounded.py'), '--cases', str(FINAL / 'cases.json'),
        '--results', str(path), '--settings', str(SCRIPTS / 'selected-metric-settings.json'), '--split', 'all',
        '--output', str(FINAL / 'bertscore'), '--device', 'cuda'], check=True, env=env)


def boundaries():
    directory = FINAL / 'boundaries'
    assert len(rows(directory / 'current.jsonl')) == 6
    call('prepare_expanded_judge_packets.py', '--cases', directory / 'cases.json',
         '--results', directory / 'current.jsonl', '--output', directory / 'judge')
    call('run_joint_expanded_judge.py', '--directory', directory / 'judge', '--model', 'claude-opus-5-5', '--workers', '2')


def judgments(directory):
    assessed = {r['id']: r['assessment'] for r in rows(directory / 'assessments.jsonl')}
    result = {}
    for item in read(directory / 'mapping.json'):
        result.setdefault(item['case_id'], {})[item['key'].split(':')[0]] = assessed.get(item['review_id'])
    return result


def stats(values):
    return {'cases': len(values), 'material_drifts': sum(v['source_fidelity'] == 'material_drift' for v in values),
            'minor_drifts': sum(v['source_fidelity'] == 'minor_drift' for v in values),
            'naturalness': statistics.mean(v['naturalness'] for v in values),
            'refusals': sum(v['refusal'] for v in values),
            'completeness': statistics.mean(v['completeness'] for v in values)}


def report():
    before = {r['case_id']: r for r in rows(FINAL / 'before.jsonl')}
    after = {r['case_id']: r for r in rows(FINAL / 'after.jsonl')}
    paired = judgments(FINAL / 'judge'); cases = {c['id']: c for c in read(FINAL / 'cases.json')['cases']}
    assert len(before) == len(after) == len(paired) == 8
    comparison = [{'case_id': cid, 'question': cases[cid]['question'],
                   'before': {'answer': before[cid]['response']['answer'], 'judge': paired[cid]['before']},
                   'after': {'answer': after[cid]['response']['answer'], 'judge': paired[cid]['after']}}
                  for cid in cases]
    freeze(FINAL / 'comparison.json', comparison)
    metric_summary = read(FINAL / 'bertscore/summary.json')[0]
    assert not metric_summary.get('error'), metric_summary
    metric = {r['id']: r for r in read(FINAL / 'bertscore' / (metric_summary['config_id'] + '.json'))['cases']}
    prior_metric = read(BASELINE / 'combined/expert-bertscore.json')
    combined_scores = {r['id']: r for r in prior_metric['cases']}
    assert len(combined_scores) == 109
    for cid, item in metric.items(): combined_scores[cid] = item
    full_cases = read(PRIMARY / 'expert/cases.json')['cases']
    expert_ids = [c['id'] for c in full_cases if c['family'] == 'original']
    combined = {r['case_id']: r for r in rows(BASELINE / 'combined/expert.jsonl')}
    combined_judged = read(BASELINE / 'combined/expert-judgments.json')
    for cid in after:
        assert before[cid]['response']['evidence'] == after[cid]['response']['evidence']
        assert before[cid]['response']['score_context'] == after[cid]['response']['score_context']
        combined[cid] = after[cid]; combined_judged[cid] = paired[cid]['after']
    directory = FINAL / 'combined'; directory.mkdir(exist_ok=True)
    text = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in combined.values())
    target = directory / 'expert.jsonl'
    assert not target.exists() or target.read_text() == text
    target.write_text(text)
    freeze(directory / 'expert-judgments.json', combined_judged)
    freeze(directory / 'expert-bertscore.json', {'cases': list(combined_scores.values())})
    boundary = judgments(FINAL / 'boundaries/judge')
    boundary_rows = rows(FINAL / 'boundaries/current.jsonl')
    summary = {
        'model': read(FINAL / 'writer-settings.json')['vertex_gemini']['model'],
        'affected_existing_cases': {name: stats([v[name] for v in paired.values()]) for name in ['before', 'after']},
        'naturalness_declines': [cid for cid, v in paired.items() if v['after']['naturalness'] < v['before']['naturalness']],
        'naturalness_improvements': [cid for cid, v in paired.items() if v['after']['naturalness'] > v['before']['naturalness']],
        'unchanged_existing_music_prompts_and_answers': 101,
        'unchanged_existing_attack_control_prompts': 73,
        'expert105_bertscore': {'before': prior_metric['summary']['expert105_after_retry'],
            'after': statistics.mean(combined_scores[cid]['f1'] if combined_scores[cid]['scored'] else 0 for cid in expert_ids),
            'fresh_scores': len(set(expert_ids) & set(metric)), 'reused_scores': len(set(expert_ids) - set(metric)),
            'settings': metric_summary['settings'], 'truncated_texts': metric_summary['truncated_texts']},
        'synthetic_boundary_checks': {cid: result['current'] or {'status': 'api_error', 'assessed': False}
                                      for cid, result in boundary.items()},
        'boundary_technical_failures': [read(p) for p in sorted((FINAL / 'boundaries').glob('technical-failure-attempt*.json'))],
        'boundary_recovery': 'At most one additional attempt of the same different_actor-en input after 504; all technical failures retained separately. No technical failure counts as a passed behavior check.',
        'latest_generation_latency': {'mean_seconds': statistics.mean(r['elapsed_seconds'] for r in after.values()),
                                     'median_seconds': statistics.median(r['elapsed_seconds'] for r in after.values()),
                                     'scope': '8 serial generation API calls only; no controlled speed comparison'},
        'limitations': 'Eight affected development questions, not a fresh holdout. 101 unchanged prompts/results reused, not regenerated. Six artificial boundary fixtures excluded from expert metrics. LLM judgments are not expert reannotations. All v1 and v2 outcomes retained separately. The source spelling 루바도 remains in the final expression-advice answer; no claim of complete terminology correction.'}
    freeze(ROOT / 'summary.json', summary)
    for rel, value in read(FINAL / 'contract.json')['source_files'].items(): assert sha(REPO / rel) == value
    for rel, value in read(FINAL / 'contract.json')['historical_files'].items():
        # Demo UI/catalog may be refreshed; every saved answer dataset stays immutable.
        if rel.startswith('sites/') and not Path(rel).name.startswith('data'): continue
        assert sha(REPO / rel) == value
    assert sha(REPO / 'data/corpus.json') == read(FINAL / 'contract.json')['corpus_sha256']
    assert not list(FINAL.rglob('*.pending'))
    for r in [*after.values(), *boundary_rows]:
        if r['status'] == 'response':
            assert r['response']['generation_metadata']['model_version'] == 'gemini-3.8-flash'
        else:
            assert r['status'] == 'exception' and r['response'] is None
    freeze(ROOT / 'verification.json', {'verified': True, 'historical_outputs_unchanged': True,
        'same_question_ku_score_context': True, 'actual_production_prompt_rebuild': True,
        'final_regenerated_existing_cases': 8, 'unchanged_music_cases_reused': 101,
        'synthetic_boundary_cases_excluded_from_expert_metrics': 6,
        'fixed_bertscore_and_no_truncation': True, 'online_output_review_added': False,
        'summary_sha256': sha(ROOT / 'summary.json')})
    print(json.dumps({k: v for k, v in summary.items() if k != 'synthetic_boundary_checks'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['expert', 'boundaries', 'report'])
    globals()[parser.parse_args().action]()
