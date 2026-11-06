"""Aggregate fixed-context writer comparisons; retain errors and exclusions."""
from pathlib import Path
import collections
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
OLD = REPO / 'evaluation/prompt_fidelity_2026-11-06'


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def latency_stats(values):
    if not values:
        return {'count': 0}
    return {'count': len(values), 'mean': statistics.mean(values),
            'median': statistics.median(values), 'max': max(values),
            'p95_nearest_rank': sorted(values)[math.ceil(.95 * len(values)) - 1]}


def judgments(run):
    output = {}
    for batch in sorted((run / 'judge-batches').iterdir()):
        assessments = {r['id']: r['assessment'] for r in rows(batch / 'assessments.jsonl')}
        for mapping in read(batch / 'mapping.json'):
            assert mapping['case_id'] not in output
            output[mapping['case_id']] = assessments.get(mapping['review_id'])
    return output


def metrics(ids, outputs, assessed):
    valid = [assessed[cid] for cid in ids if assessed[cid] is not None]
    attack_judged = [j for j in valid if isinstance(j['attack_success'], bool)]
    attack_successes = sum(j['attack_success'] for j in attack_judged)
    return {'count': len(ids), 'assessed': len(valid),
            'technical_errors': sum(outputs[cid]['status'] == 'exception' for cid in ids),
            'excluded': sum(outputs[cid]['status'] == 'excluded' for cid in ids),
            'source_fidelity': dict(collections.Counter(j['source_fidelity'] for j in valid)),
            'refusals': sum(j['refusal'] for j in valid),
            'material_drift': sum(j['source_fidelity'] == 'material_drift' for j in valid),
            'naturalness_effective_mean': sum(j['naturalness'] if not j['refusal'] else 0 for j in valid) / len(ids) if ids else None,
            'naturalness_conditional_mean': statistics.mean(j['naturalness'] for j in valid) if valid else None,
            'adequately_complete': sum(j['completeness'] == 2 and not j['refusal'] for j in valid),
            'factual_error_flags': sum(j['factual_correctness'] <= 2 and not j['refusal'] for j in valid),
            'attack_successes': attack_successes,
            'attack_assessed': len(attack_judged),
            'attack_success_rate_assessed': attack_successes / len(attack_judged) if attack_judged else None}


def version(row, judgment):
    return {'answer': (row.get('response') or {}).get('answer'), 'status': row['status'],
            'exception': row.get('exception'), 'judge': judgment,
            'latency_seconds': row['elapsed_seconds'], 'latency_scope': row.get('latency_scope'),
            'generation_metadata': (row.get('response') or {}).get('generation_metadata')}


def main():
    summary = {'model': read(ROOT / 'writer-settings.json')['vertex_gemini'],
               'contract_sha256': sha(ROOT / 'contract.json')}
    compared = {}; index = {}
    for cohort in ['expert', 'challenge']:
        new_run = ROOT / cohort; old_run = OLD / (cohort + '-j')
        cases = read(new_run / 'cases.json')['cases']
        assert cases == read(old_run / 'cases.json')['cases']
        new = {r['case_id']: r for r in rows(new_run / 'current.jsonl')}
        old = {r['case_id']: r for r in rows(old_run / 'current.jsonl')}
        assert len(new) == len(old) == len(cases)
        assert sha(new_run / 'current.jsonl') == read(new_run / 'completion.json')['result_sha256']
        nj, oj = judgments(new_run), judgments(old_run)
        assert len(nj) == len(oj) == len(cases)
        paired = [c['id'] for c in cases if new[c['id']]['status'] != 'excluded']
        for cid in paired:
            if new[cid]['status'] == 'response': assert nj[cid] is not None
            else: assert nj[cid] is None
            assert oj[cid] is not None
        details = []
        for c in cases:
            cid = c['id']; evidence = (old[cid].get('response') or {}).get('evidence', [])
            if new[cid]['status'] == 'response': assert evidence == new[cid]['response']['evidence']
            item = {'id': cid, 'question': c['question'], 'language': c['language'],
                    'kind': c['kind'], 'piece_id': c['piece_id'], 'measure_range': c.get('measure_range'),
                    'evidence': evidence, 'reference_answers': c.get('reference_answers'),
                    'qwen14b_j': version(old[cid], oj[cid]), 'gemini': version(new[cid], nj[cid])}
            details.append(item); index[cid] = item
        compared[cohort] = details
        if cohort == 'expert':
            ids = [c['id'] for c in cases if c['family'] == 'original']
            assert len(ids) == 105
            summary['expert105'] = {'qwen14b_j': metrics(ids, old, oj), 'gemini': metrics(ids, new, nj)}
            successful_ids = [cid for cid in ids if new[cid]['status'] == 'response']
            summary['expert_successful_pairs'] = {
                'case_ids': successful_ids,
                'qwen14b_j': metrics(successful_ids, old, oj),
                'gemini': metrics(successful_ids, new, nj)}
            summary['old_four_material_cases'] = {
                cid: {'new_status': new[cid]['status'], 'new_judge': nj[cid]}
                for cid in ids if oj[cid]['source_fidelity'] == 'material_drift'}
            summary['new_material_ids'] = [cid for cid in ids if nj[cid] and nj[cid]['source_fidelity'] == 'material_drift'
                                           and oj[cid]['source_fidelity'] != 'material_drift']
            refusal_ids = [c['id'] for c in read(OLD / 'profile-cases.json')['cases'] if c['timing_group'] == 'previous_refusal']
            summary['previous_five_refusals'] = {'qwen14b_j': metrics(refusal_ids, old, oj), 'gemini': metrics(refusal_ids, new, nj)}
            summary['naturalness_paired'] = dict(collections.Counter(
                'technical_error' if nj[cid] is None else 'improved' if nj[cid]['naturalness'] > oj[cid]['naturalness']
                else 'worse' if nj[cid]['naturalness'] < oj[cid]['naturalness'] else 'same' for cid in ids))
            metric_summary = read(new_run / 'bertscore/summary.json')[0]
            scored = {r['id']: r for r in read(new_run / 'bertscore' / (metric_summary['config_id'] + '.json'))['cases']}
            old_metric_summary = read(old_run / 'bertscore/summary.json')[0]
            old_scored = {r['id']: r for r in read(old_run / 'bertscore' / (old_metric_summary['config_id'] + '.json'))['cases']}
            assert metric_summary['settings'] == old_metric_summary['settings']
            assert metric_summary['model_revision'] == old_metric_summary['model_revision']
            summary['bertscore'] = {'expert105_effective_f1': statistics.mean(scored[cid]['f1'] for cid in ids),
                'expert105_local_f1': statistics.mean(old_scored[cid]['f1'] for cid in ids),
                'successful_pairs_local_f1': statistics.mean(old_scored[cid]['f1'] for cid in successful_ids),
                'expert105_scored': sum(scored[cid]['scored'] for cid in ids),
                'all109_effective_f1': metric_summary['effective']['f1'],
                'expert105_conditional_f1': statistics.mean(scored[cid]['f1'] for cid in ids if scored[cid]['scored']),
                'truncated_texts': metric_summary['truncated_texts'], 'overlength': metric_summary['unscorable_overlength_cases'],
                'settings': metric_summary['settings'], 'model_revision': metric_summary['model_revision']}
        else:
            summary['challenge'] = {}
            for kind in sorted({c['kind'] for c in cases}):
                summary['challenge'][kind] = {}
                for language in ['all', 'ko', 'en']:
                    ids = [c['id'] for c in cases if c['id'] in paired and c['kind'] == kind
                           and (language == 'all' or c['language'] == language)]
                    successful = [cid for cid in ids if new[cid]['status'] == 'response']
                    summary['challenge'][kind][language] = {
                        'qwen14b_j': metrics(ids, old, oj), 'gemini': metrics(ids, new, nj),
                        'successful_pairs': {
                            'qwen14b_j': metrics(successful, old, oj),
                            'gemini': metrics(successful, new, nj)}}
            summary['challenge_excluded'] = read(new_run / 'excluded.json')
            indirect_inputs = [r for r in read(new_run / 'writer-inputs.json')
                               if r['case_id'].startswith('indirect-')]
            summary['indirect_source_boundary'] = {
                'cases': len(indirect_inputs),
                'unsafe_source_excluded_warning': sum(r['baseline_response'].get('generation_validation_warning') == 'unsafe_source_excluded' for r in indirect_inputs),
                'empty_writer_evidence': sum(not r['baseline_response']['evidence'] for r in indirect_inputs),
                'note': 'Application-level defense includes unchanged source screening. Do not attribute every indirect defense to the writer model.'}
    profile = read(OLD / 'profile-cases.json')['cases']
    old_times = rows(OLD / 'timing-j/timings.jsonl')
    summary['latency'] = {}
    for group in sorted({c['timing_group'] for c in profile}):
        ids = [c['id'] for c in profile if c['timing_group'] == group]
        paired_ids = [cid for cid in ids if cid in index and index[cid]['gemini']['status'] == 'response']
        baseline = [t['llm_seconds_by_stage']['answer_generation'] for t in old_times if t['case_id'] in paired_ids]
        api = [index[cid]['gemini']['generation_metadata']['elapsed_seconds'] for cid in paired_ids]
        summary['latency'][group] = {'same_question_ids': paired_ids,
            'qwen_writer_mean_seconds': statistics.mean(baseline), 'qwen_observations': len(baseline),
            'gemini_writer_api_mean_seconds': statistics.mean(api), 'gemini_observations': len(api),
            'note': 'Historical Qwen writer stage, two serial repetitions; Gemini quality capture, one replay, mixed concurrency and spacing (see execution contracts and resume events). Excludes retrieval/translation. NOT a controlled latency comparison. Shared YouTube API work continued.'}
    sequential = ROOT / 'timing-sequential.jsonl'
    if sequential.exists() and len(rows(sequential)) == 10:
        timing = rows(sequential)
        ids = {r['case_id'] for r in timing}
        old_serial = [r['llm_seconds_by_stage']['answer_generation'] for r in old_times if r['case_id'] in ids]
        new_serial = [r['elapsed_seconds'] for r in timing if r['status'] == 'response']
        summary['sequential_latency'] = {
            'cases': len(ids), 'qwen_observations': len(old_serial), 'gemini_observations': len(timing),
            'gemini_errors': len(timing)-len(new_serial),
            'qwen_generation_mean_seconds': statistics.mean(old_serial),
            'qwen_generation_median_seconds': statistics.median(old_serial),
            'gemini_api_mean_success_seconds': statistics.mean(new_serial) if new_serial else None,
            'gemini_api_median_success_seconds': statistics.median(new_serial) if new_serial else None,
            'gemini_api_mean_including_errors_seconds': statistics.mean(r['elapsed_seconds'] for r in timing),
            'note': 'Same ten fixed ordinary-music questions. Qwen J: previous serial generation-stage measurements (2 repeats), Gemini: separate serial API replay after quality generation (1 repeat). Retrieval/translation excluded; first Gemini connection establishment included. Shared YouTube API workload continues. Different collection times, no SLA or model-size-only inference.'}
    api_rows = rows(ROOT / 'expert/current.jsonl') + rows(ROOT / 'challenge/current.jsonl')
    success = [r for r in api_rows if r['status'] == 'response']; failed = [r for r in api_rows if r['status'] == 'exception']
    summary['api'] = {'successful_requests': len(success), 'failed_requests': len(failed),
        'failures': [{'case_id': r['case_id'], 'exception': r['exception'], 'seconds': r['elapsed_seconds']} for r in failed],
        'successful_wall_mean_seconds': statistics.mean(r['elapsed_seconds'] for r in success),
        'all_attempts_wall_mean_seconds': statistics.mean(r['elapsed_seconds'] for r in success + failed),
        'usage': {key: sum((r['response']['generation_metadata']['usage'].get(key) or 0) for r in success)
                  for key in ['prompt_token_count', 'candidates_token_count', 'thoughts_token_count', 'total_token_count']}}
    summary['api']['quality_capture_latency_by_cohort'] = {
        cohort: latency_stats([r['elapsed_seconds'] for r in rows(ROOT / cohort / 'current.jsonl')
                               if r['status'] == 'response'])
        for cohort in ['expert', 'challenge']}
    summary['api']['usage_note'] = 'Sum of fields reported for successful responses only; absent fields contribute zero to the sum. This is not a billing estimate or proof of no internal reasoning. Failed-call billing is unknown.'
    for cohort, data in compared.items():
        (ROOT / (cohort + '-comparison.json')).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    (ROOT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
