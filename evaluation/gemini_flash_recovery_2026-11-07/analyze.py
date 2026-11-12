"""Report original first attempts and coverage after one technical retry separately."""
from pathlib import Path
import copy
import json
import runpy
import statistics

ROOT = Path(__file__).resolve().parent
PRIMARY = ROOT.parent / 'gemini_flash_writer_2026-11-07'
LOCAL = ROOT.parent / 'prompt_fidelity_2026-11-06'
REPO = ROOT.parents[1]


def main():
    helpers = runpy.run_path(str(PRIMARY / 'analyze.py'))
    read, rows, sha, metrics = [helpers[k] for k in ['read', 'rows', 'sha', 'metrics']]
    contract = read(ROOT / 'contract.json')
    for name, digest in contract['original_files'].items(): assert sha(REPO / name) == digest
    retry = {r['case_id']: r for r in rows(ROOT / 'current.jsonl')}
    assert set(retry) == {r['case_id'] for r in contract['plan']}
    assert sha(ROOT / 'current.jsonl') == read(ROOT / 'completion.json')['result_sha256']
    assert not list((ROOT / 'requests').glob('*.pending'))
    assessed = {r['id']: r['assessment'] for r in rows(ROOT / 'judge/assessments.jsonl')}
    retry_judged = {r['case_id']: assessed.get(r['review_id']) for r in read(ROOT / 'judge/mapping.json')}
    summary = {'policy': contract['policy'], 'model': contract['model'], 'retry_project': contract['project'],
               'retry_attempts': len(retry), 'retry_answers': sum(r['status'] == 'response' for r in retry.values()),
               'retry_errors': sum(r['status'] == 'exception' for r in retry.values()),
               'original_primary_artifacts_unchanged': True}
    for cohort in ['expert', 'challenge']:
        cases = read(PRIMARY / cohort / 'cases.json')['cases']
        original = {r['case_id']: r for r in rows(PRIMARY / cohort / 'current.jsonl')}
        local = {r['case_id']: r for r in rows(LOCAL / (cohort + '-j') / 'current.jsonl')}
        oj = helpers['judgments'](PRIMARY / cohort)
        lj = helpers['judgments'](LOCAL / (cohort + '-j'))
        combined = copy.deepcopy(original); combined_j = copy.deepcopy(oj)
        for cid in original:
            if cid in retry:
                assert original[cid]['status'] == 'exception' and original[cid].get('response') is None
                combined[cid] = retry[cid]; combined_j[cid] = retry_judged[cid]
                assert (retry[cid]['status'] == 'response') == (retry_judged[cid] is not None)
            else:
                assert combined[cid] == original[cid] and combined_j[cid] == oj[cid]
        directory = ROOT / 'combined'; directory.mkdir(exist_ok=True)
        (directory / (cohort + '.jsonl')).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in combined.values()))
        (directory / (cohort + '-judgments.json')).write_text(json.dumps(combined_j, ensure_ascii=False, indent=2)+'\n')
        if cohort == 'expert':
            ids = [c['id'] for c in cases if c['family'] == 'original']; assert len(ids) == 105
            summary['expert105'] = {'local': metrics(ids, local, lj),
                                   'first_attempt': metrics(ids, original, oj),
                                   'after_one_technical_retry': metrics(ids, combined, combined_j)}
            primary_metric = read(PRIMARY / 'expert/bertscore/summary.json')[0]
            new_metric = read(ROOT / 'expert/bertscore/summary.json')[0]
            assert primary_metric['settings'] == new_metric['settings']
            assert primary_metric['model_revision'] == new_metric['model_revision']
            assert primary_metric['settings']['idf'] is False
            assert read(PRIMARY / 'expert/bertscore/contract.json')['packages'] == read(ROOT / 'expert/bertscore/contract.json')['packages']
            old_scores = {r['id']: r for r in read(PRIMARY / 'expert/bertscore' / (primary_metric['config_id']+'.json'))['cases']}
            new_scores = {r['id']: r for r in read(ROOT / 'expert/bertscore' / (new_metric['config_id']+'.json'))['cases']}
            assert set(new_scores) == {cid for cid in ids if cid in retry}
            scores = copy.deepcopy(old_scores)
            for cid, score in new_scores.items():
                assert old_scores[cid]['unscorable_reason'] == 'pipeline_error_or_unavailable'
                scores[cid] = score
            result = {'expert105_first_attempt': statistics.mean(old_scores[cid]['f1'] for cid in ids),
                      'expert105_after_retry': statistics.mean(scores[cid]['f1'] for cid in ids),
                      'expert105_scored_after_retry': sum(scores[cid]['scored'] for cid in ids),
                      'settings': new_metric['settings'], 'model_revision': new_metric['model_revision'],
                      'truncated_texts': new_metric['truncated_texts'],
                      'overlength_after_retry': sum(scores[cid]['unscorable_reason'] == 'encoder_length_limit' for cid in ids),
                      'score_reuse': 'No IDF: retain all99 existing per-answer scores verbatim; score only6 previously missing expert answers with identical encoder/settings/packages.'}
            result['target_after_retry_met'] = result['expert105_after_retry'] >= .9
            summary['bertscore'] = result
            (directory / 'expert-bertscore.json').write_text(json.dumps({'summary': result, 'cases': list(scores.values())}, indent=2)+'\n')
        else:
            summary['challenge'] = {}
            for kind in sorted({c['kind'] for c in cases}):
                ids = [c['id'] for c in cases if c['kind'] == kind and original[c['id']]['status'] != 'excluded']
                summary['challenge'][kind] = {'local': metrics(ids, local, lj),
                                             'first_attempt': metrics(ids, original, oj),
                                             'after_one_technical_retry': metrics(ids, combined, combined_j)}
        details = []
        for c in cases:
            cid = c['id']
            details.append({'id': cid, 'question': c['question'], 'kind': c['kind'],
                            'reference_answers': c.get('reference_answers'),
                            'local': helpers['version'](local[cid], lj[cid]),
                            'gemini_first_attempt': helpers['version'](original[cid], oj[cid]),
                            'gemini_after_retry': helpers['version'](combined[cid], combined_j[cid]),
                            'retried': cid in retry})
        (ROOT / (cohort+'-comparison.json')).write_text(json.dumps(details, ensure_ascii=False, indent=2)+'\n')
    summary['latency_policy'] = 'No repeat latency benchmark. These7 requests were selected because the original requests failed and used a different project/time; their latency is not a controlled speed comparison.'
    summary['runtime_source_changes'] = False
    summary['default_configuration_changed'] = False
    (ROOT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    for item in contract['plan']:
        row = retry[item['case_id']]
        assert row['messages_sha256'] == item['messages_sha256']
        assert row['api_project'] == contract['project']
        if row['status'] == 'response':
            md = row['response']['generation_metadata']
            assert md['requested_model'] == md['model_version'] == 'gemini-3.8-flash'
            assert md['finish_reason'] == 'STOP'
            inputs = {r['case_id']: r for r in read(PRIMARY / item['cohort'] / 'writer-inputs.json')}
            assert row['response']['evidence'] == inputs[row['case_id']]['baseline_response']['evidence']
            assert row['response']['score_context'] == inputs[row['case_id']]['baseline_response']['score_context']
    for name, digest in contract['original_files'].items(): assert sha(REPO / name) == digest
    verification = {'verified': True, 'original_files_preserved': len(contract['original_files']),
                    'same_model_prompt_evidence_and_metric': True, 'successful_answers_regenerated': 0,
                    'original_judgments_reused_without_changes': 175, 'retry_cases': 7,
                    'first_attempt_metrics_preserved': True, 'after_retry_bertscore_target_met': result['target_after_retry_met']}
    (ROOT / 'verification.json').write_text(json.dumps(verification, indent=2)+'\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
