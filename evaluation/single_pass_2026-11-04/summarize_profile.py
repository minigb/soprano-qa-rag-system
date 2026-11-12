"""Summarize measured stage costs without counting inclusive spans twice."""
from pathlib import Path
import collections
import hashlib
import json
import statistics

ROOT = Path(__file__).resolve().parent
STAGES = ('input_admission', 'ku_selection', 'answer_generation', 'output_review',
          'instruction_review', 'question_translation')


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    report = {'scope': '20 fixed questions, two passes in forward/reverse order; cold load and one warmup excluded. Slow-path-enriched diagnostic set, not a traffic estimate. Separate sequential runs, not randomized interleaving.', 'runs': {}}
    baseline = None
    for name in ('baseline-measured', 'single-pass-a', 'single-pass-b', 'single-pass-c', 'single-pass-final', 'single-pass-e', 'single-pass-f', 'single-pass-g-clean'):
        root = ROOT / name
        if not (root / 'completion.json').exists():
            continue
        data = rows(root / 'timings.jsonl')
        assert len(data) == len({r['id'] for r in data}) == 40
        assert not any(r['exception'] for r in data)
        assert hashlib.sha256((root / 'timings.jsonl').read_bytes()).hexdigest() == read(root / 'completion.json')['results_sha256']
        indexed = {r['id']: r for r in data}
        if baseline is None:
            baseline = indexed
        info = {'groups': {}, 'loading': read(root / 'completion.json'),
                'repeated_answer_changes': sum(indexed['0:' + r['case_id']]['response']['answer'] != r['response']['answer']
                                              for r in data if r['repetition'] == 1),
                'same_retrieved_ids_as_baseline': sum([e['id'] for e in r['response']['evidence']] ==
                    [e['id'] for e in baseline[r['id']]['response']['evidence']] for r in data)}
        for group in sorted({r['timing_group'] for r in data}):
            chosen = [r for r in data if r['timing_group'] == group]
            info['groups'][group] = {
                'requests': len(chosen), 'unique_questions': len({r['case_id'] for r in chosen}),
                'mean_seconds': statistics.mean(r['elapsed_seconds'] for r in chosen),
                'median_seconds': statistics.median(r['elapsed_seconds'] for r in chosen),
                'min_seconds': min(r['elapsed_seconds'] for r in chosen),
                'max_seconds': max(r['elapsed_seconds'] for r in chosen),
                'llm_calls': dict(collections.Counter(r['llm_calls'] for r in chosen)),
                'stage_mean_seconds': {s: statistics.mean(r['llm_seconds_by_stage'].get(s, 0) for r in chosen) for s in STAGES},
                'retrieval_mean_seconds': statistics.mean(r['retrieval_seconds'] for r in chosen),
                'remaining_mean_seconds': statistics.mean(r['remaining_seconds'] for r in chosen),
                'writer_completion_tokens_mean': statistics.mean(sum(c['completion_tokens'] or 0 for c in r['completions'] if c['stage'] == 'answer_generation') for r in chosen),
                'writer_prompt_tokens_mean': statistics.mean(sum(c['prompt_tokens'] or 0 for c in r['completions'] if c['stage'] == 'answer_generation') for r in chosen),
            }
        if name.startswith('single-pass'):
            assert all(set(r['llm_seconds_by_stage']) <= {'answer_generation', 'question_translation'} for r in data)
            assert all(r['llm_calls'] == (1 if r['language'] == 'ko' else 2) for r in data)
            info['one_writer_no_model_review_verified'] = True
        report['runs'][name] = info
    (ROOT / 'profile-summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    for name, run in report['runs'].items():
        print(name, {k: round(v['mean_seconds'], 3) for k, v in run['groups'].items()})


if __name__ == '__main__':
    main()
