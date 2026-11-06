"""Reconstruct J's writer inputs through the service using its frozen retrieval.

No model is called here. Never feed evaluation references or baseline answers to
Gemini. A baseline answer is returned by a capture stub only to verify the service
metadata, and is stored separately from the messages sent to the writer.
"""
from contextlib import ExitStack
from dataclasses import fields
from pathlib import Path
from unittest.mock import patch
import ast
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
OLD = REPO / 'evaluation/prompt_fidelity_2026-11-06'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    text = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    if path.exists() and path.read_text() != text:
        raise ValueError('Frozen input changed: ' + str(path))
    path.write_text(text)


def function(path, name):
    return ast.dump(next(node for node in ast.parse(path.read_text()).body
                         if isinstance(node, ast.FunctionDef) and node.name == name))


def main():
    if (ROOT / 'contract.json').exists():
        raise SystemExit('Inputs already frozen. Use run.py; do not re-prepare.')
    old_source = OLD / 'expert-j/snapshot/soprano_qa'
    assert function(REPO / 'soprano_qa/single_pass.py', 'build_messages') == function(old_source / 'single_pass.py', 'build_messages')
    assert sha(REPO / 'soprano_qa/single_pass_prompt.py') == sha(old_source / 'single_pass_prompt.py')
    snapshot = ROOT / 'snapshot'
    shutil.copytree(REPO / 'soprano_qa', snapshot / 'soprano_qa', dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__'))
    os.environ['SOPRANO_QA_SETTINGS'] = str(OLD / 'expert-j/settings.json')
    sys.path.insert(0, str(snapshot))
    from soprano_qa import service as qa
    from soprano_qa.retrieval import SearchResult

    config = read(REPO / 'config/settings.gemini.json')['llm']
    save(ROOT / 'writer-settings.json', config)
    summary = {}
    for cohort in ['expert', 'challenge']:
        old = OLD / (cohort + '-j'); directory = ROOT / cohort
        directory.mkdir(exist_ok=True)
        cases_doc = read(old / 'cases.json'); cases = cases_doc['cases']
        save(directory / 'cases.json', cases_doc)
        rows = [json.loads(line) for line in (old / 'current.jsonl').read_text().splitlines()]
        by_id = {r['case_id']: r for r in rows}
        qa.SETTINGS = read(old / 'settings.json')
        corpus = {r['id']: r for r in read(Path(qa.SETTINGS['corpus_path']))}
        inputs = []; skipped = []; verified = []
        for case in cases:
            baseline = by_id[case['id']]; response = baseline.get('response')
            if response is None:
                skipped.append({'case_id': case['id'], 'reason': 'No baseline writer input: prior query translation failed',
                                'baseline_exception': baseline['exception']})
                continue
            def retrieve(_index, **kw):
                if response.get('generation_validation_warning') == 'unsafe_source_excluded':
                    # The prior service discarded the attacked source before
                    # writing. Reproduce that already-recorded branch; its
                    # writer input contains no source or translated query.
                    from soprano_qa.security import SecurityViolation
                    raise SecurityViolation('unsafe_retrieved_evidence')
                if cohort == 'challenge':
                    results = [SearchResult(copy.deepcopy(p['record']), p.get('score', p.get('relevance', 1)), 0, 0, 1,
                               p.get('scope', p.get('scope_match', 'general_evidence')), alias_score=p.get('alias_score', 0))
                               for p in case['fixed_evidence']]
                else:
                    results = []
                    for item in response['evidence']:
                        record = copy.deepcopy(corpus[item['id']]); assert record['answer'] == item['text']
                        args = {f.name: item[f.name] for f in fields(SearchResult) if f.name in item}
                        args.update(record=record, piece_score=1)
                        results.append(SearchResult(**args))
                return results, kw['message_builder'](kw['piece'], kw['measures'], kw['query'], results, score_context=kw['score_context'])

            captured = []
            def capture(_path, messages, _settings):
                captured.append(copy.deepcopy(messages)); return response['answer']

            with ExitStack() as stack:
                for name, value in [('_get_index', object()), ('validate_generation_requirements', None),
                                    ('load_measure_lyrics', {}), ('selected_measure_lyrics', response.get('score_context')),
                                    ('translate_question_to_korean', response['retrieval_query'] or '기록된 자료 제외 경로'),
                                    ('retrieval_diagnostics', response['retrieval'])]:
                    stack.enter_context(patch.object(qa, name, return_value=value))
                stack.enter_context(patch.object(qa, 'retrieve_generation_evidence', side_effect=retrieve))
                stack.enter_context(patch.object(qa, 'generate_llm', side_effect=capture))
                replay = qa.ask(piece_id=case['piece_id'], question=case['question'],
                    measure_range=tuple(case['measure_range']) if case.get('measure_range') else None,
                    generate=True, top_k=case.get('top_k', 6), language=case['language'])
            assert len(captured) == 1, (case['id'], len(captured))
            different = [key for key in response if key != 'model' and response[key] != replay[key]]
            assert not different, (case['id'], different)
            messages = captured[0]
            inputs.append({'case_id': case['id'], 'messages': messages,
                           'messages_sha256': hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
                           'baseline_response': response})
            verified.append(case['id'])
        save(directory / 'writer-inputs.json', inputs)
        save(directory / 'excluded.json', skipped)
        summary[cohort] = {'paired': len(inputs), 'excluded': skipped,
                           'baseline_sha256': sha(old / 'current.jsonl'),
                           'cases_sha256': sha(directory / 'cases.json'),
                           'inputs_sha256': sha(directory / 'writer-inputs.json'),
                           'replayed_service_metadata_equal': len(verified)}
        for filename in ['prepare_expanded_judge_packets.py', 'run_expanded_judge.py', 'run_joint_expanded_judge.py',
                         'run_reused_expanded_judge.py', 'watch_judge.py', 'score.py',
                         'score_expanded_bertscore_bounded.py', 'selected-metric-settings.json', 'metric-length-amendment.json']:
            if (old / filename).exists(): shutil.copy2(old / filename, directory / filename)
    save(ROOT / 'contract.json', {
        'source_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        'model': config['vertex_gemini'], 'writer_settings_sha256': sha(ROOT / 'writer-settings.json'),
        'run_sha256': sha(ROOT / 'run.py'),
        'source_files': {str(p.relative_to(snapshot)): sha(p) for p in sorted(snapshot.rglob('*.py'))},
        'cohorts': summary,
        'comparison': 'Same J prompts, questions, KUs and score context, writer only replaced. Temperature 0/top_p .8/seed42 retained. Gemini LOW thinking and 2048 total output budget differ from local no_think/768. Seeds and tokenizers are not equivalent across providers.',
        'policy': 'First completed answer per case; no quality retries, tools, web grounding, reviewers or source substitution. Excluded pre-writer baseline failures are not counted as defenses. Standard online API, sequential generation latency excludes retrieval/translation.'})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
