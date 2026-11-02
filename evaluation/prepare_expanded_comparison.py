"""Freeze the existing 460 requests and expert references without using outputs."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evaluation/expanded_460_2026-11-02'
PREVIOUS = ROOT / 'evaluation/ku_background_2026-10-31'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(path, value):
    text = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    if path.exists() and path.read_text() != text:
        raise ValueError('Frozen evaluation input changed: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def main():
    corpus = {r['id']: r for r in read(ROOT/'data/corpus.json')}
    original = read(ROOT/'evaluation/qualitative.json')
    expanded = read(ROOT/'evaluation/synthesized_question_results.json')
    expert = read(PREVIOUS/'expert_cases.json')['cases']
    expert_by_id = {c['id']: c for c in expert}
    splits = {c['pair_id']: c['split'] for c in expert}
    cases = []
    for artifact, groups in [('evaluation/qualitative.json', original['questions']),
                             ('evaluation/synthesized_question_results.json', expanded['results'])]:
        for group in groups:
            family = ('original' if artifact.endswith('/qualitative.json')
                      else {'source_question_paraphrase': 'paraphrase',
                            'knowledge_unit_derived': 'ku_derived'}[group['synthesized_question_kind']])
            for run in group['inference_runs']:
                req = run['inference_input']
                authority = req['case_reference_authority']
                assert authority['source'] == 'applicable_finalized_knowledge_units'
                assert not authority['manual_review_required']
                refs = authority['items']
                ids = list(dict.fromkeys(i['knowledge_unit_id'] for i in refs))
                assert ids
                for item in refs:
                    record = corpus[item['knowledge_unit_id']]
                    assert record['evidence_type'] == 'expert_annotation'
                    assert record['piece'] == req['piece_id']
                    assert record['answer'].strip() == item['text'].strip()
                reference = '\n'.join(corpus[i]['answer'].strip() for i in ids)
                if family == 'original':
                    case = deepcopy(expert_by_id['expert-' + run['case_id']])
                    assert case['reference_answers'] == [reference]
                    assert case['question'] == req['question']
                    assert case['measure_range'] == req['measure_range']
                else:
                    case = {'id': 'expanded-' + run['case_id'], 'kind': 'benign', 'language': 'ko',
                            'pair_id': group['source_id'], 'piece_id': req['piece_id'],
                            'question': req['question'], 'measure_range': req['measure_range'],
                            'category': 'expert_reference',
                            'split': splits.get(group['source_id'], 'expansion'),
                            'gold_evidence_ids': ids, 'reference_answers': [reference],
                            'reference_policy': 'all applicable finalized expert KUs, concatenated in frozen order',
                            'injection': None}
                case.update(family=family, reference_authority=deepcopy(authority),
                            archived_case_id=run['case_id'], source_artifact=artifact,
                            source_group_id=group['source_id'],
                            question_provenance=req.get('question_provenance'),
                            case_kind=req.get('case_kind'),
                            source_annotation_ids=group.get('source_annotation_ids', [group['source_id']]))
                cases.append(case)
    assert Counter(c['family'] for c in cases) == {'original': 105, 'paraphrase': 306, 'ku_derived': 49}
    assert len({c['id'] for c in cases}) == 460
    signature = lambda c: json.dumps({k: c[k] for k in ['piece_id', 'question', 'measure_range', 'language']}, sort_keys=True)
    random.Random(20261102).shuffle(cases)
    freeze(OUT/'cases.json', {'suite_type': 'frozen_existing_460_expert_based_requests', 'cases': cases})
    inputs = ['evaluation/qualitative.json', 'evaluation/synthesized_question_results.json',
              'data/corpus.json', 'evaluation/ku_background_2026-10-31/expert_cases.json',
              'evaluation/ku_background_2026-10-31/bertscore-development-b/selected_settings.json']
    freeze(OUT/'inventory.json', {
        'cases': len(cases), 'unique_request_inputs': len({signature(c) for c in cases}),
        'families': dict(Counter(c['family'] for c in cases)),
        'pieces': dict(Counter(c['piece_id'] for c in cases)),
        'language': {'ko': 460}, 'original_source_questions': 79,
        'paraphrase_formulations': 237, 'ku_derived_formulations': 36,
        'original_metric_splits': dict(Counter(c['split'] for c in cases if c['family'] == 'original')),
        'source_input_sha256': {name: digest(ROOT/name) for name in inputs},
        'reference_provenance': 'Exact applicable finalized expert KUs, verified equal to archived reference text. Never prior model answers.',
        'question_provenance': '79 human-source questions; 237 paraphrase formulations; 36 machine-authored KU-derived formulations. Range conditions expand them to460 cases.',
        'holdout_status': 'Reused internal benchmark; neither unseen questions nor independent expert English references.',
    })
    config = {
        'generation_python': '/home/minhee/miniconda3/envs/soprano-qa/bin/python',
        'metric_python': '/home/minhee/.cache/soprano-rag-eval-venv/bin/python',
        'cases': str((OUT/'cases.json').relative_to(ROOT)),
        'metric_settings': str((PREVIOUS/'bertscore-development-b/selected_settings.json').relative_to(ROOT)),
        'output': str(OUT.relative_to(ROOT)),
        'warmup_case': 'expert-kim-die-forelle-03__no-range',
        'systems': [
            {'name': 'candidate-e37', 'variant': 'hardened',
             'root': str((PREVIOUS/'runtime/followup-e37/snapshots/hardened').relative_to(ROOT)),
             'source_commit': '414a6c0a830ffaf9c822574a66853ddcf46ea42c'},
            {'name': 'pre-security', 'variant': 'original',
             'root': str((PREVIOUS/'runtime/baseline-pre-security/snapshots/original').relative_to(ROOT)),
             'source_commit': 'f3074700103d60cf4bd8862fd655e51656f66f0c'},
        ],
        'protocol': {
            'cases_per_system': 460, 'repetitions': 1,
            'generation': 'All920 requests newly run through each native public service; no historical answer reuse. Same frozen case order, one untimed warmup per system, serial GPU use.',
            'reference': 'Full finalized expert KU text; original question, range, reference and metric setting are unchanged.',
            'judge': 'Fresh blinded Claude Opus5.5 judgments with source-fidelity and a separate0-4 naturalness scale; exact full-packet duplicates judged once and mapped back.',
            'metric': 'Frozen xlm-roberta-large18 raw F1, IDFfalse, slow tokenizer; no tuning. Whole visible answers, errors/refusals zero. Report original105, original-development43, original-validation62, paraphrase306, KU-derived49 and total460.',
            'primary_bertscore_floor': {'cohort': 'original_validation', 'count': 62, 'f1': 0.90},
            'interpretation': 'Shared-source, reused internal evaluation; grouped question variants are correlated. Naturalness and factual meaning assessed separately from BERTScore.',
            'corpus_policy': 'Native version corpora: original946 records, E37 adds7 Sumi Jo feedback records. All946 shared records are byte-content equivalent, including all122 expert KUs. The original validator does not support sumijo_feedback. This is a full-system comparison, not a code-only effect estimate.',
            'no_score_based_retries': True, 'no_production_changes_during_run': True,
            'usage_reset_stop': False, 'notifier_used': False,
        },
    }
    for system in config['systems']:
        system['settings'] = str(Path(system['root'])/'settings.json')
        system['settings_sha256'] = digest(ROOT/system['settings'])
        corpus_path = ROOT/system['root']/'data/corpus.json'
        snapshot_corpus = {r['id']: r for r in read(corpus_path)}
        assert all(corpus[i] == record for i, record in snapshot_corpus.items())
        missing = sorted(corpus.keys() - snapshot_corpus.keys())
        assert all(corpus[i]['evidence_type'] == 'sumijo_feedback' for i in missing)
        assert len(missing) == (7 if system['name'] == 'pre-security' else 0)
        system['corpus_sha256'] = digest(corpus_path)
        system['corpus_records'] = len(snapshot_corpus)
        system['missing_from_current_corpus'] = missing
    freeze(OUT/'config.json', config)
    print(json.dumps(read(OUT/'inventory.json'), ensure_ascii=False))


if __name__ == '__main__':
    main()
