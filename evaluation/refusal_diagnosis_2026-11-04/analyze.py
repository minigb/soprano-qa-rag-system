"""Explain saved failures with pure validators; no additional model calls."""
from pathlib import Path
from types import SimpleNamespace
import json
import os
import sys

ROOT = Path(__file__).resolve().parent
CAPTURE = ROOT.parent / 'latest_demo_2026-11-03'
os.environ['SOPRANO_QA_SETTINGS'] = str(CAPTURE / 'settings.json')
sys.path.insert(0, str(CAPTURE / 'snapshot'))
from soprano_qa import answer

settings = json.loads((CAPTURE / 'settings.json').read_text())
records = {r['id']: r for r in json.loads(Path(settings['corpus_path']).read_text())}
rows = [json.loads(line) for line in (ROOT / 'traces.jsonl').read_text().splitlines()]
assert len(rows) == 5 and all(r['same_public_answer'] and r['same_reason'] for r in rows)
findings = []
for row in rows:
    retrieved = next(e for e in row['events'] if e['stage'] == 'retrieve')
    selected = next(e for e in row['events'] if e['stage'] == 'select')['selected_evidence']
    assert len(selected) == len(retrieved['retrieved_ids'])
    reconstructed = []
    for index, (unit_id, passage) in enumerate(zip(retrieved['retrieved_ids'], selected)):
        assert passage['citation_label'] == f'E{index + 1}'
        assert records[unit_id]['answer'] == passage['answer'] == retrieved['retrieved_answers'][index]
        reconstructed.append(SimpleNamespace(record=records[unit_id], scope_match=passage['query_scope_relation']))
    local = {i for i, r in enumerate(reconstructed) if r.scope_match == 'local_example'}
    general = {i for i, r in enumerate(reconstructed) if r.scope_match == 'general_evidence'}
    item = {'case_id': row['case_id'], 'question': row['question'], 'saved_reason': row['saved_reason'],
            'retrieved_ids': retrieved['retrieved_ids'], 'admission_allowed': all(
                e['outcome'] == 'returned' for e in row['events']
                if e['stage'] == 'guard' and e['guard_stage'] == 'admission'),
            'attempts': []}
    for event in row['events']:
        if event['stage'] == 'finalize' and event['outcome'] == 'raised':
            assert event['message'] == 'Generated answer overgeneralizes local evidence'
            assert answer.answer_overgeneralizes_local_examples(event['draft'], reconstructed, [])
            rejected_claims = []
            for claim in answer._answer_claims(event['draft']):
                cited = answer._claim_cited_result_indices(claim, reconstructed)
                if (cited & local and answer._claim_has_piece_wide_generalization(claim)
                        and not cited & general
                        and not answer._copies_scoped_source_sentence(claim, reconstructed, cited & local)):
                    rejected_claims.append({'claim': claim,
                        'markers': answer.LOCAL_SCOPE_GENERALIZATION_RE.findall(claim),
                        'cited_roles': [reconstructed[i].scope_match for i in sorted(cited)],
                        'matching_complete_local_source_sentence': False})
            assert rejected_claims
            item['attempts'].append({'blocking_stage': 'deterministic_scope_validator',
                'exception': event['message'], 'claims': rejected_claims})
        elif (event['stage'] == 'guard' and event['guard_stage'] == 'output'
              and event['outcome'] == 'raised'):
            parts = event['composition']
            exact = len(parts) == 1 and parts[0]['text'] == selected[0]['answer']
            item['attempts'].append({'blocking_stage': 'llm_output_guard',
                'reason': event['reason'], 'exact_source_text': exact, 'draft': event['draft'],
                'limitation': 'The real guard exposes only allow/deny; it does not provide a rationale.'})
    assert len(item['attempts']) == 2
    findings.append(item)
(ROOT / 'findings.json').write_text(json.dumps(findings, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'reproduced': 5, 'scope_validation': sum(x['saved_reason'] == 'answer_validation_failed' for x in findings),
    'output_guard': sum(x['saved_reason'] == 'security_output_rejected' for x in findings),
    'all_had_evidence': all(x['retrieved_ids'] for x in findings),
    'all_passed_admission': all(x['admission_allowed'] for x in findings)}))
