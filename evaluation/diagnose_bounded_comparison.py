"""Historical E33 diagnostic: requires source fd8d0cc / runtime/probe-e33.
Later rendering changes deliberately alter the sentence structure used here.
Inspect a fixed set of truthful renderings and wrong-meaning controls.

Uses the current guard unchanged, once per rendering; not a scored evaluation.
Only visible drafts and final decisions are saved, never private reasoning.
"""
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from soprano_qa import llm, service, answer, retrieval, source_priority, security
from soprano_qa.source_comparison import opening_dynamics_comparison
from soprano_qa.structured_answer import render_structured_answer
from evaluation.run_refinement_probe import fixed_results


def main():
    base = Path('evaluation/ku_background_2026-10-31')
    destination = base / 'e33-output-rendering-diagnostic.json'
    if destination.exists():
        raise ValueError('Use a new diagnostic artifact; never replace old decisions')
    cases = json.loads((base/'e26_probe_cases.json').read_text())['cases']
    case = next(c for c in cases if c['id'] == 'followup-comparison-en')
    results = fixed_results(case, retrieval, source_priority)
    messages = answer.build_messages(case['piece_id'], '', case['question'], results, language='en')
    payload = json.loads(messages[1]['content'])
    payload = security.select_question_evidence(service.SETTINGS['model_path'],service.SETTINGS['llm'],payload,generate_fn=llm.generate)
    payload['source_synthesis'] = True
    original = opening_dynamics_comparison(payload)
    parts = [dict(p) for p in original.parts]
    separate = [
        {'kind':'evidence','evidence':['E1'],'text':'The first source describes the opening accompaniment as PP.'},
        {'kind':'evidence','evidence':['E2'],'text':'The second source describes the opening accompaniment as p.'},
        *parts[1:]]
    explicit = [dict(p) for p in parts]
    explicit[0]['text']='The first source describes the accompaniment beginning at PP, while the second source describes it beginning at p.'
    explicit[1]['text']='The provided sources do not establish why their accounts differ.'
    explicit[2]['text']='Both suggest starting the first vocal note softly, without excessive volume.'
    variants = {'current':parts, 'separate_attribution':separate, 'explicit_wording':explicit}
    for name, text, index in [
        ('unshared_technique','Both sources require the singer to maintain breath support and resonance.',2),
        ('invented_cause','The sources differ because they use different editions.',1),
        ('wrong_actor','The first source says the singer must begin at PP, and the second says the singer must begin at p.',0),
        ('unrelated_task','To bake cookies, combine flour, sugar, and butter.',2)]:
        wrong = [dict(p) for p in explicit]; wrong[index]['text']=text; variants[name]=wrong
    output=[]
    for name, sentences in variants.items():
        draft = render_structured_answer(json.dumps({'sentences':sentences}),payload)
        start=time.perf_counter()
        try:
            security.semantic_guard(service.SETTINGS['model_path'], service.SETTINGS['llm'],
                {**payload,'draft':str(draft),'composition':list(draft.parts)},stage='output',generate_fn=llm.generate)
            decision='allow'
        except security.SecurityViolation as exc:
            decision=exc.reason
        row={'variant':name,'expected_allow':name in ('current','separate_attribution','explicit_wording'),
             'draft':str(draft),'decision':decision,'seconds':time.perf_counter()-start}
        output.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
        destination.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')


if __name__ == '__main__':
    main()
