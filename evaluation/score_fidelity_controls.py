"""Diagnostic BERTScore on frozen Korean meaning-error/control answers.

This is not the expert validation cohort and never selects metric settings.
It shows whether a high similarity score can coexist with a known meaning error.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packets', type=Path, required=True)
    parser.add_argument('--key', type=Path, required=True)
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    from bert_score import BERTScorer
    torch.set_num_threads(4)
    packets = [p for p in json.loads(args.packets.read_text()) if p['language'] == 'ko']
    keys = {k['id']: k for k in json.loads(args.key.read_text())}
    selected = json.loads(args.settings.read_text())
    if selected['settings']['idf']:
        raise ValueError('This diagnostic requires the frozen no-IDF configuration')
    scorer = BERTScorer(**selected['settings'], device='cuda', batch_size=16)
    revision = getattr(scorer._model.config, '_commit_hash', None)
    if revision != selected['model_revision']:
        raise ValueError('Frozen encoder revision mismatch')
    answers = [p['answer'] for p in packets]
    references = ['\n'.join(p['reference_answers']) for p in packets]
    lengths = [len(scorer._tokenizer.encode(s, truncation=False)) for s in answers+references]
    if any(n > scorer._tokenizer.model_max_length for n in lengths):
        raise ValueError('Would truncate a diagnostic answer/reference')
    precision, recall, f1 = scorer.score(answers, references)
    rows = [{'id': p['id'], **keys[p['id']], 'answer': p['answer'], 'reference': references[i],
             'precision': float(precision[i]), 'recall': float(recall[i]), 'f1': float(f1[i])}
            for i, p in enumerate(packets)]
    result = {
        'diagnostic_only': True, 'note': 'Known historical Korean examples, outside the target cohort; no metric tuning.',
        'contract': {'packets_sha256': digest(args.packets), 'key_sha256': digest(args.key),
                     'settings_sha256': digest(args.settings), 'runner_sha256': digest(Path(__file__)),
                     'settings': selected['settings'], 'scorer_hash': scorer.hash, 'model_revision': revision,
                     'max_input_tokens': max(lengths), 'truncated_texts': 0,
                     'packages': {p: importlib.metadata.version(p) for p in ('bert-score', 'transformers', 'torch')}},
        'cases': rows,
        'known_material_errors_at_or_above_090': sum(r['expected_fidelity'] == 'material_drift' and r['f1'] >= .9 for r in rows),
        'known_material_errors': sum(r['expected_fidelity'] == 'material_drift' for r in rows),
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)+'\n'
    if args.output.exists() and args.output.read_text() != text:
        raise ValueError('Refusing to overwrite frozen diagnostic results')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    print(json.dumps({k: result[k] for k in ('known_material_errors', 'known_material_errors_at_or_above_090')}))


if __name__ == '__main__':
    main()
