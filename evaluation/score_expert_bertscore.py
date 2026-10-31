"""Compare reproducible Korean BERTScore settings on frozen expert references.

This scores the complete delivered answer. Unavailable/error responses receive
zero in effective means. Model/IDF selection is restricted to development data.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(args):
    import torch
    from bert_score import BERTScorer
    import bert_score
    import transformers
    torch.set_num_threads(4)
    cases = json.loads(args.cases.read_text())['cases']
    chosen = [c for c in cases if c['split'] == args.split]
    rows = [json.loads(s) for s in args.results.read_text().splitlines()]
    indexed = {r['case_id']: r for r in rows}
    if len(indexed) != len(rows):
        raise ValueError('Duplicate response: select exactly one repetition/variant')
    if any(c['id'] not in indexed for c in chosen):
        raise ValueError('Incomplete expert cohort; missing answers cannot be excluded')
    idf_references = [c['reference_answers'][0] for c in cases if c['split'] == 'development']
    if args.settings:
        configs = [json.loads(args.settings.read_text())['settings']]
    else:
        if args.split != 'development':
            raise ValueError('Select settings on development data before validation')
        configs = [{'model_type': model, 'num_layers': layer, 'idf': idf,
                    'rescale_with_baseline': False, 'use_fast_tokenizer': False, 'lang': 'ko'}
                   for model, layers in [('bert-base-multilingual-cased', [8, 9, 10]),
                                         ('xlm-roberta-base', [8, 9, 10]),
                                         ('xlm-roberta-large', [16, 17, 18])]
                   for layer in layers for idf in [False, True]]
    args.output.mkdir(parents=True, exist_ok=True)
    contract = {'cases_sha256': digest(args.cases), 'results_sha256': digest(args.results),
                'split': args.split, 'cases': len(chosen),
                'references': 'one concatenation of frozen applicable expert KUs per question',
                'candidate': 'complete displayed answer; errors/refusals zero in effective metrics',
                'idf_reference_sha256': hashlib.sha256(json.dumps(idf_references, ensure_ascii=False).encode()).hexdigest(),
                'packages': {p: importlib.metadata.version(p) for p in ['bert-score', 'transformers', 'torch']}}
    contract_path = args.output / 'contract.json'
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError('Scoring contract changed; use another output directory')
    contract_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + '\n')
    delivered = [c for c in chosen if indexed[c['id']]['status'] == 'response'
                 and (indexed[c['id']].get('response') or {}).get('generation_mode') != 'unavailable'
                 and (indexed[c['id']].get('response') or {}).get('answer', '').strip()]
    summaries = []
    for config in configs:
        key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
        path = args.output / (key + '.json')
        if path.exists():
            saved = json.loads(path.read_text()); summaries.append(saved['summary']); continue
        start = time.perf_counter()
        try:
            scorer = BERTScorer(**config, device=args.device, batch_size=16,
                                idf_sents=idf_references if config['idf'] else None)
            candidates = [indexed[c['id']]['response']['answer'] for c in delivered]
            references = [c['reference_answers'][0] for c in delivered]
            maximum = scorer._tokenizer.model_max_length
            lengths = [len(scorer._tokenizer.encode(text, truncation=False)) for text in candidates + references]
            raw = scorer.score(candidates, references, batch_size=16) if delivered else ([], [], [])
            metrics = {c['id']: {'precision': float(raw[0][i]), 'recall': float(raw[1][i]),
                                 'f1': float(raw[2][i])} for i, c in enumerate(delivered)}
            per_case = [{'id': c['id'], 'delivered': c['id'] in metrics,
                         **metrics.get(c['id'], {'precision': 0., 'recall': 0., 'f1': 0.})}
                        for c in chosen]
            # No Korean baseline is bundled for these models in the installed
            # official release. Never substitute an English or invented baseline.
            baseline = Path(bert_score.__file__).parent / 'rescale_baseline' / config['lang'] / (config['model_type'] + '.tsv')
            summary = {'config_id': key, 'settings': config, 'scorer_hash': scorer.hash,
                       'model_revision': getattr(scorer._model.config, '_commit_hash', None),
                       'count': len(chosen), 'delivered': len(delivered),
                       'truncated_texts': sum(n > maximum for n in lengths),
                       'max_input_tokens': max(lengths, default=0),
                       'tokenizer_max_length': maximum,
                       'effective': {k: statistics.mean(x[k] for x in per_case) for k in ['precision', 'recall', 'f1']},
                       'conditional': {k: statistics.mean(x[k] for x in metrics.values()) if metrics else None
                                       for k in ['precision', 'recall', 'f1']},
                       'rescaled': None,
                       'rescaling_note': 'No published Korean baseline bundled; raw score reported without fabricated normalization.' if not baseline.exists() else 'Published baseline available.',
                       'seconds': time.perf_counter() - start}
            if baseline.exists():
                import pandas as pd
                values = pd.read_csv(baseline).iloc[config['num_layers']].to_numpy()[1:].astype(float)
                # Apply the official component-wise affine rescaling only to
                # delivered answers; refusals still contribute zero.
                summary['rescaled'] = {k: sum((x[k]-float(values[i]))/(1-float(values[i])) for x in metrics.values())/len(chosen)
                                      for i, k in enumerate(['precision','recall','f1'])}
                summary['baseline_sha256'] = digest(baseline)
            path.write_text(json.dumps({'summary': summary, 'cases': per_case}, indent=2) + '\n')
            summaries.append(summary)
            print(json.dumps({'config': key, 'settings': config, 'effective_f1': summary['effective']['f1'],
                              'delivered': len(delivered), 'count': len(chosen)}), flush=True)
            del scorer, raw; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except Exception as exc:
            failure = {'config_id': key, 'settings': config, 'error': type(exc).__name__ + ': ' + str(exc)}
            (args.output / (key + '.failure.json')).write_text(json.dumps(failure, indent=2) + '\n')
            print(json.dumps(failure), flush=True)
    (args.output / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n')
    if not summaries:
        raise RuntimeError('Every metric configuration failed')
    if args.split == 'development' and not args.settings:
        eligible = [s for s in summaries if not s['truncated_texts']]
        if not eligible:
            raise RuntimeError('Every setting truncates an answer/reference; no valid full-answer selection')
        best = max(eligible, key=lambda x: x['effective']['f1'])
        selected = {'selection': 'highest effective development F1; no validation outputs used',
                    'settings': best['settings'], 'development_effective_f1': best['effective']['f1'],
                    'development_contract_sha256': digest(contract_path), 'model_revision': best['model_revision']}
        (args.output / 'selected_settings.json').write_text(json.dumps(selected, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--split', choices=['development', 'validation'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--settings', type=Path)
    parser.add_argument('--device', default='cuda')
    score(parser.parse_args())
