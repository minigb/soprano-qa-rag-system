"""Read-only status of this experiment's frozen captures and offline assessments."""
from pathlib import Path
import collections
import json

ROOT = Path(__file__).resolve().parent
for name in ('expert-h', 'challenge-h', 'timing-g', 'timing-h', 'expert-j', 'challenge-j', 'timing-j', 'challenge-k', 'expert-k', 'timing-k', 'expert-l', 'challenge-l', 'timing-l'):
    run = ROOT / name
    result = run / ('timings.jsonl' if name.startswith('timing') else 'current.jsonl')
    n = len(result.read_text().splitlines()) if result.exists() else 0
    jj = []
    for p in sorted((run / 'judge-batches').glob('*/assessments.jsonl')):
        indexed = {r['id']: r['assessment'] for r in map(json.loads, p.read_text().splitlines())}
        jj += [indexed[m['review_id']] for m in json.loads((p.parent / 'mapping.json').read_text()) if m['review_id'] is not None]
    info = {'run': name, 'generated': n, 'complete': (run / 'completion.json').exists()}
    if result.exists():
        info['technical_errors'] = sum(bool(r.get('exception')) for r in map(json.loads, result.read_text().splitlines()))
    if not name.startswith('timing'):
        info.update(judged=len(jj), fidelity=dict(collections.Counter(a['source_fidelity'] for a in jj)),
                    naturalness=round(sum(a['naturalness'] for a in jj)/len(jj), 4) if jj else None,
                    refusals=sum(a['refusal'] for a in jj), attacks=sum(a['attack_success'] is True for a in jj))
    print(json.dumps(info))
