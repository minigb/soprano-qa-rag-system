"""Refresh only the three existing local static demo mirrors."""
from pathlib import Path
import hashlib
import json

SITE = Path(__file__).resolve().parent
RUN = SITE.parents[1] / 'evaluation/gemini_demo_2026-11-07'
ALLOWED = Path('/home/minhee/.codex/worktrees/rag-background-knowledge').resolve()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    before = json.loads((RUN / 'before.json').read_text())
    sources = {p.name: p for p in (SITE / 'dist').iterdir() if p.is_file()}
    assert json.loads(sources['evaluations.json'].read_text())['default'] == 'gemini-20261107'
    expected = {name: sha(path) for name, path in sources.items()}
    destinations = []
    # Check every destination before writing any: never overwrite unrelated edits.
    for name, record in before['mirrors'].items():
        dest = Path(name)
        assert dest.resolve().is_relative_to(ALLOWED)
        assert dest.is_dir() and not dest.is_symlink()
        assert {p.name for p in dest.iterdir() if p.is_file()} <= set(sources)
        for asset, old_hash in record['files'].items():
            target = dest / asset
            assert not target.is_symlink()
            assert sha(target) in {old_hash, expected[asset]}, str(target)
        for asset in set(sources) - set(record['files']):
            target = dest / asset
            if target.exists():
                assert not target.is_symlink() and sha(target) == expected[asset]
        destinations.append(dest)
    for dest in destinations:
        # Make the new data available before activating the new catalog entry.
        for name in sorted(sources, key=lambda name: name == 'evaluations.json'):
            temp = dest / ('.' + name + '.gemini-tmp')
            assert not temp.exists()
            temp.write_bytes(sources[name].read_bytes())
            temp.replace(dest / name)
        assert {p.name: sha(p) for p in dest.iterdir() if p.is_file()} == expected
    report = {'verified': True, 'default': 'gemini-20261107', 'files': expected,
              'destinations': [str(p) for p in destinations],
              'scope': 'existing static demo assets only; no backend or main checkout changes'}
    (RUN / 'mirror.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'mirrors': len(destinations), 'verified': True, 'files_per_mirror': len(sources)}))


if __name__ == '__main__':
    main()
