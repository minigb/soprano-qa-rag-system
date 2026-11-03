"""Rebuild the ignored namespace adapter from the byte-exact committed snapshot."""
from pathlib import Path
import hashlib, json
ROOT=Path(__file__).resolve().parent
manifest=json.loads((ROOT/'manifest.json').read_text())
for relative,expected in manifest['execution_sources'].items():
    target=ROOT/relative
    original=ROOT/'snapshots/current/soprano_qa'/target.name
    adapted=original.read_text().replace('soprano_qa','current_qa').encode()
    assert hashlib.sha256(adapted).hexdigest()==expected, 'Adapter/source mismatch: '+relative
    if target.exists():
        assert target.read_bytes()==adapted, 'Existing runtime was modified: '+relative
    else:
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(adapted)
print('Namespace adapter verified; protected sources untouched.')
