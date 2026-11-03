"""Refresh only existing worktree demo assets, preserving backend and main checkouts."""
from pathlib import Path
import hashlib,json
SITE=Path(__file__).resolve().parent
RUN=SITE.parents[1]/'evaluation/latest_demo_2026-11-03'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    before=json.loads((RUN/'demo-mirror-before.json').read_text());dist=SITE/'dist'
    assert json.loads((dist/'evaluations.json').read_text())['default']=='latest-14b-20261103'
    assert (RUN/'assessment-complete.json').exists()
    docs=json.loads((RUN/'demo-documentation-before.json').read_text())
    for name,digest in docs.items():assert sha(Path(name))==digest, 'Existing demo documentation changed: '+name
    destinations=[]
    for name,files in before.items():
        target=Path(name)
        assert target.resolve().is_relative_to('/home/minhee/.codex/worktrees/rag-background-knowledge')
        for relative,digest in files.items():
            assert sha(target/relative)==digest, 'Existing demo edited since capture: '+str(target/relative)
        destinations.append(target)
    files=sorted((p for p in dist.iterdir() if p.is_file()),key=lambda p:p.name=='evaluations.json')
    for target in destinations:
        for source in files:
            dest=target/source.name;temp=dest.with_suffix(dest.suffix+'.tmp')
            temp.write_bytes(source.read_bytes());temp.replace(dest)
            assert sha(dest)==sha(source)
    for name in docs:Path(name).write_bytes((SITE/'README.md').read_bytes())
    (RUN/'demo-mirror-complete.json').write_text(json.dumps({'destinations':[str(p) for p in destinations],
        'files':{p.name:sha(p) for p in files},'documentation':list(docs),
        'scope':'Static comparison assets and their serving documentation only. Backend source/config, initial answers, main checkouts and dataset unchanged.'},indent=2)+'\n')
    print(json.dumps({'mirrored_static_demos':len(destinations),'backend_changed':False}))
if __name__=='__main__':main()
