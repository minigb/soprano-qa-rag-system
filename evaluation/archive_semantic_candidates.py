"""Preserve exact experimental source changes without committing runtime data."""
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evaluation/semantic_generation_2026-11-02"


def main():
    target = OUT / "diagnostics/source-patches"
    target.mkdir(parents=True, exist_ok=True)
    for provenance in sorted(OUT.glob("e*-provenance.json")):
        record = json.loads(provenance.read_text())
        snapshot = OUT / "runtime" / record["candidate"]
        if not snapshot.exists():
            continue
        base = record["parent_commit"]
        names = subprocess.check_output(["git", "ls-tree", "--name-only", "-r", base, "soprano_qa"], cwd=ROOT, text=True).splitlines()
        original = {name: subprocess.check_output(["git", "show", f"{base}:{name}"], cwd=ROOT).decode()
                    for name in names if name.endswith(".py")}
        updated = {name: (snapshot / name).read_text() for name in record["source_files_sha256"]}
        patch = ""
        for name in sorted(original.keys() | updated.keys()):
            patch += "".join(difflib.unified_diff(
                original.get(name, "").splitlines(keepends=True), updated.get(name, "").splitlines(keepends=True),
                fromfile="a/" + name if name in original else "/dev/null",
                tofile="b/" + name if name in updated else "/dev/null"))
        path = target / (record["candidate"] + ".patch")
        if path.exists() and path.read_text() != patch:
            raise ValueError("Frozen candidate changed")
        path.write_text(patch)
        with tempfile.TemporaryDirectory(prefix="soprano-source-reproduction-") as folder:
            directory = Path(folder)
            for name, text in original.items():
                dest = directory / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(text)
            subprocess.run(["git", "apply", str(path.resolve())], cwd=directory, check=True)
            actual = {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (directory / "soprano_qa").glob("*.py")}
            assert actual == record["source_files_sha256"], record["candidate"]
        print(record["candidate"], len(patch.encode()), "bytes; all source hashes reproduced")


if __name__ == "__main__":
    main()
