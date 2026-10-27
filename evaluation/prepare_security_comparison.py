#!/usr/bin/env python3
"""Freeze original/current source and equivalent isolated evaluation settings."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from soprano_qa.corpus import corpus_input_fingerprint
from soprano_qa.settings import load_settings


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--original-ref", default="HEAD")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "snapshots").exists():
        raise SystemExit("Snapshots already exist; choose a fresh output directory")
    ref = subprocess.check_output(
        ["git", "rev-parse", args.original_ref], cwd=ROOT, text=True,
    ).strip()
    settings = load_settings(use_legacy_dataset_env=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "original_commit": ref,
        "hardened_base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip(),
        "corpus_sha256": sha256(Path(settings["corpus_path"])),
        "corpus_input_fingerprint": corpus_input_fingerprint(settings),
        "python": sys.version,
        "llama_cpp_python_version": importlib.metadata.version("llama-cpp-python"),
        "gpu": subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ], text=True).strip(),
        "models": {},
        "variants": {},
        "protocol": {
            "seed": 20261025,
            "repeats": 2,
            "primary_security_quality_repetition": 0,
            "latency": "Two repetitions; report per-case mean and repetition-specific summaries",
            "execution": "Preloaded workers; one untimed warmup each; serial paired random order",
            "indirect": "Forced exposure after clean retrieval; not retrieval-poisoning success",
            "evaluation_role": "Authored development evaluation, not an external held-out benchmark",
            "freeze": "No production changes or case-dependent tuning during measured runs",
        },
    }
    for key in ("model_path", "embedding_model_path"):
        path = Path(settings[key]).resolve()
        manifest["models"][key] = {
            "path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path),
        }
    for variant in ("original", "hardened"):
        snapshot = output / "snapshots" / variant
        (snapshot / "soprano_qa").mkdir(parents=True)
        if variant == "original":
            files = subprocess.check_output(
                ["git", "ls-tree", "-r", "--name-only", ref, "soprano_qa"],
                cwd=ROOT, text=True,
            ).splitlines()
            for name in files:
                path = snapshot / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(subprocess.check_output(["git", "show", f"{ref}:{name}"], cwd=ROOT))
        else:
            for source in sorted((ROOT / "soprano_qa").glob("*.py")):
                shutil.copy2(source, snapshot / "soprano_qa" / source.name)
        local = json.loads(json.dumps(settings))
        for key, filename in (
            ("corpus_path", "corpus.json"), ("stats_path", "corpus_stats.json"),
            ("feature_overrides_path", "feature_overrides.json"),
            ("embedding_cache_path", "embedding-cache.json"),
        ):
            destination = snapshot / "data" / filename
            destination.parent.mkdir(exist_ok=True)
            source = Path(settings[key])
            if source.exists():
                shutil.copy2(source, destination)
            local[key] = str(destination)
        coaching_cache = Path(settings["embedding_cache_path"].replace(".json", "") + "-web-coaching.json")
        if coaching_cache.exists():
            shutil.copy2(coaching_cache, snapshot / "data" / "embedding-cache-web-coaching.json")
        for key in ("model_path", "embedding_model_path", "dataset_root"):
            local[key] = str(Path(local[key]).resolve())
        settings_path = snapshot / "settings.json"
        settings_path.write_text(json.dumps(local, ensure_ascii=False, indent=2) + "\n")
        file_hashes = {
            str(path.relative_to(snapshot)): sha256(path)
            for path in sorted((snapshot / "soprano_qa").glob("*.py"))
        }
        manifest["variants"][variant] = {
            "root": str(snapshot), "settings": str(settings_path), "files": file_hashes,
            "source_sha256": hashlib.sha256(json.dumps(file_hashes, sort_keys=True).encode()).hexdigest(),
        }
    (output / "provenance.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (output / "hardening.patch").write_bytes(subprocess.check_output(
        ["git", "diff", ref, "--", "soprano_qa"], cwd=ROOT,
    ))
    print(json.dumps({"output": str(output), "original_commit": ref, "corpus_sha256": manifest["corpus_sha256"]}))


if __name__ == "__main__":
    main()
