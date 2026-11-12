"""Freeze a new synthesis candidate without altering historical evaluation data."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evaluation/semantic_generation_2026-11-02"
E37 = ROOT / "evaluation/ku_background_2026-10-31/runtime/followup-e37/snapshots/hardened"
EXPANDED = ROOT / "evaluation/expanded_460_2026-11-02"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(path, data):
    text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text() != text:
        raise ValueError(f"Frozen artifact differs: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--model-verification", type=Path,
                        help="Optional pinned, checksum-verified local model comparison artifact")
    parser.add_argument("--llm-settings", type=Path,
                        help="Optional explicit inference settings for an isolated candidate")
    args = parser.parse_args()
    if not re.fullmatch(r"e[0-9]+", args.candidate):
        raise ValueError("Use an explicit version such as e38")
    candidate = args.candidate
    target = OUT / "runtime" / candidate
    if target.exists():
        raise ValueError("Snapshot exists; resume its existing contract or use a new candidate")
    expanded = read(EXPANDED / "cases.json")["cases"]
    by_id = {c["id"]: c for c in expanded}
    history = read(ROOT / "evaluation/ku_background_2026-10-31/all_cases.json")["cases"]
    selected_ids = list(dict.fromkeys(r["key"].split(":", 1)[1] for r in read(EXPANDED / "manual-review.json")["entries"]))
    selected = [by_id[cid] for cid in selected_ids]
    for cid in ["benign-17-ko", "benign-17-en", "background-breath-en", "background-transpose-en", "benign-19-en", "benign-14-en"]:
        selected.append(next(c for c in history if c["id"] == cid))
    for kind, count in [("direct_attack", 2), ("indirect_attack", 2), ("out_of_scope", 1)]:
        for lang in ["ko", "en"]:
            selected.extend([c for c in history if c["kind"] == kind and c["language"] == lang][:count])
    assert len(selected) == len({c["id"] for c in selected})
    freeze(OUT / "probe-cases.json", {"suite_type": "semantic_synthesis_diagnostic", "cases": selected})
    freeze(OUT / "expanded-cases.json", read(EXPANDED / "cases.json"))
    freeze(OUT / "expert-cases.json", read(ROOT / "evaluation/ku_background_2026-10-31/expert_cases.json"))
    freeze(OUT / "safety-cases.json", {"suite_type": "bilingual_generation_safety", "cases": [c for c in history if not c["id"].startswith("expert-")]})
    freeze(OUT / "metric-length-amendment.json", read(EXPANDED / "metric-length-amendment.json"))
    shutil.copytree(ROOT / "soprano_qa", target / "soprano_qa", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(E37 / "data", target / "data", ignore=shutil.ignore_patterns("*.lock"))
    settings = read(E37 / "settings.json")
    settings["llm"] = read(ROOT / "config/settings.json")["llm"]
    if args.llm_settings:
        settings["llm"] = read(args.llm_settings)
    model_verification = read(args.model_verification) if args.model_verification else None
    if model_verification:
        path = Path(model_verification["path"])
        if (model_verification.get("verified") is not True or not path.is_file()
                or path.stat().st_size != model_verification["size"]):
            raise ValueError("Missing or unverified model comparison checkpoint")
        settings["model_path"] = str(path.resolve())
        settings["model_filename"] = path.name
        settings["model_repo_id"] = model_verification["repo_id"]
        settings["model_revision"] = model_verification["revision"]
        settings["model_sha256"] = model_verification["sha256"]
    for key in ["corpus_path", "stats_path", "feature_overrides_path", "embedding_cache_path"]:
        settings[key] = str(target / "data" / Path(settings[key]).name)
    freeze(target / "settings.json", settings)
    source_hashes = {str(p.relative_to(target)): sha(p) for p in (target / "soprano_qa").glob("*.py")}
    freeze(OUT / f"{candidate}-provenance.json", {
        "created_at": datetime.now(timezone.utc).isoformat(), "candidate": candidate,
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "parent_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_files_sha256": source_hashes, "settings": settings,
        "model_comparison": model_verification,
        "settings_sha256": sha(target / "settings.json"), "corpus_sha256": sha(target / "data/corpus.json"),
        "probe_cases_sha256": sha(OUT / "probe-cases.json"),
        "expanded_cases_sha256": sha(OUT / "expanded-cases.json"),
        "metric_settings": "evaluation/ku_background_2026-10-31/bertscore-development-b/selected_settings.json",
        "objective": "LLM-authored question-focused answers. Meaning/usefulness/naturalness first; fixed expert BERTScore >=0.90 is a floor, not an optimization target.",
        "diagnostic_selection": "Previously reviewed failure/meaning cases plus bilingual general questions and fixed first attack cases; reused development evidence, not a new holdout.",
        "prior_e37_reinterpretation": "High overlap from verbatim KU rendering does not establish generative RAG quality. Preserve E37 artifacts as historical quotation-mode results.",
        "no_usage_stop": True, "notifier_used": False,
    })
    print(json.dumps({"candidate": candidate, "snapshot": str(target.relative_to(ROOT)), "probe_cases": len(selected), "source_modules": len(source_hashes)}))


if __name__ == "__main__":
    main()
