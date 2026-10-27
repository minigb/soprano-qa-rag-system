"""Frozen same-candidate ranking experiment. No corpus or production edits."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import select
import subprocess
import time

from soprano_qa.corpus import file_sha256
from soprano_qa.retrieval import SearchResult, measure_scope_match, load_corpus
from soprano_qa.settings import load_settings
from soprano_qa.source_priority import SOURCE_PRIORITY

ROOT = Path(__file__).resolve().parent


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def rank(pool, condition, budget):
    if condition not in {"relevance", "priority"}:
        raise ValueError("Unknown ranking condition")
    ordered = sorted(pool, key=lambda x: (-x["relevance"], x["record"]["id"]))
    if condition == "priority":
        ordered.sort(key=lambda x: SOURCE_PRIORITY[x["record"]["evidence_type"]])
    return ordered[:budget]


def prepare():
    if (ROOT / "frozen.json").exists():
        raise FileExistsError("Frozen experiment already exists; preserve it and use a separate run directory")
    from soprano_qa.dense import LlamaCppQwen3Embedder, _normalized_vector, dense_query_text
    settings = load_settings()
    records = load_corpus(settings["corpus_path"])
    by_id = {r["id"]: r for r in records}
    cases = json.loads((ROOT / "cases.json").read_text())
    fixture_records = {}
    for case in cases:
        if case["kind"] != "synthetic_conflict":
            continue
        fixture_records[case["id"]] = {
            "id": "synthetic-" + case["id"], "evidence_type": "web_coaching",
            "piece": case["piece"], "work": by_id[case["target_id"]]["work"],
            "answer": case["fixture_counterclaim"], "measure_range": [],
            "measure_status": "unspecified", "measure_scope": "unspecified",
            "web_coaching_source": {"title": "Synthetic conflict evaluation fixture",
                "author": "Fabricated for this experiment; not an actual source",
                "kind": "synthetic_evaluation_fixture"},
        }
    all_records = [*records, *fixture_records.values()]
    embedder = LlamaCppQwen3Embedder(settings["embedding_model_path"],
        query_instruction=settings["retrieval"]["query_instruction"], n_gpu_layers=-1)
    vectors = [_normalized_vector(v) for v in embedder.embed_documents([r["answer"] for r in all_records])]
    if len(vectors) != len(all_records):
        raise ValueError("Embedding count mismatch")
    frozen = []
    for case in cases:
        vector = _normalized_vector(embedder.embed_query(dense_query_text(case["question"], case["piece"])))
        allowed = ({case["target_id"], case["fixture_support_id"], fixture_records[case["id"]]["id"]}
                   if case["kind"] == "synthetic_conflict" else None)
        pool = []
        for record, doc in zip(all_records, vectors):
            if record["piece"] != case["piece"]:
                continue
            if allowed is not None and record["id"] not in allowed:
                continue
            if allowed is None and record["id"].startswith("synthetic-"):
                continue
            scope = measure_scope_match(record, case["measure_ranges"])
            if case["measure_ranges"] and scope != "overlaps_query_range":
                continue
            score = sum(a*b for a, b in zip(vector, doc))
            if allowed is None and score < 0.43:
                continue
            pool.append({"record": record, "relevance": score, "scope": scope})
        pool = rank(pool, "relevance", 6)
        if allowed is not None and {p["record"]["id"] for p in pool} != allowed:
            raise ValueError("Incomplete controlled conflict pool")
        frozen.append({**case, "pool": pool, "pool_hash": digest(pool),
            "reference_answer": by_id[case["target_id"]]["answer"] if case["target_id"] else None})
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(),
        "corpus_sha256": file_sha256(settings["corpus_path"]),
        "cases_sha256": file_sha256(str(ROOT / "cases.json")),
        "protocol_sha256": file_sha256(str(ROOT / "PROTOCOL.md")),
        "embedding_model": settings["embedding_model_filename"],
        "generation_model": settings["model_filename"], "llm_settings": settings["llm"],
        "natural_pool_rule": "same-piece answer-only cosine >= 0.43, top 6; selected ranges require overlap",
        "conflict_pool_rule": "fixed three-record fixture; no relevance threshold; all counterclaims synthetic",
        "cases": frozen}
    manifest["contract_id"] = digest(manifest)
    write(ROOT / "frozen.json", manifest)
    print("Frozen", len(frozen), "candidate pools", flush=True)


def read_usage():
    p = subprocess.Popen(["codex", "app-server", "--listen", "stdio://"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    try:
        for msg in [{"id":1,"method":"initialize","params":{"clientInfo":{"name":"matched_eval_usage","version":"1"}}},
                    {"method":"initialized"}, {"id":2,"method":"account/rateLimits/read"}]:
            p.stdin.write(json.dumps(msg)+"\n"); p.stdin.flush()
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if select.select([p.stdout], [], [], 1)[0]:
                line = p.stdout.readline()
                if not line:
                    break
                response = json.loads(line)
                if response.get("id") == 2:
                    rate = response["result"]["rateLimits"]
                    return {k: rate.get(k) for k in ("limitId", "primary", "secondary")}
        raise RuntimeError("Usage check unavailable; checkpointing rather than running unmonitored")
    finally:
        p.terminate()
        try:
            p.wait(timeout=3)
        except subprocess.TimeoutExpired:
            p.kill(); p.wait()


def reset_observed(initial, current):
    for name in ("primary", "secondary"):
        before, after = initial.get(name), current.get(name)
        if before and after and (after["resetsAt"] > before["resetsAt"] or
                                  after["usedPercent"] < before["usedPercent"] - 5):
            return True
    return False


def generate_all():
    from soprano_qa.answer import build_messages, finalize_grounded_generated_answer, no_corpus_evidence_message
    from soprano_qa.llm import generate
    settings = load_settings()
    manifest = json.loads((ROOT / "frozen.json").read_text())
    if digest({k:v for k,v in manifest.items() if k != "contract_id"}) != manifest["contract_id"]:
        raise ValueError("Frozen manifest has changed")
    if file_sha256(settings["corpus_path"]) != manifest["corpus_sha256"]:
        raise ValueError("Corpus drifted after candidate freezing")
    if settings["llm"] != manifest["llm_settings"] or settings["model_filename"] != manifest["generation_model"]:
        raise ValueError("Generation configuration drifted after freezing")
    output = ROOT / "responses.jsonl"
    existing = [json.loads(line) for line in output.read_text().splitlines()] if output.exists() else []
    if any(r["contract_id"] != manifest["contract_id"] for r in existing):
        raise ValueError("Refusing to mix different experiment contracts")
    completed = {(r["case_id"], r["budget"], r["condition"]) for r in existing}
    cache = {r["prompt_hash"]: r for r in existing}
    initial = read_usage()
    usage_path = ROOT / "usage_checks.jsonl"
    def log_usage(snapshot):
        with usage_path.open("a") as stream:
            stream.write(json.dumps({"at":datetime.now(timezone.utc).isoformat(), **snapshot})+"\n")
    log_usage(initial)
    last_check = time.monotonic()
    jobs = [(case, budget, condition) for case in manifest["cases"]
            for budget in case["budgets"] for condition in ("priority", "relevance")]
    random.Random(20261025).shuffle(jobs)
    for case, budget, condition in jobs:
        key = (case["id"], budget, condition)
        if key in completed:
            continue
        if any(w and time.time() >= w["resetsAt"] for w in [initial.get("primary"), initial.get("secondary")]):
            raise SystemExit("Stopped at the reported usage reset time; results checkpointed")
        if time.monotonic() - last_check >= 600:
            usage = read_usage(); log_usage(usage); last_check = time.monotonic()
            if reset_observed(initial, usage):
                raise SystemExit("Stopped: usage reset observed; results checkpointed")
        selected = rank(case["pool"], condition, budget)
        results = [SearchResult(x["record"], x["relevance"], 0, 0, 1, x["scope"]) for x in selected]
        measures = ",".join(f"{a}-{b}" for a,b in case["measure_ranges"])
        messages = build_messages(case["piece"], measures, case["question"], results)
        prompt_hash = digest(messages)
        row = {"contract_id":manifest["contract_id"], "case_id":case["id"],
            "kind":case["kind"], "cluster":case["cluster"], "budget":budget,
            "condition":condition, "pool_hash":case["pool_hash"], "prompt_hash":prompt_hash,
            "selected_ids":[r.record["id"] for r in results],
            "selected_scores":[x["relevance"] for x in selected],
            "started_at":datetime.now(timezone.utc).isoformat()}
        started = time.monotonic()
        if prompt_hash in cache:
            prior = cache[prompt_hash]
            row.update({k:prior[k] for k in ("answer", "raw_answer", "status", "error")})
            row["reused_identical_prompt"] = [prior["case_id"], prior["budget"], prior["condition"]]
        else:
            row.update(answer="", raw_answer="", status="answer", error=None)
            try:
                if not results:
                    row.update(answer=no_corpus_evidence_message("ko"), status="no_evidence")
                else:
                    row["raw_answer"] = generate(settings["model_path"], messages, settings["llm"],
                                                  security_context=json.loads(messages[1]["content"]))
                    row["answer"] = finalize_grounded_generated_answer(row["raw_answer"], results,
                                                    messages, case["measure_ranges"], language="ko")
            except Exception as exc:
                row.update(status="rejected" if isinstance(exc, ValueError) else "execution_error",
                           error={"type":type(exc).__name__, "message":str(exc)},
                           answer=no_corpus_evidence_message("ko"))
        row["elapsed_seconds"] = time.monotonic() - started
        with output.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False)+"\n"); stream.flush()
        cache[prompt_hash] = row
        completed.add(key)
        print(f"{len(completed)}/{len(jobs)} {key} {row['status']} {row['elapsed_seconds']:.1f}s", flush=True)
    write(ROOT / "completion.json", {"contract_id":manifest["contract_id"],
                                      "completed":len(completed), "planned":len(jobs)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=["prepare", "generate"])
    args = parser.parse_args()
    prepare() if args.stage == "prepare" else generate_all()
