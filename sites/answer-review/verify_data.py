"""Verify visible data against independently loaded frozen result artifacts."""
from collections import Counter
import hashlib
import json
from pathlib import Path

SITE = Path(__file__).resolve().parent
EVAL = SITE.parents[1] / "evaluation/expanded_460_2026-11-02"


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    data = read(SITE / "dist/data.json")
    frozen = {c["id"]: c for c in read(EVAL / "cases.json")["cases"]}
    mapping = {r["key"]: r["review_id"] for r in read(EVAL / "judge/mapping.json")}
    judgments = {r["id"]: r["assessment"] for r in rows(EVAL / "judge/assessments.jsonl")}
    assert len(data["cases"]) == 460
    assert {c["id"] for c in data["cases"]} == set(frozen)
    counts = Counter(c["family"] for c in data["cases"])
    assert counts == {"original": 105, "paraphrase": 306, "ku_derived": 49}
    for name, digest in data["source_hashes"].items():
        assert hashlib.sha256((EVAL / name).read_bytes()).hexdigest() == digest
    for label, system in [("original", "pre-security"), ("current", "candidate-e37")]:
        raw = {r["case_id"]: r for r in rows(EVAL / (system + ".jsonl"))}
        metric_id = read(EVAL / f"bertscore-{system}/summary.json")[0]["config_id"]
        scores = {c["id"]: c for c in read(EVAL / f"bertscore-{system}/{metric_id}.json")["cases"]}
        for c in data["cases"]:
            original = frozen[c["id"]]
            assert c["question"] == original["question"]
            assert c["measure_range"] == original["measure_range"]
            assert [r["text"] for r in c["references"]] == [r["text"] for r in original["reference_authority"]["items"]]
            v = c["versions"][label]; r = raw[c["id"]]; score = scores[c["id"]]
            assert v["answer"] == r["response"]["answer"]
            assert v["latency_seconds"] == r["elapsed_seconds"]
            assert [e["text"] for e in v["evidence"]] == [e["text"] for e in r["response"]["evidence"]]
            assert v["bertscore"]["f1"] == (score["f1"] if score["scored"] else None)
            assessment = judgments[mapping[system + ":" + c["id"]]]
            assert all(value == assessment[key] for key, value in v["judge"].items())
    assert sum(not c["versions"]["current"]["bertscore"]["scored"] for c in data["cases"]) == 2
    assert sum(c["versions"]["current"]["judge"]["naturalness"] < c["versions"]["original"]["judge"]["naturalness"] for c in data["cases"]) == 272
    serialized = (SITE / "dist/data.json").read_text()
    assert not any(term in serialized for term in ["/home/minhee/", "private_trace_attempt_id", "batch_usage", "model_path", "contract_id"])
    print(json.dumps({"status": "passed", "cases": 460, "full_answers_verified": 920,
                      "expert_references": "exact", "judge_and_score_mapping": "exact",
                      "unscorable_scores_are_null": 2, "private_execution_metadata": "excluded"}))


if __name__ == "__main__":
    main()
