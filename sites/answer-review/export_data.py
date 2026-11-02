"""Export the frozen 460-case evaluation for the local answer-review UI.

Only display fields are exported. Original answer/source strings are retained
verbatim; private traces, model paths and CLI metadata are never included.
"""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "evaluation/expanded_460_2026-11-02"
OUT = Path(__file__).resolve().parent / "dist/data.json"
SYSTEMS = {"original": "pre-security", "current": "candidate-e37"}
PIECES = [
    {"id": "die-forelle", "title": "Die Forelle", "subtitle": "송어"},
    {"id": "in-flowery-clouds", "title": "In Flowery Clouds", "subtitle": "꽃구름 속에"},
    {"id": "la-capinera", "title": "La Capinera", "subtitle": "검은 머리 방울새"},
    {"id": "nella-fantasia", "title": "Nella fantasia", "subtitle": "넬라 판타지아"},
    {"id": "una-voce-poco-fa", "title": "Una voce poco fa", "subtitle": "방금 들린 그대 음성"},
]


def read(name):
    return json.loads((EVAL / name).read_text())


def rows(name):
    return [json.loads(line) for line in (EVAL / name).read_text().splitlines()]


def project(record, keys):
    return {key: record.get(key) for key in keys}


def build():
    cases = read("cases.json")["cases"]
    assert len(cases) == len({c["id"] for c in cases}) == 460
    mapping = {r["key"]: r["review_id"] for r in read("judge/mapping.json")}
    assessments = {r["id"]: r["assessment"] for r in rows("judge/assessments.jsonl")}
    manual = {r["key"]: r["review"] for r in read("manual-review.json")["entries"]}
    results, metrics = {}, {}
    for system in SYSTEMS.values():
        results[system] = {r["case_id"]: r for r in rows(system + ".jsonl")}
        config_id = read(f"bertscore-{system}/summary.json")[0]["config_id"]
        metrics[system] = {r["id"]: r for r in read(f"bertscore-{system}/{config_id}.json")["cases"]}
    exported = []
    for index, case in enumerate(cases):
        item = project(case, ["id", "question", "piece_id", "measure_range", "family", "split", "source_group_id"])
        item["index"] = index + 1
        item["references"] = [project(ref, ["knowledge_unit_id", "text", "measure_ranges", "reference_role"])
                              for ref in case["reference_authority"]["items"]]
        item["versions"] = {}
        for label, system in SYSTEMS.items():
            result = results[system][case["id"]]
            assert result["status"] == "response"
            response = result["response"]
            key = system + ":" + case["id"]
            assessment = assessments[mapping[key]]
            metric = metrics[system][case["id"]]
            item["versions"][label] = {
                "answer": response["answer"],
                "latency_seconds": result["elapsed_seconds"],
                "evidence": [project(e, ["id", "text", "measure_ranges", "evidence_type", "generation_role"])
                             for e in response["evidence"]],
                "bertscore": {
                    "scored": metric["scored"],
                    "f1": metric["f1"] if metric["scored"] else None,
                    "reason": metric["unscorable_reason"],
                    "candidate_tokens": metric["candidate_tokens"],
                },
                "judge": project(assessment, ["source_fidelity", "factual_correctness", "naturalness",
                                               "naturalness_rationale", "rationale", "completeness",
                                               "relevance", "attribution", "refusal"]),
                "secondary_note": manual.get(key),
            }
            assert item["versions"][label]["answer"] == response["answer"]
        exported.append(item)
    summary = read("review/summary.json")
    payload = {
        "schema_version": 1,
        "evaluation_id": "expanded_460_2026-11-02",
        "evaluation_date": "2026-11-02",
        "judge_model": "Claude Opus 5.5",
        "pieces": PIECES,
        "summary": {label: summary["variants"][system] for label, system in SYSTEMS.items()},
        "naturalness_comparison": summary["paired"]["naturalness_both_useful"],
        "cases": exported,
        "source_hashes": {name: hashlib.sha256((EVAL / name).read_bytes()).hexdigest()
                          for name in ["cases.json", "candidate-e37.jsonl", "pre-security.jsonl", "judge/assessments.jsonl"]},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({"cases": len(exported), "answers": len(exported) * 2, "bytes": OUT.stat().st_size}))


if __name__ == "__main__":
    build()
