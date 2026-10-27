"""Reproducible retrieval pilot; development questions are never indexed.

Run from repository root: python -m evaluation.source_priority.run
Use --hybrid for the configured local embedding model; no generation calls.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from soprano_qa.answer import retrieve_generation_evidence
from soprano_qa.corpus import corpus_input_fingerprint, file_sha256
from soprano_qa.dense import build_retrieval_index
from soprano_qa.retrieval import load_corpus
from soprano_qa.settings import load_settings
from soprano_qa.source_priority import SOURCE_PRIORITY, CONTEXT_ONLY


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--hybrid", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    settings = load_settings()
    settings["retrieval"]["mode"] = "hybrid" if args.hybrid else "lexical"
    settings["retrieval"]["source_priority"] = True
    records = load_corpus(settings["corpus_path"])
    index = build_retrieval_index(records, settings)
    questions = json.loads((root / "questions.json").read_text())
    rows = []
    for case in questions:
        kwargs = dict(query=case["question"], piece=case["piece"],
                      measure_ranges=case["measure_ranges"],
                      measures=",".join(f"{a}-{b}" for a, b in case["measure_ranges"]),
                      topic=None, top_k=6, web_coaching_supplement=4,
                      web_coaching_supplement_gate="dense_margin",
                      web_coaching_supplement_margin=0.05)
        conditions = {}
        for name, active in (("expert_web_baseline", index.base_index), ("source_priority", index)):
            results, _ = retrieve_generation_evidence(active, **kwargs)
            conditions[name] = [{"id": r.record["id"], "lineage": r.record["evidence_type"],
                                 "scope": r.scope_match} for r in results]
        rows.append({**case, "conditions": conditions})
    positives = [r for r in rows if r["expected_sumijo_id"]]
    negatives = [r for r in rows if not r["expected_sumijo_id"]]
    summary = {}
    for name in ("expert_web_baseline", "source_priority"):
        summary[name] = {
            "target_top1": sum(bool(r["conditions"][name]) and r["conditions"][name][0]["id"] == r["expected_sumijo_id"] for r in positives),
            "target_recall_at6": sum(any(h["id"] == r["expected_sumijo_id"] for h in r["conditions"][name]) for r in positives),
            "positive_questions": len(positives),
            "sumijo_false_positives": sum(any(h["lineage"] == "sumijo_feedback" for h in r["conditions"][name]) for r in negatives),
            "negative_questions": len(negatives),
            "authority_order_violations": sum(
                (tiers := [SOURCE_PRIORITY[h["lineage"]] for h in r["conditions"][name]
                           if h["scope"] not in CONTEXT_ONLY]) != sorted(tiers)
                for r in rows),
        }
    report = {"mode": settings["retrieval"]["mode"],
              "input_fingerprint": corpus_input_fingerprint(settings),
              "questions_sha256": file_sha256(str(root / "questions.json")),
              "source_priority_sha256": file_sha256(str(root.parents[1] / "soprano_qa" / "source_priority.py")),
              "sumijo_settings": settings.get("sumijo"),
              "embedding_model": settings.get("embedding_model_filename") if args.hybrid else None,
              "corpus_counts": dict(Counter(r["evidence_type"] for r in records)),
              "limitations": "Small manually authored development pilot, not an independent benchmark. Baseline lacks Sumi Jo data; recall difference measures source availability plus routing, not causal ranking quality. No answer-quality judgment or LLM generation.",
              "summary": summary, "cases": rows}
    path = root / (report["mode"] + "-results.json")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(path)


if __name__ == "__main__":
    main()
