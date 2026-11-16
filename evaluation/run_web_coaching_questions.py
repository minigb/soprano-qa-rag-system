#!/usr/bin/env python3
"""Run the web_coaching evaluation questions through the grounded pipeline.

Questions live in the dataset repository at
web_coaching/evaluation/questions/<piece>.json; each lists the web_coaching
unit IDs that answer it (empty for deliberately unanswerable questions).
For every question this records the selected evidence, the generated answer,
and gold-unit recall over the evidence, then summarizes per question type.

Usage:
  python evaluation/run_web_coaching_questions.py --output PATH [--top-k 6]
Select a pipeline configuration with SOPRANO_QA_SETTINGS.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from soprano_qa.service import SETTINGS, ask  # noqa: E402


def load_questions(dataset_root: Path) -> list[dict]:
    questions = []
    for path in sorted((dataset_root / "web_coaching" / "evaluation" / "questions").glob("*.json")):
        questions.extend(json.loads(path.read_text(encoding="utf-8")))
    if not questions:
        raise SystemExit("No web_coaching evaluation questions found under %s" % dataset_root)
    return questions


def score(question: dict, evidence_ids: list[str]) -> dict:
    gold = set(question["gold_unit_ids"])
    found = gold.intersection(evidence_ids)
    web_coaching = [i for i in evidence_ids if "-wc-" in i]
    return {
        "gold_count": len(gold),
        "gold_found": sorted(found),
        "gold_recall": round(len(found) / len(gold), 4) if gold else None,
        "gold_hit": bool(found) if gold else None,
        "web_coaching_evidence_count": len(web_coaching),
        "evidence_count": len(evidence_ids),
    }


def summarize(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups["all"].append(row)
        groups[row["question_type"]].append(row)
    summary = {}
    for name, items in sorted(groups.items()):
        answerable = [r for r in items if r["metrics"]["gold_recall"] is not None]
        summary[name] = {
            "questions": len(items),
            "mean_gold_recall": round(sum(r["metrics"]["gold_recall"] for r in answerable) / len(answerable), 4)
            if answerable else None,
            "gold_hit_rate": round(sum(r["metrics"]["gold_hit"] for r in answerable) / len(answerable), 4)
            if answerable else None,
            "mean_web_coaching_evidence": round(
                sum(r["metrics"]["web_coaching_evidence_count"] for r in items) / len(items), 3),
            "validation_warnings": sum(1 for r in items if r.get("generation_validation_warning")),
            "rejected": sum(1 for r in items if r.get("error")),
            "mean_seconds": round(sum(r["seconds"] for r in items) / len(items), 2),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--no-generate", action="store_true", help="retrieval only")
    args = parser.parse_args()

    dataset_root = Path(SETTINGS["dataset_root"])
    questions = load_questions(dataset_root)
    done = {}
    if args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        if previous.get("settings_path") == os.environ.get("SOPRANO_QA_SETTINGS", "") \
                and previous.get("generate") == (not args.no_generate):
            done = {row["qid"]: row for row in previous["results"]}

    rows = []
    for question in questions:
        if question["qid"] in done:
            rows.append(done[question["qid"]])
            continue
        started = time.time()
        result = ask(
            piece_id=question["piece_id"],
            question=question["question"],
            measure_range=None,
            generate=not args.no_generate,
            top_k=args.top_k,
        )
        evidence_ids = [item["id"] for item in result.get("evidence", [])]
        rows.append({
            **question,
            "evidence_ids": evidence_ids,
            "evidence_types": [item.get("evidence_type") for item in result.get("evidence", [])],
            "answer": result.get("answer"),
            "generation_mode": result.get("generation_mode"),
            "answer_basis": result.get("answer_basis"),
            "generation_validation_warning": result.get("generation_validation_warning"),
            "error": result.get("error"),
            "seconds": round(time.time() - started, 2),
            "metrics": score(question, evidence_ids),
        })
        print("%s recall=%s n_ev=%d %.1fs" % (
            question["qid"], rows[-1]["metrics"]["gold_recall"], len(evidence_ids),
            rows[-1]["seconds"]), flush=True)
        write(args, rows)
    write(args, rows)
    print(json.dumps(summarize(rows)["all"], ensure_ascii=False))


def write(args, rows):
    payload = {
        "settings_path": os.environ.get("SOPRANO_QA_SETTINGS", ""),
        "generate": not args.no_generate,
        "top_k": args.top_k,
        "summary": summarize(rows),
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(args.output)


if __name__ == "__main__":
    main()
