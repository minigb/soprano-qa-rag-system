"""Partition/assemble the 460 requests without generating expert requests twice.

The 105 expert requests are an exact request subset. Reference annotations differ
between suites and must be taken from the target 460-case set when judging it.
Generation rows and their original contracts are preserved, including failures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from run_joint_expanded_judge import frozen_write


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def request(case):
    # Public run_case inputs plus injection/fixed-context adapter controls.
    return {"piece_id": case["piece_id"], "question": case["question"],
            "measure_range": case.get("measure_range"), "language": case["language"],
            "top_k": case.get("top_k", 6), "generate": True,
            **{key: case.get(key) for key in (
                "injection", "fixed_evidence", "selected_pool", "pool", "condition",
                "budget", "measure_ranges", "score_context")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--expert-results", type=Path)
    parser.add_argument("--remaining-results", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    base = args.directory
    experts = read(base / "expert-cases.json")["cases"]
    expanded_document = read(base / "expanded-cases.json")
    expanded = expanded_document["cases"]
    by_id = {c["id"]: c for c in expanded}
    expert_ids = {c["id"] for c in experts}
    if len(by_id) != 460 or len(expert_ids) != 105:
        raise ValueError("Unexpected fixed cohort")
    for case in experts:
        if request(case) != request(by_id[case["id"]]):
            raise ValueError("Expert request differs from expanded request: " + case["id"])
    remaining = [c for c in expanded if c["id"] not in expert_ids]
    frozen_write(base / "expert-review-cases.json", {
        **expanded_document, "cases": [by_id[c["id"]] for c in experts]})
    frozen_write(base / "remaining-cases.json", {**expanded_document, "cases": remaining})
    frozen_write(base / "request-partition.json", {
        "expert_count": len(experts), "remaining_count": len(remaining), "total": len(expanded),
        "expert_cases_sha256": sha(base / "expert-cases.json"),
        "expanded_cases_sha256": sha(base / "expanded-cases.json"),
        "remaining_cases_sha256": sha(base / "remaining-cases.json"),
        "request_fields": list(request(experts[0])),
        "same_requests": True,
        "judging": "Use expanded reference_authority for all 460 packets; expert-only packets are not interchangeable.",
    })
    if not any((args.expert_results, args.remaining_results, args.output)):
        print(json.dumps({"expert": 105, "remaining": 355, "same_requests": True}))
        return
    if not all((args.expert_results, args.remaining_results, args.output)):
        raise ValueError("Assembly needs both result files and output")
    sources, rows, pipeline_contract = [], {}, None
    for path, cases in [(args.expert_results, experts), (args.remaining_results, remaining)]:
        data = [json.loads(line) for line in path.read_text().splitlines()]
        manifest = read(Path(str(path) + ".manifest.json"))
        contract = manifest["contract"]
        if any(row["contract_id"] != manifest["contract_id"] for row in data):
            raise ValueError("Row/manifest contract mismatch")
        if len(data) != len(cases) or {r["case_id"] for r in data} != {c["id"] for c in cases}:
            raise ValueError("Incomplete or duplicated result cohort")
        # Cohort hashes differ; all production/model/config/runner fields must match.
        current = {key: value for key, value in contract.items() if key not in ("cases_sha256", "cases_path")}
        if pipeline_contract is None:
            pipeline_contract = current
        elif current != pipeline_contract:
            raise ValueError("The two cohorts used different pipeline contracts")
        for row in data:
            if row["case_id"] in rows:
                raise ValueError("Duplicate generated request")
            rows[row["case_id"]] = row
        sources.append({"path": str(path), "sha256": sha(path), "manifest_sha256": sha(Path(str(path) + ".manifest.json")),
                        "contract_id": manifest["contract_id"], "rows": len(data)})
    text = "".join(json.dumps(rows[c["id"]], ensure_ascii=False) + "\n" for c in expanded)
    if args.output.exists() and args.output.read_text() != text:
        raise ValueError("Refusing to overwrite a completed assembly")
    args.output.write_text(text)
    frozen_write(Path(str(args.output) + ".assembly.json"), {
        "sources": sources, "sha256": sha(args.output), "rows": len(rows),
        "policy": "Original rows, failures and per-run contract IDs retained; reordered into the fixed 460 case order. Not a single generation session.",
    })
    print(json.dumps({"assembled": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
