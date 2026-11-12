"""Audit generation provenance without exporting private model traces."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trace = args.results.with_name(args.results.name + ".private-traces.jsonl")
    records = [json.loads(line) for line in args.results.read_text().splitlines()]
    attempts = {r["attempt_id"]: r for r in map(json.loads, trace.read_text().splitlines()) if not r.get("warmup")}
    details = []
    for row in records:
        response = row.get("response") or {}
        mode = response.get("generation_mode")
        text = response.get("answer", "")
        # Only retain whether the final structured composition contains actual
        # authored prose. Private reasoning and model text are not exported.
        calls = [c for c in attempts[row["private_trace_attempt_id"]]["calls"] if c["kind"] == "composition"]
        nonempty = None
        if mode == "llm":
            assert calls, row["case_id"]
            content = calls[-1]["choices"][0]["message"]["content"]
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.S | re.I).strip()
            content = re.sub(r"</?think>", "", content, flags=re.I).strip()
            parsed = json.loads(content)
            nonempty = bool(parsed["sentences"]) and all(p["text"].strip() for p in parsed["sentences"])
            assert nonempty, row["case_id"]
        legacy = mode == "llm_guarded_source_comparison" or "참고 자료의 원문입니다" in text or "Translated reference" in text
        assert not legacy, row["case_id"]
        details.append({"id": row["case_id"], "mode": mode, "authored_nonempty_composition": nonempty,
                        "legacy_source_return": legacy})
    report = {"results_sha256": hashlib.sha256(args.results.read_bytes()).hexdigest(),
              "private_trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
              "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "requests": len(records), "generation_modes": dict(Counter(d["mode"] for d in details)),
              "llm_nonempty_compositions": sum(d["authored_nonempty_composition"] is True for d in details),
              "legacy_source_returns": sum(d["legacy_source_return"] for d in details),
              "limitation": "Provenance audit, not a factuality or naturalness judgment. A generated phrase may legitimately match a short source phrase.",
              "cases": details}
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output.exists() and args.output.read_text() != encoded:
        raise ValueError("Refusing to change a completed audit")
    args.output.write_text(encoded)
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
