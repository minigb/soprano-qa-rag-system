"""Check a focused reviewer on observed bad answers and faithful source controls.

Diagnostic only: this neither changes benchmark judgments nor generates answers
for scoring. Do not read or emit model reasoning.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REVIEWER = ROOT / "evaluation/semantic_generation_2026-11-02/diagnostics/fidelity_sentence.py"
spec = importlib.util.spec_from_file_location("fidelity_diagnostic", REVIEWER)
reviewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reviewer)
review_claim_fidelity = reviewer.review_claim_fidelity
from soprano_qa.llm import generate
from soprano_qa.structured_answer import RenderedAnswer
from soprano_qa.security import SecurityViolation

BASE = ROOT / "evaluation/semantic_generation_2026-11-02"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-name", default="fidelity-review-diagnostic")
    args = parser.parse_args()
    if Path(args.output_name).name != args.output_name:
        raise ValueError("Use a filename without directories")
    settings = json.loads((BASE / "runtime/e38/settings.json").read_text())
    rows = [json.loads(line) for line in (BASE / "e38-probe.jsonl").read_text().splitlines()]
    out = BASE / (args.output_name + ".jsonl")
    if out.exists():
        raise ValueError("Retain prior diagnostic and use a new artifact after changes")
    with out.open("w") as stream:
        for row in rows[:4]:
            evidence = row["response"]["evidence"]
            sources = [{"citation_label": f"E{i+1}", "answer": e["text"]} for i, e in enumerate(evidence)]
            for condition in ["observed_generated_error", "faithful_control"]:
                text = row["response"]["answer"] if condition == "observed_generated_error" else "\n".join(e["text"] for e in evidence)
                draft = RenderedAnswer([{"kind": "evidence", "text": text, "evidence": [s["citation_label"] for s in sources]}])
                try:
                    issues = review_claim_fidelity(draft, {"Retrieved evidence": sources}, model_path=settings["model_path"],
                                                   settings=settings["llm"], generate_fn=generate)
                    result = {"case_id": row["case_id"], "condition": condition, "status": "ok", "issues": issues,
                              "reviewer_sha256": hashlib.sha256(REVIEWER.read_bytes()).hexdigest()}
                except SecurityViolation as exc:
                    result = {"case_id": row["case_id"], "condition": condition, "status": "review_failure", "reason": exc.reason}
                stream.write(json.dumps(result, ensure_ascii=False) + "\n"); stream.flush()
                print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
