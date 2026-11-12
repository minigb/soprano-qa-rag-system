"""Check literal quotation identity in public responses, independently of a judge.

This verifies text and source availability, not relevance, truth, retrieval
completeness or English translation quality. All input rows remain recorded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HEADER = "참고 자료의 원문입니다.\n"
BACKGROUND = "일반적인 음악 설명입니다.\n"


def inspect(response):
    text = response.get("answer", "")
    if not text.startswith(HEADER):
        return {"quotation": False}
    text = text[len(HEADER):]
    sources = [e for e in response.get("evidence", []) if e.get("text")]
    matched = []
    while text:
        if matched and text.startswith(BACKGROUND):
            return {"quotation": True, "exact_source_ids": matched,
                    "verified": True, "separate_background": True}
        choices = [e for e in sources if e["id"] not in matched and
                   (text == e["text"].rstrip() or text.startswith(e["text"] + "\n\n"))]
        if not choices:
            return {"quotation": True, "exact_source_ids": matched,
                    "verified": False, "unmatched_tail": text}
        # A source can contain another source as a prefix. Prefer the longest
        # complete identity, never a truncated clause or normalized spelling.
        # The display finalizer strips whitespace at the end of the entire
        # answer. Only a final whole-source match allows that outer trimming.
        source = max(choices, key=lambda e: len(e["text"]))
        matched.append(source["id"])
        text = text[len(source["text"]):]
        if text:
            text = text[2:]
    return {"quotation": True, "exact_source_ids": matched,
            "verified": bool(matched), "separate_background": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.results.read_text().splitlines()]
    cases = [{"case_id": r["case_id"], "status": r["status"],
              **inspect(r.get("response") or {})} for r in rows]
    result = {
        "results_sha256": hashlib.sha256(args.results.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limitations": "Literal text identity allowing display-time trailing whitespace removal at the end of the answer only. A quotation may still be irrelevant, incomplete, inaccurate in its source, or unsafe. Non-quotation answers and English translations require separate assessment. An unmatched tail can also be a score-lyric addition; inspect it rather than automatically declaring a semantic error.",
        "rows": len(rows), "quotation_answers": sum(c["quotation"] for c in cases),
        "verified_quotation_answers": sum(c.get("verified", False) for c in cases),
        "cases": cases,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output.exists() and args.output.read_text() != text:
        raise ValueError("Refusing to replace a different quotation audit")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    print(json.dumps({k: result[k] for k in ("rows", "quotation_answers", "verified_quotation_answers")}))


if __name__ == "__main__":
    main()
