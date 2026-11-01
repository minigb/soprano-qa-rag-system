"""Verify displayed English source text against exact-source reviewed translations.

This is a text-identity audit, not another semantic judge. It cannot establish
source truth, translation correctness, relevance or completeness of selection.
Unmatched text may be a legitimate fallback translation and needs separate review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from soprano_qa.source_translation_catalog import TRANSLATIONS_EN

HEADER = "Translated reference text:\n"
BACKGROUND = "General explanation:"


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def inspect(response, catalog=None):
    catalog = TRANSLATIONS_EN if catalog is None else catalog
    text = response.get("answer", "")
    if not text.startswith(HEADER):
        return {"translation_quotation": False}
    text = text[len(HEADER):]
    choices = {}
    for source in response.get("evidence", []):
        key = digest(source.get("text", ""))
        if key in catalog:
            choices[source["id"]] = (key, catalog[key])
    matched = []
    while text:
        if matched and text.startswith(BACKGROUND):
            return {"translation_quotation": True, "verified": True,
                    "matched_sources": matched, "separate_background": True}
        hits = [(identity, key, translated) for identity, (key, translated) in choices.items()
                if text == translated.rstrip() or text.startswith(translated + "\n")]
        if not hits:
            return {"translation_quotation": True, "verified": False,
                    "matched_sources": matched, "unmatched_tail": text}
        identity, key, translated = max(hits, key=lambda x: len(x[2]))
        matched.append({"source_id": identity, "source_sha256": key})
        del choices[identity]
        # Only the display separator is ignored; source words, punctuation and
        # internal newlines must remain exact. Outer answer trimming is allowed.
        text = text[len(translated):].lstrip("\n")
    return {"translation_quotation": True, "verified": bool(matched),
            "matched_sources": matched, "separate_background": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.results.read_text().splitlines()]
    cases = [{"case_id": row["case_id"], "status": row["status"],
              **inspect(row.get("response") or {})} for row in rows]
    result = {
        "results_sha256": hashlib.sha256(args.results.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "catalog_sha256": hashlib.sha256((ROOT/"soprano_qa/source_translation_catalog.py").read_bytes()).hexdigest(),
        "limitations": __doc__,
        "rows": len(rows),
        "translation_answers": sum(c["translation_quotation"] for c in cases),
        "verified_translation_answers": sum(c.get("verified", False) for c in cases),
        "cases": cases,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output.exists() and args.output.read_text() != text:
        raise ValueError("Refusing to replace a different translation audit")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    print(json.dumps({k: result[k] for k in ("rows", "translation_answers", "verified_translation_answers")}))


if __name__ == "__main__":
    main()
