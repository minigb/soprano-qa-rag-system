"""Re-run only the semantic guards on logged payloads, under a given rag root.

Usage: python guard_only.py <rag_root> <out.json> <log.json> [<log.json> ...]
Each log is diagnose_guards.py output. For every logged admission payload and
every logged output payload (draft), the guard of <rag_root> is asked again;
prints old -> new verdicts. English cases carry their Korean translation in
"translation" when the logging run captured it.
"""
import json
import os
import sys

rag_root, out_path, *logs = sys.argv[1:]
sys.path.insert(0, rag_root)

from soprano_qa import llm, security  # noqa: E402
from soprano_qa.service import SETTINGS  # noqa: E402

if os.environ.get("VARIANT"):
    exec(open(os.environ["VARIANT"], encoding="utf-8").read(), {"security": security})

rows = []
for log_path in logs:
    for result in json.load(open(log_path, encoding="utf-8")):
        case = result["case"]
        for entry in result["log"]:
            if entry["stage"] == "draft":
                continue
            raw = {}
            payload = dict(entry["payload"])
            key = os.environ.get("TRANSLATION_KEY")
            if key and case["language"] == "en" and result.get("retrieval_query"):
                payload[key] = result["retrieval_query"]

            def recording(*args, **kwargs):
                raw["v"] = llm.generate(*args, **kwargs)
                return raw["v"]

            try:
                security.semantic_guard(
                    SETTINGS["model_path"], SETTINGS["llm"], payload,
                    stage=entry["stage"], generate_fn=recording,
                )
                verdict = "allow"
            except security.SecurityViolation as exc:
                verdict = exc.reason
            old = "allow" if '"allow": true' in (entry.get("raw_verdict") or "").replace('":true', '": true') else "reject"
            rows.append({"case": case, "stage": entry["stage"], "old": old, "new": verdict, "raw": raw.get("v")})
            print(f"{case['language']} {case['piece_id'][:14]:14} #{case.get('n', '?')} {entry['stage']:9} {old:6} -> {verdict}", flush=True)

json.dump(rows, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
