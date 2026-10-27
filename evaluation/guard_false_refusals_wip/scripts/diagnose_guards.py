"""Ask questions through the RAG service, logging guard inputs and verdicts.

Usage (soprano-qa env, SOPRANO_QA_* env vars set as for the demo server):
    python diagnose_guards.py <rag_root> <cases.json> <out.json>
cases.json: [{"piece_id", "question", "measure_range", "language"}]
"""
import json
import sys

rag_root, cases_path, out_path = sys.argv[1:4]
sys.path.insert(0, rag_root)

from soprano_qa import llm, security, service  # noqa: E402

log = []
original_guard = security.semantic_guard
original_render = llm.render_structured_answer


def logged_guard(model_path, llm_settings, payload, *, stage, generate_fn):
    entry = {"stage": stage, "payload": payload}
    log.append(entry)

    def recording_generate(*args, **kwargs):
        raw = generate_fn(*args, **kwargs)
        entry["raw_verdict"] = raw
        return raw

    return original_guard(model_path, llm_settings, payload, stage=stage, generate_fn=recording_generate)


def logged_render(raw, payload):
    log.append({"stage": "draft", "raw_structured": raw})
    return original_render(raw, payload)


llm.semantic_guard = logged_guard
llm.render_structured_answer = logged_render

results = []
for case in json.load(open(cases_path, encoding="utf-8")):
    log = []
    rng = tuple(case["measure_range"]) if case["measure_range"] else None
    response = service.ask(
        piece_id=case["piece_id"], question=case["question"], measure_range=rng,
        generate=True, language=case.get("language", "en"),
    )
    results.append({"case": case, "answer": response.get("answer"),
                    "unavailable_reason": response.get("unavailable_reason"),
                    "evidence_ids": [e.get("id") for e in response.get("evidence", [])],
                    "retrieval_query": response.get("retrieval_query"),
                    "log": log})
    print(case["piece_id"], case["question"][:50], "->", response.get("unavailable_reason") or "answered",
          [(e["stage"], e.get("raw_verdict")) for e in log if e["stage"] != "draft"], flush=True)

json.dump(results, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
