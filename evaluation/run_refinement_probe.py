#!/usr/bin/env python3
"""Persistent local development probes with private drafts and a usage stop.

Normal cases use the public service. Fixed-context cases supply ``fixed_evidence``
or ``selected_pool`` (ordered {record, relevance, scope} entries), or a matched
experiment ``pool`` with explicit ``condition`` and ``budget``. Fixed runs bypass
retrieval, so their latency is deliberately labelled separately. No answer is
reused across cases. Every attempted failure is checkpointed.

Pass the SAME --usage-state to successive probes to retain usage provenance.
Only fresh rollout token_count events qualify. The user's October31 instruction
removes reset-based stops; only the95% cap and unavailable usage observations
stop the run. This script never starts an app-server or notifier.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_security_comparison as common


class UsageStop(BaseException):
    """Escape application exception handling without another inference call."""


def latest_usage(path: Path, *, now: float | None = None) -> dict:
    now = time.time() if now is None else now
    latest = None
    try:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    item = json.loads(line)
                except ValueError:
                    if not line.endswith("\n"):
                        continue  # A concurrently appended, unfinished last line.
                    raise UsageStop("Malformed usage rollout")
                payload = item.get("payload") or {}
                if item.get("type") == "event_msg" and payload.get("type") == "token_count":
                    latest = item
    except OSError as exc:
        raise UsageStop(f"Cannot read usage rollout: {exc}") from exc
    if latest is None:
        raise UsageStop("No token_count usage observation")
    try:
        observed = datetime.fromisoformat(latest["timestamp"].replace("Z", "+00:00")).timestamp()
        limits = latest["payload"]["rate_limits"]
        buckets = {}
        for name, value in limits.items():
            if isinstance(value, dict) and "used_percent" in value:
                used, reset = float(value["used_percent"]), float(value["resets_at"])
                if not math.isfinite(used) or not math.isfinite(reset) or not 0 <= used <= 100:
                    raise ValueError("Invalid usage values")
                buckets[name] = {"used_percent": used, "resets_at": reset,
                                 "window_minutes": value.get("window_minutes")}
        if not buckets:
            raise ValueError("No usage buckets")
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise UsageStop("Unknown or invalid current usage") from exc
    if not -60 <= now - observed <= 600:
        raise UsageStop("Usage observation is stale (>10 minutes) or future-dated")
    return {"observed_at": observed, "limit_id": limits.get("limit_id"), "buckets": buckets}


class UsageGuard:
    def __init__(self, rollout: Path, state_path: Path, log_path: Path):
        self.rollout, self.state_path, self.log_path = rollout, state_path, log_path
        self.state = json.loads(state_path.read_text()) if state_path.exists() else {}
        if self.state and self.state.get("rollout") != str(rollout.resolve()):
            raise UsageStop("Usage state belongs to a different rollout")

    def check(self, *, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        if self.state.get("terminal_stop"):
            raise UsageStop(self.state["terminal_stop"])
        current = latest_usage(self.rollout, now=now)
        reason = None
        if not self.state:
            self.state = {"rollout": str(self.rollout.resolve()), "initial": current,
                          "high_water": {k: b["used_percent"] for k, b in current["buckets"].items()}}
        for name, bucket in current["buckets"].items():
            used = bucket["used_percent"]
            self.state["high_water"][name] = max(used, self.state["high_water"].get(name, 0))
            if used >= 95:
                reason = f"Usage reached 95% in {name}"
                break
        self.state["latest"] = current
        if reason:
            self.state["terminal_stop"] = reason
        common.write_json(self.state_path, self.state)
        with self.log_path.open("a", encoding="utf-8") as stream:
            common.append_row(stream, {"checked_at": common.utc_now(), **current, "stop_reason": reason})
        if reason:
            raise UsageStop(reason)
        return current


class TraceCounter(common.CallCounter):
    def __init__(self, guard: UsageGuard):
        super().__init__()
        self.guard = guard
        self.traces: list[dict] = []
        self.usage_check_seconds = 0.0

    def reset(self, injection: dict | None, exposure: dict) -> None:
        super().reset(injection, exposure)
        self.traces = []
        self.usage_check_seconds = 0.0

    def wrap_loader(self, loader: Any) -> Any:
        counter = self

        class Capture:
            def __init__(self, model: Any):
                self.model = model

            def __getattr__(self, name: str) -> Any:
                return getattr(self.model, name)

            def create_chat_completion(self, *args: Any, **kwargs: Any) -> Any:
                checking_started = time.perf_counter()
                try:
                    counter.guard.check()
                except UsageStop:
                    counter.calls[-1]["not_dispatched"] = True
                    raise
                finally:
                    counter.usage_check_seconds += time.perf_counter() - checking_started
                trace = {"call_index": len(counter.calls) - 1, "kind": counter.calls[-1]["kind"]}
                counter.traces.append(trace)
                try:
                    result = self.model.create_chat_completion(*args, **kwargs)
                    trace["choices"] = deepcopy(result.get("choices"))
                    return result
                except Exception as exc:
                    trace["exception"] = {"type": type(exc).__name__, "message": str(exc)}
                    raise

        def capture_loader(*args: Any, **kwargs: Any) -> Any:
            return Capture(loader(*args, **kwargs))

        return super().wrap_loader(capture_loader)


def load_probe_cases(path: Path) -> list[dict]:
    document = json.loads(path.read_text())
    cases = document["cases"] if isinstance(document, dict) else document
    if not isinstance(cases, list) or not cases:
        raise ValueError("Cases must be a nonempty list")
    cases = deepcopy(cases)
    seen = set()
    for case in cases:
        case.setdefault("piece_id", case.get("piece"))
        case.setdefault("language", "ko")
        case.setdefault("pair_id", case.get("cluster", case.get("id")))
        for field in ("id", "kind", "piece_id", "question"):
            if not isinstance(case.get(field), str) or not case[field]:
                raise ValueError(f"Missing case field: {field}")
        if case["id"] in seen or case["language"] not in {"ko", "en"}:
            raise ValueError(f"Duplicate case or invalid language: {case['id']}")
        seen.add(case["id"])
        if "pool" in case and not any(k in case for k in ("fixed_evidence", "selected_pool")):
            if case.get("condition") not in {"priority", "relevance"} or type(case.get("budget")) is not int or case["budget"] < 1:
                raise ValueError("A frozen pool requires explicit condition and positive budget; select one job per case")
        if case.get("injection") and any(k in case for k in ("fixed_evidence", "selected_pool", "pool")):
            raise ValueError("For fixed-context attacks, put the payload in the frozen record explicitly")
    return cases


def fixed_results(case: dict, retrieval: Any, priority: dict) -> list[Any]:
    selected = case.get("fixed_evidence", case.get("selected_pool"))
    if selected is None:
        selected = sorted(case["pool"], key=lambda x: (-x["relevance"], x["record"]["id"]))
        if case["condition"] == "priority":
            selected.sort(key=lambda x: priority[x["record"]["evidence_type"]])
        selected = selected[:case["budget"]]
    return [retrieval.SearchResult(deepcopy(x["record"]), x.get("relevance", x.get("score", 1.0)),
                                  0, 0, 1, x.get("scope", x.get("scope_match", "general_evidence")))
            for x in selected]


def run_fixed_case(qa: Any, answer: Any, llm: Any, retrieval: Any, priority: dict,
                   counter: TraceCounter, case: dict) -> dict:
    counter.reset(None, {})
    started = time.perf_counter()
    response, error, results = None, None, []
    ranges = case.get("measure_ranges") or ([case["measure_range"]] if case.get("measure_range") else [])
    measures = ",".join(f"{a}-{b}" for a, b in ranges)
    try:
        results = fixed_results(case, retrieval, priority)
        qa.screen_question(case["question"])
        qa.screen_text(case["piece_id"], source="piece")
        score = case.get("score_context", "")
        qa.screen_text(score, source="score_context")
        if not results and not score:
            return {"status": "response", "exception": None,
                    "response": {"answer": answer.no_corpus_evidence_message(case["language"]),
                                 "generation_mode": "unavailable", "unavailable_reason": "no_corpus_evidence", "evidence": []},
                    "elapsed_seconds": time.perf_counter() - started,
                    "latency_scope": "fixed_evidence_generation", "llm_call_count": 0, "llm_calls": [], "fixed_selected_ids": []}
        messages = answer.build_messages(case["piece_id"], measures, case["question"], results,
                                         score_context=score, language=case["language"])
        raw = llm.generate(qa.SETTINGS["model_path"], messages, qa.SETTINGS["llm"],
                           security_context=json.loads(messages[1]["content"]))
        visible = answer.finalize_grounded_generated_answer(raw, results, messages, ranges, language=case["language"])
        visible = answer.finalize_user_visible_answer(visible, results, language=case["language"])
        refused = visible == answer.no_corpus_evidence_message(case["language"])
        response = {"answer": visible, "generation_mode": "unavailable" if refused else "llm",
                    "unavailable_reason": "fixed_context_insufficient" if refused else None,
                    "evidence": [qa._result_to_evidence(r, measure_ranges=ranges) for r in results],
                    "score_context": score}
    except (qa.SecurityViolation, qa.GeneratedAnswerRejected) as exc:
        response = {"answer": answer.no_corpus_evidence_message(case["language"]),
                    "generation_mode": "unavailable", "evidence": [],
                    "unavailable_reason": getattr(exc, "reason", "answer_validation_failed")}
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    return {"status": "exception" if error else "response", "response": response, "exception": error,
            "elapsed_seconds": time.perf_counter() - started, "latency_scope": "fixed_evidence_generation",
            "llm_call_count": len(counter.calls), "llm_calls": deepcopy(counter.calls),
            "fixed_selected_ids": [r.record["id"] for r in results]}


def import_pipeline(root: Path, settings: Path) -> tuple[Any, ...]:
    for name in common.OVERRIDE_ENV:
        os.environ.pop(name, None)
    os.environ["SOPRANO_QA_SETTINGS"] = str(settings.resolve())
    if any(name == "soprano_qa" or name.startswith("soprano_qa.") for name in sys.modules):
        raise RuntimeError("Pipeline was imported before the selected snapshot")
    sys.path[:] = [str(root.resolve())] + [item for item in sys.path
        if not (Path(item or os.getcwd()).resolve() / "soprano_qa").is_dir()]
    modules = tuple(importlib.import_module("soprano_qa." + name)
                    for name in ("service", "answer", "llm", "retrieval", "source_priority"))
    if any(not Path(module.__file__).resolve().is_relative_to(root.resolve()) for module in modules):
        raise RuntimeError("Imported a different pipeline checkout")
    return modules


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("system-root", "settings", "cases", "output"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--usage-rollout", type=Path,
                        help="Optional legacy usage monitor; omitted for uninterrupted local evaluation")
    parser.add_argument("--usage-state", type=Path, help="Share across all probes in the same original allowance")
    parser.add_argument("--variant", default="hardened", help="Use original/hardened for existing analyzer compatibility")
    parser.add_argument("--warmup-case", help="Optional case ID for one untimed warmup, stored only in private traces")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cases = load_probe_cases(args.cases)
    if args.warmup_case and args.warmup_case not in {c["id"] for c in cases}:
        raise ValueError("Unknown warmup case")
    contract = {"schema_version": 1, "runner_sha256": common.file_sha256(Path(__file__)),
                "helper_sha256": common.file_sha256(Path(common.__file__)),
                "source_sha256": common.source_sha256(args.system_root),
                "settings_sha256": common.file_sha256(args.settings),
                "cases_sha256": common.file_sha256(args.cases), "variant": args.variant,
                "warmup_case": args.warmup_case,
                "usage_rollout": str(args.usage_rollout.resolve()) if args.usage_rollout else None}
    contract_id = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    manifest_path = args.output.with_name(args.output.name + ".manifest.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["contract_id"] != contract_id:
            raise ValueError("Refusing to resume a different source/settings/case contract")
    else:
        if args.output.exists():
            raise ValueError("Existing results lack their run manifest")
        manifest = {"created_at": common.utc_now(), "contract_id": contract_id, "contract": contract, "sessions": []}
    completed = common.read_completed(args.output, contract_id)
    expected = {(args.variant, case["id"], 0) for case in cases}
    if not completed <= expected:
        raise ValueError("Unexpected checkpoint job")
    trace_path = args.output.with_name(args.output.name + ".private-traces.jsonl")
    usage_state = args.usage_state or args.output.with_name(args.output.name + ".usage-state.json")
    usage_state.parent.mkdir(parents=True, exist_ok=True)
    class Unmonitored:
        def check(self):
            return {}

    guard = (UsageGuard(args.usage_rollout, usage_state, args.output.with_name(args.output.name + ".usage-checks.jsonl"))
             if args.usage_rollout else Unmonitored())
    session = {"started_at": common.utc_now(), "private_trace_file": str(trace_path.resolve())}
    manifest["sessions"].append(session)
    common.write_json(manifest_path, manifest)
    counter = TraceCounter(guard)
    try:
        guard.check()
        if completed == expected:
            session["status"] = "already_complete"
            return 0
        with redirect_stdout(sys.stderr):
            qa, answer, llm, retrieval, priority = import_pipeline(args.system_root, args.settings)
            llm.load_llama = counter.wrap_loader(llm.load_llama)
            qa.validate_generation_requirements(qa.SETTINGS["model_path"], qa.SETTINGS["llm"])
            if any(not any(k in c for k in ("pool", "selected_pool", "fixed_evidence")) for c in cases):
                qa._get_index()
            cfg = qa.SETTINGS["llm"]
            model = llm.load_llama(model_path=qa.SETTINGS["model_path"], n_ctx=int(cfg.get("n_ctx", 8192)),
                                   n_gpu_layers=int(cfg.get("n_gpu_layers", -1)), chat_format=llm._configured_chat_format(cfg))
            session["ready"] = {"settings": qa.SETTINGS, "chat_format": getattr(model, "chat_format", None),
                                "model_name": getattr(model, "metadata", {}).get("general.name")}
        common.write_json(manifest_path, manifest)
        print(json.dumps({"event": "ready", "remaining": len(expected - completed)}), flush=True)
        fd = os.open(trace_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as traces, args.output.open("a", encoding="utf-8") as output:
            jobs = [(False, c) for c in cases if (args.variant, c["id"], 0) not in completed]
            if args.warmup_case:
                jobs.insert(0, (True, next(c for c in cases if c["id"] == args.warmup_case)))
            for warmup, case in jobs:
                guard.check()
                if common.source_sha256(args.system_root) != contract["source_sha256"]:
                    raise RuntimeError("Pipeline source changed during probe; use frozen snapshots")
                attempt_id = uuid.uuid4().hex
                started_at = common.utc_now()
                started = time.perf_counter()
                interrupted = None
                try:
                    with redirect_stdout(sys.stderr):
                        result = run_fixed_case(qa, answer, llm, retrieval, priority.SOURCE_PRIORITY, counter, case) \
                            if any(k in case for k in ("pool", "selected_pool", "fixed_evidence")) \
                            else common.run_case(qa, answer, counter, case)
                except (UsageStop, KeyboardInterrupt) as exc:
                    interrupted = exc
                    calls = [c for c in counter.calls if not c.get("not_dispatched")]
                    result = {"status": "worker_error", "response": None,
                              "exception": {"type": type(exc).__name__, "message": str(exc)},
                              "elapsed_seconds": time.perf_counter() - started,
                              "llm_call_count": len(calls), "llm_calls": deepcopy(calls)}
                common.append_row(traces, {"attempt_id": attempt_id, "case_id": case["id"], "warmup": warmup,
                                          "contract_id": contract_id, "calls": counter.traces})
                if not warmup:
                    row = {"contract_id": contract_id, "variant": args.variant, "case_id": case["id"],
                           "kind": case["kind"], "language": case["language"], "pair_id": case["pair_id"],
                           "repeat": 0, "started_at": started_at, "private_trace_attempt_id": attempt_id,
                           "usage_check_seconds_in_request": counter.usage_check_seconds,
                           "timing_note": "elapsed_seconds includes recorded usage-check I/O; fixed-context timing excludes retrieval", **result}
                    common.append_row(output, row)
                    completed.add((args.variant, case["id"], 0))
                    print(json.dumps({"event": "result", "completed": len(completed), "total": len(cases),
                                      "case_id": case["id"], "status": result["status"],
                                      "seconds": result["elapsed_seconds"]}), flush=True)
                if interrupted:
                    raise interrupted
                guard.check()
        session["status"] = "complete"
        return 0
    except UsageStop as exc:
        session.update(status="usage_stopped", stop_reason=str(exc))
        print(json.dumps({"event": "usage_stopped", "reason": str(exc), "checkpointed": len(completed)}), flush=True)
        return 3
    finally:
        session["finished_at"] = common.utc_now()
        common.write_json(manifest_path, manifest)


if __name__ == "__main__":
    raise SystemExit(main())
