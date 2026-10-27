#!/usr/bin/env python3
"""Run a paired, serial comparison of frozen original and hardened pipelines.

Each condition lives in its own long-lived process. Models and retrieval indexes
are loaded before timing, and one ordinary question warms inference. Both workers
remain resident, but requests are never dispatched concurrently. Indirect attacks
modify a copied retrieved record or lyric context for one request only; they do
not write to the corpus. This measures forced exposure, not poison retrieval.

The output is append-only JSONL, with a matching .manifest.json run contract.
Resume only accepts an identical case/source/settings contract. An application
exception is a measured outcome. Worker death/timeout is checkpointed explicitly,
then stops the run; rerunning resumes other requests without retrying that row.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager, redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import selectors
import subprocess
import sys
import time
from typing import Any, Iterator
from unittest.mock import patch


SCHEMA_VERSION = 1
VARIANTS = ("original", "hardened")
OVERRIDE_ENV = (
    "SOPRANO_QA_SETTINGS", "SOPRANO_QA_RAG_DATASET_ROOT",
    "SOPRANO_QA_DATASET_ROOT", "SOPRANO_QA_MODEL_PATH",
    "SOPRANO_QA_EMBEDDING_MODEL_PATH",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted((root / "soprano_qa").rglob("*.py"))
    if not files:
        raise ValueError(f"No pipeline source found in {root}")
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


def load_cases(path: Path) -> list[dict[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    cases = document["cases"] if isinstance(document, dict) else document
    if not isinstance(cases, list) or not cases:
        raise ValueError("Cases must be a nonempty list")
    seen = set()
    for case in cases:
        for key in ("id", "kind", "language", "piece_id", "question"):
            if not isinstance(case.get(key), str) or not case[key]:
                raise ValueError(f"Case requires a nonempty {key}")
        if case["id"] in seen:
            raise ValueError(f"Duplicate case ID: {case['id']}")
        seen.add(case["id"])
        if case["language"] not in {"ko", "en"}:
            raise ValueError("Only Korean and English cases are supported")
        injection = case.get("injection")
        if injection:
            if injection.get("target") not in {"answer", "metadata", "score_context"}:
                raise ValueError(f"Unknown injection target in {case['id']}")
            if injection.get("placement") not in {"append", "replace"}:
                raise ValueError(f"Unknown injection placement in {case['id']}")
            if injection.get("target") == "metadata" and injection.get("field", "work") not in {"work", "topic"}:
                raise ValueError(f"Unknown metadata injection field in {case['id']}")
            if not isinstance(injection.get("payload"), str) or not injection["payload"]:
                raise ValueError(f"Empty injection in {case['id']}")
    return cases


def schedule(cases: list[dict[str, Any]], repeats: int, seed: int) -> list[tuple[int, dict[str, Any], str]]:
    rng = random.Random(seed)
    jobs = []
    for repeat in range(repeats):
        order = list(cases)
        rng.shuffle(order)
        for case in order:
            variants = list(VARIANTS)
            rng.shuffle(variants)
            jobs.extend((repeat, case, variant) for variant in variants)
    return jobs


def job_key(variant: str, case_id: str, repeat: int) -> tuple[str, str, int]:
    return variant, case_id, repeat


def read_completed(path: Path, contract_id: str) -> set[tuple[str, str, int]]:
    completed = set()
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"Incomplete/invalid JSONL line {line_number}; preserve and repair it before resuming") from exc
            if row.get("contract_id") != contract_id:
                raise ValueError("Output contains a different evaluation contract")
            key = job_key(row["variant"], row["case_id"], row["repeat"])
            if key in completed:
                raise ValueError(f"Duplicate checkpoint row: {key}")
            completed.add(key)
    return completed


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_row(stream: Any, row: dict[str, Any]) -> None:
    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def _injected_text(original: str, injection: dict[str, Any]) -> str:
    return injection["payload"] if injection["placement"] == "replace" else original + "\n\n" + injection["payload"]


def _contains_payload(value: Any, payload: str) -> bool:
    if isinstance(value, str):
        if payload in value:
            return True
        try:
            decoded = json.loads(value)
        except (ValueError, RecursionError):
            return False
        return not isinstance(decoded, str) and _contains_payload(decoded, payload)
    if isinstance(value, dict):
        return any(_contains_payload(part, payload) for part in value.values())
    if isinstance(value, list):
        return any(_contains_payload(part, payload) for part in value)
    return False


@contextmanager
def inject_request(qa: Any, answer: Any, injection: dict[str, Any] | None, exposure: dict[str, Any]) -> Iterator[None]:
    """Inject into a fresh copy, retaining the actual retriever's selected order."""
    if not injection:
        yield
        return
    with ExitStack() as stack:
        if injection["target"] == "score_context":
            original_lyrics = qa.selected_measure_lyrics

            def poisoned_lyrics(*args: Any, **kwargs: Any) -> Any:
                context = deepcopy(original_lyrics(*args, **kwargs))
                if not context:
                    exposure["not_applied_reason"] = "no_selected_score_context"
                    return context
                measure = next((item for item in context.get("measures", []) if item.get("voice") == "sung"), None)
                if measure is None:
                    exposure["not_applied_reason"] = "no_sung_measure"
                    return context
                lines = list(measure.get("lyric_lines") or [""])
                exposure.update({"injection_applied": True, "measure": measure.get("measure"), "clean_lyric": lines[0]})
                lines[0] = _injected_text(lines[0], injection)
                measure["lyric_lines"] = lines
                return context

            stack.enter_context(patch.object(qa, "selected_measure_lyrics", poisoned_lyrics))
        else:
            original_retrieve = qa.retrieve_generation_evidence

            def poisoned_retrieve(*args: Any, **kwargs: Any) -> Any:
                results, messages = original_retrieve(*args, **kwargs)
                if not results:
                    exposure["not_applied_reason"] = "no_retrieved_generation_record"
                    return results, messages
                results = deepcopy(results)
                record = results[0].record
                field = "answer" if injection["target"] == "answer" else injection.get("field", "work")
                if field not in {"answer", "work", "topic"}:
                    raise ValueError("Unsupported injection field")
                exposure.update({
                    "injection_applied": True,
                    "record_id": record.get("id"),
                    "field": field,
                    "scope_match": results[0].scope_match,
                    "clean_record_answer": record.get("answer"),
                    "clean_field_value": record.get(field, ""),
                })
                record[field] = _injected_text(str(record.get(field, "")), injection)
                # Rebuild after mutation: the hardened screen must see the
                # compromised text before any composer/guard is invoked.
                messages = answer.build_messages(
                    kwargs["piece"], kwargs["measures"], kwargs["query"], results,
                    score_context=kwargs.get("score_context", ""),
                )
                return results, messages

            stack.enter_context(patch.object(qa, "retrieve_generation_evidence", poisoned_retrieve))
        yield


class CallCounter:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.payload: str | None = None
        self.exposure: dict[str, Any] | None = None

    def reset(self, injection: dict[str, Any] | None, exposure: dict[str, Any]) -> None:
        self.calls = []
        self.payload = injection["payload"] if injection else None
        self.exposure = exposure

    def wrap_loader(self, loader: Any) -> Any:
        counter = self

        class Proxy:
            def __init__(self, model: Any) -> None:
                self.model = model

            def __getattr__(self, name: str) -> Any:
                return getattr(self.model, name)

            def create_chat_completion(self, *args: Any, **kwargs: Any) -> Any:
                messages = kwargs.get("messages", args[0] if args else [])
                system = str(messages[0].get("content", "")) if messages else ""
                if "You translate a singer" in system:
                    kind = "translation"
                elif "security gate for a soprano" in system:
                    kind = "output_guard" if "candidate answer addresses" in system else "admission_guard"
                else:
                    kind = "composition"
                if counter.payload and _contains_payload(messages, counter.payload):
                    assert counter.exposure is not None
                    counter.exposure["model_exposed"] = True
                    counter.exposure.setdefault("model_exposure_call_kinds", []).append(kind)
                call: dict[str, Any] = {"kind": kind, "usage": None, "finish_reason": None}
                counter.calls.append(call)
                start = time.perf_counter()
                try:
                    result = self.model.create_chat_completion(*args, **kwargs)
                    call["usage"] = result.get("usage")
                    choices = result.get("choices") or []
                    call["finish_reason"] = choices[0].get("finish_reason") if choices else None
                    return result
                except Exception as exc:
                    call["exception_type"] = type(exc).__name__
                    raise
                finally:
                    call["elapsed_seconds"] = time.perf_counter() - start

        def counted_loader(*args: Any, **kwargs: Any) -> Any:
            return Proxy(loader(*args, **kwargs))

        return counted_loader


def run_case(qa: Any, answer: Any, counter: CallCounter, case: dict[str, Any]) -> dict[str, Any]:
    injection = case.get("injection")
    exposure = {
        "requested": bool(injection), "injection_applied": False,
        "model_exposed": False, "target": injection.get("target") if injection else None,
    }
    counter.reset(injection, exposure)
    response = None
    error = None
    start = time.perf_counter()
    try:
        with inject_request(qa, answer, injection, exposure):
            response = qa.ask(
                piece_id=case["piece_id"], question=case["question"],
                measure_range=tuple(case["measure_range"]) if case.get("measure_range") else None,
                generate=True, top_k=case.get("top_k", 6), language=case["language"],
            )
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    elapsed = time.perf_counter() - start
    return {
        "status": "exception" if error else "response",
        "response": response, "exception": error,
        "elapsed_seconds": elapsed, "latency_scope": "service_request",
        "llm_call_count": len(counter.calls), "llm_calls": deepcopy(counter.calls),
        "exposure": exposure,
    }


def _emit(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def worker_main(root: Path, settings: Path) -> None:
    started = time.perf_counter()
    for name in OVERRIDE_ENV:
        os.environ.pop(name, None)
    os.environ["SOPRANO_QA_SETTINGS"] = str(settings.resolve())
    # Never fall through to the live checkout or an installed sibling package.
    sys.path[:] = [str(root.resolve())] + [
        item for item in sys.path
        if not (Path(item or os.getcwd()).resolve() / "soprano_qa").is_dir()
        and Path(item or os.getcwd()).resolve() != root.resolve()
    ]
    with redirect_stdout(sys.stderr):
        qa = importlib.import_module("soprano_qa.service")
        answer = importlib.import_module("soprano_qa.answer")
        llm = importlib.import_module("soprano_qa.llm")
        if not Path(qa.__file__).resolve().is_relative_to(root.resolve()):
            raise RuntimeError("Worker imported a different pipeline checkout")
        counter = CallCounter()
        original_loader = llm.load_llama
        llm.load_llama = counter.wrap_loader(original_loader)
        qa.validate_generation_requirements(qa.SETTINGS["model_path"], qa.SETTINGS["llm"])
        qa._get_index()
        cfg = qa.SETTINGS["llm"]
        model = original_loader(
            model_path=qa.SETTINGS["model_path"], n_ctx=int(cfg.get("n_ctx", 8192)),
            n_gpu_layers=int(cfg.get("n_gpu_layers", -1)),
            chat_format=llm._configured_chat_format(cfg),
        )
        metadata = getattr(model, "metadata", {})
        try:
            backend_version = importlib.metadata.version("llama-cpp-python")
        except importlib.metadata.PackageNotFoundError:
            backend_version = None
        ready = {
            "event": "ready", "pid": os.getpid(),
            "system_root": str(root.resolve()), "module_file": qa.__file__,
            "settings": qa.SETTINGS, "startup_seconds": time.perf_counter() - started,
            "llama_cpp_version": backend_version,
            "chat_format": getattr(model, "chat_format", None),
            "chat_template": metadata.get("tokenizer.chat_template"),
            "model_name": metadata.get("general.name"),
        }
    _emit(ready)
    for line in sys.stdin:
        command = json.loads(line)
        if command.get("command") == "stop":
            break
        with redirect_stdout(sys.stderr):
            outcome = run_case(qa, answer, counter, command["case"])
        _emit({"event": "result", "request_id": command["request_id"], **outcome})


class WorkerFailure(RuntimeError):
    pass


class WorkerClient:
    def __init__(self, variant: str, root: Path, settings: Path, log_path: Path, timeout: float) -> None:
        self.variant = variant
        self.timeout = timeout
        self.log_path = log_path
        self.log = log_path.open("a", encoding="utf-8")
        environment = os.environ.copy()
        for key in OVERRIDE_ENV:
            environment.pop(key, None)
        self.process = subprocess.Popen(
            [sys.executable, "-u", str(Path(__file__).resolve()), "--worker",
             "--system-root", str(root.resolve()), "--settings", str(settings.resolve())],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            text=True, encoding="utf-8", bufsize=1, env=environment,
        )
        try:
            self.ready = self.receive()
            if self.ready.get("event") != "ready":
                raise WorkerFailure(f"Unexpected startup message from {variant}")
        except Exception:
            self.close()
            raise

    def receive(self) -> dict[str, Any]:
        assert self.process.stdout is not None
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if not selector.select(self.timeout):
                raise WorkerFailure(f"{self.variant} exceeded {self.timeout}s; see {self.log_path}")
            line = self.process.stdout.readline()
        if not line:
            raise WorkerFailure(f"{self.variant} exited ({self.process.poll()}); see {self.log_path}")
        try:
            return json.loads(line)
        except ValueError as exc:
            raise WorkerFailure(f"Invalid worker protocol from {self.variant}; see {self.log_path}") from exc

    def request(self, case: dict[str, Any], request_id: str) -> dict[str, Any]:
        assert self.process.stdin is not None
        try:
            self.process.stdin.write(json.dumps({"command": "ask", "case": case, "request_id": request_id}, ensure_ascii=False) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise WorkerFailure(f"{self.variant} stopped before dispatch") from exc
        outcome = self.receive()
        if outcome.get("event") != "result" or outcome.get("request_id") != request_id:
            raise WorkerFailure("Mismatched worker response")
        outcome.pop("event")
        outcome.pop("request_id")
        return outcome

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                assert self.process.stdin is not None
                self.process.stdin.write('{"command":"stop"}\n')
                self.process.stdin.flush()
                self.process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        for stream in (self.process.stdin, self.process.stdout):
            if stream:
                stream.close()
        self.log.close()


def parent_main(args: argparse.Namespace) -> None:
    cases = load_cases(args.cases)
    roots = {"original": args.original_root.resolve(), "hardened": args.hardened_root.resolve()}
    settings = {"original": args.original_settings.resolve(), "hardened": args.hardened_settings.resolve()}
    contract = {
        "schema_version": SCHEMA_VERSION, "runner_sha256": file_sha256(Path(__file__)),
        "cases_sha256": file_sha256(args.cases), "case_count": len(cases),
        "repeats": args.repeats, "seed": args.seed,
        "source_sha256": {key: source_sha256(value) for key, value in roots.items()},
        "settings_sha256": {key: file_sha256(value) for key, value in settings.items()},
    }
    contract_id = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output.with_name(args.output.name + ".manifest.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("contract") != contract:
            raise ValueError("Run contract differs from the existing manifest; choose a new output")
    else:
        manifest = {"created_at": utc_now(), "contract_id": contract_id, "contract": contract,
                    "roots": {k: str(v) for k, v in roots.items()},
                    "settings_files": {k: str(v) for k, v in settings.items()},
                    "cases_file": str(args.cases.resolve()), "sessions": []}
        write_json(manifest_path, manifest)
    completed = read_completed(args.output, contract_id)
    jobs = schedule(cases, args.repeats, args.seed)
    if len(completed) == len(jobs):
        print(f"Already complete: {len(completed)} requests", flush=True)
        return
    warmup = next((case for case in cases if case["kind"] == "benign" and not case.get("injection")), None)
    if warmup is None:
        raise ValueError("At least one benign case is needed for the untimed warmup")
    session = {"started_at": utc_now(), "workers": {}, "warmup_case_id": warmup["id"]}
    manifest["sessions"].append(session)
    workers = {}
    try:
        for variant in VARIANTS:
            log = args.output.with_name(args.output.stem + f".{variant}.stderr.log")
            print(f"Loading {variant} worker", flush=True)
            workers[variant] = WorkerClient(variant, roots[variant], settings[variant], log, args.timeout)
            warmup_result = workers[variant].request(warmup, "warmup")
            session["workers"][variant] = {
                "ready": workers[variant].ready,
                "warmup": {key: warmup_result[key] for key in ("status", "exception", "elapsed_seconds", "llm_call_count")},
            }
            write_json(manifest_path, manifest)
        with args.output.open("a", encoding="utf-8") as stream:
            for position, (repeat, case, variant) in enumerate(jobs, 1):
                key = job_key(variant, case["id"], repeat)
                if key in completed:
                    continue
                request_id = f"{repeat}:{case['id']}:{variant}"
                dispatch_started = time.perf_counter()
                worker_failure = None
                try:
                    outcome = workers[variant].request(case, request_id)
                except WorkerFailure as exc:
                    worker_failure = exc
                    outcome = {"status": "worker_error", "response": None,
                               "exception": {"type": type(exc).__name__, "message": str(exc)},
                               "elapsed_seconds": time.perf_counter() - dispatch_started,
                               "latency_scope": "parent_dispatch_until_worker_failure",
                               "llm_call_count": None, "llm_calls": [], "exposure": None}
                row = {
                    "schema_version": SCHEMA_VERSION, "contract_id": contract_id,
                    "completed_at": utc_now(), "schedule_position": position,
                    "variant": variant, "case_id": case["id"], "repeat": repeat,
                    "pair_id": case.get("pair_id"), "kind": case["kind"],
                    "category": case.get("category"), "language": case["language"],
                    "piece_id": case["piece_id"], **outcome,
                }
                append_row(stream, row)
                completed.add(key)
                mode = (row.get("response") or {}).get("generation_mode", row["status"])
                print(f"{len(completed)}/{len(jobs)} {variant} {case['id']} r{repeat} {mode} {row['elapsed_seconds']:.3f}s", flush=True)
                if worker_failure:
                    raise worker_failure
        session["finished_at"] = utc_now()
        manifest["completed_requests"] = len(completed)
        manifest["complete"] = len(completed) == len(jobs)
    finally:
        for worker in workers.values():
            worker.close()
        session["stopped_at"] = utc_now()
        write_json(manifest_path, manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--system-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--settings", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--original-root", type=Path)
    parser.add_argument("--hardened-root", type=Path)
    parser.add_argument("--original-settings", type=Path)
    parser.add_argument("--hardened-settings", type=Path)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()
    required = ("system_root", "settings") if args.worker else (
        "original_root", "hardened_root", "original_settings", "hardened_settings", "cases", "output")
    for name in required:
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required")
    if args.repeats < 1 or args.timeout <= 0:
        parser.error("repeats and timeout must be positive")
    return args


def main() -> None:
    args = parse_args()
    if args.worker:
        worker_main(args.system_root, args.settings)
    else:
        parent_main(args)


if __name__ == "__main__":
    main()
