from datetime import datetime, timezone
import json
from pathlib import Path
import stat
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from evaluation import run_refinement_probe as probe


def observation(now, used=19, reset=None, secondary=None):
    return {"timestamp": datetime.fromtimestamp(now, timezone.utc).isoformat(), "type": "event_msg",
            "payload": {"type": "token_count", "rate_limits": {
                "limit_id": "codex", "primary": {"used_percent": used, "resets_at": reset or now + 3600,
                                                   "window_minutes": 10080}, "secondary": secondary}}}


class UsageTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rollout = self.root / "usage.jsonl"
        self.state = self.root / "state.json"
        self.now = 1800000000

    def emit(self, **kwargs):
        self.rollout.write_text(json.dumps(observation(self.now, **kwargs)) + "\n")

    def guard(self):
        return probe.UsageGuard(self.rollout, self.state, self.root / "checks.jsonl")

    def test_unknown_and_stale_usage_fail_closed(self):
        self.rollout.write_text("")
        with self.assertRaises(probe.UsageStop):
            self.guard().check(now=self.now)
        self.emit()
        with self.assertRaisesRegex(probe.UsageStop, "stale"):
            self.guard().check(now=self.now + 601)

    def test_any_bucket_at_95_stops_permanently(self):
        self.emit(secondary={"used_percent": 95, "resets_at": self.now + 300, "window_minutes": 300})
        with self.assertRaisesRegex(probe.UsageStop, "secondary"):
            self.guard().check(now=self.now)
        self.emit(used=0)
        with self.assertRaisesRegex(probe.UsageStop, "secondary"):
            self.guard().check(now=self.now)

    def test_resets_and_clock_adjustments_do_not_stop(self):
        self.emit(used=40)
        self.guard().check(now=self.now)
        for used, reset in ((41, self.now + 3601), (0, self.now + 7200),
                            (1, self.now - 30)):
            with self.subTest(used=used, reset=reset):
                self.emit(used=used, reset=reset)
                self.guard().check(now=self.now)
        self.emit(used=95, reset=self.now + 10000)
        with self.assertRaisesRegex(probe.UsageStop, "95%"):
            self.guard().check(now=self.now)

    def test_new_usage_bucket_is_checked_after_window_change(self):
        self.emit(used=40)
        self.guard().check(now=self.now)
        self.emit(used=0, secondary={"used_percent": 96, "resets_at": self.now + 300, "window_minutes": 300})
        with self.assertRaisesRegex(probe.UsageStop, "secondary"):
            self.guard().check(now=self.now)

    def test_unfinished_append_preserves_last_complete_observation(self):
        self.emit()
        with self.rollout.open("a") as stream:
            stream.write('{"timestamp":')
        self.assertEqual(self.guard().check(now=self.now)["buckets"]["primary"]["used_percent"], 19)


class ProbeTests(unittest.TestCase):
    def test_selected_context_preserves_order_and_copies_records(self):
        entries = [{"record": {"id": "web", "evidence_type": "web_coaching"}, "relevance": .8, "scope": "general_evidence"},
                   {"record": {"id": "expert", "evidence_type": "expert_annotation"}, "relevance": .6, "scope": "local_example"}]
        retrieval = SimpleNamespace(SearchResult=lambda record, score, *rest: SimpleNamespace(record=record, score=score, scope_match=rest[-1]))
        results = probe.fixed_results({"selected_pool": entries}, retrieval, {})
        self.assertEqual([r.record["id"] for r in results], ["web", "expert"])
        results[0].record["id"] = "changed"
        self.assertEqual(entries[0]["record"]["id"], "web")
        ranked = probe.fixed_results({"pool": entries, "condition": "priority", "budget": 1}, retrieval,
                                     {"expert_annotation": 0, "web_coaching": 1})
        self.assertEqual(ranked[0].record["id"], "expert")

    def test_usage_stop_never_dispatches_completion(self):
        class StopGuard:
            def check(self):
                raise probe.UsageStop("stop")
        model = SimpleNamespace(create_chat_completion=lambda **kwargs: self.fail("Dispatched after usage stop"))
        counter = probe.TraceCounter(StopGuard())
        counter.reset(None, {})
        with self.assertRaises(probe.UsageStop):
            counter.wrap_loader(lambda: model)().create_chat_completion(messages=[])
        self.assertTrue(counter.calls[0]["not_dispatched"])
        self.assertEqual(counter.traces, [])

    def test_fixed_path_uses_production_guard_context_and_both_finalizers(self):
        events = []
        class SecurityViolation(Exception):
            reason = "security_output_rejected"
        class GeneratedAnswerRejected(Exception):
            pass
        qa = SimpleNamespace(SETTINGS={"model_path": "fake", "llm": {}},
            screen_question=lambda q: None, screen_text=lambda *a, **k: None,
            SecurityViolation=SecurityViolation, GeneratedAnswerRejected=GeneratedAnswerRejected,
            _result_to_evidence=lambda r, **k: {"id": r.record["id"]})
        answer = SimpleNamespace(
            build_messages=lambda *a, **k: [{"role": "system", "content": "policy"}, {"role": "user", "content": '{"question":"q"}'}],
            finalize_grounded_generated_answer=lambda raw, *a, **k: events.append(("grounded", raw)) or "grounded",
            finalize_user_visible_answer=lambda text, *a, **k: events.append(("visible", text)) or "visible",
            no_corpus_evidence_message=lambda lang: "refusal")
        llm = SimpleNamespace(generate=lambda *a, **k: events.append(("generate", k["security_context"])) or "draft")
        retrieval = SimpleNamespace(SearchResult=lambda record, *a: SimpleNamespace(record=record))
        counter = probe.TraceCounter(SimpleNamespace(check=lambda: None))
        case = {"piece_id": "p", "question": "q", "language": "ko", "selected_pool": [{"record": {"id": "r"}}]}
        result = probe.run_fixed_case(qa, answer, llm, retrieval, {}, counter, case)
        self.assertEqual(result["response"]["answer"], "visible")
        self.assertEqual(events, [("generate", {"question": "q"}), ("grounded", "draft"), ("visible", "grounded")])
        def rejected(*args, **kwargs):
            raise SecurityViolation()
        llm.generate = rejected
        result = probe.run_fixed_case(qa, answer, llm, retrieval, {}, counter, case)
        self.assertEqual(result["response"]["unavailable_reason"], "security_output_rejected")
        case["selected_pool"] = []
        llm.generate = lambda *a, **k: self.fail("Empty evidence dispatched generation")
        result = probe.run_fixed_case(qa, answer, llm, retrieval, {}, counter, case)
        self.assertEqual(result["llm_call_count"], 0)

    def test_subprocess_private_drafts_failures_and_resume_without_gpu(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "soprano_qa"
            package.mkdir()
            (package / "__init__.py").write_text("")
            (package / "answer.py").write_text("")
            (package / "retrieval.py").write_text("")
            (package / "source_priority.py").write_text("SOURCE_PRIORITY = {}\n")
            (package / "llm.py").write_text('''
class Model:
    metadata = {"general.name": "fake"}
    chat_format = "fake"
    def create_chat_completion(self, **kwargs):
        return {"choices": [{"message": {"content": "PRIVATE REJECTED DRAFT"}, "finish_reason": "stop"}]}
model = Model()
def load_llama(**kwargs): return model
def _configured_chat_format(cfg): return None
''')
            (package / "service.py").write_text('''
import json, os
from . import llm
SETTINGS = json.load(open(os.environ["SOPRANO_QA_SETTINGS"]))
def validate_generation_requirements(*args): pass
def _get_index(): return None
def ask(**kwargs):
    if kwargs["question"] == "fail": raise ValueError("explicit failure")
    llm.load_llama().create_chat_completion(messages=[{"role":"system", "content":"compose"}])
    return {"answer": "Safe refusal", "generation_mode":"unavailable"}
''')
            settings = root / "settings.json"
            settings.write_text(json.dumps({"model_path": "fake", "llm": {}}))
            cases = root / "cases.json"
            cases.write_text(json.dumps([{"id": q, "question": q, "kind": "benign", "language": "en", "piece_id": "piece"}
                                         for q in ("ok", "fail")]))
            rollout = root / "usage.jsonl"
            rollout.write_text(json.dumps(observation(datetime.now(timezone.utc).timestamp())) + "\n")
            output = root / "results.jsonl"
            command = [sys.executable, str(Path(probe.__file__).resolve()), "--system-root", str(root),
                       "--settings", str(settings), "--cases", str(cases), "--output", str(output),
                       "--usage-rollout", str(rollout), "--warmup-case", "ok"]
            first = subprocess.run(command, capture_output=True, text=True, timeout=20)
            self.assertEqual(first.returncode, 0, first.stderr)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([r["status"] for r in rows], ["response", "exception"])
            self.assertEqual(rows[1]["exception"]["type"], "ValueError")
            self.assertNotIn("PRIVATE REJECTED DRAFT", output.read_text() + first.stdout)
            trace = root / "results.jsonl.private-traces.jsonl"
            self.assertIn("PRIVATE REJECTED DRAFT", trace.read_text())
            self.assertEqual(stat.S_IMODE(trace.stat().st_mode), 0o600)
            before = output.read_bytes(), trace.read_bytes()
            resumed = subprocess.run(command, capture_output=True, text=True, timeout=20)
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            self.assertEqual(before, (output.read_bytes(), trace.read_bytes()))


if __name__ == "__main__":
    unittest.main()
