"""Focused tests for the read-only qualitative evaluation viewer server."""

from __future__ import annotations

import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from evaluation import server


class EvaluationViewerServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.evaluation_root = Path(self.temporary.name)
        self.assets = {
            "index.html": b"<!doctype html><title>Evaluation</title>",
            "styles.css": b"body { color: #123; }\n",
            "review-state.js": b"window.reviewState = {};\n",
            "app.js": b"window.viewerReady = true;\n",
            "manual_check_results.json": json.dumps(
                {
                    "schema_version": "1.0",
                    "summary": {"selected_questions": 50},
                    "label": "정성 검토",
                },
                ensure_ascii=False,
            ).encode("utf-8"),
            "three_piece_results.json": json.dumps(
                {
                    "schema_version": "5.0",
                    "summary": {
                        "primary_metric": {
                            "passed_cases": 1,
                            "judged_cases": 1,
                        },
                    },
                    "label": "semantic evaluation",
                },
                ensure_ascii=False,
            ).encode("utf-8"),
        }
        for filename, body in self.assets.items():
            (self.evaluation_root / filename).write_bytes(body)

        self.httpd = server.create_server(
            port=0,
            evaluation_root=self.evaluation_root,
            results_file=self.evaluation_root / "manual_check_results.json",
        )
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(
            target=self.httpd.serve_forever,
            daemon=True,
        )
        self.log_patch = mock.patch.object(
            server.EvaluationViewerRequestHandler,
            "log_message",
        )
        self.log_patch.start()
        self.thread.start()

    def tearDown(self) -> None:
        self.httpd.shutdown()
        self.thread.join(timeout=5)
        self.httpd.server_close()
        self.log_patch.stop()
        self.temporary.cleanup()

    def request(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        connection = http.client.HTTPConnection(
            "127.0.0.1",
            self.port,
            timeout=5,
        )
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        response_body = response.read()
        response_headers = {
            name.lower(): value
            for name, value in response.getheaders()
        }
        status = response.status
        connection.close()
        return status, response_headers, response_body

    def raw_host_request(
        self,
        host: str,
        *,
        origin: str | None = None,
    ) -> tuple[int, bytes]:
        connection = http.client.HTTPConnection(
            "127.0.0.1",
            self.port,
            timeout=5,
        )
        connection.putrequest("GET", "/", skip_host=True)
        connection.putheader("Host", host)
        if origin is not None:
            connection.putheader("Origin", origin)
        connection.endheaders()
        response = connection.getresponse()
        body = response.read()
        status = response.status
        connection.close()
        return status, body

    def test_default_server_binding_is_loopback(self) -> None:
        self.assertEqual(server.DEFAULT_HOST, "127.0.0.1")
        self.assertEqual(self.httpd.server_address[0], "127.0.0.1")

    def test_get_serves_only_the_exact_static_routes(self) -> None:
        expected = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/index.html": ("index.html", "text/html; charset=utf-8"),
            "/styles.css": ("styles.css", "text/css; charset=utf-8"),
            "/review-state.js": (
                "review-state.js",
                "text/javascript; charset=utf-8",
            ),
            "/app.js": ("app.js", "text/javascript; charset=utf-8"),
        }
        for route, (filename, content_type) in expected.items():
            with self.subTest(route=route):
                status, headers, body = self.request("GET", route)
                self.assertEqual(status, 200)
                self.assertEqual(headers["content-type"], content_type)
                self.assertEqual(
                    int(headers["content-length"]),
                    len(self.assets[filename]),
                )
                self.assertEqual(headers["cache-control"], "no-store")
                self.assertEqual(body, self.assets[filename])

    def test_results_api_get_and_head_serve_the_snapshot_without_caching(
        self,
    ) -> None:
        status, headers, body = self.request("GET", "/api/results")

        self.assertEqual(status, 200)
        self.assertEqual(
            headers["content-type"],
            "application/json; charset=utf-8",
        )
        self.assertEqual(headers["cache-control"], "no-store")
        self.assertEqual(headers["x-content-type-options"], "nosniff")
        self.assertEqual(
            json.loads(body),
            json.loads(self.assets["manual_check_results.json"]),
        )

        status, headers, body = self.request("HEAD", "/api/results")
        self.assertEqual(status, 200)
        self.assertEqual(
            int(headers["content-length"]),
            len(self.assets["manual_check_results.json"]),
        )
        self.assertEqual(
            headers["content-type"],
            "application/json; charset=utf-8",
        )
        self.assertEqual(headers["cache-control"], "no-store")
        self.assertEqual(body, b"")

    def test_default_results_file_prefers_semantic_then_legacy_snapshot(
        self,
    ) -> None:
        self.assertEqual(
            server.default_results_file(self.evaluation_root),
            (self.evaluation_root / "three_piece_results.json").resolve(),
        )
        (self.evaluation_root / "three_piece_results.json").unlink()
        self.assertEqual(
            server.default_results_file(self.evaluation_root),
            (self.evaluation_root / "manual_check_results.json").resolve(),
        )

    def test_server_exposes_only_the_explicit_results_file(self) -> None:
        selected = self.evaluation_root / "three_piece_results.json"
        selected_server = server.create_server(
            port=0,
            evaluation_root=self.evaluation_root,
            results_file=selected,
        )
        port = selected_server.server_address[1]
        thread = threading.Thread(
            target=selected_server.serve_forever,
            daemon=True,
        )
        thread.start()
        self.addCleanup(selected_server.server_close)
        self.addCleanup(thread.join, 5)
        self.addCleanup(selected_server.shutdown)

        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        connection.request("GET", "/api/results")
        response = connection.getresponse()
        body = response.read()
        connection.close()

        self.assertEqual(response.status, 200)
        self.assertEqual(json.loads(body)["schema_version"], "5.0")
        for route in (
            "/three_piece_results.json",
            "/manual_check_results.json",
            "/api/results?file=manual_check_results.json",
        ):
            connection = http.client.HTTPConnection(
                "127.0.0.1",
                port,
                timeout=5,
            )
            connection.request("GET", route)
            candidate = connection.getresponse()
            candidate.read()
            connection.close()
            if route.startswith("/api/results?"):
                self.assertEqual(candidate.status, 200)
            else:
                self.assertEqual(candidate.status, 404)

    def test_head_static_returns_headers_without_a_body(self) -> None:
        status, headers, body = self.request("HEAD", "/app.js")

        self.assertEqual(status, 200)
        self.assertEqual(headers["content-type"], "text/javascript; charset=utf-8")
        self.assertEqual(
            int(headers["content-length"]),
            len(self.assets["app.js"]),
        )
        self.assertEqual(body, b"")

    def test_unknown_and_traversal_routes_do_not_expose_files(self) -> None:
        secret = self.evaluation_root.parent / "viewer-secret.txt"
        secret.write_text("do not serve", encoding="utf-8")
        self.addCleanup(secret.unlink, missing_ok=True)
        routes = (
            "/manual_check_results.json",
            "/viewer-secret.txt",
            "/../viewer-secret.txt",
            "/%2e%2e/viewer-secret.txt",
            "/styles.css/extra",
            "/evaluation/manual_check_results.json",
        )

        for route in routes:
            with self.subTest(route=route):
                status, headers, body = self.request("GET", route)
                self.assertEqual(status, 404)
                self.assertEqual(
                    headers["content-type"],
                    "application/json; charset=utf-8",
                )
                self.assertEqual(json.loads(body)["code"], "not_found")
                self.assertNotIn(b"do not serve", body)

    def test_write_and_preflight_methods_are_not_allowed(self) -> None:
        original = self.assets["manual_check_results.json"]
        for method in ("POST", "PUT", "PATCH", "DELETE", "OPTIONS", "TRACE"):
            with self.subTest(method=method):
                status, headers, body = self.request(
                    method,
                    "/api/results",
                    body=b'{"replace": true}',
                    headers={"Content-Type": "application/json"},
                )
                self.assertEqual(status, 405)
                self.assertEqual(headers["allow"], "GET, HEAD")
                self.assertEqual(
                    json.loads(body)["code"],
                    "method_not_allowed",
                )
        self.assertEqual(
            (self.evaluation_root / "manual_check_results.json").read_bytes(),
            original,
        )

    def test_missing_results_file_returns_a_json_404(self) -> None:
        (self.evaluation_root / "manual_check_results.json").unlink()

        status, headers, body = self.request("GET", "/api/results")

        self.assertEqual(status, 404)
        self.assertEqual(
            headers["content-type"],
            "application/json; charset=utf-8",
        )
        self.assertEqual(json.loads(body)["code"], "results_not_found")

    def test_non_loopback_host_and_cross_origin_requests_are_rejected(
        self,
    ) -> None:
        status, body = self.raw_host_request("attacker.example")
        self.assertEqual(status, 403)
        self.assertEqual(json.loads(body)["code"], "non_loopback_request")

        host = f"127.0.0.1:{self.port}"
        status, body = self.raw_host_request(
            host,
            origin="https://attacker.example",
        )
        self.assertEqual(status, 403)
        self.assertEqual(json.loads(body)["code"], "cross_origin_request")

        status, body = self.raw_host_request(
            host,
            origin="http://127.0.0.1:not-a-port",
        )
        self.assertEqual(status, 403)
        self.assertEqual(json.loads(body)["code"], "cross_origin_request")

        status, body = self.raw_host_request(
            host,
            origin=f"http://127.0.0.1:{self.port}",
        )
        self.assertEqual(status, 200)
        self.assertEqual(body, self.assets["index.html"])


class EvaluationViewerSourceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.evaluation_root = Path(__file__).resolve().parents[1] / "evaluation"
        cls.app_source = (
            cls.evaluation_root / "app.js"
        ).read_text(encoding="utf-8")
        cls.state_source = (
            cls.evaluation_root / "review-state.js"
        ).read_text(encoding="utf-8")
        cls.index_source = (
            cls.evaluation_root / "index.html"
        ).read_text(encoding="utf-8")

    def test_schema_eight_and_in_progress_manual_lock_are_explicit(self) -> None:
        self.assertIn(
            '["1.0", "2.0", "3.0", "4.0", "5.0", "6.0", "7.0", "8.0"]',
            self.app_source,
        )
        self.assertIn(
            '["2.0", "3.0", "4.0", "5.0", "6.0", "7.0", "8.0"]',
            self.app_source,
        )
        self.assertIn(
            "manualReviewEnabledForPayload",
            self.app_source,
        )
        self.assertIn(
            "judge_complete",
            self.state_source,
        )
        self.assertIn(
            "실행 중 스냅샷: 수동 semantic-fidelity 상태 저장 비활성화",
            self.app_source,
        )

    def test_exact_case_authority_precedes_judge_and_secondary_retrieval(
        self,
    ) -> None:
        comparison_call = self.app_source.index(
            "panel.appendChild(renderCaseAuthorityComparison(caseItem.run))"
        )
        judge_call = self.app_source.index(
            "panel.appendChild(renderSemanticEvaluation(caseItem.run))"
        )
        pipeline_call = self.app_source.index(
            "renderPipeline(caseItem, result)",
            comparison_call,
        )
        self.assertLess(comparison_call, judge_call)
        self.assertLess(judge_call, pipeline_call)
        self.assertIn(
            "Original annotation (global source answer)",
            self.app_source,
        )
        self.assertIn(
            "Secondary linked-KU lineage diagnostics",
            self.app_source,
        )

    def test_hybrid_range_audit_fields_and_manual_verdict_are_visible(
        self,
    ) -> None:
        for expected in (
            "A · selected-range claims",
            "X · other-range claims",
            "raw LLM:",
            "NLI entailment",
            "NLI neutral",
            "NLI contradiction",
            "Clause-level NLI localization",
            "links:",
            "Embedding contrast delta",
            "Decision reasons",
            "Hybrid guard thresholds",
            "reference_review",
            "Manual semantic-fidelity verdict",
            "Semantic fidelity",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, self.app_source)
        self.assertIn(
            "Manual semantic fidelity",
            self.index_source,
        )

    def test_deterministic_semantic_guard_rejections_are_visible(
        self,
    ) -> None:
        for expected in (
            "covered_guard_rejected",
            "Literal pronunciation-anchor guard "
            "(raw coverage not accepted)",
            "missing literal anchors:",
            "Required-claim NLI guard "
            "(atomic or S-substitution coverage not corroborated)",
            "guard scope:",
            "Unanchored critical-error claims (manual review)",
            "Relationship labels returned as critical errors "
            "(manual review)",
            "Novel numeric-fact guard",
            "novel numeric anchors:",
            "Range deterministic parser/consistency normalizations",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, self.app_source)

    def test_supplemental_authority_and_rejections_are_auditable(
        self,
    ) -> None:
        for expected in (
            "Target R claims — completeness",
            "Retrieved S claims — factuality only",
            "Authenticated retrieved expert evidence",
            "Rejected supplemental evidence",
            "value.evidence_id",
            "severity:",
            "Supplemental-evidence integrity prevents automatic pass.",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, self.app_source)
        self.assertIn(
            'item.scope_authority !== "retrieved_expert_factuality_only"',
            self.state_source,
        )

    def test_review_disclosure_authentication_audit_is_visible(self) -> None:
        for expected in (
            "review_disclosure_validation",
            "Review-disclosure authentication audit",
            "status: reviewDisclosureValidation.status",
            "evidence_ids: listOrEmpty(reviewDisclosureValidation.evidence_ids)",
            "expected_text: reviewDisclosureValidation.expected_text",
            "stripped_from_semantic_candidate",
            "auto_pass_eligible",
            "failure_reason",
            "semantic_candidate_empty",
            "additional_disclosure_remains",
            "An authenticated review disclosure is evaluator metadata",
            "and is excluded from musical-answer judging.",
            "Review-disclosure authentication prevents automatic pass.",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, self.app_source)


if __name__ == "__main__":
    unittest.main()
