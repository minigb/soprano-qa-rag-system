"""Focused tests for the local evaluation comparison viewer."""

from __future__ import annotations

from http.client import HTTPConnection
import json
from pathlib import Path
import tempfile
import threading
import unittest

from evaluation import server as viewer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ROOT = PROJECT_ROOT / "evaluation"


class EvaluationViewerServerTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.results_file = Path(temporary.name) / "results.json"
        self.payload = {
            "artifact_type": (
                "soprano_qa_synthesized_hybrid_rag_llm_evaluation"
            ),
            "schema_version": "1.0",
            "results": [],
        }
        self.results_file.write_text(
            json.dumps(self.payload, ensure_ascii=False),
            encoding="utf-8",
        )
        self.server = viewer.create_server(
            port=0,
            evaluation_root=EVALUATION_ROOT,
            results_file=self.results_file,
        )
        self.addCleanup(self.server.server_close)
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            daemon=True,
        )
        self.thread.start()
        self.addCleanup(self._stop_server)

    def _stop_server(self) -> None:
        self.server.shutdown()
        self.thread.join(timeout=2)

    def request(self, path: str) -> tuple[int, dict[str, str], bytes]:
        port = self.server.server_address[1]
        connection = HTTPConnection("127.0.0.1", port, timeout=2)
        self.addCleanup(connection.close)
        connection.request("GET", path, headers={"Host": f"127.0.0.1:{port}"})
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()

    def test_serves_the_selected_result_file_without_rewriting_it(self) -> None:
        status, headers, body = self.request("/api/results")

        self.assertEqual(status, 200)
        self.assertEqual(body, self.results_file.read_bytes())
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertIn("application/json", headers["Content-Type"])

    def test_served_viewer_has_synthesized_comparison_filters(self) -> None:
        status, _, body = self.request("/")
        html = body.decode("utf-8")

        self.assertEqual(status, 200)
        self.assertIn('id="generation-mode-filter"', html)
        self.assertIn('id="run-status-filter"', html)
        self.assertIn('id="variant-filter"', html)

    def test_assets_support_current_synthesis_and_retained_schema_8_1(self) -> None:
        app = (EVALUATION_ROOT / "app.js").read_text(encoding="utf-8")
        review_state = (EVALUATION_ROOT / "review-state.js").read_text(
            encoding="utf-8"
        )

        self.assertIn('"8.1"', app)
        self.assertIn("Human expected/reference answer", app)
        self.assertIn("synthesized_variant_id", app)
        self.assertIn("case_reference_authority", review_state)


if __name__ == "__main__":
    unittest.main()
