#!/usr/bin/env python3
"""Serve the local, read-only RAG expected-vs-generated answer viewer."""

from __future__ import annotations

import argparse
import json
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Final
from urllib.parse import unquote, urlparse, urlsplit


EVALUATION_ROOT: Final = Path(__file__).resolve().parent
DEFAULT_HOST: Final = "127.0.0.1"
DEFAULT_PORT: Final = 8766
LOCAL_HOSTS: Final = frozenset({"127.0.0.1", "localhost", "::1"})

STATIC_ROUTES: Final = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/review-state.js": ("review-state.js", "text/javascript; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}
RESULTS_ROUTE: Final = "/api/results"


class EvaluationViewerServer(ThreadingHTTPServer):
    """HTTP server carrying the evaluation result snapshot to expose."""

    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        evaluation_root: Path = EVALUATION_ROOT,
        results_file: Path,
    ) -> None:
        self.evaluation_root = evaluation_root.resolve()
        self.results_file = results_file.resolve()
        super().__init__(server_address, EvaluationViewerRequestHandler)


class EvaluationViewerRequestHandler(BaseHTTPRequestHandler):
    """Serve only the evaluation viewer and its immutable result snapshot."""

    server_version = "SopranoQAEvaluationViewer/1.0"

    def log_message(self, format_string: str, *args: object) -> None:
        sys.stderr.write(
            f"{self.address_string()} - {format_string % args}\n"
        )

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self'; "
            "connect-src 'self'; "
            "img-src 'self' data:; "
            "object-src 'none'; "
            "base-uri 'none'; "
            "frame-ancestors 'none'",
        )
        super().end_headers()

    def _send_bytes(
        self,
        body: bytes,
        *,
        content_type: str,
        status: HTTPStatus = HTTPStatus.OK,
        head_only: bool = False,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        if head_only:
            return
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_json_error(
        self,
        message: str,
        *,
        code: str,
        status: HTTPStatus,
        head_only: bool = False,
        headers: dict[str, str] | None = None,
    ) -> None:
        body = json.dumps(
            {"error": message, "code": code},
            ensure_ascii=False,
        ).encode("utf-8")
        self._send_bytes(
            body,
            content_type="application/json; charset=utf-8",
            status=status,
            head_only=head_only,
            headers=headers,
        )

    def _allow_local_request(self, *, head_only: bool) -> bool:
        host = self.headers.get("Host", "")
        try:
            parsed_host = urlsplit(f"//{host}")
            hostname = parsed_host.hostname
            host_port = parsed_host.port
        except ValueError:
            hostname = None
            host_port = None
        if hostname not in LOCAL_HOSTS:
            self._send_json_error(
                "This viewer accepts only loopback requests.",
                code="non_loopback_request",
                status=HTTPStatus.FORBIDDEN,
                head_only=head_only,
            )
            return False

        origin = self.headers.get("Origin")
        if not origin:
            return True
        try:
            parsed_origin = urlparse(origin)
            origin_port = parsed_origin.port
        except ValueError:
            parsed_origin = None
            origin_port = None
        if (
            parsed_origin is None
            or parsed_origin.scheme != "http"
            or parsed_origin.hostname not in LOCAL_HOSTS
            or parsed_origin.hostname != hostname
            or origin_port != host_port
        ):
            self._send_json_error(
                "Cross-origin requests are not allowed.",
                code="cross_origin_request",
                status=HTTPStatus.FORBIDDEN,
                head_only=head_only,
            )
            return False
        return True

    def _send_file(
        self,
        path: Path,
        *,
        content_type: str,
        head_only: bool,
        missing_code: str = "not_found",
    ) -> None:
        try:
            body = path.read_bytes()
        except FileNotFoundError:
            self._send_json_error(
                "Not found.",
                code=missing_code,
                status=HTTPStatus.NOT_FOUND,
                head_only=head_only,
            )
            return
        except OSError as error:
            self._send_json_error(
                f"Could not read viewer data: {error}",
                code="file_read_failed",
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
                head_only=head_only,
            )
            return
        self._send_bytes(
            body,
            content_type=content_type,
            head_only=head_only,
        )

    def _dispatch_read(self, *, head_only: bool) -> None:
        if not self._allow_local_request(head_only=head_only):
            return

        path = unquote(urlsplit(self.path).path)
        evaluation_root = self.server.evaluation_root
        if path == RESULTS_ROUTE:
            self._send_file(
                self.server.results_file,
                content_type="application/json; charset=utf-8",
                head_only=head_only,
                missing_code="results_not_found",
            )
            return

        static_route = STATIC_ROUTES.get(path)
        if static_route is not None:
            filename, content_type = static_route
            self._send_file(
                evaluation_root / filename,
                content_type=content_type,
                head_only=head_only,
            )
            return

        self._send_json_error(
            "Not found.",
            code="not_found",
            status=HTTPStatus.NOT_FOUND,
            head_only=head_only,
        )

    def _method_not_allowed(self) -> None:
        if not self._allow_local_request(head_only=False):
            return
        self._send_json_error(
            "Only GET and HEAD are supported.",
            code="method_not_allowed",
            status=HTTPStatus.METHOD_NOT_ALLOWED,
            headers={"Allow": "GET, HEAD"},
        )

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch_read(head_only=False)

    def do_HEAD(self) -> None:  # noqa: N802
        self._dispatch_read(head_only=True)

    def do_POST(self) -> None:  # noqa: N802
        self._method_not_allowed()

    def do_PUT(self) -> None:  # noqa: N802
        self._method_not_allowed()

    def do_PATCH(self) -> None:  # noqa: N802
        self._method_not_allowed()

    def do_DELETE(self) -> None:  # noqa: N802
        self._method_not_allowed()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._method_not_allowed()

    def do_TRACE(self) -> None:  # noqa: N802
        self._method_not_allowed()


def create_server(
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    evaluation_root: Path = EVALUATION_ROOT,
    results_file: Path,
) -> EvaluationViewerServer:
    """Create a viewer server; callers remain responsible for serving it."""

    return EvaluationViewerServer(
        (host, port),
        evaluation_root=evaluation_root,
        results_file=results_file,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--results-file",
        type=Path,
        required=True,
        help=(
            "Evaluation JSON exposed at /api/results (current synthesized "
            "RAG+LLM or retained qualitative/schema-8 snapshot)."
        ),
    )
    arguments = parser.parse_args()
    if not arguments.results_file.is_file():
        parser.error(
            "--results-file must point to an existing evaluation JSON snapshot"
        )

    server = create_server(
        port=arguments.port,
        results_file=arguments.results_file,
    )
    host, port = server.server_address[:2]
    print(
        f"Serving RAG answer comparison at http://{host}:{port}/",
        flush=True,
    )
    print(
        f"Read-only data source: {server.results_file}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
