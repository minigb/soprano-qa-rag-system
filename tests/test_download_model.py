"""Compatibility tests for the Hugging Face model downloader."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

from scripts import download_embedding_model, download_model


class DownloadModelTests(unittest.TestCase):
    def test_existing_checkpoint_must_match_its_pin_without_redownloading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.gguf")
            with open(path, "wb") as stream:
                stream.write(b"different checkpoint")
            settings = {"model_path": path, "model_sha256": hashlib.sha256(b"evaluated checkpoint").hexdigest()}
            with mock.patch.object(download_model, "parse_args", return_value=argparse.Namespace(settings=None, force=False)), \
                 mock.patch.object(download_model, "load_settings", return_value=settings), \
                 self.assertRaisesRegex(SystemExit, "checksum differs"):
                download_model.main()
            with open(path, "rb") as stream:
                self.assertEqual(stream.read(), b"different checkpoint")

    def test_download_uses_revision_and_checks_the_received_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.gguf")
            content = b"evaluated checkpoint"
            settings = {"model_path": path, "model_repo_id": "example/repository", "model_filename": "model.gguf",
                        "model_revision": "frozen-commit", "model_sha256": hashlib.sha256(content).hexdigest()}
            def download(**kwargs):
                self.assertEqual(kwargs["revision"], "frozen-commit")
                with open(path, "wb") as stream:
                    stream.write(content)
                return path
            hub = types.ModuleType("huggingface_hub")
            hub.hf_hub_download = download
            with mock.patch.object(download_model, "parse_args", return_value=argparse.Namespace(settings=None, force=False)), \
                 mock.patch.object(download_model, "load_settings", return_value=settings), \
                 mock.patch.dict(sys.modules, {"huggingface_hub": hub}):
                download_model.main()

    def test_force_uses_current_hf_hub_download_signature(self) -> None:
        captured = {}
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "models", "model.gguf")
            settings = {
                "model_path": model_path,
                "model_repo_id": "example/repository",
                "model_filename": "model.gguf",
            }

            def fake_hf_hub_download(**kwargs):
                captured.update(kwargs)
                return model_path

            fake_hub = types.ModuleType("huggingface_hub")
            fake_hub.hf_hub_download = fake_hf_hub_download
            with (
                mock.patch.object(
                    download_model,
                    "parse_args",
                    return_value=argparse.Namespace(settings=None, force=True),
                ),
                mock.patch.object(download_model, "load_settings", return_value=settings),
                mock.patch.dict(sys.modules, {"huggingface_hub": fake_hub}),
            ):
                download_model.main()

        self.assertEqual(
            captured,
            {
                "repo_id": "example/repository",
                "filename": "model.gguf",
                "local_dir": os.path.dirname(model_path),
                "force_download": True,
            },
        )

    def test_embedding_downloader_uses_configured_gguf_artifact(self) -> None:
        captured = {}
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "models", "embedding.gguf")
            settings = {
                "embedding_model_path": model_path,
                "embedding_model_repo_id": "example/embedding-repository",
                "embedding_model_filename": "embedding.gguf",
            }

            def fake_hf_hub_download(**kwargs):
                captured.update(kwargs)
                return model_path

            fake_hub = types.ModuleType("huggingface_hub")
            fake_hub.hf_hub_download = fake_hf_hub_download
            with (
                mock.patch.object(
                    download_embedding_model,
                    "parse_args",
                    return_value=argparse.Namespace(settings=None, force=True),
                ),
                mock.patch.object(
                    download_embedding_model,
                    "load_settings",
                    return_value=settings,
                ),
                mock.patch.dict(sys.modules, {"huggingface_hub": fake_hub}),
            ):
                download_embedding_model.main()

        self.assertEqual(
            captured,
            {
                "repo_id": "example/embedding-repository",
                "filename": "embedding.gguf",
                "local_dir": os.path.dirname(model_path),
                "force_download": True,
            },
        )


if __name__ == "__main__":
    unittest.main()
