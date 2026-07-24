"""Compatibility tests for the Hugging Face model downloader."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

from scripts import download_model


class DownloadModelTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
