"""Fail-closed checks for the sole local answer-generation backend."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from soprano_qa import llm


class GenerationRequirementTests(unittest.TestCase):
    def test_missing_checkpoint_fails_before_backend_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.gguf"
            with (
                mock.patch.object(llm, "load_llama") as load,
                self.assertRaisesRegex(
                    llm.GenerationBackendUnavailable,
                    "Required answer-generation checkpoint not found",
                ),
            ):
                llm.validate_generation_requirements(
                    str(missing),
                    {"n_ctx": 4096},
                )

        load.assert_not_called()

    def test_backend_or_invalid_gguf_error_is_not_suppressed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "answer.gguf"
            checkpoint.write_bytes(b"not-a-real-checkpoint")
            backend_error = RuntimeError("llama-cpp backend unavailable")
            with (
                mock.patch.object(
                    llm,
                    "load_llama",
                    side_effect=backend_error,
                ) as load,
                self.assertRaisesRegex(
                    llm.GenerationBackendUnavailable,
                    "backend/model could not be loaded",
                ) as raised,
            ):
                llm.validate_generation_requirements(
                    str(checkpoint),
                    {
                        "n_ctx": 4096,
                        "n_gpu_layers": 0,
                        "chat_format": "chatml",
                    },
                )

        self.assertIs(raised.exception.__cause__, backend_error)
        load.assert_called_once_with(
            model_path=str(checkpoint),
            n_ctx=4096,
            n_gpu_layers=0,
            chat_format="chatml",
        )

    def test_validated_backend_is_eagerly_loaded_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "answer.gguf"
            checkpoint.write_bytes(b"checkpoint")
            with mock.patch.object(llm, "load_llama") as load:
                llm.validate_generation_requirements(
                    str(checkpoint),
                    {
                        "n_ctx": 8192,
                        "n_gpu_layers": -1,
                        "chat_format": "chatml",
                    },
                )

        load.assert_called_once_with(
            model_path=str(checkpoint),
            n_ctx=8192,
            n_gpu_layers=-1,
            chat_format="chatml",
        )

    def test_omitted_chat_format_uses_embedded_model_template(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "answer.gguf"
            checkpoint.write_bytes(b"checkpoint")
            with mock.patch.object(llm, "load_llama") as load:
                llm.validate_generation_requirements(
                    str(checkpoint),
                    {"n_ctx": 8192, "n_gpu_layers": -1},
                )

        load.assert_called_once_with(
            model_path=str(checkpoint),
            n_ctx=8192,
            n_gpu_layers=-1,
            chat_format=None,
        )


if __name__ == "__main__":
    unittest.main()
