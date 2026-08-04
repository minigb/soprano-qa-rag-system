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


class StructuredGenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        llm.load_llama.cache_clear()

    def tearDown(self) -> None:
        llm.load_llama.cache_clear()

    def test_json_generation_uses_one_grammar_constrained_call(self) -> None:
        backend = mock.Mock()
        backend.create_chat_completion.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"selected_ids":["ku-001"]}'
                    }
                }
            ]
        }
        schema = {
            "type": "object",
            "properties": {
                "selected_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                }
            },
            "required": ["selected_ids"],
        }
        with mock.patch.object(llm, "load_llama", return_value=backend):
            result = llm.generate_json(
                "/models/answer.gguf",
                [{"role": "user", "content": "select"}],
                {"n_ctx": 4096, "max_tokens": 900, "top_p": 0.8},
                schema=schema,
            )

        self.assertEqual(result, {"selected_ids": ["ku-001"]})
        backend.create_chat_completion.assert_called_once_with(
            messages=[{"role": "user", "content": "select"}],
            temperature=0.0,
            top_p=0.8,
            max_tokens=384,
            response_format={"type": "json_object", "schema": schema},
        )

    def test_invalid_json_raises_without_retry_or_parser_fallback(self) -> None:
        backend = mock.Mock()
        backend.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "not json"}}]
        }
        with (
            mock.patch.object(llm, "load_llama", return_value=backend),
            self.assertRaises(llm.StructuredGenerationError),
        ):
            llm.generate_json(
                "/models/answer.gguf",
                [{"role": "user", "content": "select"}],
                {},
                schema={"type": "object"},
            )

        backend.create_chat_completion.assert_called_once()


if __name__ == "__main__":
    unittest.main()
