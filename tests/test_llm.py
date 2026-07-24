"""Tests for local-model output sanitization."""

from __future__ import annotations

import unittest

from soprano_qa.llm import strip_thinking


class StripThinkingTests(unittest.TestCase):
    def test_complete_thinking_block_is_removed(self) -> None:
        self.assertEqual(
            strip_thinking("<think>private reasoning</think>Grounded answer"),
            "Grounded answer",
        )

    def test_truncated_thinking_block_is_never_exposed(self) -> None:
        self.assertEqual(strip_thinking("<think>private reasoning"), "")
        self.assertEqual(
            strip_thinking("Grounded answer<think>private reasoning"),
            "Grounded answer",
        )


if __name__ == "__main__":
    unittest.main()
