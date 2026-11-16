from __future__ import annotations

import json
import os
import tempfile
import unittest
from types import SimpleNamespace

from soprano_qa.answer import build_evidence_notices, supplement_web_coaching_evidence
from soprano_qa.corpus import (
    WEB_COACHING_RECORD_KEYS,
    build_corpus,
    build_web_coaching_records,
    corpus_input_paths,
    is_web_record,
    validate_release_corpus,
)
from soprano_qa.settings import load_settings

# A stand-in for the dataset's web_coaching/tools/validate.py. The corpus
# builder relies only on its collect(root) contract.
FAKE_VALIDATOR = """
import json
from pathlib import Path


def collect(root):
    data = json.loads((Path(root) / "fixture.json").read_text(encoding="utf-8"))
    return data["sources"], data["units_by_piece"], data["errors"], []
"""

SOURCE = {
    "source_id": "wc-die-forelle-501",
    "piece_ids": ["die-forelle"],
    "kind": "video",
    "url": "https://www.youtube.com/watch?v=example",
    "title": "Die Forelle lesson",
    "author": "Example channel",
    "language": "ko",
    "retrieved_at": "2026-10-12",
    "evidence_file": "transcripts/wc-die-forelle-501.json",
    "notes": "",
}
UNIT = {
    "unit_id": "die-forelle-wc-5001",
    "category": "interpretation",
    "claim_ko": "강사는 3절의 Doch부터 긴장감을 가지고 노래하라고 조언한다.",
    "excerpt": "verbatim span",
    "source_id": "wc-die-forelle-501",
    "locator": "[12:01-12:20]",
    "lyric_anchor": "Doch endlich",
    "notes": "",
}


def write_lineage(dataset_root: str, *, units=None, errors=None) -> None:
    tools = os.path.join(dataset_root, "web_coaching", "tools")
    os.makedirs(tools)
    with open(os.path.join(tools, "validate.py"), "w", encoding="utf-8") as file:
        file.write(FAKE_VALIDATOR)
    fixture = {
        "sources": [SOURCE],
        "units_by_piece": {"die-forelle": units if units is not None else [UNIT]},
        "errors": errors or [],
    }
    with open(
        os.path.join(dataset_root, "web_coaching", "fixture.json"), "w", encoding="utf-8"
    ) as file:
        json.dump(fixture, file, ensure_ascii=False)


class WebCoachingRecordTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = self.temp_dir.name

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_valid_lineage_becomes_exact_schema_records(self) -> None:
        write_lineage(self.root)
        records = build_web_coaching_records(self.root)
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(tuple(sorted(record)), tuple(sorted(WEB_COACHING_RECORD_KEYS)))
        self.assertEqual(record["evidence_type"], "web_coaching")
        self.assertTrue(is_web_record(record))
        self.assertEqual(record["answer"], UNIT["claim_ko"])
        self.assertEqual(record["knowledge_scope"], "performance_guidance")
        self.assertEqual(record["measure_range"], [])
        self.assertEqual(record["measure_scope"], "global")
        self.assertIn("Doch endlich", record["retrieval_text"])
        self.assertEqual(record["web_coaching_source"]["locator"], UNIT["locator"])
        validate_release_corpus(records)

    def test_validator_errors_refuse_the_whole_lineage(self) -> None:
        write_lineage(self.root, errors=["die-forelle-wc-5001: excerpt not found verbatim"])
        with self.assertRaisesRegex(ValueError, "failed validation"):
            build_web_coaching_records(self.root)

    def test_unknown_category_and_empty_lineage_are_rejected(self) -> None:
        write_lineage(self.root, units=[{**UNIT, "category": "gossip"}])
        with self.assertRaisesRegex(ValueError, "unsupported category"):
            build_web_coaching_records(self.root)
        other = tempfile.TemporaryDirectory()
        self.addCleanup(other.cleanup)
        write_lineage(other.name, units=[])
        with self.assertRaisesRegex(ValueError, "produced no records"):
            build_web_coaching_records(other.name)

    def test_release_corpus_rejects_unexpected_web_coaching_fields(self) -> None:
        write_lineage(self.root)
        record = dict(build_web_coaching_records(self.root)[0], excerpt="copied text")
        with self.assertRaises(ValueError):
            validate_release_corpus([record])


def fake_result(record_id: str, evidence_type: str) -> SimpleNamespace:
    return SimpleNamespace(
        record={"id": record_id, "evidence_type": evidence_type},
        scope_match="general_evidence",
    )


class WebCoachingSupplementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.expert = [fake_result("ku-1", "expert_annotation")]
        self.candidates = [
            fake_result("ku-1", "expert_annotation"),
            fake_result("wc-1", "web_coaching"),
            fake_result("web-1", "web_database"),
            fake_result("wc-2", "web_coaching"),
            fake_result("wc-3", "web_coaching"),
        ]

    def ids(self, results):
        return [result.record["id"] for result in results]

    def test_appends_best_ranked_coaching_after_expert_evidence(self) -> None:
        results = supplement_web_coaching_evidence(
            self.expert, self.candidates, measure_ranges=[], limit=2
        )
        self.assertEqual(self.ids(results), ["ku-1", "wc-1", "wc-2"])

    def test_disabled_ranged_and_duplicate_cases_are_unchanged(self) -> None:
        for limit, ranges in ((0, []), (3, [[1, 4]])):
            results = supplement_web_coaching_evidence(
                self.expert, self.candidates, measure_ranges=ranges, limit=limit
            )
            self.assertEqual(self.ids(results), ["ku-1"])
        already = [*self.expert, self.candidates[1]]
        results = supplement_web_coaching_evidence(
            already, self.candidates, measure_ranges=[], limit=1
        )
        self.assertEqual(self.ids(results), ["ku-1", "wc-1", "wc-2"])


class WebCoachingSupplementGateTests(WebCoachingSupplementTests):
    def dense(self, record_id: str, evidence_type: str, score: float) -> SimpleNamespace:
        result = fake_result(record_id, evidence_type)
        result.dense_score = score
        return result

    def test_dense_margin_gate_requires_a_closer_coaching_match(self) -> None:
        coaching = [self.dense("wc-1", "web_coaching", 0.60), self.dense("wc-2", "web_coaching", 0.55)]
        reference = [self.dense("ku-1", "expert_annotation", 0.58)]
        kwargs = dict(measure_ranges=[], limit=2, gate="dense_margin", reference_candidates=reference)
        self.assertEqual(
            self.ids(supplement_web_coaching_evidence(self.expert, coaching, margin=0.05, **kwargs)),
            ["ku-1"],
        )
        self.assertEqual(
            self.ids(supplement_web_coaching_evidence(self.expert, coaching, margin=0.0, **kwargs)),
            ["ku-1", "wc-1", "wc-2"],
        )

    def test_unknown_gate_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            supplement_web_coaching_evidence(
                self.expert, self.candidates, measure_ranges=[], limit=2, gate="sometimes"
            )


class WebCoachingSettingsTests(unittest.TestCase):
    def test_inputs_follow_the_enabled_lineages(self) -> None:
        settings = dict(load_settings(), web_coaching={"enabled": False})
        default_labels = [label for label, _ in corpus_input_paths(settings)]
        self.assertFalse(any(label.startswith("web_coaching") for label in default_labels))
        self.assertTrue(any(label.startswith("database") for label in default_labels))

        coaching_only = dict(settings, web_export_files=[], web_coaching={"enabled": True})
        labels = [label for label, _ in corpus_input_paths(coaching_only)]
        self.assertFalse(any(label.startswith("database") for label in labels))
        if os.path.isdir(os.path.join(settings["dataset_root"], "web_coaching")):
            self.assertIn(os.path.join("web_coaching", "tools", "validate.py"), labels)


class WebCoachingDatasetIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        settings = load_settings()
        root = os.path.join(settings["dataset_root"], "web_coaching")
        if not os.path.isfile(os.path.join(root, "tools", "validate.py")):
            raise unittest.SkipTest("Dataset web_coaching lineage is not available")
        cls.temp_dir = tempfile.TemporaryDirectory()
        build_settings = dict(
            settings,
            corpus_path=os.path.join(cls.temp_dir.name, "corpus.json"),
            stats_path=os.path.join(cls.temp_dir.name, "stats.json"),
            web_export_files=[],
            web_coaching={"enabled": True},
        )
        cls.result = build_corpus(build_settings)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def test_coaching_only_corpus_has_expert_and_coaching_records(self) -> None:
        counts = self.result["stats"]["records_by_evidence_type"]
        self.assertNotIn("web_database", counts)
        self.assertGreater(counts.get("web_coaching", 0), 0)
        self.assertGreater(counts.get("expert_annotation", 0), 0)
        self.assertEqual(self.result["stats"]["web_export_files"], [])
        for record in self.result["records"]:
            if record["evidence_type"] == "web_coaching":
                self.assertEqual(
                    tuple(sorted(record)), tuple(sorted(WEB_COACHING_RECORD_KEYS))
                )


if __name__ == "__main__":
    unittest.main()
