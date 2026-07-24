from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from types import SimpleNamespace

from soprano_qa.answer import (
    build_context,
    build_evidence_notices,
    ensure_corpus,
    finalize_answer_citations,
    format_evidence_notice,
    generate_with_context_retry,
)
from soprano_qa.corpus import (
    apply_overrides,
    build_corpus,
    build_facts_only_audit_index,
    build_web_claim_index,
    build_web_source_index,
    effective_source_asset_class,
    expected_inherited_licenses,
    file_sha256,
    iter_jsonl,
    normalize_ranges,
    resolve_claim_assets,
    validate_claim_asset_rights,
    validate_spellcheck_lineage,
    validate_web_chunk_lineage,
)
from soprano_qa.retrieval import BM25Index
from soprano_qa.settings import PROJECT_ROOT, load_settings


class CombinedCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        settings = load_settings()
        if not os.path.isdir(settings["dataset_root"]):
            raise unittest.SkipTest("External soprano QA dataset is not available")
        cls.temp_dir = tempfile.TemporaryDirectory()
        build_settings = dict(settings)
        build_settings["corpus_path"] = os.path.join(cls.temp_dir.name, "corpus.json")
        build_settings["stats_path"] = os.path.join(cls.temp_dir.name, "stats.json")
        result = build_corpus(build_settings)
        cls.build_settings = build_settings
        cls.records = result["records"]
        cls.stats = result["stats"]
        cls.index = BM25Index(cls.records)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def test_settings_keep_artifacts_in_this_worktree(self) -> None:
        settings = load_settings()
        self.assertEqual(settings["corpus_path"], os.path.join(PROJECT_ROOT, "data", "corpus.json"))
        self.assertEqual(
            settings["model_path"],
            os.path.join(PROJECT_ROOT, "models", "Qwen3-8B-Q4_K_M.gguf"),
        )
        self.assertTrue(settings["dataset_root"].endswith("soprano-qa-dataset"))

    def test_combines_only_answer_eligible_lineages(self) -> None:
        self.assertEqual(self.stats["corpus_schema_version"], 4)
        self.assertEqual(len(self.stats["input_fingerprint"]), 64)
        self.assertEqual(len(self.stats["corpus_sha256"]), 64)
        self.assertEqual(
            self.stats["corpus_sha256"],
            file_sha256(self.build_settings["corpus_path"]),
        )
        self.assertTrue(self.stats["dataset_root"].endswith("soprano-qa-dataset"))
        self.assertEqual(self.stats["total_records"], 211)
        self.assertEqual(self.stats["retrievable_records"], 210)
        self.assertEqual(
            self.stats["records_by_evidence_type"],
            {"expert_annotation": 110, "web_database": 101},
        )
        self.assertEqual(
            self.stats["web_records_by_usage_class"],
            {"research_conditional": 20, "research_open": 81},
        )
        self.assertNotIn("catalog_only", self.stats["web_records_by_usage_class"])
        self.assertEqual(len({record["id"] for record in self.records}), 211)
        self.assertTrue(all(record.get("relevance_text") for record in self.records))
        expert_source_ids = {
            source_id
            for record in self.records
            if record["evidence_type"] == "expert_annotation"
            for source_id in record["source_ids"]
        }
        self.assertEqual(len(expert_source_ids), 135)

    def test_strict_measure_ranges_reject_lossy_values(self) -> None:
        for invalid in ([[1.9, 4]], [[True, 2]], [["1", 2]]):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "must be integers"):
                    normalize_ranges(invalid)

    def test_feature_overrides_require_safe_targets_and_consistent_scope(self) -> None:
        record = {
            "id": "sqa-test",
            "source_ids": ["expert-source-1"],
            "topic": "호흡",
            "answer": "호흡을 준비한다.",
            "measure_range": [],
            "measure_scope": "global",
            "features": [],
        }
        with self.assertRaisesRegex(ValueError, "non-empty list"):
            apply_overrides([copy.deepcopy(record)], [{"match_source_ids": []}])
        with self.assertRaisesRegex(ValueError, "contradicts"):
            apply_overrides(
                [copy.deepcopy(record)],
                [
                    {
                        "id": "sqa-test",
                        "measure_range": [[1, 2]],
                        "measure_scope": "global",
                    }
                ],
            )
        with self.assertRaisesRegex(ValueError, "Recurring"):
            apply_overrides(
                [copy.deepcopy(record)],
                [{"id": "sqa-test", "measure_scope": "recurring"}],
            )
        recurring = copy.deepcopy(record)
        self.assertEqual(
            apply_overrides(
                [recurring],
                [
                    {
                        "id": "sqa-test",
                        "measure_range": [[4, 4], [12, 12]],
                        "measure_scope": "recurring",
                    }
                ],
            ),
            1,
        )
        self.assertEqual(recurring["measure_scope"], "recurring")
        with self.assertRaisesRegex(ValueError, "Unknown feature override fields"):
            apply_overrides(
                [copy.deepcopy(record)],
                [{"id": "sqa-test", "measure_ranges": [[1, 2]]}],
            )
        with self.assertRaisesRegex(ValueError, "at least one mutation"):
            apply_overrides([copy.deepcopy(record)], [{"id": "sqa-test"}])
        second_record = copy.deepcopy(record)
        second_record["id"] = "sqa-test-2"
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            apply_overrides(
                [copy.deepcopy(record), second_record],
                [
                    {
                        "match_source_ids": ["expert-source-1"],
                        "measure_range": [[1, 2]],
                    }
                ],
            )

    def test_spellcheck_phase_cannot_change_expert_structure(self) -> None:
        phase2 = [
            {
                "source_ids": ["expert-1"],
                "topic": "발음",
                "measure_range": [[1, 2]],
                "question": "두번째 음은?",
                "question_source": "expert",
                "answer": "두번째 음을 준비한다.",
            }
        ]
        phase3 = copy.deepcopy(phase2)
        phase3[0]["question"] = "두 번째 음은?"
        phase3[0]["answer"] = "두 번째 음을 준비한다."
        validate_spellcheck_lineage(phase2, phase3, "phase2.json", "phase3.json")
        forged = copy.deepcopy(phase3)
        forged[0]["measure_range"] = [[3, 4]]
        with self.assertRaisesRegex(ValueError, "structural field measure_range"):
            validate_spellcheck_lineage(phase2, forged, "phase2.json", "phase3.json")
        duplicate = copy.deepcopy(phase3)
        duplicate[0]["source_ids"] = ["expert-1", "expert-1"]
        with self.assertRaisesRegex(ValueError, "source_ids"):
            validate_spellcheck_lineage(duplicate, duplicate, "phase2.json", "phase3.json")

    def test_current_stats_cannot_bless_a_corrupt_corpus(self) -> None:
        settings = dict(load_settings())
        with tempfile.TemporaryDirectory() as temp_dir:
            settings["corpus_path"] = os.path.join(temp_dir, "corpus.json")
            settings["stats_path"] = os.path.join(temp_dir, "stats.json")
            build_corpus(settings)
            with open(settings["corpus_path"], "w", encoding="utf-8") as f:
                json.dump([], f)
            ensure_corpus(settings, rebuild=False)
            with open(settings["corpus_path"], encoding="utf-8") as f:
                rebuilt = json.load(f)
            self.assertEqual(len(rebuilt), 211)
            with open(settings["stats_path"], encoding="utf-8") as f:
                rebuilt_stats = json.load(f)
            self.assertEqual(
                rebuilt_stats["corpus_sha256"],
                file_sha256(settings["corpus_path"]),
            )
            with open(settings["stats_path"], "w", encoding="utf-8") as f:
                json.dump([], f)
            ensure_corpus(settings, rebuild=False)
            with open(settings["stats_path"], encoding="utf-8") as f:
                self.assertIsInstance(json.load(f), dict)

    def test_generation_context_overflow_retries_whole_records_and_stays_aligned(self) -> None:
        attempted_counts = []

        def fake_generator(messages):
            evidence_count = messages[1]["content"].count("Evidence ")
            attempted_counts.append(evidence_count)
            if evidence_count > 2:
                raise ValueError(
                    "Requested tokens (11417) exceed context window of 8192"
                )
            return "grounded answer [E1]"

        raw_answer, results, messages, context_limited = generate_with_context_retry(
            self.index,
            query="음악 곡 노래 작품 가사 표현 연주",
            piece="die-forelle",
            measure_ranges=[],
            measures="",
            topic=None,
            top_k=1000,
            generator=fake_generator,
        )
        self.assertTrue(results)
        self.assertTrue(context_limited)
        self.assertGreater(attempted_counts[0], 2)
        self.assertLessEqual(len(results), 2)
        self.assertEqual(raw_answer, "grounded answer [E1]")
        self.assertEqual(messages[1]["content"].count("Evidence "), len(results))
        notice_ids = {
            notice["record_id"] for notice in build_evidence_notices(results)
        }
        self.assertTrue(notice_ids.issubset({result.record["id"] for result in results}))

    def test_context_overflow_preserves_last_full_coverage_result_set(self) -> None:
        def always_overflow(_messages):
            raise ValueError(
                "Requested tokens (9000) exceed context window of 8192"
            )

        raw_answer, results, messages, context_limited = generate_with_context_retry(
            self.index,
            query="발음과 분위기",
            piece="die-forelle",
            measure_ranges=[],
            measures="",
            topic=None,
            top_k=2,
            generator=always_overflow,
        )

        self.assertEqual(raw_answer, "")
        self.assertTrue(context_limited)
        self.assertEqual(len(results), 2)
        self.assertEqual(messages[1]["content"].count("Evidence "), 2)

    def test_web_chunk_lineage_rejects_unknown_claims_and_wrong_piece_sources(self) -> None:
        settings = load_settings()
        source_index = build_web_source_index(settings["dataset_root"])
        claim_index = build_web_claim_index(settings["dataset_root"], source_index)
        facts_audit_index = build_facts_only_audit_index(
            settings["dataset_root"], claim_index, source_index
        )
        export_path = os.path.join(
            settings["dataset_root"],
            "database",
            "exports",
            "research-open.jsonl",
        )
        chunk = next(iter(iter_jsonl(export_path)))
        validate_web_chunk_lineage(
            chunk,
            claim_index,
            source_index,
            facts_audit_index,
        )

        rights_chunk = next(
            candidate
            for candidate in iter_jsonl(export_path)
            if any(
                claim_index[claim_id]["usage_class"] == "research_open"
                for claim_id in candidate["claim_ids"]
            )
        )
        unsafe_sources = dict(source_index)
        for source_id in rights_chunk["web_source_ids"]:
            unsafe_source = copy.deepcopy(source_index[source_id])
            for asset in unsafe_source["assets"]:
                asset["research_class"] = "catalog_only"
                asset["rights_status"] = "unknown"
                asset["permissions"] = {
                    permission: False
                    for permission in asset.get("permissions", {})
                }
            unsafe_sources[source_id] = unsafe_source
        with self.assertRaisesRegex(ValueError, "has no effective research_open asset"):
            validate_web_chunk_lineage(
                rights_chunk,
                claim_index,
                unsafe_sources,
                facts_audit_index,
            )

        unknown_claim = copy.deepcopy(chunk)
        unknown_claim["claim_ids"] = ["webclaim-does-not-exist"]
        with self.assertRaisesRegex(ValueError, "unknown claims"):
            validate_web_chunk_lineage(
                unknown_claim,
                claim_index,
                source_index,
                facts_audit_index,
            )

        wrong_source_id = next(
            source_id
            for source_id, source in source_index.items()
            if chunk["piece_id"] not in source["piece_ids"]
        )
        forged_claim = copy.deepcopy(claim_index[chunk["claim_ids"][0]])
        forged_claim["web_source_ids"] = [wrong_source_id]
        wrong_piece_chunk = copy.deepcopy(chunk)
        wrong_piece_chunk["claim_ids"] = [forged_claim["claim_id"]]
        wrong_piece_chunk["web_source_ids"] = [wrong_source_id]
        with self.assertRaisesRegex(ValueError, "outside piece"):
            validate_web_chunk_lineage(
                wrong_piece_chunk,
                {forged_claim["claim_id"]: forged_claim},
                source_index,
                facts_audit_index,
            )

        forged_rights = copy.deepcopy(chunk)
        forged_rights["inherited_licenses"].append(
            {
                "web_source_id": chunk["web_source_ids"][0],
                "asset_id": "webasset-forged",
                "asset_type": "metadata",
                "license_id": "CC0-1.0",
                "license_url": None,
                "attribution": None,
            }
        )
        with self.assertRaisesRegex(ValueError, "inherited licenses"):
            validate_web_chunk_lineage(
                forged_rights,
                claim_index,
                source_index,
                facts_audit_index,
            )

    def test_conditional_chunks_cannot_mix_governing_license_families(self) -> None:
        settings = load_settings()
        source_index = build_web_source_index(settings["dataset_root"])
        claim_index = build_web_claim_index(settings["dataset_root"], source_index)
        facts_audit_index = build_facts_only_audit_index(
            settings["dataset_root"], claim_index, source_index
        )
        export_path = os.path.join(
            settings["dataset_root"],
            "database",
            "exports",
            "research-conditional.jsonl",
        )
        chunk = next(
            item
            for item in iter_jsonl(export_path)
            if item["chunk_id"] == "webchunk-ccf3dc098b786ad55207"
        )
        linked_claims = [claim_index[claim_id] for claim_id in chunk["claim_ids"]]
        conditional_assets = [
            (evidence["web_source_id"], asset["asset_id"])
            for claim in linked_claims
            for evidence, asset in resolve_claim_assets(claim, source_index)
            if effective_source_asset_class(asset) == "research_conditional"
        ]
        self.assertGreaterEqual(len(conditional_assets), 2)
        changed_source_id, changed_asset_id = conditional_assets[0]
        mixed_sources = dict(source_index)
        changed_source = copy.deepcopy(source_index[changed_source_id])
        changed_asset = next(
            asset
            for asset in changed_source["assets"]
            if asset["asset_id"] == changed_asset_id
        )
        changed_asset["rights_status"] = "cc_by_nc_sa"
        changed_asset["license_id"] = "CC-BY-NC-SA-4.0"
        changed_asset["license_url"] = "https://creativecommons.org/licenses/by-nc-sa/4.0/"
        changed_asset["terms_url"] = "https://example.invalid/different-terms"
        mixed_sources[changed_source_id] = changed_source
        mixed_chunk = copy.deepcopy(chunk)
        mixed_chunk["inherited_licenses"] = expected_inherited_licenses(
            linked_claims,
            mixed_sources,
        )
        with self.assertRaisesRegex(ValueError, "governing license families"):
            validate_web_chunk_lineage(
                mixed_chunk,
                claim_index,
                mixed_sources,
                facts_audit_index,
            )

    def test_claim_level_expression_uses_every_potential_governing_asset(self) -> None:
        settings = load_settings()
        source_index = build_web_source_index(settings["dataset_root"])
        claim_index = build_web_claim_index(settings["dataset_root"], source_index)
        open_claim = copy.deepcopy(
            next(
                claim
                for claim in claim_index.values()
                if claim["usage_class"] == "research_open"
            )
        )
        conditional_evidence = next(
            {
                "web_source_id": source_id,
                "asset_id": asset["asset_id"],
                "asset_type": asset["asset_type"],
                "locator": "test conditional evidence",
                "original_text": None,
            }
            for source_id, source in source_index.items()
            for asset in source["assets"]
            if effective_source_asset_class(asset) == "research_conditional"
        )
        open_claim["evidence"].append(conditional_evidence)
        open_claim["contains_source_expression"] = True
        open_claim["claim_original"] = "retained expression without asset attribution"
        with self.assertRaisesRegex(ValueError, "claim-level expression"):
            validate_claim_asset_rights(open_claim, source_index)

    def test_facts_only_audit_decisions_are_recomputed(self) -> None:
        settings = load_settings()
        source_index = build_web_source_index(settings["dataset_root"])
        claim_index = build_web_claim_index(settings["dataset_root"], source_index)
        audit_path = os.path.join(
            settings["dataset_root"],
            "database",
            "reports",
            "facts-only-release-audit.json",
        )
        with open(audit_path, encoding="utf-8") as f:
            forged_audit = json.load(f)
        record = next(
            item
            for item in forged_audit["records"]
            if item["release_decision"]["export_class"] == "catalog_only"
        )
        record["release_decision"] = {
            "eligible": True,
            "export_class": "research_open",
            "reasons": [],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            report_dir = os.path.join(temp_dir, "database", "reports")
            os.makedirs(report_dir)
            with open(
                os.path.join(report_dir, "facts-only-release-audit.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(forged_audit, f, ensure_ascii=False)
            with self.assertRaisesRegex(ValueError, "decision mismatch"):
                build_facts_only_audit_index(temp_dir, claim_index, source_index)

    def test_facts_only_audit_review_payload_is_recomputed(self) -> None:
        settings = load_settings()
        source_index = build_web_source_index(settings["dataset_root"])
        claim_index = build_web_claim_index(settings["dataset_root"], source_index)
        audit_path = os.path.join(
            settings["dataset_root"],
            "database",
            "reports",
            "facts-only-release-audit.json",
        )
        with open(audit_path, encoding="utf-8") as f:
            forged_audit = json.load(f)
        forged_audit["records"][0]["review"]["notes"] = "forged review metadata"
        with tempfile.TemporaryDirectory() as temp_dir:
            report_dir = os.path.join(temp_dir, "database", "reports")
            os.makedirs(report_dir)
            with open(
                os.path.join(report_dir, "facts-only-release-audit.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(forged_audit, f, ensure_ascii=False)
            with self.assertRaisesRegex(ValueError, "review mismatch"):
                build_facts_only_audit_index(temp_dir, claim_index, source_index)

    def test_approved_facts_review_cannot_report_protected_expression(self) -> None:
        settings = load_settings()
        source_index = build_web_source_index(settings["dataset_root"])
        claim_index = build_web_claim_index(settings["dataset_root"], source_index)
        audit_path = os.path.join(
            settings["dataset_root"],
            "database",
            "reports",
            "facts-only-release-audit.json",
        )
        with open(audit_path, encoding="utf-8") as f:
            audit = json.load(f)
        target = next(
            record
            for record in audit["records"]
            if record["claim_id"] == "webclaim-317d500322e31a7ca429"
        )
        invalid_review = {
            "claim_id": target["claim_id"],
            "claim_record_sha256": target["claim_record_sha256"],
            "evidence_lineage_sha256": target["evidence_lineage_sha256"],
            "status": "approved",
            "overlap_review_status": "protected_expression_detected",
            "reviewer": "test reviewer",
            "reviewed_at": "2026-07-21T00:00:00Z",
            "method": "manual source comparison",
            "notes": "Protected overlap was found.",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            report_dir = os.path.join(temp_dir, "database", "reports")
            os.makedirs(report_dir)
            with open(
                os.path.join(report_dir, "facts-only-release-audit.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(audit, f, ensure_ascii=False)
            with open(
                os.path.join(temp_dir, "database", "facts-only-expression-reviews.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump([invalid_review], f, ensure_ascii=False)
            with self.assertRaisesRegex(ValueError, "must find no protected expression"):
                build_facts_only_audit_index(temp_dir, claim_index, source_index)

    def test_expert_and_web_lineage_stay_separate(self) -> None:
        for record in self.records:
            if record["evidence_type"] == "expert_annotation":
                self.assertIn("source_ids", record)
                self.assertNotIn("web_source_ids", record)
                self.assertNotIn("claim_ids", record)
            else:
                self.assertNotIn("source_ids", record)
                self.assertTrue(record["web_source_ids"])
                self.assertTrue(record["claim_ids"])
                self.assertTrue(all(source["url"] for source in record["sources"]))

    def test_placeholder_is_preserved_but_not_retrievable(self) -> None:
        placeholder = next(record for record in self.records if record["id"] == "sqa-0072")
        self.assertEqual(placeholder["answer"], "기존 웹 자료를 부탁한다.")
        self.assertFalse(placeholder["retrieval_eligible"])
        results = self.index.search(
            "기존 웹 자료를 부탁한다",
            piece="nella-fantasia",
            top_k=20,
        )
        self.assertNotIn("sqa-0072", [result.record["id"] for result in results])

    def test_measure_query_uses_overlap_and_general_context_only(self) -> None:
        results = self.index.search(
            "28마디부터 분위기 변화를 어떻게 표현해야 하나요?",
            piece="die-forelle",
            measure_ranges=[[28, 30]],
            top_k=10,
        )
        self.assertEqual(results[0].record["id"], "sqa-0058")
        for result in results:
            if result.record["measure_range"]:
                self.assertEqual(result.scope_match, "overlaps_query_range")

    def test_multi_range_search_checks_every_pair(self) -> None:
        results = self.index.search(
            "이중모음과 꾸밈음은 어떻게 부르나요?",
            piece="una-voce-poco-fa",
            measure_ranges=[[6, 6]],
            top_k=10,
        )
        matched = [result.record for result in results if result.scope_match == "overlaps_query_range"]
        self.assertIn([[56, 56], [6, 6]], [record["measure_range"] for record in matched])

    def test_whole_song_web_identity_and_safe_abstention(self) -> None:
        identity = self.index.search(
            "Die Forelle의 정식 표제와 작품 번호는 무엇인가요?",
            piece="die-forelle",
            top_k=3,
        )
        self.assertEqual(identity[0].record["id"], "webchunk-cecff2bace03ab67e32d")
        unsupported = self.index.search(
            "Nella fantasia와 Gabriel's Oboe의 관계는 무엇인가요?",
            piece="nella-fantasia",
            top_k=6,
        )
        self.assertEqual(unsupported, [])

    def test_title_and_creator_identity_survive_alias_normalization(self) -> None:
        cases = [
            ("die-forelle", "슈베르트는 누구인가요?", "creator_profiles"),
            ("in-flowery-clouds", "이흥렬은 누구인가요?", "creator_profiles"),
            ("nella-fantasia", "모리코네는 누구인가요?", "creator_profiles"),
            ("die-forelle", "슈베르트의 생애는 어떤가요?", "creator_profiles"),
            (
                "die-forelle",
                "Die Forelle는 무엇인가요?",
                "identity_aliases_and_catalog_ids",
            ),
            ("die-forelle", "송어는 어떤 곡인가요?", "identity_aliases_and_catalog_ids"),
            (
                "nella-fantasia",
                "넬라 판타지아는 어떤 곡인가요?",
                "identity_aliases_and_catalog_ids",
            ),
            (
                "la-capinera",
                "라 카피네라는 무슨 곡인가요?",
                "identity_aliases_and_catalog_ids",
            ),
        ]
        for piece, question, expected_topic in cases:
            with self.subTest(question=question):
                results = self.index.search(question, piece=piece, top_k=3)
                self.assertTrue(results)
                self.assertEqual(results[0].record["topic"], expected_topic)

    def test_identity_words_cannot_mask_an_unrelated_subject(self) -> None:
        cases = [
            ("nella-fantasia", "넬라 판타지아 FIFA 월드컵 우승팀은 무엇인가요?"),
            ("die-forelle", "슈베르트와 FIFA 월드컵 우승팀은 누구인가요?"),
            ("die-forelle", "송어와 프랑스 수도는 무엇인가요?"),
        ]
        for piece, question in cases:
            with self.subTest(question=question):
                self.assertEqual(self.index.search(question, piece=piece, top_k=6), [])

    def test_out_of_domain_question_abstains_with_and_without_measures(self) -> None:
        question = "2022년 FIFA 월드컵 우승팀은 어디인가요?"
        ranged = self.index.search(
            question,
            piece="die-forelle",
            measure_ranges=[[28, 30]],
            top_k=6,
        )
        whole_song = self.index.search(
            question,
            piece="die-forelle",
            top_k=6,
        )
        self.assertEqual(ranged, [])
        self.assertEqual(whole_song, [])

    def test_provenance_names_and_english_ood_queries_abstain(self) -> None:
        for question in (
            "What is the capital of France?",
            "What is France?",
            "Bibliothèque?",
            "Wikimedia?",
        ):
            with self.subTest(question=question):
                self.assertEqual(
                    self.index.search(question, piece="die-forelle", top_k=6),
                    [],
                )
        supported = self.index.search(
            "What is the title and catalog number?",
            piece="die-forelle",
            top_k=3,
        )
        self.assertEqual(supported[0].record["id"], "webchunk-cecff2bace03ab67e32d")

    def test_compound_questions_cover_each_real_concept(self) -> None:
        compound = self.index.search(
            "송어의 발음과 형식을 알려 주세요",
            piece="die-forelle",
            top_k=6,
        )
        topics = {result.record["topic"] for result in compound}
        self.assertIn("발음과 딕션", topics)
        self.assertIn("시와 유절가곡 형식", topics)
        for phrasing in (
            "발음하고 형식도 말해 주세요",
            "발음 및 형식은요?",
            "발음과 형식에 관해 알려 주세요",
            "발음과 형식을 같이 알려 주세요",
        ):
            with self.subTest(phrasing=phrasing):
                phrased_results = self.index.search(
                    phrasing,
                    piece="die-forelle",
                    top_k=6,
                )
                phrased_topics = {result.record["topic"] for result in phrased_results}
                self.assertIn("발음과 딕션", phrased_topics)
                self.assertIn("시와 유절가곡 형식", phrased_topics)
        one_chunk = self.index.search(
            "발음과 리듬을 어떻게 맞추나요?",
            piece="die-forelle",
            top_k=1,
        )
        self.assertEqual(one_chunk[0].record["id"], "sqa-0071")
        for unsupported in (
            "가사 의미없는질문",
            "반주 국제관계는 무엇인가요?",
            "형식 형식주의철학은 무엇인가요?",
        ):
            with self.subTest(unsupported=unsupported):
                self.assertEqual(
                    self.index.search(unsupported, piece="die-forelle", top_k=6),
                    [],
                )

    def test_natural_korean_predicates_do_not_hide_supported_topics(self) -> None:
        mood_questions = (
            "분위기가 바뀌는 부분을 어떻게 표현할지 알려주세요",
            "28마디부터 분위기가 어떻게 달라지는지 설명해 주세요",
        )
        for question in mood_questions:
            with self.subTest(question=question):
                results = self.index.search(
                    question,
                    piece="die-forelle",
                    measure_ranges=[[28, 30]],
                    top_k=6,
                )
                self.assertEqual(results[0].record["id"], "sqa-0058")
        for question in (
            "6마디를 어떻게 발음해야 할까요?",
            "6마디에서 발음하는 방법을 알려 주세요.",
            "6마디의 발음과 딕션은 어떻게 하나요?",
        ):
            with self.subTest(question=question):
                results = self.index.search(
                    question,
                    piece="una-voce-poco-fa",
                    measure_ranges=[[6, 6]],
                    top_k=1,
                )
                self.assertEqual(results[0].record["id"], "sqa-0032")
        form = self.index.search(
            "이 곡의 형식은 어떻게 구성되어 있나요?",
            piece="die-forelle",
            top_k=3,
        )
        self.assertEqual(form[0].record["id"], "sqa-0065")

    def test_generic_identity_and_creator_paraphrases_route_correctly(self) -> None:
        identity_questions = (
            "이 곡은 무엇인가요?",
            "송어는 어떤 노래인가요?",
        )
        for question in identity_questions:
            with self.subTest(question=question):
                results = self.index.search(question, piece="die-forelle", top_k=2)
                self.assertEqual(
                    results[0].record["topic"],
                    "identity_aliases_and_catalog_ids",
                )
        biography = self.index.search(
            "슈베르트는 어떤 사람이었나요?",
            piece="die-forelle",
            top_k=2,
        )
        self.assertTrue(biography)
        self.assertTrue(all(result.record["topic"] == "creator_profiles" for result in biography))
        for piece in (
            "die-forelle",
            "in-flowery-clouds",
            "la-capinera",
            "nella-fantasia",
            "una-voce-poco-fa",
        ):
            with self.subTest(piece=piece):
                creator = self.index.search(
                    "이 노래를 누가 작곡했나요?",
                    piece=piece,
                    top_k=2,
                )
                self.assertTrue(creator)
                self.assertEqual(creator[0].record["topic"], "creator_profiles")
                for implicit_question in (
                    "누가 작곡했나요?",
                    "작곡자는 누구예요?",
                    "이 곡을 작곡한 사람은 누구예요?",
                    "이 곡의 작곡가가 누구죠?",
                ):
                    implicit = self.index.search(
                        implicit_question,
                        piece=piece,
                        top_k=2,
                    )
                    self.assertTrue(implicit)
                    self.assertEqual(implicit[0].record["topic"], "creator_profiles")

    def test_measure_scope_priority_survives_compound_coverage(self) -> None:
        ranged = self.index.search(
            "6마디의 발음과 딕션은 어떻게 하나요?",
            piece="una-voce-poco-fa",
            measure_ranges=[[6, 6]],
            top_k=2,
        )
        self.assertEqual(ranged[0].record["id"], "sqa-0032")
        self.assertEqual(ranged[0].scope_match, "overlaps_query_range")
        whole_song = self.index.search(
            "발음과 분위기",
            piece="die-forelle",
            top_k=6,
        )
        self.assertEqual(whole_song[0].scope_match, "general_evidence")
        self.assertEqual(
            self.index.search(
                "발음과 분위기",
                piece="die-forelle",
                top_k=1,
            ),
            [],
        )

    def test_short_inflected_korean_topics_still_retrieve(self) -> None:
        form = self.index.search("형식은 무엇인가요?", piece="die-forelle", top_k=3)
        pronunciation = self.index.search("발음은 어떻게 하나요?", piece="die-forelle", top_k=3)
        natural_form = self.index.search("형식이 어떻게 되나요?", piece="die-forelle", top_k=3)
        natural_pronunciation = self.index.search("발음이 궁금해요", piece="die-forelle", top_k=3)
        genre = self.index.search("이 작품의 장르는 뭐예요?", piece="die-forelle", top_k=3)
        self.assertEqual(form[0].record["id"], "sqa-0065")
        self.assertIn(pronunciation[0].record["id"], {"sqa-0060", "sqa-0061"})
        self.assertEqual(pronunciation[0].record["topic"], "발음과 딕션")
        self.assertEqual(natural_form[0].record["id"], "sqa-0065")
        self.assertEqual(natural_pronunciation[0].record["topic"], "발음과 딕션")
        self.assertEqual(genre[0].record["topic"], "genre_and_form")
        for polite_form in (
            "어떤 형식이에요?",
            "무슨 형식이에요?",
            "이 곡은 유절가곡이에요?",
            "형식일까요?",
        ):
            with self.subTest(polite_form=polite_form):
                self.assertTrue(
                    self.index.search(polite_form, piece="die-forelle", top_k=3)
                )
        connective = self.index.search(
            "발음이랑 형식을 알려 주세요",
            piece="die-forelle",
            top_k=6,
        )
        connective_topics = {result.record["topic"] for result in connective}
        self.assertIn("발음과 딕션", connective_topics)
        self.assertIn("시와 유절가곡 형식", connective_topics)

    def test_title_only_boilerplate_abstains(self) -> None:
        results = self.index.search(
            "Die Forelle에 대해 알려 주세요",
            piece="die-forelle",
            top_k=6,
        )
        self.assertEqual(results, [])

        unsupported_cardinal_form = self.index.search(
            "이 곡은 4마디 프레이즈로 구성되나요?",
            piece="die-forelle",
            top_k=6,
        )
        self.assertEqual(unsupported_cardinal_form, [])

    def test_one_matched_word_cannot_answer_a_two_concept_question(self) -> None:
        results = self.index.search(
            "프랑스 수도는 어디인가요?",
            piece="die-forelle",
            top_k=6,
        )
        self.assertEqual(results, [])

    def test_transliterated_title_and_composer_do_not_count_as_relevance(self) -> None:
        nella = self.index.search(
            "넬라 판타지아의 프랑스 수도는 어디인가요?",
            piece="nella-fantasia",
            top_k=6,
        )
        forelle = self.index.search(
            "슈베르트 송어의 프랑스 수도는 어디인가요?",
            piece="die-forelle",
            top_k=6,
        )
        self.assertEqual(nella, [])
        self.assertEqual(forelle, [])

        imagery = self.index.search(
            "송어가 뛰어노는 모습을 어떻게 표현하나요?",
            piece="die-forelle",
            top_k=3,
        )
        self.assertEqual(imagery[0].record["id"], "sqa-0057")

    def test_context_formats_both_provenance_families(self) -> None:
        expert = self.index.search(
            "유절가곡인가요",
            piece="die-forelle",
            top_k=1,
        )
        web = self.index.search(
            "정식 표제와 작품 번호",
            piece="die-forelle",
            top_k=1,
        )
        context = build_context(expert + web)
        self.assertIn("citation_label: E1", context)
        self.assertIn("evidence_type: expert_annotation", context)
        self.assertIn("expert_source_ids:", context)
        self.assertIn("evidence_type: web_database", context)
        self.assertIn("web_source_ids:", context)
        self.assertIn("web_source:", context)

        notices = build_evidence_notices(web)
        self.assertEqual(notices[0]["record_id"], "webchunk-cecff2bace03ab67e32d")
        self.assertEqual(notices[0]["usage_class"], "research_open")
        formatted_notice = format_evidence_notice(notices[0])
        self.assertIn("generated_text_license=CC-BY-4.0", formatted_notice)
        self.assertEqual(
            notices[0]["generated_text_attribution"],
            "Soprano QA database project",
        )
        self.assertEqual(
            notices[0]["generated_text_license_url"],
            "https://creativecommons.org/licenses/by/4.0/",
        )
        self.assertIn("generated_text_attribution=Soprano QA database project", formatted_notice)
        self.assertIn("creativecommons.org/licenses/by/4.0/", formatted_notice)
        self.assertIn("source: websrc-", formatted_notice)

        labeled = finalize_answer_citations("근거입니다. [E1]", web)
        self.assertEqual(labeled, "근거입니다. [webchunk-cecff2bace03ab67e32d]")
        typo = finalize_answer_citations(
            "근거입니다. [webchunk-cecff2bace003ab67e32d]",
            web,
        )
        self.assertEqual(typo, "근거입니다. [webchunk-cecff2bace03ab67e32d]")
        grouped = finalize_answer_citations("근거입니다. [E1, E2]", expert + web)
        self.assertEqual(
            grouped,
            "근거입니다. [sqa-0065] [webchunk-cecff2bace03ab67e32d]",
        )
        spaced = finalize_answer_citations("근거입니다. [ E1 ]", expert + web)
        self.assertEqual(spaced, "근거입니다. [sqa-0065]")
        trailing = finalize_answer_citations("근거입니다. [E1;]", expert + web)
        self.assertEqual(trailing, "근거입니다. [sqa-0065]")
        uncited = finalize_answer_citations("근거입니다.", web)
        self.assertEqual(
            uncited,
            "근거입니다.\n\n제공된 검색 근거: [webchunk-cecff2bace03ab67e32d]",
        )
        unknown = finalize_answer_citations("근거입니다. [E99]", web)
        self.assertEqual(
            unknown,
            "근거입니다.\n\n제공된 검색 근거: [webchunk-cecff2bace03ab67e32d]",
        )
        malformed = finalize_answer_citations(
            "근거입니다. [websrc-fake] [sqa-ABC] [A]",
            web,
        )
        self.assertEqual(
            malformed,
            "근거입니다.   [A]\n\n제공된 검색 근거: "
            "[webchunk-cecff2bace03ab67e32d]",
        )
        prefixed = finalize_answer_citations(
            "근거입니다. [1] [출처: E99] [Source: webchunk-not-retrieved]",
            web,
        )
        self.assertNotIn("[1]", prefixed)
        self.assertNotIn("E99", prefixed)
        self.assertNotIn("not-retrieved", prefixed)
        self.assertIn("제공된 검색 근거: [webchunk-cecff2bace03ab67e32d]", prefixed)
        expert_source_id = expert[0].record["source_ids"][0]
        provenance_citation = finalize_answer_citations(
            "근거입니다. [%s]" % expert_source_id,
            expert,
        )
        self.assertNotIn(expert_source_id, provenance_citation)
        self.assertIn("제공된 검색 근거: [sqa-0065]", provenance_citation)
        grouped_provenance = finalize_answer_citations(
            "근거입니다. [%s]" % ", ".join(expert[0].record["source_ids"]),
            expert,
        )
        for source_id in expert[0].record["source_ids"]:
            self.assertNotIn(source_id, grouped_provenance)

    def test_conditional_web_evidence_has_deterministic_rights_notice(self) -> None:
        for rights_question in (
            "관할권별 권리 조건",
            "territorial rights and access conditions",
        ):
            with self.subTest(rights_question=rights_question):
                rights_results = self.index.search(
                    rights_question,
                    piece="la-capinera",
                    top_k=3,
                )
                self.assertTrue(rights_results)
                self.assertEqual(rights_results[0].record["topic"], "rights_history")
        results = self.index.search(
            "언어와 텍스트 계보",
            piece="la-capinera",
            top_k=1,
        )
        self.assertEqual(results[0].record["usage_class"], "research_conditional")
        notice = build_evidence_notices(results)[0]
        self.assertTrue(notice["inherited_licenses"])
        formatted = format_evidence_notice(notice)
        self.assertIn("usage_class=research_conditional", formatted)
        self.assertIn("inherited:", formatted)
        self.assertIn("attribution:", formatted)

        territorial_record = next(
            record
            for record in self.records
            if any(
                inherited["asset_id"] == "webasset-ae9215f1dcb72c8355b1"
                for inherited in record.get("inherited_licenses", [])
            )
        )
        territorial_notice = build_evidence_notices(
            [SimpleNamespace(record=territorial_record)]
        )[0]
        territorial_license = next(
            inherited
            for inherited in territorial_notice["inherited_licenses"]
            if inherited["asset_id"] == "webasset-ae9215f1dcb72c8355b1"
        )
        self.assertEqual(territorial_license["jurisdictions"], ["US"])
        self.assertIn("loc.gov", territorial_license["terms_url"])
        territorial_text = format_evidence_notice(territorial_notice)
        self.assertIn("asset=webasset-ae9215f1dcb72c8355b1 (metadata)", territorial_text)
        self.assertIn("jurisdiction: US", territorial_text)
        self.assertIn("rights-and-access", territorial_text)


if __name__ == "__main__":
    unittest.main()
