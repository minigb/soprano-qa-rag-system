import copy
from pathlib import Path
import shutil
import tempfile
import unittest

from soprano_qa.answer import retrieve_generation_evidence
from soprano_qa.corpus import corpus_input_paths, validate_release_corpus
from soprano_qa.dense import build_retrieval_index
from soprano_qa.retrieval import BM25Index, SearchResult
from soprano_qa.settings import load_settings
from soprano_qa.source_priority import SourcePriorityIndex, SumijoIndex, merge_evidence, prioritize
from soprano_qa.sumijo import build_records


def hit(kind, uid, score=1, scope="general_evidence"):
    return SearchResult({"id": uid, "evidence_type": kind}, score, score, 0, 0, scope)


class PriorityTests(unittest.TestCase):
    def test_authority_precedes_cross_source_score(self):
        results = [hit("web_database", "w", 100), hit("expert_annotation", "e", 50),
                   hit("sumijo_feedback", "s", 1)]
        self.assertEqual([r.record["id"] for r in prioritize(results)], ["s", "e", "w"])

    def test_context_does_not_displace_applicable_evidence(self):
        results = [hit("sumijo_feedback", "s", scope="unspecified_context"),
                   hit("expert_annotation", "e", scope="overlaps_query_range")]
        self.assertEqual(prioritize(results)[0].record["id"], "e")

    def test_stability_deduplication_and_budget(self):
        a, b = hit("expert_annotation", "a"), hit("expert_annotation", "b")
        self.assertEqual(merge_evidence([a], [b, a], 2), [a, b])
        self.assertEqual(merge_evidence([a], [b], 0), [])


class SumijoIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = load_settings()
        cls.root = Path(cls.settings["dataset_root"])
        cls.records = build_records(str(cls.root))

    def index(self):
        return build_retrieval_index(self.records, {"retrieval": {
            "mode": "lexical", "source_priority": True}})

    def test_separate_schema_and_provenance(self):
        self.assertEqual(len(self.records), 7)
        validate_release_corpus(self.records)
        self.assertTrue(all("source_ids" not in r and "web_source_ids" not in r for r in self.records))
        self.assertTrue(all(r["sumijo_source"]["speaker"] == "Sumi Jo" for r in self.records))

    def test_all_original_questions_find_their_source(self):
        index = self.index()
        for record in self.records:
            if not record["retrieval_aliases"]:
                continue
            with self.subTest(id=record["id"]):
                results = index.search(record["retrieval_aliases"][0], piece="nella-fantasia")
                self.assertEqual(results[0].record["id"], record["id"])

    def test_piece_and_range_gates(self):
        index = self.index()
        query = "마이크를 사용할 때 어떤 음색으로 노래해야 하나요?"
        self.assertTrue(index.search(query, piece="nella-fantasia"))
        self.assertEqual(index.search(query, piece="die-forelle"), [])
        self.assertEqual(index.search(query, piece="nella-fantasia", measure_ranges=[[1, 4]]), [])

    def test_unrelated_question_abstains(self):
        self.assertEqual(self.index().search("이 곡의 작곡가는 누구인가요?", piece="nella-fantasia"), [])

    def test_expert_fallback_and_no_main_index_contamination(self):
        record = copy.deepcopy(self.records[1])
        record["id"] = "expert-test"
        record["evidence_type"] = "expert_annotation"
        record["source_ids"] = ["kim-test-01"]
        del record["sumijo_source"]
        index = SourcePriorityIndex(BM25Index([record]), self.records)
        query = "마이크를 사용할 때 어떤 음색으로 노래해야 하나요?"
        hits = index.search(query, piece="nella-fantasia")
        self.assertEqual(hits[0].record["evidence_type"], "sumijo_feedback")
        self.assertIn("expert-test", [r.record["id"] for r in hits])
        self.assertEqual(len(index.base_index.records), 1)

    def test_generation_uses_priority_and_preserves_attribution(self):
        results, messages = retrieve_generation_evidence(
            self.index(), query="마이크를 사용할 때 어떤 음색으로 노래해야 하나요?",
            piece="nella-fantasia", measure_ranges=[], measures="", topic=None, top_k=2)
        self.assertEqual(results[0].record["evidence_type"], "sumijo_feedback")
        self.assertIn("Sumi Jo", messages[1]["content"])
        self.assertIn("sumijo_feedback, then expert_annotation", messages[0]["content"])

    def test_api_does_not_label_sumijo_as_web(self):
        from soprano_qa.service import _result_to_evidence
        result = self.index().search("마이크를 사용할 때 어떤 음색으로 노래해야 하나요?", piece="nella-fantasia")[0]
        evidence = _result_to_evidence(result, measure_ranges=[])
        self.assertEqual(evidence["kind"], "sumijo")
        self.assertEqual(evidence["sumijo_source"]["speaker"], "Sumi Jo")

    def test_hashes_and_validator_are_corpus_inputs(self):
        labels = {label for label, _ in corpus_input_paths(self.settings)}
        self.assertIn("sumijo-feedback/knowledge_units.json", labels)
        self.assertIn("sumijo-feedback/tools/validate.py", labels)
        self.assertIn("sumijo-feedback/audio/nella-fantasia/Memo 036.m4a", labels)

    def test_source_drift_and_broken_provenance_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shutil.copytree(self.root / "sumijo-feedback", root / "sumijo-feedback")
            path = root / "sumijo-feedback/text/nella-fantasia/Memo 036.reviewed.txt"
            path.write_text(path.read_text() + "changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "drifted"):
                build_records(str(root))
        record = copy.deepcopy(self.records[0])
        record["piece"] = "die-forelle"
        with self.assertRaises(ValueError):
            validate_release_corpus([record])

    def test_semantic_fallback_requires_confidence_and_margin(self):
        class Embedder:
            query = [1, 0]
            def embed_documents(self, texts):
                return [[1, 0], *([[0, 1]] * (len(texts) - 1))]
            def embed_query(self, text):
                return self.query
        embedder = Embedder()
        index = SumijoIndex(self.records, embedder=embedder)
        query = "첫소리를 부드럽게 시작하려면 어떻게 해야 하나요?"
        self.assertEqual(index.lexical.search(query, piece="nella-fantasia"), [])
        self.assertEqual(index.search(query, piece="nella-fantasia")[0].record["id"], self.records[0]["id"])
        embedder.query = [-1, 0]
        self.assertEqual(index.search(query, piece="nella-fantasia"), [])
        embedder.query = [1, 1]
        self.assertEqual(index.search(query, piece="nella-fantasia"), [])
        embedder.query = [1, 0]
        self.assertEqual(index.search(query, piece="die-forelle"), [])
        self.assertEqual(index.search(query, piece="nella-fantasia", measure_ranges=[[1, 4]]), [])

    def test_invalid_semantic_threshold_is_rejected(self):
        with self.assertRaises(ValueError):
            SumijoIndex(self.records, min_score=-1)


if __name__ == "__main__":
    unittest.main()
