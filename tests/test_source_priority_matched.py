import unittest
from evaluation.source_priority_matched.run import rank, digest, reset_observed
from evaluation.source_priority_matched.analyze import bootstrap, status, packet


class MatchedEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.pool = [
            {"record":{"id":"s", "evidence_type":"sumijo_feedback"}, "relevance":0.6},
            {"record":{"id":"w", "evidence_type":"web_coaching"}, "relevance":0.9},
            {"record":{"id":"e", "evidence_type":"expert_annotation"}, "relevance":0.8},
        ]

    def test_same_pool_different_order_without_mutation(self):
        before = digest(self.pool)
        self.assertEqual([p["record"]["id"] for p in rank(self.pool,"relevance",3)], ["w","e","s"])
        self.assertEqual([p["record"]["id"] for p in rank(self.pool,"priority",3)], ["s","e","w"])
        self.assertEqual(before,digest(self.pool))
        self.assertEqual({p["record"]["id"] for p in rank(self.pool,"priority",3)},
                         {p["record"]["id"] for p in rank(self.pool,"relevance",3)})

    def test_budget_changes_selection_from_same_candidates(self):
        self.assertEqual(len(rank(self.pool,"priority",2)),2)
        self.assertNotEqual(rank(self.pool,"priority",2),rank(self.pool,"relevance",2))

    def test_same_tier_preserves_relevance(self):
        for p in self.pool:
            p["record"]["evidence_type"]="expert_annotation"
        self.assertEqual(rank(self.pool,"priority",3),rank(self.pool,"relevance",3))

    def test_usage_reset_detected_without_confusing_increased_usage(self):
        initial={"primary":{"resetsAt":1000,"usedPercent":81}}
        self.assertFalse(reset_observed(initial,{"primary":{"resetsAt":1000,"usedPercent":82}}))
        self.assertTrue(reset_observed(initial,{"primary":{"resetsAt":2000,"usedPercent":1}}))

    def test_quality_interval_clusters_budget_variants(self):
        pairs=[{"cluster":cluster,"priority":{"quality_sum":4},"relevance":{"quality_sum":3}}
               for cluster in ("same-source", "same-source", "other-source")]
        result=bootstrap(pairs,"quality_sum",draws=100)
        self.assertEqual(result["clusters"],2)
        self.assertEqual(result["pairs"],3)
        self.assertEqual(result["cluster_bootstrap_95_percentile_interval"],[1,1])

    def test_validation_rejection_is_not_backend_error(self):
        row={"status":"execution_error","error":{"type":"GeneratedAnswerRejected"}}
        self.assertEqual(status(row),"rejected")
        row["error"]["type"]="MemoryError"
        self.assertEqual(status(row),"execution_error")

    def test_single_source_cluster_has_no_inferential_interval(self):
        result=bootstrap([{"cluster":"one","priority":{"quality_sum":4},
                           "relevance":{"quality_sum":3}}],"quality_sum")
        self.assertIsNone(result["cluster_bootstrap_95_percentile_interval"])

    def test_masked_packet_remaps_citations_and_hides_ranking(self):
        case={"id":"case","question":"question","piece":"piece","kind":"natural",
              "reference_requirements":"requirements","reference_answer":"reference",
              "pool":[{"record":{"id":i,"evidence_type":"expert_annotation","answer":i},
                       "scope":"general_evidence"} for i in ("a","b")]}
        row={"selected_ids":["b","a"],"answer":"claim","raw_answer":"claim [E1]",
             "status":"answer","condition":"priority"}
        p=packet(row,case)
        self.assertEqual(p["cited_answer"],"claim [b]")
        self.assertEqual([e["id"] for e in p["evidence"]],["a","b"])
        self.assertNotIn("condition",p)


if __name__ == "__main__":
    unittest.main()
