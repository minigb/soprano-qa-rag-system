"""Authority ordering after relevance and measure eligibility, never instead of them."""
from __future__ import annotations

from soprano_qa.retrieval import BM25Index, SearchResult, measure_scope_match


SOURCE_PRIORITY = {"sumijo_feedback": 0, "expert_annotation": 1,
                   "web_coaching": 2, "web_database": 2}
CONTEXT_ONLY = {"other_range_context", "unspecified_context", "global_context"}


def prioritize(results):
    """Stable ordering preserves relevance ranking within each authority tier."""
    return sorted(results, key=lambda r: (
        r.scope_match in CONTEXT_ONLY,
        SOURCE_PRIORITY.get(r.record.get("evidence_type"), 3),
    ))


def merge_evidence(primary, fallback, top_k):
    seen = set()
    combined = []
    for result in prioritize([*primary, *fallback]):
        if result.record["id"] not in seen:
            combined.append(result)
            seen.add(result.record["id"])
    return combined[:max(0, top_k)]


class SumijoIndex:
    """Lexical routing plus a high-confidence, answer-only semantic fallback."""
    def __init__(self, records, *, embedder=None, min_score=0.60, margin=0.04):
        if not 0 <= min_score <= 1 or not 0 <= margin <= 1:
            raise ValueError("Sumi Jo semantic thresholds must lie in [0, 1]")
        self.records = list(records)
        self.lexical = BM25Index(self.records)
        self.embedder = embedder
        self.min_score, self.margin = min_score, margin
        self.vectors = []
        if embedder is not None and self.records:
            from soprano_qa.dense import _normalized_vector
            self.vectors = [_normalized_vector(v) for v in embedder.embed_documents(
                [r["answer"] for r in self.records])]
            if len(self.vectors) != len(self.records) or len({len(v) for v in self.vectors}) != 1:
                raise ValueError("Invalid Sumi Jo answer embeddings")

    def search(self, query, piece=None, measure_ranges=None, topic=None, top_k=8):
        if top_k < 1:
            return []
        lexical = self.lexical.search(query, piece, measure_ranges, topic, top_k)
        if not self.vectors or measure_ranges:
            return lexical
        eligible = [i for i, r in enumerate(self.records)
                    if (not piece or r["piece"] == piece)
                    and (not topic or topic in r.get("topic", ""))]
        if not eligible:
            return lexical
        from soprano_qa.dense import _normalized_vector, dense_query_text, missing_dense_answer_constraints
        vector = _normalized_vector(self.embedder.embed_query(dense_query_text(query, piece)))
        if len(vector) != len(self.vectors[0]):
            raise ValueError("Sumi Jo query/document embedding dimensions differ")
        scores = {i: sum(a * b for a, b in zip(vector, self.vectors[i])) for i in eligible}
        ranked = sorted(eligible, key=lambda i: (-scores[i], self.records[i]["id"]))
        best = ranked[0]
        # A tie between independent claims is insufficient grounds to attribute
        # one to Sumi Jo. Keep existing lexical matches instead of guessing.
        if scores[best] < self.min_score or (len(ranked) > 1 and scores[best] - scores[ranked[1]] < self.margin):
            return lexical
        record = self.records[best]
        if missing_dense_answer_constraints(query, record["answer"]):
            return lexical
        result = SearchResult(record, scores[best], 0, 0, 1,
                              measure_scope_match(record, measure_ranges or []),
                              dense_score=scores[best], dense_content_score=scores[best],
                              semantic_match_type="sumijo_answer_semantic", retrieval_mode="hybrid")
        # Strong semantic evidence leads the same-lineage lexical neighbors.
        return merge_evidence([result], lexical, top_k)


class SourcePriorityIndex:
    """Keep the calibrated expert/web retriever intact and isolate direct feedback.

    Sumi Jo retrieval uses lexical/concept gates and conservative answer-only
    semantic matching, never provenance names or evaluation question text.
    The generation path uses the same feedback index after expert selection.
    """
    def __init__(self, base_index, sumijo_records, *, min_score=0.60, margin=0.04):
        self.base_index = base_index
        self.sumijo_index = SumijoIndex(sumijo_records,
            embedder=getattr(base_index, "embedder", None), min_score=min_score, margin=margin)
        self.records = [*sumijo_records, *base_index.records]

    def __getattr__(self, name):
        return getattr(self.base_index, name)

    def search(self, query, piece=None, measure_ranges=None, topic=None, top_k=8):
        if top_k < 1:
            return []
        kwargs = dict(query=query, piece=piece, measure_ranges=measure_ranges,
                      topic=topic, top_k=max(top_k, 24))
        fallback = self.base_index.search(**kwargs)
        from soprano_qa.dense import retrieval_abstained_for_missing_causal_authority
        if retrieval_abstained_for_missing_causal_authority(self.base_index):
            return []
        from soprano_qa.answer import select_generation_evidence
        fallback = select_generation_evidence(fallback, query=query, piece=piece,
                                              measure_ranges=measure_ranges or [])
        primary = self.sumijo_index.search(**kwargs)
        return merge_evidence(primary, fallback, top_k)


def wrap_index(index, sumijo_records, settings):
    if settings.get("retrieval", {}).get("source_priority", False):
        options = settings.get("sumijo") or {}
        return SourcePriorityIndex(index, sumijo_records,
            min_score=float(options.get("dense_min_score", 0.60)),
            margin=float(options.get("dense_margin", 0.04)))
    return index
