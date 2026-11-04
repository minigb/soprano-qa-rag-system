"""Separate retrieval index for web_coaching records.

The main hybrid index is tuned on expert and web-database records; adding the
much larger web_coaching lineage to it would shift BM25 statistics and the
calibrated thresholds for every expert query. web_coaching claims are
therefore ranked in their own small index (BM25 over the claims plus the same
dense embedding model, fused by weighted reciprocal rank) and are consulted
only for questions without a selected range.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Sequence

from soprano_qa.retrieval import SearchResult, tokenize

WEB_COACHING_EVIDENCE_TYPE = "web_coaching"


def split_web_coaching_records(
    records: Sequence[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (main-index records, web_coaching records)."""

    main = [r for r in records if r.get("evidence_type") != WEB_COACHING_EVIDENCE_TYPE]
    coaching = [r for r in records if r.get("evidence_type") == WEB_COACHING_EVIDENCE_TYPE]
    return main, coaching


class CoachingIndex:
    def __init__(
        self,
        records: Sequence[Dict[str, Any]],
        *,
        embedder: Any = None,
        cache_path: str | None = None,
        lexical_weight: float = 0.35,
        dense_weight: float = 0.65,
        rrf_k: int = 60,
        k1: float = 1.4,
        b: float = 0.75,
    ) -> None:
        self.records = list(records)
        self.embedder = embedder
        self.lexical_weight = float(lexical_weight)
        self.dense_weight = float(dense_weight)
        self.rrf_k = int(rrf_k)
        self.k1, self.b = k1, b
        texts = [str(r.get("retrieval_text") or r["answer"]) for r in self.records]
        self.tokens = [tokenize(text) for text in texts]
        self.term_freqs = [Counter(tokens) for tokens in self.tokens]
        self.avgdl = sum(map(len, self.tokens)) / max(1, len(self.tokens))
        document_frequency = Counter(t for tokens in self.tokens for t in set(tokens))
        n = len(self.tokens)
        self.idf = {
            term: math.log(1 + (n - df + 0.5) / (df + 0.5))
            for term, df in document_frequency.items()
        }
        self.vectors: List[tuple[float, ...]] = []
        if embedder is not None and self.records:
            self.vectors = self._document_vectors(texts, cache_path)

    def _document_vectors(self, texts: List[str], cache_path: str | None):
        from soprano_qa.dense import (
            _embedding_fingerprint,
            _load_cached_embeddings,
            _normalized_vector,
            _write_cached_embeddings,
        )

        fingerprint = _embedding_fingerprint(self.records, texts, texts, self.embedder)
        cached = _load_cached_embeddings(cache_path, fingerprint) if cache_path else None
        if cached is not None and len(cached) == len(texts):
            return list(cached)
        vectors = [_normalized_vector(v) for v in self.embedder.embed_documents(texts)]
        if len(vectors) != len(texts):
            raise ValueError("Embedder returned %d vectors for %d web_coaching records" % (len(vectors), len(texts)))
        if cache_path:
            _write_cached_embeddings(cache_path, fingerprint, vectors)
        return vectors

    def _bm25(self, index: int, query_tokens: List[str]) -> float:
        tf, length = self.term_freqs[index], len(self.tokens[index])
        score = 0.0
        for term in query_tokens:
            if term in tf:
                score += self.idf[term] * tf[term] * (self.k1 + 1) / (
                    tf[term] + self.k1 * (1 - self.b + self.b * length / self.avgdl)
                )
        return score

    def search(self, query: str, piece: str | None, top_k: int) -> List[SearchResult]:
        """Rank this piece's web_coaching records for a no-range question."""

        candidates = [
            i for i, record in enumerate(self.records)
            if not piece or record.get("piece") == piece
        ]
        if not candidates or top_k < 1:
            return []
        query_tokens = tokenize(query)
        lexical = {i: self._bm25(i, query_tokens) for i in candidates}
        dense: Dict[int, float] = {}
        if self.vectors:
            from soprano_qa.dense import _normalized_vector

            query_vector = _normalized_vector(self.embedder.embed_query(query))
            dense = {
                i: sum(q * d for q, d in zip(query_vector, self.vectors[i]))
                for i in candidates
            }
        lexical_rank = {
            i: rank for rank, i in enumerate(sorted(candidates, key=lambda i: -lexical[i]), start=1)
        }
        dense_rank = {
            i: rank for rank, i in enumerate(sorted(candidates, key=lambda i: -dense.get(i, 0.0)), start=1)
        }
        fused = {}
        for i in candidates:
            score = self.lexical_weight / (self.rrf_k + lexical_rank[i]) if lexical[i] > 0 else 0.0
            if dense:
                score += self.dense_weight / (self.rrf_k + dense_rank[i])
            fused[i] = score
        ranked = sorted((i for i in candidates if fused[i] > 0), key=lambda i: -fused[i])[:top_k]
        return [
            SearchResult(
                record=self.records[i],
                score=fused[i],
                text_score=lexical[i],
                measure_score=0.0,
                piece_score=1.0,
                scope_match="general_evidence",
                dense_score=dense.get(i, 0.0),
                dense_content_score=dense.get(i, 0.0),
                fusion_score=fused[i],
                retrieval_mode="hybrid" if dense else "lexical",
            )
            for i in ranked
        ]
