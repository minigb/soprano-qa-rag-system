"""Local dense embeddings and hybrid retrieval for the Soprano QA corpus."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import threading
from collections import OrderedDict
from dataclasses import replace
from typing import Any, Dict, List, Protocol, Sequence

from soprano_qa.retrieval import (
    BM25Index,
    PIECE_QUERY_ALIASES,
    SearchResult,
    concept_matches,
    concept_coverage,
    is_broad_performance_guidance_query,
    measure_boost,
    measure_scope_match,
    query_components,
    scope_priority,
    semantic_concepts,
    strip_question_measure_mentions,
    token_groups,
)


DEFAULT_QUERY_INSTRUCTION = (
    "Given a Korean question about soprano singing and musical scores, "
    "retrieve the relevant evidence passage that answers the question."
)

DENSE_DOMAIN_ANCHOR_RE = re.compile(
    r"(?:"
    r"악보|음악|작품|곡|노래|성악|가창|소프라노|피아노|건반|반주|"
    r"오케스트라|선율|멜로디|리듬|박자|강박|약박|음표|음정|음형|"
    r"화음|화성|조성|악센트|강세|스타카토|트레몰로|페르마타|"
    r"셋잇단음|꾸밈음|보표|마디|가사|발음|모음|호흡|음역|고음|"
    r"작곡|형식|프레이즈|템포|다이나믹|세뇨|코다|카덴차|지시어|"
    r"주법|기호|옥타브|도약|연주|아리아|간주|"
    r"\b(?:coda|segno|dolce|amorosa|ossia|fermata|staccato|"
    r"tremolo|piano|forte|crescendo|decrescendo)\b"
    r")",
    flags=re.IGNORECASE,
)
DENSE_COORDINATED_RELATION_RE = re.compile(
    r"(?P<left>[^?？.]{1,40}?)(?:와|과)\s*"
    r"(?P<right>[^?？.]{1,40}?)(?:은|는|이|가)\s*"
    r"(?:어떤\s*)?(?:관련|관계|차이|같)",
    flags=re.IGNORECASE,
)


def _constraint(
    name: str,
    request: str,
    support: str,
) -> tuple[str, re.Pattern[str], re.Pattern[str]]:
    return (
        name,
        re.compile(request, flags=re.IGNORECASE | re.DOTALL),
        re.compile(support, flags=re.IGNORECASE | re.DOTALL),
    )


DENSE_REQUESTED_CONSTRAINTS = (
    _constraint(
        "fingering",
        r"(?:손가락\s*(?:번호|순서)|운지(?:법|번호)?|핑거링).{0,30}"
        r"(?:무엇|어떻|어느|몇|추천|권장|알려|인가)|"
        r"(?:추천|권장|어느|어떤).{0,20}"
        r"(?:손가락\s*(?:번호|순서)?|운지(?:법)?|핑거링)",
        r"(?:[1-5]\s*번|(?:첫|두|세|네|다섯)\s*번째)\s*손가락|"
        r"손가락(?:\s*번호)?(?:은|는|이|가)?\s*[1-5]\s*번|"
        r"엄지|검지|중지|약지|소지|새끼손가락",
    ),
    _constraint(
        "handedness",
        r"(?:어느|어떤|무슨)\s*손(?:으로|이|인가|을)|"
        r"(?:왼손|오른손)\s*(?:인가요|인가|입니까|일까요|일까)|"
        r"(?:왼손|오른손)\s*(?:으로|로)\s*(?:쳐|치|연주|해야|하)"
        r"[가-힣\s]{0,10}(?:나요|습니까|건가요|맞나요|될까요)"
        r"\s*[?？.]?\s*$",
        r"(?:왼손|오른손).{0,12}"
        r"(?:연주|치|맡|사용|파트|성부|입니다|이다)",
    ),
    _constraint(
        "roman_harmony",
        r"로마\s*(?:숫자|기호).{0,24}(?:화성|화음|코드|진행|분석)|"
        r"(?:화성|화음|코드|진행|분석).{0,24}로마\s*(?:숫자|기호)",
        r"(?<![a-z])(?:vii|iii|vi|iv|ii|v|i)"
        r"(?:[°ø+]?(?:6|64|7|65|43|42|9)?)?(?![a-z])"
        r"(?:\s*(?:[-–—→>]|에서|to)\s*|\s+)"
        r"(?<![a-z])(?:vii|iii|vi|iv|ii|v|i)"
        r"(?:[°ø+]?(?:6|64|7|65|43|42|9)?)?(?![a-z])",
    ),
    _constraint(
        "chord_identity",
        r"(?:무슨|어떤)\s*(?:화음|코드)(?:으로|로)?\s*"
        r"(?:구성|이루|되어|쓰|사용|인가|일까|입니까)|"
        r"(?:화음|코드).{0,6}(?:무엇|뭐)",
        r"(?<![a-z])[a-g][#♯b♭]?(?:m|maj|min|dim|aug|sus|add|\d)+(?![a-z])|"
        r"(?:장|단|증|감)\s*(?:3|7)?\s*화음|"
        r"(?:으뜸|딸림|버금딸림)\s*화음",
    ),
    _constraint(
        "exact_count",
        r"(?:정확히|정확한)?\s*몇\s*(?:개|개의|음|음표|박|마디|번)",
        r"(?:\d+|한|두|세|네|다섯|여섯|일곱|여덟|아홉|열)\s*"
        r"개(?:의)?\s*(?:음표|음정|음(?!악)|노트)|"
        r"(?:음표|음정|노트).{0,5}"
        r"(?:\d+|한|두|세|네|다섯|여섯|일곱|여덟|아홉|열)\s*개",
    ),
    _constraint(
        "metronome_value",
        r"(?:몇|정확한|정확히|수치).{0,25}(?:bpm|메트로놈)|"
        r"(?:bpm|메트로놈).{0,25}(?:몇|정확|수치)",
        r"\d{2,3}\s*(?:bpm|박\s*/?\s*분)|"
        r"(?:m\.?m\.?|메트로놈|♩|♪)\s*(?:은|는)?\s*=?\s*\d{2,3}",
    ),
    _constraint(
        "tongue_muscle",
        r"혀.{0,8}(?:어느|어떤|무슨).{0,4}근육|"
        r"(?:어느|어떤|무슨).{0,8}혀.{0,4}근육",
        r"이설근|설골설근|경상설근|구개설근|상종설근|하종설근|"
        r"종근|횡근|수직근|내재근|외재근",
    ),
    _constraint(
        "decibels",
        r"(?:db|데시벨)",
        r"\d+(?:\.\d+)?\s*(?:db|데시벨)",
    ),
    _constraint(
        "frequency",
        r"(?:hz|헤르츠|주파수|초당\s*몇\s*번\s*진동)",
        r"\d+(?:\.\d+)?\s*(?:hz|헤르츠)|"
        r"(?:초당|1\s*초(?:에|당)?)\s*\d+(?:\.\d+)?\s*(?:번|회)",
    ),
    _constraint(
        "page_location",
        # Bare ``어느 쪽`` normally means "which option/side", not a page.
        # Treat only an explicit page noun or an unambiguous page counter as
        # a requested answer slot.
        r"(?:몇\s*(?:페이지|쪽)|어느\s*페이지)|"
        r"(?:페이지|쪽).{0,20}어디",
        r"p{1,2}\.?\s*\d+|\d+\s*(?:쪽|페이지)",
    ),
    _constraint(
        "notation_page_subject",
        r"(?:d\.?\s*s\.?|dal\s+segno|세뇨|코다|coda).{0,40}"
        r"(?:페이지|쪽)|(?:페이지|쪽).{0,40}"
        r"(?:d\.?\s*s\.?|dal\s+segno|세뇨|코다|coda)",
        r"(?:d\.?\s*s\.?|dal\s+segno|세뇨|코다|coda).{0,40}"
        r"(?:p{1,2}\.?\s*\d+|\d+\s*(?:쪽|페이지))|"
        r"(?:p{1,2}\.?\s*\d+|\d+\s*(?:쪽|페이지)).{0,40}"
        r"(?:d\.?\s*s\.?|dal\s+segno|세뇨|코다|coda)",
    ),
    _constraint(
        "biographical_date",
        r"출생(?:한)?\s*(?:연도|년도|해|날짜)|"
        r"언제\s*(?:태어|출생)",
        r"(?:출생|태어).{0,24}(?<![0-9])"
        r"(?:1[0-9]{3}|20[0-9]{2})(?![0-9])|"
        r"(?<![0-9])(?:1[0-9]{3}|20[0-9]{2})(?![0-9])"
        r".{0,24}(?:출생|태어)",
    ),
    _constraint(
        "physical_dimension",
        r"(?:가로|세로|너비|폭|길이).{0,30}"
        r"(?:몇|cm|mm|센티미터|밀리미터)|"
        r"(?:몇).{0,20}(?:cm|mm|센티미터|밀리미터)",
        r"(?:\d+(?:\.\d+)?\s*(?:cm|mm)|"
        r"\d+(?:\.\d+)?\s*(?:센티미터|밀리미터))",
    ),
    _constraint(
        "pitch_name",
        r"(?:정확한\s*)?(?:음정|음고|계이름).{0,12}(?:이름|명칭)|"
        r"(?:무슨|어떤)\s*(?:음정|음고|계이름)",
        r"(?:완전|장|단|증|감)\s*(?:\d+|일|이|삼|사|오|육|칠|팔)\s*도|"
        r"(?<![a-z])[a-g][#♯b♭]?[0-8](?![a-z0-9])|"
        r"(?:가온|높은|낮은|[1-8]\s*옥타브)\s*"
        r"(?:도|레|미|파|솔|라|시)",
    ),
    _constraint(
        "score_conversion",
        r"(?:기타\s*(?:타브|코드)|타브\s*악보).{0,30}"
        r"(?:변환|바꿔|편곡)|"
        r"(?:변환|바꿔|편곡).{0,30}"
        r"(?:기타\s*(?:타브|코드)|타브\s*악보)",
        r"(?:기타|guitar).{0,20}(?:타브|탭|tab|코드|chord)",
    ),
)


def missing_dense_answer_constraints(query: str, answer: str) -> List[str]:
    """Return explicit requested attributes absent from an answer passage."""

    return [
        name
        for name, request_pattern, support_pattern in DENSE_REQUESTED_CONSTRAINTS
        if request_pattern.search(query) and not support_pattern.search(answer)
    ]


class Embedder(Protocol):
    """Minimal embedding contract used by the in-memory dense index."""

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        ...

    def embed_query(self, text: str) -> List[float]:
        ...


class SearchIndex(Protocol):
    """Search contract shared by lexical and hybrid indexes."""

    records: List[Dict[str, Any]]

    def search(
        self,
        query: str,
        piece: str | None = None,
        measure_ranges: List[List[int]] | None = None,
        topic: str | None = None,
        top_k: int = 8,
    ) -> List[SearchResult]:
        ...


class DenseRetrievalUnavailable(RuntimeError):
    """Raised when the configured local embedding backend cannot start."""


class LlamaCppQwen3Embedder:
    """Qwen3 GGUF embedding backend using the existing llama.cpp runtime."""

    def __init__(
        self,
        model_path: str,
        *,
        query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
        n_ctx: int = 2048,
        n_batch: int = 2048,
        batch_size: int = 8,
        n_gpu_layers: int = -1,
    ) -> None:
        if not os.path.isfile(model_path):
            raise DenseRetrievalUnavailable(
                "Embedding checkpoint not found: %s" % model_path
            )
        try:
            import llama_cpp
        except ImportError as exc:
            raise DenseRetrievalUnavailable(
                "llama-cpp-python is unavailable for dense retrieval"
            ) from exc
        if not hasattr(llama_cpp.Llama, "embed") or not hasattr(
            llama_cpp,
            "LLAMA_POOLING_TYPE_LAST",
        ):
            raise DenseRetrievalUnavailable(
                "Installed llama-cpp-python lacks sequence embedding support"
            )

        self.model_path = os.path.abspath(model_path)
        self.query_instruction = query_instruction.strip()
        self.batch_size = max(1, int(batch_size))
        self.n_ctx = max(512, int(n_ctx))
        self.n_batch = max(512, int(n_batch))
        self._lock = threading.Lock()
        try:
            self._model = llama_cpp.Llama(
                model_path=self.model_path,
                embedding=True,
                pooling_type=llama_cpp.LLAMA_POOLING_TYPE_LAST,
                n_ctx=self.n_ctx,
                n_batch=self.n_batch,
                n_gpu_layers=int(n_gpu_layers),
                verbose=False,
            )
        except Exception as exc:
            raise DenseRetrievalUnavailable(
                "Could not load embedding checkpoint: %s" % exc
            ) from exc

        stat = os.stat(self.model_path)
        self.cache_identity = json.dumps(
            {
                "backend": "llama-cpp-python",
                "backend_version": getattr(llama_cpp, "__version__", None),
                "model_path": self.model_path,
                "model_size": stat.st_size,
                "model_mtime_ns": stat.st_mtime_ns,
                "n_ctx": self.n_ctx,
                "n_batch": self.n_batch,
                "pooling": "last",
                "normalize": True,
                "truncate": True,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def _coerce_vectors(
        encoded: Any,
        expected_count: int,
    ) -> List[List[float]]:
        if expected_count == 1 and encoded and isinstance(encoded[0], (int, float)):
            encoded = [encoded]
        if not isinstance(encoded, list) or len(encoded) != expected_count:
            raise DenseRetrievalUnavailable(
                "Embedding backend returned an unexpected batch shape"
            )
        vectors = []
        for vector in encoded:
            if not isinstance(vector, list) or not vector:
                raise DenseRetrievalUnavailable(
                    "Embedding backend returned an empty vector"
                )
            values = [float(value) for value in vector]
            if not all(math.isfinite(value) for value in values):
                raise DenseRetrievalUnavailable(
                    "Embedding backend returned a non-finite vector"
                )
            vectors.append(values)
        return vectors

    def _embed_many(self, texts: Sequence[str]) -> List[List[float]]:
        output: List[List[float]] = []
        with self._lock:
            for start in range(0, len(texts), self.batch_size):
                batch = list(texts[start : start + self.batch_size])
                if not batch:
                    continue
                try:
                    encoded = self._model.embed(
                        batch,
                        normalize=True,
                        truncate=True,
                    )
                    output.extend(self._coerce_vectors(encoded, len(batch)))
                except Exception as exc:
                    raise DenseRetrievalUnavailable(
                        "Embedding inference failed: %s" % exc
                    ) from exc
        return output

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return self._embed_many(texts)

    def embed_query(self, text: str) -> List[float]:
        prompt = "Instruct: %s\nQuery: %s" % (
            self.query_instruction,
            text.strip(),
        )
        return self._embed_many([prompt])[0]


def dense_document_text(record: Dict[str, Any]) -> str:
    """Return semantic unit text without provenance-only retrieval metadata."""

    return str(record.get("relevance_text") or record.get("answer") or "").strip()


def dense_query_text(query: str, piece: str | None) -> str:
    """Remove deterministic routing syntax before embedding a natural query."""

    stripped = query
    if piece:
        aliases = sorted(
            PIECE_QUERY_ALIASES.get(piece, []),
            key=len,
            reverse=True,
        )
        for alias in aliases:
            stripped = re.sub(
                re.escape(alias),
                " ",
                stripped,
                flags=re.IGNORECASE,
            )
    stripped = strip_question_measure_mentions(stripped)
    return " ".join(stripped.split()) or query.strip()


def _normalized_vector(vector: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in vector)
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("Embeddings must contain finite numeric values")
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 0:
        raise ValueError("Embeddings must not be zero vectors")
    return tuple(value / norm for value in values)


def _embedding_fingerprint(
    records: Sequence[Dict[str, Any]],
    relevance_texts: Sequence[str],
    content_texts: Sequence[str],
    embedder: Embedder,
) -> str:
    payload = {
        "version": 1,
        "embedder": getattr(embedder, "cache_identity", None)
        or "%s:%d" % (type(embedder).__qualname__, id(embedder)),
        "documents": [
            {
                "id": record.get("id"),
                "relevance_text": relevance_text,
                "content_text": content_text,
            }
            for record, relevance_text, content_text in zip(
                records,
                relevance_texts,
                content_texts,
            )
        ],
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _load_cached_embeddings(
    cache_path: str,
    fingerprint: str,
) -> List[tuple[float, ...]] | None:
    try:
        with open(cache_path, encoding="utf-8") as file:
            payload = json.load(file)
        if not isinstance(payload, dict):
            return None
        if payload.get("version") != 1 or payload.get("fingerprint") != fingerprint:
            return None
        raw_vectors = payload.get("vectors", [])
        if not isinstance(raw_vectors, list) or not raw_vectors:
            return None
        vectors = [
            _normalized_vector(vector)
            for vector in raw_vectors
        ]
        if len({len(vector) for vector in vectors}) != 1:
            return None
        return vectors
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _write_cached_embeddings(
    cache_path: str,
    fingerprint: str,
    vectors: Sequence[Sequence[float]],
) -> None:
    directory = os.path.dirname(os.path.abspath(cache_path))
    try:
        os.makedirs(directory, exist_ok=True)
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=".dense-embeddings-",
            suffix=".json",
            dir=directory,
            text=True,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "version": 1,
                        "fingerprint": fingerprint,
                        "vectors": vectors,
                    },
                    file,
                    separators=(",", ":"),
                )
            os.replace(temporary_path, cache_path)
        except Exception:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass
            raise
    except OSError:
        # A read-only deployment can still use the in-memory vectors.
        return


class HybridIndex:
    """Fuse BM25 and dense ranks while retaining deterministic scope authority."""

    retrieval_mode = "hybrid"
    configured_retrieval_mode = "hybrid"
    retrieval_fallback_reason: str | None = None

    def __init__(
        self,
        records: List[Dict[str, Any]],
        *,
        embedder: Embedder,
        dense_min_score: float = 0.44,
        dense_min_content_score: float = 0.43,
        dense_max_alias_gap: float = 0.20,
        dense_candidate_k: int = 24,
        lexical_weight: float = 0.35,
        dense_weight: float = 0.65,
        rrf_k: int = 60,
        query_cache_size: int = 128,
        fallback_to_lexical: bool = True,
        cache_path: str | None = None,
    ) -> None:
        if not -1.0 <= dense_min_score <= 1.0:
            raise ValueError("dense_min_score must be between -1 and 1")
        if not -1.0 <= dense_min_content_score <= 1.0:
            raise ValueError("dense_min_content_score must be between -1 and 1")
        if not 0.0 <= dense_max_alias_gap <= 2.0:
            raise ValueError("dense_max_alias_gap must be between 0 and 2")
        if dense_candidate_k < 1:
            raise ValueError("dense_candidate_k must be positive")
        if lexical_weight <= 0 or dense_weight <= 0:
            raise ValueError("Hybrid weights must both be positive")
        if rrf_k < 1:
            raise ValueError("rrf_k must be positive")
        if query_cache_size < 1:
            raise ValueError("query_cache_size must be positive")

        self.records = records
        self.lexical = BM25Index(records)
        self.embedder = embedder
        self.dense_min_score = float(dense_min_score)
        self.dense_min_content_score = float(dense_min_content_score)
        self.dense_max_alias_gap = float(dense_max_alias_gap)
        self.dense_candidate_k = int(dense_candidate_k)
        self.lexical_weight = float(lexical_weight)
        self.dense_weight = float(dense_weight)
        self.rrf_k = int(rrf_k)
        self.query_cache_size = int(query_cache_size)
        self.fallback_to_lexical = bool(fallback_to_lexical)
        self._search_state = threading.local()
        self._set_search_state(
            "not_searched",
            reason="index_initialized",
        )
        self.embedding_model = getattr(
            embedder,
            "model_path",
            type(embedder).__qualname__,
        )
        self._record_indices = {
            record["id"]: index for index, record in enumerate(records)
        }
        self._piece_concepts: Dict[str, set[str]] = {}
        for index, record in enumerate(records):
            self._piece_concepts.setdefault(record.get("piece", ""), set()).update(
                self.lexical.document_concepts[index]
            )
        self._query_cache: OrderedDict[str, tuple[float, ...]] = OrderedDict()
        self._query_cache_lock = threading.Lock()

        relevance_texts = [dense_document_text(record) for record in records]
        content_texts = [
            str(record.get("answer") or "").strip()
            for record in records
        ]
        fingerprint = _embedding_fingerprint(
            records,
            relevance_texts,
            content_texts,
            embedder,
        )
        cached = (
            _load_cached_embeddings(cache_path, fingerprint)
            if cache_path
            else None
        )
        if cached is not None and len(cached) == 2 * len(records):
            vectors = cached
        else:
            unique_texts = list(dict.fromkeys([*relevance_texts, *content_texts]))
            try:
                unique_vectors = embedder.embed_documents(unique_texts)
            except DenseRetrievalUnavailable:
                raise
            except Exception as exc:
                raise DenseRetrievalUnavailable(
                    "Document embedding inference failed: %s" % exc
                ) from exc
            if len(unique_vectors) != len(unique_texts):
                raise DenseRetrievalUnavailable(
                    "Embedder returned %d document vectors for %d records"
                    % (len(unique_vectors), len(unique_texts))
                )
            try:
                vector_by_text = {
                    text: _normalized_vector(vector)
                    for text, vector in zip(unique_texts, unique_vectors)
                }
            except (TypeError, ValueError) as exc:
                raise DenseRetrievalUnavailable(
                    "Document embedding normalization failed: %s" % exc
                ) from exc
            vectors = [
                *(vector_by_text[text] for text in relevance_texts),
                *(vector_by_text[text] for text in content_texts),
            ]
            if cache_path:
                _write_cached_embeddings(
                    cache_path,
                    fingerprint,
                    vectors,
                )
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) != 1:
            raise ValueError("Document embeddings have inconsistent dimensions")
        self.document_embeddings = vectors[: len(records)]
        self.content_embeddings = vectors[len(records) :]
        self.embedding_dimension = next(iter(dimensions), 0)

    @property
    def last_dense_error(self) -> str | None:
        return getattr(self._search_state, "last_dense_error", None)

    @last_dense_error.setter
    def last_dense_error(self, value: str | None) -> None:
        self._search_state.last_dense_error = value

    @property
    def last_search_mode(self) -> str:
        return getattr(self._search_state, "last_search_mode", "not_searched")

    @last_search_mode.setter
    def last_search_mode(self, value: str) -> None:
        self._search_state.last_search_mode = value

    @property
    def last_route_reason(self) -> str | None:
        return getattr(self._search_state, "last_route_reason", None)

    @last_route_reason.setter
    def last_route_reason(self, value: str | None) -> None:
        self._search_state.last_route_reason = value

    @property
    def dense_attempted(self) -> bool:
        return bool(getattr(self._search_state, "dense_attempted", False))

    @dense_attempted.setter
    def dense_attempted(self, value: bool) -> None:
        self._search_state.dense_attempted = bool(value)

    @property
    def dense_contributed(self) -> bool:
        return bool(getattr(self._search_state, "dense_contributed", False))

    @dense_contributed.setter
    def dense_contributed(self, value: bool) -> None:
        self._search_state.dense_contributed = bool(value)

    @property
    def fallback_used(self) -> bool:
        return bool(getattr(self._search_state, "fallback_used", False))

    @fallback_used.setter
    def fallback_used(self, value: bool) -> None:
        self._search_state.fallback_used = bool(value)

    def _set_search_state(
        self,
        mode: str,
        *,
        reason: str,
        dense_attempted: bool = False,
        dense_contributed: bool = False,
        fallback_used: bool = False,
        dense_error: str | None = None,
    ) -> None:
        """Record one query's route without conflating routing and failure."""

        self.last_search_mode = mode
        self.last_route_reason = reason
        self.dense_attempted = dense_attempted
        self.dense_contributed = dense_contributed
        self.fallback_used = fallback_used
        self.last_dense_error = dense_error

    def _query_embedding(self, query: str, piece: str | None) -> tuple[float, ...]:
        prepared_query = dense_query_text(query, piece)
        cache_key = hashlib.sha256(prepared_query.encode("utf-8")).hexdigest()
        with self._query_cache_lock:
            cached = self._query_cache.pop(cache_key, None)
            if cached is not None:
                self._query_cache[cache_key] = cached
                return cached
            try:
                query_vector = _normalized_vector(
                    self.embedder.embed_query(prepared_query)
                )
            except DenseRetrievalUnavailable:
                raise
            except Exception as exc:
                raise DenseRetrievalUnavailable(
                    "Query embedding inference failed: %s" % exc
                ) from exc
            if len(query_vector) != self.embedding_dimension:
                raise DenseRetrievalUnavailable(
                    "Query embedding dimension %d does not match document "
                    "dimension %d"
                    % (len(query_vector), self.embedding_dimension)
                )
            self._query_cache[cache_key] = query_vector
            while len(self._query_cache) > self.query_cache_size:
                self._query_cache.popitem(last=False)
            return query_vector

    def _dense_candidates(
        self,
        *,
        query: str,
        piece: str | None,
        measure_ranges: List[List[int]],
        topic: str | None,
    ) -> List[tuple[int, float, float, str]]:
        query_vector = self._query_embedding(query, piece)

        candidates: List[tuple[int, float, float, str]] = []
        for index, record in enumerate(self.records):
            if not record.get("retrieval_eligible", True):
                continue
            if piece and record.get("piece") != piece:
                continue
            if topic and topic not in record.get("topic", ""):
                continue
            scope_match = measure_scope_match(record, measure_ranges)
            if scope_match in {
                "unscoped_pending_context",
                "unspecified_context",
            }:
                continue
            similarity = sum(
                query_value * document_value
                for query_value, document_value in zip(
                    query_vector,
                    self.document_embeddings[index],
                )
            )
            content_similarity = sum(
                query_value * document_value
                for query_value, document_value in zip(
                    query_vector,
                    self.content_embeddings[index],
                )
            )
            if (
                similarity < self.dense_min_score
                or content_similarity < self.dense_min_content_score
            ):
                continue
            candidates.append(
                (index, similarity, content_similarity, scope_match)
            )

        candidates.sort(
            key=lambda item: (
                scope_priority(item[3], bool(measure_ranges)),
                item[1],
                item[2],
                self.records[item[0]]["id"],
            ),
            reverse=True,
        )
        return candidates[: self.dense_candidate_k]

    def _has_dense_domain_anchor(
        self,
        query: str,
        query_concepts: Sequence[str],
        piece: str | None,
    ) -> bool:
        if DENSE_DOMAIN_ANCHOR_RE.search(query):
            return True
        if piece:
            document_concepts = self._piece_concepts.get(piece, set())
        else:
            document_concepts = set().union(*self._piece_concepts.values())
        return any(
            concept_matches(concept, document_concepts)
            for concept in query_concepts
        )

    def _has_unsupported_coordinated_subject(
        self,
        query: str,
        piece: str | None,
    ) -> bool:
        """Reject relation questions that introduce an unknown second domain."""

        if piece:
            document_concepts = self._piece_concepts.get(piece, set())
        else:
            document_concepts = set().union(*self._piece_concepts.values())
        for match in DENSE_COORDINATED_RELATION_RE.finditer(query):
            for subject in (match.group("left"), match.group("right")):
                subject_concepts = semantic_concepts(
                    subject,
                    piece,
                    query=True,
                )
                if any(
                    not concept_matches(concept, document_concepts)
                    for concept in subject_concepts
                ):
                    return True
        return False

    def search(
        self,
        query: str,
        piece: str | None = None,
        measure_ranges: List[List[int]] | None = None,
        topic: str | None = None,
        top_k: int = 8,
    ) -> List[SearchResult]:
        measure_ranges = measure_ranges or []
        if top_k < 1 or not query.strip():
            self._set_search_state(
                "none",
                reason="blank_query_or_nonpositive_top_k",
            )
            return []
        lexical_results = self.lexical.search(
            query=query,
            piece=piece,
            measure_ranges=measure_ranges,
            topic=topic,
            top_k=max(top_k, self.dense_candidate_k),
        )
        if (
            not measure_ranges
            and piece
            and is_broad_performance_guidance_query(query, piece)
        ):
            self._set_search_state(
                "lexical_route",
                reason="broad_performance_guidance_router",
            )
            return lexical_results[:top_k]
        real_query, intent_anchors = query_components(query, piece)
        if not token_groups(real_query) and not intent_anchors:
            self._set_search_state(
                "lexical_route",
                reason="no_dense_query_content",
            )
            return lexical_results[:top_k]
        query_concepts = semantic_concepts(query, piece, query=True)
        has_dense_domain_anchor = self._has_dense_domain_anchor(
            query,
            query_concepts,
            piece,
        ) and not self._has_unsupported_coordinated_subject(query, piece)
        if not has_dense_domain_anchor and not lexical_results:
            self._set_search_state(
                "lexical_route",
                reason="no_domain_anchor_or_lexical_match",
            )
            return []

        try:
            dense_candidates = self._dense_candidates(
                query=query,
                piece=piece,
                measure_ranges=measure_ranges,
                topic=topic,
            )
        except DenseRetrievalUnavailable as exc:
            dense_error = "%s: %s" % (type(exc).__name__, exc)
            self._set_search_state(
                "lexical_fallback",
                reason="dense_query_error",
                dense_attempted=True,
                fallback_used=True,
                dense_error=dense_error,
            )
            if not self.fallback_to_lexical:
                raise
            return lexical_results[:top_k]

        lexical_ranks = {
            result.record["id"]: rank
            for rank, result in enumerate(lexical_results, start=1)
        }
        lexical_by_id = {
            result.record["id"]: result for result in lexical_results
        }
        dense_by_id = {}
        for rank, (
            index,
            similarity,
            content_similarity,
            scope_match,
        ) in enumerate(dense_candidates, start=1):
            record_id = self.records[index]["id"]
            if record_id not in lexical_by_id:
                if not has_dense_domain_anchor:
                    continue
                if (
                    similarity - content_similarity
                    > self.dense_max_alias_gap
                ):
                    continue
                if missing_dense_answer_constraints(
                    query,
                    str(self.records[index].get("answer") or ""),
                ):
                    continue
            dense_by_id[record_id] = (
                rank,
                similarity,
                content_similarity,
                scope_match,
            )
        candidate_ids = list(lexical_by_id)
        lexical_alias_is_authoritative = any(
            result.alias_score > 0
            for result in lexical_results
        )
        confident_other_range_subject = bool(measure_ranges) and any(
            result.scope_match == "other_range_context"
            and result.concept_coverage >= 0.8
            for result in lexical_results
        )
        allow_dense_only = (
            not lexical_results
            or (
                not lexical_alias_is_authoritative
                and not confident_other_range_subject
            )
        )
        if allow_dense_only:
            dense_only_ids = [
                record_id
                for record_id in dense_by_id
                if record_id not in lexical_by_id
            ]
            if measure_ranges:
                dense_only_ids = [
                    record_id
                    for record_id in dense_only_ids
                    if dense_by_id[record_id][3] == "overlaps_query_range"
                ]
            candidate_ids.extend(
                dense_only_ids
            )
        if not candidate_ids:
            self._set_search_state(
                "hybrid_no_dense_match",
                reason="no_retrieval_candidates",
                dense_attempted=True,
            )
            return []

        normalizer = (
            (self.lexical_weight + self.dense_weight) / (self.rrf_k + 1)
        )
        merged: List[SearchResult] = []
        for record_id in candidate_ids:
            lexical_result = lexical_by_id.get(record_id)
            (
                dense_rank,
                dense_score,
                dense_content_score,
                dense_scope,
            ) = dense_by_id.get(
                record_id,
                (None, 0.0, 0.0, None),
            )
            lexical_rank = lexical_ranks.get(record_id)
            fusion_score = 0.0
            if lexical_rank is not None:
                fusion_score += self.lexical_weight / (
                    self.rrf_k + lexical_rank
                )
            if dense_rank is not None:
                fusion_score += self.dense_weight / (
                    self.rrf_k + dense_rank
                )
            fusion_score /= normalizer

            if lexical_result is not None:
                result = replace(
                    lexical_result,
                    dense_score=dense_score,
                    dense_content_score=dense_content_score,
                    fusion_score=fusion_score,
                    retrieval_mode=(
                        "hybrid" if dense_rank is not None else "lexical"
                    ),
                )
            else:
                index = self._record_indices[record_id]
                record = self.records[index]
                result = SearchResult(
                    record=record,
                    score=dense_score,
                    text_score=0.0,
                    measure_score=measure_boost(record, measure_ranges),
                    piece_score=(
                        1.0 if piece and record.get("piece") == piece else 0.0
                    ),
                    scope_match=str(dense_scope),
                    concept_coverage=concept_coverage(
                        query_concepts,
                        self.lexical.document_concepts[index],
                    ),
                    content_concept_coverage=concept_coverage(
                        query_concepts,
                        self.lexical.content_concepts[index],
                    ),
                    semantic_match_type="dense",
                    dense_score=dense_score,
                    dense_content_score=dense_content_score,
                    fusion_score=fusion_score,
                    retrieval_mode="dense",
                )
            merged.append(result)

        merged.sort(
            key=lambda result: (
                scope_priority(result.scope_match, bool(measure_ranges)),
                result.alias_score > 0,
                result.alias_score,
                result.fusion_score,
                result.dense_score,
                result.dense_content_score,
                result.measure_score,
                result.text_score,
                result.record["id"],
            ),
            reverse=True,
        )
        returned = merged[:top_k]
        dense_contributed = any(
            result.retrieval_mode in {"hybrid", "dense"}
            for result in returned
        )
        self._set_search_state(
            "hybrid" if dense_contributed else "hybrid_no_dense_match",
            reason=(
                "dense_result_returned"
                if dense_contributed
                else "no_dense_result_returned"
            ),
            dense_attempted=True,
            dense_contributed=dense_contributed,
        )
        return returned


def _lexical_fallback(
    records: List[Dict[str, Any]],
    *,
    configured_mode: str,
    reason: str | None,
) -> BM25Index:
    index = BM25Index(records)
    index.retrieval_mode = "lexical"
    index.configured_retrieval_mode = configured_mode
    index.retrieval_fallback_reason = reason
    index.last_search_mode = (
        "lexical_fallback" if reason is not None else "lexical"
    )
    index.last_route_reason = (
        "dense_index_initialization_error"
        if reason is not None
        else "configured_lexical"
    )
    index.dense_attempted = configured_mode == "hybrid"
    index.dense_contributed = False
    index.fallback_used = reason is not None
    return index


def build_retrieval_index(
    records: List[Dict[str, Any]],
    settings: Dict[str, Any],
    *,
    embedder: Embedder | None = None,
) -> SearchIndex:
    """Build the configured local index without downloading models at runtime."""

    retrieval = settings.get("retrieval") or {}
    mode = str(retrieval.get("mode") or "lexical").lower()
    if mode == "lexical":
        return _lexical_fallback(
            records,
            configured_mode=mode,
            reason=None,
        )
    if mode != "hybrid":
        raise ValueError("Unsupported retrieval mode: %s" % mode)

    try:
        active_embedder = embedder or LlamaCppQwen3Embedder(
            settings["embedding_model_path"],
            query_instruction=str(
                retrieval.get("query_instruction")
                or DEFAULT_QUERY_INSTRUCTION
            ),
            n_ctx=int(retrieval.get("n_ctx", 2048)),
            n_batch=int(retrieval.get("n_batch", 2048)),
            batch_size=int(retrieval.get("embedding_batch_size", 8)),
            n_gpu_layers=int(retrieval.get("n_gpu_layers", -1)),
        )
        index = HybridIndex(
            records,
            embedder=active_embedder,
            dense_min_score=float(retrieval.get("dense_min_score", 0.44)),
            dense_min_content_score=float(
                retrieval.get("dense_min_content_score", 0.43)
            ),
            dense_max_alias_gap=float(
                retrieval.get("dense_max_alias_gap", 0.20)
            ),
            dense_candidate_k=int(retrieval.get("dense_candidate_k", 24)),
            lexical_weight=float(retrieval.get("lexical_weight", 0.35)),
            dense_weight=float(retrieval.get("dense_weight", 0.65)),
            rrf_k=int(retrieval.get("rrf_k", 60)),
            query_cache_size=int(retrieval.get("query_cache_size", 128)),
            fallback_to_lexical=bool(
                retrieval.get("fallback_to_lexical", True)
            ),
            cache_path=settings.get("embedding_cache_path"),
        )
        index.configured_retrieval_mode = mode
        return index
    except DenseRetrievalUnavailable as exc:
        if not retrieval.get("fallback_to_lexical", True):
            raise
        return _lexical_fallback(
            records,
            configured_mode=mode,
            reason="%s: %s" % (type(exc).__name__, exc),
        )


def retrieval_diagnostics(index: SearchIndex) -> Dict[str, Any]:
    """Return stable diagnostics for API and CLI consumers."""

    last_search_mode = getattr(index, "last_search_mode", "not_searched")
    return {
        "configured_mode": getattr(
            index,
            "configured_retrieval_mode",
            "lexical",
        ),
        "active_mode": getattr(index, "retrieval_mode", "lexical"),
        "dense_available": getattr(index, "retrieval_mode", "lexical")
        == "hybrid",
        "embedding_model": getattr(index, "embedding_model", None),
        "embedding_dimension": getattr(index, "embedding_dimension", None),
        "fallback_reason": getattr(
            index,
            "retrieval_fallback_reason",
            None,
        ),
        "last_dense_error": getattr(index, "last_dense_error", None),
        "last_search_mode": last_search_mode,
        "query_route": (
            "hybrid"
            if last_search_mode in {"hybrid", "hybrid_no_dense_match"}
            else (
                "lexical"
                if last_search_mode
                in {"lexical", "lexical_route", "lexical_fallback"}
                else "none"
            )
        ),
        "route_reason": getattr(index, "last_route_reason", None),
        "dense_attempted": bool(getattr(index, "dense_attempted", False)),
        "dense_contributed": bool(
            getattr(index, "dense_contributed", False)
        ),
        "fallback_used": bool(getattr(index, "fallback_used", False)),
    }
