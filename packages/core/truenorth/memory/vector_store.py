"""
Semantic vector store for past session search.

Embeds session summaries and enables natural-language search over
prior conversations: "find all sessions where user mentioned back pain"

Backends:
  - In-memory (numpy cosine similarity) — for tests, small deployments
  - pgvector (Postgres + vector extension) — production
  - ChromaDB — alternative embedded vector DB
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

@dataclass
class SemanticSearchResult:
    session_id: str
    score:      float
    metadata:   Dict[str, Any] = field(default_factory=dict)
    snippet:    str             = ""

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "score":      round(self.score, 4),
            "metadata":   self.metadata,
            "snippet":    self.snippet[:200],
        }

class VectorStore:
    """
    Semantic search over past session summaries.

    In-memory mode uses bag-of-words cosine similarity (no ML deps).
    Production mode: swap in pgvector or ChromaDB.
    """

    def __init__(
        self,
        postgres:  Optional[Any] = None,
        embedding_fn: Optional[Any] = None,
    ):
        self._pg          = postgres
        self._embed_fn    = embedding_fn
        self._docs:       Dict[str, dict] = {}
    async def add(
        self,
        session_id: str,
        text:       str,
        metadata:   Optional[dict] = None,
    ) -> None:
        """Add a session summary to the store."""
        vector = await self._vectorize(text)
        self._docs[session_id] = {
            "text":     text,
            "vector":   vector,
            "metadata": metadata or {},
        }
        logger.debug("vector_store: indexed session=%s chars=%d", session_id, len(text))

    async def search(
        self,
        query:    str,
        top_k:   int           = 5,
        filter_meta: Optional[dict] = None,
    ) -> List[SemanticSearchResult]:
        """Semantic search. Returns top_k most similar sessions."""
        if not self._docs:
            return []

        q_vec = await self._vectorize(query)
        scores: List[tuple] = []

        for session_id, doc in self._docs.items():

            if filter_meta:
                if not all(doc["metadata"].get(k) == v for k, v in filter_meta.items()):
                    continue
            score = self._cosine(q_vec, doc["vector"])
            scores.append((score, session_id, doc))

        scores.sort(reverse=True)
        return [
            SemanticSearchResult(
                session_id = sid,
                score      = sc,
                metadata   = doc["metadata"],
                snippet    = doc["text"][:200],
            )
            for sc, sid, doc in scores[:top_k]
        ]

    async def delete(self, session_id: str) -> bool:
        """Remove a session from the store (GDPR erasure)."""
        if session_id in self._docs:
            del self._docs[session_id]
            return True
        return False

    def count(self) -> int:
        return len(self._docs)

    async def _vectorize(self, text: str) -> Dict[str, float]:
        """
        Create a sparse bag-of-words vector.
        If an embedding_fn is configured, delegates to it.
        """
        if self._embed_fn:
            try:
                return await self._embed_fn(text)
            except Exception as e:
                logger.warning("vector_store: embedding_fn failed: %s — using BoW", e)

        words  = text.lower().split()
        tf:    Dict[str, float] = {}
        total  = max(len(words), 1)
        for w in words:

            w = w.strip(".,!?;:\"'()")
            if len(w) > 2:
                tf[w] = tf.get(w, 0) + 1 / total
        return tf

    @staticmethod
    def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
        """Cosine similarity between two sparse vectors (dicts)."""
        if not a or not b:
            return 0.0
        dot   = sum(a.get(k, 0) * v for k, v in b.items())
        mag_a = math.sqrt(sum(v * v for v in a.values()))
        mag_b = math.sqrt(sum(v * v for v in b.values()))
        denom = mag_a * mag_b
        return dot / denom if denom > 0 else 0.0
