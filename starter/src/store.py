"""Store and search vectors using Qdrant local mode (no server, no Docker required).

Design decisions:
- One Qdrant collection per strategy (strategy_A, strategy_B, strategy_C).
- Metric: cosine (fixed). Score returned is cosine similarity ∈ [-1, 1],
  normalised to [0, 1] by sentence-transformers normalised embeddings.
- Persistence: QdrantClient(path=...) writes to disk automatically.
- Upsert is idempotent: we delete+recreate the collection on each ingest,
  which is the same behaviour as the original PgVectorStore (DELETE WHERE strategy=X).
  Existing collections are skipped when force_reingest=False (run.py controls this).
- Qdrant requires integer or UUID point IDs; we use a deterministic 64-bit int
  derived from each chunk_id string (xxhash-free, uses Python built-in hash seeded
  to 0 via hashlib) so the same chunk always maps to the same ID.
- Allowed filter keys: chapter, document_type, language.
"""
from __future__ import annotations

import hashlib
import struct
from pathlib import Path
from typing import Any

from .models import Chunk

# Lazy import so other modules can be imported without qdrant installed
def _qdrant():
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import (
            Distance,
            FieldCondition,
            Filter,
            MatchValue,
            PointStruct,
            VectorParams,
        )
        return QdrantClient, Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams
    except ImportError as e:
        raise ImportError(
            "qdrant-client is required. Install with: pip install 'qdrant-client>=1.9,<2'"
        ) from e


def _chunk_id_to_int(chunk_id: str) -> int:
    """Deterministic 63-bit integer from a string (no external deps)."""
    digest = hashlib.md5(chunk_id.encode()).digest()
    # Take first 8 bytes as unsigned 64-bit, then mask to 63 bits (Qdrant wants positive int)
    val = struct.unpack(">Q", digest[:8])[0]
    return val & 0x7FFF_FFFF_FFFF_FFFF


class QdrantVectorStore:
    """Thin wrapper around Qdrant local PersistentClient.

    Parameters
    ----------
    qdrant_dir : str | Path
        Directory where Qdrant persists its data (created automatically).
    dimensions : int
        Embedding dimension.
    """

    ALLOWED_FILTER_KEYS = {"chapter", "document_type", "language"}

    def __init__(self, qdrant_dir: str | Path, dimensions: int) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.qdrant_dir = Path(qdrant_dir)
        self.dimensions = dimensions
        self.qdrant_dir.mkdir(parents=True, exist_ok=True)

        QdrantClient, *_ = _qdrant()
        self._client = QdrantClient(path=str(self.qdrant_dir))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collection_name(self, strategy: str) -> str:
        return f"strategy_{strategy}"

    def _collection_exists(self, strategy: str) -> bool:
        name = self._collection_name(strategy)
        existing = {c.name for c in self._client.get_collections().collections}
        return name in existing

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_schema(self) -> None:
        """No-op — collections are created lazily in upsert()."""
        pass

    def upsert(
        self,
        strategy: str,
        chunks: list[Chunk],
        vectors: list[list[float]],
        force_reingest: bool = False,
    ) -> None:
        """Delete-then-insert upsert (idempotent at collection level).

        If the collection already exists and force_reingest=False, it is
        left untouched (run.py skips already-ingested strategies).
        If force_reingest=True or collection doesn't exist, it is recreated.
        """
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have equal lengths")
        for v in vectors:
            if len(v) != self.dimensions:
                raise ValueError(
                    f"embedding dimension {len(v)} does not match store ({self.dimensions})"
                )

        _, Distance, _, _, _, PointStruct, VectorParams = _qdrant()
        name = self._collection_name(strategy)

        # Recreate collection (idempotent: drop if exists)
        if self._collection_exists(strategy):
            if not force_reingest:
                # Already ingested — skip
                return
            self._client.delete_collection(name)

        self._client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=self.dimensions, distance=Distance.COSINE),
        )

        if not chunks:
            return

        # Build points
        points = []
        for chunk, vector in zip(chunks, vectors):
            point_id = _chunk_id_to_int(chunk.chunk_id)
            payload: dict[str, Any] = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "source": chunk.source,
                "page": chunk.page,
                "strategy": strategy,
                "text": chunk.text,          # store text in payload for retrieval
            }
            # Flatten scalar metadata fields
            for k, v in chunk.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    payload[k] = v

            points.append(PointStruct(id=point_id, vector=vector, payload=payload))

        # Upload in batches
        BATCH = 256
        for i in range(0, len(points), BATCH):
            self._client.upsert(collection_name=name, points=points[i : i + BATCH])

    def search(
        self,
        strategy: str,
        query_vector: list[float],
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Cosine similarity search with optional metadata filters.

        Returns list of dicts with keys: chunk_id, text, source, page, metadata, score.
        score ∈ [0, 1] where 1 = identical (cosine similarity of L2-normalised vectors).
        """
        if len(query_vector) != self.dimensions:
            raise ValueError(
                f"query vector dimension {len(query_vector)} does not match store ({self.dimensions})"
            )
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        filters = filters or {}
        unknown = set(filters) - self.ALLOWED_FILTER_KEYS
        if unknown:
            raise ValueError("unsupported metadata filter: " + ", ".join(sorted(unknown)))

        _, _, FieldCondition, Filter, MatchValue, _, _ = _qdrant()

        # Build Qdrant filter
        qdrant_filter: Any = None
        if filters:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=str(v)))
                for k, v in filters.items()
            ]
            qdrant_filter = Filter(must=conditions)

        name = self._collection_name(strategy)
        # qdrant-client >= 1.7 uses query_points() instead of search()
        response = self._client.query_points(
            collection_name=name,
            query=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )
        results = response.points

        hits: list[dict[str, Any]] = []
        for r in results:
            payload = r.payload or {}
            # Qdrant cosine score is already similarity ∈ [-1, 1]
            # For normalised embeddings it is ∈ [0, 1]
            hits.append({
                "chunk_id": payload.get("chunk_id", str(r.id)),
                "text": payload.get("text", ""),
                "source": payload.get("source", ""),
                "page": payload.get("page"),
                "metadata": payload,
                "score": float(r.score),
            })
        return hits

    def collection_count(self, strategy: str) -> int:
        name = self._collection_name(strategy)
        if not self._collection_exists(strategy):
            return 0
        return self._client.get_collection(name).points_count or 0

    def delete_collection(self, strategy: str) -> None:
        """Drop a strategy's collection entirely."""
        name = self._collection_name(strategy)
        if self._collection_exists(strategy):
            self._client.delete_collection(name)
