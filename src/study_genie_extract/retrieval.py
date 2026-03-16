from __future__ import annotations

from collections.abc import Callable

from .embeddings import DEFAULT_EMBEDDING_MODEL, create_embedding
from .types import IndexedChunk, RetrievedChunk


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must have the same length")

    dot_product = 0.0
    norm_left = 0.0
    norm_right = 0.0

    for left_value, right_value in zip(left, right):
        dot_product += left_value * right_value
        norm_left += left_value * left_value
        norm_right += right_value * right_value

    denominator = (norm_left ** 0.5) * (norm_right ** 0.5)
    if denominator == 0:
        return 0.0

    return dot_product / denominator


def retrieve_top_k(
    query_text: str,
    indexed_chunks: list[IndexedChunk],
    top_k: int = 5,
    model: str = DEFAULT_EMBEDDING_MODEL,
    api_key: str | None = None,
    embed_query: Callable[[str, str, str | None], list[float]] = create_embedding,
) -> list[RetrievedChunk]:
    if not query_text or not query_text.strip():
        return []
    if not indexed_chunks:
        return []

    query_vector = embed_query(query_text, model, api_key)
    ranked_hits = [
        RetrievedChunk(chunk=indexed.chunk, similarity=cosine_similarity(query_vector, indexed.vector))
        for indexed in indexed_chunks
    ]
    ranked_hits.sort(key=lambda hit: hit.similarity, reverse=True)
    return ranked_hits[: max(top_k, 1)]
