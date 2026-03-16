"""Adapted public extract of a Study Genie retrieval component."""

from .embeddings import DEFAULT_EMBEDDING_MODEL, create_embedding, create_embeddings
from .pipeline import build_index, chunk_documents, load_documents, prepare_documents
from .retrieval import cosine_similarity, retrieve_top_k
from .types import Chunk, Document, IndexedChunk, RetrievedChunk

__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "Chunk",
    "Document",
    "IndexedChunk",
    "RetrievedChunk",
    "build_index",
    "chunk_documents",
    "cosine_similarity",
    "create_embedding",
    "create_embeddings",
    "load_documents",
    "prepare_documents",
    "retrieve_top_k",
]
