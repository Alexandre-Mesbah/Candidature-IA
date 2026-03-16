from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    document_id: str
    title: str
    text: str


@dataclass(frozen=True)
class ChunkSlice:
    content: str
    start_index: int
    end_index: int
    chunk_index: int


@dataclass(frozen=True)
class Chunk:
    document_id: str
    document_title: str
    content: str
    start_index: int
    end_index: int
    chunk_index: int


@dataclass(frozen=True)
class IndexedChunk:
    chunk: Chunk
    vector: list[float]
    model: str


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    similarity: float
