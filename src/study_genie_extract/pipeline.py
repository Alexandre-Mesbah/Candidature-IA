from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path

from .chunking import chunk_text
from .embeddings import DEFAULT_EMBEDDING_MODEL, create_embeddings
from .preprocessing import preprocess_document_text
from .types import Chunk, Document, IndexedChunk


def load_documents(path: str | Path) -> list[Document]:
    input_path = Path(path)
    documents: list[Document] = []

    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            documents.append(
                Document(
                    document_id=str(record["id"]),
                    title=str(record["title"]),
                    text=str(record["text"]),
                )
            )

    return documents


def prepare_documents(documents: Iterable[Document]) -> list[Document]:
    prepared: list[Document] = []

    for document in documents:
        prepared.append(
            Document(
                document_id=document.document_id,
                title=document.title,
                text=preprocess_document_text(document.text),
            )
        )

    return prepared


def chunk_documents(
    documents: Iterable[Document],
    max_chunk_size: int = 1000,
    overlap: int = 200,
    separator: str = "\n\n",
) -> list[Chunk]:
    chunks: list[Chunk] = []

    for document in documents:
        slices = chunk_text(
            document.text,
            max_chunk_size=max_chunk_size,
            overlap=overlap,
            separator=separator,
        )
        for chunk_slice in slices:
            chunks.append(
                Chunk(
                    document_id=document.document_id,
                    document_title=document.title,
                    content=chunk_slice.content,
                    start_index=chunk_slice.start_index,
                    end_index=chunk_slice.end_index,
                    chunk_index=chunk_slice.chunk_index,
                )
            )

    return chunks


def build_index(
    documents: Iterable[Document],
    model: str = DEFAULT_EMBEDDING_MODEL,
    api_key: str | None = None,
    max_chunk_size: int = 1000,
    overlap: int = 200,
    separator: str = "\n\n",
    embed_chunks: Callable[[list[str], str, str | None], list[list[float]]] = create_embeddings,
) -> list[IndexedChunk]:
    prepared_documents = prepare_documents(documents)
    chunks = chunk_documents(
        prepared_documents,
        max_chunk_size=max_chunk_size,
        overlap=overlap,
        separator=separator,
    )
    if not chunks:
        return []

    vectors = embed_chunks([chunk.content for chunk in chunks], model, api_key)
    if len(vectors) != len(chunks):
        raise RuntimeError("number of embeddings does not match the number of chunks")

    return [
        IndexedChunk(chunk=chunk, vector=vector, model=model)
        for chunk, vector in zip(chunks, vectors)
    ]
