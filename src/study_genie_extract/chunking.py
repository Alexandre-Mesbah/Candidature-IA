from __future__ import annotations

from .types import ChunkSlice


def chunk_text(
    text: str,
    max_chunk_size: int = 1000,
    overlap: int = 200,
    separator: str = "\n\n",
) -> list[ChunkSlice]:
    chunks: list[ChunkSlice] = []
    if not text or not text.strip():
        return chunks

    parts = text.split(separator)
    current_chunk = ""
    current_start = 0
    chunk_index = 0

    for part in parts:
        trimmed_part = part.strip()
        if not trimmed_part:
            continue

        if len(trimmed_part) > max_chunk_size:
            if current_chunk:
                chunks.append(
                    ChunkSlice(
                        content=current_chunk.strip(),
                        start_index=current_start,
                        end_index=current_start + len(current_chunk),
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1
                current_chunk = ""

            long_parts = split_long_text(trimmed_part, max_chunk_size=max_chunk_size, overlap=overlap)
            for long_part in long_parts:
                chunks.append(
                    ChunkSlice(
                        content=long_part.content,
                        start_index=current_start + long_part.start_index,
                        end_index=current_start + long_part.end_index,
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1

            current_start += len(trimmed_part) + len(separator)
            continue

        candidate = f"{current_chunk}{separator}{trimmed_part}" if current_chunk else trimmed_part
        if len(candidate) <= max_chunk_size:
            current_chunk = candidate
            continue

        if current_chunk:
            chunks.append(
                ChunkSlice(
                    content=current_chunk.strip(),
                    start_index=current_start,
                    end_index=current_start + len(current_chunk),
                    chunk_index=chunk_index,
                )
            )
            chunk_index += 1

            if overlap > 0 and len(current_chunk) > overlap:
                overlap_text = current_chunk[-overlap:]
                current_chunk = f"{overlap_text}{separator}{trimmed_part}"
                current_start = (
                    current_start
                    + len(current_chunk)
                    - len(overlap_text)
                    - len(separator)
                    - len(trimmed_part)
                )
            else:
                current_chunk = trimmed_part
                current_start = current_start + len(current_chunk) + len(separator)
        else:
            current_chunk = trimmed_part

    if current_chunk:
        chunks.append(
            ChunkSlice(
                content=current_chunk.strip(),
                start_index=current_start,
                end_index=current_start + len(current_chunk),
                chunk_index=chunk_index,
            )
        )

    return chunks


def split_long_text(text: str, max_chunk_size: int, overlap: int) -> list[ChunkSlice]:
    chunks: list[ChunkSlice] = []
    start = 0

    while start < len(text):
        end = min(start + max_chunk_size, len(text))
        content = text[start:end].strip()
        if content:
            chunks.append(
                ChunkSlice(
                    content=content,
                    start_index=start,
                    end_index=end,
                    chunk_index=0,
                )
            )

        start = end - overlap
        if start <= 0:
            start = end

    return chunks
