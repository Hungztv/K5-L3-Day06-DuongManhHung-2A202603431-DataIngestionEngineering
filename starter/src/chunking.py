"""Implement fixed-size and structure-aware chunking."""
import re

from .models import Chunk, DocumentPage


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    """Split a page into overlapping character windows."""
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be non-negative and smaller than size")
    if not page.text:
        return []

    step = size - overlap
    chunks: list[Chunk] = []
    for start in range(0, len(page.text), step):
        text = page.text[start : start + size].strip()
        if text:
            chunks.append(_make_chunk(page, len(chunks), text, "fixed"))
        if start + size >= len(page.text):
            break
    return chunks


def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    """Group paragraphs up to ``max_size`` and split oversized paragraphs."""
    if max_size <= 0:
        raise ValueError("max_size must be positive")
    if not page.text:
        return []

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", page.text) if part.strip()]
    chunks: list[Chunk] = []
    current: list[str] = []
    current_length = 0

    for paragraph in paragraphs:
        if len(paragraph) > max_size:
            if current:
                chunks.append(_make_chunk(page, len(chunks), "\n\n".join(current), "paragraph"))
                current = []
                current_length = 0
            for start in range(0, len(paragraph), max_size):
                text = paragraph[start : start + max_size].strip()
                if text:
                    chunks.append(_make_chunk(page, len(chunks), text, "paragraph"))
            continue

        proposed_length = current_length + (2 if current else 0) + len(paragraph)
        if current and proposed_length > max_size:
            chunks.append(_make_chunk(page, len(chunks), "\n\n".join(current), "paragraph"))
            current = []
            current_length = 0
        current.append(paragraph)
        current_length += (2 if current_length else 0) + len(paragraph)

    if current:
        chunks.append(_make_chunk(page, len(chunks), "\n\n".join(current), "paragraph"))
    return chunks


def _make_chunk(page: DocumentPage, index: int, text: str, strategy: str) -> Chunk:
    metadata = dict(page.metadata)
    metadata["chunk_strategy"] = strategy
    metadata["char_count"] = len(text)
    return Chunk(
        chunk_id=f"{page.document_id}-p{page.page:04d}-{strategy}-{index:04d}",
        document_id=page.document_id,
        source=page.source,
        page=page.page,
        text=text,
        metadata=metadata,
    )
