"""Sentence- and paragraph-aware chunking with overlap and source metadata."""
from __future__ import annotations
import re
from typing import Any


def split_sentences(text: str) -> list[str]:
    """Lightweight sentence splitter that requires no downloaded language model."""
    if not text.strip():
        return []
    sentences: list[str] = []
    # Newlines are retained by the chunker as paragraph boundaries.
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", line)
        sentences.extend(part.strip() for part in parts if part.strip())
    return sentences


def semantic_chunk_segments(
    segments: list[dict[str, Any]], max_words: int = 110, overlap_words: int = 20
) -> list[dict[str, Any]]:
    """Create structure-aware chunks that respect source and PDF page boundaries.

    Adjacent paragraphs are grouped only while they fit within the size limit.
    Complete sentences are retained and a small sentence-level overlap carries
    context between chunks. No external embedding model or API is required.
    """
    if max_words < 30:
        raise ValueError("max_words should be at least 30.")
    if overlap_words < 0 or overlap_words >= max_words:
        raise ValueError("overlap_words must be >= 0 and smaller than max_words.")

    grouped: dict[tuple[str, int | None], list[dict[str, Any]]] = {}
    for segment in segments:
        grouped.setdefault((segment["source"], segment.get("page")), []).append(segment)

    chunks: list[dict[str, Any]] = []
    for (source, page), page_segments in grouped.items():
        sentence_items: list[tuple[str, int]] = []
        for segment in page_segments:
            for sentence in split_sentences(segment["text"]):
                words = sentence.split()
                if len(words) <= max_words:
                    sentence_items.append((sentence, int(segment["paragraph"])))
                else:
                    for start in range(0, len(words), max_words):
                        sentence_items.append((
                            " ".join(words[start:start + max_words]),
                            int(segment["paragraph"]),
                        ))

        buffer: list[tuple[str, int]] = []
        buffer_words = 0
        chunk_number = 0

        def emit(items: list[tuple[str, int]]) -> None:
            nonlocal chunk_number
            if not items:
                return
            chunk_number += 1
            paragraph_ids = sorted({paragraph for _, paragraph in items})
            paragraph_text: list[str] = []
            current_paragraph = None
            current_sentences: list[str] = []
            for sentence, paragraph_id in items:
                if current_paragraph is not None and paragraph_id != current_paragraph:
                    paragraph_text.append(" ".join(current_sentences))
                    current_sentences = []
                current_paragraph = paragraph_id
                current_sentences.append(sentence)
            if current_sentences:
                paragraph_text.append(" ".join(current_sentences))
            chunks.append({
                "source": source,
                "page": page,
                "paragraphs": paragraph_ids,
                "chunk_id": chunk_number,
                "text": "\n".join(paragraph_text),
            })

        for sentence, paragraph_id in sentence_items:
            count = len(sentence.split())
            if buffer and buffer_words + count > max_words:
                emit(buffer)
                overlap: list[tuple[str, int]] = []
                overlap_count = 0
                for item in reversed(buffer):
                    item_count = len(item[0].split())
                    if overlap_count + item_count > overlap_words:
                        break
                    overlap.insert(0, item)
                    overlap_count += item_count
                buffer, buffer_words = overlap, overlap_count
            buffer.append((sentence, paragraph_id))
            buffer_words += count
        emit(buffer)
    return chunks
