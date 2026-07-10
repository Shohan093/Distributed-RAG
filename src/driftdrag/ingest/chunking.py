"""Fixed-window text chunking (Stage 0). Simple by design: the paper's
contribution is drift compensation/scheduling, not chunking strategy, so a
naive fixed-window splitter keeps that variable controlled across baselines.
"""

from __future__ import annotations


def chunk_text(text: str, window_chars: int = 800, overlap_chars: int = 100) -> list[str]:
    if not text:
        return []
    if window_chars <= overlap_chars:
        raise ValueError("window_chars must exceed overlap_chars")
    step = window_chars - overlap_chars
    chunks = []
    for start in range(0, len(text), step):
        chunk = text[start:start + window_chars]
        if chunk.strip():
            chunks.append(chunk)
        if start + window_chars >= len(text):
            break
    return chunks
