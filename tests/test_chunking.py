from driftdrag.ingest.chunking import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []


def test_short_text_returns_one_chunk():
    chunks = chunk_text("hello world", window_chars=800, overlap_chars=100)
    assert chunks == ["hello world"]


def test_long_text_produces_overlapping_windows():
    text = "x" * 2000
    chunks = chunk_text(text, window_chars=800, overlap_chars=100)
    assert len(chunks) > 1
    assert all(len(c) <= 800 for c in chunks)


def test_rejects_invalid_window_overlap():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("hello", window_chars=50, overlap_chars=50)
