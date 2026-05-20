"""Tests for the chunking logic in scripts/ingest.py."""

import sys
from pathlib import Path

# Ensure scripts/ is on the path
_SCRIPTS = Path(__file__).parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from ingest import chunk_text, CHUNK_SIZE, CHUNK_OVERLAP


def test_chunk_text_basic():
    # Text shorter than CHUNK_SIZE - CHUNK_OVERLAP → exactly 1 chunk
    text = "x" * (CHUNK_SIZE - CHUNK_OVERLAP - 1)
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_overlap():
    text = "x" * (CHUNK_SIZE + 100)
    chunks = chunk_text(text)
    assert len(chunks) == 2
    # Second chunk starts at CHUNK_SIZE - CHUNK_OVERLAP
    expected_start = CHUNK_SIZE - CHUNK_OVERLAP
    assert chunks[1] == text[expected_start : expected_start + CHUNK_SIZE]


def test_chunk_text_empty():
    assert chunk_text("") == []


def test_chunk_text_whitespace_only():
    assert chunk_text("   \n\t  ") == []


def test_chunk_text_short():
    text = "short text"
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text
