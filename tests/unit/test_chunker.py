import pytest

from src.reporag.ingestion.chunker import SemanticChunker


@pytest.fixture
def chunker():
    # Use a very small max_tokens so we can easily trigger splits in tests
    return SemanticChunker(max_tokens=20)


def test_small_function_one_chunk(chunker):
    source = """
def add(a, b):
    return a + b
"""
    chunks = chunker.chunk_source(source)
    assert len(chunks) == 1
    c = chunks[0]
    assert "def add(a, b):" in c.text
    assert c.start_line > 0
    assert c.end_line > 0
    assert c.language == "python"


def test_large_function_splits_with_signature(chunker):
    source = """
def large_function(x):
    a = x + 1
    b = a + 2
    c = b + 3
    d = c + 4
    e = d + 5
    return e
"""
    # 20 tokens will force this function to split
    chunks = chunker.chunk_source(source)
    assert len(chunks) > 1

    for c in chunks:
        # Every chunk of this function should include the signature
        assert "def large_function(x):" in c.text
        assert c.parent_symbol == "large_function"
        assert c.token_count > 0


def test_module_level_code(chunker):
    source = """
import os
import sys

x = 1000000
y = 2000000
z = x + y + 1000000
w = z * 2000000

def dummy():
    print("hello world this is a long string to increase tokens")
    pass
"""
    chunks = chunker.chunk_source(source)
    assert len(chunks) >= 2

    # Check that module level code is chunked
    module_text = "".join(c.text for c in chunks)
    assert "import os" in module_text
    assert "z = x + y" in module_text
    assert "def dummy():" in module_text


def test_chunk_metadata_is_correct(chunker):
    source = """
class TestClass:
    def method(self):
        pass
"""
    chunks = chunker.chunk_source(source, file_path="test.py", language="python")

    for c in chunks:
        assert c.file_path == "test.py"
        assert c.language == "python"
        assert c.start_line <= c.end_line
        assert c.token_count == chunker.count_tokens(c.text)
