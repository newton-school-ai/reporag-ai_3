import pytest

from src.reporag.ingestion.chunker import SemanticChunker


@pytest.fixture
def chunker():
    return SemanticChunker(max_tokens=20)


@pytest.mark.parametrize(
    "source,parent_symbol",
    [
        (
            """
def add(a, b):
    return a + b
""",
            "",
        ),
        (
            """
class Calculator:
    pass
""",
            "",
        ),
    ],
)
def test_small_nodes_create_single_chunk(chunker, source, parent_symbol):
    chunks = chunker.chunk_source(source)

    assert len(chunks) == 1

    chunk = chunks[0]

    assert chunk.text.strip() == source.strip()
    assert chunk.parent_symbol == parent_symbol
    assert chunk.token_count == chunker.count_tokens(chunk.text)
    assert chunk.start_line <= chunk.end_line


def test_large_function_is_split(chunker):
    source = """
def large_function(x):
    a = x + 1
    b = a + 2
    c = b + 3
    d = c + 4
    e = d + 5
    f = e + 6
    g = f + 7
    h = g + 8
    return h
"""

    chunks = chunker.chunk_source(source)

    assert len(chunks) > 1

    for chunk in chunks:
        assert chunk.parent_symbol == "large_function"
        assert chunk.token_count <= chunker.max_tokens * 1.1
        assert chunk.start_line <= chunk.end_line


def test_module_level_code_preserved(chunker):
    source = """
import os
import sys

x = 1
y = 2
z = x + y

def hello():
    print("hello")
"""

    chunks = chunker.chunk_source(source)

    combined = "\n".join(chunk.text for chunk in chunks)

    assert "import os" in combined
    assert "import sys" in combined
    assert "def hello" in combined


def test_chunk_metadata(chunker):
    source = """
class Test:
    def method(self):
        return 42
"""

    chunks = chunker.chunk_source(
        source,
        file_path="sample.py",
        language="python",
    )

    assert chunks

    for chunk in chunks:
        assert chunk.file_path == "sample.py"
        assert chunk.language == "python"
        assert chunk.start_line > 0
        assert chunk.end_line >= chunk.start_line
        assert chunk.token_count == chunker.count_tokens(chunk.text)


def test_empty_source(chunker):
    chunks = chunker.chunk_source("")
    assert chunks == []


def test_whitespace_source(chunker):
    chunks = chunker.chunk_source("\n\n\n")
    assert chunks == []


def test_token_count_matches_text(chunker):
    source = """
def foo():
    return "hello"
"""

    for chunk in chunker.chunk_source(source):
        assert chunk.token_count == chunker.count_tokens(chunk.text)


def test_parent_symbol_only_for_split_functions(chunker):
    source = """
def foo():
    return 1
"""

    chunks = chunker.chunk_source(source)

    assert len(chunks) == 1
    assert chunks[0].parent_symbol == ""
