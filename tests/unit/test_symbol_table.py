import os

import pytest

from src.reporag.graph.symbol_table import SymbolTable
from src.reporag.ingestion.symbol_extractor import Symbol


@pytest.fixture
def sample_symbols():
    return [
        Symbol(
            name="authenticate",
            type="function",
            file_path="src/auth.py",
            start_line=10,
            end_line=20,
        ),
        Symbol(
            name="authenticate",
            type="method",
            file_path="src/api.py",
            start_line=30,
            end_line=40,
            parent_class="AuthRouter",
        ),
        Symbol(
            name="test_auth",
            type="function",
            file_path="tests/test_auth.py",
            start_line=5,
            end_line=15,
        ),
        # Test collision fallback
        Symbol(
            name="helper",
            type="function",
            file_path="src/utils.py",
            start_line=1,
            end_line=5,
        ),
        Symbol(
            name="helper",
            type="function",
            file_path="src/utils.py",
            start_line=10,
            end_line=15,
        ),
    ]


def test_register_and_lookup_exact(sample_symbols):
    table = SymbolTable()
    table.register_symbols(sample_symbols)

    # Exact name lookup should return multiple
    results = table.lookup("authenticate")
    assert len(results) == 2
    assert {r.symbol.file_path for r in results} == {"src/auth.py", "src/api.py"}


def test_lookup_qualified(sample_symbols):
    table = SymbolTable()
    table.register_symbols(sample_symbols)

    # Qualified name lookup
    record = table.lookup_qualified("src.auth.authenticate")
    assert record is not None
    assert record.symbol.file_path == "src/auth.py"

    record_method = table.lookup_qualified("src.api.AuthRouter.authenticate")
    assert record_method is not None
    assert record_method.symbol.type == "method"

    # Test collision resolution
    record_helper_1 = table.lookup_qualified("src.utils.helper")
    assert record_helper_1 is not None
    assert record_helper_1.symbol.start_line == 1

    record_helper_2 = table.lookup_qualified("src.utils.helper_L10")
    assert record_helper_2 is not None
    assert record_helper_2.symbol.start_line == 10


def test_lookup_regex(sample_symbols):
    table = SymbolTable()
    table.register_symbols(sample_symbols)

    results = table.lookup_regex(r"test_.*")
    assert len(results) == 1
    assert results[0].qualified_name == "tests.test_auth.test_auth"


def test_lookup_by_file(sample_symbols):
    table = SymbolTable()
    table.register_symbols(sample_symbols)

    results = table.lookup_by_file("src/utils.py")
    assert len(results) == 2


def test_serialization(sample_symbols, tmp_path):
    table = SymbolTable()
    table.register_symbols(sample_symbols)

    file_path = tmp_path / "symbols.json"
    table.to_json(str(file_path))

    assert os.path.exists(str(file_path))

    new_table = SymbolTable()
    new_table.from_json(str(file_path))

    # Verify everything loaded correctly
    results = new_table.lookup("authenticate")
    assert len(results) == 2

    record = new_table.lookup_qualified("src.auth.authenticate")
    assert record is not None
    assert record.symbol.name == "authenticate"
    assert record.symbol.file_path == "src/auth.py"
