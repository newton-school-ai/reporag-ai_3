import os

import pytest

from src.reporag.graph.symbol_table import SymbolTable
from src.reporag.ingestion.symbol_extractor import Symbol


@pytest.fixture
def symbols():
    return [
        Symbol(
            name="login",
            type="function",
            file_path="src/auth.py",
            start_line=10,
            end_line=20,
        ),
        Symbol(
            name="login",
            type="method",
            file_path="src/api.py",
            start_line=30,
            end_line=40,
            parent_class="AuthAPI",
        ),
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
            start_line=15,
            end_line=20,
        ),
    ]


def test_register(symbols):
    table = SymbolTable()
    table.register_symbols(symbols)

    assert len(table.lookup("login")) == 2
    assert len(table.lookup("helper")) == 2


def test_lookup_qualified(symbols):
    table = SymbolTable()
    table.register_symbols(symbols)

    assert table.lookup_qualified("src.auth.login") is not None
    assert table.lookup_qualified("src.api.AuthAPI.login") is not None
    assert table.lookup_qualified("src.utils.helper") is not None
    assert table.lookup_qualified("src.utils.helper_L15") is not None


def test_regex(symbols):
    table = SymbolTable()
    table.register_symbols(symbols)

    matches = table.lookup_regex(r".*AuthAPI.*")

    assert len(matches) == 1
    assert matches[0].symbol.parent_class == "AuthAPI"


def test_lookup_file(symbols):
    table = SymbolTable()
    table.register_symbols(symbols)

    records = table.lookup_by_file("src/utils.py")

    assert len(records) == 2


def test_json_roundtrip(symbols, tmp_path):
    table = SymbolTable()
    table.register_symbols(symbols)

    file = tmp_path / "symbols.json"

    table.to_json(file)

    assert os.path.exists(file)

    new_table = SymbolTable()
    new_table.from_json(file)

    assert len(new_table.lookup("login")) == 2
    assert new_table.lookup_qualified("src.auth.login") is not None
