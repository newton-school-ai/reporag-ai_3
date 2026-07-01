import pytest

from src.reporag.ingestion.symbol_extractor import SymbolExtractor


@pytest.fixture
def extractor():
    return SymbolExtractor()


def test_extract_function(extractor):
    source = '''
@decorator
async def my_func(a: int) -> bool:
    """My docstring."""
    return True
'''
    symbols = extractor.extract_from_source(source)

    assert len(symbols) == 1
    sym = symbols[0]
    assert sym.name == "my_func"
    assert sym.type == "function"
    assert sym.signature == "(a: int)"
    assert sym.docstring == '"""My docstring."""'
    assert "@decorator" in sym.decorators
    assert sym.return_type_hint == "bool"
    assert sym.parent_class == ""
    assert sym.start_line == 2
    assert sym.end_line == 5


def test_extract_class_with_methods(extractor):
    source = '''
class MyClass(Base):
    """Class doc"""

    @property
    def my_method(self):
        return 42
'''
    symbols = extractor.extract_from_source(source)

    # Class and method
    assert len(symbols) == 2

    cls_sym = [s for s in symbols if s.type == "class"][0]
    assert cls_sym.name == "MyClass"
    assert "Base" in cls_sym.bases
    assert cls_sym.docstring == '"""Class doc"""'
    assert cls_sym.signature == "(Base)"

    method_sym = [s for s in symbols if s.type == "method"][0]
    assert method_sym.name == "my_method"
    assert method_sym.parent_class == "MyClass"
    assert "@property" in method_sym.decorators
    assert method_sym.signature == "(self)"


def test_extract_imports(extractor):
    source = """
import os
from sys import argv, path
from math import *
"""
    symbols = extractor.extract_from_source(source)
    assert len(symbols) == 3

    import_types = [s for s in symbols if s.type == "import"]
    assert len(import_types) == 3
    signatures = [s.signature for s in import_types]
    assert "import os" in signatures
    assert "from sys import argv, path" in signatures
    assert "from math import *" in signatures


def test_extract_nested_functions(extractor):
    source = """
def outer():
    def inner():
        pass
    return inner
"""
    symbols = extractor.extract_from_source(source)
    assert len(symbols) == 2
    names = [s.name for s in symbols]
    assert "outer" in names
    assert "inner" in names

    inner_sym = [s for s in symbols if s.name == "inner"][0]
    # For now, it's still treated as type "function", but with no parent_class
    assert inner_sym.type == "function"
    assert inner_sym.parent_class == ""


def test_invalid_syntax_returns_partial(extractor):
    source = """
def valid():
    pass

def invalid(
"""
    # We shouldn't crash, we should just parse what we can
    symbols = extractor.extract_from_source(source)
    assert len(symbols) >= 1
    assert any(s.name == "valid" for s in symbols)
