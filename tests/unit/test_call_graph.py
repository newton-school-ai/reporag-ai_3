import pytest

from src.reporag.graph.call_graph import CallGraphBuilder
from src.reporag.ingestion.parser import ASTParser
from src.reporag.ingestion.symbol_extractor import SymbolExtractor


@pytest.fixture
def parser():
    return ASTParser()


@pytest.fixture
def extractor(parser):
    return SymbolExtractor(parser)


@pytest.fixture
def builder():
    return CallGraphBuilder()


def test_direct_call(parser, extractor, builder):
    source = """
def b():
    pass

def a():
    b()
"""
    ast = parser.parse(source, language="python").root_node
    symbols = extractor.extract_from_source(source, file_path="test.py")

    edges = builder.build_from_symbols(symbols, {"test.py": ast})

    # a() calls b()
    assert len(edges) == 1
    edge = edges[0]
    assert edge.caller == "test.py:a"
    assert edge.callee == "test.py:b"
    assert edge.call_site_line == 6


def test_method_call(parser, extractor, builder):
    source = """
class MyClass:
    def method_b(self):
        pass

    def method_a(self):
        self.method_b()

def outside():
    obj = MyClass()
    obj.method_a()
"""
    ast = parser.parse(source, language="python").root_node
    symbols = extractor.extract_from_source(source, file_path="test.py")

    edges = builder.build_from_symbols(symbols, {"test.py": ast})

    # self.method_b() and MyClass() and obj.method_a()
    assert len(edges) == 3

    # The call to method_b from method_a
    method_b_call = next(e for e in edges if e.callee == "test.py:MyClass.method_b")
    assert method_b_call.caller == "test.py:MyClass.method_a"

    # The call to method_a from outside
    method_a_call = next(e for e in edges if e.callee == "*.method_a")
    assert method_a_call.caller == "test.py:outside"

    # The instantiation of MyClass from outside
    instantiation_call = next(e for e in edges if e.callee == "test.py:MyClass")
    assert instantiation_call.caller == "test.py:outside"


def test_cross_file_call(parser, extractor, builder):
    source_a = """
def helper():
    pass
"""
    source_b = """
from file_a import helper

def main():
    helper()
"""
    ast_a = parser.parse(source_a, language="python").root_node
    ast_b = parser.parse(source_b, language="python").root_node

    symbols_a = extractor.extract_from_source(source_a, file_path="file_a.py")
    symbols_b = extractor.extract_from_source(source_b, file_path="file_b.py")

    edges = builder.build_from_symbols(
        symbols_a + symbols_b, {"file_a.py": ast_a, "file_b.py": ast_b}
    )

    # file_b.py:main calls file_a.py:helper
    edge = next(e for e in edges if e.callee == "file_a.py:helper")
    assert edge.caller == "file_b.py:main"


def test_recursive_call(parser, extractor, builder):
    source = """
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)
"""
    ast = parser.parse(source, language="python").root_node
    symbols = extractor.extract_from_source(source, file_path="test.py")

    edges = builder.build_from_symbols(symbols, {"test.py": ast})

    assert len(edges) == 1
    edge = edges[0]
    assert edge.caller == "test.py:factorial"
    assert edge.callee == "test.py:factorial"
