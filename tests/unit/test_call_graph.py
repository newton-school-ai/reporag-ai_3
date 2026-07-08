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


def test_direct_function_call(parser, extractor, builder):
    source = """
def b():
    pass

def a():
    b()
"""

    ast = parser.parse(source).root_node
    symbols = extractor.extract_from_source(source, "test.py")

    edges = builder.build(symbols, {"test.py": ast})

    assert len(edges) == 1

    edge = edges[0]
    assert edge.caller == "test.py:a"
    assert edge.callee == "test.py:b"
    assert edge.line == 6


def test_recursive_call(parser, extractor, builder):
    source = """
def factorial(n):
    if n == 0:
        return 1
    return factorial(n - 1)
"""

    ast = parser.parse(source).root_node
    symbols = extractor.extract_from_source(source, "test.py")

    edges = builder.build(symbols, {"test.py": ast})

    assert len(edges) == 1

    edge = edges[0]
    assert edge.caller == "test.py:factorial"
    assert edge.callee == "test.py:factorial"


def test_builtin_function_call(parser, extractor, builder):
    source = """
def hello():
    print("hello")
"""

    ast = parser.parse(source).root_node
    symbols = extractor.extract_from_source(source, "test.py")

    edges = builder.build(symbols, {"test.py": ast})

    assert len(edges) == 1
    assert edges[0].caller == "test.py:hello"
    assert edges[0].callee == "print:*"


def test_multiple_calls(parser, extractor, builder):
    source = """
def a():
    pass

def b():
    pass

def c():
    a()
    b()
"""

    ast = parser.parse(source).root_node
    symbols = extractor.extract_from_source(source, "test.py")

    edges = builder.build(symbols, {"test.py": ast})

    assert len(edges) == 2

    callees = {e.callee for e in edges}

    assert callees == {
        "test.py:a",
        "test.py:b",
    }


def test_class_instantiation(parser, extractor, builder):
    source = """
class Person:
    pass

def create():
    Person()
"""

    ast = parser.parse(source).root_node
    symbols = extractor.extract_from_source(source, "test.py")

    edges = builder.build(symbols, {"test.py": ast})

    assert len(edges) == 1

    edge = edges[0]

    assert edge.caller == "test.py:create"
    assert edge.callee == "test.py:Person"


def test_self_method_call(parser, extractor, builder):
    source = """
class A:

    def bar(self):
        pass

    def foo(self):
        self.bar()
"""

    ast = parser.parse(source).root_node
    symbols = extractor.extract_from_source(source, "test.py")

    edges = builder.build(symbols, {"test.py": ast})

    assert len(edges) == 1

    edge = edges[0]

    assert edge.caller == "test.py:A.foo"
    assert edge.callee == "test.py:A.bar"


def test_cls_method_call(parser, extractor, builder):
    source = """
class A:

    @classmethod
    def bar(cls):
        pass

    @classmethod
    def foo(cls):
        cls.bar()
"""

    ast = parser.parse(source).root_node
    symbols = extractor.extract_from_source(source, "test.py")

    edges = builder.build(symbols, {"test.py": ast})

    assert len(edges) == 1

    edge = edges[0]

    assert edge.caller == "test.py:A.foo"
    assert edge.callee == "test.py:A.bar"


def test_object_method_call(parser, extractor, builder):
    source = """
class A:

    def foo(self):
        pass

def main():
    obj = A()
    obj.foo()
"""

    ast = parser.parse(source).root_node
    symbols = extractor.extract_from_source(source, "test.py")

    edges = builder.build(symbols, {"test.py": ast})

    assert len(edges) == 2

    assert any(e.callee == "test.py:A" for e in edges)

    assert any(e.callee == "*.foo" for e in edges)


def test_cross_file_import(parser, extractor, builder):
    source1 = """
def helper():
    pass
"""

    source2 = """
from file1 import helper

def main():
    helper()
"""

    ast1 = parser.parse(source1).root_node
    ast2 = parser.parse(source2).root_node

    symbols = extractor.extract_from_source(
        source1, "file1.py"
    ) + extractor.extract_from_source(source2, "file2.py")

    edges = builder.build(
        symbols,
        {
            "file1.py": ast1,
            "file2.py": ast2,
        },
    )

    assert len(edges) == 1

    edge = edges[0]

    assert edge.caller == "file2.py:main"
    assert edge.callee == "file1.py:helper"


def test_alias_import(parser, extractor, builder):
    source1 = """
def helper():
    pass
"""

    source2 = """
from file1 import helper as h

def main():
    h()
"""

    ast1 = parser.parse(source1).root_node
    ast2 = parser.parse(source2).root_node

    symbols = extractor.extract_from_source(
        source1, "file1.py"
    ) + extractor.extract_from_source(source2, "file2.py")

    edges = builder.build(
        symbols,
        {
            "file1.py": ast1,
            "file2.py": ast2,
        },
    )

    assert len(edges) == 1

    edge = edges[0]

    assert edge.caller == "file2.py:main"
    assert edge.callee == "file1.py:helper"


def test_no_calls(parser, extractor, builder):
    source = """
def foo():
    x = 1
    y = x + 2
    return y
"""

    ast = parser.parse(source).root_node
    symbols = extractor.extract_from_source(source, "test.py")

    edges = builder.build(symbols, {"test.py": ast})

    assert edges == []
