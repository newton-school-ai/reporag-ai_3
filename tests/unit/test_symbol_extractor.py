from textwrap import dedent

import pytest

from src.reporag.ingestion.symbol_extractor import SymbolExtractor


@pytest.fixture
def extractor():
    return SymbolExtractor()


def test_extract_function(extractor):
    source = dedent(
        """\
        @decorator
        async def my_func(a: int) -> bool:
            \"\"\"My docstring.\"\"\"
            return True
    """
    )

    symbols = extractor.extract_from_source(source)

    assert len(symbols) == 1

    sym = symbols[0]

    assert sym.name == "my_func"
    assert sym.type == "function"
    assert sym.signature == "(a: int)"
    assert sym.return_type_hint == "bool"
    assert sym.docstring == '"""My docstring."""'
    assert "@decorator" in sym.decorators
    assert sym.parent_class == ""
    assert sym.start_line == 1
    assert sym.end_line == 4


def test_extract_sync_function(extractor):
    source = dedent(
        """\
        def add(a, b):
            return a + b
    """
    )

    symbols = extractor.extract_from_source(source)

    assert len(symbols) == 1
    assert symbols[0].name == "add"
    assert symbols[0].type == "function"


def test_extract_class(extractor):
    source = dedent(
        """\
        class MyClass(Base1, Base2):
            \"\"\"Class doc\"\"\"
    """
    )

    symbols = extractor.extract_from_source(source)

    assert len(symbols) == 1

    cls = symbols[0]

    assert cls.type == "class"
    assert cls.name == "MyClass"
    assert cls.signature == "(Base1, Base2)"
    assert cls.docstring == '"""Class doc"""'
    assert "Base1" in cls.bases
    assert "Base2" in cls.bases


def test_extract_class_methods(extractor):
    source = dedent(
        """\
        class Foo:

            @property
            def value(self):
                return 1

            async def load(self):
                pass
    """
    )

    symbols = extractor.extract_from_source(source)

    assert len(symbols) == 3

    methods = [s for s in symbols if s.type == "method"]

    assert len(methods) == 2

    names = {m.name for m in methods}

    assert names == {"value", "load"}

    for m in methods:
        assert m.parent_class == "Foo"


def test_nested_class(extractor):
    source = dedent(
        """\
        class Outer:

            class Inner:

                def hello(self):
                    pass
    """
    )

    symbols = extractor.extract_from_source(source)

    names = {s.name for s in symbols}

    assert {"Outer", "Inner", "hello"} <= names

    hello = next(s for s in symbols if s.name == "hello")

    assert hello.type == "method"
    assert hello.parent_class == "Inner"


def test_nested_functions(extractor):
    source = dedent(
        """\
        def outer():

            def inner():
                pass

            return inner
    """
    )

    symbols = extractor.extract_from_source(source)

    assert len(symbols) == 2

    names = {s.name for s in symbols}

    assert names == {"outer", "inner"}

    inner = next(s for s in symbols if s.name == "inner")

    assert inner.type == "function"
    assert inner.parent_class == ""


def test_imports(extractor):
    source = dedent(
        """\
        import os
        import sys as system
        from pathlib import Path
        from typing import List, Dict
    """
    )

    symbols = extractor.extract_from_source(source)

    imports = [s for s in symbols if s.type == "import"]

    assert len(imports) == 4

    signatures = {s.signature for s in imports}

    assert "import os" in signatures
    assert "import sys as system" in signatures
    assert "from pathlib import Path" in signatures
    assert "from typing import List, Dict" in signatures


def test_multiple_decorators(extractor):
    source = dedent(
        """\
        @cache
        @staticmethod
        def compute():
            pass
    """
    )

    symbols = extractor.extract_from_source(source)

    decorators = symbols[0].decorators

    assert "@cache" in decorators
    assert "@staticmethod" in decorators


def test_missing_docstring(extractor):
    source = dedent(
        """\
        def hello():
            return 1
    """
    )

    symbols = extractor.extract_from_source(source)

    assert symbols[0].docstring == ""


def test_return_annotation(extractor):
    source = dedent(
        """\
        def foo() -> list[str]:
            return []
    """
    )

    symbols = extractor.extract_from_source(source)

    assert len(symbols) == 1
    assert symbols[0].return_type_hint == "list[str]"


def test_empty_source(extractor):
    assert extractor.extract_from_source("") == []


def test_invalid_syntax_returns_partial(extractor):
    source = dedent(
        """\
        def valid():
            pass

        def invalid(
    """
    )

    symbols = extractor.extract_from_source(source)

    assert any(s.name == "valid" for s in symbols)


def test_line_numbers(extractor):
    source = dedent(
        """\
        def foo():
            pass
    """
    )

    symbols = extractor.extract_from_source(source)

    sym = symbols[0]

    assert sym.start_line == 1
    assert sym.end_line == 2


def test_multiple_symbols(extractor):
    source = dedent(
        """\
        import os

        class A:
            pass

        def func():
            pass
    """
    )

    symbols = extractor.extract_from_source(source)

    assert len(symbols) == 3

    assert {s.type for s in symbols} == {
        "import",
        "class",
        "function",
    }
