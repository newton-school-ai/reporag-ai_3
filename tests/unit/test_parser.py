"""Tests for the tree-sitter AST parser."""

import pytest

from src.reporag.ingestion.parser import ASTParser, NodeData, ParseResult


@pytest.fixture
def ast_parser() -> ASTParser:
    """Create an AST parser instance."""
    return ASTParser()


def test_parse_empty_source(ast_parser):
    result = ast_parser.parse("", language="python")

    assert isinstance(result, ParseResult)
    assert result.language == "python"
    assert result.node_data.type == "module"
    assert result.has_errors is False
    assert result.node_data.children == []


def test_parse_python_function(ast_parser):
    source = """
def greet():
    return "Hello"
"""

    result = ast_parser.parse(source, language="python")

    assert result.has_errors is False

    functions = [
        node for node in result.node_data.children if node.type == "function_definition"
    ]

    assert len(functions) == 1
    assert "greet" in functions[0].text
    assert functions[0].start_line == 2


def test_parse_python_class(ast_parser):
    source = """
class Person:
    def speak(self):
        return "Hi"
"""

    result = ast_parser.parse(source, language="python")

    classes = [n for n in result.walk() if n.type == "class_definition"]
    methods = [n for n in result.walk() if n.type == "function_definition"]

    assert len(classes) == 1
    assert len(methods) == 1


def test_async_function(ast_parser):
    source = """
async def fetch():
    return await load()
"""

    result = ast_parser.parse(source)

    assert result.has_errors is False

    function_nodes = [node for node in result.walk() if "function" in node.type]

    assert function_nodes


def test_partial_ast_for_invalid_python(ast_parser):
    source = """
def broken(
    return 42
"""

    result = ast_parser.parse(source)

    assert result.has_errors is True
    assert isinstance(result, ParseResult)

    nodes = result.walk()

    assert len(nodes) > 0
    assert any(node.is_error for node in nodes)


def test_nested_class_parsing(ast_parser):
    source = """
class Outer:
    class Inner:
        def hello(self):
            pass
"""

    result = ast_parser.parse(source)

    classes = [n for n in result.walk() if n.type == "class_definition"]
    methods = [n for n in result.walk() if n.type == "function_definition"]

    assert len(classes) == 2
    assert len(methods) == 1


def test_javascript_parsing(ast_parser):
    source = """
function greet(name) {
    return name;
}
"""

    result = ast_parser.parse(source, language="javascript")

    assert result.language == "javascript"
    assert result.has_errors is False

    functions = [n for n in result.walk() if n.type == "function_declaration"]

    assert len(functions) == 1


def test_invalid_language(ast_parser):
    with pytest.raises(ValueError):
        ast_parser.parse("print(1)", language="java")


def test_walk_returns_nodes(ast_parser):
    result = ast_parser.parse("x = 10", language="python")

    nodes = result.walk()

    assert isinstance(nodes, list)
    assert len(nodes) > 0
    assert all(isinstance(node, NodeData) for node in nodes)


def test_node_metadata(ast_parser):
    result = ast_parser.parse("x = 5", language="python")

    node = result.node_data

    assert isinstance(node.type, str)
    assert isinstance(node.text, str)
    assert isinstance(node.start_line, int)
    assert isinstance(node.end_line, int)
    assert isinstance(node.children, list)
