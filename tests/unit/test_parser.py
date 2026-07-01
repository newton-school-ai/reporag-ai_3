"""Unit tests for tree-sitter AST parser (Issue 6)."""

import pytest

from src.reporag.ingestion.parser import ASTParser, NodeData, ParseResult


@pytest.fixture
def parser():
    """Return a fresh ASTParser instance."""
    return ASTParser()


# ---------------------------------------------------------------------------
# Test: empty file
# ---------------------------------------------------------------------------
class TestParseEmptyFile:
    def test_empty_string(self, parser):
        result = parser.parse("", language="python")
        assert isinstance(result, ParseResult)
        assert result.language == "python"
        assert result.has_errors is False
        assert result.node_data.type == "module"
        assert result.node_data.children == []

    def test_whitespace_only(self, parser):
        result = parser.parse("   \n\n  \n", language="python")
        assert result.has_errors is False
        assert result.node_data.type == "module"


# ---------------------------------------------------------------------------
# Test: single function
# ---------------------------------------------------------------------------
class TestParseSingleFunction:
    SOURCE = "def hello():\n    return 42\n"

    def test_root_is_module(self, parser):
        result = parser.parse(self.SOURCE, language="python")
        assert result.node_data.type == "module"
        assert result.has_errors is False

    def test_function_node_present(self, parser):
        result = parser.parse(self.SOURCE, language="python")
        func_nodes = [
            n for n in result.node_data.children if n.type == "function_definition"
        ]
        assert len(func_nodes) == 1

    def test_node_data_has_line_info(self, parser):
        result = parser.parse(self.SOURCE, language="python")
        func = result.node_data.children[0]
        assert func.start_line == 1
        assert func.end_line == 2
        assert func.start_col == 0

    def test_node_data_has_text(self, parser):
        result = parser.parse(self.SOURCE, language="python")
        func = result.node_data.children[0]
        assert "def hello" in func.text
        assert "return 42" in func.text


# ---------------------------------------------------------------------------
# Test: class with methods
# ---------------------------------------------------------------------------
class TestParseClassWithMethods:
    SOURCE = (
        "class Calculator:\n"
        "    def add(self, a, b):\n"
        "        return a + b\n"
        "\n"
        "    def subtract(self, a, b):\n"
        "        return a - b\n"
    )

    def test_class_is_parsed(self, parser):
        result = parser.parse(self.SOURCE, language="python")
        class_nodes = [
            n for n in result.node_data.children if n.type == "class_definition"
        ]
        assert len(class_nodes) == 1

    def test_methods_inside_class(self, parser):
        result = parser.parse(self.SOURCE, language="python")
        class_node = [
            n for n in result.node_data.children if n.type == "class_definition"
        ][0]
        # The class body contains a 'block' node with function_definitions
        all_nodes = []
        self._collect(class_node, all_nodes)
        method_nodes = [n for n in all_nodes if n.type == "function_definition"]
        assert len(method_nodes) == 2

    @staticmethod
    def _collect(node, result):
        result.append(node)
        for child in node.children:
            TestParseClassWithMethods._collect(child, result)


# ---------------------------------------------------------------------------
# Test: async function
# ---------------------------------------------------------------------------
class TestParseAsyncFunction:
    SOURCE = "async def fetch_data(url):\n    return await get(url)\n"

    def test_async_function_parsed(self, parser):
        result = parser.parse(self.SOURCE, language="python")
        assert result.has_errors is False
        # tree-sitter has "function_definition" for async def too
        all_nodes = result.walk()
        func_types = [n.type for n in all_nodes if "function" in n.type]
        assert len(func_types) >= 1


# ---------------------------------------------------------------------------
# Test: syntax error returns partial AST (does NOT crash)
# ---------------------------------------------------------------------------
class TestParseSyntaxError:
    SOURCE = "def broken(\n    return 42\n"

    def test_has_errors_flag(self, parser):
        result = parser.parse(self.SOURCE, language="python")
        assert result.has_errors is True

    def test_returns_parse_result(self, parser):
        result = parser.parse(self.SOURCE, language="python")
        assert isinstance(result, ParseResult)
        assert result.node_data is not None
        assert result.node_data.type == "module"

    def test_partial_ast_has_nodes(self, parser):
        result = parser.parse(self.SOURCE, language="python")
        all_nodes = result.walk()
        assert len(all_nodes) > 0

    def test_error_node_is_flagged(self, parser):
        result = parser.parse(self.SOURCE, language="python")
        all_nodes = result.walk()
        error_nodes = [n for n in all_nodes if n.is_error]
        assert len(error_nodes) > 0


# ---------------------------------------------------------------------------
# Test: language-agnostic interface (JavaScript)
# ---------------------------------------------------------------------------
class TestLanguageAgnostic:
    JS_SOURCE = "function greet(name) { return 'hello ' + name; }\n"

    def test_parse_javascript(self, parser):
        result = parser.parse(self.JS_SOURCE, language="javascript")
        assert result.language == "javascript"
        assert result.has_errors is False
        assert result.node_data.type == "program"

    def test_js_function_found(self, parser):
        result = parser.parse(self.JS_SOURCE, language="javascript")
        all_nodes = result.walk()
        func_nodes = [n for n in all_nodes if n.type == "function_declaration"]
        assert len(func_nodes) == 1

    def test_unsupported_language_raises(self, parser):
        with pytest.raises(ValueError, match="Unsupported language"):
            parser.parse("code", language="rust")


# ---------------------------------------------------------------------------
# Test: walk() and NodeData
# ---------------------------------------------------------------------------
class TestWalkAndNodeData:
    SOURCE = "x = 1\ny = 2\n"

    def test_walk_returns_list(self, parser):
        result = parser.parse(self.SOURCE, language="python")
        nodes = result.walk()
        assert isinstance(nodes, list)
        assert len(nodes) > 0

    def test_walk_max_depth(self, parser):
        result = parser.parse(self.SOURCE, language="python")
        depth0 = result.walk(max_depth=0)
        depth1 = result.walk(max_depth=1)
        # Depth 0 should only have the root
        assert len(depth0) == 1
        assert len(depth1) > len(depth0)

    def test_node_data_fields(self, parser):
        result = parser.parse("x = 1\n", language="python")
        node = result.node_data
        assert isinstance(node, NodeData)
        assert isinstance(node.type, str)
        assert isinstance(node.text, str)
        assert isinstance(node.start_line, int)
        assert isinstance(node.end_line, int)
        assert isinstance(node.start_col, int)
        assert isinstance(node.end_col, int)
        assert isinstance(node.is_error, bool)


# ---------------------------------------------------------------------------
# Test: nested classes
# ---------------------------------------------------------------------------
class TestNestedClasses:
    SOURCE = (
        "class Outer:\n"
        "    class Inner:\n"
        "        def method(self):\n"
        "            pass\n"
    )

    def test_nested_class_parsed(self, parser):
        result = parser.parse(self.SOURCE, language="python")
        assert result.has_errors is False
        all_nodes = result.walk()
        class_nodes = [n for n in all_nodes if n.type == "class_definition"]
        assert len(class_nodes) == 2

    def test_nested_method_found(self, parser):
        result = parser.parse(self.SOURCE, language="python")
        all_nodes = result.walk()
        method_nodes = [n for n in all_nodes if n.type == "function_definition"]
        assert len(method_nodes) == 1
