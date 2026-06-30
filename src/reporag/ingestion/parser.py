"""Tree-sitter based AST parser."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import tree_sitter_javascript as ts_js
import tree_sitter_python as ts_py
from tree_sitter import Language, Node, Parser, Tree

logger = logging.getLogger(__name__)


@dataclass
class NodeData:
    """Represents a parsed AST node."""

    type: str
    text: str
    start_line: int
    end_line: int
    start_col: int
    end_col: int
    is_error: bool = False
    children: list["NodeData"] = field(default_factory=list)


@dataclass
class ParseResult:
    """Represents the parsed AST."""

    language: str
    tree: Tree
    root_node: Node
    has_errors: bool
    node_data: NodeData

    def walk(self, max_depth: int = -1) -> list[NodeData]:
        """Return every node in preorder.

        Args:
            max_depth:
                -1 = unlimited (default)
                 0 = only root
                 1 = root + immediate children
                 etc.
        """
        nodes: list[NodeData] = []

        def dfs(node: NodeData, depth: int) -> None:
            nodes.append(node)

            if max_depth != -1 and depth >= max_depth:
                return

            for child in node.children:
                dfs(child, depth + 1)

        dfs(self.node_data, 0)
        return nodes


class ASTParser:
    """Language-agnostic tree-sitter parser."""

    def __init__(self) -> None:
        """Initialize parsers for supported languages."""
        self.parsers: dict[str, Parser] = {}

        # Support both old and new tree-sitter APIs
        try:
            # New API (tree-sitter >= 0.25)
            python = Parser()
            python.language = Language(ts_py.language())

            javascript = Parser()
            javascript.language = Language(ts_js.language())
        except (TypeError, AttributeError):
            # Old API
            python = Parser(Language(ts_py.language()))
            javascript = Parser(Language(ts_js.language()))

        self.parsers["python"] = python
        self.parsers["javascript"] = javascript
        self.parsers["typescript"] = javascript

    def parse(
        self,
        source: str,
        language: str = "python",
    ) -> ParseResult:
        """Parse source code."""

        if language not in self.parsers:
            raise ValueError(f"Unsupported language: {language}")

        parser = self.parsers[language]

        tree = parser.parse(source.encode("utf-8"))
        root = tree.root_node

        if root.has_error:
            logger.warning("Syntax errors detected. Returning partial AST.")

        return ParseResult(
            language=language,
            tree=tree,
            root_node=root,
            has_errors=root.has_error,
            node_data=self._convert(root),
        )

    def parse_file(
        self,
        file_path: str,
        language: str | None = None,
    ) -> ParseResult:
        """Parse a file from disk."""

        if language is None:
            language = self._infer_language(file_path)

        with open(file_path, encoding="utf-8") as file:
            source = file.read()

        return self.parse(source, language)

    def _convert(self, node: Node) -> NodeData:
        """Convert a tree-sitter node to NodeData."""

        try:
            text = node.text.decode("utf-8")
        except (UnicodeDecodeError, AttributeError):
            text = ""

        return NodeData(
            type=node.type,
            text=text,
            start_line=node.start_point.row + 1,
            end_line=node.end_point.row + 1,
            start_col=node.start_point.column,
            end_col=node.end_point.column,
            is_error=node.type == "ERROR" or node.has_error,
            children=[self._convert(child) for child in node.children],
        )

    @staticmethod
    def _infer_language(file_path: str) -> str:
        """Infer language from file extension."""

        extension = Path(file_path).suffix.lower()

        mapping = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
        }

        try:
            return mapping[extension]
        except KeyError:
            raise ValueError(f"Unsupported file extension: {extension}") from None
