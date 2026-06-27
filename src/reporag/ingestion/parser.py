"""Tree-sitter AST parser with language-agnostic interface.

Provides fast, incremental, error-tolerant parsing for Python and JavaScript/TypeScript.
Unlike Python's built-in ast module, tree-sitter preserves comments and whitespace positions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from tree_sitter import Language, Node, Parser, Tree

logger = logging.getLogger(__name__)

# Lazy-loaded language registry: maps language name -> Language object
_LANGUAGE_REGISTRY: dict[str, Language] = {}


def _get_language(language: str) -> Language:
    """Load and cache a tree-sitter Language object by name.

    Raises:
        ValueError: If the language is not supported.
    """
    if language in _LANGUAGE_REGISTRY:
        return _LANGUAGE_REGISTRY[language]

    try:
        if language == "python":
            import tree_sitter_python as tspython

            lang = Language(tspython.language())
        elif language in ("javascript", "typescript"):
            import tree_sitter_javascript as tsjs

            lang = Language(tsjs.language())
        else:
            raise ValueError(
                f"Unsupported language: {language!r}. "
                f"Supported: python, javascript, typescript"
            )
    except ImportError as exc:
        raise ValueError(
            f"tree-sitter grammar for {language!r} is not installed. "
            f"pip install tree-sitter-{language}"
        ) from exc

    _LANGUAGE_REGISTRY[language] = lang
    return lang


@dataclass(frozen=True)
class NodeData:
    """Structured representation of a single AST node."""

    type: str
    text: str
    start_line: int
    end_line: int
    start_col: int
    end_col: int
    is_error: bool = False
    children: list[NodeData] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"NodeData(type={self.type!r}, "
            f"lines={self.start_line}-{self.end_line}, "
            f"error={self.is_error})"
        )


@dataclass
class ParseResult:
    """Result of parsing a source file."""

    tree: Tree
    root_node: Node
    language: str
    has_errors: bool
    node_data: NodeData

    def walk(self, max_depth: int = -1) -> list[NodeData]:
        """Flatten the AST into a list of NodeData, optionally limited by depth.

        Args:
            max_depth: Maximum depth to traverse. -1 means unlimited.

        Returns:
            A flat list of all NodeData nodes in pre-order traversal.
        """
        result: list[NodeData] = []
        self._walk_recursive(self.node_data, result, 0, max_depth)
        return result

    @staticmethod
    def _walk_recursive(
        node: NodeData,
        result: list[NodeData],
        current_depth: int,
        max_depth: int,
    ) -> None:
        result.append(node)
        if max_depth != -1 and current_depth >= max_depth:
            return
        for child in node.children:
            ParseResult._walk_recursive(child, result, current_depth + 1, max_depth)


class ASTParser:
    """Language-agnostic AST parser backed by tree-sitter.

    Usage::

        parser = ASTParser()
        result = parser.parse("def hello():\\n    return 42\\n", language="python")
        print(result.root_node.children)

        # Or use the structured NodeData tree:
        for node in result.walk(max_depth=2):
            print(node.type, node.start_line, node.end_line)
    """

    def parse(
        self,
        source: str,
        language: str = "python",
        *,
        encoding: str = "utf-8",
    ) -> ParseResult:
        """Parse source code into a tree-sitter AST.

        Args:
            source: The source code string to parse.
            language: The language of the source code (python, javascript, typescript).
            encoding: The encoding of the source code.

        Returns:
            A ParseResult containing the tree, root node, and structured node data.

        Raises:
            ValueError: If the language is not supported.
        """
        lang = _get_language(language)
        parser = Parser(lang)

        source_bytes = source.encode(encoding)
        tree = parser.parse(source_bytes)

        root = tree.root_node
        has_errors = root.has_error

        if has_errors:
            logger.warning(
                "Source contains syntax errors; returning partial AST "
                "(language=%s, root_type=%s)",
                language,
                root.type,
            )

        node_data = self._build_node_data(root)

        return ParseResult(
            tree=tree,
            root_node=root,
            language=language,
            has_errors=has_errors,
            node_data=node_data,
        )

    def parse_file(
        self,
        file_path: str,
        language: str | None = None,
        *,
        encoding: str = "utf-8",
    ) -> ParseResult:
        """Parse a source file from disk.

        Args:
            file_path: Path to the source file.
            language: The language. If None, inferred from file extension.
            encoding: The file encoding.

        Returns:
            A ParseResult.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the language cannot be inferred or is unsupported.
        """
        if language is None:
            language = self._infer_language(file_path)

        with open(file_path, encoding=encoding) as f:
            source = f.read()

        return self.parse(source, language=language, encoding=encoding)

    @staticmethod
    def _infer_language(file_path: str) -> str:
        """Infer the language from a file extension."""
        ext_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "javascript",
            ".tsx": "typescript",
        }
        import os

        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        if ext not in ext_map:
            raise ValueError(
                f"Cannot infer language from extension {ext!r}. "
                f"Supported: {', '.join(ext_map.keys())}"
            )
        return ext_map[ext]

    def _build_node_data(self, node: Node) -> NodeData:
        """Recursively convert a tree-sitter Node into a NodeData tree."""
        children = [self._build_node_data(child) for child in node.children]

        text = ""
        try:
            text = node.text.decode("utf-8") if node.text else ""
        except (UnicodeDecodeError, AttributeError):
            text = repr(node.text)

        return NodeData(
            type=node.type,
            text=text,
            start_line=node.start_point.row + 1,
            end_line=node.end_point.row + 1,
            start_col=node.start_point.column,
            end_col=node.end_point.column,
            is_error=node.type == "ERROR" or node.has_error,
            children=children,
        )
