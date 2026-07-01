import logging
from dataclasses import dataclass, field

from tree_sitter import Node

from src.reporag.ingestion.parser import ASTParser

logger = logging.getLogger(__name__)


@dataclass
class Symbol:
    """Represents a semantic symbol extracted from source code."""

    name: str
    type: str
    file_path: str
    start_line: int
    end_line: int

    signature: str = ""
    docstring: str = ""
    decorators: list[str] = field(default_factory=list)

    parent_class: str = ""
    return_type_hint: str = ""
    bases: list[str] = field(default_factory=list)


class SymbolExtractor:
    """Extract semantic symbols from a Tree-sitter AST."""

    def __init__(self, parser: ASTParser | None = None):
        self.parser = parser or ASTParser()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def extract_from_file(
        self,
        file_path: str,
        language: str | None = None,
    ) -> list[Symbol]:
        try:
            result = self.parser.parse_file(file_path, language)
            return self.extract_from_node(result.root_node, file_path)
        except Exception:
            logger.exception("Failed extracting symbols from %s", file_path)
            return []

    def extract_from_source(
        self,
        source: str,
        file_path: str = "<memory>",
        language: str = "python",
    ) -> list[Symbol]:
        result = self.parser.parse(source, language)
        return self.extract_from_node(result.root_node, file_path)

    def extract_from_node(
        self,
        root: Node,
        file_path: str,
    ) -> list[Symbol]:
        symbols: list[Symbol] = []
        self._walk(root, file_path, symbols, parent_class=None)
        return symbols

    # ------------------------------------------------------------------ #
    # Traversal
    # ------------------------------------------------------------------ #

    def _walk(
        self,
        node: Node,
        file_path: str,
        symbols: list[Symbol],
        parent_class: str | None,
    ):
        if node.type == "decorated_definition":
            self._handle_decorated(
                node,
                file_path,
                symbols,
                parent_class,
            )
            return

        if node.type in (
            "function_definition",
            "async_function_definition",
            "class_definition",
        ):
            self._extract_definition(
                node,
                file_path,
                symbols,
                parent_class,
                decorators=[],
            )
            return

        if node.type in (
            "import_statement",
            "import_from_statement",
        ):
            self._extract_import(node, file_path, symbols)

        for child in node.children:
            self._walk(child, file_path, symbols, parent_class)

    # ------------------------------------------------------------------ #
    # Decorated definitions
    # ------------------------------------------------------------------ #

    def _handle_decorated(
        self,
        node: Node,
        file_path: str,
        symbols: list[Symbol],
        parent_class: str | None,
    ):
        decorators = []

        target = None

        for child in node.children:
            if child.type == "decorator":
                decorators.append(self._text(child))

            elif child.type in (
                "function_definition",
                "async_function_definition",
                "class_definition",
            ):
                target = child

        if target:
            self._extract_definition(
                target,
                file_path,
                symbols,
                parent_class,
                decorators,
                override_node=node,
            )

    # ------------------------------------------------------------------ #
    # Definition extraction
    # ------------------------------------------------------------------ #

    def _extract_definition(
        self,
        node: Node,
        file_path: str,
        symbols: list[Symbol],
        parent_class: str | None,
        decorators: list[str],
        override_node: Node | None = None,
    ):
        name_node = node.child_by_field_name("name") or self._find(node, "identifier")

        if name_node is None:
            return

        name = self._text(name_node)

        is_class = node.type == "class_definition"

        symbol_type = "class" if is_class else "method" if parent_class else "function"

        signature = ""
        return_type = ""
        bases = []

        if is_class:
            superclasses = node.child_by_field_name("superclasses")

            if superclasses:
                signature = self._text(superclasses)

                bases = [
                    self._text(child)
                    for child in superclasses.children
                    if child.type == "identifier"
                ]

        else:
            params = node.child_by_field_name("parameters")

            if params:
                signature = self._text(params)

            returns = node.child_by_field_name("return_type")

            if returns:
                return_type = self._text(returns)

        docstring = self._extract_docstring(node)

        position = override_node or node

        symbols.append(
            Symbol(
                name=name,
                type=symbol_type,
                file_path=file_path,
                start_line=position.start_point.row + 1,
                end_line=position.end_point.row + 1,
                signature=signature,
                docstring=docstring,
                decorators=decorators,
                parent_class=parent_class or "",
                return_type_hint=return_type,
                bases=bases,
            )
        )

        block = self._find(node, "block")

        if block:
            next_parent = name if is_class else parent_class

            for child in block.children:
                self._walk(
                    child,
                    file_path,
                    symbols,
                    next_parent,
                )

    # ------------------------------------------------------------------ #
    # Imports
    # ------------------------------------------------------------------ #

    def _extract_import(
        self,
        node: Node,
        file_path: str,
        symbols: list[Symbol],
    ):
        text = self._text(node)

        symbols.append(
            Symbol(
                name=text,
                type="import",
                file_path=file_path,
                start_line=node.start_point.row + 1,
                end_line=node.end_point.row + 1,
                signature=text,
            )
        )

    # ------------------------------------------------------------------ #
    # Docstrings
    # ------------------------------------------------------------------ #

    def _extract_docstring(self, node: Node) -> str:
        block = self._find(node, "block")

        if not block or not block.children:
            return ""

        stmt = block.children[0]

        if stmt.type != "expression_statement":
            return ""

        if not stmt.children:
            return ""

        expr = stmt.children[0]

        if expr.type == "string":
            return self._text(expr)

        return ""

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _text(node: Node) -> str:
        return node.text.decode("utf-8") if node.text else ""

    @staticmethod
    def _find(node: Node, node_type: str) -> Node | None:
        for child in node.children:
            if child.type == node_type:
                return child
        return None
