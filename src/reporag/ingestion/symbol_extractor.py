import logging
from dataclasses import dataclass, field

from tree_sitter import Node

from src.reporag.ingestion.parser import ASTParser

logger = logging.getLogger(__name__)


@dataclass
class Symbol:
    """Represents a meaningful code entity extracted from the AST."""

    name: str
    type: str  # 'function', 'class', 'method', 'import'
    file_path: str
    start_line: int
    end_line: int
    signature: str = ""
    docstring: str = ""
    decorators: list[str] = field(default_factory=list)
    parent_class: str = ""
    return_type_hint: str = ""
    bases: list[str] = field(default_factory=list)  # For classes


class SymbolExtractor:
    """Extracts semantic symbols (functions, classes, methods, imports) from code AST."""

    def __init__(self, parser: ASTParser | None = None):
        self.parser = parser or ASTParser()

    def extract_from_file(
        self, file_path: str, language: str | None = None
    ) -> list[Symbol]:
        """Parse a file and extract all symbols."""
        try:
            result = self.parser.parse_file(file_path, language=language)
            # Use the raw tree-sitter node since we need to inspect specific child field names
            # which might not be mapped perfectly in NodeData without a lot of boilerplate.
            # But wait, tree-sitter Node allows `node.child_by_field_name` but Python bindings
            # don't always expose it conveniently in old versions, but we can search children.
            return self.extract_from_node(result.root_node, file_path)
        except Exception as e:
            logger.error(f"Failed to extract symbols from {file_path}: {e}")
            return []

    def extract_from_source(
        self, source: str, file_path: str = "<memory>", language: str = "python"
    ) -> list[Symbol]:
        """Parse source code string and extract all symbols."""
        result = self.parser.parse(source, language=language)
        return self.extract_from_node(result.root_node, file_path)

    def extract_from_node(self, root_node: Node, file_path: str) -> list[Symbol]:
        """Traverse the AST node and yield extracted symbols."""
        symbols = []
        self._traverse(root_node, file_path, symbols, parent_class=None)
        return symbols

    def _traverse(
        self,
        node: Node,
        file_path: str,
        symbols: list[Symbol],
        parent_class: str | None,
    ):
        """Recursively traverse the AST and build Symbol objects."""
        # Handle decorators by looking at decorated_definition wrapper
        if node.type == "decorated_definition":
            decorators = []
            target_node = None
            for child in node.children:
                if child.type == "decorator":
                    decorators.append(child.text.decode("utf-8") if child.text else "")
                elif child.type in ("function_definition", "class_definition"):
                    target_node = child

            if target_node:
                self._extract_definition(
                    target_node,
                    file_path,
                    symbols,
                    parent_class,
                    decorators,
                    override_node=node,
                )
            return

        if node.type in ("function_definition", "class_definition"):
            self._extract_definition(
                node, file_path, symbols, parent_class, decorators=[]
            )
            return

        if node.type in ("import_statement", "import_from_statement"):
            self._extract_import(node, file_path, symbols)
            return

        # Continue traversing down
        for child in node.children:
            self._traverse(child, file_path, symbols, parent_class)

    def _extract_definition(
        self,
        node: Node,
        file_path: str,
        symbols: list[Symbol],
        parent_class: str | None,
        decorators: list[str],
        override_node: Node | None = None,
    ):
        name = self._get_first_child_text(node, "identifier")
        if not name:
            return

        is_class = node.type == "class_definition"
        sym_type = "class" if is_class else ("method" if parent_class else "function")

        # Signature / Bases
        signature = ""
        bases = []
        if is_class:
            bases_node = self._get_child_by_type(node, "argument_list")
            if bases_node:
                signature = bases_node.text.decode("utf-8") if bases_node.text else ""
                # Very basic bases extraction
                bases = [
                    c.text.decode("utf-8")
                    for c in bases_node.children
                    if c.type == "identifier"
                ]
        else:
            params_node = self._get_child_by_type(node, "parameters")
            if params_node:
                signature = params_node.text.decode("utf-8") if params_node.text else ""

        # Return type hint
        return_type_hint = ""
        if not is_class:
            return_type_node = self._get_child_by_type(node, "type")
            if return_type_node:
                return_type_hint = (
                    return_type_node.text.decode("utf-8")
                    if return_type_node.text
                    else ""
                )

        # Docstring
        docstring = ""
        block_node = self._get_child_by_type(node, "block")
        if block_node and len(block_node.children) > 0:
            first_stmt = block_node.children[0]
            if (
                first_stmt.type == "expression_statement"
                and len(first_stmt.children) > 0
            ):
                expr = first_stmt.children[0]
                if expr.type == "string":
                    docstring = expr.text.decode("utf-8") if expr.text else ""

        pos_node = override_node if override_node else node

        sym = Symbol(
            name=name,
            type=sym_type,
            file_path=file_path,
            start_line=pos_node.start_point.row + 1,
            end_line=pos_node.end_point.row + 1,
            signature=signature,
            docstring=docstring,
            decorators=decorators,
            parent_class=parent_class if parent_class else "",
            return_type_hint=return_type_hint,
            bases=bases,
        )
        symbols.append(sym)

        # Traverse children to catch nested functions or methods
        # If it's a class, set parent_class so child functions become methods.
        new_parent = name if is_class else parent_class
        if block_node:
            for child in block_node.children:
                self._traverse(child, file_path, symbols, new_parent)

    def _extract_import(self, node: Node, file_path: str, symbols: list[Symbol]):
        text = node.text.decode("utf-8") if node.text else ""
        sym = Symbol(
            name=text.split(" ")[1] if " " in text else text,  # simple name fallback
            type="import",
            file_path=file_path,
            start_line=node.start_point.row + 1,
            end_line=node.end_point.row + 1,
            signature=text,
        )
        symbols.append(sym)

    def _get_first_child_text(self, node: Node, child_type: str) -> str:
        for child in node.children:
            if child.type == child_type:
                return child.text.decode("utf-8") if child.text else ""
        return ""

    def _get_child_by_type(self, node: Node, child_type: str) -> Node | None:
        for child in node.children:
            if child.type == child_type:
                return child
        return None
