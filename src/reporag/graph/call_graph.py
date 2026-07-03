import logging
from dataclasses import dataclass

from tree_sitter import Node

from src.reporag.ingestion.symbol_extractor import Symbol

logger = logging.getLogger(__name__)


@dataclass
class CallEdge:
    caller: str  # e.g. "file_path:ClassName.method" or "file_path:function"
    callee: str  # resolved target, e.g., "file_path:function" or "*.method"
    call_site_line: int


class CallGraphBuilder:
    """Builds a call graph from AST nodes and resolved symbols."""

    def __init__(self):
        # file_path -> list of defined Symbols
        self.symbols_by_file: dict[str, list[Symbol]] = {}
        # file_path -> {alias: target_file_path + ":" + target_name}
        self.import_map: dict[str, dict[str, str]] = {}

    def build_from_symbols(
        self, symbols: list[Symbol], file_asts: dict[str, Node]
    ) -> list[CallEdge]:
        self._index_symbols(symbols)
        edges = []
        for file_path, ast_root in file_asts.items():
            file_edges = self._extract_calls(ast_root, file_path)
            edges.extend(file_edges)
        return edges

    def _index_symbols(self, symbols: list[Symbol]):
        for sym in symbols:
            if sym.file_path not in self.symbols_by_file:
                self.symbols_by_file[sym.file_path] = []
                self.import_map[sym.file_path] = {}

            if sym.type == "import":
                self._parse_and_register_import(sym)
            else:
                self.symbols_by_file[sym.file_path].append(sym)

    def _parse_and_register_import(self, import_sym: Symbol):
        text = import_sym.signature.strip()
        if text.startswith("from "):
            parts = text.split(" import ")
            if len(parts) == 2:
                # Naive conversion from 'a.b.c' -> 'a/b/c.py'
                module_path = parts[0][5:].replace(".", "/") + ".py"
                names = parts[1].split(",")
                for n in names:
                    n = n.strip()
                    alias = n
                    if " as " in n:
                        orig, alias = n.split(" as ")
                        alias = alias.strip()
                        orig = orig.strip()
                    else:
                        orig = n
                    self.import_map[import_sym.file_path][
                        alias
                    ] = f"{module_path}:{orig}"
        elif text.startswith("import "):
            names = text[7:].split(",")
            for n in names:
                n = n.strip()
                alias = n
                if " as " in n:
                    orig, alias = n.split(" as ")
                    alias = alias.strip()
                    orig = orig.strip()
                else:
                    orig = n
                self.import_map[import_sym.file_path][alias] = f"{orig}:*"

    def _extract_calls(self, root_node: Node, file_path: str) -> list[CallEdge]:
        edges = []
        self._traverse(
            root_node, file_path, current_scope="", current_class="", edges=edges
        )
        return edges

    def _traverse(
        self,
        node: Node,
        file_path: str,
        current_scope: str,
        current_class: str,
        edges: list[CallEdge],
    ):
        new_scope = current_scope
        new_class = current_class

        if node.type == "class_definition":
            name = self._get_first_child_text(node, "identifier")
            new_class = name if name else current_class
            new_scope = name if name else current_scope
        elif node.type == "function_definition":
            name = self._get_first_child_text(node, "identifier")
            new_scope = f"{current_class}.{name}" if current_class else name

        if node.type == "call":
            func_expr = node.children[0] if node.children else None

            callee_resolved = ""
            if func_expr and func_expr.type == "identifier":
                func_name = self._get_node_text(func_expr)
                callee_resolved = self._resolve_target(func_name, file_path)
            elif func_expr and func_expr.type == "attribute":
                # Handle obj.method
                obj_node = func_expr.child_by_field_name("object")
                attr_node = func_expr.child_by_field_name("attribute")
                if not obj_node and func_expr.children:
                    obj_node = func_expr.children[0]
                if not attr_node and func_expr.children:
                    attr_node = func_expr.children[-1]

                if obj_node and attr_node:
                    obj_name = self._get_node_text(obj_node)
                    method_name = self._get_node_text(attr_node)

                    if (
                        obj_name == "self"
                        and current_class
                        or obj_name == "cls"
                        and current_class
                        or obj_name == "super"
                        and current_class
                    ):
                        callee_resolved = f"{file_path}:{current_class}.{method_name}"
                    else:
                        callee_resolved = f"*.{method_name}"

            if callee_resolved:
                caller_str = (
                    f"{file_path}:{new_scope}" if new_scope else f"{file_path}:<module>"
                )
                edges.append(
                    CallEdge(
                        caller=caller_str,
                        callee=callee_resolved,
                        call_site_line=node.start_point.row + 1,
                    )
                )

        # Recurse
        for child in node.children:
            self._traverse(child, file_path, new_scope, new_class, edges)

    def _resolve_target(self, name: str, file_path: str) -> str:
        # Check imports
        imports = self.import_map.get(file_path, {})
        if name in imports:
            return imports[name]

        # Check local definitions
        local_symbols = self.symbols_by_file.get(file_path, [])
        for sym in local_symbols:
            if sym.name == name and sym.type in ("function", "class"):
                return f"{file_path}:{name}"

        # Unresolved (built-in or unmapped)
        return f"{name}:*"

    def _get_first_child_text(self, node: Node, child_type: str) -> str:
        for child in node.children:
            if child.type == child_type:
                return self._get_node_text(child)
        return ""

    def _get_node_text(self, node: Node) -> str:
        return node.text.decode("utf-8") if node.text else ""
