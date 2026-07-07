import logging
from collections import defaultdict
from dataclasses import dataclass

from tree_sitter import Node

from src.reporag.ingestion.symbol_extractor import Symbol

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CallEdge:
    caller: str
    callee: str
    line: int


class CallGraphBuilder:

    def __init__(self):
        self.symbols = defaultdict(dict)
        self.imports = defaultdict(dict)

    def build(self, symbols: list[Symbol], asts: dict[str, Node]) -> list[CallEdge]:
        self._build_symbol_index(symbols)

        edges = []
        for file_path, root in asts.items():
            edges.extend(self._walk_file(root, file_path))

        return edges

    def _build_symbol_index(self, symbols: list[Symbol]):
        for sym in symbols:

            if sym.type == "import":
                self._register_import(sym)
            else:
                self.symbols[sym.file_path][sym.name] = sym

    def _register_import(self, sym: Symbol):
        text = sym.signature.strip()

        if text.startswith("from "):
            module, names = text[5:].split(" import ")

            for item in names.split(","):
                item = item.strip()

                if " as " in item:
                    original, alias = item.split(" as ")
                else:
                    original = alias = item

                self.imports[sym.file_path][
                    alias.strip()
                ] = f"{module.replace('.', '/')}.py:{original.strip()}"

        elif text.startswith("import "):
            for item in text[7:].split(","):
                item = item.strip()

                if " as " in item:
                    original, alias = item.split(" as ")
                else:
                    original = alias = item

                self.imports[sym.file_path][alias.strip()] = f"{original.strip()}:*"

    def _walk_file(self, root: Node, file_path: str) -> list[CallEdge]:

        edges = []

        stack = [(root, "", "")]

        while stack:

            node, cls, scope = stack.pop()

            next_cls = cls
            next_scope = scope

            if node.type == "class_definition":
                name = self._identifier(node)
                next_cls = name
                next_scope = name

            elif node.type == "function_definition":
                fn = self._identifier(node)
                next_scope = f"{cls}.{fn}" if cls else fn

            elif node.type == "call":
                edge = self._extract_call(node, file_path, next_scope, next_cls)
                if edge:
                    edges.append(edge)

            for child in reversed(node.children):
                stack.append((child, next_cls, next_scope))

        return edges

    def _extract_call(
        self,
        node: Node,
        file_path: str,
        scope: str,
        cls: str,
    ) -> CallEdge | None:

        if not node.children:
            return None

        func = node.children[0]

        callee = None

        if func.type == "identifier":
            callee = self._resolve(func.text.decode(), file_path)

        elif func.type == "attribute":

            obj = func.child_by_field_name("object")
            attr = func.child_by_field_name("attribute")

            if obj and attr:

                obj_name = obj.text.decode()
                method = attr.text.decode()

                if obj_name in {"self", "cls", "super"} and cls:
                    callee = f"{file_path}:{cls}.{method}"
                else:
                    callee = f"*.{method}"

        if callee is None:
            return None

        caller = f"{file_path}:{scope or '<module>'}"

        return CallEdge(
            caller=caller,
            callee=callee,
            line=node.start_point.row + 1,
        )

    def _resolve(self, name: str, file_path: str) -> str:

        if name in self.imports[file_path]:
            return self.imports[file_path][name]

        if name in self.symbols[file_path]:
            return f"{file_path}:{name}"

        return f"{name}:*"

    @staticmethod
    def _identifier(node: Node) -> str:

        ident = node.child_by_field_name("name")

        if ident:
            return ident.text.decode()

        for child in node.children:
            if child.type == "identifier":
                return child.text.decode()

        return ""
