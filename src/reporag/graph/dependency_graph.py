import logging
from dataclasses import dataclass

from src.reporag.ingestion.symbol_extractor import Symbol

logger = logging.getLogger(__name__)


@dataclass
class DependencyEdge:
    source_module: str
    target_module: str
    import_type: str  # "import", "from_import", "star_import"
    imported_names: list[str]


class DependencyGraphBuilder:
    """Builds an import dependency graph from extracted symbols."""

    def __init__(self):
        self.edges: list[DependencyEdge] = []
        self.cycles: list[list[str]] = []
        self.modules: set[str] = set()

    def build(self, symbols_by_file: dict[str, list[Symbol]]) -> list[DependencyEdge]:
        self.edges = []
        self.cycles = []
        self.modules = set()

        for file_path, symbols in symbols_by_file.items():
            source_module, is_package = self._file_to_module(file_path)
            self.modules.add(source_module)
            for sym in symbols:
                if sym.type == "import":
                    self._parse_and_add_edge(source_module, is_package, sym)

        self._detect_cycles()
        return self.edges

    def _parse_and_add_edge(self, source_module: str, is_package: bool, sym: Symbol):
        text = sym.signature.strip()
        if text.startswith("import "):
            # Handle "import a, b as c"
            text = text[7:]
            parts = text.split(",")
            for p in parts:
                name = p.split(" as ")[0].strip()
                target_module = name
                self.edges.append(
                    DependencyEdge(
                        source_module=source_module,
                        target_module=target_module,
                        import_type="import",
                        imported_names=[name],
                    )
                )
        elif text.startswith("from "):
            # Handle "from a import b" or "from .a import b"
            if " import " not in text:
                return

            module_part, names_part = text[5:].split(" import ")
            module_part = module_part.strip()
            names_part = names_part.strip()

            if module_part.startswith("."):
                dots = 0
                for char in module_part:
                    if char == ".":
                        dots += 1
                    else:
                        break
                relative_path = module_part[dots:]
                target_module = self._resolve_relative(
                    source_module, is_package, dots, relative_path
                )
            else:
                target_module = module_part

            names = [n.split(" as ")[0].strip() for n in names_part.split(",")]
            import_type = "star_import" if "*" in names else "from_import"

            if import_type == "star_import":
                logger.warning(
                    f"Star import detected in {source_module} from {target_module}"
                )

            self.edges.append(
                DependencyEdge(
                    source_module=source_module,
                    target_module=target_module,
                    import_type=import_type,
                    imported_names=names,
                )
            )

    def _file_to_module(self, file_path: str) -> tuple[str, bool]:
        clean_path = file_path
        is_package = False
        if clean_path.endswith("/__init__.py"):
            clean_path = clean_path[:-12]
            is_package = True
        elif clean_path.endswith(".py"):
            clean_path = clean_path[:-3]

        return clean_path.replace("/", "."), is_package

    def _resolve_relative(
        self, source_module: str, is_package: bool, dots: int, relative_path: str
    ) -> str:
        parts = source_module.split(".")

        levels_to_drop = dots if not is_package else dots - 1

        if levels_to_drop > len(parts):
            return relative_path

        base = parts[:-levels_to_drop] if levels_to_drop > 0 else parts[:]
        if relative_path:
            base.append(relative_path)
        return ".".join(base)

    def _detect_cycles(self):
        graph = {}
        for edge in self.edges:
            if edge.source_module not in graph:
                graph[edge.source_module] = []
            if edge.target_module not in graph[edge.source_module]:
                graph[edge.source_module].append(edge.target_module)

        visited = set()
        stack = set()
        path = []

        def dfs(node):
            if node in stack:
                idx = path.index(node)
                cycle = path[idx:] + [node]
                self.cycles.append(cycle)
                logger.warning(f"Circular import detected: {' -> '.join(cycle)}")
                return

            if node in visited:
                return

            visited.add(node)
            stack.add(node)
            path.append(node)

            for neighbor in graph.get(node, []):
                dfs(neighbor)

            stack.remove(node)
            path.pop()

        for node in list(graph.keys()):
            if node not in visited:
                dfs(node)
