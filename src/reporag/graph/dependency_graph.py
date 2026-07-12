import logging
from dataclasses import dataclass

from src.reporag.ingestion.symbol_extractor import Symbol

logger = logging.getLogger(__name__)


@dataclass
class DependencyEdge:
    source_module: str
    target_module: str
    import_type: str
    imported_names: list[str]


class DependencyGraphBuilder:
    """Builds a dependency graph from extracted import symbols."""

    def __init__(self):
        self.edges: list[DependencyEdge] = []
        self.cycles: list[list[str]] = []
        self.modules: set[str] = set()

    def build(self, symbols_by_file: dict[str, list[Symbol]]) -> list[DependencyEdge]:
        self.edges.clear()
        self.cycles.clear()
        self.modules.clear()

        for file_path, symbols in symbols_by_file.items():
            source_module, is_package = self._module_from_path(file_path)
            self.modules.add(source_module)

            for symbol in symbols:
                if symbol.type != "import":
                    continue

                statement = symbol.signature.strip()

                if statement.startswith("import "):
                    self._parse_import(source_module, statement)

                elif statement.startswith("from "):
                    self._parse_from_import(
                        source_module,
                        is_package,
                        statement,
                    )

        self._detect_cycles()

        return self.edges

    def _parse_import(self, source_module: str, statement: str):
        modules = statement[7:].split(",")

        for module in modules:
            module = module.strip().split(" as ")[0]

            self.edges.append(
                DependencyEdge(
                    source_module=source_module,
                    target_module=module,
                    import_type="import",
                    imported_names=[module],
                )
            )

    def _parse_from_import(
        self,
        source_module: str,
        is_package: bool,
        statement: str,
    ):
        module_part, names_part = statement[5:].split(" import ", 1)

        module_part = module_part.strip()
        names = [name.strip().split(" as ")[0] for name in names_part.split(",")]

        if module_part.startswith("."):
            target_module = self._resolve_relative(
                source_module,
                is_package,
                module_part,
            )
        else:
            target_module = module_part

        import_type = "star_import" if "*" in names else "from_import"

        if import_type == "star_import":
            logger.warning(
                "Star import detected in %s from %s",
                source_module,
                target_module,
            )

        self.edges.append(
            DependencyEdge(
                source_module=source_module,
                target_module=target_module,
                import_type=import_type,
                imported_names=names,
            )
        )

    def _module_from_path(self, file_path: str) -> tuple[str, bool]:
        is_package = False

        if file_path.endswith("/__init__.py"):
            file_path = file_path[:-12]
            is_package = True
        elif file_path.endswith(".py"):
            file_path = file_path[:-3]

        return file_path.replace("/", "."), is_package

    def _resolve_relative(
        self,
        source_module: str,
        is_package: bool,
        module_part: str,
    ) -> str:
        level = 0

        while level < len(module_part) and module_part[level] == ".":
            level += 1

        relative = module_part[level:]

        parts = source_module.split(".")

        drop = level if not is_package else level - 1

        if drop > len(parts):
            return relative

        base = parts[:-drop] if drop > 0 else parts[:]

        if relative:
            base.append(relative)

        return ".".join(base)

    def _detect_cycles(self):
        graph: dict[str, list[str]] = {}

        for edge in self.edges:
            graph.setdefault(edge.source_module, [])

            if edge.target_module not in graph[edge.source_module]:
                graph[edge.source_module].append(edge.target_module)

        visited = set()
        stack = set()

        def dfs(node: str, path: list[str]):
            visited.add(node)
            stack.add(node)
            path.append(node)

            for neighbour in graph.get(node, []):
                if neighbour not in visited:
                    dfs(neighbour, path)

                elif neighbour in stack:
                    start = path.index(neighbour)
                    cycle = path[start:] + [neighbour]

                    if cycle not in self.cycles:
                        self.cycles.append(cycle)
                        logger.warning(
                            "Circular import detected: %s",
                            " -> ".join(cycle),
                        )

            stack.remove(node)
            path.pop()

        for node in list(graph.keys()):
            if node not in visited:
                dfs(node, [])
