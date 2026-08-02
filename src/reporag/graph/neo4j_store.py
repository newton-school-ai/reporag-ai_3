"""Neo4j graph store with Cypher query layer.

Persists the code knowledge graph (call graph + dependency graph + symbol
table) in Neo4j. Provides Cypher query helpers for neighbors, shortest
path, and subgraph extraction. Includes NetworkX fallback for testing.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

try:
    from neo4j import GraphDatabase
    from neo4j.exceptions import ServiceUnavailable, SessionExpired
except ImportError:
    GraphDatabase = None

try:
    import networkx as nx
except ImportError:
    nx = None

from src.reporag.graph.call_graph import CallEdge
from src.reporag.graph.dependency_graph import DependencyEdge
from src.reporag.graph.symbol_table import SymbolTable

logger = logging.getLogger(__name__)


class GraphStore(ABC):
    @abstractmethod
    def clear(self):
        """Clear the entire graph."""
        pass

    @abstractmethod
    def persist_graph(
        self,
        call_edges: list[CallEdge],
        dep_edges: list[DependencyEdge],
        symbol_table: SymbolTable,
    ):
        """Persist the symbols, call edges, and dependency edges to the graph."""
        pass

    @abstractmethod
    def query(self, query_str: str, **kwargs) -> Any:
        """Run a generic query."""
        pass

    @abstractmethod
    def get_neighbors(self, node_id: str, depth: int = 1) -> list[str]:
        """Get neighbors of a node up to a certain depth."""
        pass

    @abstractmethod
    def shortest_path(self, source_id: str, target_id: str) -> list[str]:
        """Find the shortest path between two nodes."""
        pass

    @abstractmethod
    def subgraph(self, node_ids: list[str]) -> Any:
        """Extract a subgraph for a set of nodes."""
        pass


class NetworkXGraphStore(GraphStore):
    def __init__(self):
        if nx is None:
            raise ImportError("networkx is not installed")
        self.graph = nx.DiGraph()

    def clear(self):
        self.graph.clear()

    def persist_graph(
        self,
        call_edges: list[CallEdge],
        dep_edges: list[DependencyEdge],
        symbol_table: SymbolTable,
    ):
        # 1. Add Nodes
        for qname, record in symbol_table._records.items():
            sym = record.symbol
            label = sym.type.capitalize()
            self.graph.add_node(
                qname,
                label=label,
                name=sym.name,
                file_path=sym.file_path,
                start_line=sym.start_line,
                signature=sym.signature,
            )

        # 2. Add Call Edges
        for ce in call_edges:
            self.graph.add_edge(ce.caller, ce.callee, type="CALLS", line=ce.line)

        # 3. Add Dependency Edges
        for de in dep_edges:
            self.graph.add_edge(
                de.source_module,
                de.target_module,
                type="IMPORTS",
                import_type=de.import_type,
            )

        # 4. Infer CONTAINS edges and INHERITS edges
        for qname, record in symbol_table._records.items():
            sym = record.symbol
            if sym.type == "method" and sym.parent_class:
                parts = qname.split(".")
                class_qname = ".".join(parts[:-1])
                if class_qname in self.graph:
                    self.graph.add_edge(class_qname, qname, type="CONTAINS")

            if sym.type == "class" and sym.bases:
                for base in sym.bases:
                    parts = qname.split(".")
                    module_qname = ".".join(parts[:-1])
                    base_qname = f"{module_qname}.{base}"
                    if base_qname in self.graph:
                        self.graph.add_edge(qname, base_qname, type="INHERITS")

    def query(self, query_str: str, **kwargs) -> Any:
        raise NotImplementedError(
            "NetworkXGraphStore does not support Cypher queries directly."
        )

    def get_neighbors(self, node_id: str, depth: int = 1) -> list[str]:
        if node_id not in self.graph:
            return []

        visited = set()
        queue = [(node_id, 0)]

        while queue:
            current, d = queue.pop(0)
            if current not in visited:
                visited.add(current)
                if d < depth:
                    for neighbor in self.graph.neighbors(current):
                        queue.append((neighbor, d + 1))

        visited.remove(node_id)
        return list(visited)

    def shortest_path(self, source_id: str, target_id: str) -> list[str]:
        if source_id not in self.graph or target_id not in self.graph:
            return []
        try:
            return nx.shortest_path(self.graph, source=source_id, target=target_id)
        except nx.NetworkXNoPath:
            return []
        except nx.NodeNotFound:
            return []

    def subgraph(self, node_ids: list[str]) -> Any:
        return self.graph.subgraph(node_ids)


class Neo4jGraphStore(GraphStore):
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
    ):
        if GraphDatabase is None:
            raise ImportError("neo4j driver is not installed")

        self.uri = uri
        self.user = user
        self.password = password
        self.driver = self._connect_with_retry()

    def _connect_with_retry(self, retries=5, delay=2):
        for attempt in range(retries):
            try:
                driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
                driver.verify_connectivity()
                return driver
            except (ServiceUnavailable, SessionExpired) as e:
                if attempt == retries - 1:
                    logger.error("Could not connect to Neo4j after multiple retries.")
                    raise e
                time.sleep(delay)

    def close(self):
        if self.driver:
            self.driver.close()

    def clear(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def persist_graph(
        self,
        call_edges: list[CallEdge],
        dep_edges: list[DependencyEdge],
        symbol_table: SymbolTable,
    ):
        nodes_batch = []
        for qname, record in symbol_table._records.items():
            sym = record.symbol
            nodes_batch.append(
                {
                    "id": qname,
                    "label": sym.type.capitalize(),
                    "name": sym.name,
                    "file_path": sym.file_path,
                    "start_line": sym.start_line,
                    "signature": sym.signature,
                }
            )

        calls_batch = []
        for ce in call_edges:
            calls_batch.append(
                {"source": ce.caller, "target": ce.callee, "line": ce.line}
            )

        imports_batch = []
        for de in dep_edges:
            imports_batch.append(
                {
                    "source": de.source_module,
                    "target": de.target_module,
                    "type": de.import_type,
                }
            )

        contains_batch = []
        inherits_batch = []

        for qname, record in symbol_table._records.items():
            sym = record.symbol
            if sym.type == "method" and sym.parent_class:
                parts = qname.split(".")
                class_qname = ".".join(parts[:-1])
                contains_batch.append({"source": class_qname, "target": qname})

            if sym.type == "class" and sym.bases:
                for base in sym.bases:
                    parts = qname.split(".")
                    module_qname = ".".join(parts[:-1])
                    base_qname = f"{module_qname}.{base}"
                    inherits_batch.append({"source": qname, "target": base_qname})

        with self.driver.session() as session:
            label_groups = {}
            for n in nodes_batch:
                label_groups.setdefault(n["label"], []).append(n)

            for label, batch in label_groups.items():
                query = f"""
                UNWIND $batch AS row
                MERGE (n:{label} {{id: row.id}})
                SET n.name = row.name,
                    n.file_path = row.file_path,
                    n.start_line = row.start_line,
                    n.signature = row.signature
                """
                session.run(query, batch=batch)

            if calls_batch:
                session.run(
                    """
                UNWIND $batch AS row
                MATCH (s {id: row.source})
                MATCH (t {id: row.target})
                MERGE (s)-[r:CALLS]->(t)
                SET r.line = row.line
                """,
                    batch=calls_batch,
                )

            if imports_batch:
                session.run(
                    """
                UNWIND $batch AS row
                MERGE (s:Module {id: row.source})
                MERGE (t:Module {id: row.target})
                MERGE (s)-[r:IMPORTS]->(t)
                SET r.type = row.type
                """,
                    batch=imports_batch,
                )

            if contains_batch:
                session.run(
                    """
                UNWIND $batch AS row
                MATCH (s {id: row.source})
                MATCH (t {id: row.target})
                MERGE (s)-[r:CONTAINS]->(t)
                """,
                    batch=contains_batch,
                )

            if inherits_batch:
                session.run(
                    """
                UNWIND $batch AS row
                MATCH (s {id: row.source})
                MATCH (t {id: row.target})
                MERGE (s)-[r:INHERITS]->(t)
                """,
                    batch=inherits_batch,
                )

    def query(self, query_str: str, **kwargs) -> Any:
        with self.driver.session() as session:
            result = session.run(query_str, **kwargs)
            return [record.data() for record in result]

    def get_neighbors(self, node_id: str, depth: int = 1) -> list[str]:
        query = (
            """
        MATCH (start {id: $node_id})-[*1.."""
            + str(depth)
            + """]->(neighbor)
        RETURN DISTINCT neighbor.id AS id
        """
        )
        results = self.query(query, node_id=node_id)
        return [r["id"] for r in results]

    def shortest_path(self, source_id: str, target_id: str) -> list[str]:
        query = """
        MATCH p=shortestPath((s {id: $source_id})-[:CALLS|IMPORTS|CONTAINS|INHERITS*]-(t {id: $target_id}))
        RETURN [n IN nodes(p) | n.id] AS path
        """
        results = self.query(query, source_id=source_id, target_id=target_id)
        if results and results[0]["path"]:
            return results[0]["path"]
        return []

    def subgraph(self, node_ids: list[str]) -> Any:
        query = """
        MATCH (n)-[r]->(m)
        WHERE n.id IN $node_ids AND m.id IN $node_ids
        RETURN n, r, m
        """
        return self.query(query, node_ids=node_ids)
