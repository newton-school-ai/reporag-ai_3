import pytest

from src.reporag.graph.call_graph import CallEdge
from src.reporag.graph.dependency_graph import DependencyEdge
from src.reporag.graph.neo4j_store import NetworkXGraphStore
from src.reporag.graph.symbol_table import SymbolTable
from src.reporag.ingestion.symbol_extractor import Symbol


@pytest.fixture
def mock_data():
    symbols = [
        Symbol(
            name="func_a",
            type="function",
            file_path="src/a.py",
            start_line=10,
            end_line=20,
        ),
        Symbol(
            name="func_b",
            type="function",
            file_path="src/b.py",
            start_line=5,
            end_line=15,
        ),
        Symbol(
            name="BaseClass",
            type="class",
            file_path="src/c.py",
            start_line=1,
            end_line=5,
        ),
        Symbol(
            name="MyClass",
            type="class",
            file_path="src/c.py",
            start_line=10,
            end_line=50,
            bases=["BaseClass"],
        ),
        Symbol(
            name="my_method",
            type="method",
            file_path="src/c.py",
            start_line=10,
            end_line=20,
            parent_class="MyClass",
        ),
    ]

    table = SymbolTable()
    table.register_symbols(symbols)

    call_edges = [
        CallEdge(caller="src.a.func_a", callee="src.b.func_b", line=15),
        CallEdge(caller="src.b.func_b", callee="src.c.MyClass.my_method", line=10),
    ]

    dep_edges = [
        DependencyEdge(
            source_module="src.a",
            target_module="src.b",
            import_type="import",
            imported_names=["b"],
        )
    ]

    return table, call_edges, dep_edges


@pytest.fixture
def nx_store(mock_data):
    table, call_edges, dep_edges = mock_data
    store = NetworkXGraphStore()
    store.persist_graph(call_edges, dep_edges, table)
    return store


def test_persist_nodes(nx_store):
    assert "src.a.func_a" in nx_store.graph
    node = nx_store.graph.nodes["src.a.func_a"]
    assert node["label"] == "Function"
    assert node["file_path"] == "src/a.py"

    assert "src.c.MyClass" in nx_store.graph
    assert nx_store.graph.nodes["src.c.MyClass"]["label"] == "Class"


def test_persist_edges(nx_store):
    # CALLS edge
    assert nx_store.graph.has_edge("src.a.func_a", "src.b.func_b")
    assert nx_store.graph.edges["src.a.func_a", "src.b.func_b"]["type"] == "CALLS"

    # IMPORTS edge
    assert nx_store.graph.has_edge("src.a", "src.b")
    assert nx_store.graph.edges["src.a", "src.b"]["type"] == "IMPORTS"

    # CONTAINS edge (Class -> Method)
    assert nx_store.graph.has_edge("src.c.MyClass", "src.c.MyClass.my_method")
    assert (
        nx_store.graph.edges["src.c.MyClass", "src.c.MyClass.my_method"]["type"]
        == "CONTAINS"
    )

    # INHERITS edge
    assert nx_store.graph.has_edge("src.c.MyClass", "src.c.BaseClass")
    assert (
        nx_store.graph.edges["src.c.MyClass", "src.c.BaseClass"]["type"] == "INHERITS"
    )


def test_get_neighbors(nx_store):
    neighbors = nx_store.get_neighbors("src.a.func_a", depth=1)
    assert "src.b.func_b" in neighbors
    assert "src.c.MyClass.my_method" not in neighbors

    neighbors_d2 = nx_store.get_neighbors("src.a.func_a", depth=2)
    assert "src.b.func_b" in neighbors_d2
    assert "src.c.MyClass.my_method" in neighbors_d2


def test_shortest_path(nx_store):
    path = nx_store.shortest_path("src.a.func_a", "src.c.MyClass.my_method")
    assert path == ["src.a.func_a", "src.b.func_b", "src.c.MyClass.my_method"]

    no_path = nx_store.shortest_path("src.c.MyClass.my_method", "src.a.func_a")
    assert no_path == []
