import pytest

from src.reporag.graph.dependency_graph import DependencyGraphBuilder
from src.reporag.ingestion.symbol_extractor import Symbol


@pytest.fixture
def graph():
    return DependencyGraphBuilder()


def make_symbol(signature, path):
    return Symbol(
        name=signature,
        type="import",
        file_path=path,
        start_line=1,
        end_line=1,
        signature=signature,
    )


def test_import_and_from_import(graph):
    files = {
        "src/app.py": [
            make_symbol("import os", "src/app.py"),
            make_symbol("import sys as s", "src/app.py"),
            make_symbol("from src.config import settings", "src/app.py"),
        ]
    }

    edges = graph.build(files)

    assert len(edges) == 3

    edge_map = {(e.target_module, e.import_type): e for e in edges}

    assert ("os", "import") in edge_map
    assert ("sys", "import") in edge_map

    config_edge = edge_map[("src.config", "from_import")]
    assert config_edge.source_module == "src.app"
    assert config_edge.imported_names == ["settings"]


def test_relative_import_resolution(graph):
    files = {
        "src/pkg/module.py": [
            make_symbol("from . import helper", "src/pkg/module.py"),
            make_symbol("from ..core import base", "src/pkg/module.py"),
        ]
    }

    edges = graph.build(files)

    helper = next(e for e in edges if "helper" in e.imported_names)
    base = next(e for e in edges if "base" in e.imported_names)

    assert helper.target_module == "src.pkg"
    assert helper.source_module == "src.pkg.module"

    assert base.target_module == "src.core"
    assert base.source_module == "src.pkg.module"


def test_package_relative_import(graph):
    files = {
        "src/pkg/__init__.py": [
            make_symbol("from . import api", "src/pkg/__init__.py"),
        ]
    }

    edges = graph.build(files)

    assert len(edges) == 1

    edge = edges[0]

    assert edge.source_module == "src.pkg"
    assert edge.target_module == "src.pkg"
    assert edge.imported_names == ["api"]


def test_star_import_logs_warning(graph, caplog):
    files = {"src/main.py": [make_symbol("from src.utils import *", "src/main.py")]}

    edges = graph.build(files)

    assert edges[0].import_type == "star_import"
    assert edges[0].imported_names == ["*"]

    messages = [record.message for record in caplog.records]
    assert any("Star import detected" in msg for msg in messages)


def test_detects_simple_cycle(graph, caplog):
    files = {
        "src/x.py": [make_symbol("from src.y import b", "src/x.py")],
        "src/y.py": [make_symbol("from src.x import a", "src/y.py")],
    }

    graph.build(files)

    assert len(graph.cycles) == 1

    cycle = graph.cycles[0]

    assert cycle[0] == cycle[-1]
    assert {"src.x", "src.y"} <= set(cycle)

    messages = [record.message for record in caplog.records]
    assert any("Circular import detected" in msg for msg in messages)
