import pytest

from src.reporag.graph.dependency_graph import DependencyGraphBuilder
from src.reporag.ingestion.symbol_extractor import Symbol


@pytest.fixture
def builder():
    return DependencyGraphBuilder()


def test_absolute_imports(builder):
    symbols_by_file = {
        "src/utils.py": [
            Symbol(
                name="import os",
                type="import",
                file_path="src/utils.py",
                start_line=1,
                end_line=1,
                signature="import os",
            ),
            Symbol(
                name="from src.config import settings",
                type="import",
                file_path="src/utils.py",
                start_line=2,
                end_line=2,
                signature="from src.config import settings",
            ),
        ]
    }

    edges = builder.build(symbols_by_file)
    assert len(edges) == 2

    edge_os = edges[0]
    assert edge_os.source_module == "src.utils"
    assert edge_os.target_module == "os"
    assert edge_os.import_type == "import"

    edge_config = edges[1]
    assert edge_config.source_module == "src.utils"
    assert edge_config.target_module == "src.config"
    assert edge_config.import_type == "from_import"
    assert edge_config.imported_names == ["settings"]


def test_relative_imports(builder):
    symbols_by_file = {
        "src/reporag/utils.py": [
            Symbol(
                name="from . import config",
                type="import",
                file_path="src/reporag/utils.py",
                start_line=1,
                end_line=1,
                signature="from . import config",
            ),
            Symbol(
                name="from ..db import models",
                type="import",
                file_path="src/reporag/utils.py",
                start_line=2,
                end_line=2,
                signature="from ..db import models",
            ),
        ],
        "src/reporag/__init__.py": [
            Symbol(
                name="from . import core",
                type="import",
                file_path="src/reporag/__init__.py",
                start_line=1,
                end_line=1,
                signature="from . import core",
            )
        ],
    }

    edges = builder.build(symbols_by_file)

    # Check src.reporag.utils -> src.reporag.config
    edge_1 = next(e for e in edges if e.imported_names == ["config"])
    assert edge_1.source_module == "src.reporag.utils"
    assert edge_1.target_module == "src.reporag"

    # Check src.reporag.utils -> src.db.models
    edge_2 = next(e for e in edges if e.imported_names == ["models"])
    assert edge_2.source_module == "src.reporag.utils"
    assert edge_2.target_module == "src.db"

    # Check src.reporag -> src.reporag.core
    edge_3 = next(e for e in edges if e.imported_names == ["core"])
    assert edge_3.source_module == "src.reporag"
    assert edge_3.target_module == "src.reporag"


def test_star_imports(builder, caplog):
    symbols_by_file = {
        "src/bad.py": [
            Symbol(
                name="from src.utils import *",
                type="import",
                file_path="src/bad.py",
                start_line=1,
                end_line=1,
                signature="from src.utils import *",
            )
        ]
    }
    edges = builder.build(symbols_by_file)
    assert len(edges) == 1
    edge = edges[0]
    assert edge.import_type == "star_import"
    assert edge.imported_names == ["*"]

    # Check if a warning was logged
    assert any("Star import detected" in record.message for record in caplog.records)


def test_circular_imports(builder, caplog):
    symbols_by_file = {
        "src/a.py": [
            Symbol(
                name="from src.b import y",
                type="import",
                file_path="src/a.py",
                start_line=1,
                end_line=1,
                signature="from src.b import y",
            )
        ],
        "src/b.py": [
            Symbol(
                name="from src.a import x",
                type="import",
                file_path="src/b.py",
                start_line=1,
                end_line=1,
                signature="from src.a import x",
            )
        ],
    }

    edges = builder.build(symbols_by_file)
    assert len(edges) == 2

    # Should detect cycle src.a -> src.b -> src.a
    assert len(builder.cycles) == 1
    assert builder.cycles[0] == ["src.a", "src.b", "src.a"] or builder.cycles[0] == [
        "src.b",
        "src.a",
        "src.b",
    ]

    assert any(
        "Circular import detected" in record.message for record in caplog.records
    )
