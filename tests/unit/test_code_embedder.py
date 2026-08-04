import numpy as np
import pytest
import sqlite3
import os

from src.reporag.embedding.code_embedder import CodeEmbedder


@pytest.fixture(scope="module")
def embedder(tmp_path_factory):
    # Using a small fast model for tests to avoid downloading 500MB if possible,
    # but for simplicity we'll just use the default or a tiny model if one is set.
    # We will use the default uniXcoder, which is typical for this component.
    cache_dir = tmp_path_factory.mktemp("cache")
    cache_db = str(cache_dir / "test_cache.db")
    return CodeEmbedder(
        model_name="microsoft/unixcoder-base",
        cache_path=cache_db
    )


def test_embed_batch_shape_and_norm(embedder):
    code_strings = [
        "def hello_world():\n    print('hello world')",
        "class Foo:\n    pass"
    ]
    vectors = embedder.embed_batch(code_strings)
    
    assert vectors.shape == (2, 768)
    
    # Check L2 normalization (dot product with itself should be close to 1)
    norm1 = np.dot(vectors[0], vectors[0])
    norm2 = np.dot(vectors[1], vectors[1])
    
    assert np.isclose(norm1, 1.0, atol=1e-5)
    assert np.isclose(norm2, 1.0, atol=1e-5)


def test_caching(embedder):
    code_str = "def sum(a, b): return a + b"
    
    # First embed to populate cache
    vectors1 = embedder.embed_batch([code_str])
    
    # Verify it is in sqlite
    with sqlite3.connect(embedder.cache_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM embeddings")
        count = cursor.fetchone()[0]
        assert count >= 1
        
    # Re-embed (should hit cache)
    vectors2 = embedder.embed_batch([code_str])
    
    # Arrays should be exactly identical since one was loaded from cache
    np.testing.assert_array_equal(vectors1, vectors2)


def test_semantic_similarity(embedder):
    str_a = "def add(a, b):\n    return a + b"
    str_b = "def sum_two_numbers(x, y):\n    return x + y"
    str_c = "class DatabaseConnection:\n    def __init__(self, uri):\n        self.uri = uri"
    
    vectors = embedder.embed_batch([str_a, str_b, str_c])
    vec_a, vec_b, vec_c = vectors[0], vectors[1], vectors[2]
    
    # Cosine similarity is just dot product because they are L2 normalized
    sim_ab = np.dot(vec_a, vec_b)
    sim_ac = np.dot(vec_a, vec_c)
    
    # a and b should be very similar
    assert sim_ab > 0.7
    # a and c should be dissimilar
    assert sim_ac < sim_ab
    assert sim_ac < 0.5
