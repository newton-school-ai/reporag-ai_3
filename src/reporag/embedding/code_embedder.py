"""Code embedding pipeline.

Embeds code chunks using CodeBERT or UniXcoder. Produces 768-dim L2-normalized
vectors. Supports batch embedding with GPU acceleration and CPU fallback.
"""

import hashlib
import sqlite3
import numpy as np
import logging
from typing import Optional

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    torch = None
    AutoTokenizer = None
    AutoModel = None

logger = logging.getLogger(__name__)


class CodeEmbedder:
    def __init__(
        self,
        model_name: str = "microsoft/unixcoder-base",
        device: Optional[str] = None,
        cache_path: str = ".reporag_cache.db"
    ):
        if torch is None:
            raise ImportError("torch and transformers are required for CodeEmbedder")

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
            
        logger.info(f"Initializing CodeEmbedder with model {model_name} on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        self.cache_path = cache_path
        self._init_cache()

    def _init_cache(self):
        """Initialize the SQLite caching database."""
        with sqlite3.connect(self.cache_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    hash TEXT PRIMARY KEY,
                    vector BLOB
                )
                """
            )
            conn.commit()

    def _get_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _get_cached(self, hashes: list[str]) -> dict[str, np.ndarray]:
        """Fetch cached embeddings for a list of hashes."""
        results = {}
        if not hashes:
            return results
            
        placeholders = ",".join(["?"] * len(hashes))
        with sqlite3.connect(self.cache_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT hash, vector FROM embeddings WHERE hash IN ({placeholders})",
                hashes
            )
            for row in cursor.fetchall():
                results[row[0]] = np.frombuffer(row[1], dtype=np.float32)
        return results

    def _set_cached(self, hash_to_vec: dict[str, np.ndarray]):
        """Save new embeddings to cache."""
        if not hash_to_vec:
            return
            
        rows = [(h, v.tobytes()) for h, v in hash_to_vec.items()]
        with sqlite3.connect(self.cache_path) as conn:
            cursor = conn.cursor()
            cursor.executemany(
                "INSERT OR IGNORE INTO embeddings (hash, vector) VALUES (?, ?)",
                rows
            )
            conn.commit()

    def embed_batch(self, code_strings: list[str], batch_size: int = 32) -> np.ndarray:
        """Embed a list of code strings, returning an (N, 768) normalized numpy array."""
        if not code_strings:
            return np.empty((0, 768), dtype=np.float32)
            
        hashes = [self._get_hash(s) for s in code_strings]
        cached = self._get_cached(hashes)
        
        to_compute = []
        to_compute_hashes = []
        
        for s, h in zip(code_strings, hashes):
            if h not in cached:
                to_compute.append(s)
                to_compute_hashes.append(h)
                
        new_embeddings = {}
        
        for i in range(0, len(to_compute), batch_size):
            batch_strs = to_compute[i : i + batch_size]
            batch_hashes = to_compute_hashes[i : i + batch_size]
            
            inputs = self.tokenizer(
                batch_strs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                
            cls_embeddings_np = cls_embeddings.cpu().numpy().astype(np.float32)
            
            for h, vec in zip(batch_hashes, cls_embeddings_np):
                new_embeddings[h] = vec
                cached[h] = vec
                
        self._set_cached(new_embeddings)
        
        final_vectors = np.array([cached[h] for h in hashes], dtype=np.float32)
        
        norms = np.linalg.norm(final_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        final_vectors = final_vectors / norms
        
        return final_vectors
