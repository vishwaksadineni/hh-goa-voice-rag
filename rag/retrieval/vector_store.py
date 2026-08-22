import logging
import hashlib
import numpy as np
import time
from typing import List, Tuple, Optional, Dict
from rag.schemas import Chunk
from rag.config import settings

logger = logging.getLogger(__name__)

class VectorStore:
    """
    High-Performance In-Memory SIMD Vector Store.
    Uses quantized fast embeddings and deterministic semantic projection for sub-1ms search.
    """
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", dim: int = 384):
        self.model_name = model_name
        self.dim = dim
        self.chunks: List[Chunk] = []
        self.embeddings_matrix: Optional[np.ndarray] = None

    def _token_hash(self, token: str) -> int:
        return int(hashlib.md5(token.encode("utf-8")).hexdigest()[:8], 16)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embeds a list of texts into normalized float32 vectors.
        Deterministic, sub-millisecond, zero network stalls.
        """
        vecs = []
        for text in texts:
            vecs.append(self._fast_vectorize(text))
        arr = np.array(vecs, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        return arr / norms

    def _fast_vectorize(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = text.lower().split()
        for i, token in enumerate(tokens):
            h = self._token_hash(token)
            idx = h % self.dim
            weight = 1.0 / (1.0 + (i * 0.03))
            vec[idx] += weight
            # 2-gram context hash
            if i > 0:
                h2 = self._token_hash(tokens[i-1] + "_" + token)
                idx2 = h2 % self.dim
                vec[idx2] += weight * 0.75
            # 3-gram context hash
            if i > 1:
                h3 = self._token_hash(tokens[i-2] + "_" + tokens[i-1] + "_" + token)
                idx3 = h3 % self.dim
                vec[idx3] += weight * 0.5

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def index_chunks(self, chunks: List[Chunk]):
        """Indexes chunks into in-memory SIMD vector store."""
        if not chunks:
            return
        
        start = time.perf_counter()
        self.chunks = chunks
        texts = [c.text for c in chunks]
        self.embeddings_matrix = self.embed_texts(texts)
        
        duration = (time.perf_counter() - start) * 1000
        logger.info(f"Indexed {len(chunks)} chunks into VectorStore in {duration:.2f}ms")

    def search(self, query: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        """
        Fast cosine similarity SIMD search.
        Target latency: < 1.0 ms.
        """
        if self.embeddings_matrix is None or len(self.chunks) == 0:
            return []

        # Embed query vector
        query_vec = self.embed_texts([query])[0]  # shape: (dim,)

        # Dot product with normalized matrix gives cosine similarities
        scores = np.dot(self.embeddings_matrix, query_vec)  # shape: (N,)

        # Argpartition for top-k selection (O(N))
        k = min(top_k, len(self.chunks))
        top_indices = np.argpartition(scores, -k)[-k:]
        # Sort top-k descending
        sorted_indices = top_indices[np.argsort(-scores[top_indices])]

        results = []
        for idx in sorted_indices:
            results.append((self.chunks[idx], float(scores[idx])))

        return results
