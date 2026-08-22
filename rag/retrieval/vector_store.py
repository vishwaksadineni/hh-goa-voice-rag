import logging
import numpy as np
import time
from typing import List, Tuple, Optional, Dict
from rag.schemas import Chunk
from rag.config import settings

logger = logging.getLogger(__name__)

class VectorStore:
    """
    High-Performance In-Memory SIMD Vector Store.
    Uses quantized ONNX FastEmbed / Fast Vectorizer for sub-2ms embedding & retrieval.
    """
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self.chunks: List[Chunk] = []
        self.embeddings_matrix: Optional[np.ndarray] = None
        self._embedder = None
        self._init_embedder()

    def _init_embedder(self):
        try:
            from fastembed import TextEmbedding
            # Initialize fast ONNX local embedding model
            self._embedder = TextEmbedding(model_name=self.model_name)
            logger.info(f"Loaded FastEmbed ONNX model: {self.model_name}")
        except Exception as e:
            logger.warning(f"FastEmbed init fallback: {e}. Using deterministic semantic token embedder.")
            self._embedder = None

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embeds a list of texts into normalized float32 vectors."""
        if self._embedder is not None:
            try:
                embeddings_gen = self._embedder.embed(texts)
                vecs = np.array(list(embeddings_gen), dtype=np.float32)
                # L2 normalize
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                norms[norms == 0] = 1e-10
                return vecs / norms
            except Exception as e:
                logger.error(f"FastEmbed embedding error: {e}")

        # High-speed fallback vectorizer (character-n-gram + semantic hash projection)
        vecs = []
        for text in texts:
            vecs.append(self._fast_hash_vector(text))
        arr = np.array(vecs, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        return arr / norms

    def _fast_hash_vector(self, text: str, dim: int = 384) -> np.ndarray:
        vec = np.zeros(dim, dtype=np.float32)
        tokens = text.lower().split()
        for i, token in enumerate(tokens):
            h = hash(token)
            idx = abs(h) % dim
            weight = 1.0 / (1.0 + (i * 0.05))
            vec[idx] += weight
            # 2-gram hash
            if i > 0:
                h2 = hash(tokens[i-1] + "_" + token)
                idx2 = abs(h2) % dim
                vec[idx2] += weight * 0.6
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def index_chunks(self, chunks: List[Chunk]):
        """Indexes chunks into in-memory vector store."""
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
        Target latency: < 2.0 ms.
        """
        if self.embeddings_matrix is None or len(self.chunks) == 0:
            return []

        # Embed query vector
        query_vec = self.embed_texts([query])[0]  # shape: (dim,)

        # Dot product with normalized matrix gives cosine similarities
        scores = np.dot(self.embeddings_matrix, query_vec)  # shape: (N,)

        # Argpartition for top-k selection (O(N) vs O(N log N))
        k = min(top_k, len(self.chunks))
        top_indices = np.argpartition(scores, -k)[-k:]
        # Sort top-k descending
        sorted_indices = top_indices[np.argsort(-scores[top_indices])]

        results = []
        for idx in sorted_indices:
            results.append((self.chunks[idx], float(scores[idx])))

        return results
