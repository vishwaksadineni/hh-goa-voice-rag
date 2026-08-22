import re
import logging
import time
from typing import List, Tuple, Optional
from rank_bm25 import BM25Okapi
from rag.schemas import Chunk

logger = logging.getLogger(__name__)

class BM25Index:
    """
    High-Speed BM25 Sparse Keyword Index.
    Complements dense neural embeddings by capturing exact keywords, acronyms, and numeric queries.
    """
    def __init__(self):
        self.chunks: List[Chunk] = []
        self.bm25: Optional[BM25Okapi] = None

    def _tokenize(self, text: str) -> List[str]:
        # Lowercase and clean alphanumeric and Indic unicode characters
        cleaned = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = [t for t in cleaned.split() if len(t) > 1]
        return tokens or text.lower().split()

    def index_chunks(self, chunks: List[Chunk]):
        if not chunks:
            return
        
        start = time.perf_counter()
        self.chunks = chunks
        corpus_tokenized = [self._tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(corpus_tokenized)
        
        duration = (time.perf_counter() - start) * 1000
        logger.info(f"Indexed {len(chunks)} chunks into BM25 in {duration:.2f}ms")

    def search(self, query: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        """Fast BM25 scoring. Target latency: < 1.0 ms."""
        if self.bm25 is None or len(self.chunks) == 0:
            return []

        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
        
        # Max score normalization
        max_score = float(max(scores)) if len(scores) > 0 and max(scores) > 0 else 1.0
        normalized_scores = [float(s) / max_score for s in scores]

        # Top-k selection
        k = min(top_k, len(self.chunks))
        top_indices = sorted(range(len(normalized_scores)), key=lambda i: normalized_scores[i], reverse=True)[:k]

        return [(self.chunks[i], normalized_scores[i]) for i in top_indices]
