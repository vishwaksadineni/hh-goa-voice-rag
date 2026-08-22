import logging
import time
from typing import List, Dict, Any, Tuple
from rag.schemas import Chunk, RetrievalResult
from rag.retrieval.vector_store import VectorStore
from rag.retrieval.bm25_index import BM25Index
from rag.config import settings

logger = logging.getLogger(__name__)

class HybridSearchEngine:
    """
    Hybrid Retrieval Engine combining Dense SIMD Vector Search and Sparse BM25
    using Reciprocal Rank Fusion (RRF).
    """
    def __init__(self, vector_store: VectorStore, bm25_index: BM25Index):
        self.vector_store = vector_store
        self.bm25_index = bm25_index

    def search(
        self, 
        query: str, 
        top_k: int = 3, 
        dense_weight: float = 0.7, 
        sparse_weight: float = 0.3,
        rrf_k: int = 60
    ) -> List[RetrievalResult]:
        start = time.perf_counter()

        # Step 1: Retrieve candidate sets from both Dense and Sparse index
        fetch_k = top_k * 3
        dense_results = self.vector_store.search(query, top_k=fetch_k)
        sparse_results = self.bm25_index.search(query, top_k=fetch_k)

        # Step 2: Reciprocal Rank Fusion (RRF)
        # RRF Score = w_dense / (k + rank_dense) + w_sparse / (k + rank_sparse)
        chunk_map: Dict[str, Chunk] = {}
        dense_ranks: Dict[str, Tuple[int, float]] = {}
        sparse_ranks: Dict[str, Tuple[int, float]] = {}

        for rank, (chunk, score) in enumerate(dense_results, 1):
            chunk_map[chunk.chunk_id] = chunk
            dense_ranks[chunk.chunk_id] = (rank, score)

        for rank, (chunk, score) in enumerate(sparse_results, 1):
            chunk_map[chunk.chunk_id] = chunk
            sparse_ranks[chunk.chunk_id] = (rank, score)

        fused_scores: Dict[str, float] = {}
        for chunk_id in chunk_map:
            score = 0.0
            if chunk_id in dense_ranks:
                d_rank, _ = dense_ranks[chunk_id]
                score += dense_weight * (1.0 / (rrf_k + d_rank))
            if chunk_id in sparse_ranks:
                s_rank, _ = sparse_ranks[chunk_id]
                score += sparse_weight * (1.0 / (rrf_k + s_rank))
            fused_scores[chunk_id] = score

        # Step 3: Sort by fused score
        sorted_ids = sorted(fused_scores.keys(), key=lambda cid: fused_scores[cid], reverse=True)[:top_k]

        results: List[RetrievalResult] = []
        for rank, cid in enumerate(sorted_ids, 1):
            chunk = chunk_map[cid]
            d_score = dense_ranks[cid][1] if cid in dense_ranks else 0.0
            s_score = sparse_ranks[cid][1] if cid in sparse_ranks else 0.0
            f_score = fused_scores[cid]

            results.append(RetrievalResult(
                chunk=chunk,
                score=f_score,
                dense_score=d_score,
                sparse_score=s_score,
                rank=rank
            ))

        return results
