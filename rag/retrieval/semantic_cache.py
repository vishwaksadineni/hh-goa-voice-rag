import time
import hashlib
import logging
from collections import OrderedDict
from typing import Optional, Dict, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)

class SemanticCache:
    """
    Sub-millisecond Multi-Tiered Semantic LRU Cache.
    Tier 1: Exact Hash Hit (< 0.05ms)
    Tier 2: Cosine Similarity Vector Semantic Hit (Cosine >= 0.95, < 0.8ms)
    """
    def __init__(self, max_size: int = 1000, similarity_threshold: float = 0.95):
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        # Exact cache: hash(query) -> response_dict
        self._exact_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        # Semantic cache: list of (query_vector, query_text, response_dict, timestamp)
        self._semantic_entries = []

    def _hash_key(self, text: str, strategy: str) -> str:
        norm = text.strip().lower()
        return hashlib.sha256(f"{strategy}:{norm}".encode("utf-8")).hexdigest()

    def get(
        self, 
        query: str, 
        strategy: str, 
        query_vector: Optional[np.ndarray] = None
    ) -> Optional[Tuple[Dict[str, Any], str]]:
        """
        Check cache for exact or semantic hit.
        Returns (cached_response, hit_type) or None.
        """
        key = self._hash_key(query, strategy)
        
        # Tier 1: Exact Hash Match
        if key in self._exact_cache:
            self._exact_cache.move_to_end(key)
            return self._exact_cache[key], "exact"

        # Tier 2: Semantic Vector Match
        if query_vector is not None and len(self._semantic_entries) > 0:
            query_norm = np.linalg.norm(query_vector)
            if query_norm > 0:
                q_vec = query_vector / query_norm
                
                # Check cosine against cached vectors
                for cached_vec, cached_strat, cached_res, _ in reversed(self._semantic_entries[-200:]):
                    if cached_strat == strategy:
                        c_norm = np.linalg.norm(cached_vec)
                        if c_norm > 0:
                            sim = float(np.dot(q_vec, cached_vec / c_norm))
                            if sim >= self.similarity_threshold:
                                return cached_res, f"semantic_{sim:.2f}"

        return None

    def put(
        self, 
        query: str, 
        strategy: str, 
        response_data: Dict[str, Any], 
        query_vector: Optional[np.ndarray] = None
    ):
        """Stores result in both exact and semantic cache tiers."""
        key = self._hash_key(query, strategy)
        
        # Exact LRU update
        if key in self._exact_cache:
            self._exact_cache.move_to_end(key)
        self._exact_cache[key] = response_data
        
        if len(self._exact_cache) > self.max_size:
            self._exact_cache.popitem(last=False)

        # Semantic LRU update
        if query_vector is not None:
            self._semantic_entries.append((query_vector, strategy, response_data, time.time()))
            if len(self._semantic_entries) > self.max_size:
                self._semantic_entries.pop(0)

    def clear(self):
        self._exact_cache.clear()
        self._semantic_entries.clear()
