import logging
from typing import List, Tuple
from rag.schemas import RetrievalResult
from rag.config import settings

logger = logging.getLogger(__name__)

class RetrievalGuardrail:
    """
    Retrieval Sufficiency Guardrail.
    Ensures that retrieved chunks meet the minimum relevance threshold
    across dense and sparse hybrid metrics before synthesis.
    """
    def __init__(self, min_similarity: float = 0.20):
        self.min_similarity = min_similarity

    def evaluate(self, results: List[RetrievalResult]) -> Tuple[bool, float, str]:
        """
        Evaluates retrieval sufficiency across hybrid dense and sparse signals.
        Returns: (is_sufficient, confidence_score, reason)
        """
        if not results or len(results) == 0:
            return False, 0.0, "NO_RETRIEVED_DOCUMENTS"

        r0 = results[0]
        confidence = max(r0.dense_score, r0.sparse_score)
        if confidence == 0.0 and r0.score > 0:
            confidence = r0.score
        
        if confidence < self.min_similarity:
            return False, float(confidence), f"LOW_RETRIEVAL_CONFIDENCE ({confidence:.3f} < {self.min_similarity})"

        return True, float(confidence), "SUFFICIENT_RETRIEVAL"
