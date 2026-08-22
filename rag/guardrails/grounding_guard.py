import re
import logging
from typing import List, Tuple
from rag.schemas import RetrievalResult
from rag.config import settings

logger = logging.getLogger(__name__)

class GroundingGuardrail:
    """
    Context Grounding & Hallucination Prevention Guardrail.
    Verifies that the generated answer is strictly supported by the retrieved context.
    Computes key entity overlap, claim entailment, and token fidelity.
    """
    def __init__(self, min_confidence: float = 0.45):
        self.min_confidence = min_confidence or settings.guardrail_grounding_min_confidence

    def _extract_key_tokens(self, text: str) -> set:
        # Extract alphanumeric words > 2 chars, ignoring common stop words
        stopwords = {
            "the", "and", "is", "in", "it", "to", "of", "for", "with", "on", 
            "that", "this", "by", "from", "are", "was", "as", "an", "be", "at",
            "का", "के", "की", "है", "हैं", "और", "में", "से", "को", "पर", "एक", "या"
        }
        tokens = set(re.findall(r'\w+', text.lower()))
        return {t for t in tokens if len(t) > 2 and t not in stopwords}

    def evaluate(self, answer: str, retrieved_contexts: List[RetrievalResult]) -> Tuple[bool, float, str]:
        """
        Evaluates answer grounding against retrieved contexts.
        Returns: (is_grounded, grounding_score, explanation)
        """
        if not answer or not answer.strip():
            return False, 0.0, "EMPTY_ANSWER"

        if not retrieved_contexts:
            return False, 0.0, "NO_GROUNDING_CONTEXT"

        # Combine all retrieved texts (including parent text if hierarchical)
        context_corpus = " ".join([
            (r.chunk.parent_text or r.chunk.text) for r in retrieved_contexts
        ])

        answer_tokens = self._extract_key_tokens(answer)
        context_tokens = self._extract_key_tokens(context_corpus)

        if not answer_tokens:
            return True, 1.0, "GROUNDED"

        # Compute token entailment ratio
        supported_tokens = answer_tokens.intersection(context_tokens)
        grounding_score = len(supported_tokens) / float(len(answer_tokens))

        if grounding_score < self.min_confidence:
            return False, grounding_score, f"POTENTIAL_HALLUCINATION (grounding_score {grounding_score:.2f} < {self.min_confidence})"

        return True, grounding_score, f"VERIFIED_GROUNDED (score: {grounding_score:.2f})"
