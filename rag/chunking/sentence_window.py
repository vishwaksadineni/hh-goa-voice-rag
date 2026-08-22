import re
from typing import List
from rag.schemas import DocumentPassage, Chunk
from rag.chunking.base import BaseChunker

class SentenceWindowChunker(BaseChunker):
    """
    Sentence-Window Chunking Strategy.
    Indexes individual focal sentences for high-precision query matching,
    while attaching a sliding context window (k surrounding sentences)
    to feed the LLM complete surrounding context.
    """
    def __init__(self, window_size: int = 2):
        super().__init__(
            name="sentence_window",
            description="Sentence-Window: focal sentence embedding with sliding k-neighbor context window."
        )
        self.window_size = window_size
        self.sentence_pattern = re.compile(r'(?<=[.!?])\s+')

    def _split_sentences(self, text: str) -> List[str]:
        raw = self.sentence_pattern.split(text)
        return [s.strip() for s in raw if s.strip()]

    def chunk_document(self, doc: DocumentPassage) -> List[Chunk]:
        sentences = self._split_sentences(doc.text)
        chunks: List[Chunk] = []

        if not sentences:
            sentences = [doc.text]

        for idx, focal_sentence in enumerate(sentences):
            # Window calculation: idx - window_size to idx + window_size
            start_idx = max(0, idx - self.window_size)
            end_idx = min(len(sentences), idx + self.window_size + 1)
            window_context = " ".join(sentences[start_idx:end_idx])

            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}_sw_{idx}",
                doc_id=doc.doc_id,
                text=focal_sentence,  # Focal sentence for precise embedding lookup
                parent_text=window_context,  # Expanded window context for LLM synthesis
                strategy=self.name,
                token_count=len(focal_sentence.split()),
                metadata={
                    **doc.metadata,
                    "sentence_idx": idx,
                    "total_sentences": len(sentences),
                    "window_size": self.window_size,
                    "window_text": window_context,
                    "is_gold": doc.is_gold,
                    "query_id": doc.query_id,
                    "query_type": doc.query_type
                }
            ))

        return chunks
