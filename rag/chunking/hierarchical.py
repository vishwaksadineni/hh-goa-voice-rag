import re
from typing import List
from rag.schemas import DocumentPassage, Chunk
from rag.chunking.base import BaseChunker

class HierarchicalChunker(BaseChunker):
    """
    Hierarchical (Parent-Child) Chunking Strategy.
    Divides documents into small, high-granularity child chunks for fine-grained
    vector similarity lookup while retaining the parent passage for grounded synthesis.
    """
    def __init__(self, child_token_size: int = 35, child_overlap: int = 10):
        super().__init__(
            name="hierarchical",
            description="Parent-Child hierarchy: fine-grained child chunk search with full parent context expansion."
        )
        self.child_token_size = child_token_size
        self.child_overlap = child_overlap

    def chunk_document(self, doc: DocumentPassage) -> List[Chunk]:
        words = doc.text.split()
        chunks: List[Chunk] = []

        if len(words) <= self.child_token_size:
            # If text is already small, treat as single chunk
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}_c0",
                doc_id=doc.doc_id,
                text=doc.text,
                parent_text=doc.text,
                strategy=self.name,
                token_count=len(words),
                metadata={
                    **doc.metadata,
                    "level": "child",
                    "parent_doc_id": doc.doc_id,
                    "is_gold": doc.is_gold,
                    "query_id": doc.query_id,
                    "query_type": doc.query_type
                }
            ))
            return chunks

        # Sliding window with overlap
        step = max(1, self.child_token_size - self.child_overlap)
        child_idx = 0
        for i in range(0, len(words), step):
            child_words = words[i:i + self.child_token_size]
            if not child_words:
                break
            child_text = " ".join(child_words)
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}_c{child_idx}",
                doc_id=doc.doc_id,
                text=child_text,
                parent_text=doc.text,  # Full parent context
                strategy=self.name,
                token_count=len(child_words),
                metadata={
                    **doc.metadata,
                    "level": "child",
                    "child_index": child_idx,
                    "start_word_idx": i,
                    "end_word_idx": i + len(child_words),
                    "parent_doc_id": doc.doc_id,
                    "is_gold": doc.is_gold,
                    "query_id": doc.query_id,
                    "query_type": doc.query_type
                }
            ))
            child_idx += 1
            if i + self.child_token_size >= len(words):
                break

        return chunks
