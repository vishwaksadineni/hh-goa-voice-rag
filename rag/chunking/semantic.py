import re
from typing import List
from rag.schemas import DocumentPassage, Chunk
from rag.chunking.base import BaseChunker

class SemanticChunker(BaseChunker):
    """
    Semantic Boundary Chunking Strategy.
    Splits passages along natural discourse boundaries, sentence terminations,
    and transition connectives (e.g. 'However', 'Therefore', 'In contrast', semicolons)
    to keep cohesive conceptual units intact.
    """
    def __init__(self, max_tokens: int = 60, min_tokens: int = 15):
        super().__init__(
            name="semantic",
            description="Semantic Boundary: splits along discourse transitions and cohesive linguistic units."
        )
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.discourse_pattern = re.compile(
            r'(?<=[.!?])\s+|;\s*|(?<=[,\.\?!])\s+(?=(?:However|Furthermore|Therefore|Consequently|In contrast|Moreover|Although|Additionally|Specifically|For example)\b)',
            re.IGNORECASE
        )

    def _split_into_propositions(self, text: str) -> List[str]:
        # Split on sentence boundaries and discourse markers
        raw_parts = self.discourse_pattern.split(text)
        cleaned = [p.strip() for p in raw_parts if p.strip()]
        if not cleaned:
            return [text]
        return cleaned

    def chunk_document(self, doc: DocumentPassage) -> List[Chunk]:
        propositions = self._split_into_propositions(doc.text)
        chunks: List[Chunk] = []
        
        current_chunk_parts: List[str] = []
        current_tokens = 0
        chunk_idx = 0

        for prop in propositions:
            prop_tokens = len(prop.split())
            if current_tokens + prop_tokens > self.max_tokens and current_tokens >= self.min_tokens:
                # Flush current chunk
                chunk_text = " ".join(current_chunk_parts)
                chunks.append(Chunk(
                    chunk_id=f"{doc.doc_id}_sem_{chunk_idx}",
                    doc_id=doc.doc_id,
                    text=chunk_text,
                    parent_text=doc.text,
                    strategy=self.name,
                    token_count=current_tokens,
                    metadata={
                        **doc.metadata,
                        "semantic_segments_count": len(current_chunk_parts),
                        "is_gold": doc.is_gold,
                        "query_id": doc.query_id,
                        "query_type": doc.query_type
                    }
                ))
                chunk_idx += 1
                current_chunk_parts = [prop]
                current_tokens = prop_tokens
            else:
                current_chunk_parts.append(prop)
                current_tokens += prop_tokens

        if current_chunk_parts:
            chunk_text = " ".join(current_chunk_parts)
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}_sem_{chunk_idx}",
                doc_id=doc.doc_id,
                text=chunk_text,
                parent_text=doc.text,
                strategy=self.name,
                token_count=current_tokens,
                metadata={
                    **doc.metadata,
                    "semantic_segments_count": len(current_chunk_parts),
                    "is_gold": doc.is_gold,
                    "query_id": doc.query_id,
                    "query_type": doc.query_type
                }
            ))

        return chunks
