from typing import List
from rag.schemas import DocumentPassage, Chunk
from rag.chunking.base import BaseChunker

class MetadataAwareChunker(BaseChunker):
    """
    Metadata-Aware Contextual Chunking Strategy.
    Enriches chunks with contextual metadata tags (Source Language, Query Type,
    Document Origin, Intent Category) to enhance dense and sparse vector relevance.
    """
    def __init__(self, max_tokens: int = 55):
        super().__init__(
            name="metadata_aware",
            description="Metadata-Aware: contextual metadata tagging with query-type and language provenance."
        )
        self.max_tokens = max_tokens

    def chunk_document(self, doc: DocumentPassage) -> List[Chunk]:
        words = doc.text.split()
        chunks: List[Chunk] = []
        
        # Construct rich metadata header
        lang_tag = doc.source_lang.upper() if doc.source_lang else "EN"
        type_tag = doc.query_type or "GENERAL"
        meta_prefix = f"[{type_tag} | LANG:{lang_tag} | DOC:{doc.doc_id}]"

        step = self.max_tokens
        chunk_idx = 0
        for i in range(0, len(words), step):
            slice_words = words[i:i + step]
            if not slice_words:
                break
            
            body_text = " ".join(slice_words)
            # Prepend metadata header for contextualized indexing
            indexed_text = f"{meta_prefix} {body_text}"
            
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}_meta_{chunk_idx}",
                doc_id=doc.doc_id,
                text=indexed_text,
                parent_text=doc.text,
                strategy=self.name,
                token_count=len(slice_words) + 6,
                metadata={
                    **doc.metadata,
                    "metadata_prefix": meta_prefix,
                    "raw_chunk_text": body_text,
                    "query_type": type_tag,
                    "source_lang": doc.source_lang,
                    "is_gold": doc.is_gold,
                    "query_id": doc.query_id
                }
            ))
            chunk_idx += 1

        return chunks
