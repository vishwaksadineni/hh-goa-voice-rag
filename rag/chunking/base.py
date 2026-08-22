from abc import ABC, abstractmethod
from typing import List
from rag.schemas import DocumentPassage, Chunk

class BaseChunker(ABC):
    """Abstract Base Class for all Chunking Strategies."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def chunk_document(self, doc: DocumentPassage) -> List[Chunk]:
        """Split a single DocumentPassage into a list of Chunks."""
        pass

    def chunk_documents(self, docs: List[DocumentPassage]) -> List[Chunk]:
        """Split multiple DocumentPassages into chunks."""
        chunks: List[Chunk] = []
        for doc in docs:
            chunks.extend(self.chunk_document(doc))
        return chunks
