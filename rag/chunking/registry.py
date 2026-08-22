from typing import Dict, List, Optional
from rag.chunking.base import BaseChunker
from rag.chunking.hierarchical import HierarchicalChunker
from rag.chunking.semantic import SemanticChunker
from rag.chunking.metadata_aware import MetadataAwareChunker
from rag.chunking.sentence_window import SentenceWindowChunker

class ChunkingRegistry:
    """Registry and factory for all chunking strategies."""
    
    def __init__(self):
        self._strategies: Dict[str, BaseChunker] = {
            "hierarchical": HierarchicalChunker(),
            "semantic": SemanticChunker(),
            "metadata_aware": MetadataAwareChunker(),
            "sentence_window": SentenceWindowChunker()
        }

    def get_strategy(self, name: Optional[str] = None) -> BaseChunker:
        if not name or name not in self._strategies:
            name = "hierarchical"
        return self._strategies[name]

    def list_strategies(self) -> List[Dict[str, str]]:
        return [
            {"id": k, "name": v.name, "description": v.description}
            for k, v in self._strategies.items()
        ]

chunking_registry = ChunkingRegistry()
