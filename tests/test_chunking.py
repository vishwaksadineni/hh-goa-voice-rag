import pytest
from rag.schemas import DocumentPassage
from rag.chunking.hierarchical import HierarchicalChunker
from rag.chunking.semantic import SemanticChunker
from rag.chunking.metadata_aware import MetadataAwareChunker
from rag.chunking.sentence_window import SentenceWindowChunker
from rag.chunking.registry import chunking_registry

@pytest.fixture
def sample_passage():
    return DocumentPassage(
        doc_id="doc_test_101",
        query_id=101,
        query_type="DESCRIPTION",
        text="Panaji is the capital of Goa. It is famous for Portuguese colonial architecture and the Mandovi river promenade. Calangute beach attracts tourists.",
        source_lang="en",
        target_lang="hi",
        is_gold=True,
        metadata={"domain": "travel"}
    )

def test_hierarchical_chunker(sample_passage):
    chunker = HierarchicalChunker(child_token_size=10, child_overlap=2)
    chunks = chunker.chunk_document(sample_passage)
    
    assert len(chunks) >= 2
    for c in chunks:
        assert c.strategy == "hierarchical"
        assert c.doc_id == sample_passage.doc_id
        # Crucial: hierarchical chunks retain full parent context for generation
        assert c.parent_text == sample_passage.text
        assert c.metadata["level"] == "child"

def test_semantic_chunker(sample_passage):
    chunker = SemanticChunker(max_tokens=15, min_tokens=5)
    chunks = chunker.chunk_document(sample_passage)
    
    assert len(chunks) >= 1
    for c in chunks:
        assert c.strategy == "semantic"
        assert len(c.text) > 0
        assert "is_gold" in c.metadata

def test_metadata_aware_chunker(sample_passage):
    chunker = MetadataAwareChunker(max_tokens=20)
    chunks = chunker.chunk_document(sample_passage)
    
    assert len(chunks) >= 1
    for c in chunks:
        assert c.strategy == "metadata_aware"
        # Metadata prefix should be injected into the text
        assert "[DESCRIPTION | LANG:EN | DOC:doc_test_101]" in c.text

def test_sentence_window_chunker(sample_passage):
    chunker = SentenceWindowChunker(window_size=1)
    chunks = chunker.chunk_document(sample_passage)
    
    assert len(chunks) == 3  # 3 sentences
    assert chunks[0].text.startswith("Panaji is the capital")
    assert chunks[0].parent_text is not None  # window context attached

def test_chunking_registry():
    strategies = chunking_registry.list_strategies()
    assert len(strategies) == 4
    strat_ids = [s["id"] for s in strategies]
    assert "hierarchical" in strat_ids
    assert "semantic" in strat_ids
    assert "metadata_aware" in strat_ids
    assert "sentence_window" in strat_ids
